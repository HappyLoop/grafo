import asyncio
import asyncio.log
from logging import Logger, getLogger
from typing import Any, Optional
from uuid import uuid4


from .components import Node, PickerNode, UnionNode

logger = getLogger("grafo")


class AsyncTreeExecutor:
    """
    An executor that processes a tree of nodes concurrently. Rules:
    - Each node is processed by a worker.
    - A worker executes the coroutine of a node and enqueues its children.
    - A worker stops when it receives a None from the queue.
    - The executor stops when all workers have stopped.

    :param root: The root node of the tree.
    :param num_workers: The number of workers to process the tree.
    :param min_workers: The minimum number of workers allowed.
    :param max_workers: The maximum number of workers allowed.
    :param use_dynamic_workers: Whether to automatically adjust the number of workers.
    :param forward_results: Whether to forward the results of each node to its children as arguments.
    :param logger: A logger to log errors.
    :param cutoff_branch_on_error: Whether to stop the branch when a node fails.
    :param quit_on_tree_error: Whether to stop the tree when a node fails.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        root: Optional[Node] = None,
        num_workers: int = 1,
        min_workers: int = 1,
        max_workers: int = 10,
        use_dynamic_workers: bool = True,
        logger: Logger | None = None,
        cutoff_branch_on_error: bool = False,
        quit_tree_on_error: bool = False,
    ):
        if num_workers < min_workers:
            raise ValueError(
                "'num_workers' must be greater than or equal to 'min_workers'."
            )
        if max_workers < num_workers:
            raise ValueError(
                "'max_workers' must be greater than or equal to 'num_workers'."
            )

        self._name = name

        self._root = root
        self._queue = asyncio.Queue()
        self._num_workers = num_workers
        self._min_workers = min_workers
        self._max_workers = max_workers
        self._use_dynamic_workers = use_dynamic_workers
        self._workers = []
        self._output = dict()

        self._logger = logger
        self._cutoff_branch_on_error = cutoff_branch_on_error
        self._quit_tree_on_error = quit_tree_on_error
        self._visited_nodes = set()
        self._enqueued_nodes = set()

        self._graceful_stop_flag = False
        self._graceful_stop_nodes = set()

    @property
    def name(self):
        return self._name

    @property
    def root(self):
        return self._root

    @property
    def queue(self):
        return self._queue

    @property
    def min_workers(self):
        return self._min_workers

    @property
    def max_workers(self):
        return self._max_workers

    @property
    def workers(self):
        return self._workers

    @property
    def output(self):
        return self._output

    @property
    def logger(self):
        return self._logger

    @property
    def cutoff_branch_on_error(self):
        return self._cutoff_branch_on_error

    @property
    def quit_tree_on_error(self):
        return self._quit_tree_on_error

    @property
    def visited_nodes(self):
        return self._visited_nodes

    @property
    def enqueued_nodes(self):
        return self._enqueued_nodes

    @property
    def global_stop_flag(self):
        return self._graceful_stop_flag

    @property
    def graceful_stop_nodes(self):
        return self._graceful_stop_nodes

    def __or__(self, tree_dict: dict[Node, Any]):
        """
        Override the `|` operator to create an instance of AsyncTreeExecutor and builds
        a tree of nodes from nested dictionaries.

        Example:
        {
            Node1: {
                Node2: {
                    Node3: {
                        Node4: {}
                    }
                },
                Node5: {}
            }
        }

        The structure can go on indefinitely.

        NOTE: If the tree contains more than one node in the 1st level, the executor will
        inject a mockup root node and use it as the root of the tree.
        """
        if self._root:
            raise ValueError(
                "A root has been provided, indicating a manual tree construction. Cannot use the | operator syntax."
            )

        if len(tree_dict) > 1:
            logger.warning(
                "Tree contains more than one node in the 1st level. Defaulting to a mockup root node."
            )

            async def mockup_coroutine(name):
                return "root mockup"

            tree_dict = {
                Node(
                    uuid=str(uuid4()),
                    metadata={
                        "__mockup__": True,
                    },
                    coroutine=mockup_coroutine,
                ): tree_dict
            }

        def connect_children(
            parent_node: Node, children_iterable: dict[Node, Any] | list[Node]
        ):
            self.__validate_element(parent_node)

            if isinstance(children_iterable, dict):
                for child_node, descendants_iterable in children_iterable.items():
                    if isinstance(parent_node, PickerNode) and isinstance(
                        child_node, UnionNode
                    ):
                        raise ValueError(
                            "UnionNodes cannot be children of PickerNodes."
                        )

                    self.__validate_element(child_node)

                    parent_node.connect(child_node)
                    if isinstance(child_node, UnionNode):
                        child_node.add_parent(parent_node)

                    connect_children(child_node, descendants_iterable)

            elif isinstance(children_iterable, list):
                for child_node in children_iterable:
                    if isinstance(parent_node, PickerNode) and isinstance(
                        child_node, UnionNode
                    ):
                        raise ValueError(
                            "UnionNodes cannot be children of PickerNodes."
                        )

                    self.__validate_element(child_node)
                    parent_node.connect(child_node)
                    if isinstance(child_node, UnionNode):
                        child_node.add_parent(parent_node)

        for root_node, children_iterable in tree_dict.items():
            self._root = root_node

            if isinstance(children_iterable, dict):
                connect_children(root_node, children_iterable)

            elif isinstance(children_iterable, list):
                for child_node in children_iterable:
                    if isinstance(self, PickerNode) and isinstance(
                        child_node, UnionNode
                    ):
                        raise ValueError(
                            "UnionNodes cannot be children of PickerNodes."
                        )

                    self.__validate_element(child_node)
                    root_node.connect(child_node)

        return self

    async def __worker(self):
        """
        A worker that executes the work contained in a Node.
        """
        result = None
        while True:
            node: Node = await self._queue.get()

            if node is None:
                self._queue.task_done()
                break

            # Run the node
            try:
                if node.uuid not in self._visited_nodes:
                    result = await node.run()
            except Exception as e:
                if self._quit_tree_on_error:
                    self._graceful_stop_flag = True
                    self._graceful_stop_nodes.add(node)
                    self._queue.task_done()
                    if self._logger:
                        self._logger.error(f"Quit at {node}. Error: {e}")
                    break

                elif self._cutoff_branch_on_error:
                    self._queue.task_done()

                    if self._logger:
                        self._logger.error(f"Cutoff at {node}. Error: {e}")
                    break

                elif self._logger:
                    self._logger.error(f"Error on {node}: {e}")
            finally:
                self._visited_nodes.add(node.uuid)

            self._output[str(node.uuid)] = result
            node.set_output(result)

            # Update children
            if isinstance(node, PickerNode):
                children = result or []
            else:
                children = node.children or []

            for child in children or []:
                args = []
                kwargs = {}
                if isinstance(result, list):
                    for res in result:
                        if isinstance(res, dict):
                            kwargs.update(res)
                        else:
                            args.append(res)

                elif isinstance(result, dict):
                    kwargs.update(result)
                else:
                    args.append(result)

                if node.forward_output:
                    if isinstance(child, UnionNode):
                        child.parent_completed(node.uuid, node.output)
                        child.append_arguments(args=args, kwargs=kwargs)
                    else:
                        child.update(args=args, kwargs=kwargs)
                else:
                    if isinstance(child, UnionNode):
                        child.parent_completed(node.uuid, node.output)

            # Enqueue children and adjust workers
            if not self._graceful_stop_flag:
                async with asyncio.Lock():
                    for child_node in children:
                        if child_node not in self._enqueued_nodes:
                            self._queue.put_nowait(child_node)
                            self._enqueued_nodes.add(child_node)
                            # Adjust workers if needed
                            if (
                                self._use_dynamic_workers
                                and self._queue.qsize() > len(self._workers)
                                and len(self._workers) < self._max_workers
                            ):
                                new_worker = asyncio.create_task(self.__worker())
                                self._workers.append(new_worker)
                                if self.logger:
                                    self.logger.debug(
                                        f"Added worker. Current workers: {len(self._workers)}"
                                    )
            # Remove workers if needed
            if (
                self._use_dynamic_workers
                and self._queue.qsize() < len(self._workers) - 1
                and len(self._workers) > self._min_workers
            ):
                self._queue.put_nowait(None)
                self._workers.pop()
                if self.logger:
                    self.logger.debug(
                        f"Removed worker. Current workers: {len(self._workers)}"
                    )

            self._queue.task_done()

    async def __stop_all_workers(self):
        """
        Stops all workers.
        """
        for _ in range(self._num_workers):
            self._queue.put_nowait(None)

    def __validate_element(self, obj: Any):
        """
        Check if an object is an instance of Node.
        """
        if not isinstance(obj, Node):
            raise ValueError(f"Object is not a Node instance. Object: {obj}")

    async def run(self):
        """
        Runs the tree with the specified number of workers.
        """
        self._queue.put_nowait(self.root)
        self._workers = [
            asyncio.create_task(self.__worker()) for _ in range(self._num_workers)
        ]

        if len(self._workers) == 0:
            raise ValueError("No workers were created.")

        if self.logger:
            self.logger.debug(
                f"Running tree{' {}'.format(self.name) if self.name else ''}..."
            )

        await self._queue.join()
        await self.__stop_all_workers()
        await asyncio.gather(*self._workers, return_exceptions=True)

        if self.logger:
            self.logger.debug("Tree execution complete.")
            if self._graceful_stop_flag:
                self.logger.debug(
                    f"Graceful stop due to errors in nodes: {self._graceful_stop_nodes}"
                )

        return self._output
