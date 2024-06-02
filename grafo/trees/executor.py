import asyncio
from logging import Logger
from typing import Any, Optional

from .components import Node, PickerNode


class AsyncTreeExecutor:
    """
    An executor that processes a tree of nodes concurrently. Rules:
    - Each node is processed by a worker.
    - A worker executes the coroutine of a node and enqueues its children.
    - A worker stops when it receives a None from the queue.
    - The executor stops when all workers have stopped.

    :param root: The root node of the tree.
    :param num_workers: The number of workers to process the tree.
    :param forward_results: Whether to forward the results of each node to its children as arguments.
    :param logger: A logger to log errors.
    :param cutoff_branch_on_error: Whether to stop the branch when a node fails.
    :param quit_on_tree_error: Whether to stop the tree when a node fails.
    """

    def __init__(
        self,
        root: Optional[Node] = None,
        num_workers: int = 1,
        forward_results: bool = False,
        logger: Logger | None = None,
        cutoff_branch_on_error: bool = False,
        quit_tree_on_error: bool = False,
    ):
        self._root = root
        self._queue = asyncio.Queue()
        self._num_workers = num_workers
        self._forward_results = forward_results
        self._output = dict()

        self._logger = logger
        self._cutoff_branch_on_error = cutoff_branch_on_error
        self._quit_tree_on_error = quit_tree_on_error
        self._visited_nodes = set()

    @property
    def root(self):
        return self._root

    @property
    def queue(self):
        return self._queue

    @property
    def num_workers(self):
        return self._num_workers

    @property
    def forward_results(self):
        return self._forward_results

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

    def __or__(self, tree_dict: dict[Node, Any]):
        """
        Override the `|` operator to create an instance of AsyncTreeExecutor and build the tree.
        """
        self.__build_tree(tree_dict)
        return self

    async def __worker(self):
        """
        A worker that executes the work contained in a Node.
        """
        result = None
        while True:
            node: Node = await self._queue.get()
            if node is None:
                break

            try:
                result = await node.run()
                self._visited_nodes.add(node.uuid)
            except Exception as e:
                result = e
                if self._logger:
                    self._logger.error(f"Error running node {node.uuid}: {e}")

            if isinstance(result, Exception) and self._cutoff_branch_on_error:
                self._queue.task_done()
                if self._logger:
                    self._logger.error(
                        f"Stopping branch at node {node.uuid} due to error."
                    )
                break

            if isinstance(result, Exception) and self._quit_tree_on_error:
                self._queue.task_done()
                await self.__stop_all_workers()
                if self._logger:
                    self._logger.error(
                        f"Stopping tree at node {node.uuid} due to error."
                    )
                break

            self._output[str(node.uuid)] = result
            node.set_output(result)

            ###########################################################################
            # Instead of doing this, we should check isinstace(node, PickerNode) and then execute that class' pick method
            if isinstance(node, PickerNode):
                children = await node.choose()
            # if node.picker:
            #     try:
            #         children = await node.picker(node, result, node.children) or []
            #     except Exception as e:
            #         if self._logger:
            #             self._logger.error(
            #                 f"Error picking children for node {node.uuid}: {e}"
            #             )
            #         children = []
            else:
                children = node.children or []
            ###########################################################################

            for child in children:
                if self._forward_results:
                    if isinstance(result, list):
                        child.update(args=result)
                    else:
                        child.update(args=[result])

                if child.uuid not in self._visited_nodes:
                    self._queue.put_nowait(child)

            self._queue.task_done()

    async def __stop_all_workers(self):
        """
        Queue a None for each worker to stop them.
        """
        for _ in range(self._num_workers):
            await self._queue.put(None)

    def __validate_element(self, obj: Any):
        "Check if an object is an instance of Node."
        if not isinstance(obj, Node):
            raise ValueError(f"Object is not a Node instance. Object: {obj}")

    def __build_tree(self, tree_dict: dict[Node, Any]):
        """
        Builds a tree of nodes from nested dictionaries.

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
        """
        if not isinstance(tree_dict, dict):
            raise ValueError("'tree_dict' must be a dictionary.")
        elif len(tree_dict) != 1:
            raise ValueError("'tree_dict' must contain exactly one root node.")

        self._num_workers = 0

        def connect_children(
            parent_node: Node, children_iterable: dict[Node, Any] | list[Node]
        ):
            self.__validate_element(parent_node)

            if isinstance(children_iterable, dict):
                for child_node, descendants_iterable in children_iterable.items():
                    self.__validate_element(child_node)

                    parent_node.connect(child_node)
                    self._num_workers += 1
                    connect_children(child_node, descendants_iterable)

            elif isinstance(children_iterable, list):
                for child_node in children_iterable:
                    self.__validate_element(child_node)
                    parent_node.connect(child_node)
                    self._num_workers += 1

        for root_node, children_iterable in tree_dict.items():
            self._root = root_node

            if isinstance(children_iterable, dict):
                connect_children(root_node, children_iterable)

            elif isinstance(children_iterable, list):
                for child_node in children_iterable:
                    self.__validate_element(child_node)
                    root_node.connect(child_node)
                    self._num_workers += 1

    async def run(self):
        """
        Runs the tree with the specified number of workers.
        """
        await self._queue.put(self.root)

        workers = [
            asyncio.create_task(self.__worker()) for _ in range(self._num_workers)
        ]

        await self._queue.join()
        await self.__stop_all_workers()
        await asyncio.gather(*workers)

        return self._output
