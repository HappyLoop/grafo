import asyncio
import asyncio.log
from typing import Any, AsyncGenerator, Optional
from uuid import uuid4
from grafo._internal import logger

from .components import Node


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
        uuid: Optional[str] = None,
        root: Optional[Node] = None,
        num_workers: int = 1,
        min_workers: int = 1,
        max_workers: int = 10,
        use_dynamic_workers: bool = True,
    ):
        if num_workers < min_workers:
            raise ValueError(
                "'num_workers' must be greater than or equal to 'min_workers'."
            )
        if max_workers < num_workers:
            raise ValueError(
                "'max_workers' must be greater than or equal to 'num_workers'."
            )

        self._uuid = uuid
        self._root = root
        self._queue = asyncio.Queue()
        self._num_workers = num_workers
        self._min_workers = min_workers
        self._max_workers = max_workers
        self._use_dynamic_workers = use_dynamic_workers
        self._workers = []
        self._output = []

        self._enqueued_nodes = set()
        self._stop: asyncio.Event = asyncio.Event()

    @property
    def name(self):
        return self._uuid

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
            if isinstance(children_iterable, dict):
                for child_node, descendants_iterable in children_iterable.items():
                    parent_node.connect(child_node)

                    connect_children(child_node, descendants_iterable)

            elif isinstance(children_iterable, list):
                for child_node in children_iterable:
                    parent_node.connect(child_node)

        for root_node, children_iterable in tree_dict.items():
            self._root = root_node

            if isinstance(children_iterable, dict):
                connect_children(root_node, children_iterable)

            elif isinstance(children_iterable, list):
                for child_node in children_iterable:
                    root_node.connect(child_node)

        return self

    async def __worker(self):
        """
        A worker that executes the work contained in a Node.
        """
        while True:
            node: Node = await self._queue.get()

            if node is None or self._stop.is_set():
                self._queue.task_done()
                break

            try:
                # Run the node
                logger.debug(f"Running {node}")
                await node.run()

                # Enqueue children and adjust workers
                for child in node.children:
                    if child in self._enqueued_nodes:
                        continue
                    self._queue.put_nowait(child)
                    self._enqueued_nodes.add(child)
                    if (
                        self._use_dynamic_workers
                        and self._queue.qsize() > len(self._workers)
                        and len(self._workers) < self._max_workers
                    ):
                        new_worker = asyncio.create_task(self.__worker())
                        self._workers.append(new_worker)
                        logger.debug(
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
                    logger.debug(
                        f"Removed worker. Current workers: {len(self._workers)}"
                    )

                self._output.append(node)
            except Exception as e:
                self._stop.set()
                logger.error(f"Error on {node}: {e}")
                break
            finally:
                self._queue.task_done()

    async def stop_tree(self):
        """
        Gracefully stops all workers.
        """
        self._stop.set()
        for _ in range(self._num_workers):
            self._queue.put_nowait(None)

    async def run(self) -> list[Node]:
        """
        Runs the tree with the specified number of workers.
        """
        self._queue.put_nowait(self._root)
        self._workers = [
            asyncio.create_task(self.__worker()) for _ in range(self._num_workers)
        ]

        if len(self._workers) == 0:
            raise ValueError("No workers were created.")

        logger.debug(f"Running {' {}'.format(self._uuid) if self._uuid else ''}...")

        await self._queue.join()
        await self.stop_tree()
        await asyncio.gather(*self._workers, return_exceptions=True)

        logger.debug("Tree execution complete.")
        return self._output

    async def yielding(
        self,
        latency: float = 0.05,
    ) -> AsyncGenerator[Node, None]:
        """
        Runs the tree with the specified number of workers and yields results as they are set.
        """
        self._queue.put_nowait(self._root)
        self._workers = [
            asyncio.create_task(self.__worker()) for _ in range(self._num_workers)
        ]

        if len(self._workers) == 0:
            raise ValueError("No workers were created.")

        logger.debug(f"Running {'{}'.format(self._uuid) if self._uuid else ''}...")

        while any(not worker.done() for worker in self._workers):
            if self._stop.is_set():
                break

            while self._output:
                node = self._output.pop(0)
                yield node

            await asyncio.sleep(
                latency
            )  # ? REASON: Small delay to prevent busy-waiting

        await self._queue.join()
        await self.stop_tree()
        await asyncio.gather(*self._workers, return_exceptions=True)

        logger.debug(f"{self._uuid} complete.")
        # ? REASON: Yield any remaining results safely
        while self._output:
            yield self._output.pop(0)
