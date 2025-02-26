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

    :param roots: The root node(s) of the tree. Can be a single Node or a list of Nodes.
    :param num_workers: The number of workers to process the tree. If multiple roots are provided,
                        this will be automatically adjusted to at least the number of roots.
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
        roots: Optional[list[Node]] = None,
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

        self._uuid = uuid or str(uuid4())
        self._roots = roots or []
        self._num_workers = num_workers
        self._min_workers = min_workers
        self._max_workers = max_workers
        self._use_dynamic_workers = use_dynamic_workers

        self._workers = []
        self._output = []

        self._queue = asyncio.Queue()
        self._enqueued_nodes = set()
        self._lock = asyncio.Lock()
        self._stop: asyncio.Event = asyncio.Event()

        # Adjust number of workers if multiple roots are provided
        if self._use_dynamic_workers:
            self._num_workers = max(len(self._roots), min_workers)
            if self._num_workers > max_workers:
                self._num_workers = max_workers
            logger.debug(
                f"Initial number of workers set to {self._num_workers} to accommodate {len(self._roots)} root node(s)."
            )

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
        treat them as multiple root nodes.
        """
        if self._roots:
            raise ValueError(
                "Root nodes have been provided, indicating a manual tree construction. Cannot use the | operator syntax."
            )

        # Store the tree structure for later async processing
        if len(tree_dict) == 1:
            root_node, children_iterable = next(iter(tree_dict.items()))
            self._roots = [root_node]
            self._pending_connections = [(root_node, children_iterable)]
        else:
            # Multiple roots in the dictionary
            self._roots = list(tree_dict.keys())
            self._pending_connections = [
                (node, tree_dict[node]) for node in self._roots
            ]

            # Adjust workers based on number of roots
            if len(self._roots) > self._num_workers:
                self._num_workers = len(self._roots)
                if self._num_workers > self._max_workers:
                    self._max_workers = self._num_workers

        return self

    async def _build_tree(self):
        """Helper method to build the tree structure asynchronously"""
        if not hasattr(self, "_pending_connections"):
            return

        async def connect_children(
            parent_node: Node, children_iterable: dict[Node, Any] | list[Node]
        ):
            if isinstance(children_iterable, dict):
                for child_node, descendants_iterable in children_iterable.items():
                    await parent_node.connect(child_node)
                    await connect_children(child_node, descendants_iterable)
            elif isinstance(children_iterable, list):
                for child_node in children_iterable:
                    await parent_node.connect(child_node)

        for parent_node, children_iterable in self._pending_connections:
            if isinstance(children_iterable, dict):
                await connect_children(parent_node, children_iterable)
            elif isinstance(children_iterable, list):
                for child_node in children_iterable:
                    await parent_node.connect(child_node)

        delattr(self, "_pending_connections")

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
                await node.run_wrapper()

                # Enqueue children
                for child in node.children:
                    if child in self._enqueued_nodes:
                        continue
                    self._queue.put_nowait(child)
                    self._enqueued_nodes.add(child)

                # Adjust workers based on queue size and current worker count
                if self._use_dynamic_workers:
                    async with self._lock:
                        if self._queue.qsize() > len(self._workers):
                            workers_to_add = min(
                                self._max_workers - len(self._workers),
                                len(node.children),
                            )
                            for _ in range(workers_to_add):
                                self._workers.append(
                                    asyncio.create_task(self.__worker())
                                )
                            logger.debug(
                                f"Added {workers_to_add} workers. Current workers: {len(self._workers)}"
                            )
                        else:
                            workers_to_remove = min(
                                len(self._workers) - self._queue.qsize(),
                                len(self._workers) - self._min_workers,
                            )
                            for _ in range(workers_to_remove):
                                self._queue.put_nowait(None)
                                self._workers.pop()
                            logger.debug(
                                f"Removed {workers_to_remove} workers. Current workers: {len(self._workers)}"
                            )

                self._output.append(node)
            except Exception as e:
                logger.error(f"Error on {node}: {e}", exc_info=True)
                self._stop.set()
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
        await self._build_tree()  # Build the tree before running

        # Enqueue all root nodes
        for root in self._roots:
            self._queue.put_nowait(root)
            self._enqueued_nodes.add(root)

        self._workers = [
            asyncio.create_task(self.__worker()) for _ in range(self._num_workers)
        ]

        if len(self._workers) == 0:
            raise ValueError("No workers were created.")

        logger.debug(
            f"Running {' {}'.format(self._uuid) if self._uuid else ''} with {len(self._roots)} root nodes..."
        )

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
        await self._build_tree()

        # Enqueue all root nodes
        for root in self._roots:
            self._queue.put_nowait(root)
            self._enqueued_nodes.add(root)

        self._workers = [
            asyncio.create_task(self.__worker()) for _ in range(self._num_workers)
        ]

        if len(self._workers) == 0:
            raise ValueError("No workers were created.")

        logger.debug(
            f"Running {'{}'.format(self._uuid) if self._uuid else ''} with {len(self._roots)} root nodes..."
        )

        while not self._stop.is_set():
            if self._queue.empty():
                break
            while self._output:
                node = self._output.pop(0)
                yield node

            await asyncio.sleep(latency)  # Small delay to prevent busy-waiting

        await self._queue.join()
        await self.stop_tree()
        await asyncio.gather(*self._workers, return_exceptions=True)

        logger.debug(f"{self._uuid} complete.")
        # ? REASON: Yield any remaining results safely
        while self._output:
            yield self._output.pop(0)
