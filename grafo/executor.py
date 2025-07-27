import asyncio
import asyncio.log
import inspect
import time
from typing import Any, AsyncGenerator, Generic, Optional, TypeVar
from uuid import uuid4

from grafo._internal import logger

from .components import Chunk, Node

N = TypeVar("N")
C = TypeVar("C")


class AsyncTreeExecutor(Generic[N, C]):
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
        description: Optional[str] = "",
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
        self._description = description
        self._roots = roots or []
        self._num_workers = num_workers
        self._min_workers = min_workers
        self._max_workers = max_workers
        self._use_dynamic_workers = use_dynamic_workers

        self._workers = []
        self._output_nodes: list[Node[Any]] = []
        self._output_values: list[Chunk[Any]] = []
        self._errors = []

        self._queue = asyncio.Queue()
        self._enqueued_nodes = set()  # ? REASON: avoid duplicate nodes being enqueued
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

    def __repr__(self):
        expression = (
            f"UUID: {self._uuid}\nDescription: {self._description}\nStructure:\n"
        )
        for root in self._roots:
            expression += f"\tRoot {root.uuid}:\n"
            branch_expression, _ = self.__branch_depth_first_search(root)
            expression += branch_expression
        return expression

    @property
    def name(self):
        return self._uuid

    @property
    def results(self) -> list[Chunk]:
        return self._output_values

    def __branch_depth_first_search(
        self, node: Node, expression: str = "", leaf_nodes: list[Node] | None = None
    ):
        if leaf_nodes is None:
            leaf_nodes = []
        if len(node.children) == 0:
            leaf_nodes.append(node)
        for child in node.children:
            expression += f"\t\t{node.uuid} -> {child.uuid}\n"
            expression, childless_nodes = self.__branch_depth_first_search(
                child, expression
            )
            leaf_nodes.extend(childless_nodes)
        return expression, leaf_nodes

    async def __adjust_dynamic_workers(self, node: Node):
        """
        Adjusts the number of workers based on the queue size and current worker count.
        """
        if not self._use_dynamic_workers:
            return
        async with self._lock:
            if self._queue.qsize() > len(self._workers):
                workers_to_add = min(
                    self._max_workers - len(self._workers),
                    len(node.children),
                )
                for _ in range(workers_to_add):
                    self._workers.append(asyncio.create_task(self.__worker()))
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
                if inspect.isasyncgenfunction(node.coroutine):
                    async for result in node.run_yielding():
                        self._output_values.append(result)
                else:
                    await node.run()
                self._output_nodes.append(node)

                # Enqueue children
                for child in node.children:
                    if child not in self._enqueued_nodes:
                        self._enqueued_nodes.add(child)
                        self._queue.put_nowait(child)
                await self.__adjust_dynamic_workers(node)

                # Add node to output nodes and tree results
                self._enqueued_nodes.remove(node)
            except Exception as e:
                self._errors.append(e)
                logger.error(
                    f"{'|   ' * (node.metadata.level - 1) + ('|---' if node.metadata.level > 0 else '')}\033[4;31mError\033[0m on {node}: {e}",
                    exc_info=True,
                )
                self._stop.set()
                if not node._is_running:
                    raise e
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

    async def run(self) -> list[Node[N]]:
        """
        Runs the tree with the specified number of workers.
        """
        levels = []
        for root in self._roots:
            levels.append(root.metadata.level)
            self._queue.put_nowait(root)
            self._enqueued_nodes.add(root)
        base_level = min(levels)

        self._workers = [
            asyncio.create_task(self.__worker()) for _ in range(self._num_workers)
        ]

        if len(self._workers) == 0:
            raise ValueError("No workers were created.")

        logger.info(
            f"{'|   ' * (base_level - 1) + ('|---' if base_level > 0 else '')}\033[4m\033[90mRunning {'{}'.format(self._uuid) if self._uuid else ''} with {len(self._roots)} root node(s)...\033[0m"
        )
        start_time = time.time()

        await self._queue.join()
        await self.stop_tree()
        await asyncio.gather(*self._workers, return_exceptions=True)

        end_time = time.time()
        logger.info(
            f"{'|   ' * (base_level - 1) + ('|---' if base_level > 0 else '')}\033[4m\033[90m{self._uuid} complete in {end_time - start_time:.2f} seconds.\033[0m"
        )
        return self._output_nodes

    async def yielding(
        self,
        latency: float = 0.01,
    ) -> AsyncGenerator[Node[N] | Chunk[C], None]:
        """
        Runs the tree with the specified number of workers and yields results as they are set.
        """
        levels = []
        for root in self._roots:
            levels.append(root.metadata.level)
            self._queue.put_nowait(root)
            self._enqueued_nodes.add(root)
        base_level = min(levels)

        self._workers = [
            asyncio.create_task(self.__worker()) for _ in range(self._num_workers)
        ]

        if len(self._workers) == 0:
            raise ValueError("No workers were created.")

        logger.info(
            f"{'|   ' * (base_level - 1) + ('|---' if base_level > 0 else '')}\033[4m\033[90mRunning {'{}'.format(self._uuid) if self._uuid else ''} with {len(self._roots)} root node(s)...\033[0m"
        )
        start_time = time.time()

        while len(self._enqueued_nodes) > 0 or self._output_nodes:
            while self._output_nodes:
                yield self._output_nodes.pop(0)
            while self._output_values:
                yield self._output_values.pop(0)
            if (
                self._stop.is_set()
                and not self._output_nodes
                and not self._output_values
            ):
                break
            await asyncio.sleep(latency)  # ? REASON: prevent busy-waiting

        await self._queue.join()
        await self.stop_tree()
        await asyncio.gather(*self._workers, return_exceptions=True)

        # ? REASON: Yield any remaining results safely
        while self._output_nodes:
            yield self._output_nodes.pop(0)

        end_time = time.time()
        logger.info(
            f"{'|   ' * (base_level - 1) + ('|---' if base_level > 0 else '')}\033[4m\033[90m{self._uuid} complete in {end_time - start_time:.2f} seconds.\033[0m"
        )

    def get_leaves(self) -> list[Node]:
        """
        Returns the leaf nodes of the tree.

        ATTENTION: The node states are returned as they were during the time of this method's execution.
        """
        leaf_nodes = []
        for root in self._roots:
            _, branch_leaf_nodes = self.__branch_depth_first_search(root)
            for leaf in branch_leaf_nodes:
                if leaf not in leaf_nodes:
                    leaf_nodes.append(leaf)
        return leaf_nodes
