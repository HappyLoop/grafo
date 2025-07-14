import asyncio
import inspect
import time
from collections import namedtuple
from typing import Any, AsyncGenerator, Callable, Generic, Optional, TypeVar
from uuid import uuid4

from grafo._internal import logger


class SafeExecutionError(Exception):
    """
    Exception raised when a method is called on a running node.
    """


def safe_execution(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator that prevents a method from being called if the node is already running.
    """

    def wrapper(self: "Node", *args: Any, **kwargs: Any) -> Any:
        if not self._is_running:
            return func(self, *args, **kwargs)
        raise SafeExecutionError(
            f"Skipped <{func.__name__}> call because {self} is running."
        )  # NOTE: this error will stop the entire process

    return wrapper


Metadata = namedtuple("Metadata", ["runtime", "level"])
T = TypeVar("T")


class Node(Generic[T]):
    """
    A Node is a unit of work that can be executed concurrently. It contains a coroutine function that is executed by a worker.

    :param coroutine: The coroutine function to execute.
    :param kwargs: Optional; the keyword arguments to pass to the coroutine.
    :param uuid: The unique identifier of the node.
    :param state: Optional; a dict containing some state to be held by the node.
    :param metadata: Optional; a dict containing at least "name" and "description" for the node.
    :param timeout: Optional; the timeout for the node. If not provided, a warning will be logged.

    Additionally, the following optional event callback parameters can be provided as a tuple:
      (callback, fixed_kwargs)
    where fixed_kwargs is an optional dict of fixed keyword arguments (defaulting to an empty dict if not provided).

    :param on_connect: Optional; a tuple (callback, fixed_kwargs) triggered when `connect()` is called.
    :param on_disconnect: Optional; a tuple (callback, fixed_kwargs) triggered when `disconnect()` is called.
    :param on_before_run: Optional; a tuple (callback, fixed_kwargs) triggered before the node's coroutine is executed via `run()`.
    :param on_after_run: Optional; a tuple (callback, fixed_kwargs) triggered after the node's coroutine is executed via `run()`.

    **Important:** All coroutines and callbacks are automatically called with the node instance (self) as the first (positional) argument.
    """

    def __init__(
        self,
        coroutine: Callable,
        kwargs: Optional[dict[str, Any]] = None,
        uuid: Optional[str] = None,
        state: Optional[dict] = None,
        timeout: Optional[float] = 60.0,
        on_connect: Optional[
            tuple[Callable[..., Any], Optional[dict[str, Any]]]
        ] = None,
        on_disconnect: Optional[
            tuple[Callable[..., Any], Optional[dict[str, Any]]]
        ] = None,
        on_before_run: Optional[
            tuple[Callable[..., Any], Optional[dict[str, Any]]]
        ] = None,
        on_after_run: Optional[
            tuple[Callable[..., Any], Optional[dict[str, Any]]]
        ] = None,
        _shared_lock: Optional[asyncio.Lock] = None,
    ):
        self.uuid: str = uuid or str(uuid4())
        if not timeout:
            logger.warning(
                "Node %s was given no timeout. Defaulting to 60 seconds to avoid running indefinitely.",
                self.uuid,
            )

        self.coroutine: Callable = coroutine
        self.kwargs: dict[str, Any] = kwargs if kwargs is not None else {}
        self.state: dict = state or {}
        self.metadata: Metadata = Metadata(runtime=0, level=0)
        self.on_connect = on_connect
        self.on_disconnect = on_disconnect
        self.on_before_run = on_before_run
        self.on_after_run = on_after_run

        self.shared_lock = _shared_lock

        self.children: list["Node"] = []
        self._output: Optional[T] = None
        self._aggregated_output: list[T] = []

        # * Inner flags
        self._event: asyncio.Event = asyncio.Event()
        self._is_running: bool = False
        self._parent_events: list[asyncio.Event] = []
        self._timeout: Optional[float] = timeout

    def __repr__(self) -> str:
        return f"Node(uuid={self.uuid}, level={self.metadata.level})"

    def __setattr__(self, name: str, value: Any) -> None:
        # ? REASON: check if _is_running exists to avoid interfering during __init__
        if (
            name not in ["_level", "_is_running", "_output"]
            and hasattr(self, "_is_running")
            and self._is_running
        ):
            raise SafeExecutionError(
                f"Cannot change property '{name}' while the node is running."
            )
        super().__setattr__(name, value)

    @property
    def output(self) -> T | None:
        return self._output

    @property
    def aggregated_output(self) -> list[T] | None:
        if not inspect.isasyncgenfunction(self.coroutine):
            raise ValueError(
                "Cannot access aggregated_output because Node does not contain a coroutine that yields values."
            )
        return self._aggregated_output

    def _add_event(self, event: asyncio.Event):
        """
        Adds an event to this node so that it waits for it to be set before running.
        """
        self._parent_events.append(event)

    def _remove_event(self, event: asyncio.Event):
        """
        Removes an event from this node so that it no longer waits for it to be set before running.
        """
        self._parent_events.remove(event)

    def set_level(self, level: int):
        """
        Sets the level of this node.
        """
        self.metadata = self.metadata._replace(level=level)

    def _eval_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """
        Evaluates any lambda functions in kwargs, leaving other objects unchanged.
        Returns a new dict with evaluated values.
        """
        lambda_type = type(lambda: None)
        return {k: v() if isinstance(v, lambda_type) else v for k, v in kwargs.items()}

    async def _run_callback(
        self,
        prop: tuple[Callable[..., Any], Optional[dict[str, Any]]],
    ):
        """
        Runs a callback with the given fixed kwargs.
        """
        callback, fixed_kwargs = prop
        if not inspect.iscoroutinefunction(callback):
            raise ValueError("callback must be a coroutine function")
        runtime_kwargs = self._eval_kwargs(fixed_kwargs or {})
        await callback(**runtime_kwargs)

    async def connect(self, child: "Node"):
        """
        Connects a child to this node.
        """
        self.children.append(child)
        child._add_event(self._event)
        child.set_level(self.metadata.level + 1)
        if self.on_connect:
            await self._run_callback(self.on_connect)

    async def disconnect(self, child: "Node"):
        """
        Disconnects a child from this node.
        """
        if child not in self.children:
            logger.warning(
                f"{'|   ' * (self.metadata.level - 1) + ('|   ' if self.metadata.level > 0 else '')}\033[91m\033[4mWarning\033[0m {self} is trying to disconnect a child that is not in its children: {child}. No action taken."
            )
            return
        self.children.remove(child)
        child._remove_event(self._event)
        # ? NOTE: no level removal because nodes can have multiple parents
        if self.on_disconnect:
            await self._run_callback(self.on_disconnect)

    async def redirect(self, target: "Node"):
        """
        Convenience method to disconnect all children and connect to a new target.
        """
        for child in self.children:
            await self.disconnect(child)
        await self.connect(target)

    async def _on_before_run(self):
        """
        Runs the on_before_run callback.
        """
        if self.on_before_run:
            await self._run_callback(self.on_before_run)

    async def _on_after_run(self):
        """
        Runs the on_after_run callback.
        """
        if self.on_after_run:
            await self._run_callback(self.on_after_run)

    @safe_execution
    async def _run(self) -> Any:
        """
        Asynchronously runs the coroutine of in this node.
        """
        try:
            start_time = time.time()
            logger.info(
                f"{'|   ' * self.metadata.level}\033[4m\033[93mRunning\033[0m {self}"
            )
            self._is_running = True

            runtime_kwargs = self._eval_kwargs(self.kwargs)
            self._output = await self.coroutine(**runtime_kwargs)
            self._event.set()
        finally:
            self._is_running = False
            end_time = time.time()
            self.metadata = self.metadata._replace(runtime=end_time - start_time)
            logger.info(
                f"{'|   ' * (self.metadata.level - 1) + ('|   ' if self.metadata.level > 0 else '')}\033[92m\033[4mCompleted\033[0m {self} in {self.metadata.runtime} seconds"
            )

    @safe_execution
    async def _run_yielding(self) -> AsyncGenerator[Any, None]:
        """
        Asynchronously runs the coroutine of in this node.
        """
        try:
            start_time = time.time()
            logger.info(
                f"{'|   ' * self.metadata.level}\033[4m\033[93mRunning\033[0m {self}"
            )
            self._is_running = True

            runtime_kwargs = self._eval_kwargs(self.kwargs)
            async for result in self.coroutine(**runtime_kwargs):
                self._aggregated_output.append(result)
                self._output = result
                yield result
            self._event.set()
        finally:
            self._is_running = False
            end_time = time.time()
            self.metadata = self.metadata._replace(runtime=end_time - start_time)
            logger.info(
                f"{'|   ' * (self.metadata.level - 1) + ('|   ' if self.metadata.level > 0 else '')}\033[92m\033[4mCompleted\033[0m {self} in {self.metadata.runtime} seconds"
            )

    async def run(self) -> Any:
        """
        Wraps the run method to run the on_before_run and on_after_run callbacks.
        """
        logger.debug(f"{'|   ' * self.metadata.level}Awaiting {self} parents...")
        await asyncio.wait_for(
            asyncio.gather(*[e.wait() for e in self._parent_events]),
            timeout=self._timeout,
        )
        await self._on_before_run()
        await self._run()
        await self._on_after_run()

    async def run_yielding(self) -> AsyncGenerator[Any, None]:
        """
        Wraps the run method to run the on_before_run and on_after_run callbacks.
        """
        logger.debug(f"{'|   ' * self.metadata.level}Awaiting {self} parents...")
        await asyncio.wait_for(
            asyncio.gather(*[e.wait() for e in self._parent_events]),
            timeout=self._timeout,
        )
        await self._on_before_run()
        async for result in self._run_yielding():
            yield result
        await self._on_after_run()
