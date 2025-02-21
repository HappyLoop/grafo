from collections import namedtuple
import inspect
import time
from typing import Any, Callable, Generic, Optional, TypeVar

import asyncio
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


Metadata = namedtuple("Metadata", ["runtime"])
T = TypeVar("T")


class Node(Generic[T]):
    """
    A Node is a unit of work that can be executed concurrently. It contains a coroutine function that is executed by a worker.

    :param coroutine: The coroutine function to execute.
    :param kwargs: Optional; the keyword arguments to pass to the coroutine.
    :param uuid: The unique identifier of the node.
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
        shared_lock: Optional[asyncio.Lock] = None,
    ):
        self.uuid: str = uuid or str(uuid4())
        if not timeout:
            logger.warning(
                "Node %s has no timeout, which can cause trees to run indefinitely. Consider setting a timeout.",
                self.uuid,
            )

        self.coroutine: Callable = coroutine
        self.kwargs: dict[str, Any] = kwargs if kwargs is not None else {}
        self.state: dict = state or {}
        self.metadata: Metadata = Metadata(
            runtime=0,
        )
        self.on_connect = on_connect
        self.on_disconnect = on_disconnect
        self.on_before_run = on_before_run
        self.on_after_run = on_after_run

        self.shared_lock = shared_lock

        self.children: list["Node"] = []
        self.output: Optional[T] = None

        # * Inner flags
        self._event: asyncio.Event = asyncio.Event()
        self._is_running: bool = False
        self._parent_events: list[asyncio.Event] = []
        self._timeout: Optional[float] = timeout

    def __repr__(self) -> str:
        return f"Node(uuid={self.uuid})"

    def __setattr__(self, name: str, value: Any) -> None:
        # ? REASON: check if _is_running exists to avoid interfering during __init__
        if (
            name not in ["_is_running", "output"]
            and hasattr(self, "_is_running")
            and self._is_running
        ):
            raise SafeExecutionError(
                f"Cannot change property '{name}' while the node is running."
            )
        super().__setattr__(name, value)

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

    def _eval_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """
        Evaluates any lambda functions in kwargs, leaving other objects unchanged.
        Returns a new dict with evaluated values.
        """
        lambda_type = type(lambda: None)
        return {k: v() if isinstance(v, lambda_type) else v for k, v in kwargs.items()}

    async def connect(self, child: "Node[T]"):
        """
        Connects a child to this node.
        """
        self.children.append(child)
        child._add_event(self._event)
        if self.on_connect:
            if not inspect.iscoroutinefunction(self.on_connect[0]):
                raise ValueError("on_connect must be a coroutine function")
            callback, fixed_kwargs = self.on_connect
            runtime_kwargs = self._eval_kwargs(fixed_kwargs or {})
            await callback(**runtime_kwargs)

    async def disconnect(self, child: "Node[T]"):
        """
        Disconnects a child from this node.
        """
        self.children.remove(child)
        child._remove_event(self._event)
        if self.on_disconnect:
            if not inspect.iscoroutinefunction(self.on_disconnect[0]):
                raise ValueError("on_disconnect must be a coroutine function")
            callback, fixed_kwargs = self.on_disconnect
            runtime_kwargs = self._eval_kwargs(fixed_kwargs or {})
            await callback(**runtime_kwargs)

    @safe_execution
    async def run(self) -> Any:
        """
        Asynchronously runs the coroutine of in this node.
        """
        await asyncio.wait_for(
            asyncio.gather(*[e.wait() for e in self._parent_events]),
            timeout=self._timeout,
        )

        try:
            start_time = time.time()
            self._is_running = True

            if self.on_before_run:
                if not inspect.iscoroutinefunction(self.on_before_run[0]):
                    raise ValueError("on_before_run must be a coroutine function")
                callback, fixed_kwargs = self.on_before_run
                runtime_kwargs = self._eval_kwargs(fixed_kwargs or {})
                await callback(**runtime_kwargs)

            runtime_kwargs = self._eval_kwargs(self.kwargs)
            self.output = await self.coroutine(**runtime_kwargs)
            self._event.set()

            if self.on_after_run:
                if not inspect.iscoroutinefunction(self.on_after_run[0]):
                    raise ValueError("on_after_run must be a coroutine function")
                callback, fixed_kwargs = self.on_after_run
                runtime_kwargs = self._eval_kwargs(fixed_kwargs or {})
                await callback(**runtime_kwargs)
        finally:
            self._is_running = False
            end_time = time.time()
            self.metadata = Metadata(runtime=end_time - start_time)
            logger.debug(
                f"{self} coroutine completed in {self.metadata.runtime} seconds"
            )
