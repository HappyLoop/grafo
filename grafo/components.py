import asyncio
import inspect
import time
from typing import (
    get_args,
    Any,
    AsyncGenerator,
    Callable,
    Generic,
    Optional,
    TypeVar,
)
from uuid import uuid4

from grafo._internal import (
    AwaitableCallback,
    Chunk,
    Metadata,
    OnForwardCallable,
    logger,
)
from grafo.errors import (
    ForwardingOverrideError,
    MismatchChunkType,
    NotAsyncCallableError,
    SafeExecutionError,
)


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


N = TypeVar("N")


class Node(Generic[N]):
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
        coroutine: AwaitableCallback,
        kwargs: Optional[dict[str, Any]] = None,
        uuid: Optional[str] = None,
        timeout: Optional[float] = 60.0,
        on_connect: Optional[tuple[AwaitableCallback, Optional[dict[str, Any]]]] = None,
        on_disconnect: Optional[
            tuple[AwaitableCallback, Optional[dict[str, Any]]]
        ] = None,
        on_before_run: Optional[
            tuple[AwaitableCallback, Optional[dict[str, Any]]]
        ] = None,
        on_after_run: Optional[
            tuple[AwaitableCallback, Optional[dict[str, Any]]]
        ] = None,
    ):
        self.uuid: str = uuid or str(uuid4())
        self.coroutine: Callable = coroutine
        self.kwargs: dict[str, Any] = kwargs or {}
        self.metadata: Metadata = Metadata(runtime=0, level=0)
        self.children: list["Node"] = []

        # * Output
        self._output: Optional[N] = None
        self._aggregated_output: list[N] = []

        # * Events
        self.on_connect = on_connect
        self.on_disconnect = on_disconnect
        self.on_before_run = on_before_run
        self.on_after_run = on_after_run

        # * Inner flags
        self._event: asyncio.Event = asyncio.Event()
        self._is_running: bool = False
        self._parent_events: list[asyncio.Event] = []
        self._timeout: Optional[float] = timeout
        self._forward_map: dict[
            str,
            tuple[
                str,
                Optional[tuple[OnForwardCallable, Optional[dict[str, Any]]]],
            ],
        ] = {}
        if not timeout:
            logger.warning(
                "Node %s was given no timeout. Defaulting to 60 seconds to avoid running indefinitely.",
                self.uuid,
            )

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
    def output(self) -> N | None:
        if inspect.isasyncgenfunction(self.coroutine):
            raise NotAsyncCallableError(
                "Node contains a yielding coroutine. Use `aggregated_output` instead."
            )
        return self._output

    @property
    def aggregated_output(self) -> list[N]:
        if not inspect.isasyncgenfunction(self.coroutine):
            raise NotAsyncCallableError(
                "Node does not contain a yielding coroutine. Use `output` instead."
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
        self.metadata = Metadata(runtime=self.metadata.runtime, level=level)

    def _eval_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """
        Evaluates any lambda functions in kwargs, leaving other objects unchanged.
        Returns a new dict with evaluated values.
        """
        lambda_type = type(lambda: None)
        return {k: v() if isinstance(v, lambda_type) else v for k, v in kwargs.items()}

    async def _run_callback(
        self,
        prop: tuple[AwaitableCallback, Optional[dict[str, Any]]],
        **kwargs,
    ):
        """
        Runs a callback with the given fixed kwargs.
        """
        callback, fixed_kwargs = prop
        if not inspect.iscoroutinefunction(callback):
            raise NotAsyncCallableError("Callback must be a coroutine function")
        runtime_kwargs = self._eval_kwargs(fixed_kwargs or {})
        if kwargs:
            runtime_kwargs.update(kwargs)
        return await callback(**runtime_kwargs)

    async def connect(
        self,
        child: "Node",
        forward_as: Optional[str] = None,
        on_before_forward: tuple[OnForwardCallable, Optional[dict[str, Any]]]
        | None = None,
    ):
        """
        Connects a child to this node.

        :param child: The child node to connect.
        :param forward_as: Optional; the name of the argument to forward the output to.
        :param on_before_forward: Optional; a tuple (callback, fixed_kwargs) triggered before the output is forwarded to the child.

        >>> async def node_a_coroutine():
        ...     return 1
        >>> async def node_b_coroutine(output: int):
        ...     return output + 1
        >>> async def on_before_forward(output: int):
        ...     return output + 1
        >>> node_a = Node(uuid="nodeA", coroutine=node_a_coroutine)
        >>> node_b = Node(uuid="nodeB", coroutine=node_b_coroutine)
        >>> await node_a.connect(node_b, forward_as="output", on_before_forward=(on_before_forward, {}))
        """
        self.children.append(child)
        child._add_event(self._event)
        if self.metadata.level + 1 > child.metadata.level:
            child.set_level(self.metadata.level + 1)
        if self.on_connect:
            await self._run_callback(self.on_connect)
        if forward_as:
            self._forward_map[child.uuid] = (forward_as, on_before_forward)

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
        if child.uuid in self._forward_map:
            del self._forward_map[child.uuid]

    async def redirect(self, targets: list["Node"]):
        """
        Convenience method to disconnect all children and connect to a new target.
        """
        for child in self.children:
            await self.disconnect(child)
        for target in targets:
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
            self._is_running = True
            runtime_kwargs = self._eval_kwargs(self.kwargs)
            self._output = await self.coroutine(**runtime_kwargs)
            self._event.set()
        finally:
            self._is_running = False
            end_time = time.time()
            self.metadata = Metadata(
                runtime=end_time - start_time, level=self.metadata.level
            )

    @safe_execution
    async def _run_yielding(self) -> AsyncGenerator[Chunk[N], None]:
        """
        Asynchronously runs the coroutine of in this node.
        """
        try:
            start_time = time.time()
            self._is_running = True
            runtime_kwargs = self._eval_kwargs(self.kwargs)
            async for result in self.coroutine(**runtime_kwargs):
                self._aggregated_output.append(result)
                self._output = result
                if not isinstance(result, Chunk):
                    yield Chunk[N](self.uuid, result)
                else:
                    # TODO: check this
                    if hasattr(self, "__orig_class__"):
                        runtime_type = get_args(self.__orig_class__)[0]  # type: ignore
                        if runtime_type != Any and not isinstance(
                            result.output, runtime_type
                        ):
                            raise MismatchChunkType(
                                f"Node {self} yielded a chunk of type {type(result.output)} but expected {runtime_type}"
                            )
                    yield result

            self._event.set()
        finally:
            self._is_running = False
            end_time = time.time()
            self.metadata = Metadata(
                runtime=end_time - start_time, level=self.metadata.level
            )

    async def _forward_output(self):
        """
        Forwards the output of this node to a child.
        """
        for child in self.children:
            if child.uuid in self._forward_map:
                forward_as, on_before_forward = self._forward_map[child.uuid]
                if forward_as in child.kwargs:
                    raise ForwardingOverrideError(
                        f"{self} is trying to forward its output as `{forward_as}` to {child} but it already has an argument with that name."
                    )

                forward_data = self._output
                if inspect.isasyncgenfunction(self.coroutine):
                    forward_data = self._aggregated_output
                if on_before_forward:
                    forward_data = await self._run_callback(
                        on_before_forward, forward_data=forward_data
                    )
                child.kwargs[forward_as] = forward_data

    async def run(self) -> None:
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
        await self._forward_output()
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
        await self._forward_output()
        await self._on_after_run()
