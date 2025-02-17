import inspect
from typing import Any, Callable, Optional, Self, Type

import asyncio

from grafo._internal import logger


def safe_execution(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator that prevents a method from being called if the node is already running.
    """

    def wrapper(self: "Node", *args: Any, **kwargs: Any) -> Any:
        if not self._is_running:
            return func(self, *args, **kwargs)
        # NOTE: on ELSE, error is not raised so as to not trigger branch/tree cutoff

    return wrapper


class Node:
    """
    A Node is a unit of work that can be executed concurrently. It contains a coroutine function that is executed by a worker.

    :param uuid: The unique identifier of the node.
    :param metadata: A dict containing at least "name" and "description" for the node.
    :param coroutine: The coroutine function to execute.
    :param args: The arguments to pass to the coroutine.
    :param kwargs: The keyword arguments to pass to the coroutine.
    :param children: The children nodes of this node.
    :param forward_output: Whether to forward the output of this node to its children as arguments.

    Additionally, the following optional event callback parameters can be passed to the constructor:

    :param on_connect: Optional; a callback triggered when the `connect()` method is called.
                       The callback receives the connected child as `child_`.
    :param on_connect_kwargs: Optional; additional fixed keyword arguments for the `on_connect` callback.

    :param on_disconnect: Optional; a callback triggered when the `disconnect()` method is called.
                          The callback receives the disconnected child as `child_`.
    :param on_disconnect_kwargs: Optional; additional fixed keyword arguments for the `on_disconnect` callback.

    :param on_update: Optional; a callback triggered when the `update()` method is called.
                      The callback receives the update parameters with an underscore appended:
                      `metadata_`, `children_`, `coroutine_`, `args_`, and `kwargs_`.
    :param on_update_kwargs: Optional; additional fixed keyword arguments for the `on_update` callback.

    :param on_before_run: Optional; a callback triggered before the node's coroutine is executed via `run()`.
                          No additional timed parameters are provided, but fixed kwargs can be passed.
    :param on_before_run_kwargs: Optional; additional fixed keyword arguments for the `on_before_run` callback.

    :param on_after_run: Optional; a callback triggered after the node's coroutine is executed via `run()`.
                         No additional timed parameters are provided, but fixed kwargs can be passed.
    :param on_after_run_kwargs: Optional; additional fixed keyword arguments for the `on_after_run` callback.

    **Note:** For each event callback in Node, if an associated function passes a parameter named `param`,
            the callback will receive that parameter as `param_`. For example, in `set_output(output)`,
            the callback is invoked with `output_=output`.
    """

    _output: Any

    def __init__(
        self,
        uuid: str,
        metadata: dict,
        coroutine: Callable,
        args: Optional[list[Any]] = None,
        kwargs: Optional[dict[str, Any]] = None,
        forward_output: Optional[bool] = False,
        on_connect: Optional[Callable[..., Any]] = None,
        on_connect_kwargs: Optional[dict[str, Any]] = None,
        on_disconnect: Optional[Callable[..., Any]] = None,
        on_disconnect_kwargs: Optional[dict[str, Any]] = None,
        on_update: Optional[Callable[..., Any]] = None,
        on_update_kwargs: Optional[dict[str, Any]] = None,
        on_before_run: Optional[Callable[..., Any]] = None,
        on_before_run_kwargs: Optional[dict[str, Any]] = None,
        on_after_run: Optional[Callable[..., Any]] = None,
        on_after_run_kwargs: Optional[dict[str, Any]] = None,
    ):
        self.__validate_param(uuid, "uuid", str)
        self.__validate_param(args, "args", list, allow_none=True)
        self.__validate_param(kwargs, "kwargs", dict, allow_none=True)
        self._metadata = metadata

        if not inspect.iscoroutinefunction(coroutine):
            raise ValueError(
                "The coroutine parameter must be a coroutine function (async function)."
            )

        self._uuid = uuid
        self._coroutine = coroutine
        self._args = args if args is not None else []
        self._kwargs = kwargs if kwargs is not None else {}
        self._output = None
        self._children = []
        self._is_running = False
        self._forward_output = forward_output
        self._event = asyncio.Event()

        self._on_connect_callback = (
            (on_connect, on_connect_kwargs or {}) if on_connect is not None else None
        )
        self._on_disconnect_callback = (
            (on_disconnect, on_disconnect_kwargs or {})
            if on_disconnect is not None
            else None
        )
        self._on_update_callback = (
            (on_update, on_update_kwargs or {}) if on_update is not None else None
        )
        self._on_before_run_callback = (
            (on_before_run, on_before_run_kwargs or {})
            if on_before_run is not None
            else None
        )
        self._on_after_run_callback = (
            (on_after_run, on_after_run_kwargs or {})
            if on_after_run is not None
            else None
        )

    def __repr__(self) -> str:
        return f"Node(uuid={self.uuid}, metadata={self.metadata}, output={self.output})"

    @property
    def uuid(self):
        return self._uuid

    @property
    def metadata(self):
        return self._metadata

    @property
    def children(self):
        return self._children

    @property
    def coroutine(self):
        return self._coroutine

    @property
    def args(self):
        return self._args

    @property
    def kwargs(self):
        return self._kwargs

    @property
    def output(self):
        return self._output

    def __validate_param(
        self, param: Any, param_name: str, expected_type: Type, allow_none: bool = False
    ):
        """
        Validates that a parameter is of the expected type.

        :param param: The parameter to validate.
        :param param_name: The name of the parameter (used in the error message).
        :param expected_type: The expected type of the parameter.
        :param allow_none: Whether None is allowed as a valid value.
        """
        if param is None and allow_none:
            return

        if not isinstance(param, expected_type):
            raise ValueError(
                f"'{param_name}' is required and must be of type {expected_type.__name__}."
            )

    def _run_event(self, callback: Callable, *args: Any, **kwargs: Any):
        """
        Executes an event callback which can be either synchronous or asynchronous.
        It checks whether we're in an async context (i.e. if there is a current task)
        and returns an awaitable if so; otherwise it runs the callback completely.

        If in an async context and the callback is synchronous, its result is wrapped
        as an awaitable via a zero-second asyncio.sleep call.
        """
        in_async = asyncio.current_task() is not None
        if inspect.iscoroutinefunction(callback):
            coro = callback(*args, **kwargs)
            if in_async:
                # In an async context, return the coroutine so the caller can await it.
                return coro
            else:
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    # No running loop; run the coroutine to completion.
                    return asyncio.run(coro)
                else:
                    return loop.run_until_complete(coro)
        else:
            result = callback(*args, **kwargs)
            if in_async:
                # Wrap synchronous result as an awaitable (immediately available).
                return asyncio.sleep(0, result=result)
            else:
                return result

    @safe_execution
    def connect(self, child: Self):
        """
        Connects a child to this node.
        """
        if not issubclass(type(child), Node):
            raise ValueError("The 'child' parameter must be a Node instance.")
        if isinstance(child, UnionNode) and isinstance(self, PickerNode):
            raise ValueError(
                "A UnionNode cannot have a PickerNode as a child. Use a PickerNode as a parent instead."
            )
        self.children.append(child)
        if isinstance(child, UnionNode):
            child._add_event(self._event)
        if self._on_connect_callback:
            callback, fixed_kwargs = self._on_connect_callback
            self._run_event(callback, child_=child, **fixed_kwargs)

    @safe_execution
    def disconnect(self, child: Self):
        """
        Disconnects a child from this node.
        """
        if not issubclass(type(child), Node):
            raise ValueError("The 'child' parameter must be a Node instance.")
        self.children.remove(child)
        if isinstance(child, UnionNode):
            child._remove_event(self._event)
        if self._on_disconnect_callback:
            callback, fixed_kwargs = self._on_disconnect_callback
            self._run_event(callback, child_=child, **fixed_kwargs)

    @safe_execution
    def update(
        self,
        metadata: Optional[dict] = None,
        children: Optional[list[Self]] = None,
        coroutine: Optional[Callable] = None,
        args: Optional[list[Any]] = None,
        kwargs: Optional[dict[str, Any]] = None,
        append_mode: bool = False,
    ):
        """
        Updates the node with new data.

        :param metadata: Optional; a dict containing the node's metadata.
        :param children: Optional; a list of children nodes.
        :param coroutine: Optional; a coroutine function to execute.
        :param args: Optional; a list of arguments to pass to the coroutine.
        :param kwargs: Optional; a dict of keyword arguments to pass to the coroutine.
        :param append_mode: Optional; if True, the args and kwargs will be appended to the existing data.
        """

        if metadata is not None:
            self._metadata = metadata

        if children is not None:
            self._children = children

        if coroutine is not None:
            if not inspect.iscoroutinefunction(coroutine):
                raise ValueError(
                    "The 'coroutine' parameter must be a coroutine function (async function)."
                )
            self._coroutine = coroutine

        self.__validate_param(args, "args", list, allow_none=True)
        self.__validate_param(kwargs, "kwargs", dict, allow_none=True)

        if append_mode:
            if args:
                self._args.extend(args)
            if kwargs:
                self._kwargs.update(kwargs)
        else:
            self._args = args if args is not None else []
            self._kwargs = kwargs if kwargs is not None else {}

        if self._on_update_callback:
            callback, fixed_kwargs = self._on_update_callback
            self._run_event(callback, children_=children, **fixed_kwargs)

    @safe_execution
    async def run(self) -> Any:
        """
        Asynchronously runs the coroutine of in this node.
        """
        logger.debug(f"Running {self}")
        if self._on_before_run_callback:
            callback, fixed_kwargs = self._on_before_run_callback
            # In an async context, await the event callback.
            await self._run_event(callback, **fixed_kwargs)

        try:
            self._is_running = True
            result = await self._coroutine(*self.args, **self.kwargs)
            self._output = result
            self._event.set()
            return result
        finally:
            if self._on_after_run_callback:
                callback, fixed_kwargs = self._on_after_run_callback
                # Await the after-run event callback.
                await self._run_event(callback, output_=result, **fixed_kwargs)
            self._is_running = False


class PickerNode(Node):
    """
    A node that uses an LLM to determine which children to queue next.

    :param uuid: The unique identifier of the node.
    :param metadata: A dict containing at least "name" and "description" for the node.
    :param coroutine: The coroutine (picker) function to execute.
    :param args: The arguments to pass to the coroutine.
    :param kwargs: The keyword arguments to pass to the coroutine.
    :param children: The children nodes of this node.
    :param forward_output: Whether to forward the output of this node to its children as arguments.

    Additionally, the following optional event callback parameters can be passed to the constructor:

    :param on_connect: Optional; a callback triggered when the `connect()` method is called.
                       The callback receives the connected child as `child_`.
    :param on_connect_kwargs: Optional; additional fixed keyword arguments for the `on_connect` callback.

    :param on_disconnect: Optional; a callback triggered when the `disconnect()` method is called.
                          The callback receives the disconnected child as `child_`.
    :param on_disconnect_kwargs: Optional; additional fixed keyword arguments for the `on_disconnect` callback.

    :param on_update: Optional; a callback triggered when the `update()` method is called.
                      The callback receives the update parameters with an underscore appended:
                      `metadata_`, `children_`, `coroutine_`, `args_`, and `kwargs_`.
    :param on_update_kwargs: Optional; additional fixed keyword arguments for the `on_update` callback.

    :param on_before_run: Optional; a callback triggered before the node's coroutine is executed via `run()`.
                          No additional timed parameters are provided, but fixed kwargs can be passed.
    :param on_before_run_kwargs: Optional; additional fixed keyword arguments for the `on_before_run` callback.

    :param on_after_run: Optional; a callback triggered after the node's coroutine is executed via `run()`.
                         No additional timed parameters are provided, but fixed kwargs can be passed.
    :param on_after_run_kwargs: Optional; additional fixed keyword arguments for the `on_after_run` callback.
    """

    def __init__(
        self,
        uuid: str,
        metadata: dict,
        coroutine: Callable,
        args: Optional[list[Any]] = None,
        kwargs: Optional[dict[str, Any]] = None,
        forward_output: Optional[bool] = False,
        on_connect: Optional[Callable[..., Any]] = None,
        on_connect_kwargs: Optional[dict[str, Any]] = None,
        on_disconnect: Optional[Callable[..., Any]] = None,
        on_disconnect_kwargs: Optional[dict[str, Any]] = None,
        on_update: Optional[Callable[..., Any]] = None,
        on_update_kwargs: Optional[dict[str, Any]] = None,
        on_before_run: Optional[Callable[..., Any]] = None,
        on_before_run_kwargs: Optional[dict[str, Any]] = None,
        on_after_run: Optional[Callable[..., Any]] = None,
        on_after_run_kwargs: Optional[dict[str, Any]] = None,
    ):
        super().__init__(
            uuid,
            metadata,
            coroutine,
            args,
            kwargs,
            forward_output,
            on_connect=on_connect,
            on_connect_kwargs=on_connect_kwargs,
            on_disconnect=on_disconnect,
            on_disconnect_kwargs=on_disconnect_kwargs,
            on_update=on_update,
            on_update_kwargs=on_update_kwargs,
            on_before_run=on_before_run,
            on_before_run_kwargs=on_before_run_kwargs,
            on_after_run=on_after_run,
            on_after_run_kwargs=on_after_run_kwargs,
        )

    def __repr__(self) -> str:
        return f"PickerNode(uuid={self.uuid}, metadata={self.metadata}, output={self.output})"

    @safe_execution
    async def run(self) -> list["Node"]:
        """
        Picks the children to queue next based on the result of the node.
        """
        if self._on_before_run_callback:
            callback, fixed_kwargs = self._on_before_run_callback
            # Await the before-run callback via _run_event.
            await self._run_event(callback, **fixed_kwargs)

        try:
            self._is_running = True
            result = await self.coroutine(
                self, self.children, *self.args, **self.kwargs
            )
            if not isinstance(result, list):
                raise ValueError("The picker coroutine must return a list of children.")
            for child in result:
                if not isinstance(child, Node):
                    raise ValueError(
                        "The picker coroutine must return a list of Node instances."
                    )
            self._output = result
            return result
        finally:
            if self._on_after_run_callback:
                callback, fixed_kwargs = self._on_after_run_callback
                # Await the after-run callback via _run_event.
                await self._run_event(callback, output_=result, **fixed_kwargs)
            self._is_running = False


class UnionNode(Node):
    """
    A node that waits for all its parents to finish executing before continuing.

    :param uuid: The unique identifier of the node.
    :param metadata: A dict containing at least "name" and "description" for the node.
    :param coroutine: The coroutine function to execute.
    :param args: The arguments to pass to the coroutine.
    :param kwargs: The keyword arguments to pass to the coroutine.
    :param parent_events: The events to wait for from the parent nodes.
    :param timeout: The timeout for waiting for parents to complete.
    :param forward_output: Whether to forward the output of this node to its children as arguments.

    Additionally, the following optional event callback parameters can be passed to the constructor:

    :param on_connect: Optional; a callback triggered when the `connect()` method is called.
                       The callback receives the connected child as `child_`.
    :param on_connect_kwargs: Optional; additional fixed keyword arguments for the `on_connect` callback.

    :param on_disconnect: Optional; a callback triggered when the `disconnect()` method is called.
                          The callback receives the disconnected child as `child_`.
    :param on_disconnect_kwargs: Optional; additional fixed keyword arguments for the `on_disconnect` callback.

    :param on_update: Optional; a callback triggered when the `update()` method is called.
                      The callback receives the update parameters with an underscore appended:
                      `metadata_`, `children_`, `coroutine_`, `args_`, and `kwargs_`.
    :param on_update_kwargs: Optional; additional fixed keyword arguments for the `on_update` callback.

    :param on_before_run: Optional; a callback triggered before the node's coroutine is executed via `run()`.
                          No additional timed parameters are provided, but fixed kwargs can be passed.
    :param on_before_run_kwargs: Optional; additional fixed keyword arguments for the `on_before_run` callback.

    :param on_after_run: Optional; a callback triggered after the node's coroutine is executed via `run()`.
                         No additional timed parameters are provided, but fixed kwargs can be passed.
    :param on_after_run_kwargs: Optional; additional fixed keyword arguments for the `on_after_run` callback.

    NOTE: USE WITH CARE! This node can cause deadlocks if not used properly.
    """

    def __init__(
        self,
        uuid: str,
        metadata: dict,
        coroutine: Callable,
        args: Optional[list[Any]] = None,
        kwargs: Optional[dict[str, Any]] = None,
        parent_events: Optional[list[asyncio.Event]] = None,
        timeout: Optional[float] = None,
        forward_output: Optional[bool] = False,
        on_connect: Optional[Callable[..., Any]] = None,
        on_connect_kwargs: Optional[dict[str, Any]] = None,
        on_disconnect: Optional[Callable[..., Any]] = None,
        on_disconnect_kwargs: Optional[dict[str, Any]] = None,
        on_update: Optional[Callable[..., Any]] = None,
        on_update_kwargs: Optional[dict[str, Any]] = None,
        on_before_run: Optional[Callable[..., Any]] = None,
        on_before_run_kwargs: Optional[dict[str, Any]] = None,
        on_after_run: Optional[Callable[..., Any]] = None,
        on_after_run_kwargs: Optional[dict[str, Any]] = None,
    ):
        super().__init__(
            uuid,
            metadata,
            coroutine,
            args,
            kwargs,
            forward_output=forward_output,
            on_connect=on_connect,
            on_connect_kwargs=on_connect_kwargs,
            on_disconnect=on_disconnect,
            on_disconnect_kwargs=on_disconnect_kwargs,
            on_update=on_update,
            on_update_kwargs=on_update_kwargs,
            on_before_run=on_before_run,
            on_before_run_kwargs=on_before_run_kwargs,
            on_after_run=on_after_run,
            on_after_run_kwargs=on_after_run_kwargs,
        )
        self._parent_events = parent_events if parent_events is not None else []
        self._forward_output = forward_output
        self._timeout = timeout

    def __repr__(self) -> str:
        return f"UnionNode(uuid={self.uuid}, metadata={self.metadata}, timeout={self._timeout}, output={self.output})"

    @property
    def parent_events(self):
        return self._parent_events

    @safe_execution
    def _add_event(self, event: asyncio.Event):
        """
        Adds an event to this node so that it waits for it to be set before running.
        """
        self._parent_events.append(event)

    @safe_execution
    def _remove_event(self, event: asyncio.Event):
        """
        Removes an event from this node so that it no longer waits for it to be set before running.
        """
        self._parent_events.remove(event)

    @safe_execution
    async def run(self) -> Any:
        """
        Waits for all parents to complete before running the coroutine.
        """
        if not self._timeout:
            logger.warning(
                "UnionNode %s has no timeout. This will cause a deadlock if one of its parent coroutines does not finish.",
                self.uuid,
            )
        logger.debug(f"Number of events to wait for: {len(self.parent_events)}")
        await asyncio.wait_for(
            asyncio.gather(*[e.wait() for e in self.parent_events]),
            timeout=self._timeout,
        )

        try:
            return await super().run()
        except Exception as e:
            self._children = []
            raise e
