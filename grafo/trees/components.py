import inspect
from typing import Any, Callable, Optional, Self, Type

import asyncio


def safe_execution(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator that prevents a method from being called if the node is already running.
    """

    def wrapper(self, *args: Any, **kwargs: Any) -> Any:
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

    :param on_result: Optional; a callback triggered when the node's output is set via `set_output()`.
                      The callback receives the output as `output_`.
    :param on_result_kwargs: Optional; additional fixed keyword arguments for the `on_result` callback.

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
        on_result: Optional[Callable[..., Any]] = None,
        on_result_kwargs: Optional[dict[str, Any]] = None,
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

        # Initialize event callbacks from constructor parameters.
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
        self._on_result_callback = (
            (on_result, on_result_kwargs or {}) if on_result is not None else None
        )

    def __repr__(self) -> str:
        return f"Node(uuid={self.uuid}, metadata={self.metadata})"

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

    @property
    def forward_output(self):
        return self._forward_output

    @property
    def is_running(self):
        return self._is_running

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
        if self._on_connect_callback:
            callback, fixed_kwargs = self._on_connect_callback
            callback(child_=child, **fixed_kwargs)

    @safe_execution
    def disconnect(self, child: Self):
        """
        Disconnects a child from this node.
        """
        if not issubclass(type(child), Node):
            raise ValueError("The 'child' parameter must be a Node instance.")
        self.children.remove(child)
        if self._on_disconnect_callback:
            callback, fixed_kwargs = self._on_disconnect_callback
            callback(child_=child, **fixed_kwargs)

    @safe_execution
    def update(
        self,
        metadata: Optional[dict] = None,
        children: Optional[list[Self]] = None,
        coroutine: Optional[Callable] = None,
        args: Optional[list[Any]] = None,
        kwargs: Optional[dict[str, Any]] = None,
    ):
        """
        Updates the node with new data.
        """
        if metadata is not None:
            if "name" not in metadata or "description" not in metadata:
                raise ValueError("metadata must contain 'name' and 'description' keys.")
            self._metadata = metadata

        self.__validate_param(args, "args", list, allow_none=True)
        self.__validate_param(kwargs, "kwargs", dict, allow_none=True)

        if children and any(not isinstance(child, Node) for child in children):
            raise ValueError("'children' parameter must be a list of <Node> instances")

        if children:
            self._children = children

        if inspect.iscoroutinefunction(coroutine):
            self._coroutine = coroutine

        self._args = args if args is not None else []
        self._kwargs = kwargs if kwargs is not None else {}

        if self._on_update_callback:
            callback, fixed_kwargs = self._on_update_callback
            callback(
                children_=children,
                **fixed_kwargs,
            )

    @safe_execution
    async def run(self) -> Any:
        """
        Asynchronously runs the coroutine of in this node.
        """
        if self._on_before_run_callback:
            callback, fixed_kwargs = self._on_before_run_callback
            callback(**fixed_kwargs)

        self._is_running = True
        try:
            result = await self._coroutine(*self.args, **self.kwargs)  # type: ignore
        finally:
            self._is_running = False
        return result

    @safe_execution
    def set_output(self, output: Any):
        """
        Sets the output of the node.

        Event Callback: on_result callback is triggered after setting the node's output.
        Naming Convention: The argument 'output' is passed as 'output_' to the callback.
        """
        self._output = output
        if self._on_result_callback:
            callback, fixed_kwargs = self._on_result_callback
            callback(output_=output, **fixed_kwargs)


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

    :param on_result: Optional; a callback triggered when the node's output is set via `set_output()`.
                      The callback receives the output as `output_`.
    :param on_result_kwargs: Optional; additional fixed keyword arguments for the `on_result` callback.
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
        on_result: Optional[Callable[..., Any]] = None,
        on_result_kwargs: Optional[dict[str, Any]] = None,
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
            on_result=on_result,
            on_result_kwargs=on_result_kwargs,
        )

    def __repr__(self) -> str:
        return f"PickerNode(uuid={self.uuid}, metadata={self.metadata})"

    @safe_execution
    async def run(self) -> list["Node"]:
        """
        Picks the children to queue next based on the result of the node.
        """
        if self._on_before_run_callback:
            callback, fixed_kwargs = self._on_before_run_callback
            callback(**fixed_kwargs)

        result = await self.coroutine(self, self.children, *self.args, **self.kwargs)
        if not isinstance(result, list):
            raise ValueError("The picker coroutine must return a list of children.")
        for child in result:
            if not isinstance(child, Node):
                raise ValueError(
                    "The picker coroutine must return a list of Node instances."
                )
        return result


class UnionNode(Node):
    """
    A node that waits for all its parents to finish executing before continuing.

    :param uuid: The unique identifier of the node.
    :param metadata: A dict containing at least "name" and "description" for the node.
    :param coroutine: The coroutine function to execute.
    :param args: The arguments to pass to the coroutine.
    :param kwargs: The keyword arguments to pass to the coroutine.
    :param parents: The parent nodes of this node.
    :param parent_timeout: The timeout for waiting for parents to complete.
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

    :param on_result: Optional; a callback triggered when the node's output is set via `set_output()`.
                      The callback receives the output as `output_`.
    :param on_result_kwargs: Optional; additional fixed keyword arguments for the `on_result` callback.



    >>> USE WITH CARE!
    >>> This node can cause deadlocks if not used properly.
    """

    def __init__(
        self,
        uuid: str,
        metadata: dict,
        coroutine: Callable,
        args: Optional[list[Any]] = None,
        kwargs: Optional[dict[str, Any]] = None,
        parents: Optional[list["UnionNode"]] = None,
        parent_timeout: Optional[float] = None,
        forward_output: Optional[bool] = False,
        on_connect: Optional[Callable[..., Any]] = None,
        on_connect_kwargs: Optional[dict[str, Any]] = None,
        on_disconnect: Optional[Callable[..., Any]] = None,
        on_disconnect_kwargs: Optional[dict[str, Any]] = None,
        on_update: Optional[Callable[..., Any]] = None,
        on_update_kwargs: Optional[dict[str, Any]] = None,
        on_before_run: Optional[Callable[..., Any]] = None,
        on_before_run_kwargs: Optional[dict[str, Any]] = None,
        on_result: Optional[Callable[..., Any]] = None,
        on_result_kwargs: Optional[dict[str, Any]] = None,
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
            on_result=on_result,
            on_result_kwargs=on_result_kwargs,
        )
        self._parents = parents if parents is not None else []
        self._parent_outputs = {}
        self._num_parents_completed = 0
        self._forward_output = forward_output
        self._parent_timeout = parent_timeout
        self._timeout_flag = False

    def __repr__(self) -> str:
        return f"UnionNode(uuid={self.uuid}, metadata={self.metadata})"

    @property
    def parents(self):
        return self._parents

    @property
    def parent_outputs(self):
        return self._parent_outputs

    @property
    def num_parents_completed(self):
        return self._num_parents_completed

    @property
    def timeout_flag(self):
        return self._timeout_flag

    @safe_execution
    def append_arguments(
        self,
        args: Optional[list[Any]] = None,
        kwargs: Optional[dict[str, Any]] = None,
    ):
        """
        Appends arguments to the node.
        """
        if args:
            self._args.extend(args)

        if kwargs:
            self._kwargs.update(kwargs)

    @safe_execution
    def add_parent(self, parent: "UnionNode"):
        """
        Adds a parent to this node.
        """
        if not issubclass(type(parent), Node):
            raise ValueError("The 'parent' parameter must be a UnionNode instance.")
        self._parents.append(parent)

    @safe_execution
    def parent_completed(self, parent_uuid: str, output: Any):
        """
        Marks a parent as completed and stores its output.
        """
        self._parent_outputs[parent_uuid] = output
        self._num_parents_completed += 1

    @safe_execution
    async def run(self) -> Any:
        """
        Waits for all parents to complete before running the coroutine.
        """
        self._is_running = True
        try:
            await asyncio.wait_for(
                self._wait_for_parents(), timeout=self._parent_timeout
            )
        except asyncio.TimeoutError:
            self._timeout_flag = True
            self._is_running = False  # Ensure the running flag is reset
            raise asyncio.TimeoutError

        if self._on_before_run_callback:
            callback, fixed_kwargs = self._on_before_run_callback
            callback(**fixed_kwargs)

        try:
            async with asyncio.Lock():
                result = await self._coroutine(
                    *self.args,
                    **self.kwargs,
                )
        finally:
            self._is_running = False
        return result

    async def _wait_for_parents(self):
        while self.num_parents_completed < len(self.parents):
            await asyncio.sleep(0.1)
