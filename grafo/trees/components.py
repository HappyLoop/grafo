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
    """

    _output: Any

    def __init__(
        self,
        uuid: str,
        metadata: dict,
        coroutine: Callable,
        args: Optional[list[Any]] = None,
        kwargs: Optional[dict[str, Any]] = None,
        children: Optional[list["Node"]] = None,
        forward_output: Optional[bool] = False,
    ):
        self.__validate_param(uuid, "uuid", str)
        self.__validate_param(args, "args", list, allow_none=True)
        self.__validate_param(kwargs, "kwargs", dict, allow_none=True)
        self._metadata = metadata

        if not inspect.iscoroutinefunction(coroutine):
            raise ValueError(
                "The coroutine parameter must be a coroutine function (async function)."
            )

        if any(not isinstance(child, Node) for child in children or []):
            raise ValueError("'children' parameter must be a list of <Node> instances")

        self._uuid = uuid
        self._coroutine = coroutine
        self._args = args if args is not None else []
        self._kwargs = kwargs if kwargs is not None else {}
        self._output = None
        self._children = children if children is not None else []
        self._is_running = False
        self._forward_output = forward_output

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

    @safe_execution
    def disconnect(self, child: Self):
        """
        Disconnects a child from this node.
        """
        if not issubclass(type(child), Node):
            raise ValueError("The 'child' parameter must be a Node instance.")
        self.children.remove(child)

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

    @safe_execution
    async def run(self) -> Any:
        """
        Asynchronously runs the coroutine of in this node.
        """
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
        """
        self._output = output


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
    """

    def __init__(
        self,
        uuid: str,
        metadata: dict,
        coroutine: Callable,
        args: Optional[list[Any]] = None,
        kwargs: Optional[dict[str, Any]] = None,
        children: Optional[list["Node"]] = None,
        forward_output: Optional[bool] = False,
    ):
        super().__init__(
            uuid, metadata, coroutine, args, kwargs, children, forward_output
        )

    def __repr__(self) -> str:
        return f"PickerNode(uuid={self.uuid}, metadata={self.metadata})"

    @safe_execution
    async def run(self) -> list["Node"]:
        """
        Picks the children to queue next based on the result of the node.
        """
        return await self.coroutine(self, self.children, *self.args, **self.kwargs)


class UnionNode(Node):
    """
    A node that waits for all its parents to finish executing before continuing.

    :param uuid: The unique identifier of the node.
    :param metadata: A dict containing at least "name" and "description" for the node.
    :param coroutine: The coroutine function to execute.
    :param args: The arguments to pass to the coroutine.
    :param kwargs: The keyword arguments to pass to the coroutine.
    :param parents: The parent nodes of this node.
    :param forward_output: Whether to forward the output of this node to its children as arguments.
    :param timeout: The timeout for waiting for parents to complete.

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
        forward_output: Optional[bool] = False,
        timeout: Optional[float] = None,
    ):
        super().__init__(uuid, metadata, coroutine, args, kwargs)
        self._parents = parents if parents is not None else []
        self._parent_outputs = {}
        self._num_parents_completed = 0
        self._forward_output = forward_output
        self._timeout = timeout
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
            await asyncio.wait_for(self._wait_for_parents(), timeout=self._timeout)
        except asyncio.TimeoutError:
            self._timeout_flag = True
            self._is_running = False  # Ensure the running flag is reset
            raise asyncio.TimeoutError

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
