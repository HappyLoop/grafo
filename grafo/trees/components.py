import inspect
from typing import Any, Callable, Optional, Self, Type

from grafo.interpreters.base import LLM


class Node:
    """
    A Node is a unit of work that can be executed concurrently. It contains a coroutine function that is executed by a worker.

        :param uuid: The unique identifier of the node.
        :param name: The name of the node.
        :param description: The description of the node.
        :param coroutine: The coroutine function to execute.
        :param args: The arguments to pass to the coroutine.
        :param kwargs: The keyword arguments to pass to the coroutine.
        :param children: The children nodes of this node.
        :param picker: A function that receives the current node, the result of running the node, and its children. It then returns a list of children to be queued. If None, all children are queued.
    """

    _output: Any

    def __init__(
        self,
        uuid: str,
        name: str,
        description: str,
        coroutine: Callable,
        args: Optional[list[Any]] = None,
        kwargs: Optional[dict[str, Any]] = None,
        children: Optional[list["Node"]] = None,
        picker: Optional[Callable] = None,
    ):
        self.__validate_param(uuid, "uuid", str)
        self.__validate_param(name, "name", str)
        self.__validate_param(description, "description", str)
        self.__validate_param(args, "args", list, allow_none=True)
        self.__validate_param(kwargs, "kwargs", dict, allow_none=True)

        if not inspect.iscoroutinefunction(coroutine):
            raise ValueError(
                "The coroutine parameter must be a coroutine function (async function)."
            )

        if any(not isinstance(child, Node) for child in children or []):
            raise ValueError("'children' parameter must be a list of <Node> instances")

        if picker and not callable(picker):
            raise ValueError("'picker' parameter must be a callable function")

        self._uuid = uuid
        self._name = name
        self._description = description
        self._coroutine = coroutine
        self._args = args if args is not None else []
        self._kwargs = kwargs if kwargs is not None else {}
        self._output = None
        self._children = children if children is not None else []
        self._picker = picker
        self._is_running = False

    def __repr__(self) -> str:
        return f"Node(uuid={self.uuid}, name={self.name})"

    @property
    def uuid(self):
        return self._uuid

    @property
    def name(self):
        return self._name

    @property
    def description(self):
        return self._description

    @property
    def children(self):
        return self._children

    @property
    def picker(self):
        return self._picker

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

    def __safe_execution(func: Callable):  # type: ignore
        """
        Decorator that prevents a method from being called if the node is already running.
        """

        def wrapper(self, *args, **kwargs):
            if self.output is not None:
                raise

            if not self._is_running:
                return func(self, *args, **kwargs)
            else:
                raise ValueError("This node is already running.")

        return wrapper

    @__safe_execution
    def connect(self, child: Self):
        """
        Connects a child to this node.
        """
        if not isinstance(child, Node):
            raise ValueError("The 'child' parameter must be a Node instance.")
        self.children.append(child)

    @__safe_execution
    def disconnect(self, child: Self):
        """
        Disconnects a child from this node.
        """
        if not isinstance(child, Node):
            raise ValueError("The 'child' parameter must be a Node instance.")
        self.children.remove(child)

    @__safe_execution
    def update(
        self,
        name: str | None = None,
        description: str | None = None,
        children: list[Self] | None = None,
        coroutine: Callable | None = None,
        args: list[Any] | None = None,
        kwargs: dict[str, Any] | None = None,
    ):
        """
        Updates the node with new data.
        """
        self.__validate_param(name, "name", str, allow_none=True)
        self.__validate_param(description, "description", str, allow_none=True)
        self.__validate_param(args, "args", list, allow_none=True)
        self.__validate_param(kwargs, "kwargs", dict, allow_none=True)

        if children and any(not isinstance(child, Node) for child in children):
            raise ValueError("'children' parameter must be a list of <Node> instances")

        if name:
            self._name = name

        if description:
            self._description = description

        if children:
            self._children = children

        if inspect.iscoroutinefunction(coroutine):
            self._coroutine = coroutine

        self._args = args if args is not None else []
        self._kwargs = kwargs if kwargs is not None else {}

    @__safe_execution
    async def run(self):
        """
        Asynchronously runs the coroutine of in this node.
        """
        self._is_running = True
        try:
            result = await self._coroutine(*self.args, **self.kwargs)  # type: ignore
        finally:
            self._is_running = False
        return result

    @__safe_execution
    def set_output(self, output: Any):
        """
        Sets the output of the node.
        """
        self._output = output


class PickerNode(Node):
    """
    A node that uses an LLM to determine which children to queue next.
    """

    def __init__(
        self,
        uuid: str,
        name: str,
        description: str,
        coroutine: Callable,
        llm: LLM,
        args: Optional[list[Any]] = None,
        kwargs: Optional[dict[str, Any]] = None,
        children: Optional[list["Node"]] = None,
        picker: Optional[Callable] = None,
    ):
        super().__init__(
            uuid, name, description, coroutine, args, kwargs, children, picker
        )

        self._llm = llm

    @property
    def llm(self):
        return self._llm

    async def choose(self) -> list[Node]:
        """
        Picks the children to queue next based on the result of the node.
        """
        # Use a child's name, description and coroutine with an LLM's tool calling functionality to pick among children
        raise NotImplementedError(
            "You must implement the 'picker' method in a PickerNode subclass."
        )
