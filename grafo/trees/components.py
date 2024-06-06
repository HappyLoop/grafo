import inspect
from typing import Any, Callable, Optional, Self, Type

from grafo.interpreters.base import LLM
import asyncio


def safe_execution(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator that prevents a method from being called if the node is already running.
    """

    def wrapper(self, *args: Any, **kwargs: Any) -> Any:
        if self.output is not None:
            raise ValueError("This node has already produced an output.")

        if not self._is_running:
            return func(self, *args, **kwargs)
        else:
            raise ValueError("This node is already running.")

    return wrapper


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
        :param forward_output: Whether to forward the output of this node to its children as arguments.
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
        forward_output: Optional[bool] = False,
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
        self._forward_output = forward_output

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
        if not isinstance(child, Node):
            raise ValueError("The 'child' parameter must be a Node instance.")
        self.children.append(child)

    @safe_execution
    def disconnect(self, child: Self):
        """
        Disconnects a child from this node.
        """
        if not isinstance(child, Node):
            raise ValueError("The 'child' parameter must be a Node instance.")
        self.children.remove(child)

    @safe_execution
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

    @safe_execution
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
    :param name: The name of the node.
    :param description: The description of the node.
    :param coroutine: The coroutine function to execute.
    :param llm: The LLM to use for picking children.
    :param picker: A function that receives the current node, the result of running the node, and its children. It then returns a list of children to be queued. If None, all children are queued.
    :param args: The arguments to pass to the coroutine.
    :param kwargs: The keyword arguments to pass to the coroutine.
    :param children: The children nodes of this node.
    :param forward_output: Whether to forward the output of this node to its children as arguments.
    """

    def __init__(
        self,
        uuid: str,
        name: str,
        description: str,
        coroutine: Callable,
        llm: LLM,
        picker: Callable,
        args: Optional[list[Any]] = None,
        kwargs: Optional[dict[str, Any]] = None,
        children: Optional[list["Node"]] = None,
        forward_output: Optional[bool] = False,
    ):
        super().__init__(
            uuid, name, description, coroutine, args, kwargs, children, picker
        )

        self._llm = llm
        self._picker = picker
        self._forward_output = forward_output

    @property
    def llm(self):
        return self._llm

    @property
    def picker(self):
        return self._picker

    @safe_execution
    async def choose(self):
        """
        Picks the children to queue next based on the result of the node.
        """
        if self.picker:
            return await self.picker(self, self.output, self.children)
        return None


class UnionNode(Node):
    """
    A node that waits for all its parents to finish executing before continuing.

    :param uuid: The unique identifier of the node.
    :param name: The name of the node.
    :param description: The description of the node.
    :param coroutine: The coroutine function to execute.
    :param args: The arguments to pass to the coroutine.
    :param kwargs: The keyword arguments to pass to the coroutine.
    :param parents: The parent nodes of this node.
    :param use_parents_output: Whether to use the output of the parents as arguments to the coroutine.
    :param forward_output: Whether to forward the output of this node to its children as arguments.
    """

    def __init__(
        self,
        uuid: str,
        name: str,
        description: str,
        coroutine: Callable,
        args: Optional[list[Any]] = None,
        kwargs: Optional[dict[str, Any]] = None,
        parents: Optional[list["UnionNode"]] = None,
        use_parents_output: Optional[bool] = False,
        forward_output: Optional[bool] = False,
    ):
        super().__init__(uuid, name, description, coroutine, args, kwargs)
        self._parents = parents if parents is not None else []
        self._parent_outputs = {}
        self._num_parents_completed = 0
        self._use_parents_output = use_parents_output
        self._forward_output = forward_output

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
    def use_parents_output(self):
        return self._use_parents_output

    @safe_execution
    def add_parent(self, parent: "UnionNode"):
        """
        Adds a parent to this node.
        """
        if not isinstance(parent, Node):
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
    async def run(self):
        """
        Waits for all parents to complete before running the coroutine.
        """
        while self.num_parents_completed < len(self.parents):
            await asyncio.sleep(0.1)

        if self.use_parents_output:
            self._args = [output for output in self.parent_outputs.values()]
        ##############################
        # Need a way to pass the output of parents as kwargs too
        # result = await self._coroutine(*self.args, **self.kwargs)
        ##############################

        self._is_running = True
        try:
            result = await self._coroutine(
                *self.args  # NOTE: currently not passing kwargs
            )
        finally:
            self._is_running = False
        return result
