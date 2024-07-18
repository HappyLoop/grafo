from typing import Annotated, Callable, Optional, Type

from pydantic import BaseModel

from examples.brain.components import TaskManager, ToolManager, ContextPoolManager
from grafo.llm.base import BaseLLM


type ToolMap = Annotated[
    dict[Type[BaseModel], Optional[Callable]],
    "A map of tools schemas to functions. If no function is provided, the tool is passice, which means it is only a schema to be filled.",
]


class Brain:
    def __init__(
        self,
        llm_handler: BaseLLM,
        tool_map: ToolMap,
        context_pool_manager: ContextPoolManager,
        user_clarification: bool = False,  # ! to be implemented in the future
    ):
        self._llm = llm_handler
        self._task_manager = TaskManager(llm_handler)
        self._tool_manager = ToolManager(llm_handler, tool_map)
        self._user_clarification = user_clarification

    def __str__(self) -> str:
        return f"Brain({self._task_manager}, {self._tool_manager})"

    @property
    def llm(self):
        return self._llm

    @property
    def task_manager(self):
        return self._task_manager

    @property
    def tool_manager(self):
        return self._tool_manager

    @property
    def user_clarification(self):
        return self._user_clarification

    async def pipeline(
        self, input: str
    ):  # ? Should this be a root node during runtime?
        """
        The Brain's pipeline.
        """
        # 1. Get the sub-prompt & split the tasks
        tools_decriptions = self.tool_manager.get_descriptions()
        await self.task_manager.split_tasks(input, tools_decriptions)

        # 2. Perform clarifications (RAG or ask user)
        # a) Clarifications always search a knowledge base first (if available)
        # b) If no knowledge base is available, or the information is not found, ask the user

        # 3. Add clarification context to task prompt

        # 3. Create tree with tasks


# Splitter -> Clarifier -> RAG/Input ->
