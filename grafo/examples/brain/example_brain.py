from typing import Callable, Optional, Type

from pydantic import BaseModel

from grafo.examples.brain.services import TaskManager, ToolManager
from grafo.handlers.llm.base import BaseLLM


class Brain:
    def __init__(
        self,
        llm_manager: BaseLLM,
        tools_manager: dict[Type[BaseModel], Optional[Callable]],
        user_clarification: bool = False,  # ! to be implemented in the future
        db_handler: Optional[Callable] = None,  # ! to be implemented in the future
        knowledge_manager: Optional[
            dict[Type[BaseModel], Optional[Callable]]
        ] = None,  # ! to be implemented in the future, can use db handler
    ):
        self._llm = llm_manager
        self._task_manager = TaskManager(llm_manager)
        self._tool_manager = ToolManager(llm_manager, tools_manager)
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
