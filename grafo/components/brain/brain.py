# Brain:
# - Has awareness of its tools (ToolManager)
# - Has access to a knowledge base (RAG)
# - Has centralizd state (TaskManager)
# - Can understand inputs and split them into tasks (TaskManager)

from typing import Callable, Optional
from pydantic import BaseModel
from grafo.handlers.base import LLM

from ..splitter import TaskManager
from ..tools import ToolManager


class Brain:
    def __init__(self, llm: LLM, tools: dict[BaseModel, Optional[Callable]]):
        self._llm = llm
        self._task_manager = TaskManager(llm)
        self._tool_manager = ToolManager(llm, tools)

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

    async def pipeline(self, input: str):
        """
        The Brain's pipeline.
        """
        # 1. Get the sub-prompt & split the tasks
        tools_decriptions = self.tool_manager.get_descriptions()
        await self.task_manager.split_tasks(input, tools_decriptions)

        # 2. Perform clarifications (RAG or ask user)

        # 3. Add clarification context to task prompt

        # 3. Create tree with tasks
