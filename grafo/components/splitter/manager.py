from grafo.handlers import LLM
from grafo.components.splitter.schemas import TaskGroup


class TaskManager:
    """
    Splits tasks.
    """

    def __init__(self, llm: LLM):
        self._llm = llm

    def __str__(self) -> str:
        if not hasattr(self, "group"):
            return "TaskManager(None)"
        return f"TaskManager({self.group.tasks})"

    @property
    def llm(self):
        return self._llm

    @property
    def group(self):
        return self._group

    async def split_tasks(
        self,
        input: str,
        tools_description: str,
    ):
        self._group: TaskGroup = await self.llm.handler.asend(
            messages=[
                {
                    "role": "system",
                    "content": f"You are a helpful assistant. You will receive an input from a user, which you must split into tasks. You have access to the following tools: \n{tools_description}",
                },
                {
                    "role": "user",
                    "content": input,
                },
            ],
            response_model=TaskGroup,
        )
