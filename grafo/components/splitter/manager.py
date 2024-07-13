from grafo.handlers import LLM
from grafo.components.splitter.schemas import UserRequest


class TaskManager:
    """
    Manages the state of Tasks within a UserRequest.
    """

    def __init__(self, llm: LLM):
        self._llm = llm

    def __str__(self) -> str:
        if not hasattr(self, "user_request"):
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
        """
        Create a UserRequest object from a user input. It includes a list of tasks and clarifications.
        """
        self._group: UserRequest = await self.llm.handler.asend(
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
            response_model=UserRequest,
        )
