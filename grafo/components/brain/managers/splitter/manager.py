from grafo.components.brain.managers.splitter.schemas import UserRequest
from grafo.handlers.llm.base import BaseLLM


class TaskManager:
    """
    Manages the state of Tasks within a UserRequest.
    """

    def __init__(self, llm: BaseLLM):
        self._llm = llm

    def __str__(self) -> str:
        if not hasattr(self, "user_request"):
            return "TaskManager(None)"
        if self.user_request is None:
            return "TaskManager(None)"
        return f"TaskManager({self.user_request.tasks})"

    @property
    def llm(self):
        return self._llm

    @property
    def user_request(self):
        return self._user_request

    async def split_tasks(
        self,
        input: str,
        tools_description: str,
    ):
        """
        Create a UserRequest object from a user input. It includes a list of tasks and clarifications.
        """
        self._user_request: UserRequest = await self.llm.asend(
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
