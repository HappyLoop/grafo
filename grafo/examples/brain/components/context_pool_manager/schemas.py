from pydantic import BaseModel, Field


class Context(BaseModel):
    """
    A response to a clarification request.
    """

    task_id: int = Field(..., description="The ID of the task.")
    answer: str = Field(..., description="The answer to the clarification question.")
