from typing import Optional
from pydantic import BaseModel, Field


class Task(BaseModel):
    """
    An unit of work to be performed.
    """

    priority: Optional[int] = Field(
        0,
        description="A number representing the priority order of the task. Zero-indexed.",
    )
    description: str = Field(
        ..., description="Your description of the task. Be descriptive."
    )
    tool: Optional[str] = Field(
        None,
        description="The name of the tool to use. Only set if this task is not a request for more information.",
    )
    quote: str = Field(
        ..., description="A quote from the user input that describes the task."
    )


class TaskGroup(BaseModel):
    """
    Build a list of tasks from the user input. I necessary, include tasks
    where you ask the user for any additional information required to
    complete the task. Example:

    User: 'summarize the provided document and send an email'
    Tasks: [
        'Summarize the provided document',
        'Ask the user for the address',
        'Ask the user for the subject',
        'Ask the user for the body',
        'Ask the user for the recipient',
        'Send an email',
    ]
    """

    tasks: list[Task] = Field(..., description="The list of tasks.")
    chain_of_thought: str = Field(
        ..., description="The chain of thought behind the tasks."
    )
