from typing import Optional
from pydantic import BaseModel, Field


class Task(BaseModel):
    """
    An unit of work to be performed.
    """

    description: str = Field(
        ...,
        description="A description of the task.",
    )
    additional_info: Optional[list[str]] = Field(
        None,
        description="A list of questions requesting additional information necessary to complete the task.",
    )
    tool: Optional[str] = Field(
        None,
        description="The name of the tool to use. Only set if this task is not a request for more information.",
    )
    essential: bool = Field(
        False,
        description="Whether the task is essential to the main goal.",
    )


class TaskGroup(BaseModel):
    """
    Build a list of tasks from the user input. Rules:
    - Be sure to fill additional_info with questions that may be necessary to complete each task. E.g., "What is the name of the cake?"
    - The main goal is a short description of the user's main objective. It may not encompass all tasks.
    """

    tasks: Optional[list[Task]] = Field(None, description="The list of tasks.")
    chain_of_thought: str = Field(
        ..., description="The chain of thought behind the tasks."
    )
    main_goal: str = Field(
        ...,
        description="A concise description of the tasks directly related to the main goal of the user's input.",
    )
    secondary_goals: Optional[list[str]] = Field(
        None,
        description="A list of secondary goals that are not directly related to the main goal.",
    )
