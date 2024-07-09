from typing import Optional
from pydantic import BaseModel, Field


class Task(BaseModel):
    """
    An unit of work to be performed. Rules:
    - Tasks that can be performed in parallel should have the same layer.
    """

    description: str = Field(
        ...,
        description="A description of the task.",
    )
    tool: Optional[str] = Field(
        None,
        description="The name of the tool to use. Only set if this task is not a request for more information.",
    )
    essential: bool = Field(
        False,
        description="Whether the task is related to the main goal.",
    )
    layer: int = Field(
        0,
        description="The layer of the task within a dependency tree. Lower numbers are executed first. Zero-indexed.",
    )


class TaskGroup(BaseModel):
    """
    A list of tasks from the user input. Rules:
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
        title="Main Goal",
    )
    primary_task: Task = Field(
        None,
        description="The most important task to be performed. It is likely that its completion will achieve the final steps of the Main Goal.",
    )
    # secondary_goals: Optional[list[str]] = Field(
    #     None,
    #     description="A list of descriptions of tasks that are not directly related to the main goal. They are not necessary to achieve the main goal.",
    # )
    secondary_tasks: Optional[list[Task]] = Field(
        None,
        description="A list of tasks whose completion is not present in the tree of dependencies of the most_important_task. They are not necessary to achieve the main goal.",
    )
    additional_info: Optional[list[str]] = Field(
        None,
        description="A list of questions requesting contextual information useful to the completion of the tasks.",
    )
