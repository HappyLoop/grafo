from typing import Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Task(BaseModel):
    """
    An unit of work to be performed. Rules:
    - Tasks that can be performed in parallel should have the same layer.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: int = Field(
        ...,
        description="The ID of the task.",
    )
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
    result: Union[BaseModel, Exception, None] = Field(
        None,
        description="The result of the task.",
    )


class TaskClarification(BaseModel):
    """
    A request for more information about a task.
    """

    task_id: int = Field(..., description="The ID of the task.")
    question: str = Field(..., description="The clarification needed.")


class UserRequest(BaseModel):
    """
    A list of tasks from the user input. Rules:
    - Be sure to fill additional_info with questions that may be necessary to complete each task. E.g., "What is the name of the cake?"
    - The main goal is a short description of the user's main objective. It may not encompass all tasks.
    """

    tasks: list[Task] = Field(..., description="The list of tasks.")
    chain_of_thought: str = Field(
        ..., description="The chain of thought behind the tasks."
    )
    main_goal: str = Field(
        ...,
        description="A concise description of the tasks directly related to the main goal of the user's input.",
    )
    clarifications: list[TaskClarification] = (
        Field(  # TODO: what if no additional info is needed?
            ...,
            description="A list of questions requesting contextual information useful to the completion of the tasks. These questions can also be used to disambiguate the user's input.",
        )
    )

    @model_validator(mode="after")
    def perform_validations(self):
        task_ids = [task.id for task in self.tasks]
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("Task IDs must be unique.")
        if not all(cl.task_id in task_ids for cl in self.clarifications):
            raise ValueError("All tasks must be clarified.")
        return self
