import pytest

from grafo.examples.brain.services import TaskManager
from grafo.handlers.llm import OpenAIHandler


@pytest.mark.asyncio
async def test_split_tasks():
    openai = OpenAIHandler()
    manager = TaskManager(openai)
    await manager.split_tasks(
        """
        Write a recipe for a cake, find a fitting occasion to serve it, research about orca whales, bake the cake, take a picture of the it and send to paulo's email.
        """,
        """
        recipe_writer
        occasion_finder
        url_finder
        cake_baker
        camera
        email_sender
        """,
    )
    user_request = manager.user_request
    assert user_request is not None
    assert user_request.tasks is not None
    assert [task.layer for task in user_request.tasks] == [0, 0, 0, 1, 2, 3]
    assert [task.essential for task in user_request.tasks] == [
        True,
        True,
        False,
        True,
        True,
        True,
    ]
    assert len(user_request.clarifications) == 2

    for task in user_request.tasks:
        print(task)
    print("\n\t", user_request.chain_of_thought)
    print("\t", user_request.main_goal)
    print("\t", user_request.clarifications)
