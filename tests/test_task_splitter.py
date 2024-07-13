import asyncio
import pytest

from grafo.components.splitter import TaskManager
from grafo.handlers import LLM, OpenAIHandler

openai = LLM[OpenAIHandler]()


@pytest.mark.asyncio
async def test_split_tasks():
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
    group = manager.group
    assert group is not None
    assert group.tasks is not None
    assert [task.layer for task in group.tasks] == [0, 0, 0, 1, 2, 3]
    assert [task.essential for task in group.tasks] == [
        True,
        True,
        False,
        True,
        True,
        True,
    ]
    assert len(group.clarifications) == 2

    for task in group.tasks:
        print(task)
    print("\n\t", group.chain_of_thought)
    print("\t", group.main_goal)
    print("\t", group.clarifications)


asyncio.run(test_split_tasks())
