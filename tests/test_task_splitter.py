from grafo.handlers import LLM, OpenAIHandler
from grafo.components.splitter import split_tasks, TaskGroup


openai = LLM[OpenAIHandler]()


def test_split_tasks():
    group: TaskGroup = split_tasks(
        openai,
        """
        Write a recipe for a cake, find a fitting occasion to serve it, research about orca whales, bake the cake, take a picture of the it and send to paulo's email.

        You have the following tools:
        recipe_writer
        occasion_finder
        url_finder
        cake_baker
        camera
        email_sender
        """,
    )

    for task in group.tasks or []:
        print(task)

    print("\n\t", group.chain_of_thought)
    print("\t", group.main_goal)
    # print("\t", group.secondary_goals)
    print("\t", group.additional_info)
    print("\t", group.primary_task)
    print("\t", group.secondary_tasks)


test_split_tasks()
