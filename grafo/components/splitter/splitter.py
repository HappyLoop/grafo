from grafo.handlers import LLM, OpenAIHandler
from grafo.components.splitter.schemas import TaskGroup


openai = LLM[OpenAIHandler]()


def split_tasks(input: str):
    return openai.handler.send(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {
                "role": "user",
                "content": input,
            },
        ],
        response_model=TaskGroup,
    )


group: TaskGroup = split_tasks(
    """
    Write a recipe for a cake, give me a list of occasions where I can use it,
    and then send me an email (paulo@test.com) with a link to the wiki page for cakes. Also search the web for orca whales. Afterwards, bake the cake and finally take a picture of it!

    You have the following tools:
    recipe_writer
    occasion_finder
    url_finder
    email_sender
    cake_baker
    camera
    """,
)


for task in group.tasks or []:
    print(task)

print("\n\t", group.chain_of_thought, "\n")

print("\t", group.main_goal)
print("\t", group.secondary_goals)
