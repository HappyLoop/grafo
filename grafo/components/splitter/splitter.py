from grafo.interpreters import LLM, OpenAIHandler
from grafo.components.splitter.schemas import TaskGroup


def split_tasks(input: str):
    openai = LLM[OpenAIHandler]()

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
    and then send me an email (paul@test.com) with the history of cakes.

    You have the following tools:
    recipe_writer
    occasion_finder
    email_sender
    """
)

for task in group.tasks:
    print("\n", task)

print("\n", group.chain_of_thought)
