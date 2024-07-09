from grafo.handlers import LLM
from grafo.components.splitter.schemas import TaskGroup


def split_tasks(llm: LLM, input: str):
    return llm.handler.send(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. You will receive an input from a user, which you must split into tasks.",
            },
            {
                "role": "user",
                "content": input,
            },
        ],
        response_model=TaskGroup,
    )
