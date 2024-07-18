import json
from pydantic import BaseModel, Field

from grafo.llm import OpenAIHandler
from grafo._internal import logger


class Message(BaseModel):
    """
    A message in a conversation.
    """

    content: str = Field(..., description="Your answer to the user's message.")
    summary: str = Field(
        ...,
        description="A summary of what the user asked and what you answered.",
    )
    is_done: bool = Field(
        False, description="Whether the user is finished with the conversation."
    )


async def test_chat():
    """
    Summarize a conversation between the user and an AI.
    """

    openai = OpenAIHandler()
    context = ""

    while True:
        user_input = input("Enter your message: ")

        response: Message = openai.send(
            messages=[
                {
                    "role": "system",
                    "content": f"You are a helpful assistant. Keep the conversation going by asking if the user wants to know more about something you mentioned. The conversation so far: {context}",
                },
                {
                    "role": "user",
                    "content": user_input,
                },
            ],
            response_model=Message,
        )

        logger.debug(json.dumps(response.model_dump(), indent=2))
        context += "\n\n" + response.summary
        if response.is_done:
            print(context)
            break
