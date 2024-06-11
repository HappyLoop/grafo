import asyncio
import json
import logging

from langsmith import traceable
from pydantic import BaseModel, Field

from grafo.interpreters import LLM, OpenAIHandler
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


@traceable(name="test-chat")
async def test_chat():
    """
    Summarize a conversation between the user and an AI.
    """
    # disable loggers
    logging.getLogger("instructor").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("langsmith").setLevel(logging.WARNING)

    openai = LLM[
        OpenAIHandler
    ]()  # NOTE: don't forget to set API keys in environment variables
    context = ""

    while True:
        user_input = input("Enter your message: ")

        response: Message = openai.handler.send(
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


asyncio.run(test_chat())
