import os
from typing import Optional, Type, TypeVar
from pydantic import BaseModel

import instructor
from langsmith.wrappers import wrap_openai
from openai import AsyncOpenAI, OpenAI

from grafo.llm import BaseLLM

T = TypeVar("T", bound=BaseModel)


class OpenAIHandler(BaseLLM):
    """Handles interactions with OpenAI's models."""

    def __init__(
        self,
        model: Optional[str] = "gpt-4o",
        temperature: Optional[int] = 0,
    ):
        super().__init__()

        self.model = os.getenv("OPENAI_MODEL") or model
        self.temperature = temperature

        self._langsmith = True
        if os.getenv("LANGCHAIN_API_KEY") is None:
            self._langsmith = False

    @property
    def langsmith(self):
        return self._langsmith

    def send(
        self,
        messages: list,
        response_model: Type[T],
        model: Optional[str] = "gpt-4o",
        max_retries: int = 3,
    ) -> T:
        """
        Send data to the model.
        """
        if self.langsmith:
            client = wrap_openai(OpenAI())
        else:
            client = OpenAI()

        client = instructor.from_openai(client=client)

        return client.chat.completions.create(
            model=self.model or model,
            messages=messages,
            response_model=response_model,
            max_retries=max_retries,
            temperature=self.temperature,
        )

    async def asend(
        self,
        messages: list,
        response_model: Type[T],
        model: Optional[str] = "gpt-4o",
        max_retries: int = 3,
    ) -> T:
        """
        Sends data to the model asyncronously.
        """
        if self.langsmith:
            client = wrap_openai(AsyncOpenAI())
        else:
            client = AsyncOpenAI()

        client = instructor.from_openai(client=client)

        return await client.chat.completions.create(
            model=self.model or model,
            messages=messages,
            response_model=response_model,
            max_retries=max_retries,
            temperature=self.temperature,
        )

    def create_embedding(self, text: str, model: str = "text-embedding-3-small"):
        """
        Get embeddings from the model.
        """
        if self.langsmith:
            client = wrap_openai(OpenAI())
        else:
            client = OpenAI()

        return client.embeddings.create(
            model=model,
            input=text,
        )
