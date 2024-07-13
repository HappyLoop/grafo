import os
from typing import Optional

import instructor
from langsmith import traceable
from langsmith.wrappers import wrap_openai
from openai import AsyncOpenAI, OpenAI

from grafo.handlers.base import BaseLLM


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

    @traceable(name="send")
    def send(
        self,
        messages: list,
        response_model,
        model: Optional[str] = "gpt-4o",
        max_retries: int = 3,
    ):
        """
        Send data to the model.
        """
        if self.langsmith:
            client = wrap_openai(OpenAI())
        else:
            client = OpenAI()

        client = instructor.from_openai(
            client=client, mode=instructor.Mode.TOOLS
        )  # NOTE: keep TOOLS instead of PARALLEL_TOOLS for now

        return client.chat.completions.create(
            model=self.model or model,
            messages=messages,
            response_model=response_model,
            max_retries=max_retries,
            temperature=self.temperature,
        )

    @traceable(name="send-async")
    async def asend(
        self,
        messages: list,
        response_model,
        model: Optional[str] = "gpt-4o",
        max_retries: int = 3,
    ):
        """
        Sends data to the model asyncronously.
        """
        if self.langsmith:
            client = wrap_openai(AsyncOpenAI())
        else:
            client = AsyncOpenAI()

        client = instructor.from_openai(
            client=client, mode=instructor.Mode.TOOLS
        )  # NOTE: keep TOOLS instead of PARALLEL_TOOLS for now

        return await client.chat.completions.create(
            model=self.model or model,
            messages=messages,
            response_model=response_model,
            max_retries=max_retries,
            temperature=self.temperature,
        )
