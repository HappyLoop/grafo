import os
from typing import Optional

import instructor
from openai import OpenAI

from .base import BaseLLM


class OpenAIHandler(BaseLLM):
    """Handles interactions with OpenAI's models."""

    def __init__(
        self,
        model: Optional[str] = "gpt-4o",
        temperature: Optional[int] = 0,
    ):
        super().__init__()

        self.client = instructor.from_openai(client=OpenAI())
        self.model = os.getenv("OPENAI_MODEL") or model
        self.temperature = temperature

    def send(self, messages: list, response_model, model: Optional[str] = "gpt-4o"):
        """
        Send data to the model.
        """
        return self.client.chat.completions.create(
            model=model, messages=messages, response_model=response_model
        )
