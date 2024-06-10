import os
from typing import Literal, Optional

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

    def create__assistant(
        self,
        client: OpenAI,
        name: str,
        instructions: str,
        tools: list,
        tool_resources: dict,
        model: str = "gpt-4o",
        **kwargs,
    ):
        """
        Create an assistant.
        """
        return client.beta.assistants.create(
            name=name,
            instructions=instructions,
            tools=tools,
            tool_resources=tool_resources,  # type: ignore
            model=model,
            **kwargs,
        )

    def create_thread(self, client: OpenAI):
        """
        Create a thread.
        """
        return client.beta.threads.create()

    def create_message(
        self,
        client: OpenAI,
        thread_id: str,
        role: Literal["user", "assistant"],
        content: str,
    ):
        """
        Create a message.
        """
        return client.beta.threads.messages.create(
            thread_id=thread_id, role=role, content=content
        )

    def create_file(self, client: OpenAI, file: str):
        """
        Create a file and returns a content object.
        """
        new_file = client.files.create(file=open(file, "rb"), purpose="vision")
        return {
            "type": "image_file",
            "image_file": {"file_id": new_file.id},
        }
