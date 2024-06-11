import os
from typing import Literal, Optional

import instructor
from langsmith import traceable
from langsmith.wrappers import wrap_openai
from openai import AsyncOpenAI, OpenAI

from grafo.interpreters.base import BaseLLM


class OpenAIHandler(BaseLLM):
    """Handles interactions with OpenAI's models."""

    def __init__(
        self,
        model: Optional[str] = "gpt-4o",
        temperature: Optional[int] = 0,
    ):
        super().__init__()

        # self.client = instructor.from_openai(client=OpenAI())
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
            model=model,
            messages=messages,
            response_model=response_model,
            max_retries=max_retries,
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
            model=model,
            messages=messages,
            response_model=response_model,
            max_retries=max_retries,
        )

    ################################################################################
    # Beta
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

    #############################################################################
