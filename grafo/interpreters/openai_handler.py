import json

from openai import OpenAI

from .base import BaseLLM


class OpenAIHandler(BaseLLM):
    """Handles interactions with OpenAI's models."""

    def __init__(self, api_key: str, model: str, temperature: int = 0):
        super().__init__()

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature

    def send(self, messages: list, json_response: bool = False):
        """Send data to the model."""
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
            response_format={"type": "json_object"}
            if json_response
            else {"type": "text"},
        )

    def json_parser(self, data: str) -> str:
        """Parses the JSON response from OpenAI."""
        return json.loads(data[7:-3])
