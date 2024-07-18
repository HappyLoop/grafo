from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class BaseLLM(ABC, Generic[T]):
    """Base class for LLM models."""

    @abstractmethod
    def send(self, *args, **kwargs) -> T:
        """Send data to the model."""
        pass

    @abstractmethod
    async def asend(self, *args, **kwargs) -> T:
        """Send data to the model asynchronously."""
        pass
