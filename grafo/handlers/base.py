from abc import ABC, abstractmethod
from typing import Generic, Type, TypeVar

T = TypeVar("T", bound="BaseLLM")


class BaseLLM(ABC):
    """Base class for LLM models."""

    _allow_instantiation = False

    def __init__(self) -> None:
        if not self._allow_instantiation:
            raise TypeError(
                f"{self.__class__.__name__} cannot be instantiated directly"
            )

    @abstractmethod
    def send(self, messages: list, response_model, model: str, max_retries: int):
        """Send data to the model."""
        pass


class LLM(Generic[T]):
    """Handles interactions with LLM models."""

    cls: Type[T]

    def __init__(self, *args, **kwargs) -> None:
        self.handler: T = self._create_handler(*args, **kwargs)

    def __str__(self) -> str:
        return f"LLM({self.handler.__class__.__name__})"

    @classmethod
    def __class_getitem__(cls, item: Type[T]) -> Type["LLM"]:
        cls.cls = item
        return cls

    def _create_handler(self, *args, **kwargs) -> T:
        if not issubclass(self.cls, BaseLLM):
            raise TypeError(f"{self.cls.__name__} is not a subclass of BaseLLM")

        self.cls._allow_instantiation = True
        try:
            instance = self.cls(*args, **kwargs)
        finally:
            self.cls._allow_instantiation = False
        return instance
