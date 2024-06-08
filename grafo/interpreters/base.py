from abc import ABC, abstractmethod
from typing import Generic, Type, TypeVar

T = TypeVar("T", bound="BaseLLM")


class BaseLLM(ABC):
    """Base class for LLM models."""

    _instantiation_allowed = False

    def __init__(self) -> None:
        if not self._instantiation_allowed:
            raise TypeError(
                f"{self.__class__.__name__} cannot be instantiated directly"
            )

    @abstractmethod
    def send(self, data: str) -> None:
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
        if not hasattr(self.cls, "_instantiation_allowed"):
            raise TypeError(
                f"{self.cls.__name__} cannot be instantiated directly by LLM."
            )

        self.cls._instantiation_allowed = True
        try:
            instance = self.cls(*args, **kwargs)
        finally:
            self.cls._instantiation_allowed = False
        return instance
