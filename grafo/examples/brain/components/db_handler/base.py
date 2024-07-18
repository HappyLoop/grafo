from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound="BaseModel")


class BaseMultiModalDB(ABC, Generic[T]):
    """Base class for vector dabatase handlers."""

    @abstractmethod
    def select(self, *args, **kwargs) -> T:
        pass

    @abstractmethod
    def insert(self, *args, **kwargs) -> T:
        pass

    @abstractmethod
    def update(self, *args, **kwargs) -> T:
        pass

    @abstractmethod
    def upsert(self, *args, **kwargs) -> T:
        pass

    @abstractmethod
    def delete(self, *args, **kwargs) -> T:
        pass

    @abstractmethod
    def execute(self, *args, **kwargs) -> T:
        pass

    @abstractmethod
    def select_nearest_neighbours(self, *args, **kwargs) -> T:
        pass
