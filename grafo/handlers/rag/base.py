from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from pydantic import BaseModel


T = TypeVar("T", bound="BaseModel")


class BaseRAG(ABC, Generic[T]):
    """Base class for vector dabatase handlers."""

    @abstractmethod
    def retrieve(self, *args, **kwargs) -> T:
        """Retrieve data from the RAG."""
        pass

    @abstractmethod
    def insert(self, *args, **kwargs) -> T:
        """Insert data into the RAG."""
        pass

    @abstractmethod
    def update(self, *args, **kwargs) -> T:
        """Update data in the RAG."""
        pass

    @abstractmethod
    def upsert(self, *args, **kwargs) -> T:
        """Upsert data in the RAG."""
        pass

    @abstractmethod
    def delete(self, *args, **kwargs) -> T:
        """Delete data from the RAG."""
        pass
