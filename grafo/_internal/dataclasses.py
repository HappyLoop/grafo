from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(slots=True, frozen=True)
class Metadata:
    runtime: float
    level: int


@dataclass(slots=True, frozen=True)
class Chunk(Generic[T]):
    """
    A Chunk represents an intermediate output of a node. Useful for trying filter the different outputs from a tree.
    """

    uuid: str
    output: T
