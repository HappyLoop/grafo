from dataclasses import dataclass
from typing import Generic, TypeVar


C = TypeVar("C")


@dataclass(slots=True, frozen=True)
class Metadata:
    runtime: float
    level: int


@dataclass(slots=True, frozen=True)
class Chunk(Generic[C]):
    """
    A Chunk represents an intermediate output of a node. Useful for trying filter the different outputs from a tree.
    """

    uuid: str
    output: C
