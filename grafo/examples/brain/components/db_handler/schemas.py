from typing import Optional, TypeVar

from pydantic import BaseModel
from sqlmodel import Field, SQLModel

T = TypeVar("T", bound=SQLModel)


class VectorSearch(BaseModel):
    field: str = Field(..., description="The field to compare the input to.")
    embedding: list[float] = Field(
        ..., description="The embedding to compare results to."
    )
    threshold: Optional[float] = Field(
        None, description="The threshold for the comparison."
    )
    limit: Optional[int] = Field(None, description="The limit for the query.")
