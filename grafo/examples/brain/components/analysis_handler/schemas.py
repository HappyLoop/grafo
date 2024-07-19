from typing import Optional
from pydantic import BaseModel, Field


class SQLQuery(BaseModel):
    chain_of_thought: str = Field(
        ..., description="The chain of thought behind your query.", exclude=True
    )
    statement: str = Field(..., description="The SQL query.")
    schema: Optional[str] = Field(
        None,
        description="Only use if this is a CREATE query. A Pydantic schema to represent data in this table.",
    )
