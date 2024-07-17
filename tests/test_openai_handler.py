import json

import pytest
from pydantic import BaseModel, Field

from grafo.handlers import LLM, OpenAIHandler
from grafo._internal import logger


class SQLWriter(BaseModel):
    """
    Write an SQL query that returns data from a table given by the user.
    The query should be a valid SQL query.

    Example:
    ```
    SELECT * FROM table_name;
    ```
    """

    chain_of_thought: str = Field(
        ..., description="The chain of thought behind your query.", exclude=True
    )
    table: str = Field(..., description="The table name.")
    columns: list[str] = Field(..., description="The columns to select.")
    conditions: list[str] = Field(..., description="The conditions to filter the data.")
    query: str = Field(..., description="The SQL query.")
    python_code: str = Field(
        ..., description="The Python code to build a pydantic schema for your query."
    )


# Tests
@pytest.mark.asyncio
async def test_openai_handler():
    """
    Test a tool that with a single message sent.
    """
    openai = LLM[OpenAIHandler]()
    response: SQLWriter = openai.handler.send(
        messages=[
            {
                "role": "system",
                "content": "You are an expert in SQL.",
            },
            {
                "role": "user",
                "content": "Get me user.id and user.name from users table for every user whose name includes 'John'.",
            },
        ],
        response_model=SQLWriter,
    )
    logger.debug(json.dumps(response.model_dump(), indent=2))
    assert response.query == "SELECT id, name FROM users WHERE name LIKE '%John%';"
