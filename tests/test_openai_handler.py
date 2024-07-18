import json
import os
import re

import pytest
from pydantic import BaseModel, Field

from grafo.llm import OpenAIHandler


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
    SELF_PATH = os.path.dirname(os.path.abspath(__file__))
    with open(f"{SELF_PATH}/../.vscode/launch.json", "r") as f:
        content = f.read()
        content = re.sub(r"//.*?\n|/\*.*?\*/", "", content, flags=re.S)
        data = json.loads(content)
        api_key = data["configurations"][0]["env"]["OPENAI_API_KEY"]
        os.environ["OPENAI_API_KEY"] = api_key

    openai = OpenAIHandler()
    response: SQLWriter = openai.send(
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
    print(json.dumps(response.model_dump(), indent=2))
    assert response.query == "SELECT id, name FROM users WHERE name LIKE '%John%';"
