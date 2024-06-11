import asyncio
import json
from typing import Optional

import pytest
from pydantic import BaseModel, Field

from grafo.interpreters import LLM, OpenAIHandler
from grafo._internal import logger


# Define tools
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


class TextSummarizer(BaseModel):
    """
    Summarize an interaction between the user and an AI. Your output should be a
    paragraph that represents the interaction, keeping the most important parts.
    Be sure to specify what was asked and what was answered, keeping specific details.

    Example:
    ```
    The conversation is about cars. The user asked about the price of a car and the
    AI answered that the price of a car depends on the model and the year of the car.
    ```
    """

    chain_of_thought: str = Field(
        ..., description="The chain of thought behind your summary.", exclude=True
    )
    summary: str = Field(..., description="The summary of the text.")


# Define an Agent-like entity
class AgentLikeEntity(
    BaseModel
):  # NOTE: not really an Agent, just a simulation of tool choosing
    """
    Use your tools to process the user's input.
    """

    text_summary: Optional[TextSummarizer] = Field(
        None, description="Use if user asked for a summary."
    )
    sql_writer: Optional[SQLWriter] = Field(
        None, description="Use if the user asked for an SQL query."
    )


# Tests
@pytest.mark.asyncio
async def test_single_message():
    """
    Test a tool that with a single message sent.
    """
    openai = LLM[OpenAIHandler]()
    response = openai.handler.send(
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


@pytest.mark.asyncio
async def test_multiple_messages():
    """
    Test a tool that requires multiple messages to be sent.
    """
    openai = LLM[OpenAIHandler]()
    response = openai.handler.send(
        messages=[
            {
                "role": "system",
                "content": "You always use tools.",
            },
            {
                "role": "user",
                "content": "Do you know butterflies?",
            },
            {
                "role": "assistant",
                "content": "Yes, I know butterflies. What would you like to know about them?",
            },
            {
                "role": "user",
                "content": "Tell me more about them!",
            },
            {
                "role": "system",
                "content": """
                Butterflies are insects in the macrolepidopteran clade Rhopalocera from the order 
                Lepidoptera, which also includes moths. Adult butterflies have large, often brightly 
                coloured wings, and conspicuous, fluttering flight. The group comprises the large 
                superfamily Papilionoidea, which contains at least one former group, the skippers 
                (formerly the superfamily "Hesperioidea"), and the most recent analyses suggest it
                also contains the moth-butterflies (formerly the superfamily "Hedyloidea"). 
                Butterfly fossils date to the Paleocene, about 56 million years ago.
                """,
            },
        ],
        response_model=TextSummarizer,
    )

    logger.debug(json.dumps(response.model_dump(), indent=2))


@pytest.mark.asyncio
async def test_agent_with_tools():
    """
    Use an agent with tools to process the user's input.
    """
    openai = LLM[OpenAIHandler]()
    response = openai.handler.send(
        messages=[
            {
                "role": "system",
                "content": "You always use tools.",
            },
            {
                "role": "user",
                "content": """Write me an SQL query to retrieve name and tag from the products 
                              table where the tag includes 'abc'. Afterwards, write a summary of 
                              of what I've asked you.
                            """,
            },
        ],
        response_model=AgentLikeEntity,
    )

    logger.debug(json.dumps(response.model_dump(), indent=2))


asyncio.run(test_agent_with_tools())
