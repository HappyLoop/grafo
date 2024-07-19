from typing import Union
import uuid

from sqlmodel import text

from grafo.examples.brain.components.analysis_handler import SQLQuery
from grafo.examples.brain.components.db_handler.db_handler import DBHandler
from grafo.llm.base import BaseLLM


class AnalysisHandler:  # ! THIS ENTIRE CLASS IS ONGOING WORK
    def __init__(self, llm: BaseLLM):
        self._llm = llm
        self._db = DBHandler(in_memory=True)

    @property
    def llm(self):
        return self._llm

    @property
    def db(self):
        return self._db

    def _build_table(self, data_schema: str) -> SQLQuery:
        table_name = f"table_{uuid.uuid4().hex}"
        create_query: SQLQuery = self.llm.send(
            messages=[
                {
                    "role": "system",
                    "content": f"You are tasked with building a CREATE TABLE query for a table named {table_name}. The table must accomodate the data in the following format: {data_schema}",
                }
            ],
            response_model=SQLQuery,
        )
        self.db.execute(text(create_query.statement))
        return create_query

    def _insert_data(self, data_schema: Union[str, None], data: list[dict[str, str]]):
        "Generates a pydantic model from the data schema and inserts the data into the table."
        if not data_schema:
            raise ValueError("Cannot insert data without a schema.")
        pass

    def _build_analysis(self, create_table: SQLQuery, input: str) -> SQLQuery:
        analytical_query: SQLQuery = self.llm.send(
            messages=[
                {
                    "role": "system",
                    "content": f"You must build a SELECT query that is able to satisfy the request in the context. \nCREATE query: {create_table.statement}\nContext: {input}",
                }
            ],
            response_model=SQLQuery,
        )
        self.db.execute(text(analytical_query.statement))
        return analytical_query

    def analyze(self, input: str, data_schema: str, data: list[dict[str, str]]):
        # ! ONGOING WORK
        create_query = self._build_table(data_schema)
        print(create_query.schema)

        self._insert_data(create_query.schema or "", data)  # TODO: write this method

        analytical_query = self._build_analysis(create_query, input)
        print(analytical_query.schema)

        return self.db.execute(text(analytical_query.statement))
        # ! ONGOING WORK
