import os
from typing import Any, Optional, Sequence, Type, TypeVar

from sqlmodel import Session, SQLModel, create_engine
from sqlmodel import select as sqlmodel_select

from grafo.examples.brain.components.db_handler import BaseMultiModalDB
from grafo.examples.brain.components.db_handler.schemas import VectorSearch

T = TypeVar("T", bound=SQLModel)


class PostgresHandler(BaseMultiModalDB):
    def __init__(self, db_url: Optional[str] = None):
        connection_string = db_url or os.getenv("DB_URL")
        if not connection_string:
            raise ValueError("No connection string provided.")
        self.engine = create_engine(connection_string)

    def select(
        self,
        model: Type[T],
        where: Optional[dict[str, list[str]]] = None,
        group_by: Optional[str] = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Sequence[T]:
        "Selects entries from the database."
        with Session(self.engine) as session:
            statement = sqlmodel_select(model)
            if where:
                where_statements = []
                for key, values in where.items():
                    for val in values:
                        where_statements.append(getattr(model, key) == val)
                statement = statement.where(*where_statements)
            if group_by:
                statement = statement.group_by(getattr(model, group_by))
            if order_by:
                statement = statement.order_by(getattr(model, order_by))
            if limit:
                statement = statement.limit(limit)
            return session.exec(statement).all()

    def insert(self, data: list[Type[T]]) -> list[Type[T]]:
        "Adds entries to the database."
        return self.upsert(data)

    def update(self, data: list[Type[T]]) -> list[Type[T]]:
        "Updates data in the database."
        return self.upsert(data)

    def upsert(self, data: list[Type[T]]) -> list[Type[T]]:
        "Inserts or updates entries to the database."
        with Session(self.engine) as session:
            session.add_all(data)
            session.commit()
            for item in data:
                session.refresh(item)
            return data

    def delete(self, data: list[Type[T]]) -> list[Type[T]]:
        "Deletes entries from the database."
        with Session(self.engine) as session:
            for item in data:
                session.delete(item)
            session.commit()
            return data

    def execute(self, statement: Any):
        "Executes a custom SQL statement."
        with Session(self.engine) as session:
            result = session.exec(statement)
            session.commit()
            return result

    def select_nearest_neighbours(
        self, model: Type[T], search: VectorSearch
    ) -> Sequence[T]:
        "Selects the nearest neighbours to a vector."
        with Session(self.engine) as session:
            attr = getattr(model, search.field)
            statement = sqlmodel_select(model)

            if search.threshold:
                statement = statement.filter(
                    attr.cosine_distance(search.embedding) < search.threshold
                )

            if search.limit:
                statement = statement.limit(search.limit)

            statement = statement.order_by(attr.cosine_distance(search.embedding))
            return session.exec(statement).all()
