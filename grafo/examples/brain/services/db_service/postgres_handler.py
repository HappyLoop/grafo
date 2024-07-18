from typing import Any, Optional, Type

from pydantic import BaseModel
from sqlmodel import Field, Session, SQLModel, create_engine
from sqlmodel import select as sqlmodel_select

from grafo.examples.brain.services.db_service import BaseMultiModalDB


class VectorComparison(BaseModel):
    field: str = Field(..., description="The field to compare the input to.")
    input: list[int] = Field(..., description="The input vector.")
    threshold: Optional[float] = Field(
        None, description="The threshold for the comparison."
    )
    limit: Optional[int] = Field(None, description="The limit for the query.")


class PostgresHandler(BaseMultiModalDB):
    def __init__(self, db_url: str):
        self.engine = create_engine(db_url)
        SQLModel.metadata.create_all(self.engine)

    def select(
        self,
        model: Type[SQLModel],
        where: Optional[dict[str, list[str]]] = None,
        group_by: Optional[str] = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = None,
    ):
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

    def insert(self, data: list[Type[SQLModel]]):
        "Adds entries to the database."
        return self.upsert(data)

    def update(self, data: list[Type[SQLModel]]):
        "Updates data in the database."
        return self.upsert(data)

    def upsert(self, data: list[Type[SQLModel]]):
        "Inserts or updates entries to the database."
        with Session(self.engine) as session:
            session.add_all(data)
            session.commit()
            for item in data:
                session.refresh(item)
            return data

    def delete(self, data: list[Type[SQLModel]]):
        "Deletes entries from the database."
        with Session(self.engine) as session:
            for item in data:
                session.delete(item)
            session.commit()
            return data

    def execute(self, statement: Any):
        "Executes a custom SQL statement."
        with Session(self.engine) as session:
            return session.exec(statement)

    def select_nearest_neighbours(
        self, model: Type[SQLModel], vector: VectorComparison
    ):
        "Selects the nearest neighbours to a vector."
        with Session(self.engine) as session:
            attr = getattr(model, vector.field)
            comparison = attr.l2_distance(vector.input)
            if vector.threshold:
                comparison = attr.l2_distance(vector.input) < vector.threshold
            statement = sqlmodel_select(comparison)
            if vector.limit:
                statement = statement.limit(vector.limit)
            return session.exec(statement).all()
