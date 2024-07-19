import concurrent.futures
import uuid
from typing import Any
from uuid import UUID

from pgvector.sqlalchemy import Vector
from pydantic import BaseModel
from sqlmodel import Column, Field, SQLModel, text

from grafo.examples.brain.components.db_handler import DBHandler, VectorSearch
from grafo.llm.openai_handler import OpenAIHandler


class Embedding(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    collection: str = Field(..., nullable=False, index=True)
    content: str = Field(..., nullable=False)
    embedding: Any = Field(sa_column=Column(Vector(1536), nullable=False))


class DogInfo(BaseModel):
    """
    A model with information about a dog.
    """

    name: str = Field(..., description="The name of the dog.")
    color: str = Field(..., description="The color of the dog.")


def test_vector_search():
    db = DBHandler()
    openai = OpenAIHandler()

    SQLModel.metadata.create_all(db.engine)

    messages = [
        "You have two dogs, Nina and Cherie.",
        "Nina is 12 years old, and Cherie is 5 months old.",
        "Nina's fur is orange, and Cherie's fur is white.",
        "Cherie is very playful, while Nina is grumpy.",
    ]

    animals_collection = []

    def build_embedding(msg):
        embedding = openai.create_embedding(msg).data[0].embedding
        return Embedding(collection="animals", content=msg, embedding=embedding)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_msg = {executor.submit(build_embedding, msg): msg for msg in messages}

        for future in concurrent.futures.as_completed(future_to_msg):
            try:
                animals_collection.append(future.result())
            except Exception as exc:
                print(f"An error occurred: {exc}")
    db.insert(animals_collection)

    input = "What color is Nina?"
    embedding = openai.create_embedding(input).data[0].embedding
    search = VectorSearch(
        field="embedding", embedding=embedding, limit=3, threshold=0.5
    )
    result = db.select_nearest_neighbours(Embedding, search)
    context = [item.content for item in result]
    context.append(input)

    dog_info = openai.send(
        messages=[
            {
                "role": "system",
                "content": "Given contextual information, fill the model with information.",
            },
            {"role": "user", "content": "\n".join(context)},
        ],
        response_model=DogInfo,
    )
    print(dog_info)
    db.execute(text("DROP TABLE IF EXISTS embedding;"))
    assert dog_info.name == "Nina"
    assert dog_info.color == "orange"


test_vector_search()
