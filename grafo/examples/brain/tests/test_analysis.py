import json

from grafo.examples.brain.components.analysis_handler.analysis_handler import (
    AnalysisHandler,
)
from grafo.llm.openai_handler import OpenAIHandler


def test_analysis():
    data_schema = json.dumps(
        {
            "name": "str",
            "age": "int",
            "address": "str",
        }
    )
    data = [
        {
            "name": "John",
            "age": 30,
            "address": "123 Main St.",
        },
        {
            "name": "Jane",
            "age": 25,
            "address": "456 Elm St.",
        },
        {
            "name": "Alice",
            "age": 35,
            "address": "789 Oak St.",
        },
    ]

    context = "What is the average age of the people in the dataset?"
    openai = OpenAIHandler()
    analysis_handler = AnalysisHandler(openai)
    result = analysis_handler.analyze(context, data_schema, data)

    for row in result:
        print(row)


test_analysis()
