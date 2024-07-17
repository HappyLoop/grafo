import asyncio

from grafo.components.brain import Brain
from grafo.handlers import LLM, OpenAIHandler

from tests.tools import CakeRecipeWriterTool, OccasionFinderTool, ResearchTool


async def call_pipeline():
    llm = LLM[OpenAIHandler]()
    brain = Brain(
        llm,
        {
            CakeRecipeWriterTool: None,
            OccasionFinderTool: None,
            ResearchTool: None,
        },  # type: ignore
    )
    await brain.pipeline(
        """
        Write a recipe for a cake, find a fitting occasion to serve it, research about orca whales, bake the cake, take a picture of the it and send to paulo's email.
        """
    )
    print(brain.task_manager.user_request)


asyncio.run(call_pipeline())
