from grafo.components.brain import Brain
from grafo.handlers.llm import OpenAIHandler

from tools import CakeRecipeWriterTool, OccasionFinderTool, ResearchTool


async def test_pipeline():
    brain = Brain(
        llm=OpenAIHandler(),
        tools={
            CakeRecipeWriterTool: None,
            OccasionFinderTool: None,
            ResearchTool: None,
        },
    )
    await brain.pipeline(
        """
        Write a recipe for a cake, find a fitting occasion to serve it, research about orca whales, bake the cake, take a picture of the it and send to paulo's email.
        """
    )
    print(brain.task_manager.user_request)
