from pydantic import BaseModel, Field


class CakeRecipeWriterTool(BaseModel):
    """
    This tool is used to write cake recipes.
    """

    name: str = Field(..., title="Cake Name", description="The name of the cake")
    ingredients: list[str] = Field(
        ..., title="Ingredients", description="The ingredients of the cake"
    )
    steps: list[str] = Field(
        ..., title="Steps", description="The steps to bake the cake"
    )
