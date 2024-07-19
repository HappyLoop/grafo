from typing import Callable, Optional, Type

from pydantic import BaseModel

from grafo.llm.base import BaseLLM


class ToolObject(BaseModel):
    model: Type[BaseModel]
    fn: Optional[Callable]


class ToolManager:
    """
    Manages tools for the Brain.
    """

    def __init__(self, llm: BaseLLM, tools: dict[Type[BaseModel], Optional[Callable]]):
        self._llm = llm
        self._tool_map = {
            model.__name__: ToolObject(model=model, fn=fn)
            for model, fn in tools.items()
        }

    def __str__(self):
        repr = {name: tool.__doc__ for name, tool in self.tool_map.items()}
        return f"ToolManager({repr})"

    @property
    def llm(self):
        return self._llm

    @property
    def tool_map(self):
        return self._tool_map

    def get_tool(self, tool_name: str):
        """
        Retrieve a tool by its name.
        """
        return self.tool_map.get(tool_name)

    def get_descriptions(self):
        """
        Get the descriptions of all tools in a <name>:<description> format.
        """
        sub_prompt = ""
        for name, tool in self.tool_map.items():
            sub_prompt += f"{name}: {tool.model.__doc__}\n"
        return sub_prompt

    async def call_tool(self, tool_name: str, task_description: str):
        """
        Calls a tool by its name.
        """
        tool_object = self.get_tool(tool_name)
        if not tool_object:
            raise ValueError(f"Tool {tool_name} not found.")
        filled_tool_model: BaseModel = await self.llm.asend(
            messages=[
                {
                    "role": "user",
                    "content": task_description,
                }
            ],
            response_model=tool_object.model,
        )
        if tool_object.fn is None:
            return filled_tool_model
        return tool_object.fn(**filled_tool_model.model_dump())
