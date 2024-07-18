# This is a consumer class, that consumes ClarificationRequests to produce Context objects

from typing import Callable

from grafo.examples.brain.components.task_manager.schemas import ClarificationRequest
from grafo.examples.brain.components.context_pool_manager.schemas import Context


class ContextPoolManager:
    """
    Manages the context pool.
    """

    def __init__(self, user_pipeline_method: Callable, rag_pipeline_method: Callable):
        self._user_pipeline_method = user_pipeline_method
        self._rag_pipeline_method = rag_pipeline_method

    def __str__(self) -> str:
        return f"ContextPoolManager({self._user_pipeline_method.__doc__}, {self._rag_pipeline_method.__doc__})"

    @property
    def user_pipeline_method(self):
        return self._user_pipeline_method

    @property
    def rag_pipeline_method(self):
        return self._rag_pipeline_method

    def consume(self, clarification_request: ClarificationRequest) -> Context:
        """
        Consume a ClarificationRequest to produce a Context object.
        """
        return self.user_pipeline_method(clarification_request)
