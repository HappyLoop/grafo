from typing import Any, Protocol


class OnForwardCallable(Protocol):
    async def __call__(self, forward_data: Any, **kwargs: Any) -> Any: ...


class AwaitableCallback(Protocol):
    async def __call__(self, *args: Any, **kwargs: Any) -> Any: ...
