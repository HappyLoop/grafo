from typing import Any, Protocol, AsyncGenerator, Union, Awaitable


class OnForwardCallable(Protocol):
    async def __call__(self, forward_data: Any, **kwargs: Any) -> Any: ...


class AwaitableCallback(Protocol):
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> Union[Awaitable[Any], AsyncGenerator[Any, None]]: ...
