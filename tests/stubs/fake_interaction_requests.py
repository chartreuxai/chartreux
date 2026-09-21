from __future__ import annotations

from collections.abc import Awaitable, Callable

from pydantic import BaseModel

from chartreux.core.tools.base import ToolError

type UserInputRequestHandler = Callable[[BaseModel], Awaitable[BaseModel]]


class FakeInteractionRequests:
    def __init__(self, *, user_input: UserInputRequestHandler | None = None) -> None:
        self._user_input = user_input

    async def request_user_input(self, args: BaseModel, tool_call_id: str) -> BaseModel:
        del tool_call_id
        if self._user_input is None:
            raise ToolError("User input is not available")
        return await self._user_input(args)
