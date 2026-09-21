from __future__ import annotations

import asyncio
from uuid import uuid4

from pydantic import BaseModel

from chartreux.core.events import BaseEvent, UserInputRequestEvent
from chartreux.core.tools.base import InteractionRequestPort, ToolError


class InteractionRequestBroker(InteractionRequestPort):
    def __init__(self) -> None:
        self._events: asyncio.Queue[BaseEvent | None] | None = None
        self._user_input_requests: dict[str, asyncio.Future[BaseModel]] = {}

    def bind(self, events: asyncio.Queue[BaseEvent | None]) -> None:
        if self._events is not None:
            raise RuntimeError("An interaction request stream is already active")
        self._events = events

    def unbind(self, events: asyncio.Queue[BaseEvent | None]) -> None:
        if self._events is not events:
            raise RuntimeError("Cannot close an inactive interaction request stream")
        self._events = None

    async def request_user_input(self, args: BaseModel, tool_call_id: str) -> BaseModel:
        events = self._events
        if events is None:
            raise ToolError("User input is not available outside an active turn")

        request_id = str(uuid4())
        future = asyncio.get_running_loop().create_future()
        self._user_input_requests[request_id] = future
        await events.put(
            UserInputRequestEvent(
                request_id=request_id, args=args, tool_call_id=tool_call_id
            )
        )
        try:
            return await future
        finally:
            self._user_input_requests.pop(request_id, None)

    def resolve_user_input(self, request_id: str, result: BaseModel) -> None:
        future = self._user_input_requests.get(request_id)
        if future is not None and not future.done():
            future.set_result(result)

    def reject(self, request_id: str, error: BaseException) -> None:
        future = self._user_input_requests.get(request_id)
        if future is not None and not future.done():
            future.set_exception(error)
