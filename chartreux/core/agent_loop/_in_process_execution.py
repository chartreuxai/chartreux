"""In-process adapter for the agent-side execution port.

The adapter contributes the port's ordering and callback behavior, not agent
logic: it runs the bound loop's own turn stream, so the in-process escape path
(``CHARTREUX_TRANSPORT=in_process``) and the multi-process path exercise the same
components with the process boundary removed. Everything it adds is what a
process-backed adapter has to reproduce across the wire.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import aclosing
from typing import TYPE_CHECKING

from pydantic import BaseModel

from chartreux.core.agent_loop._execution_port import (
    ExecutionTurnRequest,
    TurnOrderingConflict,
)
from chartreux.core.agent_loop._request_broker import (
    KeepWaitingOutcome,
    UserInputSettleOutcome,
)
from chartreux.core.events import BaseEvent, UserInputRequestEvent

if TYPE_CHECKING:
    from chartreux.core.agent_loop._loop import AgentLoop


class InProcessTurnExecution:
    """Serve the execution port from the loop living in this process."""

    def __init__(self, agent_loop: AgentLoop) -> None:
        self._agent_loop = agent_loop
        self._turn_in_flight = False
        self._active_turn_id: str | None = None

    async def run_turn(
        self, request: ExecutionTurnRequest
    ) -> AsyncGenerator[BaseEvent, None]:
        if self._turn_in_flight:
            raise TurnOrderingConflict(self._active_turn_id)
        options = request.turn_options
        self._turn_in_flight = True
        self._active_turn_id = options.turn_id if options is not None else None
        try:
            async with aclosing(
                self._agent_loop.act(
                    request.message,
                    client_message_id=request.client_message_id,
                    auto_title=request.auto_title,
                    images=request.images,
                    user_display_content=request.user_display_content,
                    input_text=request.input_text,
                    resources=request.resources,
                    subagent_runner=request.subagent_runner,
                    tool_io=request.tool_io,
                    turn_options=options,
                )
            ) as events:
                async for event in events:
                    yield event
        finally:
            self._active_turn_id = None
            self._turn_in_flight = False

    def resolve_user_input(self, request_id: str, result: BaseModel) -> None:
        self._agent_loop.resolve_user_input_request(request_id, result)

    def reject_user_input(self, request_id: str, error: BaseException) -> None:
        self._agent_loop.reject_request(request_id, error)

    def relay_user_input_request(self, event: UserInputRequestEvent) -> None:
        self._agent_loop.request_broker.relay_user_input_request(event)

    async def settle_user_input_answer(
        self, request_id: str, result: BaseModel
    ) -> UserInputSettleOutcome:
        return await self._agent_loop.request_broker.settle_answer(request_id, result)

    async def settle_user_input_rejection(
        self, request_id: str, error: BaseException
    ) -> UserInputSettleOutcome:
        return await self._agent_loop.request_broker.settle_rejection(request_id, error)

    async def keep_waiting_user_input(self, request_id: str) -> KeepWaitingOutcome:
        return await self._agent_loop.request_broker.keep_waiting(request_id)

    async def cancel_user_input(self, request_id: str, error: BaseException) -> None:
        await self._agent_loop.request_broker.cancel(request_id, error)
