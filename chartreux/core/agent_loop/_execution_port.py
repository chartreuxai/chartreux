"""Agent-side execution port: the surface a host drives an agent turn through.

The port belongs to the agent side. A host asks the agent to run one turn and
answers the interactions that turn opens, without reaching into the agent's
internals. Today the host is the in-process app server and the port is served by
:mod:`chartreux.core.agent_loop._in_process_execution`; a process-backed adapter
implements the same contract over the wire, so the in-process escape path and
the multi-process path run the same agent components with the process boundary
removed.

Two properties are part of the contract, not of one implementation:

* Turn ordering. At most one turn runs on a port at a time, and every event of
  that turn is delivered in the order the agent produced it. A turn offered
  while another is in flight fails typed instead of interleaving two streams.
* Callbacks. A turn's user-input requests are answered through the same port,
  by request id, so the agent that asked settles its own waiter.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel

from chartreux.core.agent_loop._loop_types import AgentTurnOptions
from chartreux.core.agent_loop._request_broker import (
    KeepWaitingOutcome,
    UserInputSettleOutcome,
)
from chartreux.core.events import BaseEvent, UserInputRequestEvent
from chartreux.core.llm_models import ImageAttachment
from chartreux.core.subagents import SubagentRunnerPort
from chartreux.core.tools.io_port import ToolIOPort
from chartreux.user_content import UserDisplayContent, UserResource


class ExecutionPortError(RuntimeError):
    """Base class for typed execution-port failures."""


class TurnOrderingConflict(ExecutionPortError):
    """A turn was offered while another turn was still in flight."""

    def __init__(self, active_turn_id: str | None) -> None:
        running = f" {active_turn_id!r}" if active_turn_id else ""
        super().__init__(f"a turn{running} is already running on this agent")
        self.active_turn_id = active_turn_id


@dataclass(frozen=True, slots=True)
class ExecutionTurnRequest:
    """One turn handed to the agent: the loop's turn inputs, nothing else."""

    message: str | None = None
    client_message_id: str | None = None
    auto_title: str | None = None
    images: list[ImageAttachment] | None = None
    user_display_content: UserDisplayContent | None = None
    input_text: str | None = None
    resources: list[UserResource] | None = None
    subagent_runner: SubagentRunnerPort | None = None
    tool_io: ToolIOPort | None = None
    turn_options: AgentTurnOptions | None = None


class TurnExecutionPort(Protocol):
    """The agent-side turn execution surface a host drives."""

    def run_turn(
        self, request: ExecutionTurnRequest
    ) -> AsyncGenerator[BaseEvent, None]:
        """Run one turn, yielding its events in production order."""
        ...

    def resolve_user_input(self, request_id: str, result: BaseModel) -> None:
        """Settle the open user-input request *request_id* with *result*."""
        ...

    def reject_user_input(self, request_id: str, error: BaseException) -> None:
        """Fail the open user-input request *request_id* with *error*."""
        ...

    def relay_user_input_request(self, event: UserInputRequestEvent) -> None:
        """Adopt a host-relayed user-input request; its record lives agent-side."""
        ...

    async def settle_user_input_answer(
        self, request_id: str, result: BaseModel
    ) -> UserInputSettleOutcome:
        """Arbitrated answer relay; the decision is made agent-side."""
        ...

    async def settle_user_input_rejection(
        self, request_id: str, error: BaseException
    ) -> UserInputSettleOutcome:
        """Arbitrated rejection relay; the decision is made agent-side."""
        ...

    async def keep_waiting_user_input(self, request_id: str) -> KeepWaitingOutcome:
        """Relay a keep-waiting request; the deadline decision is agent-side."""
        ...

    async def cancel_user_input(self, request_id: str, error: BaseException) -> None:
        """Relay a cancellation; the request is closed agent-side, fail-closed."""
        ...
