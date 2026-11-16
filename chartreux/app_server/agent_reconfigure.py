"""Agent-side adapters for the reconfigure modes this release carries.

The coordinator reaches the root session through one reconfigure request per
invocation mode, and the modes that mutate the session -- the settings
apply-and-ack, the /clear history reset, and the config write with its
prepare/commit-reload legs -- are served here, next to the loop that owns the
state they change. Nothing the coordinator owns moves across the boundary.

The clear acks success only after verifying, against LIVE runtime state, that
the history is gone and no session model override remains -- the override clear
is driven from that state rather than from the session metadata a reset has
just rewritten, and the receipt's ``stage`` names exactly the steps this call
completed, so a retry after a partial failure re-enters and finishes the rest.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from chartreux.app_server._session_model import (
    clear_session_active_model_override,
    override_active_model,
)
from chartreux.app_server.agent_config_write import AgentConfigWriteService
from chartreux.app_server.agent_snapshot import AgentSessionSnapshot, RootOpenAccounting
from chartreux.app_server.agent_wire import (
    CLEAR_STAGE_ALREADY_CLEAR,
    CLEAR_STAGE_HISTORY,
    CLEAR_STAGE_MODEL_OVERRIDE,
    RECONFIGURE_OUTCOME_APPLIED,
    RECONFIGURE_OUTCOME_CLEARED,
    RECONFIGURE_OUTCOME_CLEARED_WITH_FOLLOWUP_FAILURE,
    AgentReconfigureConflictError,
    AgentReconfigureFailedError,
    AgentReconfigureRefusedError,
    AgentSessionNotFoundError,
    ReconfigureReply,
    ReconfigureRequest,
)
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.llm_models import LLMMessage, Role

type AfterResponse = Callable[[str], Awaitable[str | None]]
type SnapshotSource = Callable[[], AgentSessionSnapshot]


class AgentReconfigureService:
    """Serve the reconfigure modes against the loop this agent owns."""

    def __init__(
        self,
        loop: AgentLoop,
        *,
        turn_active: Callable[[], bool],
        snapshot: SnapshotSource,
        after_response: AfterResponse | None = None,
    ) -> None:
        self._loop = loop
        self._turn_active = turn_active
        self._snapshot = snapshot
        self._after_response = after_response

    def settings_update(self, request: ReconfigureRequest) -> ReconfigureReply:
        """Apply the requested bounds and acknowledge what was applied."""
        self._require_session(request.session_id)
        if request.max_turns is not None:
            self._loop.set_max_turns(request.max_turns)
        if request.max_tokens is not None:
            self._loop.set_max_tokens(request.max_tokens)
        return ReconfigureReply(
            mode=request.mode,
            outcome=RECONFIGURE_OUTCOME_APPLIED,
            max_turns=request.max_turns,
            max_tokens=request.max_tokens,
        )

    async def history_clear(self, request: ReconfigureRequest) -> ReconfigureReply:
        """Clear the session history and its model pin, or report what failed."""
        self._require_session(request.session_id)
        if self._turn_active():
            raise AgentReconfigureConflictError(
                "Cannot clear history while a turn is active"
            )
        loop = self._loop
        orchestrator = loop.config_orchestrator
        stages: list[str] = []
        if _conversation_messages(loop):
            try:
                await loop.clear_history()
            except Exception as exc:
                raise AgentReconfigureFailedError(
                    f"Failed to clear history: {exc}"
                ) from exc
            stages.append(CLEAR_STAGE_HISTORY)
        # The override clear is driven from LIVE runtime state: the metadata a
        # reset just rewrote is not the authority for whether a pin remains.
        if override_active_model(orchestrator) is not None:
            failures = await clear_session_active_model_override(
                orchestrator, reason="clear session active model"
            )
            if failures:
                return self._cleared_with_followup(
                    request,
                    stages,
                    f"Failed to clear session active model: {failures[0]}",
                )
            stages.append(CLEAR_STAGE_MODEL_OVERRIDE)
            try:
                await loop.reload_with_initial_messages()
            except Exception as exc:
                return self._cleared_with_followup(
                    request, stages, f"The prompt refresh after the clear failed: {exc}"
                )
        # The terminal-state check that gates the success ack (S6-B5).
        if _conversation_messages(loop):
            raise AgentReconfigureFailedError(
                "The session history is still present after the clear"
            )
        if override_active_model(orchestrator) is not None:
            return self._cleared_with_followup(
                request,
                stages,
                "The session active model override is still active after the "
                "clear; retry the clear to remove it",
            )
        followup = await self._run_after_response(request.mode)
        return ReconfigureReply(
            mode=request.mode,
            outcome=RECONFIGURE_OUTCOME_CLEARED,
            stage=",".join(stages) or CLEAR_STAGE_ALREADY_CLEAR,
            followup=followup,
            snapshot=self._snapshot(),
        )

    async def config_write(self, request: ReconfigureRequest) -> ReconfigureReply:
        """Run the config transaction here and carry its shipped result back.

        The receipt is the shipped response (persistence x application, the
        rejected flag, the failure details, the revision, the saved values and
        the launch-metadata outcome), so the outcome this reply names is that
        response's own runtime application dimension -- nothing is collapsed
        into a smaller class, and an applied-but-undurable transaction stays the
        composition it is.
        """
        candidate = request.config_write
        if candidate is None:
            raise AgentReconfigureRefusedError(
                "The config write carries no candidate transaction"
            )
        if self._turn_active():
            raise AgentReconfigureConflictError(
                "Cannot write the configuration while a turn is active; "
                "finish or stop the work first"
            )
        response = await AgentConfigWriteService(self._loop).write(candidate)
        return ReconfigureReply(
            mode=request.mode,
            outcome=response.application,
            snapshot=self._snapshot(),
            accounting=RootOpenAccounting.from_loop(self._loop),
            config_write=response,
        )

    async def _run_after_response(self, mode: str) -> str:
        """Run the post-response step inside the operation boundary.

        The in-process handlers answer with a callable the coordinator runs
        after the reply; the pipe carries no deferred action, so the work runs
        here, before the reply is written, and a failure becomes the receipt's
        ``followup`` instead of a promise the peer cannot keep.
        """
        if self._after_response is None:
            return ""
        try:
            outcome = await self._after_response(mode)
        except Exception as exc:
            return f"the post-response step failed ({type(exc).__name__}: {exc})"
        return outcome or ""

    def _cleared_with_followup(
        self, request: ReconfigureRequest, stages: list[str], detail: str
    ) -> ReconfigureReply:
        """The receipt for a clear whose history landed but a stage did not."""
        return ReconfigureReply(
            mode=request.mode,
            outcome=RECONFIGURE_OUTCOME_CLEARED_WITH_FOLLOWUP_FAILURE,
            stage=",".join(stages),
            detail=detail,
            snapshot=self._snapshot(),
        )

    def _require_session(self, session_id: str) -> None:
        if session_id != self._loop.session_id:
            raise AgentSessionNotFoundError(f"Session not found: {session_id}")


def _conversation_messages(loop: AgentLoop) -> list[LLMMessage]:
    """The non-system messages the live loop still carries."""
    return [message for message in loop.messages if message.role is not Role.system]


__all__ = ["AfterResponse", "AgentReconfigureService"]
