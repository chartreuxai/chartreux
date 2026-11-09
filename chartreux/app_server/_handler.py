from __future__ import annotations

import asyncio
import base64
import binascii
from collections.abc import Awaitable, Callable, Iterator
from contextlib import ExitStack, contextmanager, suppress
from dataclasses import dataclass
import errno
import os
from pathlib import Path
import stat
from typing import Any

from pydantic import TypeAdapter

from chartreux.app_server._agent_transcript import (
    read_agent_transcript,
    read_live_agent_transcript,
)
from chartreux.app_server._dispatch import (
    DispatchResult,
    RequestFailure,
    method_not_found,
)
from chartreux.app_server._execution import (
    SessionExecution,
    SessionExecutionConflict,
    SessionExecutionKind,
)
from chartreux.app_server._model import ProtocolModel, validate_wire
from chartreux.app_server._plan_record import session_plan_record
from chartreux.app_server._projection import (
    history_user_message_index,
    project_history,
    project_message_history,
    project_session_log,
)
from chartreux.app_server._resources import ResourceRequestHandler
from chartreux.app_server._review import ReviewRequestHandler
from chartreux.app_server._root_session import RootSessionCoordinator, SessionHandoff
from chartreux.app_server._runtime import (
    AgentRuntimeFactory,
    RuntimeSessionNotFoundError,
    close_agent_loop,
)
from chartreux.app_server._session_model import (
    clear_session_active_model_override,
    clear_session_agent_identity_override,
    override_active_model,
    override_agent_identity,
    override_thinking_level,
    remove_session_thinking_override,
    set_session_active_model_override,
    set_session_agent_identity_override,
    set_session_thinking_override,
)
from chartreux.app_server._sessions import (
    SessionRuntime,
    SessionRuntimeRegistry,
    TranscriptReadSnapshot,
)
from chartreux.app_server._shell import ShellConflictError
from chartreux.app_server._shell_requests import ShellRequestHandler
from chartreux.app_server._state import (
    _turns_from_history,
    build_public_state,
    build_stored_public_state,
    history_page,
)
from chartreux.app_server._turn_queue import (
    TurnQueueFullError,
    TurnQueueIdempotencyConflictError,
    TurnQueueItemNotFoundError,
)
from chartreux.app_server._turns import (
    CallbackClosedError,
    CallbackConflictError,
    CallbackNotFoundError,
    ModelChoicePendingError,
    StaleTurnError,
    TurnConflictError,
    TurnController,
)
from chartreux.app_server._workspace import (
    PromptPreparationError,
    WorkspaceTrustError,
    decide_workspace_trust,
    prepare_prompt,
)
from chartreux.app_server._worktree_session import SessionWorktrees
from chartreux.app_server.models import (
    CallbackOutput,
    PublicCallbackEntry,
    PublicHistoryEntry,
    PublicRecoveryEligibility,
    PublicSessionState,
)
from chartreux.app_server.protocol import (
    AgentsCancelParams,
    AgentsCancelResponse,
    AgentsReconcileParams,
    AgentsReconcileResponse,
    AgentsReleaseParams,
    AgentsReleaseResponse,
    AgentsResumeParams,
    AgentsResumeResponse,
    AgentTranscriptGetParams,
    AgentTranscriptGetResponse,
    AgentTranscriptState,
    CallbackKeepWaitingParams,
    CallbackKeepWaitingResponse,
    CallbackRespondParams,
    CallbackRespondResponse,
    CallbackResultError,
    CallbackResultParams,
    CallbackResultResponse,
    CancelOutcome,
    ContextInjectParams,
    EmptyResponse,
    ProtocolErrorCode,
    ReconcileOutcome,
    ReleaseOutcome,
    ResumeOutcome,
    RootRecoveryAdmitParams,
    RootRecoveryCancelParams,
    RootRecoveryScheduleParams,
    RunStopReason as PublicRunStopReason,
    RuntimeUpdatedParams,
    SessionCompactParams,
    SessionCompactResponse,
    SessionContinueParams,
    SessionContinueResponse,
    SessionForkParams,
    SessionForkResponse,
    SessionHistoryClearParams,
    SessionHistoryClearResponse,
    SessionHistoryListParams,
    SessionHistoryListResponse,
    SessionLogReadParams,
    SessionLogReadResponse,
    SessionPlanHandoffParams,
    SessionPlanHandoffResponse,
    SessionPlanRecord,
    SessionReadParams,
    SessionReadResponse,
    SessionReadyReadParams,
    SessionReadyReadResponse,
    SessionReadyWaitParams,
    SessionReadyWaitResponse,
    SessionRelocateParams,
    SessionRelocateResponse,
    SessionResumeParams,
    SessionResumeResponse,
    SessionRewindParams,
    SessionRewindReadParams,
    SessionRewindReadResponse,
    SessionRewindResponse,
    SessionSettingsUpdateParams,
    SessionStartParams,
    SessionStartResponse,
    SessionStopParams,
    SessionStopResponse,
    SessionTitleUpdateParams,
    SessionTitleUpdateResponse,
    SessionTurnsListParams,
    SessionTurnsListResponse,
    TurnEnqueueParams,
    TurnEnqueueResponse,
    TurnInterruptParams,
    TurnInterruptResponse,
    TurnQueueReadParams,
    TurnQueueReadResponse,
    TurnQueueRemoveParams,
    TurnQueueRemoveResponse,
    TurnQueueReplaceParams,
    TurnQueueReplaceResponse,
    TurnQueueResumeParams,
    TurnQueueResumeResponse,
    TurnStartParams,
    TurnStartResponse,
    TurnSteerParams,
    TurnSteerResponse,
    WorkspacePromptPrepareParams,
    WorkspacePromptPrepareResponse,
    WorkspaceTrustDecisionParams,
)
from chartreux.core.agent_loop import AgentLoop, AgentLoopStateError
from chartreux.core.compaction import CompactionFailedError
from chartreux.core.compaction.manager import CompactionSettlementError
from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
from chartreux.core.config.orchestrator import ConfigOrchestrator
from chartreux.core.git.worktree import (
    TransferAttempt,
    abort_transfer,
    begin_transfer,
    commit_transfer,
)
from chartreux.core.git.worktree.record import OwnershipToken, release_holder
from chartreux.core.llm_models import LLMMessage
from chartreux.core.plan_episode import (
    LOGGING_REQUIRED_MESSAGE,
    STATE_ACCEPTED,
    STATE_ACCEPTING,
    CommitAttempt,
    LoggingDomain,
    LoggingObservation,
    PlanDigestMismatchError,
    PlanEpisodeError,
    PlanEpisodeWriteError,
    PlanFenceBusyError,
    PlanTransitionError,
    begin_commit,
    build_handoff_context,
    observe_logging_domain,
    prepare_accept,
    read_sidecar,
    release_claim,
    require_no_executor,
    sha256_bytes,
)
from chartreux.core.plan_mode import extract_plan_summary
from chartreux.core.rewind import ArchivedHistorySource, archived_history_for
from chartreux.core.session.image_snapshot import ImageSnapshotError
from chartreux.core.session.session_loader import (
    SessionFileContainmentError,
    SessionLoader,
)
from chartreux.core.session.session_logger import SessionDiskFullError
from chartreux.core.session_types import (
    PlanModeRecord,
    PlanScanDomain,
    ScheduledLoop as CoreScheduledLoop,
    SessionMetadata,
)
from chartreux.core.subagents import (
    EffectVerification,
    ReleaseAgentOutcome,
    RunStopReason,
    UnknownAgentError,
)
from chartreux.observability.logging import logger
from chartreux.utils.io import decode_safe

DEFAULT_HISTORY_LIMIT = 200

# A manual /compact summarizes the whole conversation behind the LIFECYCLE
# reserve, so a stalled provider would hold every other request off for as long
# as it waits. This budget requests cooperative cancellation of that wait — it
# is not a cap on the operation or on the response: settlement still runs after
# it fires, so an already-published boundary is reported, never dropped.
_MANUAL_COMPACTION_TIMEOUT_SECONDS = 720.0

# A competing accept either busy-retries a bounded moment and then loses to the
# expected-state verify, or reports that the transaction is in progress. The
# commit fence is never waited on indefinitely (DR-44).
_PLAN_COMMIT_FENCE_RETRY_SECONDS = 5.0

_RELEASE_REASONS: dict[ReleaseAgentOutcome, str] = {
    ReleaseAgentOutcome.RELEASED: (
        "Agent released; retained results and browser access are discarded."
    ),
    ReleaseAgentOutcome.EVICTED: (
        "Terminal metadata removed; retained results and browser access are discarded."
    ),
    ReleaseAgentOutcome.CONFLICT: (
        "Agent is active, reserved for replacement, or its latest run changed; "
        "nothing was released."
    ),
    ReleaseAgentOutcome.UNKNOWN: "Unknown agent; nothing was released.",
}

type ReplaceRoot = Callable[[str, int], Awaitable[PublicSessionState]]
type AdoptRoot = Callable[
    [AgentLoop, int, list[CoreScheduledLoop]], Awaitable[PublicSessionState]
]
type StageRoot = Callable[[AgentLoop], Awaitable[None]]
type SpawnResumeTask = Callable[[asyncio.Task[None]], None]


@dataclass(frozen=True, slots=True)
class RootLifecycle:
    replace: ReplaceRoot
    adopt: AdoptRoot
    stage: StageRoot | None = None


@dataclass(frozen=True, slots=True)
class ResumeOrchestration:
    """Server-owned callbacks that manage the fast-resume background lifecycle.

    Grouped separately from ``RootLifecycle`` because they are server concerns
    (task tracking, post-response notification) rather than session-state concerns.
    """

    runtime_factory: AgentRuntimeFactory
    current_event_id: Callable[[str], int]
    spawn_resume_task: SpawnResumeTask


class CoreRequestHandler:
    def __init__(
        self,
        agent_loop: AgentLoop,
        turns: TurnController,
        execution: SessionExecution,
        notify: Callable[[str, ProtocolModel], Awaitable[None]],
        sessions: SessionRuntimeRegistry,
        resources: ResourceRequestHandler,
        root_session: RootSessionCoordinator,
        root_lifecycle: RootLifecycle,
        resume_orchestration: ResumeOrchestration,
    ) -> None:
        self._agent_loop = agent_loop
        self._turns = turns
        self._execution = execution
        self._notify = notify
        self._sessions = sessions
        self._root_session = root_session
        self._current_event_id = resume_orchestration.current_event_id
        self._shell = ShellRequestHandler(
            agent_loop, turns, execution, self._require_attached, self._current_event_id
        )
        self._resources = resources
        self._review = ReviewRequestHandler(
            agent_loop.review_manager,
            self._require_session,
            self._execution.require_idle,
        )
        self._root_lifecycle = root_lifecycle
        self._runtime_factory = resume_orchestration.runtime_factory
        self._closed = False
        self._spawn_resume_task = resume_orchestration.spawn_resume_task
        self.worktree_token: OwnershipToken | None = None

    async def dispatch(self, method: str, raw_params: dict[str, Any]) -> DispatchResult:
        steer_params: TurnSteerParams | None = None
        try:
            if method == "turn/steer":
                steer_params = validate_wire(TurnSteerParams, raw_params)
            return await self._dispatch_checked(method, raw_params, steer_params)
        except RequestFailure as exc:
            if method == "turn/steer" and (
                steer_params is None
                or self._turns.steer_committed(steer_params) is False
            ):
                data = dict(exc.data) if isinstance(exc.data, dict) else {}
                data["steerCommitted"] = False
                exc.data = data
            raise

    async def _dispatch_checked(
        self,
        method: str,
        raw_params: dict[str, Any],
        steer_params: TurnSteerParams | None,
    ) -> DispatchResult:
        try:
            active = self._execution.active
            if (
                active is not None
                and active.kind is SessionExecutionKind.LIFECYCLE
                and method not in {"turn/interrupt", "agents/cancel"}
                and not (
                    steer_params is not None
                    and self._turns.known_steer_receipt(steer_params)
                )
            ):
                raise SessionExecutionConflict(
                    f"Session lifecycle transition is active: {active.id}"
                )
            if steer_params is not None:
                return await self._dispatch_turn(method, raw_params, steer_params)
            return await self._dispatch(method, raw_params)
        except TurnConflictError as exc:
            raise RequestFailure(ProtocolErrorCode.CONFLICT, str(exc)) from exc
        except TurnQueueFullError as exc:
            raise RequestFailure(
                ProtocolErrorCode.CONFLICT, str(exc), data={"maxItems": exc.max_items}
            ) from exc
        except TurnQueueIdempotencyConflictError as exc:
            raise RequestFailure(
                ProtocolErrorCode.CONFLICT,
                str(exc),
                data={"idempotencyKey": exc.idempotency_key},
            ) from exc
        except TurnQueueItemNotFoundError as exc:
            raise RequestFailure(
                ProtocolErrorCode.NOT_FOUND,
                str(exc),
                data={"queueItemId": exc.queue_item_id},
            ) from exc
        except StaleTurnError as exc:
            raise RequestFailure(
                ProtocolErrorCode.STALE_TURN,
                str(exc),
                data={"activeTurnId": exc.active_turn_id},
            ) from exc
        except CallbackNotFoundError as exc:
            raise RequestFailure(ProtocolErrorCode.NOT_FOUND, str(exc)) from exc
        except CallbackClosedError as exc:
            raise RequestFailure(ProtocolErrorCode.CALLBACK_CLOSED, str(exc)) from exc
        except CallbackConflictError as exc:
            raise RequestFailure(ProtocolErrorCode.CONFLICT, str(exc)) from exc
        except (ImageSnapshotError, PromptPreparationError, WorkspaceTrustError) as exc:
            raise RequestFailure(ProtocolErrorCode.INVALID_PARAMS, str(exc)) from exc
        except ModelChoicePendingError as exc:
            # A recovered session has no usable model selection; the message
            # routes the client to pick one instead of failing opaquely.
            raise RequestFailure(ProtocolErrorCode.INVALID_PARAMS, str(exc)) from exc
        except ShellConflictError as exc:
            raise RequestFailure(ProtocolErrorCode.CONFLICT, str(exc)) from exc
        except SessionExecutionConflict as exc:
            raise RequestFailure(ProtocolErrorCode.CONFLICT, str(exc)) from exc

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._shell.close()

    async def _dispatch(
        self, method: str, raw_params: dict[str, Any]
    ) -> DispatchResult:
        if method.startswith("app_server/session/turn/"):
            return await self._dispatch_turn(method, raw_params)
        namespace = method.partition("/")[0]
        match namespace:
            case "agents":
                result = await self._dispatch_agents(method, raw_params)
            case "agent":
                result = await self._dispatch_agent(method, raw_params)
            case "session":
                result = await self._dispatch_session(method, raw_params)
            case "turn":
                result = await self._dispatch_turn(method, raw_params)
            case "workspace":
                result = await self._dispatch_workspace(method, raw_params)
            case "callback":
                result = await self._dispatch_callback(method, raw_params)
            case "recovery":
                result = await self._dispatch_recovery(method, raw_params)
            case "review":
                result = self._review.dispatch(method, raw_params)
            case (
                "account"
                | "identity"
                | "runtime"
                | "config"
                | "skills"
                | "tools"
                | "stats"
                | "diagnostics"
                | "mcp"
                | "loops"
                | "narration"
            ):
                result = await self._resources.dispatch(method, raw_params)
            case _:
                raise method_not_found(method)
        return result

    async def _dispatch_agents(
        self, method: str, raw_params: dict[str, Any]
    ) -> DispatchResult:
        if method == "agents/cancel":
            params = validate_wire(AgentsCancelParams, raw_params)
            self._require_attached(self._agent_loop.session_id)
            if self._closed:
                raise RequestFailure(ProtocolErrorCode.CONFLICT, "Session is closing")
            result = await self._sessions.cancel_run(
                params.agent_id,
                params.run_id,
                reason=RunStopReason.USER_CANCELLED,
                requester_session_id=self._agent_loop.session_id,
            )
            return DispatchResult(
                AgentsCancelResponse(
                    outcome=CancelOutcome(result.outcome.value),
                    run_id=result.run_id,
                    stop_reason=(
                        PublicRunStopReason(result.stop_reason.value)
                        if result.stop_reason is not None
                        else None
                    ),
                )
            )
        if method == "agents/release":
            # Guarded UI release: keeps the standard lifecycle guards (no
            # cancel-style exemption) plus the registry's atomic race guards.
            params = validate_wire(AgentsReleaseParams, raw_params)
            self._require_attached(self._agent_loop.session_id)
            if self._closed:
                raise RequestFailure(ProtocolErrorCode.CONFLICT, "Session is closing")
            outcome = await self._sessions.release_agent_guarded(
                params.agent_id, params.expected_run_id
            )
            return DispatchResult(
                AgentsReleaseResponse(
                    outcome=ReleaseOutcome(outcome.value),
                    reason=_RELEASE_REASONS[outcome],
                )
            )
        if method == "agents/resume":
            # WP2: explicit warm resume of a suspended failed run. The
            # owning parent's attached session is the requester.
            params = validate_wire(AgentsResumeParams, raw_params)
            self._require_attached(self._agent_loop.session_id)
            if self._closed:
                raise RequestFailure(ProtocolErrorCode.CONFLICT, "Session is closing")
            idempotency_key = params.idempotency_key or (
                f"{params.agent_id}:{params.expected_run_id}"
            )
            result = await self._sessions.resume_agent(
                params.agent_id,
                params.expected_run_id,
                idempotency_key,
                requester_session_id=self._agent_loop.session_id,
            )
            return DispatchResult(
                AgentsResumeResponse(
                    outcome=ResumeOutcome(result.outcome.value),
                    agent_id=result.agent_id,
                    run_id=result.run_id,
                    successor_run_id=result.successor_run_id,
                    recovery_operation_id=result.recovery_operation_id,
                    retry_at=result.retry_at,
                    eligibility=(
                        PublicRecoveryEligibility(result.eligibility.value)
                        if result.eligibility is not None
                        else None
                    ),
                    reason_codes=result.reason_codes,
                    reason=result.reason,
                    same_agent_resumes_used=result.same_agent_resumes_used,
                )
            )
        if method == "agents/reconcile":
            # WP2: reconcile_result pinned to the failed run and its
            # transcript boundary; clears or prohibits the barrier.
            params = validate_wire(AgentsReconcileParams, raw_params)
            self._require_attached(self._agent_loop.session_id)
            if self._closed:
                raise RequestFailure(ProtocolErrorCode.CONFLICT, "Session is closing")
            result = await self._sessions.reconcile_result(
                params.agent_id,
                params.expected_run_id,
                params.transcript_boundary,
                [
                    EffectVerification(
                        tool_call_id=effect.tool_call_id,
                        disposition=effect.disposition,
                        note=effect.note,
                    )
                    for effect in params.verified_effects
                ],
                requester_session_id=self._agent_loop.session_id,
            )
            return DispatchResult(
                AgentsReconcileResponse(
                    outcome=ReconcileOutcome(result.outcome.value),
                    agent_id=result.agent_id,
                    run_id=result.run_id,
                    transcript_boundary=result.transcript_boundary,
                    eligibility=(
                        PublicRecoveryEligibility(result.eligibility.value)
                        if result.eligibility is not None
                        else None
                    ),
                    reason_codes=result.reason_codes,
                    reason=result.reason,
                )
            )
        raise method_not_found(method)

    async def _dispatch_agent(
        self, method: str, raw_params: dict[str, Any]
    ) -> DispatchResult:
        if method != "agent/transcript/get":
            raise method_not_found(method)
        params = validate_wire(AgentTranscriptGetParams, raw_params)
        parent_identity = (
            self._agent_loop.session_id,
            self._agent_loop._session_generation,
        )
        self._require_attached(parent_identity[0])
        try:
            resident_snapshot = await self._sessions.resolve_resident_transcript_read(
                params.agent_id
            )
        except UnknownAgentError as exc:
            raise RequestFailure(
                ProtocolErrorCode.NOT_FOUND, "Agent not found"
            ) from exc
        if resident_snapshot is not None:
            if resident_snapshot.parent_identity != parent_identity:
                raise RequestFailure(
                    ProtocolErrorCode.CONFLICT,
                    "Transcript parent changed while reading",
                )
            response = await asyncio.to_thread(
                read_live_agent_transcript,
                resident_snapshot.messages,
                before=params.before,
                limit=params.limit,
            )
            if not await self._sessions.resident_transcript_read_is_current(
                resident_snapshot
            ):
                raise RequestFailure(
                    ProtocolErrorCode.CONFLICT,
                    "Transcript read is no longer authorized",
                )
            return DispatchResult(response)
        try:
            snapshot = await self._sessions.resolve_transcript_read(params.agent_id)
        except UnknownAgentError as exc:
            raise RequestFailure(
                ProtocolErrorCode.NOT_FOUND, "Agent not found"
            ) from exc
        if snapshot.parent_identity != parent_identity:
            raise RequestFailure(
                ProtocolErrorCode.CONFLICT, "Transcript parent changed while reading"
            )
        if not snapshot.has_saved_transcript:
            return DispatchResult(
                AgentTranscriptGetResponse(
                    state=AgentTranscriptState.NO_SAVED_TRANSCRIPT
                )
            )
        try:
            response = await asyncio.to_thread(
                _read_authorized_agent_transcript, snapshot, params.before, params.limit
            )
        except SessionFileContainmentError as exc:
            raise RequestFailure(
                ProtocolErrorCode.CONFLICT, "Saved transcript failed containment checks"
            ) from exc
        if not await self._sessions.transcript_read_is_current(snapshot):
            raise RequestFailure(
                ProtocolErrorCode.CONFLICT, "Transcript read is no longer authorized"
            )
        return DispatchResult(response)

    async def _dispatch_session(
        self, method: str, raw_params: dict[str, Any]
    ) -> DispatchResult:
        if method == "session/shellCommand":
            return await self._shell.dispatch(method, raw_params)
        if method in {
            "session/start",
            "session/resume",
            "session/continue",
            "session/stop",
        }:
            return await self._dispatch_session_lifecycle(method, raw_params)
        if (
            method == "session/read"
            or method.startswith("session/rewind")
            or method
            in {
                "session/history/list",
                "session/turns/list",
                "session/rename",
                "session/compact",
            }
        ):
            return await self._dispatch_session_delegated(method, raw_params)
        runtime_updated = False
        session_attached = False
        after_response: Callable[[], None] | None = None
        match method:
            case "session/ready/wait":
                response: ProtocolModel = await self._wait_ready(
                    validate_wire(SessionReadyWaitParams, raw_params)
                )
            case "session/ready/read":
                params = validate_wire(SessionReadyReadParams, raw_params)
                self._require_session(params.session_id)
                response = SessionReadyReadResponse(
                    ready=self._agent_loop.is_initialized
                )
            case "session/fork":
                params = validate_wire(SessionForkParams, raw_params)
                response = await self._session_fork(params)
                session_attached = params.attach
            case "session/settings/update":
                response = self._session_settings_update(
                    validate_wire(SessionSettingsUpdateParams, raw_params)
                )
            case "session/relocate":
                response = await self._relocate(
                    validate_wire(SessionRelocateParams, raw_params)
                )
                runtime_updated = True
            case "session/log/read":
                response = self._session_log_read(
                    validate_wire(SessionLogReadParams, raw_params)
                )
            case "session/context/inject":
                params = validate_wire(ContextInjectParams, raw_params)
                self._require_attached(params.session_id)
                response = await self._turns.inject(params)
            case "session/history/clear":
                response, after_response = await self._history_clear(
                    validate_wire(SessionHistoryClearParams, raw_params)
                )
            case "session/plan/handoff":
                response, after_response = await self._plan_handoff(
                    validate_wire(SessionPlanHandoffParams, raw_params)
                )
            case _:
                raise method_not_found(method)
        return DispatchResult(
            response,
            after_response=after_response,
            runtime_updated=runtime_updated,
            session_attached=session_attached,
        )

    async def _dispatch_session_delegated(
        self, method: str, raw_params: dict[str, Any]
    ) -> DispatchResult:
        if method == "session/read":
            return await self._dispatch_session_records(method, raw_params)
        if method.startswith("session/rewind"):
            return await self._dispatch_rewind(method, raw_params)
        if method == "session/rename":
            return DispatchResult(
                await self._session_title_update(
                    validate_wire(SessionTitleUpdateParams, raw_params)
                )
            )
        if method == "session/compact":
            response, after_response = await self._compact(
                validate_wire(SessionCompactParams, raw_params)
            )
            return DispatchResult(response, after_response=after_response)
        return await self._dispatch_session_catalog(method, raw_params)

    async def _dispatch_session_catalog(
        self, method: str, raw_params: dict[str, Any]
    ) -> DispatchResult:
        match method:
            case "session/history/list":
                return DispatchResult(
                    await self._history_list(
                        validate_wire(SessionHistoryListParams, raw_params)
                    )
                )
            case "session/turns/list":
                return DispatchResult(
                    await self._session_turns_list(
                        validate_wire(SessionTurnsListParams, raw_params)
                    )
                )
        raise method_not_found(method)

    async def _read_child_transcript(
        self, session_id: str
    ) -> tuple[SessionRuntime | None, list[LLMMessage] | None, SessionMetadata | None]:
        """Read a linked inactive child without materializing its runtime."""
        for _ in range(2):
            child = self._sessions.resolve_child(session_id)
            if child is None:
                return None, None, None
            if isinstance(child, SessionRuntime):
                return child, None, None
            try:
                messages, raw_metadata = await asyncio.to_thread(
                    SessionLoader.load_session, child.child_dir
                )
                metadata = SessionMetadata.model_validate(raw_metadata)
            except Exception:
                return None, None, None
            if self._sessions.stored_child_is_current(child):
                return None, messages, metadata
        child = self._sessions.resolve_child(session_id)
        return (
            (child, None, None)
            if isinstance(child, SessionRuntime)
            else (None, None, None)
        )

    async def _dispatch_session_records(
        self, method: str, raw_params: dict[str, Any]
    ) -> DispatchResult:
        match method:
            case "session/read":
                params = validate_wire(SessionReadParams, raw_params)
                child, messages, metadata = await self._read_child_transcript(
                    params.session_id
                )
                if child is not None:
                    public = self._sessions.public_state(
                        params.session_id,
                        params.history_limit,
                        turns_limit=params.turns_limit,
                        include_history=params.include_history,
                        include_turns=params.include_turns,
                    )
                elif messages is not None and metadata is not None:
                    public = build_stored_public_state(
                        params.session_id,
                        messages,
                        metadata,
                        history_limit=params.history_limit,
                        turns_limit=params.turns_limit,
                        include_history=params.include_history,
                        include_turns=params.include_turns,
                    )
                else:
                    self._require_session(params.session_id)
                    public = self._public_state(
                        params.history_limit,
                        turns_limit=params.turns_limit,
                        include_history=params.include_history,
                        include_turns=params.include_turns,
                    )
                response: ProtocolModel = SessionReadResponse(
                    state=public, last_event_id=public.event_id
                )
            case _:
                raise method_not_found(method)
        return DispatchResult(response)

    async def _dispatch_session_lifecycle(
        self, method: str, raw_params: dict[str, Any]
    ) -> DispatchResult:
        if method == "session/start":
            params = validate_wire(SessionStartParams, raw_params)
            if self._root_session.attached_session_id is not None:
                raise RequestFailure(
                    ProtocolErrorCode.CONFLICT, "A session is already attached"
                )
            if (
                params.cwd is not None
                and Path(params.cwd).resolve() != self._agent_loop.cwd
            ):
                raise RequestFailure(
                    ProtocolErrorCode.INVALID_PARAMS,
                    "The app server was started in a different working directory",
                )
            self._root_session.attach(self._agent_loop.session_id)
            return DispatchResult(
                SessionStartResponse(
                    state=(state := self._public_state(params.history_limit)),
                    last_event_id=state.event_id,
                ),
                session_attached=True,
            )
        if method == "session/resume":
            return await self._session_resume(
                validate_wire(SessionResumeParams, raw_params)
            )
        if method == "session/continue":
            params = validate_wire(SessionContinueParams, raw_params)
            if self._root_session.attached_session_id is not None:
                raise RequestFailure(
                    ProtocolErrorCode.CONFLICT, "A session is already attached"
                )
            cwd = (
                Path(params.cwd).resolve()
                if params.cwd is not None
                else self._agent_loop.cwd
            )
            try:
                session_id = self._runtime_factory.resolve_latest(self._agent_loop, cwd)
            except RuntimeSessionNotFoundError as exc:
                raise RequestFailure(ProtocolErrorCode.NOT_FOUND, str(exc)) from exc
            state = await self._root_lifecycle.replace(session_id, params.history_limit)
            return DispatchResult(
                SessionContinueResponse(state=state, last_event_id=state.event_id),
                session_attached=True,
                runtime_updated=True,
            )
        if method == "session/stop":
            stop_params = validate_wire(SessionStopParams, raw_params)
            self._require_attached(stop_params.session_id)
            await self.close()
            return DispatchResult(SessionStopResponse())
        raise method_not_found(method)

    async def _dispatch_rewind(
        self, method: str, raw_params: dict[str, Any]
    ) -> DispatchResult:
        after_response: Callable[[], None] | None = None
        match method:
            case "session/rewind/read":
                response: ProtocolModel = self._rewind_read(
                    validate_wire(SessionRewindReadParams, raw_params)
                )
            case "session/rewind":
                response, after_response = await self._rewind(
                    validate_wire(SessionRewindParams, raw_params)
                )
            case _:
                raise method_not_found(method)
        return DispatchResult(response, after_response=after_response)

    async def _dispatch_turn(
        self,
        method: str,
        raw_params: dict[str, Any],
        steer_params: TurnSteerParams | None = None,
    ) -> DispatchResult:
        response: ProtocolModel
        after_response: Callable[[], None] | None = None
        match method:
            case "app_server/session/turn/enqueue":
                params = validate_wire(TurnEnqueueParams, raw_params)
                self._require_attached(params.session_id)
                result, after_response = self._turns.enqueue(params)
                response = TurnEnqueueResponse(
                    queue_item_id=result.record.queued_turn.id
                )
            case "app_server/session/turn/queue/read":
                params = validate_wire(TurnQueueReadParams, raw_params)
                self._require_attached(params.session_id)
                response = TurnQueueReadResponse(queue=self._turns.queue_state)
            case "app_server/session/turn/queue/remove":
                params = validate_wire(TurnQueueRemoveParams, raw_params)
                self._require_attached(params.session_id)
                _, after_response = self._turns.remove_queued_turn(params.queue_item_id)
                response = TurnQueueRemoveResponse()
            case "app_server/session/turn/queue/replace":
                params = validate_wire(TurnQueueReplaceParams, raw_params)
                self._require_attached(params.session_id)
                result, after_response = self._turns.replace_queued_turn(
                    params.queue_item_id, params.as_enqueue_params()
                )
                response = TurnQueueReplaceResponse(
                    queue_item_id=result.record.queued_turn.id
                )
            case "app_server/session/turn/queue/resume":
                params = validate_wire(TurnQueueResumeParams, raw_params)
                self._require_attached(params.session_id)
                after_response = self._turns.resume_queue()
                response = TurnQueueResumeResponse()
            case "turn/start":
                start_params = validate_wire(TurnStartParams, raw_params)
                self._require_attached(start_params.session_id)
                vibe_turn, after_response = self._turns.start(start_params)
                response = TurnStartResponse(
                    turn=vibe_turn.turn,
                    last_event_id=self._current_event_id(start_params.session_id),
                )
            case "turn/steer":
                assert steer_params is not None
                if not self._turns.known_steer_receipt(steer_params):
                    if self._closed and steer_params.require_waiting_only:
                        raise TurnConflictError("Session is closing")
                    self._require_turn_route(
                        steer_params.session_id, steer_params.expected_turn_id
                    )
                await self._turns.steer(steer_params)
                response = TurnSteerResponse(
                    last_event_id=self._current_event_id(steer_params.session_id)
                )
            case "turn/interrupt":
                params = validate_wire(TurnInterruptParams, raw_params)
                if self._turns.active_turn is None:
                    self._require_attached(params.session_id)
                else:
                    self._require_turn_route(params.session_id, params.expected_turn_id)
                self._turns.interrupt(params)
                response = TurnInterruptResponse(
                    last_event_id=self._current_event_id(params.session_id)
                )
            case _:
                raise method_not_found(method)
        return DispatchResult(response, after_response)

    async def _dispatch_workspace(
        self, method: str, raw_params: dict[str, Any]
    ) -> DispatchResult:
        match method:
            case "workspace/prompt/prepare":
                params = validate_wire(WorkspacePromptPrepareParams, raw_params)
                self._require_session(params.session_id)
                prompt = await asyncio.to_thread(
                    prepare_prompt,
                    self._agent_loop,
                    params.message,
                    params.title_content,
                )
                response: ProtocolModel = WorkspacePromptPrepareResponse(prompt=prompt)
                runtime_updated = False
            case "workspace/trust/decision":
                params = validate_wire(WorkspaceTrustDecisionParams, raw_params)
                response = await self._workspace_trust_decision(params)
                runtime_updated = params.decision in {"trust_repo", "trust_cwd"}
            case _:
                raise method_not_found(method)
        return DispatchResult(response, runtime_updated=runtime_updated)

    async def _workspace_trust_decision(
        self, params: WorkspaceTrustDecisionParams
    ) -> ProtocolModel:
        if params.session_id is None:
            raise RequestFailure(
                ProtocolErrorCode.INVALID_PARAMS,
                "Active workspace trust decisions require a session ID",
            )
        grant = params.decision in {"trust_repo", "trust_cwd"}
        self._require_session(params.session_id)
        if grant:
            self._execution.require_idle()

        session_cwd = self._agent_loop.cwd.expanduser().resolve()
        session_cwd_stat = session_cwd.stat(follow_symlinks=False)
        session_cwd_identity = (session_cwd_stat.st_dev, session_cwd_stat.st_ino)
        cwd = (
            Path(params.cwd).expanduser().resolve()
            if params.cwd is not None
            else session_cwd
        )
        if cwd != session_cwd:
            raise RequestFailure(
                ProtocolErrorCode.INVALID_PARAMS,
                "Trust decision cwd must match the session working directory",
            )
        response = await asyncio.to_thread(
            decide_workspace_trust,
            cwd,
            params.decision,
            self._agent_loop.harness_files.trust_store,
            expected_cwd=session_cwd,
            expected_cwd_identity=session_cwd_identity,
        )
        if not grant:
            return response

        await self._agent_loop.config_orchestrator.reload()
        await self._agent_loop.reload_with_initial_messages(reload_hooks=True)
        return response

    async def _dispatch_callback(
        self, method: str, raw_params: dict[str, Any]
    ) -> DispatchResult:
        if method == "callback/result":
            return DispatchResult(
                await self._callback_result(
                    validate_wire(CallbackResultParams, raw_params)
                )
            )
        if method == "callback/keepWaiting":
            return DispatchResult(
                await self._keep_waiting_callback(
                    validate_wire(CallbackKeepWaitingParams, raw_params)
                )
            )
        raise method_not_found(method)

    async def _dispatch_recovery(
        self, method: str, raw_params: dict[str, Any]
    ) -> DispatchResult:
        """R2/OD3: the root automatic-recovery gate's wire surface.

        ``recovery/root/schedule|admit|cancel`` are root-session operations
        (``_require_attached`` rejects a child session's id); the gate itself
        owns authoritative eligibility, episode accounting, and single-flight
        admission. The TUI controller is the only production caller.
        """
        if method == "recovery/root/schedule":
            params = validate_wire(RootRecoveryScheduleParams, raw_params)
            self._require_attached(params.session_id)
            return DispatchResult(await self._turns.root_recovery.schedule(params))
        if method == "recovery/root/admit":
            params = validate_wire(RootRecoveryAdmitParams, raw_params)
            self._require_attached(params.session_id)
            return DispatchResult(await self._turns.root_recovery.admit(params))
        if method == "recovery/root/cancel":
            params = validate_wire(RootRecoveryCancelParams, raw_params)
            self._require_attached(params.session_id)
            return DispatchResult(await self._turns.root_recovery.cancel(params))
        raise method_not_found(method)

    async def _keep_waiting_callback(
        self, params: CallbackKeepWaitingParams
    ) -> CallbackKeepWaitingResponse:
        # Child sessions have no root turn controller; their callbacks are
        # answered through the parent, so a keep-waiting request has no
        # deadline to lift (plan U/OD1).
        if await self._sessions.ensure_child(params.session_id):
            return CallbackKeepWaitingResponse(status="no_deadline")
        self._require_attached(params.session_id)
        status = await self._turns.keep_waiting_callback(params.callback_id)
        return CallbackKeepWaitingResponse(status=status)

    async def _callback_result(
        self, params: CallbackResultParams
    ) -> CallbackResultResponse:
        if params.result.error is not None:
            await self._reject_callback(
                params.session_id, params.callback_id, params.result.error
            )
        elif params.result.output is None:
            raise RequestFailure(
                ProtocolErrorCode.INVALID_PARAMS,
                "Callback result must include output or error",
            )
        else:
            await self._callback_respond(
                CallbackRespondParams(
                    session_id=params.session_id,
                    callback_id=params.callback_id,
                    output=TypeAdapter(CallbackOutput).validate_python(
                        params.result.output
                    ),
                )
            )
        return CallbackResultResponse(
            last_event_id=self._current_event_id(params.session_id)
        )

    async def _session_resume(self, params: SessionResumeParams) -> DispatchResult:
        if params.session_id != self._agent_loop.session_id:
            state = await self._root_lifecycle.replace(
                params.session_id, params.history_limit
            )
            agent_loop = self._agent_loop
            session_id = params.session_id

            def _spawn() -> None:
                task = asyncio.create_task(self._finish_resume(agent_loop, session_id))
                self._spawn_resume_task(task)

            return DispatchResult(
                SessionResumeResponse(
                    state=state,
                    last_event_id=state.event_id,
                    plan_record=self._plan_record(),
                ),
                after_response=_spawn,
                session_attached=True,
                runtime_updated=True,
            )
        self._root_session.attach(params.session_id)
        return DispatchResult(
            SessionResumeResponse(
                state=(state := self._public_state(params.history_limit)),
                last_event_id=state.event_id,
                plan_record=self._plan_record(),
            ),
            session_attached=True,
            runtime_updated=True,
        )

    def _plan_record(self) -> SessionPlanRecord | None:
        """The resumed session's persisted plan mode, read from its metadata.

        Sent with the resume response because the restore has to be in place
        before the session is marked ready; a separate read would land after
        that point.
        """
        metadata = self._agent_loop.session_logger.session_metadata
        if metadata is None or metadata.plan_mode is None:
            return None
        return session_plan_record(metadata.plan_mode)

    async def _finish_resume(self, agent_loop: AgentLoop, session_id: str) -> None:
        session_id = agent_loop.session_id
        await self._runtime_factory.finish_resume_root(agent_loop, session_id)
        try:
            await self._notify(
                "runtime/updated",
                RuntimeUpdatedParams(
                    session_id=session_id, runtime=self._resources.runtime_snapshot()
                ),
            )
        except Exception:
            logger.exception(
                "Failed to emit runtime/updated after resuming session_id=%s",
                session_id,
            )

    async def _session_title_update(
        self, params: SessionTitleUpdateParams
    ) -> SessionTitleUpdateResponse:
        self._require_session(params.session_id)
        session_logger = self._agent_loop.session_logger
        try:
            updated_at = await session_logger.apply_manual_title(params.title)
        except ValueError as exc:
            raise RequestFailure(ProtocolErrorCode.INVALID_PARAMS, str(exc)) from exc
        except RuntimeError as exc:
            # Corrupt on-disk metadata is a genuine internal fault; keep its
            # message instead of collapsing to an opaque INTERNAL_ERROR.
            raise RequestFailure(ProtocolErrorCode.INTERNAL_ERROR, str(exc)) from exc
        title = session_logger.title
        if title is None:
            raise RuntimeError("The session title was not updated")
        if updated_at is None and session_logger.session_metadata is not None:
            updated_at = session_logger.session_metadata.end_time
        return SessionTitleUpdateResponse(title=title, updated_at=updated_at)

    async def _session_fork(self, params: SessionForkParams) -> SessionForkResponse:
        self._require_attached(params.source_session_id)
        if self._agent_loop._is_subagent:
            raise RequestFailure(
                ProtocolErrorCode.INVALID_PARAMS, "Child sessions cannot be forked"
            )
        if (
            not params.attach
            and self._root_lifecycle.stage is None
            and not self._agent_loop.session_logger.enabled
        ):
            raise RequestFailure(
                ProtocolErrorCode.CONFLICT,
                "Detached forks require session logging to be enabled",
            )

        message_id: str | None = None
        if params.entry_id is not None:
            index = history_user_message_index(self._agent_loop, params.entry_id)
            if index is None:
                raise RequestFailure(
                    ProtocolErrorCode.NOT_FOUND,
                    f"Forkable history entry not found: {params.entry_id}",
                )
            message_id = self._agent_loop.messages[index].message_id
            if message_id is None:
                raise RequestFailure(
                    ProtocolErrorCode.CONFLICT,
                    "The selected history entry has no stable source message ID",
                )

        with self._execution.reserve(
            SessionExecutionKind.LIFECYCLE, f"fork:{params.source_session_id}"
        ):
            try:
                forked: AgentLoop | None = await self._runtime_factory.fork(
                    self._agent_loop, message_id
                )
            except ValueError as exc:
                raise RequestFailure(
                    ProtocolErrorCode.INVALID_PARAMS, str(exc)
                ) from exc

            if params.attach:
                state = await self._root_lifecycle.adopt(
                    forked, params.history_limit, self._resources.loop_snapshot()
                )
                return SessionForkResponse(
                    source_session_id=params.source_session_id,
                    state=state,
                    last_event_id=state.event_id,
                )

            try:
                await self._resources.copy_loops_to(forked)
                history = project_history(forked)
                state = build_public_state(
                    forked,
                    history=history,
                    current_history=[],
                    callbacks=[],
                    turns=[],
                    retrying=None,
                    history_limit=params.history_limit,
                )
                if (
                    self._root_lifecycle.stage is not None
                    and not forked.session_logger.enabled
                ):
                    await self._root_lifecycle.stage(forked)
                    forked = None
            finally:
                if forked is not None:
                    await close_agent_loop(forked)

        return SessionForkResponse(
            source_session_id=params.source_session_id,
            state=state,
            last_event_id=state.event_id,
        )

    def _rewind_read(
        self, params: SessionRewindReadParams
    ) -> SessionRewindReadResponse:
        self._require_session(params.session_id)
        index = self._rewind_index(params.entry_id)
        paths = self._agent_loop.rewind_manager.restorable_paths_at(index)
        return SessionRewindReadResponse(has_file_changes=bool(paths), paths=paths)

    async def _rewind(
        self, params: SessionRewindParams
    ) -> tuple[SessionRewindResponse, Callable[[], None] | None]:
        self._require_attached(params.session_id)
        index = self._rewind_index(params.entry_id)
        history = self._all_history()
        history_index = next(
            (
                position
                for position, entry in enumerate(history)
                if entry.id == params.entry_id
            ),
            None,
        )
        if history_index is None:
            raise RuntimeError(
                f"Rewindable core message is missing from public history: {params.entry_id}"
            )
        with self._execution.reserve(
            SessionExecutionKind.LIFECYCLE, f"rewind:{params.entry_id}"
        ):
            (
                message,
                restore_errors,
                restored_paths,
            ) = await self._agent_loop.rewind_manager.rewind_to_message(
                index,
                restore_files=params.restore_files,
                inplace=params.inplace,
                archive=self._rewind_archive(),
            )
            after_response = await self._turns.reset()
            handoff = self._root_session.replace_idle_with_history(
                params.session_id,
                history=history[:history_index],
                checkpoint_kind="rewind",
                checkpoint_message="Conversation rewound",
                checkpoint_details={
                    "entryId": params.entry_id,
                    "restoreFiles": params.restore_files,
                    "inplace": params.inplace,
                },
            )
        return (
            SessionRewindResponse(
                message=message,
                restore_errors=restore_errors,
                restored_paths=restored_paths,
                state=handoff.state,
                session_log=handoff.session_log,
            ),
            after_response,
        )

    def _rewind_archive(self) -> ArchivedHistorySource | None:
        """The sealed prefix this rewind operation hydrates from (M1 §3).

        Operation-scoped by construction: the source is resolved from the
        live message list and transcript at the moment of the rewind, so a
        rewind targeting an archived turn reads complete records while
        ordinary saves never hydrate.
        """
        return archived_history_for(
            self._agent_loop.messages, self._agent_loop.session_logger
        )

    def _rewind_index(self, entry_id: str) -> int:
        index = history_user_message_index(self._agent_loop, entry_id)
        if index is None:
            raise RequestFailure(
                ProtocolErrorCode.NOT_FOUND,
                f"Rewindable history entry not found: {entry_id}",
            )
        return index

    async def _history_list(
        self, params: SessionHistoryListParams
    ) -> SessionHistoryListResponse:
        child, messages, metadata = await self._read_child_transcript(params.session_id)
        if child is not None:
            history = self._sessions.history(params.session_id)
        elif messages is not None and metadata is not None:
            history = project_message_history(params.session_id, messages, metadata)
        else:
            self._require_session(params.session_id)
            history = self._all_history()
        page = history_page(
            history,
            turn_id=params.turn_id,
            before=params.cursor if params.sort_direction == "backward" else None,
            after=params.cursor if params.sort_direction == "forward" else None,
            limit=params.limit,
        )
        return SessionHistoryListResponse(
            items=page.entries,
            next_cursor=(
                page.cursor.before
                if params.sort_direction == "backward"
                else page.cursor.after
            ),
            previous_cursor=(
                page.cursor.after
                if params.sort_direction == "backward"
                else page.cursor.before
            ),
        )

    async def _session_turns_list(
        self, params: SessionTurnsListParams
    ) -> SessionTurnsListResponse:
        child, messages, metadata = await self._read_child_transcript(params.session_id)
        if child is not None:
            turns = self._sessions.turns(params.session_id)
        elif messages is not None and metadata is not None:
            history = project_message_history(params.session_id, messages, metadata)
            turns = _turns_from_history(history, params.session_id)
        else:
            self._require_session(params.session_id)
            turns = self._turns.turns
        if params.sort_direction == "backward":
            if params.cursor is None:
                page = turns[-params.limit :]
                first_index = max(0, len(turns) - len(page))
            else:
                end = next(
                    (
                        index
                        for index, turn in enumerate(turns)
                        if turn.id == params.cursor
                    ),
                    0,
                )
                first_index = max(0, end - params.limit)
                page = turns[first_index:end]
            last_index = first_index + len(page) - 1
        else:
            first_index = (
                0
                if params.cursor is None
                else next(
                    (
                        index + 1
                        for index, turn in enumerate(turns)
                        if turn.id == params.cursor
                    ),
                    len(turns),
                )
            )
            page = turns[first_index : first_index + params.limit]
            last_index = first_index + len(page) - 1
        next_cursor = page[0].id if page and first_index > 0 else None
        backwards_cursor = page[-1].id if page and last_index < len(turns) - 1 else None
        if params.sort_direction == "forward":
            next_cursor, backwards_cursor = backwards_cursor, next_cursor
        return SessionTurnsListResponse(
            items=page, next_cursor=next_cursor, previous_cursor=backwards_cursor
        )

    async def _wait_ready(
        self, params: SessionReadyWaitParams
    ) -> SessionReadyWaitResponse:
        self._require_session(params.session_id)
        await self._agent_loop.wait_until_ready()
        return SessionReadyWaitResponse(
            init_duration_ms=self._agent_loop.init_duration_ms
        )

    def _session_settings_update(
        self, params: SessionSettingsUpdateParams
    ) -> EmptyResponse:
        self._require_session(params.session_id)
        if params.max_turns is not None:
            self._agent_loop.set_max_turns(params.max_turns)
        if params.max_tokens is not None:
            self._agent_loop.set_max_tokens(params.max_tokens)
        return EmptyResponse()

    def _session_log_read(self, params: SessionLogReadParams) -> SessionLogReadResponse:
        self._require_session(params.session_id)
        return SessionLogReadResponse(log=project_session_log(self._agent_loop))

    async def _callback_respond(
        self, params: CallbackRespondParams
    ) -> CallbackRespondResponse:
        if await self._sessions.ensure_child(params.session_id):
            attached = self._root_session.attached_session_id
            if attached is None or not self._sessions.child_belongs_to(
                params.session_id, attached
            ):
                raise RequestFailure(
                    ProtocolErrorCode.CONFLICT,
                    "Child session is not linked to the attached session",
                )
            status = await self._sessions.answer_callback(
                params.session_id, params.callback_id, params.output
            )
            return CallbackRespondResponse.model_validate({"status": status})
        self._require_attached(params.session_id)
        status = await self._turns.answer_callback(params.callback_id, params.output)
        return CallbackRespondResponse.model_validate({"status": status})

    async def _reject_callback(
        self, session_id: str, callback_id: str, error: CallbackResultError
    ) -> str:
        if await self._sessions.ensure_child(session_id):
            attached = self._root_session.attached_session_id
            if attached is None or not self._sessions.child_belongs_to(
                session_id, attached
            ):
                raise RequestFailure(
                    ProtocolErrorCode.CONFLICT,
                    "Child session is not linked to the attached session",
                )
            return await self._sessions.reject_callback(session_id, callback_id, error)
        self._require_attached(session_id)
        return await self._turns.reject_callback(callback_id, error)

    async def _history_clear(
        self, params: SessionHistoryClearParams
    ) -> tuple[SessionHistoryClearResponse, Callable[[], None] | None]:
        self._require_session(params.session_id)
        if self._turns.active_turn is not None:
            raise RequestFailure(
                ProtocolErrorCode.CONFLICT,
                "Cannot clear history while a turn is active",
            )
        with self._execution.reserve(
            SessionExecutionKind.LIFECYCLE, f"clear:{params.session_id}"
        ):
            previous_history = self._turns.history
            session_model_pinned = (
                self._agent_loop.session_logger.active_model is not None
            )
            await self._agent_loop.clear_history()
            if session_model_pinned:
                failures = await clear_session_active_model_override(
                    self._agent_loop.config_orchestrator,
                    reason="clear session active model",
                )
                if failures:
                    raise RequestFailure(
                        ProtocolErrorCode.INTERNAL_ERROR,
                        f"Failed to clear session active model: {failures[0]}",
                    )
                await self._agent_loop.reload_with_initial_messages()
            after_response = await self._turns.reset()
            handoff = self._root_session.replace_idle(
                params.session_id,
                current_history=previous_history,
                checkpoint_kind="clear",
                checkpoint_message="New conversation started",
            )
        return (
            SessionHistoryClearResponse(
                state=handoff.state, session_log=handoff.session_log
            ),
            after_response,
        )

    async def _plan_handoff(  # noqa: PLR0914, PLR0915 - one ordered transaction
        self, params: SessionPlanHandoffParams
    ) -> tuple[SessionPlanHandoffResponse, Callable[[], None] | None]:
        """Start the consented executor session for an accepted plan.

        Admission mirrors the clear route: an active turn or a live background
        job refuses the handoff with an actionable message, and the job check
        is re-run inside the reserve, right before the reset retires jobs. The
        consent snapshot in ``params`` is applied exactly — any patch failure
        or a read-back that cannot match the displayed selection is reported
        and execution is not started.

        An episode-bound request runs the FULL transaction. PREPARE (P1-P3) runs
        first; COMMIT (C1-C5) then acquires the episode's transition fence and
        holds it to the end, so the pre-launch re-scan, the executor's durable
        marker, the terminal flip, and the derived rename are one bounded,
        uninterleaved sequence.

        The session's plan record decides the route. A session with no record
        keeps the pre-episode (legacy) request shape and runs none of the
        transaction; a record that names no episode is legacy-unverified and
        never accepts; an episode-bound record admits only the integrity path
        carrying its own episode identity.

        A request that re-enters an ACCEPTING claim held by this session skips
        PREPARE and enters COMMIT directly: the re-offer continuation carries
        the surviving snapshot's bytes, and post-consent the draft is never
        consulted again, so no draft read may stand between the consent and the
        commit (design sec.4.2/4.4).
        """
        self._require_session(params.session_id)
        if self._turns.active_turn is not None:
            raise RequestFailure(
                ProtocolErrorCode.CONFLICT,
                "Cannot start plan execution while a turn is active; "
                "finish or stop the work first",
            )
        with self._execution.reserve(
            SessionExecutionKind.LIFECYCLE, f"plan/handoff:{params.session_id}"
        ):
            # Re-checked at handoff: the reset below retires background jobs.
            try:
                self._agent_loop.require_background_jobs_idle()
            except AgentLoopStateError as exc:
                raise RequestFailure(
                    ProtocolErrorCode.CONFLICT,
                    f"Cannot start plan execution while background work is "
                    f"active: {exc}. Finish or stop the work first.",
                ) from exc
            # The record decides the route before anything is read or claimed.
            body = _handoff_verified_body(params)
            self._plan_handoff_require_route(body)
            # A re-offer continuation re-enters the commit of a claim this
            # session already holds from the surviving snapshot's bytes; the
            # draft is never consulted post-consent, so PREPARE is skipped.
            if body is None or not self._plan_handoff_continuation(params, body):
                # PREPARE (P1-P3): claim the episode, verify the previewed bytes
                # against the draft, and publish the consent snapshot. Each step
                # fences itself; every observed failure returns the episode to
                # draft, and success leaves the claim held for the commit half.
                await self._plan_handoff_prepare(params)
            with ExitStack() as stack:
                commit: CommitAttempt | None = None
                retired: PlanModeRecord | None = None
                if body is not None:
                    # COMMIT's fence is acquired here and held through C5.
                    commit = self._plan_handoff_begin_commit(params, body)
                    stack.callback(commit.close)
                    # C2 runs while params.session_id still names the root: the
                    # reset below rotates it, and the source record becomes
                    # unaddressable on the sanctioned write path.
                    retired = await self._plan_handoff_retire_source(params, commit)
                orchestrator = self._agent_loop.config_orchestrator
                # The executor identity rides as a session-scoped override
                # written BEFORE the fresh root bind, so the reset binds it; the
                # bound bundle freezes it and the pin is removed again after it.
                identity_failures = await set_session_agent_identity_override(
                    orchestrator,
                    params.executor_identity,
                    reason="plan handoff executor identity",
                )
                if identity_failures:
                    raise RequestFailure(
                        ProtocolErrorCode.INTERNAL_ERROR,
                        f"Failed to install plan executor identity: "
                        f"{identity_failures[0]}",
                    )
                previous_history = self._turns.history
                session_model_pinned = (
                    self._agent_loop.session_logger.active_model is not None
                )
                try:
                    await self._agent_loop.clear_history()
                except BaseException as exc:
                    # The reset failed after the identity write: best-effort
                    # removal of the leftover pin, reported honestly.
                    cleanup = await clear_session_agent_identity_override(
                        orchestrator, reason="plan handoff reset failure cleanup"
                    )
                    if not isinstance(exc, Exception):
                        raise
                    detail = (
                        "leftover identity pin removed"
                        if not cleanup
                        else f"leftover identity pin could not be removed: {cleanup[0]}"
                    )
                    raise RequestFailure(
                        ProtocolErrorCode.INTERNAL_ERROR,
                        f"Plan handoff reset failed: {exc} ({detail})",
                    ) from exc
                unpin_failures = await clear_session_agent_identity_override(
                    orchestrator, reason="plan handoff identity unpin after bind"
                )
                if unpin_failures:
                    raise RequestFailure(
                        ProtocolErrorCode.INTERNAL_ERROR,
                        f"Failed to remove plan executor identity pin: "
                        f"{unpin_failures[0]}",
                    )
                if session_model_pinned:
                    failures = await clear_session_active_model_override(
                        orchestrator, reason="plan handoff active model unpin"
                    )
                    if failures:
                        raise RequestFailure(
                            ProtocolErrorCode.INTERNAL_ERROR,
                            f"Failed to clear session active model: {failures[0]}",
                        )
                # Restore the planner-alias thinking key /plan wrote at entry
                # BEFORE the executor install, so the executor install stays the
                # final write on its own alias even when both share it.
                restore_failures = await self._plan_handoff_restore_planner_key(
                    orchestrator, params
                )
                if restore_failures:
                    raise RequestFailure(
                        ProtocolErrorCode.INTERNAL_ERROR,
                        f"Failed to restore planner thinking key: "
                        f"{restore_failures[0]}",
                    )
                # Executor install, pinned between the unpin above and the
                # reload below so the reload refreshes the prompt with
                # post-install values instead of reset-time config.
                model_failures = await set_session_active_model_override(
                    orchestrator,
                    params.executor_model,
                    reason="plan handoff executor model",
                )
                thinking_failures = await set_session_thinking_override(
                    orchestrator,
                    params.executor_model,
                    params.executor_thinking,
                    reason="plan handoff executor thinking",
                )
                install_failures = [*model_failures, *thinking_failures]
                if install_failures:
                    raise RequestFailure(
                        ProtocolErrorCode.INTERNAL_ERROR,
                        f"Failed to install plan executor selection: "
                        f"{install_failures[0]}",
                    )
                # Apply-exactly-or-fail: read the installed session state back
                # and compare with the consented snapshot before starting
                # execution.
                mismatch = self._plan_handoff_selection_mismatch(orchestrator, params)
                if mismatch is not None:
                    raise RequestFailure(
                        ProtocolErrorCode.CONFLICT,
                        "Plan handoff could not apply the consented executor "
                        "selection exactly; execution was not started",
                        mismatch,
                    )
                if commit is not None:
                    # C3: one frozen logging observation, the pre-marker domain
                    # record, the fenced pre-launch re-scan, the durable marker,
                    # then the injection of the carried bytes.
                    assert retired is not None
                    await self._plan_handoff_mark_and_inject(commit, retired)
                # Prompt refresh: reload_with_initial_messages is the only
                # refresh hook, and the clear route reaches it solely inside its
                # logger-gated pinned branch — so for the unpinned case this
                # call is the named refresh step, and it runs after the executor
                # install in both cases.
                await self._agent_loop.reload_with_initial_messages()
                after_response = await self._turns.reset()
                handoff = self._root_session.replace_idle(
                    params.session_id,
                    current_history=previous_history,
                    checkpoint_kind="plan",
                    checkpoint_message="Plan execution started",
                    checkpoint_details=(
                        {"planPath": params.plan_path} if params.plan_path else None
                    ),
                )
                if commit is not None:
                    # C4/C5: the executor session exists, so its id is recorded
                    # first (the executor scan's discriminator against a restored
                    # planning mirror), the terminal flip is the last
                    # arbiter-visible change, and the derived rename follows it.
                    self._plan_handoff_record_executor(commit, handoff.new_session_id)
                    self._plan_handoff_finalize(commit, handoff.new_session_id)
        return (
            SessionPlanHandoffResponse(
                state=handoff.state, session_log=handoff.session_log
            ),
            after_response,
        )

    def _plan_handoff_require_route(self, body: tuple[str, Path, bytes] | None) -> None:
        """Refuse a handoff whose shape contradicts this session's plan record.

        Once ANY episode record exists the integrity path is the only handoff
        path. A record naming no episode is legacy-unverified and never accepts
        (criterion 9: "accept never arms on legacy") — the user re-binds into a
        new episode or cancels. An episode-bound record admits only a request
        carrying that episode's verifiable body. A session with NO record at all
        is the pre-P10 client shape and keeps its old behavior untouched.
        """
        record = self._plan_handoff_record()
        if record is None:
            return
        if record.episode_id is None:
            raise RequestFailure(
                ProtocolErrorCode.CONFLICT,
                "This plan record is legacy-unverified, so it cannot accept: "
                "re-bind the draft into a new plan episode and accept again, or "
                "cancel plan mode",
            )
        if body is None:
            raise RequestFailure(
                ProtocolErrorCode.INVALID_PARAMS,
                f"The plan record for episode {record.episode_id} admits only the "
                "integrity handoff: the request must carry the episode id, the "
                "snapshot path and digest, and the snapshot bytes",
            )
        if body[0] != record.episode_id:
            raise RequestFailure(
                ProtocolErrorCode.INVALID_PARAMS,
                f"The handoff episode {body[0]} is not this session's episode "
                f"{record.episode_id}; execution was not started",
            )

    def _plan_handoff_continuation(
        self, params: SessionPlanHandoffParams, body: tuple[str, Path, bytes]
    ) -> bool:
        """Whether this request RE-ENTERS the commit of a claim already held.

        Row 3's re-offer keeps the claim held and re-enters COMMIT with the
        located surviving snapshot's bytes, so the continuation leg must not run
        PREPARE's draft read at all (post-consent the draft is never consulted).
        Only an ACCEPTING claim held by this same session, already carrying a
        recorded consent digest, qualifies. The digest itself is verified by C1
        against the carried bytes — that is where a wrong body refuses and
        releases the claim — so this predicate decides the route only. Every
        other shape is a fresh accept and prepares normally, where its own
        refusals (already claimed, foreign claim, domain mismatch) surface.
        """
        episode_id, snapshot_path, _content = body
        try:
            sidecar = read_sidecar(snapshot_path.parent)
        except PlanEpisodeError:
            return False
        if sidecar is None:
            return False
        return (
            sidecar.episode_id == episode_id
            and sidecar.state == STATE_ACCEPTING
            and sidecar.claiming_session_id == params.session_id
            and sidecar.consent_sha256 is not None
        )

    def _plan_handoff_record(self) -> PlanModeRecord | None:
        """The root session's own plan record, or ``None`` when it has none."""
        metadata = self._agent_loop.session_logger.session_metadata
        return None if metadata is None else metadata.plan_mode

    def _plan_handoff_logging_domain(self) -> LoggingDomain:
        """The episode's PREPARE-side logging domain (the P1 owner domain)."""
        logger = self._agent_loop.session_logger
        save_dir = logger.save_dir
        session_prefix = logger.session_prefix
        if not logger.enabled or save_dir is None or session_prefix is None:
            raise RequestFailure(ProtocolErrorCode.CONFLICT, LOGGING_REQUIRED_MESSAGE)
        return LoggingDomain(save_dir=str(save_dir), session_prefix=session_prefix)

    def _plan_handoff_begin_commit(
        self, params: SessionPlanHandoffParams, body: tuple[str, Path, bytes]
    ) -> CommitAttempt:
        """COMMIT entry: acquire the fence and run C1, or refuse.

        A mismatch releases the claim (the episode returns to draft and the
        planning session is untouched — no destructive step has run) and surfaces
        as a pre-replacement CONFLICT.
        """
        episode_id, snapshot_path, content = body
        domain = self._plan_handoff_logging_domain()
        try:
            return begin_commit(
                snapshot_path.parent,
                episode_id=episode_id,
                session_id=params.session_id,
                owner_domain=domain,
                carried_bytes=content,
                snapshot_path=snapshot_path,
                retry_timeout=_PLAN_COMMIT_FENCE_RETRY_SECONDS,
            )
        except PlanFenceBusyError as exc:
            raise RequestFailure(
                ProtocolErrorCode.CONFLICT,
                f"{exc} — an accept transaction is in progress; retry after it "
                "completes, or inspect the episode status",
            ) from exc
        except PlanDigestMismatchError as exc:
            detail = self._plan_handoff_release_refused_claim(
                snapshot_path.parent, params.session_id, domain
            )
            raise RequestFailure(ProtocolErrorCode.CONFLICT, f"{exc}{detail}") from exc
        except PlanTransitionError as exc:
            raise RequestFailure(ProtocolErrorCode.CONFLICT, str(exc)) from exc
        except PlanEpisodeError as exc:
            raise RequestFailure(ProtocolErrorCode.INTERNAL_ERROR, str(exc)) from exc

    def _plan_handoff_release_refused_claim(
        self, directory: Path, session_id: str, domain: LoggingDomain
    ) -> str:
        """Release a refused C1 claim; report, never hide, a release failure."""
        try:
            release_claim(directory, session_id=session_id, owner_domain=domain)
        except PlanEpisodeError as exc:
            return (
                f" (the accept claim could not be released: {exc}; inspect the "
                "episode status before retrying)"
            )
        return " The accept claim was released and planning can continue."

    async def _plan_handoff_retire_source(
        self, params: SessionPlanHandoffParams, commit: CommitAttempt
    ) -> PlanModeRecord:
        """C2: retire the planning session's record durably as accepted.

        The retirement is a FORCED, checked durable write: a write that cannot be
        proven durable fails the handoff loudly instead of reporting a clean
        success, and it runs before the reset, while the request's session id
        still names the root.
        """
        if params.session_id != self._agent_loop.session_id:
            raise RequestFailure(
                ProtocolErrorCode.CONFLICT,
                "The plan record belongs to the root session",
            )
        logger = self._agent_loop.session_logger
        metadata = logger.session_metadata
        source = None if metadata is None else metadata.plan_mode
        if source is None:
            raise RequestFailure(
                ProtocolErrorCode.CONFLICT,
                "The planning session records no plan mode to retire",
            )
        retired = source.model_copy(
            update={
                "episode_id": commit.episode_id,
                "state": STATE_ACCEPTED,
                "consent_sha256": commit.sidecar.consent_sha256,
                "snapshot_path": commit.snapshot_path.name,
                "attempt_generation": commit.attempt_generation,
                "scan_domain": _record_scan_domain(commit.sidecar.owner_domain),
            }
        )
        try:
            await logger.persist_plan_mode_durable(retired)
        except (PlanEpisodeWriteError, SessionDiskFullError) as exc:
            raise RequestFailure(
                ProtocolErrorCode.CONFLICT,
                f"Plan handoff could not retire the planning record durably: "
                f"{exc}. The planning session is unchanged, but this accept "
                "attempt is committed — inspect the episode status before "
                "retrying or cancelling.",
            ) from exc
        return retired

    async def _plan_handoff_mark_and_inject(
        self, commit: CommitAttempt, retired: PlanModeRecord
    ) -> None:
        """C3: one logging observation, the domain record, the marker, injection.

        The observation is read ONCE: its domain is recorded in the sidecar and
        its session directory is the marker's destination, so the recorded
        discovery domain can always find the marker it was recorded for. The
        pre-launch re-scan then runs INSIDE the held fence, the marker is the
        executor's first durable act, and only then are the carried bytes
        injected. A marker that cannot be made durable is a loud transition
        failure with row-3 semantics: it is never reported as success and never
        auto-retried.
        """
        logger = self._agent_loop.session_logger
        observation: LoggingObservation = observe_logging_domain(
            save_dir=logger.save_dir,
            session_prefix=logger.session_prefix,
            session_dir=logger.session_dir,
        )
        try:
            commit.record_scan_domain(observation.domain)
            # DR-19's final guard: a marker that appeared after the recovery
            # decision refuses the launch; an untrusted scan fails closed. The
            # source session's own record carries the episode too, so it is
            # excluded — it is the caller, never a launched executor.
            require_no_executor(
                observation.domain,
                commit.episode_id,
                excluding_session_id=commit.sidecar.claiming_session_id,
            )
        except PlanEpisodeError as exc:
            raise RequestFailure(ProtocolErrorCode.INTERNAL_ERROR, str(exc)) from exc
        if logger.session_dir != observation.session_dir:
            raise RequestFailure(
                ProtocolErrorCode.INTERNAL_ERROR,
                "The executor session dir moved away from the observed logging "
                "configuration; the episode marker was not written",
            )
        marker = retired.model_copy(
            update={
                "state": STATE_ACCEPTING,
                "episode_id": commit.episode_id,
                "consent_sha256": commit.sidecar.consent_sha256,
                "snapshot_path": commit.snapshot_path.name,
                "attempt_generation": commit.attempt_generation,
                "scan_domain": _record_scan_domain(observation.domain),
            }
        )
        try:
            await logger.persist_plan_mode_durable(marker)
        except (PlanEpisodeWriteError, SessionDiskFullError) as exc:
            raise RequestFailure(
                ProtocolErrorCode.INTERNAL_ERROR,
                f"Plan handoff could not mark the executor session durably: "
                f"{exc}. The episode was not reported as accepted; inspect its "
                "status before retrying.",
            ) from exc
        try:
            await self._agent_loop.inject_user_context(
                build_handoff_context(
                    content=commit.carried_bytes,
                    episode_id=commit.episode_id,
                    snapshot_path=commit.snapshot_path,
                    consent_sha256=commit.sidecar.consent_sha256 or "",
                ),
                as_message=True,
            )
        except Exception as exc:
            raise RequestFailure(
                ProtocolErrorCode.INTERNAL_ERROR,
                f"Plan handoff could not inject the accepted plan into the "
                f"executor session: {exc}. The executor marker is durable, so "
                "the episode is possibly executed — inspect its status.",
            ) from exc

    def _plan_handoff_record_executor(
        self, commit: CommitAttempt, executor_session_id: str
    ) -> None:
        """C4: record the rotated executor id while the episode is accepting.

        design §4.2 C4: the id is known only after the replacement. Recording it
        before C5 is what lets the executor scan tell the launch apart from a
        planning session that restored the same accepting episode; a write that
        cannot be made durable fails loudly with row-3 semantics — the marker is
        already durable, so the episode is possibly executed.
        """
        try:
            commit.record_executor_session(executor_session_id)
        except PlanEpisodeError as exc:
            raise RequestFailure(
                ProtocolErrorCode.INTERNAL_ERROR,
                f"Plan handoff could not record the executor session: {exc}. The "
                "executor marker is durable, so the episode is possibly executed "
                "— inspect its status.",
            ) from exc

    def _plan_handoff_finalize(
        self, commit: CommitAttempt, executor_session_id: str
    ) -> None:
        """C5: the terminal flip once the executor exists, then the rename."""
        try:
            commit.finalize(executor_session_id=executor_session_id)
        except PlanEpisodeError as exc:
            raise RequestFailure(
                ProtocolErrorCode.INTERNAL_ERROR,
                f"Plan handoff could not record the accepted episode: {exc}",
            ) from exc

    async def _plan_handoff_prepare(self, params: SessionPlanHandoffParams) -> None:
        """Run the PREPARE half (P1-P3) of an episode-bound handoff.

        The request's snapshot bytes are the sole verified artifact. The
        episode is claimed under the fence, the draft's raw bytes are retained
        and byte-compared against the previewed bytes (a change demands a
        re-preview), and the retained bytes are published EXCLUSIVELY with
        their digest recorded in the sidecar. Every observed failure releases
        the claim and returns the episode to draft; success leaves the claim
        held for the commit half.

        A request without the episode fields is the legacy, pre-episode path
        and prepares nothing.

        The consented bytes must also BE a usable plan document. The consent
        modal refuses an empty or summary-less draft too; the server repeats the
        check (design sec.4.2's refusal matrix) so the raw wire route cannot
        consent to a document the matrix forbids. The refusal happens BEFORE the
        claim, so it leaves the episode exactly as it was and names its recovery.
        """
        body = _handoff_verified_body(params)
        if body is None:
            return
        episode_id, snapshot_path, content = body
        refusal = _draft_refusal(content)
        if refusal is not None:
            raise RequestFailure(ProtocolErrorCode.CONFLICT, refusal)
        try:
            prepare_accept(
                snapshot_path.parent,
                episode_id=episode_id,
                session_id=params.session_id,
                owner_domain=self._plan_handoff_logging_domain(),
                previewed_bytes=content,
                previewed_sha256=sha256_bytes(content),
            )
        except PlanTransitionError as exc:
            raise RequestFailure(ProtocolErrorCode.CONFLICT, str(exc)) from exc
        except PlanEpisodeError as exc:
            raise RequestFailure(ProtocolErrorCode.INTERNAL_ERROR, str(exc)) from exc

    async def _plan_handoff_restore_planner_key(
        self,
        orchestrator: ConfigOrchestrator[ChartreuxConfigSchema],
        params: SessionPlanHandoffParams,
    ) -> list[BaseException]:
        """Restore the planner-alias thinking key to its pre-entry value.

        Per-key only: a key absent pre-entry (``planner_thinking_value=None``)
        is removed, never whole-map replaced.
        """
        if not params.planner_thinking_alias:
            return []
        if params.planner_thinking_value is None:
            return await remove_session_thinking_override(
                orchestrator,
                params.planner_thinking_alias,
                reason="plan handoff planner thinking key removal",
            )
        return await set_session_thinking_override(
            orchestrator,
            params.planner_thinking_alias,
            params.planner_thinking_value,
            reason="plan handoff planner thinking key restore",
        )

    def _plan_handoff_selection_mismatch(
        self,
        orchestrator: ConfigOrchestrator[ChartreuxConfigSchema],
        params: SessionPlanHandoffParams,
    ) -> dict[str, Any] | None:
        """Read back the installed selection; None when it matches the consent.

        The bound identity is included: the fresh bind froze the executor
        identity, and the session-scoped identity pin must be gone again.
        """
        applied_model = override_active_model(orchestrator)
        applied_thinking = override_thinking_level(orchestrator, params.executor_model)
        bound_identity = self._agent_loop._bound_identity
        if (
            applied_model == params.executor_model
            and applied_thinking == params.executor_thinking
            and bound_identity == params.executor_identity
            and override_agent_identity(orchestrator) is None
        ):
            return None
        return {
            "appliedModel": applied_model,
            "appliedThinking": applied_thinking,
            "boundIdentity": bound_identity,
        }

    async def _compact(
        self, params: SessionCompactParams
    ) -> tuple[SessionCompactResponse, Callable[[], None] | None]:
        self._require_session(params.session_id)
        if self._turns.active_turn is not None:
            raise RequestFailure(
                ProtocolErrorCode.CONFLICT, "Cannot compact while a turn is active"
            )
        # Compaction runs an LLM summarization, so a recovered session must
        # pick a model first — exactly like a turn start — instead of
        # summarizing on a silently-committed default.
        self._turns.require_model_choice()
        with self._execution.reserve(
            SessionExecutionKind.LIFECYCLE, f"compact:{params.session_id}"
        ):
            previous_history = self._turns.history
            # The manager marks publication on the carrier as it happens, so an
            # interrupted call can still be classified by what it published.
            outcome = self._agent_loop.compaction_outcome
            deadline = asyncio.timeout(_MANUAL_COMPACTION_TIMEOUT_SECONDS)
            summary: str | None = None
            settlement: CompactionSettlementError | None = None
            try:
                async with deadline:
                    summary = await self._agent_loop.compact(
                        params.extra_instructions, manual=True
                    )
            except CompactionSettlementError as exc:
                # The boundary is in the live message list; only its persistence
                # failed. Deadline expiry cannot change what there is to settle.
                settlement = exc
            except CompactionFailedError as exc:
                if not deadline.expired():
                    raise RequestFailure(
                        ProtocolErrorCode.COMPACTION_FAILED,
                        str(exc),
                        {"reason": exc.reason},
                    ) from exc
                # A provider that turned the deadline's own cancellation into a
                # failure of its own is still an expired run: row 1 governs.
                raise self._expired_compaction_failure() from exc
            except asyncio.CancelledError as exc:
                if not deadline.expired() and not outcome.published:
                    # Nothing was published, so no compaction is owed: release
                    # the session view the cancelled request still owns — with
                    # no checkpoint, which would claim a compaction — and let
                    # the cancellation out.
                    after_reset = await self._turns.reset()
                    if after_reset is not None:
                        after_reset()
                    raise
                if deadline.expired() and not outcome.published:
                    # The deadline and an external cancellation both fired, and
                    # the cancellation count keeps the timeout from surfacing as
                    # TimeoutError. Expiry still governs an unpublished run.
                    raise self._expired_compaction_failure() from exc
                # A published boundary whose settlement was itself interrupted:
                # classified below.
            except TimeoutError as exc:
                if not deadline.expired():
                    # A provider timeout inside a live deadline is the
                    # provider's failure, not deadline expiry.
                    raise
                if not outcome.published:
                    raise self._expired_compaction_failure() from exc
                # A published boundary still owes its settlement: classified
                # below, whatever type the interruption arrived as.
            except Exception as exc:
                if not deadline.expired():
                    raise
                # Deadline expiry governs the outcome whatever type the failure
                # arrives as (row 9): a published boundary is still settled.
                if not outcome.published:
                    raise self._expired_compaction_failure() from exc
                raise await self._published_compaction_failure(
                    params, previous_history, None
                ) from exc
            if summary is None:
                raise await self._published_compaction_failure(
                    params, previous_history, settlement
                ) from settlement
            handoff, after_response = await self._hand_off_compacted_session(
                params, previous_history, summary
            )
        return (
            SessionCompactResponse(
                summary=summary, state=handoff.state, session_log=handoff.session_log
            ),
            after_response,
        )

    @staticmethod
    def _expired_compaction_failure() -> RequestFailure:
        """Classify a deadline-expired manual compaction that published nothing."""
        return RequestFailure(
            ProtocolErrorCode.COMPACTION_FAILED,
            "Manual compaction timed out before it published a summary",
            {"reason": "timeout"},
        )

    async def _hand_off_compacted_session(
        self,
        params: SessionCompactParams,
        previous_history: list[PublicHistoryEntry],
        summary: str,
    ) -> tuple[SessionHandoff, Callable[[], None] | None]:
        """Reset and hand off the session a published compaction leaves behind.

        The boundary is already live, so a failure here is a persistence
        failure — the client reconciles instead of retrying a summarization
        that already happened — and never a raw exception.
        """
        reset_settled = False
        after_response: Callable[[], None] | None = None
        try:
            after_response = await self._turns.reset()
            reset_settled = True
            handoff = self._root_session.replace_idle(
                params.session_id,
                current_history=previous_history,
                checkpoint_kind="compaction",
                checkpoint_message="Context compacted",
                checkpoint_details={"summaryLength": len(summary)},
            )
        except asyncio.CancelledError:
            # The boundary is already live, so a cancelled hand-off still owes
            # its settlement: the reset and the idle replacement publish the
            # state a client reconciles with, and only then does the
            # cancellation go on. It is never turned into a response.
            if after_response is not None:
                with suppress(Exception):
                    after_response()
            # A settlement fault of its own must not replace the cancellation.
            with suppress(Exception):
                await self._settle_published_compaction(
                    params,
                    previous_history,
                    summary_length=len(summary),
                    reset_settled=reset_settled,
                )
            raise
        except Exception as exc:
            if after_response is not None:
                # The typed error carries no deferred callback, so the queue
                # reset is published here — without letting it mask the error
                # that brought us to it.
                with suppress(Exception):
                    after_response()
            raise await self._published_compaction_failure(
                params,
                previous_history,
                None,
                summary_length=len(summary),
                reset_settled=reset_settled,
            ) from exc
        return handoff, after_response

    async def _settle_published_compaction(
        self,
        params: SessionCompactParams,
        previous_history: list[PublicHistoryEntry],
        *,
        summary_length: int | None = None,
        reset_settled: bool = False,
    ) -> dict[str, Any]:
        """Publish the state a live compaction boundary still owes a client.

        The boundary is in the live message list, so the reset and the idle
        replacement run whatever interrupted the hand-off. Failures are logged,
        not raised: the caller already owns the error it is reporting.
        """
        if not reset_settled:
            try:
                after_reset = await self._turns.reset()
            except Exception as exc:
                # The replacement below still runs, and the state it publishes
                # is what a client reconciles with: a reset failure must not
                # replace the typed error with a raw exception.
                logger.warning(
                    "Published compaction could not reset the turn state", exc_info=exc
                )
            else:
                if after_reset is not None:
                    # The error response carries no deferred callback, and the
                    # replacement below can still fail, so run it before that —
                    # without letting it mask the error that brought us here.
                    try:
                        after_reset()
                    except Exception as exc:
                        logger.warning(
                            "Published compaction could not run the deferred reset",
                            exc_info=exc,
                        )
        data: dict[str, Any] = {"reason": "persistence"}
        try:
            handoff = self._root_session.replace_idle(
                params.session_id,
                current_history=previous_history,
                checkpoint_kind="compaction",
                checkpoint_message="Context compacted",
                checkpoint_details=(
                    {"summaryLength": summary_length}
                    if summary_length is not None
                    else None
                ),
            )
        except Exception as exc:
            # Without a settled replacement there is no authoritative state to
            # hand over; the client has to reconcile later.
            logger.warning(
                "Published compaction could not replace the idle session", exc_info=exc
            )
        else:
            data["state"] = handoff.state.model_dump(mode="json", by_alias=True)
        return data

    async def _published_compaction_failure(
        self,
        params: SessionCompactParams,
        previous_history: list[PublicHistoryEntry],
        cause: CompactionSettlementError | None,
        *,
        summary_length: int | None = None,
        reset_settled: bool = False,
    ) -> RequestFailure:
        """Settle what a published-but-unsettled manual compaction still owes.

        The boundary is already live, so the reset and the idle replacement run
        regardless of the failure: they publish the state a client needs to
        reconcile. The failure is typed as persistence rather than compaction
        so a client does not retry a summarization that already happened.
        """
        if summary_length is None and cause is not None:
            summary_length = cause.summary_length
        data = await self._settle_published_compaction(
            params,
            previous_history,
            summary_length=summary_length,
            reset_settled=reset_settled,
        )
        detail = (
            str(cause)
            if cause is not None
            else "The published compaction was interrupted before it settled."
        )
        return RequestFailure(
            ProtocolErrorCode.COMPACTION_FAILED,
            f"{detail} The conversation is compacted; reconcile the session "
            "state before retrying.",
            data,
        )

    async def _relocate(  # noqa: PLR0912, PLR0915
        self, params: SessionRelocateParams
    ) -> SessionRelocateResponse:
        self._require_attached(params.session_id)
        # The reserve is stricter than the loop's own guards, and it is what
        # makes every busy case a conflict rather than a bad request: a turn, a
        # teleport and another lifecycle transition all hold the same slot. What
        # reaches the loop is therefore only ever a target it rejects on merit.
        with self._execution.reserve(  # noqa: PLR1702
            SessionExecutionKind.LIFECYCLE, "relocate"
        ):
            previous_cwd = self._agent_loop.cwd
            session_id = self._agent_loop.session_id
            source_token = self.worktree_token
            # Off the loop, and in one hop. Both halves are filesystem reads --
            # `resolve` walks symlinks, and locating a claim resolves twice more
            # -- so a relocate onto a slow mount would otherwise stall every
            # other session while this one worked out where it was going.
            target, changes_worktree = await asyncio.to_thread(
                _relocation_target, params.cwd, previous_cwd
            )
            if target != previous_cwd:
                try:
                    self._agent_loop.require_background_jobs_idle()
                except AgentLoopStateError as exc:
                    raise RequestFailure(ProtocolErrorCode.CONFLICT, str(exc)) from exc
            transfer: TransferAttempt | None = None
            acquired_token: OwnershipToken | None = None
            if changes_worktree:
                if (
                    source_token is not None
                    and SessionWorktrees.root(target) is not None
                ):
                    begin = asyncio.create_task(
                        asyncio.to_thread(begin_transfer, source_token, target)
                    )
                    try:
                        transfer = await asyncio.shield(begin)
                    except asyncio.CancelledError:
                        # A thread already acquiring the target cannot be stopped.
                        # Drain it and return that acquisition before propagating.
                        while not begin.done():
                            with suppress(asyncio.CancelledError):
                                await asyncio.shield(begin)
                        transfer = begin.result()
                        abort = asyncio.create_task(
                            asyncio.to_thread(abort_transfer, transfer)
                        )
                        while not abort.done():
                            with suppress(asyncio.CancelledError):
                                await asyncio.shield(abort)
                        abort.result()
                        raise
                else:
                    acquire = asyncio.create_task(
                        asyncio.to_thread(SessionWorktrees.hold, target, session_id)
                    )
                    try:
                        acquired_token = await asyncio.shield(acquire)
                    except asyncio.CancelledError:
                        while not acquire.done():
                            with suppress(asyncio.CancelledError):
                                await asyncio.shield(acquire)
                        acquired_token = acquire.result()
                        if acquired_token is not None:
                            release = asyncio.create_task(
                                asyncio.to_thread(release_holder, acquired_token)
                            )
                            while not release.done():
                                with suppress(asyncio.CancelledError):
                                    await asyncio.shield(release)
                            release.result()
                        raise
            try:
                await self._agent_loop.relocate(target)
            except BaseException as exc:
                if transfer is not None:
                    abort = asyncio.create_task(
                        asyncio.to_thread(abort_transfer, transfer)
                    )
                    try:
                        await asyncio.shield(abort)
                    except asyncio.CancelledError:
                        while not abort.done():
                            with suppress(asyncio.CancelledError):
                                await asyncio.shield(abort)
                        abort.result()
                elif acquired_token is not None:
                    release = asyncio.create_task(
                        asyncio.to_thread(release_holder, acquired_token)
                    )
                    try:
                        await asyncio.shield(release)
                    except asyncio.CancelledError:
                        while not release.done():
                            with suppress(asyncio.CancelledError):
                                await asyncio.shield(release)
                        release.result()
                if isinstance(exc, AgentLoopStateError):
                    raise RequestFailure(
                        ProtocolErrorCode.INVALID_PARAMS, str(exc)
                    ) from exc
                raise
            destination = self._agent_loop.cwd
            if destination == previous_cwd:
                return SessionRelocateResponse(
                    state=self._public_state(DEFAULT_HISTORY_LIMIT)
                )
            # The checkpoint folds the turns into the stored history and hands
            # back a state carrying none of its own, so the controller has to
            # let go of them too. Left in place they are read a second time
            # beside the copy now in history, with the relocation mark between
            # the two. Clear and compact do the same thing for the same reason.
            if changes_worktree:
                if transfer is not None:
                    commit = asyncio.create_task(
                        asyncio.to_thread(commit_transfer, transfer)
                    )
                    try:
                        self.worktree_token = await asyncio.shield(commit)
                    except asyncio.CancelledError:
                        while not commit.done():
                            with suppress(asyncio.CancelledError):
                                await asyncio.shield(commit)
                        self.worktree_token = commit.result()
                        raise
                else:
                    if source_token is not None:
                        release = asyncio.create_task(
                            asyncio.to_thread(release_holder, source_token)
                        )
                        try:
                            await asyncio.shield(release)
                        except asyncio.CancelledError:
                            while not release.done():
                                with suppress(asyncio.CancelledError):
                                    await asyncio.shield(release)
                            release.result()
                            self.worktree_token = acquired_token
                            raise
                    self.worktree_token = acquired_token
            previous_history = self._turns.history
            await self._turns.reset()
            state = self._root_session.append_checkpoint(
                current_history=previous_history,
                kind="relocation",
                message=f"Moved to {destination}",
                details={"cwd": str(destination), "previousCwd": str(previous_cwd)},
            )
        return SessionRelocateResponse(state=state)

    def _public_state(
        self,
        history_limit: int,
        *,
        turns_limit: int | None = None,
        include_history: bool = True,
        include_turns: bool = True,
    ) -> PublicSessionState:
        callbacks = [
            entry
            for entry in self._all_history()
            if isinstance(entry, PublicCallbackEntry)
        ]
        return self._root_session.public_state(
            current_history=self._turns.history,
            callbacks=callbacks,
            turns=self._turns.turns,
            retrying=self._turns.retrying,
            history_limit=history_limit,
            turns_limit=turns_limit,
            include_history=include_history,
            include_turns=include_turns,
            turn_queue=self._turns.queue_state,
        )

    def _all_history(self) -> list[PublicHistoryEntry]:
        return self._root_session.all_history(self._turns.history)

    def _require_session(self, session_id: str) -> None:
        if not self._root_session.is_current(session_id):
            raise RequestFailure(
                ProtocolErrorCode.NOT_FOUND, f"Session not found: {session_id}"
            )

    def _require_attached(self, session_id: str) -> None:
        self._require_session(session_id)
        if not self._root_session.is_attached(session_id):
            raise RequestFailure(ProtocolErrorCode.CONFLICT, "Session is not attached")

    def _require_turn_route(self, session_id: str, turn_id: str) -> None:
        active_turn = self._turns.active_turn
        if active_turn is None:
            raise RequestFailure(ProtocolErrorCode.CONFLICT, "No active turn")
        if active_turn.id != turn_id:
            raise StaleTurnError(active_turn.id)
        if not self._root_session.routes_active_turn(session_id, active_turn):
            raise RequestFailure(
                ProtocolErrorCode.NOT_FOUND, f"Session not found: {session_id}"
            )


def _record_scan_domain(domain: LoggingDomain | None) -> PlanScanDomain | None:
    """The persisted record's scan-domain mirror of a logging domain."""
    if domain is None:
        return None
    return PlanScanDomain(
        save_dir=domain.save_dir, session_prefix=domain.session_prefix
    )


def _handoff_verified_body(
    params: SessionPlanHandoffParams,
) -> tuple[str, Path, bytes] | None:
    """Decode and validate an episode-bound handoff's consent body.

    The episode identity, snapshot path, digest, and bytes travel together or
    not at all. The bytes are the authority: their digest must equal the
    carried digest, and the snapshot path must be one of the episode's own names
    in the episode's own directory — the path locates the workspace, it never
    supplies the content. ``None`` means the legacy, pre-episode request.
    """
    episode_id = params.episode_id
    snapshot_path = params.snapshot_path
    snapshot_sha256 = params.snapshot_sha256
    encoded = params.snapshot_bytes_b64
    if (
        episode_id is None
        and snapshot_path is None
        and snapshot_sha256 is None
        and encoded is None
    ):
        return None
    if (
        episode_id is None
        or snapshot_path is None
        or snapshot_sha256 is None
        or encoded is None
    ):
        raise RequestFailure(
            ProtocolErrorCode.INVALID_PARAMS,
            "An episode handoff carries its episode id, snapshot path, "
            "snapshot digest, and snapshot bytes together",
        )
    try:
        content = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise RequestFailure(
            ProtocolErrorCode.INVALID_PARAMS,
            "The plan handoff snapshot bytes are not valid base64",
        ) from exc
    if sha256_bytes(content) != snapshot_sha256:
        raise RequestFailure(
            ProtocolErrorCode.INVALID_PARAMS,
            "The plan handoff snapshot digest does not match its bytes",
        )
    path = Path(snapshot_path)
    if not _is_episode_snapshot_name(path.name, episode_id):
        raise RequestFailure(
            ProtocolErrorCode.INVALID_PARAMS,
            f"The plan handoff snapshot path must be the episode's "
            f"accepted-{episode_id}.md, or its surviving "
            f"provenance-{episode_id}-<generation>.md",
        )
    return episode_id, path, content


def _is_episode_snapshot_name(name: str, episode_id: str) -> bool:
    """True for either of the episode's own snapshot names.

    The live name is the pre-completion carrier; the attempt's provenance name
    is what the re-offer locates after a crash in the cancel window. Both are
    legitimate locators — the carried bytes and the sidecar's digest decide the
    content, never the name.
    """
    if name == f"accepted-{episode_id}.md":
        return True
    prefix = f"provenance-{episode_id}-"
    if not name.startswith(prefix) or not name.endswith(".md"):
        return False
    generation = name[len(prefix) : len(name) - len(".md")]
    return generation.isdigit() and int(generation) >= 1


def _draft_refusal(content: bytes) -> str | None:
    """The refusal for consented bytes that are not a usable plan document.

    The design's refusal matrix (:234) refuses a bound draft that is empty or
    summary-less, for the same reason ``read_plan_document`` does: the executor
    would be seeded with a document that has nothing to execute. The message
    names the recovery the matrix names — bind another document and re-consent,
    or cancel.
    """
    recovery = (
        " Bind another document (re-bind the draft and re-consent) or cancel plan mode."
    )
    if not content.strip():
        return f"The plan document is empty; execution was not started.{recovery}"
    summary = extract_plan_summary(decode_safe(content).text)
    if summary is None:
        return (
            "The plan document has no ## Summary section; execution was not "
            f"started.{recovery}"
        )
    if not summary:
        return (
            "The plan document's ## Summary section is empty; execution was not "
            f"started.{recovery}"
        )
    return None


def _read_authorized_agent_transcript(
    snapshot: TranscriptReadSnapshot, before: str | None, limit: int
) -> AgentTranscriptGetResponse:
    """Load transcript files relative to opened, non-symlink directories."""
    assert snapshot.child_dir is not None
    assert snapshot.parent_dir is not None
    try:
        relative_child = snapshot.child_dir.relative_to(snapshot.parent_dir)
    except ValueError as exc:
        raise SessionFileContainmentError(
            "Child session is outside its parent"
        ) from exc
    try:
        with _opened_directory(snapshot.parent_dir) as parent_fd:
            with _opened_descendant_directory(relative_child, parent_fd) as child_fd:
                return read_agent_transcript(
                    snapshot.child_dir,
                    lambda: True,
                    before=before,
                    limit=limit,
                    session_dir_fd=child_fd,
                )
    except FileNotFoundError:
        return AgentTranscriptGetResponse(
            state=AgentTranscriptState.NO_SAVED_TRANSCRIPT
        )


@contextmanager
def _opened_descendant_directory(path: Path, parent_fd: int) -> Iterator[int]:
    """Open a relative directory one stable, non-symlink component at a time."""
    parts = path.parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise SessionFileContainmentError(f"Invalid child session path: {path}")

    with ExitStack() as stack:
        current_fd = parent_fd
        for part in parts:
            # Keep every ancestor descriptor open until the final descriptor is
            # acquired so a rename cannot redirect subsequent traversal.
            current_fd = stack.enter_context(
                _opened_directory(Path(part), dir_fd=current_fd)
            )
        yield current_fd


@contextmanager
def _opened_directory(path: Path, *, dir_fd: int | None = None) -> Iterator[int]:
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    target: str | Path = path.name if dir_fd is not None else path
    before = None
    if not nofollow:
        before = os.stat(target, dir_fd=dir_fd, follow_symlinks=False)
        if not stat.S_ISDIR(before.st_mode):
            raise SessionFileContainmentError(
                f"Session directory is not a regular directory: {path.name}"
            )
    try:
        fd = os.open(target, os.O_RDONLY | os.O_DIRECTORY | nofollow, dir_fd=dir_fd)
    except OSError as exc:
        if nofollow and exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise SessionFileContainmentError(
                f"Session directory is a symbolic link or not a directory: {path.name}"
            ) from exc
        raise
    try:
        opened = os.fstat(fd)
        if not stat.S_ISDIR(opened.st_mode):
            raise SessionFileContainmentError(
                f"Session directory is not a regular directory: {path.name}"
            )
        if before is not None and (before.st_dev, before.st_ino) != (
            opened.st_dev,
            opened.st_ino,
        ):
            raise SessionFileContainmentError(
                f"Session directory changed while opening: {path.name}"
            )
        yield fd
    finally:
        os.close(fd)


# Where the move is going, and whether it changes which worktree is held.
#
# One function because both answers come from the same filesystem reads, and
# the caller wants them together on one trip off the event loop.
#
# A move that stays inside one managed worktree -- into a subdirectory of it,
# or to a sibling path the loop then refuses -- resolves to the claim the
# session already holds. Taking that hold is a no-op, so giving it back would
# drop the one the session is still standing on and leave the checkout readable
# as idle for another session to delete.
def _relocation_target(requested: str, previous_cwd: Path) -> tuple[Path, bool]:
    # Expanded here rather than passed through raw, because the loop expands
    # before it moves: a `~` target would relocate the session and leave the
    # holder on a path that never existed.
    target = Path(requested).expanduser().resolve()
    changes_worktree = SessionWorktrees.root(target) != SessionWorktrees.root(
        previous_cwd
    )
    return target, changes_worktree
