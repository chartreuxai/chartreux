from __future__ import annotations

import asyncio
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
from chartreux.app_server._projection import (
    history_user_message_index,
    project_history,
    project_message_history,
    project_session_log,
)
from chartreux.app_server._resources import ResourceRequestHandler
from chartreux.app_server._review import ReviewRequestHandler
from chartreux.app_server._root_session import RootSessionCoordinator
from chartreux.app_server._runtime import (
    AgentRuntimeFactory,
    RuntimeSessionNotFoundError,
    close_agent_loop,
)
from chartreux.app_server._session_model import clear_session_active_model_override
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
    PublicSessionState,
)
from chartreux.app_server.protocol import (
    AgentTranscriptGetParams,
    AgentTranscriptGetResponse,
    AgentTranscriptState,
    CallbackRespondParams,
    CallbackRespondResponse,
    CallbackResultError,
    CallbackResultParams,
    CallbackResultResponse,
    ContextInjectParams,
    EmptyResponse,
    ProtocolErrorCode,
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
from chartreux.core.git.worktree import (
    TransferAttempt,
    abort_transfer,
    begin_transfer,
    commit_transfer,
)
from chartreux.core.git.worktree.record import OwnershipToken, release_holder
from chartreux.core.llm_models import LLMMessage
from chartreux.core.session.image_snapshot import ImageSnapshotError
from chartreux.core.session.session_loader import (
    SessionFileContainmentError,
    SessionLoader,
)
from chartreux.core.session_types import (
    ScheduledLoop as CoreScheduledLoop,
    SessionMetadata,
)
from chartreux.core.subagents import UnknownAgentError
from chartreux.observability.logging import logger

DEFAULT_HISTORY_LIMIT = 200

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
        try:
            active = self._execution.active
            if (
                active is not None
                and active.kind is SessionExecutionKind.LIFECYCLE
                and method != "turn/interrupt"
            ):
                raise SessionExecutionConflict(
                    f"Session lifecycle transition is active: {active.id}"
                )
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
        self, method: str, raw_params: dict[str, Any]
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
                steer_params = validate_wire(TurnSteerParams, raw_params)
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
        raise method_not_found(method)

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
                SessionResumeResponse(state=state, last_event_id=state.event_id),
                after_response=_spawn,
                session_attached=True,
                runtime_updated=True,
            )
        self._root_session.attach(params.session_id)
        return DispatchResult(
            SessionResumeResponse(
                state=(state := self._public_state(params.history_limit)),
                last_event_id=state.event_id,
            ),
            session_attached=True,
            runtime_updated=True,
        )

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
                index, restore_files=params.restore_files, inplace=params.inplace
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
            try:
                summary = await self._agent_loop.compact(params.extra_instructions)
            except CompactionFailedError as exc:
                raise RequestFailure(
                    ProtocolErrorCode.COMPACTION_FAILED,
                    str(exc),
                    {"reason": exc.reason},
                ) from exc
            after_response = await self._turns.reset()
            handoff = self._root_session.replace_idle(
                params.session_id,
                current_history=previous_history,
                checkpoint_kind="compaction",
                checkpoint_message="Context compacted",
                checkpoint_details={"summaryLength": len(summary)},
            )
        return (
            SessionCompactResponse(
                summary=summary, state=handoff.state, session_log=handoff.session_log
            ),
            after_response,
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
