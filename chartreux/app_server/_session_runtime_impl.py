from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from chartreux.app_server._dispatch import RequestFailure
from chartreux.app_server._execution import (
    SessionExecution,
    SessionExecutionConflict,
    SessionExecutionKind,
    cancel_tasks,
)
from chartreux.app_server._handler import (
    CoreRequestHandler,
    ResumeOrchestration,
    RootLifecycle,
)
from chartreux.app_server._host import HostRequestHandler
from chartreux.app_server._projection import project_history
from chartreux.app_server._resources import ResourceRequestHandler
from chartreux.app_server._root_session import RootSessionCoordinator
from chartreux.app_server._runtime import (
    AgentRuntimeFactory,
    RootOpenRequest,
    RuntimeAuthenticationError,
    RuntimeConfigurationError,
    RuntimeSessionNotFoundError,
    close_agent_loop,
)
from chartreux.app_server._session_backend_impl import (
    SessionBackendHostImpl,
    SessionBackendImpl,
)
from chartreux.app_server._session_backend_port import ResolvedMCPCatalog
from chartreux.app_server._session_backend_services import SessionBackendServices
from chartreux.app_server._session_history import SessionHistory
from chartreux.app_server._sessions import SessionRuntime, SessionRuntimeRegistry
from chartreux.app_server._tool_io import ClientToolIO
from chartreux.app_server._turns import (
    ModelChoicePendingError,
    TurnConflictError,
    TurnController,
)
from chartreux.app_server._utils import now_ms, public_error
from chartreux.app_server._worktree_effects import WorktreeEffect
from chartreux.app_server._worktree_session import SessionWorktrees, WorktreeResolution
from chartreux.app_server.mcp_catalog import MCPCatalogService
from chartreux.app_server.models import (
    JsonPatchOperation,
    PublicCallbackEntry,
    PublicEntryGenerationStatus,
    PublicNoticeEntry,
    PublicSessionState,
    SessionTitleUpdatedNoticeDetail,
    TextContentBlock,
)
from chartreux.app_server.protocol import (
    AgentConfig,
    AgentEvictionModel,
    AgentSummaryModel,
    AgentsUpdateParams,
    HistoryEntryAddedParams,
    ProtocolErrorCode,
    ServerErrorParams,
    SessionContinueParams,
    SessionOpenParams,
    SessionResumeParams,
    SessionStartParams,
    SessionStartResponse,
    SessionUpdatedParams,
    TurnStartParams,
)
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.config import ChartreuxConfigSchema
from chartreux.core.events import BackgroundWorkEvent, SessionTitleUpdatedEvent
from chartreux.core.git.errors import GitError
from chartreux.core.git.worktree import PreparedWorktree, acquire_for_attachment
from chartreux.core.git.worktree.record import AcquireResult, release_holder
from chartreux.core.session import last_session_pointer
from chartreux.core.session.resume_sessions import resume_directories
from chartreux.core.session.session_lease import SessionBusyError
from chartreux.core.session.worktrees import ResumableDirectories
from chartreux.core.session_types import (
    ScheduledLoop as CoreScheduledLoop,
    WorktreeContext,
)
from chartreux.core.subagents import AgentEviction
from chartreux.observability.logging import logger

type OpenRoot = Callable[[RootOpenRequest], Awaitable[AgentLoop]]
type StageRoot = Callable[[AgentLoop], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class OpenedRuntime:
    agent_loop: AgentLoop
    worktree_resolution: WorktreeResolution


class SessionRuntimeControllerImpl:
    def __init__(
        self,
        *,
        open_root: OpenRoot,
        runtime_factory: AgentRuntimeFactory,
        host_handler: HostRequestHandler,
        stage_root: StageRoot | None,
        services: SessionBackendServices,
        mcp_catalog_service: MCPCatalogService | None = None,
    ) -> None:
        self._open_root = open_root
        self._runtime_factory = runtime_factory
        self._host_handler = host_handler
        self._stage_root = stage_root
        self._services = services
        self._mcp_catalog_service = mcp_catalog_service
        self._scheduler_enabled = False
        self._scheduler_task: asyncio.Task[None] | None = None
        self._title_drain_task: asyncio.Task[None] | None = None
        # The agent loop whose out-of-band queue the drain is bound to, so a
        # restart on the same loop doesn't add a second consumer to one queue.
        self._title_drain_loop: AgentLoop | None = None
        self._resume_tasks: set[asyncio.Task[None]] = set()
        self._tasks: set[asyncio.Task[None]] = set()
        self._worktrees = SessionWorktrees()
        self._root: SessionBackendImpl | None = None
        self._tool_io = ClientToolIO(services)
        self._sessions = SessionRuntimeRegistry(
            services.record_child_notification,
            services.publish_callback,
            services.event_watermark,
            self._tool_io,
            runtime_factory,
            self._notify_agents,
        )

    async def _notify_agents(
        self, agents: list[AgentSummaryModel], evictions: list[AgentEviction]
    ) -> None:
        root = self._root
        if root is None:
            return
        try:
            session_id = self._services.current_session_id()
        except RuntimeError:
            return
        # Root replacement binds the runtime before the server activates its
        # replacement backend. Do not send an update for the new registry over
        # the old session's event stream.
        if session_id != root.session_id:
            return
        await self._services.notify(
            "agents/update",
            AgentsUpdateParams(
                event_id=0,
                session_id=session_id,
                emitted_at=now_ms(),
                agents=agents,
                evictions=[
                    AgentEvictionModel(
                        agent_id=item.agent_id,
                        run_id=item.run_id,
                        reason=item.reason,
                        idle_duration_seconds=item.idle_duration_seconds,
                        root_generation=item.root_generation,
                    )
                    for item in evictions
                ],
            ),
        )

    def create_host(self) -> SessionBackendHostImpl:
        return SessionBackendHostImpl(
            start=self.start,
            resume=self.resume,
            continue_latest=self.continue_latest,
            current_session=self.current_session,
            host_handler=self._host_handler,
            stop_background_tasks=self.stop_background_tasks,
            after_lifecycle_response=self._after_lifecycle_response,
        )

    def current_session(self) -> SessionBackendImpl | None:
        return self._root

    async def start(self, params: SessionStartParams) -> SessionBackendImpl:
        self._scheduler_enabled = not params.agent_config.headless
        if root := self._root:
            await root.handler.dispatch(
                "session/start", params.model_dump(mode="json", by_alias=True)
            )
            return root
        opened = await self._open_runtime(params, None)
        await self._attach_opened_runtime(opened, params.history_limit)
        return self._require_root()

    async def resume(
        self, params: SessionResumeParams
    ) -> tuple[SessionBackendImpl, Callable[[], None] | None]:
        self._scheduler_enabled = not params.agent_config.headless
        self._worktrees.reject_input(params.agent_config)
        if root := self._root:
            result = await root.handler.dispatch(
                "session/resume", params.model_dump(mode="json", by_alias=True)
            )
            return root, result.after_response
        try:
            opened = await self._open_runtime(params, params.session_id)
        except RuntimeSessionNotFoundError as exc:
            raise RequestFailure(
                ProtocolErrorCode.NOT_FOUND, f"Session not found: {params.session_id}"
            ) from exc
        await self._attach_opened_runtime(opened, params.history_limit, resumed=True)
        return self._require_root(), None

    async def continue_latest(
        self, params: SessionContinueParams
    ) -> SessionBackendImpl:
        self._scheduler_enabled = not params.agent_config.headless
        self._worktrees.reject_input(params.agent_config)
        if root := self._root:
            await root.handler.dispatch(
                "session/continue", params.model_dump(mode="json", by_alias=True)
            )
            return self._require_root()
        try:
            opened = await self._open_runtime(params, None, continue_latest=True)
        except RuntimeSessionNotFoundError as exc:
            raise RequestFailure(ProtocolErrorCode.NOT_FOUND, str(exc)) from exc
        await self._attach_opened_runtime(opened, params.history_limit, resumed=True)
        return self._require_root()

    async def stop_background_tasks(self, current: object) -> list[BaseException]:
        tasks = [task for task in self._tasks if task is not current]
        self._tasks.clear()
        scheduler = self._scheduler_task
        self._scheduler_task = None
        if scheduler is not None and scheduler is not current:
            tasks.append(scheduler)
        drain = self._title_drain_task
        self._title_drain_task = None
        self._title_drain_loop = None
        if drain is not None and drain is not current:
            tasks.append(drain)
        self._resume_tasks.clear()
        return await cancel_tasks(tasks, label="legacy session runtime")

    def _after_lifecycle_response(self) -> None:
        if self._scheduler_enabled:
            self._ensure_scheduler()
        self._restart_title_drain()

    def _require_root(self) -> SessionBackendImpl:
        if self._root is None:
            raise RuntimeError("The active backend has no legacy runtime capability")
        return self._root

    @property
    def _agent_loop(self) -> AgentLoop:
        return self._require_root().session.agent_loop

    @property
    def _resources(self) -> ResourceRequestHandler:
        return self._require_root().resources

    @property
    def _root_session(self) -> RootSessionCoordinator:
        return self._require_root().coordinator

    @property
    def _turns(self) -> TurnController:
        return self._require_root().session.turns

    @property
    def _handler(self) -> CoreRequestHandler:
        return self._require_root().handler

    def _track_task(self, task: asyncio.Task[None]) -> None:
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        task.add_done_callback(self._services.task_finished)

    def _spawn_resume_task(self, task: asyncio.Task[None]) -> None:
        self._resume_tasks.add(task)
        task.add_done_callback(self._resume_tasks.discard)
        self._track_task(task)

    def _bind_root(
        self,
        agent_loop: AgentLoop,
        mcp_catalog: ResolvedMCPCatalog | None,
        *,
        initial_loops: list[CoreScheduledLoop] | None = None,
    ) -> None:
        execution = SessionExecution()
        history = SessionHistory(project_history(agent_loop))
        resources = ResourceRequestHandler(
            agent_loop,
            execution,
            self._services.notify,
            current_event_id=self._services.event_watermark,
            initial_loops=initial_loops,
            reserve_config=self._sessions.reserve_config,
        )
        coordinator = RootSessionCoordinator(
            agent_loop,
            resources,
            self._sessions,
            self._services.event_watermark,
            history,
        )
        turns: TurnController | None = None

        def snapshot_state() -> PublicSessionState:
            if turns is None:
                raise RuntimeError("Turn controller is not bound")
            callbacks = [
                entry
                for entry in coordinator.all_history(turns.history)
                if isinstance(entry, PublicCallbackEntry)
            ]
            return coordinator.public_state(
                current_history=turns.history,
                callbacks=callbacks,
                turns=turns.turns,
                retrying=turns.retrying,
                history_limit=200,
            )

        turns = TurnController(
            agent_loop,
            self._services.notify,
            self._services.publish_callback,
            execution,
            self._sessions,
            snapshot_state=snapshot_state,
            tool_io=self._tool_io,
            background_work_sink=self._handle_background_work,
            session_coordinator=coordinator,
        )
        session = SessionRuntime(agent_loop, turns, execution, history)
        handler = CoreRequestHandler(
            agent_loop,
            turns,
            execution,
            self._services.notify,
            self._sessions,
            resources,
            coordinator,
            RootLifecycle(
                replace=self._replace_root,
                adopt=self._adopt_root,
                stage=self._stage_root,
            ),
            ResumeOrchestration(
                runtime_factory=self._runtime_factory,
                current_event_id=self._services.event_watermark,
                spawn_resume_task=self._spawn_resume_task,
            ),
        )
        self._sessions.bind_root(session)
        self._root = SessionBackendImpl(
            session,
            resources,
            coordinator,
            handler,
            self._sessions,
            last_session_pointer.record,
            mcp_catalog=mcp_catalog,
            mcp_authorization_provider=(
                self._mcp_catalog_service.authentication
                if self._mcp_catalog_service is not None
                else None
            ),
            mcp_descriptor_cache_root=(
                Path(agent_loop.config.session_logging.save_dir)
                .expanduser()
                .resolve()
                .parent
                / "mcp-descriptors"
                # Preserve this descriptor-cache path component for compatibility.
                / "legacy"
            ),
        )

    async def _replace_root(
        self, session_id: str, history_limit: int
    ) -> PublicSessionState:
        previous = self._require_root()
        with previous.session.execution.reserve(
            SessionExecutionKind.LIFECYCLE, f"resume:{session_id}"
        ):
            async with self._services.lifecycle_transition():
                for task in list(self._resume_tasks):
                    task.cancel()
                self._resume_tasks.clear()
                agent_loop = previous.session.agent_loop
                try:
                    await self._sessions.drain_children()
                except Exception:
                    logger.exception(
                        "Failed to drain child sessions while resuming session_id=%s",
                        session_id,
                    )
                try:
                    await self._runtime_factory.resume_root(agent_loop, session_id)
                except RuntimeSessionNotFoundError as exc:
                    raise RequestFailure(
                        ProtocolErrorCode.NOT_FOUND, f"Session not found: {session_id}"
                    ) from exc
                except SessionBusyError as exc:
                    raise RequestFailure(ProtocolErrorCode.CONFLICT, str(exc)) from exc
                except RequestFailure:
                    raise
                except Exception as exc:
                    raise RequestFailure(
                        ProtocolErrorCode.INTERNAL_ERROR,
                        f"Failed to resume session {session_id}: {exc}",
                    ) from exc
                # The core rebind has committed. Refresh and attach the public
                # projection before the next await so cancellation cannot leave
                # the coordinator pointing at the old session.
                self._root_session.replace_from_core()
                self._root_session.attach(agent_loop.session_id)
                try:
                    await previous.session.turns.reset()
                except Exception:
                    logger.exception(
                        "Failed to reset the turn controller while resuming "
                        "session_id=%s",
                        session_id,
                    )
                state = self._root_session.append_checkpoint(
                    current_history=[],
                    kind="resume",
                    message=(
                        "Session resumed. Live and retained agent handles were not "
                        "restored; old agent IDs now raise UnknownAgentError."
                    ),
                    history_limit=history_limit,
                )
                self._sessions.begin_root_generation()
                self._sessions._admission_closed = False
                return state

    async def _adopt_root(
        self,
        replacement: AgentLoop,
        history_limit: int,
        inherited_loops: list[CoreScheduledLoop],
    ) -> PublicSessionState:
        async with self._services.lifecycle_transition():
            previous = self._require_root()
            return await self._install_root(
                previous,
                replacement,
                history_limit,
                resume_checkpoint=False,
                initial_loops=inherited_loops,
            )

    async def _install_root(  # noqa: PLR0915
        self,
        previous: SessionBackendImpl,
        replacement: AgentLoop,
        history_limit: int,
        *,
        resume_checkpoint: bool,
        initial_loops: list[CoreScheduledLoop] | None = None,
    ) -> PublicSessionState:
        acquire = asyncio.ensure_future(
            asyncio.to_thread(acquire_for_attachment, replacement.cwd)
        )
        try:
            acquired = await asyncio.shield(acquire)
        except asyncio.CancelledError:
            while not acquire.done():
                with suppress(asyncio.CancelledError):
                    await asyncio.shield(acquire)
            acquired = acquire.result()
            if acquired.outcome is AcquireResult.SUCCESS and acquired.token is not None:
                release_holder(acquired.token)
            raise
        if acquired.outcome not in {AcquireResult.SUCCESS, AcquireResult.UNMANAGED}:
            await close_agent_loop(replacement)
            raise RuntimeError("Unable to reserve replacement worktree")
        pending_token = acquired.token
        try:
            await self._sessions.drain_children()
        except Exception:
            logger.exception(
                "Failed to drain child sessions while installing session_id=%s",
                replacement.session_id,
            )
        self._sessions.release_root(previous.session)
        try:
            self._bind_root(
                replacement,
                await self._resolve_mcp_catalog(replacement),
                initial_loops=initial_loops,
            )
        except BaseException:
            if pending_token is not None:
                release_holder(pending_token)
            self._sessions.bind_root(previous.session)
            self._root = previous
            self._sessions._admission_closed = False
            await close_agent_loop(replacement)
            raise
        replacement_backend = self._require_root()
        try:
            replacement_backend.worktree_token = self._worktrees.hold(
                replacement.cwd, replacement.session_id, pending_token
            )
            await self._sessions.drain_children()
            self._root_session.attach(replacement.session_id)
            if initial_loops is not None:
                await replacement_backend.resources.persist_loops()
        except BaseException:
            with suppress(BaseException):
                await replacement_backend.shutdown()
            # Failed attached replacements must leave the previous root serving.
            self._sessions.bind_root(previous.session)
            self._root = previous
            self._sessions._admission_closed = False
            raise
        try:
            await previous.shutdown()
        except BaseException as exc:
            logger.warning(
                "Failed to close previous root while resuming session_id=%s",
                replacement.session_id,
                exc_info=exc,
            )
        if resume_checkpoint:
            state = self._root_session.append_checkpoint(
                current_history=[],
                kind="resume",
                message="Session resumed",
                history_limit=history_limit,
            )
        else:
            state = self._root_session.public_state(
                current_history=[],
                callbacks=[],
                turns=[],
                retrying=None,
                history_limit=history_limit,
            )
        replacement_backend.adopt_state(state)
        self._sessions._admission_closed = False
        return state

    async def _attach_opened_session(
        self,
        agent_loop: AgentLoop,
        history_limit: int,
        *,
        resumed: bool = False,
        created_worktree: PreparedWorktree | None = None,
        opened_worktree: WorktreeResolution,
    ) -> PublicSessionState:
        try:
            self._bind_root(agent_loop, await self._resolve_mcp_catalog(agent_loop))
            if created_worktree is not None:
                await self._report_created_worktree(agent_loop, created_worktree)
            started = await self._handler.dispatch(
                "session/start",
                SessionStartParams(
                    agent_config=AgentConfig(cwd=str(agent_loop.cwd)),
                    history_limit=history_limit,
                ).model_dump(mode="json", by_alias=True),
            )
            assert isinstance(started.response, SessionStartResponse)
            state = started.response.state
            token = self._worktrees.hold(
                agent_loop.cwd, agent_loop.session_id, opened_worktree.pending_token
            )
            self._require_root().worktree_token = token
            self._worktrees.start_sweep(
                agent_loop.cwd, _resumable_directories_impl(agent_loop.config)
            )
            if resumed:
                state = self._root_session.append_checkpoint(
                    current_history=[],
                    kind="resume",
                    message=(
                        "Session resumed. Live and retained agent handles were not "
                        "restored; old agent IDs now raise UnknownAgentError."
                    ),
                    history_limit=history_limit,
                )
            self._require_root().adopt_state(state)
            return state
        except BaseException:
            root = self._root
            self._root = None
            if root is not None:
                await root.shutdown()
            else:
                await close_agent_loop(agent_loop)
            raise

    async def _resolve_mcp_catalog(
        self, agent_loop: AgentLoop
    ) -> ResolvedMCPCatalog | None:
        if self._mcp_catalog_service is None:
            return None
        return await self._mcp_catalog_service.resolve_catalog(
            agent_loop.config_orchestrator
        )

    async def _open_runtime(
        self,
        params: SessionOpenParams,
        session_id: str | None,
        *,
        continue_latest: bool = False,
    ) -> OpenedRuntime:
        options = params.agent_config.model_copy(update={"cwd": params.cwd})
        try:
            worktree_resolution = await self._worktrees.resolve_for_start(options)
            agent_loop: AgentLoop | None = None
            try:
                agent_loop = await self._open_root(
                    RootOpenRequest(
                        options=worktree_resolution.options,
                        client_info=self._services.client_info(),
                        client_capabilities=self._services.client_capabilities(),
                        session_id=session_id,
                        continue_latest=continue_latest,
                    )
                )
                actual_cwd = Path(agent_loop.cwd).expanduser().resolve()
                resolved_cwd = (
                    Path(worktree_resolution.options.cwd or Path.cwd())
                    .expanduser()
                    .resolve()
                )
                if (
                    worktree_resolution.prepared_worktree is None
                    and actual_cwd != resolved_cwd
                ):
                    actual = await self._worktrees.resolve_for_start(
                        worktree_resolution.options.model_copy(
                            update={"cwd": str(actual_cwd)}
                        )
                    )
                    await self._worktrees.cleanup(worktree_resolution)
                    worktree_resolution = actual
            except BaseException as startup_error:
                cleanup_errors: list[BaseException] = []
                if agent_loop is not None:
                    try:
                        await close_agent_loop(agent_loop)
                    except BaseException as exc:
                        cleanup_errors.append(exc)
                try:
                    await self._worktrees.cleanup(worktree_resolution)
                except BaseException as exc:
                    cleanup_errors.append(exc)
                if cleanup_errors:
                    raise BaseExceptionGroup(
                        "Runtime startup and cleanup failed",
                        [startup_error, *cleanup_errors],
                    )
                raise
            assert agent_loop is not None
            return OpenedRuntime(
                agent_loop=agent_loop, worktree_resolution=worktree_resolution
            )
        except RuntimeAuthenticationError as exc:
            data: dict[str, str] = {"provider": exc.provider}
            if exc.classification is not None:
                data["classification"] = exc.classification.value
            raise RequestFailure(
                ProtocolErrorCode.UNAUTHORIZED,
                str(exc),
                data=data,  # type: ignore[arg-type]
            ) from exc
        except RuntimeConfigurationError as exc:
            raise RequestFailure(
                ProtocolErrorCode.INVALID_PARAMS,
                str(exc),
                data={"kind": "configuration"},
            ) from exc
        except GitError as exc:
            raise RequestFailure(ProtocolErrorCode.INVALID_PARAMS, str(exc)) from exc

    async def _attach_opened_runtime(
        self, opened: OpenedRuntime, history_limit: int, *, resumed: bool = False
    ) -> PublicSessionState:
        created = opened.worktree_resolution.prepared_worktree
        if created is not None and not created.created:
            created = None
        try:
            state = await self._attach_opened_session(
                opened.agent_loop,
                history_limit,
                resumed=resumed,
                created_worktree=created,
                opened_worktree=opened.worktree_resolution,
            )
        except BaseException:
            await self._worktrees.cleanup(opened.worktree_resolution)
            raise
        root = self._root
        if root is not None and created is not None:
            root.created_worktree = created
        return state

    async def _report_created_worktree(
        self, agent_loop: AgentLoop, worktree: PreparedWorktree
    ) -> None:
        entry_id = f"worktree-{worktree.name}"
        effect = WorktreeEffect.created(worktree)
        try:
            await self._turns.start_effect(
                session_id=agent_loop.session_id,
                entry_id=entry_id,
                title="worktree",
                detail=effect.detail,
            )
            await self._turns.complete_effect(entry_id, effect.state)
        except Exception as exc:
            logger.warning("Failed to show the created worktree", exc_info=exc)
        try:
            await agent_loop.session_logger.persist_created_worktree(
                WorktreeContext(
                    entry_id=entry_id,
                    name=worktree.name,
                    branch=worktree.branch,
                    path=str(worktree.root),
                    created_at=now_ms(),
                )
            )
        except Exception as exc:
            logger.warning("Failed to record the created worktree", exc_info=exc)

    def _ensure_scheduler(self) -> None:
        if self._scheduler_task is not None and not self._scheduler_task.done():
            return
        self._scheduler_task = asyncio.create_task(
            self._run_scheduler(), name="vibe-scheduled-loops"
        )
        self._scheduler_task.add_done_callback(self._scheduler_finished)

    @staticmethod
    def _scheduler_finished(task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            logger.error(
                "Scheduled loops task failed",
                exc_info=(type(error), error, error.__traceback__),
            )

    def _restart_title_drain(self) -> None:
        # Rebind to the current agent loop; resume/continue swaps it in place.
        # If we're already draining this loop's queue, do nothing: cancelling and
        # immediately recreating would briefly leave two consumers on the one
        # queue, and a title event could be delivered to the dying task and lost.
        loop = self._agent_loop
        task = self._title_drain_task
        if task is not None and not task.done() and self._title_drain_loop is loop:
            return
        if task is not None and not task.done():
            task.cancel()
        self._title_drain_loop = loop
        self._title_drain_task = asyncio.create_task(
            self._run_title_drain(), name="vibe-title-drain"
        )
        self._title_drain_task.add_done_callback(self._title_drain_finished)

    @staticmethod
    def _title_drain_finished(task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            logger.error(
                "Session title drain task failed",
                exc_info=(type(error), error, error.__traceback__),
            )

    async def _run_title_drain(self) -> None:
        async for event in self._agent_loop.out_of_band_events():
            try:
                if isinstance(event, SessionTitleUpdatedEvent):
                    await self._notify_title(event)
                elif isinstance(event, BackgroundWorkEvent):
                    await self._handle_background_work(event)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await self._services.notify(
                    "error", ServerErrorParams(error=public_error(exc))
                )

    async def _handle_background_work(self, event: BackgroundWorkEvent) -> None:
        if self._root_session.update_background_work(event):
            await self._turns.emit_snapshot(include_history=False, include_turns=False)

    async def _notify_title(self, event: SessionTitleUpdatedEvent) -> None:
        # The background title is session metadata: reflect it in the public
        # projection (a single /title patch) so every projection consumer, e.g.
        # the ACP session state, sees it live. The terminal notice is the
        # separate, CLI-owned concern for the tab title. Both are emitted by this
        # single drain owner, so nothing competes with the turn projector.
        session_id = event.session_id
        # A reset (/new, /clear) can swap in a new session while a title for the
        # old one is still queued. Dropping the stale event keeps its title from
        # reviving on the fresh session's tab.
        if session_id != self._agent_loop.session_id:
            return
        title = event.title
        now = now_ms()
        await self._services.notify(
            "session/updated",
            SessionUpdatedParams(
                event_id=0,
                session_id=session_id,
                patch=[JsonPatchOperation(op="replace", path="/title", value=title)],
                emitted_at=now,
            ),
        )
        entry = PublicNoticeEntry(
            id=str(uuid4()),
            session_id=session_id,
            turn_id=None,
            created_at=now,
            updated_at=now,
            generation_status=PublicEntryGenerationStatus.COMPLETED,
            level="info",
            message="Session title updated",
            detail=SessionTitleUpdatedNoticeDetail(title=title),
        )
        await self._services.notify(
            "history/entryAdded",
            HistoryEntryAddedParams(
                event_id=0,
                session_id=session_id,
                turn_id=None,
                entry=entry,
                emitted_at=now,
            ),
        )

    async def _run_scheduler(self) -> None:
        while True:
            try:
                delay = self._resources.next_loop_due_in()
                await asyncio.sleep(max(0.05, min(delay, 1.0)))
                if (
                    self._require_root().session.execution.active is not None
                    or self._turns.has_queued_turns
                ):
                    continue
                loop = self._resources.due_loop()
                if loop is None:
                    continue
                try:
                    _, start = self._turns.start(
                        TurnStartParams(
                            session_id=self._agent_loop.session_id,
                            message=[TextContentBlock(text=loop.prompt)],
                        ),
                        scheduled_loop_id=loop.id,
                    )
                except (SessionExecutionConflict, TurnConflictError):
                    continue
                except ModelChoicePendingError:
                    # A recovered session must wait for an explicit model
                    # choice; the pending state is already surfaced as a
                    # runtime issue. ``due()`` does not advance
                    # ``next_fire_at``, so skipping this cycle would retry
                    # (and fail) on every scheduler pass. Mark the loop fired
                    # to back off one full interval without running it; after
                    # an explicit selection it fires normally again.
                    await self._resources.mark_loop_fired(loop.id)
                    continue
                start()
                await self._resources.mark_loop_fired(loop.id)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await self._services.notify(
                    "error", ServerErrorParams(error=public_error(exc))
                )


def create_session_backend_host_impl(
    *,
    open_root: OpenRoot,
    runtime_factory: AgentRuntimeFactory,
    host_handler: HostRequestHandler,
    stage_root: StageRoot | None,
    services: SessionBackendServices,
    mcp_catalog_service: MCPCatalogService | None = None,
) -> SessionBackendHostImpl:
    return SessionRuntimeControllerImpl(
        open_root=open_root,
        runtime_factory=runtime_factory,
        host_handler=host_handler,
        stage_root=stage_root,
        services=services,
        mcp_catalog_service=mcp_catalog_service,
    ).create_host()


def _resumable_directories_impl(config: ChartreuxConfigSchema) -> ResumableDirectories:
    """The legacy store's answer for the sweep, off the event loop."""

    async def _resumable() -> tuple[Path, ...]:
        return await asyncio.to_thread(resume_directories, config)

    return _resumable
