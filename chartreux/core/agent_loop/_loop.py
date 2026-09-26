from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable, Generator, Sequence
import contextlib
import copy
from dataclasses import dataclass, replace
from enum import StrEnum, auto
from functools import wraps
import inspect
import json
import os
from pathlib import Path
import shutil
import threading
import time
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

from pydantic import BaseModel, JsonValue

from chartreux.core.agent_loop._request_broker import InteractionRequestBroker
from chartreux.core.agent_loop._title_cadence import TitleCadence
from chartreux.core.agent_loop.backend_lifetime import BackendLifetime, BackendPublish
from chartreux.core.agent_loop.errors import (
    AgentLoopError,
    AgentLoopLLMResponseError as AgentLoopLLMResponseError,
    AgentLoopStateError,
    ImagesNotSupportedError,
)
from chartreux.core.agent_loop.llm_gateway import (
    CallResources,
    CompletionInputs,
    LLMGateway,
    TranscriptAppend,
    select_backend,
)
from chartreux.core.agent_loop.title_controller import (
    ScheduledTitle,
    TitleController,
    TitleEventSink,
    TitleGateInputs,
    TitleScheduleInputs,
)
from chartreux.core.agent_loop_hooks import AgentLoopHooksMixin, PostToolFinalization
from chartreux.core.agents.launch import LaunchCandidate
from chartreux.core.agents.manager import AgentManager
from chartreux.core.autocompletion.path_prompt import build_path_prompt_payload
from chartreux.core.checkpoints import Checkpointer, CheckpointRecorder, FileStore
from chartreux.core.compaction import (
    CompactionFailedError as CompactionFailedError,
    CompactionManager,
    reorder_for_tool_adjacency,
)
from chartreux.core.compaction.context import select_model_context
from chartreux.core.config import ChartreuxConfigSchema, ModelConfig, ProviderConfig
from chartreux.core.config._restrictions import SourceRestrictions
from chartreux.core.config.harness_files import (
    HarnessFilesManager,
    get_harness_files_manager,
)
from chartreux.core.config.layers.project import ProjectConfigLayer
from chartreux.core.config.orchestrator import ConfigOrchestrator
from chartreux.core.errors import ContextTooLongError
from chartreux.core.events import (
    AssistantEvent,
    BackgroundWorkEvent,
    BaseEvent,
    CompactEndEvent,
    CompactStartEvent,
    ReasoningEvent,
    SessionTitleUpdatedEvent,
    ToolCallEvent,
    ToolResultEvent,
    ToolStreamEvent,
    UserMessageEvent,
)
from chartreux.core.git.errors import GitError
from chartreux.core.git.worktree.repository import WorktreeRepository
from chartreux.core.hooks.config import load_hooks_from_fs
from chartreux.core.hooks.manager import HooksManager
from chartreux.core.hooks.models import HookConfigResult, HookEvent
from chartreux.core.llm.backend.factory import create_backend
from chartreux.core.llm.failures import RequestRetryBudget, classify
from chartreux.core.llm.format import (
    APIToolFormatHandler,
    FailedToolCall,
    ResolvedMessage,
    ResolvedToolCall,
)
from chartreux.core.llm.types import BackendLike
from chartreux.core.llm.utility_completion import is_fast_utility_model
from chartreux.core.llm_models import (
    AvailableTool,
    FunctionCall,
    ImageAttachment,
    LLMChunk,
    LLMMessage,
    ManualShellContext,
    PersistedToolResult,
    Role,
    StrToolChoice,
    ToolCall,
)
from chartreux.core.message_list import MessageList
from chartreux.core.middleware import (
    AutoCompactMiddleware,
    ContextWarningMiddleware,
    ConversationContext,
    MiddlewareAction,
    MiddlewarePipeline,
    MiddlewareResult,
    PriceLimitMiddleware,
    ResetReason,
    TokenLimitMiddleware,
    TurnLimitMiddleware,
)
from chartreux.core.model_catalog.availability import (
    EligibleDeployment,
    eligible_deployments,
)
from chartreux.core.plan_session import PlanSession
from chartreux.core.review import ReviewManager
from chartreux.core.rewind import RewindManager
from chartreux.core.scratchpad import cleanup_scratchpad, init_scratchpad
from chartreux.core.session.session_id import extract_suffix, generate_session_id
from chartreux.core.session.session_lease import SessionLease
from chartreux.core.session.session_logger import SessionLogger
from chartreux.core.session_types import (
    AgentStats,
    ChildSessionLink,
    CommittedModelIdentity,
    LaunchMetadataV2,
    LaunchPersonaV1,
    SessionMetadata,
)
from chartreux.core.skills.manager import SkillManager
from chartreux.core.subagents import (
    LaunchConfig,
    SubagentManagementPort,
    SubagentRunnerPort,
)
from chartreux.core.system_prompt import get_universal_system_prompt
from chartreux.core.tools import secret_redaction
from chartreux.core.tools.base import (
    BaseTool,
    CancellableToolResult,
    InvokeContext,
    ToolError,
    ToolPermission,
    ToolPermissionError,
)
from chartreux.core.tools.builtins.read_file import ReadFileArgs
from chartreux.core.tools.builtins.skill import (
    Skill as SkillTool,
    SkillArgs,
    build_skill_result,
    skill_content_marker,
)
from chartreux.core.tools.builtins.todo import TodoState
from chartreux.core.tools.io_port import ToolIOPort
from chartreux.core.tools.manager import NoSuchToolError, ToolManager
from chartreux.core.tools.ui import ToolUIDataAdapter
from chartreux.core.utils import (
    TOOL_ERROR_TAG,
    VIBE_STOP_EVENT_TAG,
    CancellationReason,
    RetryObserver,
    RetryReason,
    get_user_cancellation_message,
    is_user_cancellation_event,
)
from chartreux.core.workspace import Workspace
from chartreux.observability.logging import logger
from chartreux.user_content import UserDisplayContent, UserResource
from chartreux.utils.cache_store import CacheStore, InMemoryCacheStore
from chartreux.utils.http import get_user_agent


def _is_git_executable_available() -> bool:
    executable = os.environ.get("GIT_PYTHON_GIT_EXECUTABLE")
    if not executable:
        return shutil.which("git") is not None

    path = Path(executable).expanduser()
    if path.is_absolute() or os.sep in executable:
        return path.is_file() and os.access(path, os.X_OK)
    return shutil.which(executable) is not None


if TYPE_CHECKING:
    from chartreux.core.tools.mcp.pool import MCPConnectionPool
    from chartreux.core.tools.mcp.registry import MCPRegistry


class ToolExecutionResponse(StrEnum):
    SKIP = auto()
    EXECUTE = auto()


class ToolDecision(BaseModel):
    verdict: ToolExecutionResponse
    approval_type: ToolPermission
    feedback: str | None = None


@dataclass(frozen=True, slots=True)
class AgentRuntimePolicy:
    max_turns: int | None
    max_price: float | None
    max_tokens: int | None
    max_session_tokens: int | None
    enable_streaming: bool
    launch_context: dict[str, object] | None
    headless: bool
    hook_config_result: HookConfigResult | None
    cache_store: CacheStore
    # Whether this surface wants background LLM session titles. Core owns the
    # capability; the delivery layer (app server) owns this policy decision.
    auto_title_enabled: bool = False
    inherited_workspace: Workspace | None = None
    inherited_restrictions: tuple[SourceRestrictions, ...] = ()
    inherited_mode_restrictions: tuple[SourceRestrictions, ...] = ()
    inherited_plan_write_scopes: tuple[tuple[Path, Path | None], ...] = ()
    parent_authority_getter: Callable[[], ToolManager] | None = None
    parent_authority_revision_getter: Callable[[], int] | None = None


class _SwappableConfigSource:
    """Config getter for reload-prepared managers.

    Points at the target agent's config while preparation runs off-loop, then is
    repointed to the live config inside the synchronous commit. The prepared
    managers are not shared with the running turn until commit, so the running
    turn only ever observes the live getter.
    """

    def __init__(self, getter: Callable[[], ChartreuxConfigSchema]) -> None:
        self._getter = getter
        self.live = False

    def get(self) -> ChartreuxConfigSchema:
        return self._getter()

    def point_to(self, getter: Callable[[], ChartreuxConfigSchema]) -> None:
        self._getter = getter
        self.live = True


@dataclass(frozen=True, slots=True)
class _PreparedPolicyReplacement:
    owner: AgentLoop
    session_generation: int
    expected_token: object
    orchestrator: ConfigOrchestrator[ChartreuxConfigSchema]
    previous_manager: ToolManager
    tool_manager: ToolManager
    inherited_restrictions: tuple[SourceRestrictions, ...]
    inherited_workspace: Workspace | None


@dataclass(frozen=True, slots=True)
class _PreparedReload:
    config: ChartreuxConfigSchema
    backend: BackendLike
    agent_manager: AgentManager
    tool_manager: ToolManager
    skill_manager: SkillManager
    system_prompt: str
    config_source: _SwappableConfigSource
    hook_config_result: HookConfigResult | None
    skills_adopted: int


@dataclass(frozen=True, slots=True)
class _PreparedLaunchReconfiguration:
    candidate: LaunchCandidate
    orchestrator: ConfigOrchestrator[ChartreuxConfigSchema]
    consumers: _PreparedReload
    previous_orchestrator: ConfigOrchestrator[ChartreuxConfigSchema]
    previous_agent_manager: AgentManager
    previous_tool_manager: ToolManager
    previous_skill_manager: SkillManager
    previous_system_message: LLMMessage | None
    previous_prices: tuple[float, float, float | None]
    previous_launch_overrides: LaunchConfig | None
    previous_committed_model: CommittedModelIdentity | None
    session_generation: int
    parent_authority_revision: int | None
    owns_backend: bool


class _QueueTitleEventSink(TitleEventSink):
    def __init__(self, queue: asyncio.Queue[BaseEvent]) -> None:
        self._queue = queue

    def publish(self, event: BaseEvent) -> None:
        self._queue.put_nowait(event)


@dataclass(frozen=True, slots=True)
class AgentTurnOptions:
    retry_sink: RetryObserver | None = None
    injected: bool = False
    user_initiated_retry: bool = False


@dataclass(frozen=True, slots=True)
class _ActiveTurn:
    """Collaborators lent to the loop for one turn; its presence means one is running."""

    subagent_runner: SubagentRunnerPort | None = None
    tool_io: ToolIOPort | None = None
    retry_sink: RetryObserver | None = None


_NO_TURN = _ActiveTurn()


class _PrepareScratchpad:
    pass


_PREPARE_SCRATCHPAD = _PrepareScratchpad()

# Test-only kill switch for harnesses that run the real CLI against a mock model.
_DISABLE_AUTO_TITLE_ENV_VAR = "CHARTREUX_TEST_DISABLE_AUTO_TITLE"


def requires_init(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator that awaits deferred initialization before executing the method."""
    if inspect.isasyncgenfunction(fn):

        @wraps(fn)
        async def gen_wrapper(self: AgentLoop, *args: Any, **kwargs: Any) -> Any:
            with secret_redaction.bind_policy(self.scrub_policy):
                await self.wait_until_ready()
            agen = fn(self, *args, **kwargs)
            sent: Any = None
            try:
                while True:
                    with secret_redaction.bind_policy(self.scrub_policy):
                        try:
                            event = await agen.asend(sent)
                        except StopAsyncIteration:
                            return
                    sent = yield event
            finally:
                with secret_redaction.bind_policy(self.scrub_policy):
                    await agen.aclose()

        return gen_wrapper

    @wraps(fn)
    async def wrapper(self: AgentLoop, *args: Any, **kwargs: Any) -> Any:
        with secret_redaction.bind_policy(self.scrub_policy):
            await self.wait_until_ready()
            return await fn(self, *args, **kwargs)

    return wrapper


class AgentLoop(AgentLoopHooksMixin):  # noqa: PLR0904
    def __init__(  # noqa: PLR0913, PLR0915
        self,
        config_orchestrator: ConfigOrchestrator[ChartreuxConfigSchema],
        *,
        max_turns: int | None = None,
        max_price: float | None = None,
        max_tokens: int | None = None,
        max_session_tokens: int | None = None,
        backend: BackendLike | None = None,
        enable_streaming: bool = False,
        launch_context: dict[str, object] | None = None,
        is_subagent: bool = False,
        defer_heavy_init: bool = False,
        headless: bool = False,
        hook_config_result: HookConfigResult | None = None,
        mcp_registry: MCPRegistry | None = None,
        cache_store: CacheStore | None = None,
        auto_title_enabled: bool = False,
        parent_session_id: str | None = None,
        cwd: Path | None = None,
        harness_files: HarnessFilesManager | None = None,
        session_id: str | None = None,
        session_dir: Path | None = None,
        session_lease: SessionLease | None = None,
        inherited_workspace: Workspace | None = None,
        inherited_restrictions: tuple[SourceRestrictions, ...] = (),
        inherited_mode_restrictions: tuple[SourceRestrictions, ...] = (),
        inherited_plan_write_scopes: tuple[tuple[Path, Path | None], ...] = (),
        parent_authority_getter: Callable[[], ToolManager] | None = None,
        parent_authority_revision_getter: Callable[[], int] | None = None,
        launch_profile: str | None = None,
        launch_overrides: LaunchConfig | None = None,
        frozen_system_prompt_id: str | None = None,
        frozen_instructions: str | None = None,
        committed_model: CommittedModelIdentity | None = None,
    ) -> None:
        self._inherited_restrictions = tuple(inherited_restrictions)
        self._inherited_mode_restrictions = tuple(inherited_mode_restrictions)
        self._inherited_plan_write_scopes = tuple(inherited_plan_write_scopes)
        self._parent_authority_getter = parent_authority_getter
        self._parent_authority_revision_getter = parent_authority_revision_getter
        self._authority_revision = 0
        self.launch_profile = launch_profile
        self.launch_overrides = (
            launch_overrides.model_copy(deep=True) if launch_overrides else None
        )
        self.frozen_system_prompt_id = frozen_system_prompt_id
        self.frozen_instructions = frozen_instructions
        self.committed_model = committed_model
        self._committed_selection: str | None = None
        self.cwd = (cwd or Path.cwd()).resolve()
        # A child's own cwd/config cannot reconstruct a narrower parent ceiling.
        # Direct callers must carry explicit parent authority, just like the factory.
        if is_subagent and inherited_workspace is None:
            raise ValueError("Subagent construction requires inherited_workspace")
        self._inherited_workspace = inherited_workspace
        self.harness_files = replace(
            harness_files or get_harness_files_manager(), cwd=self.cwd
        )
        self._config_orchestrator = config_orchestrator
        self._auto_title_enabled = auto_title_enabled
        self._headless = headless
        self._is_subagent = is_subagent
        self.cache_store = cache_store or InMemoryCacheStore()

        self._defer_heavy_init = defer_heavy_init
        self._deferred_init_thread: threading.Thread | None = None
        self._deferred_init_lock = threading.Lock()
        self._closing = False
        self._close_task: asyncio.Task[None] | None = None
        self._init_error: Exception | None = None
        self._init_start_time = time.monotonic()
        self._skills_adopted: int = 0
        self._plan_attrs_task: asyncio.Task[None] | None = None
        self._reload_generation: int = 0
        self._session_generation: int = 0
        self._init_duration_pending: bool = defer_heavy_init
        self._last_init_duration_ms: int | None = None
        # Background title results land here. An app-server drain surfaces them
        # immediately (between turns); otherwise the next turn drains them.
        self._out_of_band_events: asyncio.Queue[BaseEvent] = asyncio.Queue()

        self.session_id = session_id or generate_session_id()
        self._session_lease = session_lease
        self.parent_session_id = parent_session_id
        self.scratchpad_dir = (
            init_scratchpad(self.session_id) if not is_subagent else None
        )

        self.agent_manager = AgentManager(
            self._config_orchestrator, harness_files=self.harness_files
        )
        # Configuration parsing is pure; runtime IO is explicit and must precede
        # construction of any clients, including the MCP pool.
        _ = self.config.system_prompt, self.config.compaction_prompt
        self.mcp_registry: MCPRegistry | None = (
            mcp_registry
            if defer_heavy_init
            else mcp_registry or self._create_mcp_registry()
        )
        self.scrub_policy = secret_redaction.ScrubPolicy.from_config(self.config)
        secret_redaction.register_session_policy(self, self.scrub_policy)
        self._retired_mcp_pools: list[MCPConnectionPool] = []
        self._mcp_pool: MCPConnectionPool | None = (
            None if defer_heavy_init else self._create_mcp_pool()
        )
        self.config.require_active_provider_api_key()
        config = self.config
        self._committed_selection = config.active_model
        if self.committed_model is None and config.catalog_snapshot is not None:
            from chartreux.core.model_catalog.resolver import resolver_for

            selection = config.active_model or config.resolve_default_model_alias()
            self.committed_model = (
                resolver_for(config)
                .resolve(selection, allowed_models=config.allowed_models)
                .identity
            )
        if self.committed_model is not None:
            config.attach_committed_model(self.committed_model)
        self.tool_manager = ToolManager(
            lambda: self.config,
            mcp_registry=self.mcp_registry,
            defer_mcp=True,
            restriction_getter=lambda: self.config_orchestrator.restrictions,
            accepted_token_getter=lambda: self.config_orchestrator.accepted_token,
            inherited_restrictions=self._inherited_restrictions,
            inherited_workspace=self._inherited_workspace,
            inherited_plan_write_scopes=self._inherited_plan_write_scopes,
            parent_authority_getter=self._parent_authority_getter,
            parent_authority_revision_getter=self._parent_authority_revision_getter,
            cwd=self.cwd,
            harness_files=self.harness_files,
            scratchpad_dir=self.scratchpad_dir,
        )
        self.skill_manager = SkillManager(
            lambda: self.config, harness_files=self.harness_files
        )
        self._max_turns = max_turns
        self._max_price = max_price
        self._max_tokens = max_tokens
        self._max_session_tokens = max_session_tokens
        self._plan_session = PlanSession()
        self._user_plan: str | None = None

        self.format_handler = APIToolFormatHandler()
        self._llm_gateway = LLMGateway()

        self._injected_backend = backend
        self._backend_lifetime = BackendLifetime(self.backend_factory())
        self._completion_providers: list[tuple[str, ...]] = []
        self._failover_switches: list[dict[str, str]] = []

        self.enable_streaming = enable_streaming
        self.middleware_pipeline = MiddlewarePipeline()
        self._setup_middleware()

        self.messages = MessageList()

        self.stats = AgentStats()
        self._tool_event_queue: asyncio.Queue[BaseEvent | None] | None = None
        # Retain admission names and resolved values only until outward emission.
        self._admitted_tool_policies: dict[str, secret_redaction.ScrubPolicy] = {}
        # Tool-response ordering: while a tool batch is active, response
        # messages are staged and appended in tool-call index order even
        # though their events keep streaming in completion order.
        self._tool_response_order: dict[str, int] | None = None
        self._staged_tool_responses: dict[int, list[LLMMessage]] = {}
        self._next_tool_response_index = 0
        self._request_broker = InteractionRequestBroker()
        self._active_turn: _ActiveTurn | None = None
        # Operations that are not turns but still hold the session: anything
        # reading the working directory or acting on its repository for longer
        # than an instant. They exclude each other through _take_session, so an
        # operation announces itself rather than every other one having to know
        # it exists. A turn will not start while one is held either, which is
        # what makes it safe for a holder to await.
        self._holders: list[str] = []
        # The directory a move granted session trust to, so a later move
        # releases that and not a grant somebody else made.
        self._trust_taken_by_move: Path | None = None
        self.launch_context = launch_context
        config = self.config
        try:
            active_model = config.get_active_model()
            self.stats.input_price_per_million = active_model.input_price
            self.stats.output_price_per_million = active_model.output_price
            self.stats.cached_input_price_per_million = active_model.cached_input_price
        except ValueError:
            pass

        self._current_user_message_id: str | None = None
        self._is_user_prompt_call: bool = False
        self._reactive_recovery_used: bool = False
        self._pending_injected_messages: list[LLMMessage] = []
        self._pending_clear_context: bool = False

        self.session_logger = SessionLogger(
            config.session_logging,
            self.session_id,
            cwd=self.cwd,
            session_dir=session_dir,
        )
        self._title_controller = TitleController(
            read_at_gate=lambda: TitleGateInputs(
                config=self.config, previous_title=self.session_logger.title
            ),
            writer=self.session_logger,
            event_sink=_QueueTitleEventSink(self._out_of_band_events),
        )
        if self.session_logger.session_metadata is not None:
            self.session_logger.session_metadata.parent_session_id = parent_session_id
            self.install_launch_metadata()
        self._hook_config_result = hook_config_result
        self._hooks_manager = (
            HooksManager(hook_config_result.hooks, cwd=self.cwd)
            if hook_config_result
            else None
        )
        self.hook_config_issues = (
            hook_config_result.issues if hook_config_result else []
        )
        self.hooks_count = len(hook_config_result.hooks) if hook_config_result else 0
        checkpointer = Checkpointer()
        file_store = FileStore()
        self.checkpoint_recorder = CheckpointRecorder(
            checkpointer, self.messages, file_store
        )
        self.review_manager = ReviewManager(checkpointer, file_store)
        self.rewind_manager = RewindManager(
            checkpointer,
            messages=self.messages,
            save_messages=self._save_messages,
            reset_session=self._reset_session,
            files=file_store,
        )
        self.compaction_manager = CompactionManager(
            messages=self.messages,
            stats_getter=lambda: self.stats,
            config_getter=lambda: self.config,
            complete=self._complete,
            available_tools=lambda: self.format_handler.get_available_tools(
                self.tool_manager
            ),
            tool_choice=self.format_handler.get_tool_choice,
            save=self._save_messages,
        )

        if defer_heavy_init:
            self._start_deferred_init()
        else:
            self._complete_init()
            if err := self._init_error:
                raise err

    def _start_deferred_init(self) -> threading.Thread:
        """Spawn a daemon thread that finishes deferred heavy I/O once."""
        with self._deferred_init_lock:
            if self._deferred_init_thread is not None:
                return self._deferred_init_thread
            if self._closing:
                raise AgentLoopStateError("Cannot initialize a closing agent loop")

            thread = threading.Thread(
                target=self._complete_init, daemon=True, name="agent_loop_init"
            )
            self._deferred_init_thread = thread
            thread.start()
            return thread

    @property
    def is_initialized(self) -> bool:
        """Whether deferred initialization has completed (successfully or not)."""
        if not self._defer_heavy_init:
            return True
        thread = self._deferred_init_thread
        return thread is not None and not thread.is_alive()

    def _complete_init(self) -> None:
        """Run deferred heavy I/O: MCP discovery.

        Intended to be called from a background thread when
        ``defer_heavy_init=True`` was passed to ``__init__``.
        """
        with self._deferred_init_lock:
            if self._closing:
                return
        try:
            with secret_redaction.bind_policy(self.scrub_policy):
                self._ensure_remote_registries()
                self.tool_manager.integrate_all(raise_on_mcp_failure=True)
                self.messages.update_system_prompt(self._build_system_prompt())
        except Exception as exc:
            self._init_error = exc

    async def wait_until_ready(self) -> None:
        """Await deferred initialization and record its duration."""
        await self._await_deferred_init()
        self._ensure_init_duration_recorded()

    async def _await_deferred_init(self) -> None:
        """Await the deferred init thread and plan-attribute task."""
        if self._defer_heavy_init:
            thread = self._start_deferred_init()
            await asyncio.to_thread(thread.join)
            if err := self._init_error:
                raise copy.copy(err).with_traceback(err.__traceback__)
        for task in (self._plan_attrs_task,):
            if task is None or task is asyncio.current_task():
                continue
            with contextlib.suppress(asyncio.CancelledError):
                await task

    def _ensure_init_duration_recorded(self) -> None:
        """Record initialization duration exactly once when initialization is armed."""
        if self._last_init_duration_ms is not None:
            return
        if not self._init_duration_pending and not self._defer_heavy_init:
            return
        self._last_init_duration_ms = int(
            (time.monotonic() - self._init_start_time) * 1000
        )
        self._init_duration_pending = False

    @property
    def config_orchestrator(self) -> ConfigOrchestrator[ChartreuxConfigSchema]:
        return self._config_orchestrator

    @property
    def config(self) -> ChartreuxConfigSchema:
        return self.agent_manager.config

    @property
    def runtime_policy(self) -> AgentRuntimePolicy:
        return AgentRuntimePolicy(
            max_turns=self._max_turns,
            max_price=self._max_price,
            max_tokens=self._max_tokens,
            max_session_tokens=self._max_session_tokens,
            enable_streaming=self.enable_streaming,
            launch_context=self.launch_context,
            headless=self._headless,
            hook_config_result=self._hook_config_result,
            cache_store=self.cache_store,
            auto_title_enabled=self._auto_title_enabled,
            inherited_restrictions=self._inherited_restrictions,
            inherited_mode_restrictions=self._inherited_mode_restrictions,
            inherited_workspace=self._inherited_workspace,
            inherited_plan_write_scopes=self._inherited_plan_write_scopes,
            parent_authority_getter=self._parent_authority_getter,
            parent_authority_revision_getter=self._parent_authority_revision_getter,
        )

    @property
    def child_runtime_policy(self) -> AgentRuntimePolicy:
        """Capture accepted parent authority before child overrides or awaits.

        Only construction fixes this snapshot. Future explicit revocation must
        refresh descendants; ordinary child config/grants cannot replace it.
        Mode restrictions are execution scope, not persistent user policy.
        """
        sources = self.config_orchestrator.restrictions
        return replace(
            self.runtime_policy,
            inherited_workspace=self.tool_manager.workspace,
            parent_authority_getter=lambda: self.tool_manager,
            parent_authority_revision_getter=lambda: self._authority_revision,
            inherited_restrictions=tuple(
                dict.fromkeys(
                    self._inherited_restrictions
                    + tuple(source for source in sources if source.kind == "source")
                )
            ),
        )

    async def record_child_session(
        self, child: AgentLoop, tool_call_id: str, agent_name: str
    ) -> ChildSessionLink:
        parent_dir = self.session_logger.session_dir
        child_dir = child.session_logger.session_dir
        relative_path = (
            str(child_dir.relative_to(parent_dir))
            if parent_dir is not None and child_dir is not None
            else None
        )
        link = ChildSessionLink(
            session_id=child.session_id,
            tool_call_id=tool_call_id,
            agent=agent_name,
            relative_path=relative_path,
        )
        metadata = self.session_logger.session_metadata
        if metadata is None:
            return link
        existing = next(
            (
                item
                for item in metadata.child_sessions
                if item.tool_call_id == tool_call_id
            ),
            None,
        )
        if existing is not None:
            if existing != link:
                raise RuntimeError(
                    f"Tool call {tool_call_id} is already linked to a child session"
                )
            return existing
        metadata.child_sessions.append(link)
        try:
            await self._save_messages()
            await self.session_logger.persist_child_sessions()
        except BaseException:
            metadata.child_sessions.remove(link)
            raise
        return link

    async def replace_child_session(
        self, old_session_id: str, child: AgentLoop, tool_call_id: str
    ) -> ChildSessionLink:
        parent_dir = self.session_logger.session_dir
        child_dir = child.session_logger.session_dir
        replacement = ChildSessionLink(
            session_id=child.session_id,
            tool_call_id=tool_call_id,
            agent="subagent",
            relative_path=(
                str(child_dir.relative_to(parent_dir))
                if parent_dir is not None and child_dir is not None
                else None
            ),
        )
        metadata = self.session_logger.session_metadata
        if metadata is None:
            return replacement
        index = next(
            (
                index
                for index, link in enumerate(metadata.child_sessions)
                if link.session_id == old_session_id
                and link.tool_call_id == tool_call_id
            ),
            None,
        )
        if index is None:
            raise RuntimeError(f"Child session link not found: {old_session_id}")
        previous = metadata.child_sessions[index]
        metadata.child_sessions[index] = replacement
        try:
            await self.session_logger.persist_child_sessions()
        except BaseException:
            metadata.child_sessions[index] = previous
            raise
        return replacement

    async def forget_child_session(
        self, child_session_id: str, tool_call_id: str
    ) -> None:
        metadata = self.session_logger.session_metadata
        if metadata is None:
            return
        index = next(
            (
                index
                for index, link in enumerate(metadata.child_sessions)
                if link.session_id == child_session_id
                and link.tool_call_id == tool_call_id
            ),
            None,
        )
        if index is None:
            return
        link = metadata.child_sessions.pop(index)
        try:
            await self.session_logger.persist_child_sessions()
        except BaseException:
            metadata.child_sessions.insert(index, link)
            raise

    async def persist_empty_session(self) -> None:
        await self._save_messages(allow_empty=True)

    async def refresh_config(self) -> None:
        await self._config_orchestrator.reload()
        self.scrub_policy = secret_redaction.ScrubPolicy.from_config(self.config)
        secret_redaction.register_session_policy(self, self.scrub_policy)
        secret_redaction.reset_cache()
        self._retire_mcp_pool()
        self._ensure_remote_registries()
        if self.mcp_registry is not None:
            self.mcp_registry.sync_active_servers(self.config.mcp_servers)

    def _drain_pending_injections(self) -> bool:
        if not self._pending_injected_messages:
            return False
        for injected in self._pending_injected_messages:
            self.messages.append(injected)
        self._pending_injected_messages.clear()
        return True

    def resolve_user_input_request(self, request_id: str, result: BaseModel) -> None:
        self._request_broker.resolve_user_input(request_id, result)

    def reject_request(self, request_id: str, error: BaseException) -> None:
        self._request_broker.reject(request_id, error)

    @property
    def init_duration_ms(self) -> int | None:
        return self._last_init_duration_ms

    @property
    def _auto_title_task(self) -> asyncio.Task[None] | None:
        return self._title_controller.task

    @property
    def _title_cadence(self) -> TitleCadence:
        return self._title_controller.cadence

    @_title_cadence.setter
    def _title_cadence(self, cadence: TitleCadence) -> None:
        self._title_controller.cadence = cadence

    async def aclose(self) -> None:
        task = self._close_task
        if task is None or (task.done() and task.exception() is not None):
            task = asyncio.create_task(
                self._aclose_owned(), name=f"vibe-agent-close:{self.session_id}"
            )
            self._close_task = task
            task.add_done_callback(self._observe_close_outcome)
        await asyncio.shield(task)
        secret_redaction.unregister_session_policy(self)

    def _observe_close_outcome(self, task: asyncio.Task[None]) -> None:
        if task.cancelled():
            logger.error("Owned agent cleanup was cancelled: %s", self.session_id)
            return
        if exc := task.exception():
            logger.error(
                "Owned agent cleanup failed: %s", self.session_id, exc_info=exc
            )

    async def _aclose_owned(self) -> None:
        with self._deferred_init_lock:
            self._closing = True
            deferred_init_thread = self._deferred_init_thread
        if deferred_init_thread is not None:
            await asyncio.to_thread(deferred_init_thread.join, 5.0)
            if deferred_init_thread.is_alive():
                raise TimeoutError(
                    "Deferred initialization did not stop during shutdown"
                )
        if self._mcp_pool is not None:
            self._mcp_pool.retire()
        await self._title_controller.aclose()
        for task in (self._plan_attrs_task,):
            if task is not None and not task.done():
                task.cancel()
                with contextlib.suppress(BaseException):
                    await task
        pools = list(self._retired_mcp_pools)
        if self._mcp_pool is not None:
            pools.append(self._mcp_pool)
        outcomes = await asyncio.gather(
            *(pool.aclose() for pool in pools), return_exceptions=True
        )
        self._retired_mcp_pools = [
            pool for pool in self._retired_mcp_pools if not pool.cleanup_complete
        ]
        errors = [result for result in outcomes if isinstance(result, BaseException)]
        if errors:
            # Keep ownership and session resources until cleanup actually finishes.
            # A later close can join a timed-out worker; it must not replay its call.
            raise BaseExceptionGroup("MCP shutdown cleanup incomplete", errors)
        await self._backend_lifetime.aclose()
        if self._trust_taken_by_move is not None:
            self.harness_files.trust_store.revoke_session_trust(
                self._trust_taken_by_move
            )
            self._trust_taken_by_move = None
        self.harness_files.revoke_session_trusts()
        cleanup_scratchpad(self.scratchpad_dir)
        lease = self._session_lease
        self._session_lease = None
        if lease is not None:
            await asyncio.to_thread(lease.release)

    @staticmethod
    def _create_mcp_registry() -> MCPRegistry:
        from chartreux.core.tools.mcp.registry import MCPRegistry

        return MCPRegistry()

    def _create_mcp_pool(self) -> MCPConnectionPool:
        from chartreux.core.tools.mcp.pool import MCPConnectionPool

        return MCPConnectionPool(policy=self.scrub_policy)

    def _retire_mcp_pool(self) -> None:
        self._retired_mcp_pools = [
            pool for pool in self._retired_mcp_pools if not pool.cleanup_complete
        ]
        if self._mcp_pool is not None:
            self._mcp_pool.retire(drain_active=True)
            self._retired_mcp_pools.append(self._mcp_pool)
            self._mcp_pool = None

    def _ensure_remote_registries(self) -> None:
        if self.mcp_registry is None and self.config.mcp_servers:
            self.mcp_registry = self._create_mcp_registry()
            self.tool_manager.set_mcp_registry(self.mcp_registry)

        if self._mcp_pool is None and self.config.mcp_servers:
            self._mcp_pool = self._create_mcp_pool()

    def _render_system_prompt(
        self,
        skill_manager: SkillManager,
        config: ChartreuxConfigSchema | None = None,
        tool_manager: ToolManager | None = None,
        agent_manager: AgentManager | None = None,
    ) -> str:
        prompt_config = config or self.config
        if self.frozen_system_prompt_id is not None:
            prompt_config = prompt_config.model_copy(
                update={"system_prompt_id": self.frozen_system_prompt_id}
            )
        return get_universal_system_prompt(
            prompt_config,
            skill_manager,
            agent_manager or self.agent_manager,
            scratchpad_dir=self.scratchpad_dir,
            headless=self._headless,
            cwd=self.cwd,
            harness_files=self.harness_files,
            tool_manager=tool_manager or self.tool_manager,
            role_instructions=self.frozen_instructions,
        )

    def _build_system_prompt(self) -> str:
        return self._render_system_prompt(self.skill_manager)

    async def prepare_launch_reconfiguration(
        self,
        candidate: LaunchCandidate,
        *,
        expected_session_generation: int,
        expected_parent_authority_revision: int,
    ) -> _PreparedLaunchReconfiguration:
        """Stage only changed retained-child consumers without publishing them."""
        await self.wait_until_ready()
        self._require_policy_idle()
        if expected_session_generation != self._session_generation:
            raise AgentLoopStateError("Stale launch reconfiguration generation")
        parent_revision = (
            self._parent_authority_revision_getter()
            if self._parent_authority_revision_getter is not None
            else None
        )
        if parent_revision != expected_parent_authority_revision:
            raise AgentLoopStateError("Parent authority changed during reservation")
        orchestrator = candidate.orchestrator.copy()
        reuse_backend = (
            self._injected_backend is not None
            or orchestrator.config.get_active_provider()
            == self.config.get_active_provider()
        )
        tool_fields = ("enabled_tools", "disabled_tools", "tools")
        previous_overrides = self.launch_overrides or LaunchConfig()
        replace_tools = any(
            getattr(previous_overrides, field)
            != getattr(candidate.semantic_overrides, field)
            for field in tool_fields
        )
        preparation = asyncio.create_task(
            asyncio.to_thread(
                self._prepare_launch_consumers,
                orchestrator.config,
                replace_tools=replace_tools,
                reuse_backend=reuse_backend,
            )
        )
        try:
            consumers = await asyncio.shield(preparation)
        except asyncio.CancelledError as cancellation:
            while not preparation.done():
                try:
                    await asyncio.shield(preparation)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            if not preparation.cancelled():
                try:
                    abandoned = preparation.result()
                except Exception:
                    abandoned = None
                if abandoned is not None and not reuse_backend:
                    with contextlib.suppress(Exception):
                        await abandoned.backend.__aexit__(None, None, None)
            raise cancellation
        system_message = (
            self.messages[0].model_copy(deep=True)
            if self.messages and self.messages[0].role is Role.system
            else None
        )
        return _PreparedLaunchReconfiguration(
            candidate=candidate,
            orchestrator=orchestrator,
            consumers=consumers,
            previous_orchestrator=self._config_orchestrator,
            previous_agent_manager=self.agent_manager,
            previous_tool_manager=self.tool_manager,
            previous_skill_manager=self.skill_manager,
            previous_system_message=system_message,
            previous_prices=(
                self.stats.input_price_per_million,
                self.stats.output_price_per_million,
                self.stats.cached_input_price_per_million,
            ),
            previous_launch_overrides=(
                self.launch_overrides.model_copy(deep=True)
                if self.launch_overrides is not None
                else None
            ),
            previous_committed_model=self.committed_model,
            session_generation=expected_session_generation,
            parent_authority_revision=parent_revision,
            owns_backend=not reuse_backend,
        )

    def _prepare_launch_consumers(
        self,
        target_config: ChartreuxConfigSchema,
        *,
        replace_tools: bool,
        reuse_backend: bool,
    ) -> _PreparedReload:
        """Build the narrow consumer set required by a retained re-task."""
        config_source = _SwappableConfigSource(lambda: target_config)
        tool_manager = self.tool_manager
        if replace_tools:
            tool_manager = ToolManager(
                config_source.get,
                mcp_registry=self.mcp_registry,
                defer_mcp=True,
                discovery_source=self.tool_manager,
                restriction_getter=lambda: self.config_orchestrator.restrictions,
                accepted_token_getter=lambda: (
                    self.config_orchestrator.accepted_token
                    if config_source.live
                    else None
                ),
                inherited_restrictions=self._inherited_restrictions,
                inherited_workspace=self._inherited_workspace,
                inherited_plan_write_scopes=self._inherited_plan_write_scopes,
                parent_authority_getter=self._parent_authority_getter,
                parent_authority_revision_getter=self._parent_authority_revision_getter,
                cwd=self.cwd,
                harness_files=self.harness_files,
                scratchpad_dir=self.scratchpad_dir,
            )
        system_prompt = self._render_system_prompt(
            self.skill_manager, target_config, tool_manager, self.agent_manager
        )
        return _PreparedReload(
            config=target_config,
            backend=self.backend
            if reuse_backend
            else self.backend_factory(target_config),
            agent_manager=self.agent_manager,
            tool_manager=tool_manager,
            skill_manager=self.skill_manager,
            system_prompt=system_prompt,
            config_source=config_source,
            hook_config_result=self._hook_config_result,
            skills_adopted=self._skills_adopted,
        )

    def publish_launch_reconfiguration(
        self, prepared: _PreparedLaunchReconfiguration
    ) -> BackendPublish:
        """Synchronously publish reversible state immediately before turn start."""
        self._require_policy_idle()
        parent_revision = (
            self._parent_authority_revision_getter()
            if self._parent_authority_revision_getter is not None
            else None
        )
        if (
            prepared.previous_orchestrator is not self._config_orchestrator
            or prepared.session_generation != self._session_generation
            or prepared.parent_authority_revision != parent_revision
        ):
            raise AgentLoopStateError("Stale launch reconfiguration")
        backend_publish = self._backend_lifetime.publish_reversible(
            prepared.consumers.backend
        )
        self._config_orchestrator = prepared.orchestrator
        prepared.consumers.agent_manager._orchestrator = prepared.orchestrator
        self.agent_manager = prepared.consumers.agent_manager
        self.tool_manager = prepared.consumers.tool_manager
        self.skill_manager = prepared.consumers.skill_manager
        prepared.consumers.config_source.point_to(lambda: self.config)
        self.messages.update_system_prompt(prepared.consumers.system_prompt)
        model = prepared.candidate.effective_model
        self.stats.update_pricing(
            model.input_price, model.output_price, model.cached_input_price
        )
        self.launch_overrides = prepared.candidate.semantic_overrides.model_copy(
            deep=True
        )
        self.committed_model = prepared.candidate.committed_model
        self.config.attach_committed_model(self.committed_model)
        return backend_publish

    def rollback_launch_reconfiguration(
        self, prepared: _PreparedLaunchReconfiguration, backend_publish: BackendPublish
    ) -> None:
        backend_publish.rollback()
        self._config_orchestrator = prepared.previous_orchestrator
        prepared.previous_agent_manager._orchestrator = prepared.previous_orchestrator
        self.agent_manager = prepared.previous_agent_manager
        self.tool_manager = prepared.previous_tool_manager
        self.skill_manager = prepared.previous_skill_manager
        if prepared.previous_system_message is not None:
            self.messages.update_system_prompt(
                str(prepared.previous_system_message.content or "")
            )
        self.stats.update_pricing(*prepared.previous_prices)
        self.launch_overrides = prepared.previous_launch_overrides
        self.committed_model = prepared.previous_committed_model
        self.config.attach_committed_model(self.committed_model)

    def _model_choice_pending(self) -> bool:
        """True while a recovered session awaits an explicit model selection.

        Mirrors ``committed_model_recovery_issue`` in the app-server
        projection: the stored launch envelope still commits a model while
        the loop no longer carries one, so resolving a default here would
        silently re-pin the session and dissolve the pending-choice issue.
        """
        if self.committed_model is not None:
            return False
        metadata = self.session_logger.session_metadata
        envelope = metadata.launch_config if metadata is not None else None
        return isinstance(envelope, LaunchMetadataV2)

    def _launch_metadata(self) -> LaunchMetadataV2 | None:
        if self.committed_model is None:
            return None
        return LaunchMetadataV2(
            version=2,
            profile=self.launch_profile,
            overrides=self.launch_overrides or LaunchConfig(),
            persona=LaunchPersonaV1(
                system_prompt_id=self.frozen_system_prompt_id
                or self.config.system_prompt_id,
                instructions=self.frozen_instructions,
            ),
            committed_model=self.committed_model,
        )

    def install_launch_metadata(self) -> None:
        """Synchronously install accepted launch state before any lock wait."""
        if (metadata := self._launch_metadata()) is not None:
            self.session_logger.install_launch_config(metadata)

    async def persist_launch_metadata(self) -> None:
        if (metadata := self._launch_metadata()) is not None:
            await self.session_logger.persist_launch_config(metadata)

    async def abort_launch_reconfiguration(
        self, prepared: _PreparedLaunchReconfiguration
    ) -> None:
        if prepared.owns_backend:
            with contextlib.suppress(Exception):
                await prepared.consumers.backend.__aexit__(None, None, None)

    def finalize_launch_reconfiguration(
        self, prepared: _PreparedLaunchReconfiguration, backend_publish: BackendPublish
    ) -> None:
        backend_publish.finalize(whole_turn_active=True)
        if prepared.previous_tool_manager is not prepared.consumers.tool_manager:
            prepared.previous_tool_manager._retire_authority()

    @requires_init
    async def refresh_system_prompt(self) -> None:
        """Rebuild and replace the system prompt with current tool/skill state."""
        prompt = await asyncio.to_thread(self._build_system_prompt)
        self.messages.update_system_prompt(prompt)

    @property
    def backend(self) -> BackendLike:
        return self._backend_lifetime.active

    @property
    def _turn(self) -> _ActiveTurn:
        return self._active_turn or _NO_TURN

    def _take_session(self, operation: str) -> None:
        """Claim the session for *operation*, refusing if something else holds it.

        Both the claim and the refusal, because mutual exclusion needs each side
        to do both. An operation that only announced itself could still start
        inside another one's awaits.
        """
        if self._holders:
            raise AgentLoopStateError(
                f"Cannot start {operation} while {self._holders[0]} is running"
            )
        self._holders.append(operation)

    def _release_session(self, operation: str) -> None:
        self._holders.remove(operation)

    async def notice_retry(self, reason: RetryReason) -> None:
        if (sink := self._turn.retry_sink) is not None:
            await sink(reason)

    def backend_factory(
        self, config: ChartreuxConfigSchema | None = None
    ) -> BackendLike:
        return self._injected_backend or self._select_backend(config)

    def _select_backend(
        self, config: ChartreuxConfigSchema | None = None
    ) -> BackendLike:
        return select_backend(
            config or self.config, on_retry=self.notice_retry, factory=create_backend
        )

    async def _save_messages(self, *, allow_empty: bool = False) -> None:
        await self.session_logger.save_interaction(
            self.messages,
            self.stats,
            self.config,
            self.tool_manager,
            None,
            allow_empty=allow_empty,
        )

    @requires_init
    async def inject_user_context(
        self,
        content: str,
        *,
        as_message: bool = False,
        inject_implicit: bool = False,
        images: list[ImageAttachment] | None = None,
        input_text: str | None = None,
        resources: list[UserResource] | None = None,
        client_message_id: str | None = None,
        manual_shell: ManualShellContext | None = None,
    ) -> list[BaseEvent]:
        events: list[BaseEvent] = []
        if as_message:
            message = LLMMessage(
                role=Role.user,
                content=content,
                message_id=client_message_id or str(uuid4()),
                images=images or None,
                input_text=input_text,
                resources=resources or None,
                manual_shell=manual_shell,
            )
            self.messages.append(message)
            if message.message_id is None:
                raise AgentLoopError("User message must have a message_id")
            events.append(
                UserMessageEvent(
                    content=input_text if input_text is not None else content,
                    message_id=message.message_id,
                    images=list(message.images or []),
                    resources=list(message.resources or []),
                )
            )
            if inject_implicit:
                async for event in self._inject_invoked_skill(content):
                    events.append(event)
                async for event in self._inject_mentioned_files(content):
                    events.append(event)
        else:
            self.messages.append(
                LLMMessage(
                    role=Role.user,
                    content=content,
                    injected=True,
                    images=images or None,
                    input_text=input_text,
                    resources=resources or None,
                    manual_shell=manual_shell,
                )
            )
        await self._save_messages()
        return events

    @requires_init
    async def act(
        self,
        msg: str | None,
        client_message_id: str | None = None,
        *,
        auto_title: str | None = None,
        images: list[ImageAttachment] | None = None,
        user_display_content: UserDisplayContent | None = None,
        input_text: str | None = None,
        resources: list[UserResource] | None = None,
        subagent_runner: SubagentRunnerPort | None = None,
        tool_io: ToolIOPort | None = None,
        turn_options: AgentTurnOptions | None = None,
    ) -> AsyncGenerator[BaseEvent, None]:
        try:
            active_model = self.config.get_active_model()
        except ValueError:
            active_model = None
        if images and active_model is not None and not active_model.supports_images:
            raise ImagesNotSupportedError(
                active_model.display_name or active_model.alias
            )
        if self._active_turn is not None:
            raise AgentLoopStateError("A turn is already active")
        # A holder is not a turn, so the check above cannot see one. A move
        # refuses while a turn is active and then awaits twice; without this a
        # turn could begin in that window and bind its tools to the directory
        # being moved out from under them.
        if self._holders:
            raise AgentLoopStateError(
                f"Cannot start a turn while {self._holders[0]} is running"
            )
        self._backend_lifetime.drain(whole_turn_active=False)
        options = turn_options or AgentTurnOptions()
        if options.user_initiated_retry and self.committed_model is not None:
            self.config_orchestrator.availability_registry.reset_base(
                self.committed_model.base_model
            )
        self._admitted_tool_policies.clear()
        self._active_turn = _ActiveTurn(
            subagent_runner=subagent_runner,
            tool_io=tool_io,
            retry_sink=options.retry_sink,
        )
        try:
            self._clean_message_history()
            self.checkpoint_recorder.create_checkpoint()
            try:
                async with contextlib.aclosing(
                    self._conversation_loop(
                        msg,
                        client_message_id=client_message_id,
                        auto_title=auto_title,
                        images=images,
                        user_display_content=user_display_content,
                        input_text=input_text,
                        resources=resources,
                        injected=options.injected,
                    )
                ) as conversation:
                    async for event in self._sanitize_outward_events(conversation):
                        yield event
            finally:
                self.checkpoint_recorder.seal_turn()
        finally:
            self._admitted_tool_policies.clear()
            self._active_turn = None
            self._backend_lifetime.drain(whole_turn_active=False)

    async def _sanitize_outward_events(
        self, events: AsyncGenerator[BaseEvent]
    ) -> AsyncGenerator[BaseEvent]:
        # Hold events behind the earliest unfinished stream so concurrent tool
        # completions cannot reorder the events seen by consumers.
        marker = "[Unchecked tool stream remainder suppressed]"
        pending: list[BaseEvent] = []
        streams: dict[str, list[ToolStreamEvent]] = {}
        stream_sizes: dict[str, int] = {}
        truncated: set[str] = set()
        async for event in events:
            ready: list[BaseEvent] = []
            with secret_redaction.bind_policy(
                secret_redaction.ScrubPolicy.for_redaction(
                    self._admitted_tool_policies.values(), self.scrub_policy
                )
            ):
                if isinstance(event, ToolStreamEvent):
                    call_id = event.tool_call_id
                    is_first = call_id not in streams
                    used = stream_sizes.get(call_id, 0)
                    remaining = max(0, 8192 * 128 - used)
                    if len(event.message) > remaining:
                        truncated.add(call_id)
                    if remaining and event.message:
                        streams.setdefault(call_id, []).append(
                            event.model_copy(
                                update={"message": event.message[:remaining]}
                            )
                        )
                    elif call_id not in streams:
                        streams[call_id] = [event.model_copy(update={"message": ""})]
                    stream_sizes[call_id] = used + min(len(event.message), remaining)
                    if is_first:
                        pending.append(streams[call_id][0])
                elif isinstance(event, ToolResultEvent):
                    chunks = streams.pop(event.tool_call_id, [])
                    stream_sizes.pop(event.tool_call_id, None)
                    if chunks:
                        replacement = self._sanitize_stream(
                            chunks, marker, event.tool_call_id in truncated
                        )
                        first = pending.index(chunks[0])
                        pending[first] = replacement
                    truncated.discard(event.tool_call_id)
                    pending.append(self._sanitize_outward_event(event))
                    self._admitted_tool_policies.pop(event.tool_call_id, None)
                else:
                    pending.append(self._sanitize_outward_event(event))
                while pending and not (
                    isinstance(pending[0], ToolStreamEvent)
                    and pending[0].tool_call_id in streams
                ):
                    ready.append(pending.pop(0))
            for item in ready:
                yield item
        # A stream with no result is still output; never expose its unchecked
        # remainder if the source generator ended unexpectedly.
        for item in pending:
            if isinstance(item, ToolStreamEvent) and item.tool_call_id in streams:
                with secret_redaction.bind_policy(
                    secret_redaction.ScrubPolicy.for_redaction(
                        self._admitted_tool_policies.values(), self.scrub_policy
                    )
                ):
                    item = item.model_copy(
                        update={
                            "message": marker,
                            "tool_name": secret_redaction.redact(item.tool_name),
                            "tool_call_id": secret_redaction.redact(item.tool_call_id),
                        }
                    )
            yield item

    @staticmethod
    def _sanitize_stream(
        chunks: list[ToolStreamEvent], marker: str, truncated: bool
    ) -> ToolStreamEvent:
        body = "".join(chunk.message for chunk in chunks)
        # Streams are buffered, so scan the entire retained body once. A window
        # cannot prove that a credential crossing its edge was checked (even at
        # the documented 16 KiB maximum); never emit independently checked
        # windows. The 1 MiB retention cap still bounds this whole-body scan.
        sanitized = secret_redaction.redact(body)
        if truncated:
            sanitized += marker
        return chunks[0].model_copy(
            update={
                "message": sanitized,
                "tool_name": secret_redaction.redact(chunks[0].tool_name),
                "tool_call_id": secret_redaction.redact(chunks[0].tool_call_id),
            }
        )

    def _sanitize_outward_event(self, event: BaseEvent) -> BaseEvent:
        """Sanitize tool and hook payloads before UI/ACP consumers see them."""
        if not isinstance(event, (ToolResultEvent, HookEvent)):
            return event
        try:
            return secret_redaction.redact_model(event)
        except Exception:
            logger.warning(
                "Tool event sanitation failed; suppressing payload", exc_info=True
            )
            if isinstance(event, ToolResultEvent):
                return ToolResultEvent(
                    tool_name="tool",
                    tool_class=None,
                    tool_call_id=secret_redaction.redact(event.tool_call_id),
                    error="Tool result unavailable",
                )
            return HookEvent()

    def _last_user_message(self) -> LLMMessage | None:
        return AgentLoop._last_user_message_from(select_model_context(self.messages))

    def _current_model_context(self) -> list[LLMMessage]:
        return select_model_context(self.messages)

    @staticmethod
    def _last_user_message_from(messages: Sequence[LLMMessage]) -> LLMMessage | None:
        return next(
            (m for m in reversed(messages) if m.role == Role.user and not m.injected),
            None,
        )

    def set_max_turns(self, max_turns: int) -> None:
        self._max_turns = max_turns
        self._setup_middleware()

    def set_max_tokens(self, max_tokens: int) -> None:
        self._max_tokens = max_tokens

    def _setup_middleware(self) -> None:
        self.middleware_pipeline = self._build_middleware_pipeline(self.config)

    def _build_middleware_pipeline(
        self, config: ChartreuxConfigSchema
    ) -> MiddlewarePipeline:
        """Construct the candidate pipeline without clearing the active one."""
        pipeline = MiddlewarePipeline()

        if self._max_turns is not None:
            pipeline.add(TurnLimitMiddleware(self._max_turns))

        if self._max_price is not None:
            pipeline.add(PriceLimitMiddleware(self._max_price))

        if self._max_session_tokens is not None:
            pipeline.add(TokenLimitMiddleware(self._max_session_tokens))

        pipeline.add(AutoCompactMiddleware())
        if config.context_warnings:
            pipeline.add(ContextWarningMiddleware(0.5))

        return pipeline

    async def _handle_middleware_result(
        self, result: MiddlewareResult
    ) -> AsyncGenerator[BaseEvent]:
        match result.action:
            case MiddlewareAction.STOP:
                yield AssistantEvent(
                    content=f"<{VIBE_STOP_EVENT_TAG}>{result.reason}</{VIBE_STOP_EVENT_TAG}>",
                    stopped_by_middleware=True,
                )

            case MiddlewareAction.INJECT_MESSAGE:
                if result.message:
                    injected_message = LLMMessage(
                        role=Role.user, content=result.message, injected=True
                    )
                    self.messages.append(injected_message)

            case MiddlewareAction.COMPACT:
                async for event in self._run_compaction():
                    yield event

            case MiddlewareAction.CONTINUE:
                pass

    async def _run_compaction(self) -> AsyncGenerator[BaseEvent]:
        # Auto/reactive compaction: emit boundary events, compact, report status.
        old_tokens = self.stats.context_tokens
        threshold = self.config.get_active_model().auto_compact_threshold
        old_session_id = self.session_id
        tool_call_id = str(uuid4())

        yield CompactStartEvent(
            tool_call_id=tool_call_id,
            current_context_tokens=old_tokens,
            threshold=threshold,
        )

        try:
            summary = await self.compact()
        except asyncio.CancelledError:
            raise

        yield CompactEndEvent(
            tool_call_id=tool_call_id,
            summary_length=len(summary),
            old_session_id=old_session_id,
            new_session_id=self.session_id,
        )

    @property
    def user_plan(self) -> str | None:
        # The legacy ``_user_plan`` field is a fallback for paths that resolve a
        # plan without an explicit user selection.
        return self._user_plan

    def set_user_plan(self, user_plan: str | None) -> None:
        self._user_plan = user_plan

    def _should_self_heal(self) -> bool:
        # Recover from an overflow at most once per turn; strict mode surfaces it.
        return (
            not self._reactive_recovery_used
            and not self.config.raise_on_compaction_failure
        )

    def _get_context(self) -> ConversationContext:
        return ConversationContext(
            messages=self.messages, stats=self.stats, config=self.config
        )

    def _build_backend_metadata(self, call_type: str | None = None) -> dict[str, str]:
        metadata = {
            "session_id": self.session_id,
            "call_type": call_type
            or ("main_call" if self._is_user_prompt_call else "secondary_call"),
        }
        if self.parent_session_id is not None:
            metadata["parent_session_id"] = self.parent_session_id
        if self._current_user_message_id is not None:
            metadata["message_id"] = self._current_user_message_id
        if self.user_plan is not None:
            metadata["user_plan"] = self.user_plan
        if isinstance(self.launch_context, dict):
            for key in (
                "agent_entrypoint",
                "agent_version",
                "client_name",
                "client_version",
                "terminal_emulator",
            ):
                value = self.launch_context.get(key)
                if value is not None:
                    metadata[key] = str(value)
        return metadata

    def _get_extra_headers(
        self, provider: ProviderConfig | None = None
    ) -> dict[str, str]:
        provider = self.config.get_active_provider() if provider is None else provider
        headers: dict[str, str] = {}
        headers["user-agent"] = get_user_agent(provider.backend)
        headers["x-affinity"] = self.session_id
        return headers

    async def _open_user_turn(
        self,
        user_msg: str | None,
        *,
        client_message_id: str | None = None,
        auto_title: str | None = None,
        images: list[ImageAttachment] | None = None,
        user_display_content: UserDisplayContent | None = None,
        input_text: str | None = None,
        resources: list[UserResource] | None = None,
        injected: bool = False,
    ) -> AsyncGenerator[BaseEvent]:
        if user_msg is None:
            # Idle Plan preparation already appended the reviewed seed. Do not
            # fabricate a second user message for the implementation phase.
            last_user = self._last_user_message()
            self._current_user_message_id = (
                last_user.message_id if last_user is not None else None
            )
            return
        if injected:
            self.messages.append(
                LLMMessage(
                    role=Role.user,
                    content=user_msg,
                    injected=True,
                    images=images or None,
                    user_display_content=user_display_content,
                    input_text=input_text,
                    resources=resources or None,
                )
            )
            last_user = self._last_user_message()
            self._current_user_message_id = (
                last_user.message_id if last_user is not None else None
            )
            return

        user_message = LLMMessage(
            role=Role.user,
            content=user_msg,
            message_id=client_message_id,
            images=images or None,
            user_display_content=user_display_content,
            input_text=input_text,
            resources=resources or None,
        )
        self.messages.append(user_message)
        self.stats.steps += 1
        self._current_user_message_id = user_message.message_id

        if user_message.message_id is None:
            raise AgentLoopError("User message must have a message_id")

        yield UserMessageEvent(
            content=input_text if input_text is not None else user_msg,
            message_id=user_message.message_id,
            images=list(user_message.images or []),
            user_display_content=user_message.user_display_content,
            resources=list(user_message.resources or []),
        )

        async for event in self._inject_invoked_skill(user_msg):
            yield event

        async for event in self._inject_mentioned_files(user_msg):
            yield event

        if auto_title is not None and self.session_logger.set_initial_auto_title(
            auto_title
        ):
            yield SessionTitleUpdatedEvent(title=auto_title, session_id=self.session_id)

        if self._hooks_manager:
            self._hooks_manager.reset_retry_count()

    async def _conversation_loop(  # noqa: PLR0912
        self,
        user_msg: str | None,
        client_message_id: str | None = None,
        *,
        auto_title: str | None = None,
        images: list[ImageAttachment] | None = None,
        user_display_content: UserDisplayContent | None = None,
        input_text: str | None = None,
        resources: list[UserResource] | None = None,
        injected: bool = False,
    ) -> AsyncGenerator[BaseEvent]:
        async for event in self._open_user_turn(
            user_msg,
            client_message_id=client_message_id,
            auto_title=auto_title,
            images=images,
            user_display_content=user_display_content,
            input_text=input_text,
            resources=resources,
            injected=injected,
        ):
            yield event

        completed_normally = False
        try:
            should_break_loop = False
            first_llm_turn = True
            self._reactive_recovery_used = False
            while not should_break_loop:
                self._is_user_prompt_call = False
                result = await self.middleware_pipeline.run_before_turn(
                    self._get_context()
                )
                async for event in self._handle_middleware_result(result):
                    yield event

                if result.action == MiddlewareAction.STOP:
                    return

                user_cancelled = False
                self._is_user_prompt_call = first_llm_turn
                try:
                    async with contextlib.aclosing(self._perform_llm_turn()) as turn:
                        async for event in turn:
                            user_cancelled = (
                                user_cancelled or is_user_cancellation_event(event)
                            )
                            yield event
                except ContextTooLongError:
                    if not self._should_self_heal():
                        raise
                    self._reactive_recovery_used = True
                    async for event in self._run_compaction():
                        yield event
                    continue  # retry the turn — still the user's first response
                # A turn ran to completion: count it against the turn budget (so
                # an overflow-and-retry never does) and mark later turns as
                # follow-ups.
                self.stats.steps += 1
                first_llm_turn = False
                # Per-turn save so the on-disk log stays fresh; after the
                # inner loop so pre_tool rewrites land in the snapshot.
                await self._save_messages()
                self._is_user_prompt_call = False

                # Schedule after each model step, not only at turn end: a single
                # tool-heavy turn can run for minutes, and the first title should
                # land as soon as there is usable context. The step ending with
                # an assistant answer (not a tool result) means the turn is
                # completing, which the cadence uses to time the initial title.
                async for event in self._schedule_title_generation_events(
                    turn_completing=self.messages[-1].role != Role.tool
                ):
                    yield event

                last_message = self.messages[-1]
                drained = self._drain_pending_injections()
                should_break_loop = last_message.role != Role.tool and not drained

                if user_cancelled:
                    return

                if should_break_loop:
                    retry_msg, hook_events = await self._dispatch_post_turn_hooks()
                    for hook_event in hook_events:
                        yield hook_event
                    should_break_loop = self._queue_post_turn_retry(retry_msg)
                    # The final save can yield long enough for a background-agent
                    # completion to queue an injection. Save before the last drain so
                    # no await remains between observing an empty queue and exit.
                    if should_break_loop:
                        await self._save_messages()
                    if self._drain_pending_injections():
                        should_break_loop = False
            completed_normally = True
        finally:
            if not completed_normally:
                await self._save_messages()

    def _queue_post_turn_retry(self, retry_msg: LLMMessage | None) -> bool:
        # Returns whether the loop should still break (no retry queued).
        if retry_msg is None:
            return True
        self.messages.append(retry_msg)
        return False

    async def _schedule_title_generation_events(
        self, *, turn_completing: bool
    ) -> AsyncGenerator[BackgroundWorkEvent, None]:
        scheduled = self._maybe_schedule_title_generation(
            turn_completing=turn_completing
        )
        if scheduled is None:
            return
        try:
            yield scheduled.started_event
        except (GeneratorExit, asyncio.CancelledError):
            scheduled.abort_delivery()
            raise
        else:
            scheduled.release()

    def _maybe_schedule_title_generation(
        self, *, turn_completing: bool
    ) -> ScheduledTitle | None:
        return self._title_controller.schedule(
            TitleScheduleInputs(
                messages=tuple(self.messages),
                session_id=self.session_id,
                turn_completing=turn_completing,
                enabled=self._auto_title_enabled,
                disabled_by_test_switch=(
                    os.environ.get(_DISABLE_AUTO_TITLE_ENV_VAR) == "1"
                ),
                logging_enabled=self.session_logger.enabled,
                title_is_manual=self.session_logger.title_source == "manual",
                periodic=is_fast_utility_model(self.config),
            )
        )

    async def out_of_band_events(self) -> AsyncGenerator[BaseEvent, None]:
        """Stream events produced outside a turn.

        The delivery layer drains this as the single consumer, so title and
        background-work events stay ordered and surface once.
        """
        while True:
            yield await self._out_of_band_events.get()

    def _cancel_auto_title_task(self) -> None:
        self._title_controller.cancel()

    def _reset_title_state(self) -> None:
        self._title_controller.reset()
        while not self._out_of_band_events.empty():
            self._out_of_band_events.get_nowait()

    def _skill_already_loaded(self, name: str) -> bool:
        marker = skill_content_marker(name)
        return any(
            m.role == Role.tool and m.name == "skill" and marker in (m.content or "")
            for m in self._current_model_context()
        )

    async def _inject_invoked_skill(
        self, user_msg: str
    ) -> AsyncGenerator[BaseEvent, None]:
        parsed = self.skill_manager.parse_skill_command(user_msg)
        if parsed is None:
            return
        skill_info = self.skill_manager.get_skill(parsed.name)
        if skill_info is None:
            return

        result = await build_skill_result(
            skill_info, already_loaded=self._skill_already_loaded(parsed.name)
        )
        call_id = str(uuid4())
        tool_class = self.tool_manager.available_tools.get("skill", SkillTool)
        call_event = ToolCallEvent(
            tool_call_id=call_id,
            tool_call_index=0,
            tool_name="skill",
            tool_class=tool_class,
            args=SkillArgs(name=parsed.name),
        )
        call_event = call_event.model_copy(
            update={
                "presentation": ToolUIDataAdapter(
                    tool_class, harness_files=self.harness_files
                ).get_call_presentation(call_event)
            }
        )
        result_event = ToolResultEvent(
            tool_name="skill",
            tool_class=tool_class,
            result=result,
            tool_call_id=call_id,
        )
        result_event = result_event.model_copy(
            update={
                "presentation": ToolUIDataAdapter(
                    tool_class, harness_files=self.harness_files
                ).get_result_presentation(result_event)
            }
        )

        self.messages.append(
            LLMMessage(
                role=Role.assistant,
                content="",
                tool_calls=[
                    ToolCall(
                        id=call_id,
                        index=0,
                        function=FunctionCall(
                            name="skill", arguments=json.dumps({"name": parsed.name})
                        ),
                        presentation=call_event.presentation,
                    )
                ],
            )
        )
        result_text = "\n".join(f"{k}: {v}" for k, v in result.model_dump().items())
        self.messages.append(
            LLMMessage(
                role=Role.tool,
                tool_call_id=call_id,
                name="skill",
                content=result_text,
                tool_result=PersistedToolResult(
                    output=cast(dict[str, JsonValue], result.model_dump(mode="json")),
                    presentation=result_event.presentation,
                ),
            )
        )

        yield call_event
        yield result_event

    async def _inject_mentioned_files(
        self, user_msg: str
    ) -> AsyncGenerator[BaseEvent, None]:
        payload = build_path_prompt_payload(user_msg, base_dir=self.cwd)
        file_resources = [r for r in payload.resources if r.kind == "file"]
        if not file_resources:
            return
        try:
            tool_instance = self.tool_manager.get("read_file")
        except NoSuchToolError:
            return
        tool_class = type(tool_instance)

        for resource in file_resources:
            file_path = str(resource.path)
            call_id = str(uuid4())
            self.messages.append(
                LLMMessage(
                    role=Role.assistant,
                    content="",
                    tool_calls=[
                        ToolCall(
                            id=call_id,
                            index=0,
                            function=FunctionCall(
                                name="read_file",
                                arguments=json.dumps({"file_path": file_path}),
                            ),
                        )
                    ],
                )
            )
            tool_call = ResolvedToolCall(
                tool_name="read_file",
                tool_class=tool_class,
                validated_args=ReadFileArgs(file_path=file_path),
                call_id=call_id,
            )
            call_event = ToolCallEvent(
                tool_call_id=call_id,
                tool_call_index=0,
                tool_name="read_file",
                tool_class=tool_class,
                args=ReadFileArgs(file_path=file_path),
            )
            call_event = call_event.model_copy(
                update={
                    "presentation": ToolUIDataAdapter(
                        tool_class, harness_files=self.harness_files
                    ).get_call_presentation(call_event)
                }
            )
            self._record_tool_call_presentation(call_event)
            yield call_event
            async for event in self._process_one_tool_call(tool_call):
                yield event

    async def _perform_llm_turn(self) -> AsyncGenerator[BaseEvent, None]:
        if self.enable_streaming:
            async for event in self._stream_assistant_events():
                yield event
        else:
            assistant_event = await self._get_assistant_event()
            if assistant_event.content:
                yield assistant_event

        last_message = self.messages[-1]

        parsed = self.format_handler.parse_message(last_message)
        resolved = self.format_handler.resolve_tool_calls(parsed, self.tool_manager)

        if not resolved.tool_calls and not resolved.failed_calls:
            return

        async with contextlib.aclosing(self._handle_tool_calls(resolved)) as calls:
            async for event in calls:
                yield event

    def _build_tool_call_events(
        self, tool_calls: list[ToolCall] | None, emitted_ids: set[str]
    ) -> Generator[ToolCallEvent, None, None]:
        for tc in tool_calls or []:
            if tc.id is None or not tc.function.name:
                continue
            if tc.id in emitted_ids:
                continue

            tool_class = self.tool_manager.available_tools.get(tc.function.name)
            if tool_class is None:
                continue

            event = ToolCallEvent(
                tool_call_id=tc.id,
                tool_call_index=tc.index,
                tool_name=tc.function.name,
                tool_class=tool_class,
            )
            yield event.model_copy(
                update={
                    "presentation": ToolUIDataAdapter(
                        tool_class, harness_files=self.harness_files
                    ).get_call_presentation(event)
                }
            )

    async def _stream_assistant_events(
        self,
    ) -> AsyncGenerator[AssistantEvent | ReasoningEvent | ToolCallEvent]:
        message_id: str | None = None
        reasoning_message_id: str | None = None
        emitted_tool_call_ids = set[str]()

        async for chunk in self._chat_streaming():
            if message_id is None:
                message_id = chunk.message.message_id
            if reasoning_message_id is None:
                reasoning_message_id = chunk.message.reasoning_message_id

            if chunk.message.reasoning_content:
                yield ReasoningEvent(
                    content=chunk.message.reasoning_content,
                    message_id=reasoning_message_id,
                )

            if chunk.message.content:
                yield AssistantEvent(
                    content=chunk.message.content, message_id=message_id
                )

            for event in self._build_tool_call_events(
                chunk.message.tool_calls, emitted_tool_call_ids
            ):
                emitted_tool_call_ids.add(event.tool_call_id)
                yield event

    async def _get_assistant_event(self) -> AssistantEvent:
        llm_result = await self._chat()
        return AssistantEvent(
            content=llm_result.message.content or "",
            message_id=llm_result.message.message_id,
        )

    async def _handle_tool_calls(
        self, resolved: ResolvedMessage
    ) -> AsyncGenerator[BaseEvent]:
        self._begin_tool_response_ordering()
        try:
            async for event in self._emit_failed_tool_events(resolved.failed_calls):
                yield event
            if not resolved.tool_calls:
                return

            for batch in [resolved.tool_calls]:
                async with contextlib.aclosing(
                    self._handle_tool_batch(batch)
                ) as events:
                    async for event in events:
                        yield event
        finally:
            self._end_tool_response_ordering()

    async def _handle_tool_batch(
        self, tool_calls: list[ResolvedToolCall]
    ) -> AsyncGenerator[BaseEvent]:
        for tool_call in tool_calls:
            event = ToolCallEvent(
                tool_name=tool_call.tool_name,
                tool_class=tool_call.tool_class,
                args=tool_call.validated_args,
                tool_call_id=tool_call.call_id,
            )
            event = event.model_copy(
                update={
                    "presentation": ToolUIDataAdapter(
                        tool_call.tool_class, harness_files=self.harness_files
                    ).get_call_presentation(event)
                }
            )
            self._record_tool_call_presentation(event)
            yield event

        async with contextlib.aclosing(
            self._run_tools_concurrently(tool_calls)
        ) as events:
            async for event in events:
                yield event

    def _record_tool_call_presentation(self, event: ToolCallEvent) -> None:
        if event.presentation is None:
            return
        for message in reversed(self.messages):
            for tool_call in message.tool_calls or []:
                if tool_call.id == event.tool_call_id:
                    tool_call.presentation = event.presentation
                    return

    async def _emit_failed_tool_events(
        self, failed_calls: list[FailedToolCall]
    ) -> AsyncGenerator[ToolResultEvent]:
        for failed in failed_calls:
            error_msg = f"<{TOOL_ERROR_TAG}>{failed.tool_name}: {failed.error}</{TOOL_ERROR_TAG}>"
            yield ToolResultEvent(
                tool_name=failed.tool_name,
                tool_class=None,
                error=error_msg,
                tool_call_id=failed.call_id,
            )
            self.stats.tool_calls_failed += 1
            # Same choke point as successful responses: pydantic validation
            # errors echo model-supplied argument values, so the failure text
            # can carry a secret and must be redacted before it joins the
            # message list.
            self._append_tool_response(
                failed.call_id,
                self.format_handler.create_failed_tool_response_message(
                    failed, secret_redaction.redact(error_msg)
                ),
            )

    async def _run_tools_concurrently(
        self, tool_calls: list[ResolvedToolCall]
    ) -> AsyncGenerator[BaseEvent]:
        """Execute multiple tool calls concurrently, yielding events as they arrive."""
        queue: asyncio.Queue[BaseEvent | None] = asyncio.Queue()
        if self._tool_event_queue is not None:
            raise AgentLoopStateError("A tool batch is already active")
        self._tool_event_queue = queue
        self._request_broker.bind(queue)

        # Execution retains only names/passthrough; the value snapshot is never
        # bound in a tool task or placed in a queue/event.
        admitted = self.scrub_policy
        redaction_snapshot = admitted.capture_for_redaction()
        self._admitted_tool_policies.update(
            (tc.call_id, redaction_snapshot) for tc in tool_calls
        )
        tasks = [
            asyncio.create_task(self._execute_tool_to_queue(tc, queue, admitted))
            for tc in tool_calls
        ]

        async def _signal_when_all_done() -> None:
            try:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for tool_call, result in zip(tool_calls, results, strict=True):
                    if not isinstance(result, BaseException):
                        continue
                    if isinstance(result, asyncio.CancelledError):
                        continue
                    error_msg = (
                        f"<{TOOL_ERROR_TAG}>{tool_call.tool_name} failed during "
                        f"pre-tool processing: {result}</{TOOL_ERROR_TAG}>"
                    )
                    logger.warning(
                        "Pre-tool processing failed tool=%s tool_call_id=%s error=%s",
                        tool_call.tool_name,
                        tool_call.call_id,
                        type(result).__name__,
                    )
                    self.stats.tool_calls_failed += 1
                    with secret_redaction.bind_policy(admitted):
                        await queue.put(self._tool_failure_event(tool_call, error_msg))
            finally:
                await queue.put(None)

        monitor = asyncio.create_task(_signal_when_all_done())

        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield event
        finally:
            # Closing the event stream must join tools before act() releases its
            # active-turn guard, just as cancellation and normal completion do.
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self._request_broker.unbind(queue)
            self._tool_event_queue = None
            if not monitor.done():
                monitor.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await monitor

    async def _execute_tool_to_queue(
        self,
        tc: ResolvedToolCall,
        queue: asyncio.Queue[BaseEvent | None],
        admitted: secret_redaction.ScrubPolicy,
    ) -> None:
        """Run a single tool call under its admission policy, sending events to the queue."""
        with secret_redaction.bind_policy(admitted):
            async with contextlib.aclosing(self._process_one_tool_call(tc)) as events:
                async for event in events:
                    await queue.put(event)

    async def _process_one_tool_call(
        self, tool_call: ResolvedToolCall
    ) -> AsyncGenerator[ToolResultEvent | ToolStreamEvent | HookEvent]:
        try:
            tool_instance = self.tool_manager.get(tool_call.tool_name)
        except Exception as exc:
            error_msg = f"Error getting tool '{tool_call.tool_name}': {exc}"
            yield self._tool_failure_event(tool_call, error_msg)
            return

        try:
            tool_input = self._serialize_tool_input(tool_call)
        except Exception as exc:
            error_msg = (
                f"<{TOOL_ERROR_TAG}>Failed to serialize tool input for "
                f"'{tool_call.tool_name}': {exc}</{TOOL_ERROR_TAG}>"
            )
            self.stats.tool_calls_failed += 1
            yield ToolResultEvent(
                tool_name=tool_call.tool_name,
                tool_class=tool_call.tool_class,
                error=error_msg,
                tool_call_id=tool_call.call_id,
            )
            self._handle_tool_response(tool_call, error_msg)
            return

        events, resolution = await self._run_pre_tool_pipeline(tool_call, tool_input)
        for ev in events:
            yield ev
        if resolution.denial_event is not None:
            yield resolution.denial_event
            return
        tool_call = resolution.tool_call
        tool_input = resolution.tool_input

        decision: ToolDecision | None = None
        tool_started = False
        try:
            decision = await self._should_execute_tool(
                tool_instance, tool_call.validated_args
            )

            if decision.verdict == ToolExecutionResponse.SKIP:
                async for ev in self._handle_tool_skip(tool_call, decision):
                    yield ev
                return

            tool_started = True
            async for ev in self._invoke_tool(
                tool_call, tool_instance, tool_input, decision
            ):
                yield ev

        except asyncio.CancelledError:
            cancel = str(
                get_user_cancellation_message(CancellationReason.TOOL_INTERRUPTED)
            )
            logger.info(
                "Tool call cancelled tool=%s tool_call_id=%s outcome=cancelled",
                tool_call.tool_name,
                tool_call.call_id,
            )
            self.stats.tool_calls_failed += 1
            yield ToolResultEvent(
                tool_name=tool_call.tool_name,
                tool_class=tool_call.tool_class,
                error=cancel,
                cancelled=True,
                tool_call_id=tool_call.call_id,
            )
            async for ev in self._finalize_cancelled_tool(
                tool_call, tool_input, cancel, tool_started=tool_started
            ):
                yield ev
            raise

        except Exception as exc:
            # One prefix for both: the model reads `error`, the client `display`.
            failure = f"{tool_instance.get_name()} failed: "
            error_msg = f"<{TOOL_ERROR_TAG}>{failure}{exc}</{TOOL_ERROR_TAG}>"
            if isinstance(exc, ToolPermissionError):
                logger.info(
                    "Tool call denied tool=%s tool_call_id=%s outcome=denied",
                    tool_call.tool_name,
                    tool_call.call_id,
                )
                self.stats.tool_calls_agreed -= 1
                self.stats.tool_calls_rejected += 1
            else:
                logger.warning(
                    "Tool call failed tool=%s tool_call_id=%s outcome=error error=%s",
                    tool_call.tool_name,
                    tool_call.call_id,
                    type(exc).__name__,
                )
                self.stats.tool_calls_failed += 1
            yield ToolResultEvent(
                tool_name=tool_call.tool_name,
                tool_class=tool_call.tool_class,
                error=error_msg,
                error_display=(
                    f"{failure}{exc.display}" if isinstance(exc, ToolError) else None
                ),
                tool_call_id=tool_call.call_id,
            )
            async for ev in self._run_post_tool_and_finalize(
                tool_call,
                PostToolFinalization(
                    tool_input=tool_input,
                    tool_status="failure",
                    response_status="failure",
                    tool_error=str(exc),
                    initial_text=error_msg,
                ),
            ):
                yield ev

    async def _invoke_tool(
        self,
        tool_call: ResolvedToolCall,
        tool_instance: BaseTool,
        tool_input: dict[str, Any],
        decision: ToolDecision,
    ) -> AsyncGenerator[ToolResultEvent | ToolStreamEvent | HookEvent]:
        self.stats.tool_calls_agreed += 1

        snapshot = await asyncio.to_thread(
            tool_instance.get_file_snapshot, tool_call.validated_args
        )
        if snapshot is not None:
            self.checkpoint_recorder.add_snapshot(snapshot)

        start_time = time.perf_counter()
        logger.debug(
            "Tool call starting tool=%s tool_call_id=%s",
            tool_call.tool_name,
            tool_call.call_id,
        )
        result_model = None
        async for item in tool_instance.invoke(
            ctx=InvokeContext(
                tool_call_id=tool_call.call_id,
                agent_manager=self.agent_manager,
                session_dir=self.session_logger.session_dir,
                launch_context=self.launch_context,
                interaction_requests=self._request_broker,
                subagent_runner=self._turn.subagent_runner,
                subagent_manager=(
                    None
                    if self._is_subagent
                    else cast(SubagentManagementPort, self._turn.subagent_runner)
                ),
                plan_file_path=self._plan_session.plan_file_path,
                request_clear_context_callback=self._request_clear_context,
                skill_manager=self.skill_manager,
                is_skill_loaded=self._skill_already_loaded,
                scratchpad_dir=self.scratchpad_dir,
                hook_config_result=self._hook_config_result,
                session_id=self.session_id,
                mcp_pool=self._mcp_pool,
                tool_io=self._turn.tool_io,
                is_subagent=self._is_subagent,
            ),
            **tool_call.args_dict,
        ):
            if isinstance(item, ToolStreamEvent):
                yield item
            else:
                result_model = item

        duration = time.perf_counter() - start_time
        if result_model is None:
            raise ToolError("Tool did not yield a result")

        result_dict = result_model.model_dump(mode="json")
        text = "\n".join(f"{k}: {v}" for k, v in result_dict.items())
        extra = tool_instance.get_result_extra(result_model)
        if extra:
            text += "\n\n" + extra

        result_cancelled = (
            isinstance(result_model, CancellableToolResult) and result_model.cancelled
        )
        images = (
            None if result_cancelled else tool_instance.get_result_images(result_model)
        )
        result_event = ToolResultEvent(
            tool_name=tool_call.tool_name,
            tool_class=tool_call.tool_class,
            result=result_model,
            cancelled=result_cancelled,
            duration=duration,
            tool_call_id=tool_call.call_id,
        )
        result_event = result_event.model_copy(
            update={
                "presentation": ToolUIDataAdapter(
                    tool_call.tool_class, harness_files=self.harness_files
                ).get_result_presentation(result_event)
            }
        )
        yield result_event
        async for ev in self._run_post_tool_and_finalize(
            tool_call,
            PostToolFinalization(
                tool_input=tool_input,
                tool_status="cancelled" if result_cancelled else "success",
                response_status="success",
                tool_output=result_dict,
                tool_presentation=result_event.presentation,
                images=images,
                duration_ms=duration * 1000.0,
                initial_text=text,
            ),
        ):
            yield ev
        self.stats.tool_calls_succeeded += 1
        logger.info(
            "Tool call completed tool=%s tool_call_id=%s duration_ms=%d outcome=%s",
            tool_call.tool_name,
            tool_call.call_id,
            int(duration * 1000),
            "cancelled" if result_cancelled else "success",
        )

    async def _should_execute_tool(
        self, tool: BaseTool, args: BaseModel
    ) -> ToolDecision:
        tool_name = tool.get_name()
        # Always evaluate invocation guards, even with old grants/bypass flags.
        ctx = tool.resolve_permission(args)
        config_perm = self.tool_manager.get_tool_config(tool_name).permission
        if config_perm == ToolPermission.NEVER or (
            ctx is not None and ctx.permission == ToolPermission.NEVER
        ):
            return ToolDecision(
                verdict=ToolExecutionResponse.SKIP,
                approval_type=ToolPermission.NEVER,
                feedback=(ctx.reason if ctx is not None else None)
                or f"Tool '{tool_name}' is disabled by policy",
            )
        # ASK is no longer an execution approval mechanism. Positive accident
        # guards must return NEVER at their resolver; opaque extensions are
        # trusted code, not proven filesystem-contained implementations.
        return ToolDecision(
            verdict=ToolExecutionResponse.EXECUTE, approval_type=ToolPermission.ALWAYS
        )

    def _handle_tool_response(
        self,
        tool_call: ResolvedToolCall,
        text: str,
        persisted_result: PersistedToolResult | None = None,
        images: list[ImageAttachment] | None = None,
    ) -> None:
        # Single choke point for every tool response entering the message list:
        # redacting here covers both model input and the persisted session log,
        # including the structured tool_result payload carried alongside.
        text = secret_redaction.redact(text)
        if persisted_result is not None:
            persisted_result = secret_redaction.redact_persisted_result(
                persisted_result
            )
        message = LLMMessage.model_validate(
            self.format_handler.create_tool_response_message(tool_call, text)
        )
        updates: dict[str, Any] = {}
        if persisted_result is not None:
            updates["tool_result"] = persisted_result
        if images is not None:
            updates["images"] = images
        self._append_tool_response(
            tool_call.call_id,
            message.model_copy(update=updates) if updates else message,
        )

    def _tool_failure_event(
        self, tool_call: ResolvedToolCall, error_msg: str, cancelled: bool = False
    ) -> ToolResultEvent:
        """Create a ToolResultEvent for a failed tool and record the failure."""
        self._handle_tool_response(tool_call, error_msg)
        return ToolResultEvent(
            tool_name=tool_call.tool_name,
            tool_class=tool_call.tool_class,
            error=error_msg,
            cancelled=cancelled,
            tool_call_id=tool_call.call_id,
        )

    def _begin_tool_response_ordering(self) -> None:
        """Stage tool-response messages so they append in tool-call order.

        Providers require ``tool_response`` messages to follow the order of
        the ``tool_calls`` that produced them, but batched tools complete out
        of order and their events must keep streaming in completion order.
        While ordering is active, ``_append_tool_response`` buffers each
        message and appends index ``i`` only once index ``i - 1`` has
        appended, so streaming stays responsive and the message list stays
        deterministic.
        """
        ordered_calls: list[ToolCall] = []
        for message in reversed(self.messages):
            if message.role == Role.assistant and message.tool_calls:
                ordered_calls = list(message.tool_calls)
                break
        order: dict[str, int] = {}
        for index, tool_call in enumerate(ordered_calls):
            if tool_call.id and tool_call.id not in order:
                order[tool_call.id] = index
        self._tool_response_order = order
        self._staged_tool_responses = {}
        self._next_tool_response_index = 0

    def _append_tool_response(self, call_id: str, message: LLMMessage) -> None:
        """Append a tool-response message, in call order during a batch."""
        order = self._tool_response_order
        index = order.get(call_id) if order is not None else None
        if index is None:
            # No active batch (or a call outside it): append directly.
            self.messages.append(message)
            return
        self._staged_tool_responses.setdefault(index, []).append(message)
        while (
            staged := self._staged_tool_responses.pop(
                self._next_tool_response_index, None
            )
        ) is not None:
            self.messages.extend(staged)
            self._next_tool_response_index += 1

    def _end_tool_response_ordering(self) -> None:
        """Flush staged responses and close the ordering window.

        A call aborted before producing any response leaves a gap here;
        ``_fill_missing_tool_responses`` backfills it before the next
        completion, exactly as it did before ordering existed.
        """
        order = self._tool_response_order
        staged = self._staged_tool_responses
        self._tool_response_order = None
        self._staged_tool_responses = {}
        self._next_tool_response_index = 0
        if order is None:
            return
        for index in sorted(staged):
            self.messages.extend(staged[index])

    def _completion_inputs(
        self,
        *,
        model: ModelConfig,
        messages: Sequence[LLMMessage],
        tools: list[AvailableTool] | None,
        tool_choice: StrToolChoice | AvailableTool | None,
        call_type: str | None = None,
    ) -> CompletionInputs:
        provider = self.config.get_provider_for_model(model)
        return CompletionInputs(
            model=model,
            provider_name=provider.name,
            emits_finish_reason=provider.emits_finish_reason,
            messages=tuple(messages),
            tools=tools,
            tool_choice=tool_choice,
            extra_headers=self._get_extra_headers(provider),
            metadata=self._build_backend_metadata(call_type),
            max_tokens=self._max_tokens,
        )

    def _call_resources(self, backend: BackendLike) -> CallResources:
        return CallResources(
            backend=backend,
            stats=self.stats,
            process_message=self.format_handler.process_api_response_message,
        )

    def _append_transcript(self, outcome: TranscriptAppend) -> None:
        message = outcome.message
        if outcome.committed_model is not None:
            message = message.model_copy(
                update={"deployment_identity": outcome.committed_model.model_dump()}
            )
        self.messages.append(message)

    def completion_metadata_mark(self) -> tuple[int, int]:
        """Return a cursor for completion visibility produced after this point."""
        return len(self._completion_providers), len(self._failover_switches)

    def completion_metadata_since(self, mark: tuple[int, int]) -> dict[str, JsonValue]:
        """Return additive result metadata for completions after ``mark``."""
        providers_mark, switches_mark = mark
        return cast(
            dict[str, JsonValue],
            {
                "providers_used": [
                    list(providers)
                    for providers in self._completion_providers[providers_mark:]
                ],
                "switch_notices": self._failover_switches[switches_mark:],
            },
        )

    def _record_completion_providers(self, providers: Sequence[str]) -> None:
        self._completion_providers.append(tuple(providers))

    def _record_failover_switch(
        self, *, base_model: str, old_provider: str, new_provider: str, reason: str
    ) -> None:
        self._failover_switches.append({
            "base_model": base_model,
            "old_provider": old_provider,
            "new_provider": new_provider,
            "reason": reason,
        })

    def count_history_images_unsupported_by_active_model(self) -> int:
        try:
            active_model = self.config.get_active_model()
        except ValueError:
            return 0
        if active_model.supports_images:
            return 0
        return sum(1 for message in self._current_model_context() if message.images)

    def _failover_candidates(
        self,
        messages: Sequence[LLMMessage],
        attempted_model: ModelConfig,
        *,
        call_type: str | None = None,
    ) -> tuple[EligibleDeployment, ...]:
        config = self.config
        snapshot = config.catalog_snapshot
        committed = self.committed_model
        if snapshot is None or committed is None:
            return ()
        result = eligible_deployments(
            snapshot=snapshot,
            committed=committed,
            registry=self.config_orchestrator.availability_registry,
            config=config,
            history=reorder_for_tool_adjacency(select_model_context(messages)),
            thinking=attempted_model.thinking,
            compaction_base=config.compaction_model or None,
            compaction_only=call_type == "secondary_call",
            allowed_models=config.allowed_models,
        )
        return tuple(result.candidates)

    def _backend_for_attempt(
        self, model: ModelConfig, budget: RequestRetryBudget
    ) -> BackendLike:
        if self._injected_backend is not None:
            return self._injected_backend
        provider = self.config.get_provider_for_model(model)
        return create_backend(
            provider=provider,
            on_retry=self.notice_retry,
            timeout=self.config.api_timeout,
            retry_max_elapsed_time=self.config.api_retry_max_elapsed_time,
            retry_budget=budget,
            connect_timeout=self.config.api_connect_timeout,
            write_timeout=self.config.api_write_timeout,
            pool_timeout=self.config.api_pool_timeout,
            enable_system_trust_store=self.config.enable_system_trust_store,
        )

    def _attempt_model(
        self, resolved: object, *, call_type: str | None, thinking: str | None = None
    ) -> ModelConfig:
        """Materialize the completion model for a candidate provider.

        A dedicated compaction base follows the selected conversation provider;
        eligibility has already established that such a deployment exists.
        """
        from chartreux.core.model_catalog.resolver import ResolvedModel, resolver_for

        assert isinstance(resolved, ResolvedModel)
        config = self.config
        target = resolved
        if call_type == "secondary_call" and config.compaction_model:
            base = resolver_for(config).canonicalize(config.compaction_model)
            definition = config.catalog_snapshot.catalog.models[base]  # type: ignore[union-attr]
            deployment = next(
                item
                for item in definition.deployments
                if item.provider == resolved.deployment.provider and not item.disabled
            )
            target = ResolvedModel(
                base,
                definition,
                deployment,
                config.catalog_snapshot.catalog.providers[deployment.provider],  # type: ignore[union-attr]
                config.catalog_snapshot.revision,  # type: ignore[union-attr]
            )
        return target.materialize(
            auto_compact_threshold=config.auto_compact_threshold,
            thinking=(
                thinking
                if thinking is not None
                else config.thinking_overrides.get(target.base_model)
            ),
        )

    def _release_unattempted_probes(
        self, candidates: Sequence[EligibleDeployment], attempted: set[str]
    ) -> None:
        registry = self.config_orchestrator.availability_registry
        for candidate in candidates:
            if (
                candidate.recovery_probe
                and candidate.resolved.deployment.provider not in attempted
            ):
                assert candidate.recovery_probe_token is not None
                registry.release_probe(
                    candidate.resolved.base_model,
                    candidate.resolved.deployment.provider,
                    candidate.recovery_probe_token,
                )

    def _attempt_retry_budget(
        self, budget: RequestRetryBudget, *, has_alternative: bool
    ) -> RequestRetryBudget:
        """Use the shared deadline for a lone deployment, otherwise fail over now."""
        if not has_alternative:
            return budget
        return RequestRetryBudget(0.0)

    def _rollback_attempt(
        self,
        publication: object,
        previous_identity: CommittedModelIdentity | None,
        candidate: EligibleDeployment,
    ) -> None:
        restored = publication.rollback()  # type: ignore[attr-defined]
        try:
            if restored:
                self.committed_model = previous_identity
                self.config.attach_committed_model(previous_identity)
                self.install_launch_metadata()
        finally:
            if candidate.recovery_probe:
                assert candidate.recovery_probe_token is not None
                self.config_orchestrator.availability_registry.release_probe(
                    candidate.resolved.base_model,
                    candidate.resolved.deployment.provider,
                    candidate.recovery_probe_token,
                )

    def _finalize_attempt(
        self, publication: object, candidate: EligibleDeployment
    ) -> None:
        self.install_launch_metadata()
        self.config_orchestrator.availability_registry.record_success(
            candidate.resolved.base_model, candidate.resolved.deployment.provider
        )
        publication.finalize(whole_turn_active=True)  # type: ignore[attr-defined]

    async def _attempt_completion(  # noqa: PLR0912, PLR0914, PLR0915
        self,
        *,
        messages: Sequence[LLMMessage],
        tools: list[AvailableTool] | None,
        tool_choice: StrToolChoice | AvailableTool | None,
        call_type: str | None,
        transcript: bool,
        streaming: bool,
        fallback_model: ModelConfig,
    ) -> AsyncGenerator[LLMChunk]:
        """Run one shared preparation/publication/rollback attempt lifecycle."""
        candidates = self._failover_candidates(
            messages, fallback_model, call_type=call_type
        )
        if not candidates:  # noqa: PLR1702
            inputs = self._completion_inputs(
                model=fallback_model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                call_type=call_type,
            )
            provider_name = self.config.get_provider_for_model(fallback_model).name
            try:
                with self._backend_lifetime.borrow() as backend:
                    resources = self._call_resources(backend)
                    if streaming:
                        stream = self._llm_gateway.chat_streaming(
                            inputs, resources, transcript=self._append_transcript
                        )
                        async with contextlib.aclosing(stream):
                            async for chunk in stream:
                                yield chunk
                    elif transcript:
                        yield await self._llm_gateway.chat(
                            inputs, resources, transcript=self._append_transcript
                        )
                    else:
                        yield await self._llm_gateway.complete(inputs, resources)
            finally:
                self._record_completion_providers((provider_name,))
            return

        budget = RequestRetryBudget(self.config.api_retry_max_elapsed_time)
        registry = self.config_orchestrator.availability_registry
        last_error: BaseException | None = None
        attempted: set[str] = set()
        providers_used: list[str] = []
        try:  # noqa: PLR1702
            for candidate_index, candidate in enumerate(candidates):
                resolved = candidate.resolved
                provider_name = resolved.deployment.provider
                if provider_name in attempted:
                    continue

                replacement: BackendLike | None = None
                try:
                    model = self._attempt_model(
                        resolved, call_type=call_type, thinking=fallback_model.thinking
                    )
                    attempt_budget = self._attempt_retry_budget(
                        budget, has_alternative=candidate_index < len(candidates) - 1
                    )
                    replacement = self._backend_for_attempt(model, attempt_budget)
                    inputs = self._completion_inputs(
                        model=model,
                        messages=messages,
                        tools=tools,
                        tool_choice=tool_choice,
                        call_type=call_type,
                    )
                except BaseException:
                    if replacement is not None:
                        with contextlib.suppress(Exception):
                            await replacement.__aexit__(None, None, None)
                    raise

                if replacement is None:
                    raise AgentLoopLLMResponseError(
                        "Backend preparation produced no backend for publication"
                    )

                previous_identity = self.committed_model

                def transcript_sink(
                    outcome: TranscriptAppend,
                    identity: CommittedModelIdentity = resolved.identity,
                ) -> None:
                    self._append_transcript(replace(outcome, committed_model=identity))

                publication: BackendPublish | None = None
                semantic_delta = False
                try:
                    publication = self._backend_lifetime.publish_reversible(replacement)
                    attempted.add(provider_name)
                    providers_used.append(provider_name)
                    self.committed_model = resolved.identity
                    self.config.attach_committed_model(self.committed_model)
                    with self._backend_lifetime.borrow() as backend:
                        resources = self._call_resources(backend)
                        if streaming:
                            stream = self._llm_gateway.chat_streaming(
                                inputs, resources, transcript=transcript_sink
                            )
                            async with contextlib.aclosing(stream):
                                async for chunk in stream:
                                    message = chunk.message
                                    semantic_delta = semantic_delta or bool(
                                        message.content
                                        or message.reasoning_content
                                        or message.tool_calls
                                    )
                                    yield chunk
                        elif transcript:
                            yield await self._llm_gateway.chat(
                                inputs, resources, transcript=transcript_sink
                            )
                        else:
                            yield await self._llm_gateway.complete(inputs, resources)
                    self._finalize_attempt(publication, candidate)
                except (asyncio.CancelledError, GeneratorExit):
                    if publication is not None:
                        self._rollback_attempt(
                            publication, previous_identity, candidate
                        )
                    raise
                except Exception as error:
                    if publication is not None:
                        self._rollback_attempt(
                            publication, previous_identity, candidate
                        )
                    last_error = error
                    failure = classify(error)
                    if failure.failover_eligible:
                        registry.record_failure(
                            resolved.base_model, provider_name, failure
                        )
                    self._backend_lifetime.drain(whole_turn_active=False)
                    if (
                        semantic_delta
                        or not failure.failover_eligible
                        or budget.exhausted
                    ):
                        raise
                    continue
                else:
                    if (
                        previous_identity is not None
                        and previous_identity.provider != provider_name
                    ):
                        reason = (
                            classify(last_error).category.value
                            if last_error
                            else "availability"
                        )
                        self._record_failover_switch(
                            base_model=resolved.base_model,
                            old_provider=previous_identity.provider,
                            new_provider=provider_name,
                            reason=reason,
                        )
                        logger.warning(
                            "Model deployment switched base=%s old_provider=%s new_provider=%s reason=%s",
                            resolved.base_model,
                            previous_identity.provider,
                            provider_name,
                            reason,
                        )
                    return
            assert last_error is not None
            raise last_error
        finally:
            self._record_completion_providers(providers_used)
            self._release_unattempted_probes(candidates, attempted)

    async def _attempt_nonstreaming(
        self,
        *,
        messages: Sequence[LLMMessage],
        tools: list[AvailableTool] | None,
        tool_choice: StrToolChoice | AvailableTool | None,
        call_type: str | None,
        transcript: bool,
        fallback_model: ModelConfig,
    ) -> LLMChunk:
        result: LLMChunk | None = None
        async for chunk in self._attempt_completion(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            call_type=call_type,
            transcript=transcript,
            streaming=False,
            fallback_model=fallback_model,
        ):
            result = chunk
        if result is None:
            raise AgentLoopLLMResponseError("Completion returned no result")
        return result

    async def _complete(
        self,
        *,
        model: ModelConfig,
        messages: Sequence[LLMMessage],
        tools: list[AvailableTool] | None,
        tool_choice: StrToolChoice | AvailableTool | None,
        call_type: str | None,
    ) -> LLMChunk:
        return await self._attempt_nonstreaming(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            call_type=call_type,
            transcript=False,
            fallback_model=model,
        )

    async def _chat(
        self, model_override: ModelConfig | None = None, *, call_type: str | None = None
    ) -> LLMChunk:
        active_model = model_override or self.config.get_active_model()
        return await self._attempt_nonstreaming(
            messages=self.messages,
            tools=self.format_handler.get_available_tools(self.tool_manager),
            tool_choice=self.format_handler.get_tool_choice(),
            call_type=call_type,
            transcript=True,
            fallback_model=active_model,
        )

    async def _chat_streaming(self) -> AsyncGenerator[LLMChunk]:
        active_model = self.config.get_active_model()
        stream = self._attempt_completion(
            messages=self.messages,
            tools=self.format_handler.get_available_tools(self.tool_manager),
            tool_choice=self.format_handler.get_tool_choice(),
            call_type=None,
            transcript=True,
            streaming=True,
            fallback_model=active_model,
        )
        async with contextlib.aclosing(stream):
            async for chunk in stream:
                yield chunk

    def _clean_message_history(self) -> None:
        ACCEPTABLE_HISTORY_SIZE = 2
        if len(self.messages) < ACCEPTABLE_HISTORY_SIZE:
            return
        self._fill_missing_tool_responses()

    def _fill_missing_tool_responses(self) -> None:
        i = 1
        while i < len(self.messages):  # noqa: PLR1702
            msg = self.messages[i]

            if msg.role == "assistant" and msg.tool_calls:
                expected_responses = len(msg.tool_calls)

                if expected_responses > 0:
                    responded_ids: set[str] = set()
                    j = i + 1
                    while j < len(self.messages) and self.messages[j].role == "tool":
                        tool_call_id = self.messages[j].tool_call_id
                        if tool_call_id is not None:
                            responded_ids.add(tool_call_id)
                        j += 1

                    if len(responded_ids) < expected_responses:
                        insertion_point = j

                        for tool_call_data in msg.tool_calls:
                            if (tool_call_data.id or "") in responded_ids:
                                continue

                            empty_response = LLMMessage(
                                role=Role.tool,
                                tool_call_id=tool_call_data.id or "",
                                name=(
                                    (tool_call_data.function.name or "")
                                    if tool_call_data.function
                                    else ""
                                ),
                                content=str(
                                    get_user_cancellation_message(
                                        CancellationReason.TOOL_NO_RESPONSE
                                    )
                                ),
                            )

                            if insertion_point < len(self.messages):
                                # An interior insertion invalidates the cursor's
                                # boundary proof, even if repeated records make the
                                # shifted boundary slot look unchanged.
                                self.session_logger.invalidate_transcript_cursor()
                            self.messages.insert(insertion_point, empty_response)
                            insertion_point += 1

                    i = i + 1 + expected_responses
                    continue

            i += 1

    async def _reset_session(self, keep_parent: bool = True) -> None:
        old_session_id = self.session_id
        suffix = extract_suffix(self.session_id)
        session_id = generate_session_id(suffix=suffix)
        lease_root = (
            self._session_lease.path.parent.parent
            if self._session_lease is not None
            else Path(self.config.session_logging.save_dir)
        )
        lease = (
            await asyncio.to_thread(SessionLease(lease_root, session_id).acquire)
            if self.config.session_logging.enabled
            else None
        )
        parent_session_id = (
            self.parent_session_id
            if keep_parent and self._is_subagent
            else old_session_id
            if keep_parent
            else None
        )
        try:
            self.session_logger.reset_session(
                session_id, parent_session_id=parent_session_id
            )
        except BaseException:
            if lease is not None:
                await asyncio.to_thread(lease.release)
            raise
        self._session_generation += 1
        self.session_id = session_id
        self.parent_session_id = parent_session_id
        self.replace_session_lease(lease)
        self._reset_title_state()

    def replace_session_lease(self, lease: SessionLease | None) -> None:
        previous = self._session_lease
        self._session_lease = lease
        if previous is not None:
            previous.release()

    def prepare_scratchpad_for_session(self, session_id: str) -> Path | None:
        """Create the scratchpad needed to rebind without changing live state."""
        return None if self._is_subagent else init_scratchpad(session_id)

    def rebind_to_session(
        self,
        session_id: str,
        session_dir: Path,
        loaded_messages: list[LLMMessage],
        *,
        session_metadata: SessionMetadata,
        parent_session_id: str | None = None,
        stats: AgentStats | None = None,
        prepared_scratchpad: Path | None | _PrepareScratchpad = _PREPARE_SCRATCHPAD,
    ) -> None:
        """Swap session identity in-place, reusing expensive runtime infrastructure.

        Kept (session-independent): MCP pools, the tool and skill
        registries, git context, config, and the backend client.

        Reimported from the resumed session: session ID, parent, message history,
        stats, and the session-logger binding.

        Reset so nothing leaks across the session boundary: tool-permission
        approvals, checkpoint/rewind state, the plan session, per-turn middleware
        and tool state, and the scratchpad directory.

        Atomicity: the only intentionally fail-able work (scratchpad creation)
        runs before any mutation. The commit section is designed to be infallible
        — pure assignments and resets — so a resume either fully applies or leaves
        the loop untouched. A bug in any commit step would leave the loop
        half-rebound; callers should treat an unexpected raise as fatal.
        """
        # Prepare — no mutation of the live loop.
        previous_scratchpad = self.scratchpad_dir
        scratchpad_dir = (
            self.prepare_scratchpad_for_session(session_id)
            if isinstance(prepared_scratchpad, _PrepareScratchpad)
            else prepared_scratchpad
        )

        # Commit — assignments and in-place resets only, from here on infallible.
        self._session_generation += 1
        self.session_id = session_id
        self.parent_session_id = parent_session_id
        self.scratchpad_dir = scratchpad_dir
        self.tool_manager.set_scratchpad_dir(scratchpad_dir)
        self.session_logger.apply_resumed_session(
            session_id, session_dir, session_metadata
        )
        # Atomically preserve any system prompt the deferred-init thread may
        # have inserted between snapshot and commit, instead of snapshotting
        # system messages outside the lock and racing update_system_prompt.
        self.messages.reset_preserving_system(loaded_messages)
        if stats is not None:
            self.stats = stats
        else:
            self.stats = AgentStats.create_fresh(self.stats)
            self._apply_active_model_pricing()
        self._reset_session_scoped_state()
        self._restore_todo_state(loaded_messages)
        cleanup_scratchpad(previous_scratchpad)

    def _restore_todo_state(self, messages: Sequence[LLMMessage]) -> None:
        try:
            todo = self.tool_manager.get("todo")
        except NoSuchToolError:
            return
        todo.state = TodoState.replay(messages)

    def _apply_active_model_pricing(self) -> None:
        try:
            active_model = self.config.get_active_model()
        except ValueError:
            return
        self.stats.update_pricing(
            active_model.input_price,
            active_model.output_price,
            active_model.cached_input_price,
        )

    def _reset_session_scoped_state(self) -> None:
        # Clear any duration the picker recorded so it doesn't leak into the
        # resumed session. ``_init_start_time`` is intentionally kept: the
        # metric measures ``__init__ -> ready``.
        self._init_duration_pending = False
        self._last_init_duration_ms = None
        self.checkpoint_recorder.reset()
        self.middleware_pipeline.reset()
        self.tool_manager.reset_all()
        self._plan_session = PlanSession()
        self._user_plan = None
        self._pending_injected_messages = []
        self._pending_clear_context = False
        self._current_user_message_id = None
        self._is_user_prompt_call = False
        self._reactive_recovery_used = False
        self._reset_title_state()

    @requires_init
    async def clear_history(self) -> None:
        await self.session_logger.save_interaction(
            self.messages, self.stats, self.config, self.tool_manager, None
        )
        self.messages.reset(self.messages[:1])

        self.stats = AgentStats.create_fresh(self.stats)
        self.stats.trigger_listeners()
        self._apply_active_model_pricing()

        self.middleware_pipeline.reset()
        self.tool_manager.reset_all()
        await self._reset_session(keep_parent=False)

    @requires_init
    async def compact(self, extra_instructions: str = "") -> str:
        # Summary generation and the context boundary live in the manager; the loop
        # keeps the surrounding lifecycle (clean history, save, middleware).
        try:
            self._clean_message_history()
            await self._save_messages()
            summary = await self.compaction_manager.compact(extra_instructions)
            self.middleware_pipeline.reset(reset_reason=ResetReason.COMPACT)
            # A compacted conversation reads very differently from its first
            # turn; force a title refresh at the next scheduling point.
            self._title_controller.mark_compaction()
            return summary
        except Exception:
            await self._save_messages()
            raise

    async def _request_clear_context(self) -> None:
        """Signal that the context should be cleared at the next turn boundary.

        The actual clear is deferred so the in-flight tool turn can finish
        appending its tool-result before ``self.messages`` is wiped.
        """
        self._pending_clear_context = True

    @requires_init
    async def relocate(self, cwd: Path) -> None:
        """Move the session to *cwd*, or leave it exactly where it was.

        Refused while a turn is active. Tool calls in a batch run as concurrent
        tasks, so a move alongside a file or shell call would leave that call
        writing to the old directory, and swapping the objects it holds cannot
        stop one already running. A subagent runs inside its parent's turn, so
        this covers those too.

        Refused as well while an operation holds the session. A turn is not
        the only thing that reads the working directory or acts on its
        repository, and the rest are invisible to the turn check.

        Raises:
            AgentLoopStateError: Something is in flight, or *cwd* is not a
                directory.
        """
        if self._active_turn is not None:
            raise AgentLoopStateError("Cannot relocate while a turn is active")
        target = cwd.expanduser().resolve()
        if target == self.cwd:
            return
        if not target.is_dir():
            raise AgentLoopStateError(f"Not a directory: {target}")

        # Taken before the first await, and held across both. Checking once and
        # then awaiting would leave a window for an operation to start against a
        # directory that is moving under it.
        self._take_session("relocation")
        try:
            checkout = await asyncio.to_thread(self._destination_checkout, target)
            if checkout is None:
                raise AgentLoopStateError(
                    f"Not a worktree of this session's repository: {target}"
                )

            previous_files = self.harness_files
            previous_cwd = self.cwd
            moved = self.harness_files.moved_to(target)
            # The checkout rather than the position inside it. The project layer
            # finds `.chartreux` by walking up from the working directory and asks
            # whether the directory holding it is trusted, and trust resolves
            # upward too - so a grant on a subdirectory never reaches the root
            # the file sits at, and a subdirectory move would read no project
            # config at all. That is the case re-rooting exists for.
            #
            # Safe only because of the check above. Session trust outranks an
            # explicit untrust at the same path and reaches every descendant,
            # so granting it for an arbitrary directory would hand the session
            # a tree the user never opened. Restricted to a checkout of the
            # repository it is already working in, it inherits authority rather
            # than creating it.
            moved.trust_store.trust_for_session(checkout)
            try:
                await self._bind_workspace(target, moved)
            except BaseException:
                moved.trust_store.revoke_session_trust(checkout)
                await self._restore_workspace(previous_cwd, previous_files)
                raise
            # Release the grant a move took, and only that one. The trust store
            # is process-wide and counts grants, so releasing the departure
            # outright would take back one the session never made: at session
            # start for this session, or for another session sitting there.
            if self._trust_taken_by_move is not None:
                moved.trust_store.revoke_session_trust(self._trust_taken_by_move)
            self._trust_taken_by_move = checkout
        finally:
            self._release_session("relocation")

    def _destination_checkout(self, target: Path) -> Path | None:
        """The checkout *target* sits in, or None when it is not one to move to.

        The precondition the rest of the move rests on, and it has to be exactly
        this narrow. Anything wider grants trust over a tree the session never
        occupied; anything narrower strands a session opened in a subdirectory,
        since it moves into the matching subdirectory of the worktree and could
        not name its way back.

        So the destinations are the counterparts of the current position: the
        same relative path in the main checkout, and in each linked worktree.
        Not the parent checkout, which would widen a subdirectory session to the
        whole repository, and not a sibling directory, which the session has no
        claim on either.

        The checkout comes back with the answer because trust is granted on it
        rather than on the position: a `.chartreux` at the root is out of reach of a
        grant made on a subdirectory below it.
        """
        try:
            with WorktreeRepository.open(self.cwd) as repository:
                for worktree in repository.linked():
                    if worktree.path.resolve() == target:
                        return worktree.root.resolve()
                if repository.repository_counterpart == target:
                    return repository.root
        except GitError:
            return None
        return None

    def _project_config_layer(self) -> ProjectConfigLayer | None:
        """The live project config layer, or None when there is none to re-root.

        Absent when the project source is disabled, and when the orchestrator
        carries no layer stack at all, which is how the test doubles are built.
        """
        with contextlib.suppress(AttributeError, KeyError):
            layer = self.config_orchestrator.get_layer(ProjectConfigLayer.NAME)
            if isinstance(layer, ProjectConfigLayer):
                return layer
        return None

    async def _bind_workspace(
        self, cwd: Path, harness_files: HarnessFilesManager
    ) -> None:
        """Point everything rooted in the working directory at *cwd*."""
        self.cwd = cwd
        self.harness_files = harness_files
        # The logger holds its own copy and is what writes the directory into
        # session metadata, so a move that skipped it would be recorded as
        # never having happened.
        self.session_logger.relocated_to(cwd)
        if (layer := self._project_config_layer()) is not None:
            await layer.reroot(cwd)
            await self.config_orchestrator.reload()
        self.agent_manager.rebind(harness_files)
        # Rebuilds the tools, skills, prompt and hooks bound to the old
        # directory. Preparing happens off-thread and the commit is synchronous,
        # so that half is not observable partly applied. It runs last so those
        # objects are built from the config the destination resolves to.
        #
        # It returns without committing, and without raising, when a newer
        # reload supersedes it. The destination still wins: that newer reload
        # prepares from the live self.cwd, which is already the new one. What is
        # not guaranteed is that the rebuild has finished by the time relocate
        # returns. Treating supersession as failure would be worse, since the
        # rollback would then fight a reload that is legitimately in flight.
        await self.reload_with_initial_messages(reload_hooks=True)

    async def _restore_workspace(
        self, cwd: Path, harness_files: HarnessFilesManager
    ) -> None:
        """Undo a partly applied move.

        Failures are logged rather than raised. The caller is already unwinding
        the exception that says why the move did not happen, and replacing it
        with a second one would hide that.
        """
        self.cwd = cwd
        self.harness_files = harness_files
        self.session_logger.relocated_to(cwd)
        try:
            if (layer := self._project_config_layer()) is not None:
                await layer.reroot(cwd)
                await self.config_orchestrator.reload()
            self.agent_manager.rebind(harness_files)
        except Exception:
            logger.exception(
                "Failed to restore the workspace after a rejected move cwd=%s", cwd
            )

    @requires_init
    async def reload_with_initial_messages(
        self,
        max_turns: int | None = None,
        max_price: float | None = None,
        reset_middleware: bool = True,
        reload_hooks: bool = False,
        reload_config: bool = False,
    ) -> None:
        self._reload_generation += 1
        generation = self._reload_generation

        await self.session_logger.save_interaction(
            self.messages, self.stats, self.config, self.tool_manager, None
        )

        # A newer reload superseded us while we were saving; don't mutate state.
        if generation != self._reload_generation:
            return

        if max_turns is not None:
            self._max_turns = max_turns
        if max_price is not None:
            self._max_price = max_price
        self._ensure_remote_registries()

        if reload_config:
            # The orchestrator owns source reads and acceptance. Preparing runtime
            # consumers must finish before its paired snapshot becomes visible.
            prepared_reload: _PreparedReload | None = None

            async def prepare(candidate: ChartreuxConfigSchema) -> None:
                nonlocal prepared_reload
                prepared_reload = await asyncio.to_thread(
                    self._prepare_reload, candidate, reload_hooks
                )
                if generation != self._reload_generation:
                    raise AgentLoopStateError("Configuration reload was superseded")
                self._commit_reload(prepared_reload, reset_middleware)

            await self._config_orchestrator.reload(preflight=prepare)
            return

        target_config = self.config

        # Off-loop: skill discovery and system prompt I/O. reload() is awaited within a
        # turn, so that turn is suspended here -- nothing mutates the shared state this
        # reads, and the new objects are built locally before the commit below.
        prepared = await asyncio.to_thread(
            self._prepare_reload, target_config, reload_hooks
        )

        # A newer reload superseded us; let it own the commit.
        if generation != self._reload_generation:
            return

        # Synchronous swap: no await, so an in-flight turn can't observe a partial
        # update. Keep it that way -- don't make it async or move it off-thread.
        self._commit_reload(prepared, reset_middleware)

    async def _prepare_policy_replacement(
        self, *, source: str, tools: dict[str, Any], expected_token: object
    ) -> _PreparedPolicyReplacement:
        """Prepare local authority under the server's tree reservation."""
        self._require_policy_idle()
        previous_manager = self.tool_manager
        session_generation = self._session_generation
        staged = await self.config_orchestrator._stage_policy_replacement(
            source=source, tools=tools, expected_token=expected_token
        )
        return self._prepare_policy_manager(
            staged,
            expected_token,
            previous_manager,
            session_generation,
            self._inherited_restrictions,
        )

    async def _prepare_root_replacement(
        self, *, source: str, roots: dict[str, list[str]], expected_token: object
    ) -> _PreparedPolicyReplacement:
        self._require_policy_idle()
        previous_manager = self.tool_manager
        generation = self._session_generation
        staged = await self.config_orchestrator._prepare_root_replacement(
            source=source, roots=roots, expected_token=expected_token
        )
        return self._prepare_policy_manager(
            staged,
            expected_token,
            previous_manager,
            generation,
            self._inherited_restrictions,
        )

    def _require_policy_idle(self) -> None:
        if (
            self._active_turn is not None
            or self._tool_event_queue is not None
            or self._holders
        ):
            raise AgentLoopStateError("Policy replacement requires an idle loop")

    async def _prepare_policy_refresh(
        self,
        *,
        previous: SourceRestrictions,
        replacement: SourceRestrictions,
        expected_token: object,
        inherited_workspace: Workspace | None = None,
    ) -> _PreparedPolicyReplacement:
        self._require_policy_idle()
        previous_manager = self.tool_manager
        session_generation = self._session_generation
        inherited = tuple(
            replacement if source.identity == previous.identity else source
            for source in self._inherited_restrictions
        )
        if any(source.identity is None for source in inherited):
            raise ValueError("Policy source provenance unavailable")
        staged = await self.config_orchestrator._stage_policy_refresh(
            previous=previous, replacement=replacement, expected_token=expected_token
        )
        return self._prepare_policy_manager(
            staged,
            expected_token,
            previous_manager,
            session_generation,
            inherited,
            inherited_workspace=inherited_workspace,
        )

    def _prepare_policy_manager(
        self,
        staged: ConfigOrchestrator[ChartreuxConfigSchema],
        expected_token: object,
        previous_manager: ToolManager,
        session_generation: int,
        inherited: tuple[SourceRestrictions, ...],
        *,
        inherited_workspace: Workspace | None = None,
    ) -> _PreparedPolicyReplacement:
        inherited_workspace = inherited_workspace or self._inherited_workspace

        # During preparation use the candidate, after synchronous publication use
        # live authority. Stale preparations are rejected, never installed.
        def authority() -> ConfigOrchestrator[ChartreuxConfigSchema]:
            return (
                staged
                if self.config_orchestrator.accepted_token is expected_token
                else self.config_orchestrator
            )

        manager = ToolManager(
            lambda: authority().config,
            mcp_registry=self.mcp_registry,
            defer_mcp=True,
            discovery_source=previous_manager,
            restriction_getter=lambda: authority().restrictions,
            accepted_token_getter=lambda: authority().accepted_token,
            inherited_restrictions=inherited,
            inherited_workspace=inherited_workspace,
            inherited_plan_write_scopes=self._inherited_plan_write_scopes,
            parent_authority_getter=self._parent_authority_getter,
            parent_authority_revision_getter=self._parent_authority_revision_getter,
            cwd=self.cwd,
            harness_files=self.harness_files,
            scratchpad_dir=self.scratchpad_dir,
        )
        for name in manager.registered_tools:
            manager.get_tool_config(name)
        for name in previous_manager._instances.keys() & manager.available_tools.keys():
            manager.get(name)
        self.config_orchestrator._check_policy_token(expected_token)
        return _PreparedPolicyReplacement(
            self,
            session_generation,
            expected_token,
            staged,
            previous_manager,
            manager,
            inherited,
            inherited_workspace,
        )

    def _validate_policy_replacement(
        self, prepared: _PreparedPolicyReplacement
    ) -> None:
        """Validate every participant before the tree performs any publication."""
        if (
            prepared.owner is not self
            or prepared.session_generation != self._session_generation
            or prepared.previous_manager is not self.tool_manager
        ):
            raise ValueError("Stale policy runtime preparation")
        self._require_policy_idle()
        self.config_orchestrator._check_policy_commit(
            prepared.expected_token, prepared.orchestrator
        )

    def _commit_policy_replacement(
        self, prepared: _PreparedPolicyReplacement, *, retire: bool = True
    ) -> None:
        self._validate_policy_replacement(prepared)
        self.config_orchestrator._commit_policy_replacement(
            prepared.orchestrator, expected_token=prepared.expected_token
        )
        self._inherited_restrictions = prepared.inherited_restrictions
        self._inherited_workspace = prepared.inherited_workspace
        if retire:
            prepared.previous_manager._retire_authority()
        self.tool_manager = prepared.tool_manager
        self._authority_revision += 1

    def _prepare_reload(
        self,
        target_config: ChartreuxConfigSchema,
        reload_hooks: bool,
        *,
        mcp_registry: MCPRegistry | None = None,
    ) -> _PreparedReload:
        # Preserve the retained deployment before constructing a replacement backend.
        # An unchanged selection expression is provenance, not a request to resolve a
        # different deployment for the same base model.
        if (
            self.committed_model is not None
            and target_config.active_model == self._committed_selection
        ):
            from chartreux.core.model_catalog.resolver import resolver_for

            resolved = resolver_for(target_config).resolve_committed(
                self.committed_model, allowed_models=target_config.allowed_models
            )
            target_config.attach_committed_model(resolved.identity)
        # Load both configured prompts before any runtime mutation or client IO.
        _ = target_config.system_prompt, target_config.compaction_prompt
        # The candidate trust policy is carried by the backend being prepared.
        return self._prepare_reload_consumers(
            target_config, reload_hooks, mcp_registry=mcp_registry
        )

    def _prepare_reload_consumers(
        self,
        target_config: ChartreuxConfigSchema,
        reload_hooks: bool,
        *,
        mcp_registry: MCPRegistry | None = None,
        discovery_source: ToolManager | None = None,
        reuse_backend: bool = False,
    ) -> _PreparedReload:
        skills_adopted = self._skills_adopted
        config_source = _SwappableConfigSource(lambda: target_config)
        tool_manager = ToolManager(
            config_source.get,
            mcp_registry=mcp_registry
            if mcp_registry is not None
            else self.mcp_registry,
            defer_mcp=discovery_source is not None,
            discovery_source=discovery_source,
            restriction_getter=lambda: self.config_orchestrator.restrictions,
            accepted_token_getter=lambda: (
                self.config_orchestrator.accepted_token if config_source.live else None
            ),
            inherited_restrictions=self._inherited_restrictions,
            inherited_workspace=self._inherited_workspace,
            inherited_plan_write_scopes=self._inherited_plan_write_scopes,
            parent_authority_getter=self._parent_authority_getter,
            parent_authority_revision_getter=self._parent_authority_revision_getter,
            cwd=self.cwd,
            harness_files=self.harness_files,
            scratchpad_dir=self.scratchpad_dir,
        )
        skill_manager = SkillManager(
            config_source.get, harness_files=self.harness_files
        )
        candidate_orchestrator = self._config_orchestrator.copy(config=target_config)
        agent_manager = AgentManager(
            candidate_orchestrator, harness_files=self.harness_files
        )
        system_prompt = self._render_system_prompt(
            skill_manager, target_config, tool_manager, agent_manager
        )
        hook_config_result = (
            load_hooks_from_fs(harness_files=self.harness_files)
            if reload_hooks
            else self._hook_config_result
        )
        return _PreparedReload(
            config=target_config,
            backend=self.backend
            if reuse_backend
            else self.backend_factory(target_config),
            agent_manager=agent_manager,
            tool_manager=tool_manager,
            skill_manager=skill_manager,
            system_prompt=system_prompt,
            config_source=config_source,
            hook_config_result=hook_config_result,
            skills_adopted=skills_adopted,
        )

    def _commit_reload(  # noqa: PLR0915
        self, prepared: _PreparedReload, reset_middleware: bool
    ) -> None:
        # Resolve identity before publishing any candidate runtime state. A changed
        # selection is an explicit reconfiguration; an unchanged expression is only
        # provenance and must retain the already committed deployment.
        from chartreux.core.model_catalog.resolver import resolver_for

        resolver = resolver_for(prepared.config)
        identity: CommittedModelIdentity | None
        if (
            self.committed_model is not None
            and prepared.config.active_model == self._committed_selection
        ):
            identity = resolver.resolve_committed(
                self.committed_model, allowed_models=prepared.config.allowed_models
            ).identity
        elif (
            self.committed_model is None
            and self._model_choice_pending()
            and not prepared.config.active_model
        ):
            # Committed-model recovery: an unrelated config write must not
            # silently commit the default. Stay unpinned; only an explicit
            # active_model selection (the branch below) opens the gate.
            identity = None
        else:
            identity = resolver.resolve(
                prepared.config.active_model
                or prepared.config.resolve_default_model_alias(),
                allowed_models=prepared.config.allowed_models,
            ).identity
        prepared.config.attach_committed_model(identity)
        # Finish fallible rendering and hook construction before retiring authority
        # or replacing any runtime object. A failed candidate leaves the old loop live.
        adopt_skills = prepared.skills_adopted == self._skills_adopted
        system_prompt = (
            prepared.system_prompt
            if adopt_skills
            else self._render_system_prompt(
                self.skill_manager,
                prepared.config,
                prepared.tool_manager,
                prepared.agent_manager,
            )
        )
        hooks_manager = (
            HooksManager(prepared.hook_config_result.hooks, cwd=self.cwd)
            if prepared.hook_config_result is not None
            else None
        )
        middleware = (
            self._build_middleware_pipeline(prepared.config)
            if reset_middleware
            else self.middleware_pipeline
        )
        # The prepared backend owns its candidate trust policy; publishing it
        # requires no process-global TLS mutation.
        # Now that the profile is live, the prepared managers should track the live
        # config so later refreshes (refresh_config) propagate to them.
        prepared.config_source.point_to(lambda: self.config)
        prepared.agent_manager._orchestrator = self._config_orchestrator
        previous_identity = self.committed_model
        previous_selection = self._committed_selection
        publication = self._backend_lifetime.publish_reversible(prepared.backend)
        try:
            self.committed_model = identity
            self._committed_selection = prepared.config.active_model
            self.config.attach_committed_model(self.committed_model)
            self.install_launch_metadata()
        except BaseException:
            publication.rollback()
            self.committed_model = previous_identity
            self._committed_selection = previous_selection
            self.config.attach_committed_model(previous_identity)
            with contextlib.suppress(Exception):
                self.install_launch_metadata()
            raise
        publication.finalize(whole_turn_active=self._active_turn is not None)
        self._retire_mcp_pool()
        self.agent_manager = prepared.agent_manager
        self.tool_manager._retire_authority()
        self.tool_manager = prepared.tool_manager
        self._authority_revision += 1
        self.scrub_policy = secret_redaction.ScrubPolicy.from_config(prepared.config)
        secret_redaction.register_session_policy(self, self.scrub_policy)
        if prepared.config.mcp_servers:
            self._mcp_pool = self._create_mcp_pool()
        if adopt_skills:
            self.skill_manager = prepared.skill_manager
        self.messages.update_system_prompt(system_prompt)
        self._hook_config_result = prepared.hook_config_result
        self._hooks_manager = hooks_manager
        self.hook_config_issues = (
            prepared.hook_config_result.issues
            if prepared.hook_config_result is not None
            else []
        )
        self.hooks_count = (
            len(prepared.hook_config_result.hooks)
            if prepared.hook_config_result is not None
            else 0
        )

        if len(self.messages) == 1:
            self.stats.reset_context_state()

        try:
            active_model = prepared.config.get_active_model()
            self.stats.update_pricing(
                active_model.input_price,
                active_model.output_price,
                active_model.cached_input_price,
            )
        except ValueError:
            pass

        self.middleware_pipeline = middleware
        # The value cache remains process-wide; reload invalidates stored/keyring
        # values. Already inherited child environments require pool retirement.
        secret_redaction.reset_cache()
