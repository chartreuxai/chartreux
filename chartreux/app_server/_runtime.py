from __future__ import annotations

import asyncio
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
import threading
from typing import TYPE_CHECKING, Literal

from pydantic import ValidationError

from chartreux import __version__
from chartreux.app_server._host import HostRequestHandler
from chartreux.app_server._projection import (
    project_config_view,
    project_skill_summaries,
)
from chartreux.app_server._session_backend_port import SessionBackendHost
from chartreux.app_server._session_backend_services import SessionBackendServices
from chartreux.app_server._session_model import (
    active_model_is_pinned,
    clear_session_active_model_override,
    config_active_model,
    set_session_active_model_override,
)
from chartreux.app_server.client import AppServerClient
from chartreux.app_server.client_tools import ClientToolHandler
from chartreux.app_server.models import (
    AgentStatsSnapshot,
    ConfigIssue,
    MCPState,
    ToolSummary,
    TurnErrorCode,
)
from chartreux.app_server.protocol import (
    ClientCapabilities,
    ClientInfo,
    RuntimeSnapshot,
    SessionMCPHttpServer,
    SessionMCPServer,
    SessionMCPStdioServer,
    SessionOptions,
    TransportKind,
)
from chartreux.app_server.transport import JsonRpcTransport, memory_transport_pair
from chartreux.core.agent_loop import AgentLoop, AgentRuntimePolicy
from chartreux.core.agents.launch import FrozenPersona, LaunchCandidate, resolve_launch
from chartreux.core.agents.manager import AgentManager
from chartreux.core.config import (
    ChartreuxConfigSchema,
    MCPHttp,
    MCPServer,
    MCPStaticAuth,
    MCPStdio,
    MissingAPIKeyError,
    SessionLoggingConfig,
    build_default_orchestrator,
)
from chartreux.core.config.harness_files import HarnessFilesManager
from chartreux.core.config.layers.overrides import OverridesLayer
from chartreux.core.config.orchestrator import ConfigOrchestrator
from chartreux.core.hooks.config import load_hooks_from_fs
from chartreux.core.hooks.models import HookConfigResult
from chartreux.core.llm_models import LLMMessage, Role
from chartreux.core.paths import WORKTREES_DIR
from chartreux.core.scratchpad import cleanup_scratchpad
from chartreux.core.session import last_session_pointer
from chartreux.core.session.session_id import extract_suffix, generate_session_id
from chartreux.core.session.session_index import warm_session_index
from chartreux.core.session.session_lease import SessionLease
from chartreux.core.session.session_loader import SessionLoader
from chartreux.core.session_types import (
    AgentStats,
    CommittedModelIdentity,
    LaunchMetadataV2,
    LaunchPersonaV1,
    SessionMetadata,
)
from chartreux.core.skills.models import SkillInfo
from chartreux.core.subagents import (
    InvalidLaunchConfigError,
    LaunchConfig,
    UnsupportedChildForkError,
)
from chartreux.observability.logging import logger, set_config_log_level
from chartreux.utils.cache_store import FileSystemCacheStore

_SHORT_SESSION_ID_LENGTH = 8


def _launch_metadata(candidate: LaunchCandidate) -> LaunchMetadataV2:
    return LaunchMetadataV2(
        version=2,
        profile=candidate.profile.name,
        overrides=candidate.semantic_overrides.model_copy(deep=True),
        persona=LaunchPersonaV1(
            system_prompt_id=candidate.persona.system_prompt_id,
            instructions=candidate.persona.instructions,
        ),
        committed_model=candidate.committed_model,
    )


if TYPE_CHECKING:
    from chartreux.app_server.server import AppServer
    from chartreux.core.tools.mcp.registry import MCPRegistry


@dataclass(frozen=True, slots=True)
class NewSessionIntent:
    pass


@dataclass(frozen=True, slots=True)
class ContinueSessionIntent:
    pass


@dataclass(frozen=True, slots=True)
class ResumeSessionIntent:
    session_id: str

    def __post_init__(self) -> None:
        if not self.session_id:
            raise ValueError("A session ID is required to resume a session")


type LocalSessionIntent = NewSessionIntent | ContinueSessionIntent | ResumeSessionIntent


@dataclass(frozen=True, slots=True)
class ClientDescriptor:
    info: ClientInfo
    capabilities: ClientCapabilities = field(default_factory=ClientCapabilities)


def _default_client() -> ClientDescriptor:
    return ClientDescriptor(info=ClientInfo(name="vibe_client", version=__version__))


@dataclass(frozen=True, slots=True)
class LocalHarnessOptions:
    client: ClientDescriptor = field(default_factory=_default_client)
    session_options: SessionOptions = field(
        default_factory=lambda: SessionOptions(cwd=str(Path.cwd().resolve()))
    )
    session: LocalSessionIntent = field(default_factory=NewSessionIntent)
    client_tool_handler: ClientToolHandler | None = None


class RuntimeSessionNotFoundError(RuntimeError):
    pass


class RuntimeAuthenticationError(RuntimeError):
    def __init__(
        self, provider: str, *, kind: Literal["missing", "invalid"] = "missing"
    ) -> None:
        self.provider = provider
        self.kind = kind
        self.classification = (
            TurnErrorCode.INVALID_API_KEY if kind == "invalid" else None
        )
        detail = (
            f"Invalid API key for provider: {provider}"
            if kind == "invalid"
            else f"Authentication is required for provider: {provider}"
        )
        super().__init__(detail)


class RuntimeConfigurationError(RuntimeError):
    pass


class CommittedModelResumeError(RuntimeConfigurationError):
    """A persisted committed model cannot be used by the resuming process.

    Raised by the blueprint resume path (fresh session open): that flow has
    no interactive recovery, so it must fail fast with an actionable error
    instead of loading the session unpinned. Only the root-session resume
    RPC keeps the recovery flow.
    """

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class RootOpenRequest:
    options: SessionOptions
    client_info: ClientInfo
    session_id: str | None = None
    continue_latest: bool = False
    client_capabilities: ClientCapabilities = field(default_factory=ClientCapabilities)

    def __post_init__(self) -> None:
        if self.session_id is not None and self.continue_latest:
            raise ValueError("Cannot resume a session and continue the latest")


@dataclass(frozen=True, slots=True)
class _AgentLoopBlueprint:
    config_orchestrator: ConfigOrchestrator[ChartreuxConfigSchema]
    policy: AgentRuntimePolicy
    cwd: Path
    harness_files: HarnessFilesManager
    is_subagent: bool = False
    parent_session_id: str | None = None
    session_id: str | None = None
    session_dir: Path | None = None
    session_lease: SessionLease | None = None
    mcp_registry: MCPRegistry | None = None
    launch_profile: str | None = None
    launch_overrides: LaunchConfig | None = None
    frozen_system_prompt_id: str | None = None
    frozen_instructions: str | None = None
    committed_model: CommittedModelIdentity | None = None

    def build(self) -> AgentLoop:
        return AgentLoop(
            config_orchestrator=self.config_orchestrator,
            max_turns=self.policy.max_turns,
            max_price=self.policy.max_price,
            max_tokens=self.policy.max_tokens,
            max_session_tokens=self.policy.max_session_tokens,
            enable_streaming=self.policy.enable_streaming,
            launch_context=self.policy.launch_context,
            is_subagent=self.is_subagent,
            defer_heavy_init=True,
            headless=self.policy.headless,
            hook_config_result=self.policy.hook_config_result,
            inherited_workspace=self.policy.inherited_workspace,
            inherited_restrictions=self.policy.inherited_restrictions,
            inherited_mode_restrictions=self.policy.inherited_mode_restrictions,
            inherited_plan_write_scopes=self.policy.inherited_plan_write_scopes,
            parent_authority_getter=self.policy.parent_authority_getter,
            parent_authority_revision_getter=self.policy.parent_authority_revision_getter,
            launch_profile=self.launch_profile,
            launch_overrides=self.launch_overrides,
            frozen_system_prompt_id=self.frozen_system_prompt_id,
            frozen_instructions=self.frozen_instructions,
            committed_model=self.committed_model,
            mcp_registry=self.mcp_registry,
            cache_store=self.policy.cache_store,
            auto_title_enabled=self.policy.auto_title_enabled,
            parent_session_id=self.parent_session_id,
            cwd=self.cwd,
            harness_files=self.harness_files,
            session_id=self.session_id,
            session_dir=self.session_dir,
            session_lease=self.session_lease,
        )


@dataclass(frozen=True, slots=True)
class _SessionConfig:
    config_orchestrator: ConfigOrchestrator[ChartreuxConfigSchema]
    harness_files: HarnessFilesManager


@dataclass(frozen=True, slots=True)
class _RootRuntimeBlueprint:
    config_orchestrator: ConfigOrchestrator[ChartreuxConfigSchema]
    harness_files: HarnessFilesManager
    options: SessionOptions
    client_info: ClientInfo
    client_capabilities: ClientCapabilities
    hook_config_result: HookConfigResult
    cache_store: FileSystemCacheStore
    mcp_registry: MCPRegistry | None = None

    @property
    def cwd(self) -> Path:
        return Path(self.options.cwd or Path.cwd()).expanduser().resolve()

    @property
    def config(self) -> ChartreuxConfigSchema:
        return self.config_orchestrator.config

    def build(
        self,
        *,
        parent_session_id: str | None = None,
        session_id: str | None = None,
        session_dir: Path | None = None,
        session_lease: SessionLease | None = None,
        committed_model: CommittedModelIdentity | None = None,
    ) -> AgentLoop:
        policy = AgentRuntimePolicy(
            max_turns=self.options.max_turns,
            max_price=self.options.max_price,
            max_tokens=None,
            max_session_tokens=self.options.max_session_tokens,
            enable_streaming=True,
            launch_context={
                "agent_entrypoint": self.client_info.entrypoint,
                "agent_version": __version__,
                "client_name": self.client_info.name,
                "client_version": self.client_info.version,
                "terminal_emulator": self.client_info.terminal_emulator,
            },
            headless=self.options.headless,
            hook_config_result=self.hook_config_result,
            cache_store=self.cache_store,
            # The legacy AgentLoop generates background titles for CLI and Desktop.
            # Other clients retain the preview, and config can disable generation.
            auto_title_enabled=(
                self.client_info.entrypoint in {"cli", "desktop"}
                and self.config.session_logging.generate_titles
            ),
        )
        return _AgentLoopBlueprint(
            config_orchestrator=self.config_orchestrator.copy(),
            policy=policy,
            parent_session_id=parent_session_id,
            cwd=self.cwd,
            harness_files=self.harness_files,
            session_id=session_id,
            session_dir=session_dir,
            session_lease=session_lease,
            committed_model=committed_model,
            mcp_registry=self.mcp_registry,
        ).build()


@dataclass(frozen=True, slots=True)
class HarnessServer:
    _server: AppServer
    _transport: JsonRpcTransport
    _reconnectable: bool = False

    async def serve(self) -> None:
        await self._server.serve_connection(
            self._transport, close_on_disconnect=not self._reconnectable
        )

    def connect_client(self) -> AppServerClient:
        if not self._reconnectable:
            raise RuntimeError("This app-server transport cannot reconnect")
        client_transport, server_transport = memory_transport_pair()
        return AppServerClient(
            client_transport,
            run_peer=lambda: self._server.serve_connection(
                server_transport, close_on_disconnect=False
            ),
        )


class AgentRuntimeFactory:
    def resolve_latest(self, source: AgentLoop, cwd: Path) -> str:
        _require_session_logging(source.config)
        return _find_session_to_continue(source.config, cwd=cwd)

    async def resume_root(self, source: AgentLoop, session_id: str) -> None:
        """Resume a stored session by rebinding the existing loop in place.

        The existing MCP connections, tool registry, git context, and config
        are all reused — only session-scoped state (ID, messages, stats,
        session logger) is swapped. This avoids the cold-rebuild overhead of
        creating a fresh AgentLoop on every resume.

        The rebind runs before waiting for deferred init so the UI can render
        the resumed transcript immediately. ``finish_resume_root`` must be
        called afterward to await readiness.
        """
        session_id = await asyncio.to_thread(
            _resolve_resume_session_id, source.config, session_id
        )
        lease = await asyncio.to_thread(
            _acquire_session_lease, source.config, session_id
        )
        previous_model = source.config.get_active_model().alias
        previous_identity = source.committed_model
        previous_session_pinned = source.session_logger.active_model is not None
        prepared_scratchpad: Path | None = None
        target_model_applied = False
        try:
            session_path, loaded_messages, metadata = await asyncio.to_thread(
                _load_session, source.config, session_id
            )
            # Prepare the only fallible part of rebinding before the reload. This
            # keeps a scratchpad failure from leaving the old session's runtime
            # consumers configured for the target session.
            prepared_scratchpad = source.prepare_scratchpad_for_session(session_id)
            await source.config_orchestrator.reload()
            session_metadata = SessionMetadata.model_validate(metadata)
            resume_identity = _resume_identity(source.config, session_metadata)
            active_model = config_active_model(metadata)
            if resume_identity is not None:
                source.config.attach_committed_model(resume_identity)
            elif _is_legacy_root_metadata(session_metadata):
                await _restore_session_active_model(
                    source.config_orchestrator,
                    active_model,
                    clear_existing=previous_session_pinned,
                )
            else:
                # Committed-model recovery: drop the unresolvable identity and
                # clear any previous session pin instead of restoring the
                # missing selection. The user must pick a model before the next
                # turn; the pending choice is surfaced as a runtime issue.
                await _restore_session_active_model(
                    source.config_orchestrator, None, clear_existing=True
                )
            target_model_applied = True
            # ``_load_session`` already parsed metadata.json into ``metadata``;
            # parse that dict instead of re-reading the file from disk.
            stats = _build_stats(source, metadata)
            # Reload while this loop still identifies as the old session. Its
            # initial save therefore persists the old transcript and config.
            source.committed_model = resume_identity
            source._committed_selection = source.config.active_model
            if not _same_concrete_identity(source.committed_model, previous_identity):
                await source.reload_with_initial_messages()
            if resume_identity is None and not _is_legacy_root_metadata(
                session_metadata
            ):
                # Committed-model recovery: the reload re-commits the
                # configured default whenever no identity is held, so drop it
                # again here — the pending user choice must gate turn starts.
                source.committed_model = None
                source.config.attach_committed_model(None)
            # Identity transitions are durable at the next session save; this
            # accepted boundary deliberately does not add memory-disk transactions.
            _mark_legacy_root_cost_incomplete(stats, session_metadata)
            source.rebind_to_session(
                session_id,
                session_path,
                loaded_messages,
                session_metadata=session_metadata,
                parent_session_id=_parent_session_id(metadata),
                stats=stats,
                prepared_scratchpad=prepared_scratchpad,
            )
            source.replace_session_lease(lease)
        except BaseException:
            source.committed_model = previous_identity
            source.config.attach_committed_model(previous_identity)
            if prepared_scratchpad is not None:
                cleanup_scratchpad(prepared_scratchpad)
            if target_model_applied:
                await _restore_session_active_model(
                    source.config_orchestrator, previous_model, clear_existing=False
                )
            if lease is not None:
                await asyncio.to_thread(lease.release)
            raise

    async def finish_resume_root(self, source: AgentLoop, session_id: str) -> None:
        """Finish a resume by awaiting deferred initialization.

        Called after the ``session/resume`` RPC response is sent so the client
        can render the transcript while MCP init completes in the
        background. Deferred-init failure must not abort the caller before it
        emits ``runtime/updated`` — the degraded state (e.g. MCP discovery
        errors) is carried in the runtime snapshot instead.

        Init-duration recording lives in ``wait_until_ready`` via
        ``_ensure_init_duration_recorded``, not here.
        """
        try:
            await source._await_deferred_init()
        except Exception:
            logger.exception(
                "Deferred init failed after resuming session_id=%s", session_id
            )

    async def resume_blueprint(
        self,
        blueprint: _RootRuntimeBlueprint,
        session_id: str,
        session_lease: SessionLease | None = None,
    ) -> AgentLoop:
        session_path, loaded_messages, metadata = await asyncio.to_thread(
            _load_session, blueprint.config, session_id
        )
        await blueprint.config_orchestrator.reload()
        session_metadata = SessionMetadata.model_validate(metadata)
        # Blueprint resume (fresh session open) has no interactive recovery:
        # an unusable persisted committed model fails fast as a configuration
        # error instead of loading the session unpinned.
        resume_identity = _resume_identity(
            blueprint.config, session_metadata, fail_fast=True
        )
        if resume_identity is not None:
            blueprint.config.attach_committed_model(resume_identity)
        else:
            # Only legacy (absent or V1) envelopes reach here: re-resolve the
            # stored session pin from the current configuration.
            await _restore_session_active_model(
                blueprint.config_orchestrator, config_active_model(metadata)
            )
        replacement = blueprint.build(
            parent_session_id=_parent_session_id(metadata),
            session_id=session_id,
            session_dir=session_path,
            session_lease=session_lease,
            committed_model=resume_identity,
        )
        # Set messages and stats immediately so the UI can render the stored
        # transcript while the runtime (git, MCP) warms up in the background.
        # MessageList.update_system_prompt() inserts at position 0 when the
        # background thread eventually sets the system prompt, so no system
        # message needs to be present here.
        try:
            replacement.messages.reset_preserving_system(loaded_messages)
            _apply_stored_stats(replacement, metadata)
            _mark_legacy_root_cost_incomplete(replacement.stats, session_metadata)
        except BaseException:
            await close_agent_loop(replacement)
            raise
        return replacement

    async def create_child(
        self,
        parent: AgentLoop,
        candidate: LaunchCandidate | str,
        *,
        session_id: str | None = None,
        session_dir: Path | None = None,
    ) -> AgentLoop:
        if isinstance(candidate, str):
            candidate = resolve_launch(
                profile_name=candidate,
                config=None,
                parent_orchestrator=parent.config_orchestrator,
                tool_inventory={
                    name: object() for name in parent.tool_manager.registered_tools
                },
                authorized_tool_names=frozenset(parent.tool_manager.available_tools),
                agent_manager=parent.agent_manager,
            )
        policy = parent.child_runtime_policy
        agent_profile = candidate.profile
        agent_name = agent_profile.name
        parent_session_dir = parent.session_logger.session_dir
        orchestrator = candidate.orchestrator.copy()
        session_logging = SessionLoggingConfig(
            save_dir=(
                str(parent_session_dir / "agents")
                if parent_session_dir is not None
                else ""
            ),
            session_prefix=agent_name,
            enabled=parent_session_dir is not None,
        )
        failures = await orchestrator.set_field(
            "/session_logging",
            session_logging.model_dump(mode="json"),
            reason="configure child session logging",
            target_layer=OverridesLayer.NAME,
        )
        if failures:
            raise RuntimeConfigurationError(
                "Failed to configure child session logging"
            ) from failures[0]
        child_session_id = session_id or generate_session_id()
        lease = await asyncio.to_thread(
            _acquire_session_lease, parent.config, child_session_id
        )
        try:
            child = self._create_like(
                parent,
                config_orchestrator=orchestrator,
                is_subagent=True,
                parent_session_id=parent.session_id,
                session_id=child_session_id,
                session_dir=session_dir,
                session_lease=lease,
                policy=policy,
                launch_candidate=candidate,
            )
            metadata = child.session_logger.session_metadata
            if metadata is not None:
                metadata.launch_config = _launch_metadata(candidate)
            return child
        except BaseException:
            if lease is not None:
                await asyncio.to_thread(lease.release)
            raise

    async def resume_child(
        self, parent: AgentLoop, agent_name: str, session_id: str, session_dir: Path
    ) -> AgentLoop:
        try:
            loaded_messages, raw_metadata = await asyncio.to_thread(
                SessionLoader.load_session, session_dir
            )
        except ValueError as exc:
            raise InvalidLaunchConfigError(
                "launch_config", "Stored launch configuration is malformed"
            ) from exc
        try:
            metadata = SessionMetadata.model_validate(raw_metadata)
        except ValidationError as exc:
            raise InvalidLaunchConfigError(
                "launch_config", "Stored launch configuration is malformed"
            ) from exc
        envelope = metadata.launch_config
        if envelope is not None and envelope.profile != agent_name:
            raise InvalidLaunchConfigError(
                "launch_config.profile", "Stored launch profile does not match its link"
            )
        if envelope is not None and envelope.profile is None:
            raise InvalidLaunchConfigError(
                "launch_config.profile", "Stored child launch profile is missing"
            )
        candidate = resolve_launch(
            profile_name=envelope.profile if envelope is not None else agent_name,
            config=None,
            parent_orchestrator=parent.config_orchestrator,
            tool_inventory={
                name: object() for name in parent.tool_manager.registered_tools
            },
            authorized_tool_names=frozenset(parent.tool_manager.available_tools),
            history=loaded_messages,
            agent_manager=parent.agent_manager,
            accumulated_overrides=(
                envelope.overrides if envelope is not None else None
            ),
            frozen_persona=(
                FrozenPersona(
                    system_prompt_id=envelope.persona.system_prompt_id,
                    instructions=envelope.persona.instructions,
                )
                if envelope is not None
                else None
            ),
            committed_model=(
                envelope.committed_model
                if isinstance(envelope, LaunchMetadataV2)
                else None
            ),
        )
        child = await self.create_child(
            parent, candidate, session_id=session_id, session_dir=session_dir
        )
        # Eager message setting: background thread inserts system prompt at position 0
        # when _complete_init finishes, same pattern as resume_blueprint.
        try:
            child.messages.reset_preserving_system(loaded_messages)
            child.session_logger.apply_resumed_session(
                session_id, session_dir, metadata
            )
            if child.session_logger.session_metadata is not None:
                child.session_logger.session_metadata.launch_config = (
                    _launch_metadata(candidate) if envelope is not None else None
                )
            _apply_stored_stats(child, raw_metadata)
            if envelope is not None and envelope.version == 1:
                child.stats.has_unknown_cost = True
                child.stats.known_cost_total = 0.0
        except BaseException:
            await close_agent_loop(child)
            raise
        return child

    async def fork(self, source: AgentLoop, message_id: str | None) -> AgentLoop:
        metadata = source.session_logger.session_metadata
        if source._is_subagent or (
            metadata is not None
            and metadata.launch_config is not None
            and metadata.launch_config.profile is not None
        ):
            raise UnsupportedChildForkError(
                "session", "Child sessions cannot be forked"
            )
        session_id = generate_session_id(suffix=extract_suffix(source.session_id))
        lease = await asyncio.to_thread(
            _acquire_session_lease, source.config, session_id
        )
        forked: AgentLoop | None = None
        try:
            forked = self._create_like(
                source,
                is_subagent=source._is_subagent,
                # Fork preserves identity and existing ceilings; it is not a new
                # child capture of the source's current local mode.
                policy=source.runtime_policy,
                parent_session_id=source.session_id,
                session_id=session_id,
                session_lease=lease,
            )
            await forked.wait_until_ready()
            forked.messages.extend(_messages_for_fork(source, message_id))
            await forked.session_logger.save_interaction(
                forked.messages, forked.stats, forked.config, forked.tool_manager, None
            )
        except BaseException:
            if forked is not None:
                await close_agent_loop(forked)
            elif lease is not None:
                await asyncio.to_thread(lease.release)
            raise
        return forked

    @staticmethod
    def _create_like(
        source: AgentLoop,
        *,
        config_orchestrator: ConfigOrchestrator[ChartreuxConfigSchema] | None = None,
        is_subagent: bool = False,
        parent_session_id: str | None = None,
        session_id: str | None = None,
        session_dir: Path | None = None,
        session_lease: SessionLease | None = None,
        policy: AgentRuntimePolicy | None = None,
        launch_candidate: LaunchCandidate | None = None,
    ) -> AgentLoop:
        if policy is None:
            policy = (
                source.child_runtime_policy if is_subagent else source.runtime_policy
            )
        if is_subagent:
            policy = replace(policy, enable_streaming=False)
        replacement = _AgentLoopBlueprint(
            config_orchestrator=(
                config_orchestrator or source.config_orchestrator.copy()
            ),
            policy=policy,
            is_subagent=is_subagent,
            parent_session_id=parent_session_id,
            cwd=source.cwd,
            harness_files=source.harness_files,
            session_id=session_id,
            session_dir=session_dir,
            session_lease=session_lease,
            launch_profile=(
                launch_candidate.profile.name if launch_candidate else None
            ),
            launch_overrides=(
                launch_candidate.semantic_overrides if launch_candidate else None
            ),
            frozen_system_prompt_id=(
                launch_candidate.persona.system_prompt_id if launch_candidate else None
            ),
            frozen_instructions=(
                launch_candidate.persona.instructions if launch_candidate else None
            ),
            committed_model=(
                launch_candidate.committed_model
                if launch_candidate
                else source.committed_model
            ),
            mcp_registry=(
                source.mcp_registry.clone_configuration()
                if source.mcp_registry is not None
                else None
            ),
        ).build()
        return replacement


class HarnessProcess:
    def __init__(self, harness_files: HarnessFilesManager | None = None) -> None:
        from chartreux.app_server._mcp_auth import MCPAuthenticationService
        from chartreux.app_server.mcp_catalog import MCPCatalogService

        self.runtime_factory = AgentRuntimeFactory()
        self.cache_store = FileSystemCacheStore()
        self.harness_files = harness_files or HarnessFilesManager(
            sources=("user", "project")
        )
        self._configuration_lock = threading.Lock()
        self._configured = False
        self._staged_roots: dict[str, AgentLoop] = {}
        self._staged_roots_lock = asyncio.Lock()
        self._closed = False
        self.host_handler = HostRequestHandler(self.harness_files)
        self.mcp_authentication = MCPAuthenticationService()
        self.mcp_catalog = MCPCatalogService(
            self.mcp_authentication,
            sessionless_catalog_factory=self.build_sessionless_mcp_catalog,
        )

    def create_session_backend_host(
        self, services: SessionBackendServices
    ) -> SessionBackendHost:
        from chartreux.app_server._session_runtime_impl import (
            create_session_backend_host_impl,
        )

        return create_session_backend_host_impl(
            open_root=self.open_root,
            runtime_factory=self.runtime_factory,
            host_handler=self.host_handler,
            stage_root=self.stage_root,
            services=services,
            mcp_catalog_service=self.mcp_catalog,
        )

    async def stage_root(self, root: AgentLoop) -> None:
        superseded: AgentLoop | None = None
        async with self._staged_roots_lock:
            if self._closed:
                superseded = root
            else:
                superseded = self._staged_roots.get(root.session_id)
                self._staged_roots[root.session_id] = root
        if superseded is not None and superseded is not root:
            await close_agent_loop(superseded)
        if self._closed:
            if superseded is root:
                await close_agent_loop(root)
            raise RuntimeError("The app-server harness process is closed")

    async def close(self) -> None:
        async with self._staged_roots_lock:
            if self._closed:
                return
            self._closed = True
            staged = list(self._staged_roots.values())
            self._staged_roots.clear()
        errors: list[BaseException] = []
        for root in staged:
            try:
                await close_agent_loop(root)
            except BaseException as exc:
                errors.append(exc)
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup("Failed to close staged session runtimes", errors)

    async def build_session_runtime(self, options: SessionOptions) -> RuntimeSnapshot:
        session_config = await self._build_session_config(options)
        return build_runtime_snapshot(
            session_config.config_orchestrator, session_config.harness_files
        )

    async def build_sessionless_mcp_catalog(
        self,
    ) -> ConfigOrchestrator[ChartreuxConfigSchema]:
        session_config = await self._build_session_config(SessionOptions())
        return session_config.config_orchestrator

    async def _build_session_config(self, options: SessionOptions) -> _SessionConfig:
        cwd = Path(options.cwd or Path.cwd()).expanduser().resolve()
        workspace_roots = [
            Path(root).expanduser().resolve() for root in options.workspace_roots
        ]
        harness_files = self.harness_files.for_session(
            cwd, workspace_roots=workspace_roots
        )
        if options.trust_workspace:
            harness_files = harness_files.trust_for_session(cwd)
        overrides = _session_config_overrides(options)
        config_orchestrator = await build_default_orchestrator(
            overrides, harness_files=harness_files
        )
        return _SessionConfig(
            config_orchestrator=config_orchestrator, harness_files=harness_files
        )

    async def build_root_blueprint(
        self,
        options: SessionOptions,
        client_info: ClientInfo,
        client_capabilities: ClientCapabilities | None = None,
    ) -> _RootRuntimeBlueprint:
        session_config = await self._build_session_config(options)
        config_orchestrator = session_config.config_orchestrator
        harness_files = session_config.harness_files
        hook_config_result = await asyncio.to_thread(
            load_hooks_from_fs, harness_files=harness_files
        )
        await asyncio.to_thread(self._configure_process, config_orchestrator.config)
        return _RootRuntimeBlueprint(
            config_orchestrator=config_orchestrator,
            harness_files=harness_files,
            options=options,
            client_info=client_info,
            client_capabilities=client_capabilities or ClientCapabilities(),
            hook_config_result=hook_config_result,
            cache_store=self.cache_store,
            mcp_registry=await self._build_mcp_registry_impl(config_orchestrator),
        )

    async def open_root(self, request: RootOpenRequest) -> AgentLoop:
        try:
            if request.session_id is not None:
                staged = await self._claim_staged_root(request.session_id)
                if staged is not None:
                    return staged
            blueprint = await self.build_root_blueprint(
                request.options, request.client_info, request.client_capabilities
            )
            session_id = request.session_id
            if request.continue_latest:
                session_id = _find_session_to_continue(
                    blueprint.config, cwd=blueprint.cwd
                )
            if session_id is not None:
                session_id = await asyncio.to_thread(
                    _resolve_resume_session_id, blueprint.config, session_id
                )
                lease = await asyncio.to_thread(
                    _acquire_session_lease, blueprint.config, session_id
                )
                try:
                    return await self.runtime_factory.resume_blueprint(
                        blueprint, session_id, lease
                    )
                except BaseException:
                    if lease is not None:
                        await asyncio.to_thread(lease.release)
                    raise
            session_id = generate_session_id()
            lease = await asyncio.to_thread(
                _acquire_session_lease, blueprint.config, session_id
            )
            try:
                return blueprint.build(session_id=session_id, session_lease=lease)
            except BaseException:
                if lease is not None:
                    await asyncio.to_thread(lease.release)
                raise
        except MissingAPIKeyError as exc:
            raise RuntimeAuthenticationError(exc.provider_name) from exc
        except (ValidationError, ValueError) as exc:
            raise RuntimeConfigurationError(str(exc)) from exc

    async def _build_mcp_registry_impl(
        self, orchestrator: ConfigOrchestrator[ChartreuxConfigSchema]
    ) -> MCPRegistry:
        from chartreux.app_server._session_backend_impl import (
            configure_mcp_registry_impl,
        )
        from chartreux.core.tools.mcp.registry import MCPRegistry

        # Bound in the registry's name, not the orchestrator's: this is the
        # blueprint's orchestrator and the session's loop is handed a copy of
        # it, so nothing holds this one once the blueprint has been consumed,
        # while the registry it arms here goes on resolving through the
        # binding. The registry is how long that binding is needed for.
        registry = MCPRegistry()
        configuration = await self.mcp_catalog.resolve_catalog(
            orchestrator, owner=registry
        )
        cache_root = (
            Path(orchestrator.config.session_logging.save_dir)
            .expanduser()
            .resolve()
            .parent
            / "mcp-descriptors"
            # Preserve this descriptor-cache path component for compatibility.
            / "legacy"
        )
        configure_mcp_registry_impl(
            registry,
            configuration,
            self.mcp_authentication,
            descriptor_cache_root=cache_root,
        )
        return registry

    async def _claim_staged_root(self, session_id: str) -> AgentLoop | None:
        async with self._staged_roots_lock:
            if self._closed:
                raise RuntimeError("The app-server harness process is closed")
            return self._staged_roots.pop(session_id, None)

    def _configure_process(self, config: ChartreuxConfigSchema) -> None:
        with self._configuration_lock:
            if self._configured:
                return
            warm_session_index(config.session_logging)
            set_config_log_level(config.log_level)
            self._configured = True


async def create_harness_server(
    transport: JsonRpcTransport,
    *,
    transport_kind: TransportKind,
    process: HarnessProcess | None = None,
) -> HarnessServer:
    """Build a server over ``transport``."""
    from chartreux.app_server.server import AppServer

    process = process or HarnessProcess()
    return HarnessServer(
        _server=AppServer(
            transport,
            transport_kind=transport_kind,
            host_handler=process.host_handler,
            session_backend_host_factory=process.create_session_backend_host,
            mcp_catalog_service=process.mcp_catalog,
        ),
        _transport=transport,
        _reconnectable=transport_kind == "in_process",
    )


def build_runtime_snapshot(
    config_orchestrator: ConfigOrchestrator[ChartreuxConfigSchema],
    harness_files: HarnessFilesManager,
    *,
    issues: Sequence[ConfigIssue] = (),
) -> RuntimeSnapshot:
    agents = AgentManager(config_orchestrator, harness_files=harness_files)
    return build_unified_runtime_snapshot(config_orchestrator, agents, issues=issues)


def build_unified_runtime_snapshot(
    config_orchestrator: ConfigOrchestrator[ChartreuxConfigSchema],
    agents: AgentManager,
    *,
    issues: Sequence[ConfigIssue] = (),
    skills: Iterable[SkillInfo] = (),
    tools: Iterable[str] = (),
    custom_tool_names: frozenset[str] = frozenset(),
    hooks_count: int = 0,
) -> RuntimeSnapshot:
    """Project the layered config into the runtime state the client observes.

    The tools are the resolved Chartreux catalogue used to derive the local Runtime's
    executable tool modes. ``skills`` and ``issues`` are supplied by the caller
    because they come from discovery, not from the layered
    config. ``hooks_count`` is supplied by the caller because hook discovery
    happens outside the layered config. Integration projections are filled by
    their owning adapters.
    """
    config = config_orchestrator.config
    active_model = config.get_active_model()
    return RuntimeSnapshot(
        config=project_config_view(
            config, active_model_pinned=active_model_is_pinned(config_orchestrator)
        ),
        skills=project_skill_summaries(skills),
        tools=[
            ToolSummary(name=name, is_custom=name in custom_tool_names)
            for name in tools
        ],
        stats=AgentStatsSnapshot(
            input_price_per_million=active_model.input_price,
            output_price_per_million=active_model.output_price,
            cached_input_price_per_million=active_model.cached_input_price,
        ),
        context_window=active_model.auto_compact_threshold,
        issues=list(issues),
        hooks_count=hooks_count,
        mcp=MCPState(),
    )


def _project_session_mcp_server(server: SessionMCPServer) -> MCPServer:
    match server:
        case SessionMCPHttpServer():
            return MCPHttp(
                transport="streamable-http",
                name=server.name,
                url=server.url,
                auth=MCPStaticAuth(headers=server.headers),
            )
        case SessionMCPStdioServer():
            return MCPStdio(
                transport="stdio",
                name=server.name,
                command=server.command,
                args=server.args,
                env=server.env,
                cwd=server.cwd,
            )
        case _:
            raise TypeError(f"Unsupported session MCP server: {type(server).__name__}")


def _session_config_overrides(options: SessionOptions) -> dict[str, object]:
    overrides: dict[str, object] = {}
    if options.enabled_tools is not None:
        overrides["enabled_tools"] = options.enabled_tools
    if options.disabled_tools:
        overrides["disabled_tools"] = options.disabled_tools
    if options.mcp_servers:
        overrides["mcp_servers"] = [
            _project_session_mcp_server(server).model_dump(
                mode="json", exclude_none=True
            )
            for server in options.mcp_servers
        ]
    return overrides


def _require_session_logging(config: ChartreuxConfigSchema) -> None:
    if config.session_logging.enabled:
        return
    raise RuntimeSessionNotFoundError(
        "Session logging is disabled. Enable it in config to use --continue or --resume"
    )


def _find_session_to_continue(config: ChartreuxConfigSchema, *, cwd: Path) -> str:
    cwd = cwd.resolve()
    pointer_session_id = last_session_pointer.load(config.session_logging)
    if pointer_session_id is not None:
        session = SessionLoader.find_session_by_id(
            pointer_session_id, config.session_logging, working_directory=cwd
        )
        if session is not None:
            return pointer_session_id

    session = SessionLoader.find_latest_session(
        config.session_logging, working_directory=cwd
    )
    if session is not None:
        _, metadata = SessionLoader.load_session(session)
        session_id = metadata.get("session_id")
        if isinstance(session_id, str) and session_id:
            return session_id
        raise RuntimeSessionNotFoundError(f"Saved session has no session ID: {session}")

    message = (
        f"No previous sessions found in {config.session_logging.save_dir} for cwd={cwd}"
    )
    if cwd.is_relative_to(WORKTREES_DIR.path.resolve()):
        message = (
            f"{message}. This worktree has no sessions yet; start a new one or "
            "use --resume <ID> to continue an existing session here"
        )
    raise RuntimeSessionNotFoundError(message)


def _load_session(
    config: ChartreuxConfigSchema, session_id: str
) -> tuple[Path, list[LLMMessage], dict[str, object]]:
    session_path = SessionLoader.find_session_by_id(session_id, config.session_logging)
    if session_path is None:
        raise RuntimeSessionNotFoundError(session_id)
    loaded_messages, metadata = SessionLoader.load_session(session_path)
    return session_path, loaded_messages, metadata


def _resolve_resume_session_id(config: ChartreuxConfigSchema, session_id: str) -> str:
    sessions = SessionLoader.list_sessions(config.session_logging)
    exact = [session for session in sessions if session["session_id"] == session_id]
    matches = exact or [
        session
        for session in sessions
        if session["session_id"][:_SHORT_SESSION_ID_LENGTH] == session_id
    ]
    canonical_ids = {session["session_id"] for session in matches}
    if len(canonical_ids) > 1:
        raise ValueError(f"Session ID is ambiguous: {session_id}")
    return next(iter(canonical_ids), session_id)


def _acquire_session_lease(
    config: ChartreuxConfigSchema, session_id: str
) -> SessionLease | None:
    if not config.session_logging.enabled:
        return None
    return SessionLease(Path(config.session_logging.save_dir), session_id).acquire()


async def _restore_session_active_model(
    orchestrator: ConfigOrchestrator[ChartreuxConfigSchema],
    active_model: str | None,
    *,
    clear_existing: bool = False,
) -> None:
    if active_model is not None:
        failures = await set_session_active_model_override(
            orchestrator, active_model, reason="restore session active model"
        )
    elif clear_existing:
        failures = await clear_session_active_model_override(
            orchestrator, reason="clear previous session active model"
        )
    else:
        return
    if failures:
        raise RuntimeConfigurationError(
            f"Failed to restore session active model: {failures[0]}"
        )


def _same_concrete_identity(
    left: CommittedModelIdentity | None, right: CommittedModelIdentity | None
) -> bool:
    if left is None or right is None:
        return left is right
    return (left.base_model, left.provider, left.wire_name) == (
        right.base_model,
        right.provider,
        right.wire_name,
    )


def _is_legacy_root_metadata(metadata: SessionMetadata) -> bool:
    """Recognize root sessions that predate persisted committed identity."""
    envelope = metadata.launch_config
    return envelope is None or envelope.version == 1


def _mark_legacy_root_cost_incomplete(
    stats: AgentStats | None, metadata: SessionMetadata
) -> None:
    if stats is not None and _is_legacy_root_metadata(metadata):
        stats.has_unknown_cost = True
        stats.known_cost_total = 0.0


def _legacy_identity_policy() -> None:
    """Apply V1 compatibility: re-resolve current config and never reprice history."""
    # V1 envelopes predate committed identity. Resume therefore selects from the
    # current configuration; callers mark historical cost incomplete separately.
    return None


def _resume_identity(
    config: ChartreuxConfigSchema, metadata: SessionMetadata, *, fail_fast: bool = False
) -> CommittedModelIdentity | None:
    envelope = metadata.launch_config
    if envelope is None or envelope.version == 1:
        return _legacy_identity_policy()
    from chartreux.core.model_catalog.resolver import ModelResolutionError, resolver_for

    try:
        resolver_for(config).resolve_committed(
            envelope.committed_model, allowed_models=config.allowed_models
        )
    except ModelResolutionError as exc:
        if fail_fast:
            # Blueprint resume (fresh session open) has no interactive
            # recovery: fail fast with an actionable error instead of
            # loading the session unpinned.
            identity = envelope.committed_model
            raise CommittedModelResumeError(
                exc.code,
                f"Cannot resume session {metadata.session_id}: its committed "
                f"model {identity.provider}/{identity.wire_name} (base "
                f"{identity.base_model!r}) is unusable in the current "
                f"configuration: {exc}. Choose another model before resuming "
                f"this session.",
            ) from exc
        # Recovery: the committed deployment left the catalog (or was disabled
        # or disallowed). Load the session unpinned instead of failing the
        # resume — the user must choose a model before the next turn. The
        # pending choice stays derivable from the stored envelope and is
        # surfaced as a runtime issue; see committed_model_recovery_issue.
        logger.warning(
            "Committed model unavailable on resume session_id=%s: %s",
            metadata.session_id,
            exc,
        )
        return None
    return envelope.committed_model


def _parent_session_id(metadata: dict[str, object]) -> str | None:
    value = metadata.get("parent_session_id")
    return value if isinstance(value, str) else None


def _build_stats(loop: AgentLoop, metadata: dict[str, object]) -> AgentStats | None:
    if not isinstance(raw_stats := metadata.get("stats"), dict):
        return None
    stats = AgentStats.model_validate(raw_stats)
    if stats.cached_input_price_per_million is None:
        try:
            stats.cached_input_price_per_million = (
                loop.config.get_active_model().cached_input_price
            )
        except ValueError:
            pass
    return stats


def _apply_stored_stats(loop: AgentLoop, metadata: dict[str, object]) -> None:
    stats = _build_stats(loop, metadata)
    if stats is not None:
        loop.stats = stats


def _messages_for_fork(source: AgentLoop, message_id: str | None) -> list[LLMMessage]:
    messages = [
        message for message in source.messages if message.role is not Role.system
    ]
    if message_id is None:
        return [message.model_copy(deep=True) for message in messages]

    anchor = next(
        (
            index
            for index, message in enumerate(messages)
            if message.message_id == message_id
        ),
        None,
    )
    if anchor is None:
        raise ValueError(f"Cannot fork from unknown message_id: {message_id}")
    if messages[anchor].role is not Role.user:
        raise ValueError("Fork from message_id is only supported for user messages")

    end = next(
        (
            index
            for index, message in enumerate(messages[anchor + 1 :], start=anchor + 1)
            if message.role is Role.user
        ),
        len(messages),
    )
    return [message.model_copy(deep=True) for message in messages[:end]]


async def close_agent_loop(agent_loop: AgentLoop) -> None:
    await agent_loop.aclose()
