from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypedDict, cast

from pydantic import JsonValue

from chartreux.app_server._agent_transcript import (
    archive_transcript_source,
    iter_archived_messages,
)
from chartreux.app_server._session_model import active_model_is_pinned
from chartreux.app_server._shell import restored_shell_effect_state, shell_effect_detail
from chartreux.app_server._tool_projection import (
    project_effect_detail,
    project_effect_output_value,
)
from chartreux.app_server._utils import now_ms
from chartreux.app_server._worktree_effects import WorktreeEffect
from chartreux.app_server.config import (
    THINKING_LEVELS,
    ConfigView,
    EffectiveThinkingView,
    ModelConfigView,
    RouteTargetView,
    RouteView,
    StatusLineConfigView,
    ThinkingLevel,
)
from chartreux.app_server.models import (
    COMMITTED_MODEL_RECOVERY_ISSUE_FILE,
    AgentStatsSnapshot,
    CancelledEffectState,
    CompletedEffectState,
    ConfigIssue,
    ContentBlock,
    DebugLogEntry,
    DebugLogPage,
    EffectCallDisplay,
    EffectResultDisplay,
    EffectState,
    FailedEffectState,
    GenericEffectDetail,
    ImageAttachment,
    ImageContentBlock,
    MCPSourceStatus,
    MCPSourceSummary,
    MCPState,
    MCPToolSummary,
    PublicCheckpointEntry,
    PublicEffectEntry,
    PublicEntryGenerationStatus,
    PublicError,
    PublicHistoryEntry,
    PublicMessageEntry,
    PublicReasoningEntry,
    ResourceContentBlock,
    SessionLogSummary,
    SkillSummary,
    SubagentEffectDetail,
    TextContentBlock,
    ToolSummary,
)
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.config import ChartreuxConfigSchema, ModelConfig
from chartreux.core.llm.thinking_levels import (
    declared_thinking_levels,
    get_thinking_levels,
    narrow_thinking_levels,
    note_thinking_fallback,
    resolve_enabled_thinking_level,
    resolve_thinking_level,
)
from chartreux.core.llm_models import (
    ImageAttachment as CoreImageAttachment,
    LLMMessage,
    Role,
)
from chartreux.core.log_reader import PaginatedLogs
from chartreux.core.model_catalog.resolver import ModelResolutionError
from chartreux.core.session.transcript_archive import (
    ArchiveIntegrityError,
    verify_archived_prefix_on_disk,
)
from chartreux.core.session_types import (
    LaunchMetadataV2,
    SessionMetadata,
    WorktreeContext,
)
from chartreux.core.skills.models import SkillInfo, SkillSource
from chartreux.core.subagents import InvalidLaunchThinkingError
from chartreux.core.tools.base import ToolPermission
from chartreux.core.tools.builtins.web_search import (
    SearchProviderDiagnostic,
    WebSearch,
    effective_web_search_config,
)
from chartreux.core.tools.remote import AuthStatus, MCPTool
from chartreux.core.utils import CANCELLATION_TAG, TOOL_ERROR_TAG, TaggedText
from chartreux.core.utils.matching import name_matches
from chartreux.model_display import format_model_display_name
from chartreux.observability.logging import logger
from chartreux.user_content import UserResource
from chartreux.utils.mcp import format_tool_display_description
from chartreux.utils.tool_presentation import ToolCallPresentation

if TYPE_CHECKING:
    from chartreux.app_server._subagent_records import ArchiveTranscriptSource


def project_launch_purposes(agent_loop: AgentLoop) -> list[str]:
    """The purpose label a routed launch declared at launch resolution.

    Package 16b retired the dispatch slot/roster surface: a launch's purpose
    is the label recorded on the loop (``AgentLoop.launch_purpose``), set
    when the launch resolved through a purpose route. A direct launch
    carries no purpose.
    """
    purpose = agent_loop.launch_purpose
    if purpose is None:
        return []
    return [purpose]


def project_config(agent_loop: AgentLoop) -> ConfigView:
    return project_config_view(
        agent_loop.config,
        active_model_pinned=active_model_is_pinned(agent_loop.config_orchestrator),
        effective_thinking=project_effective_thinking(agent_loop),
    )


def project_effective_thinking(agent_loop: AgentLoop) -> EffectiveThinkingView:
    """The next request's thinking estimate on the committed deployment.

    Computed from the committed identity and the loop's current message
    history — the same inputs the next request resolves against — so a
    history-dependent coercion appears as soon as the entry lands. The
    declared-level fallback is resolved and reported here too, at the
    effective-selection boundary. An unresolvable deployment reports unknown
    instead of claiming the requested level is effective.
    """
    config = agent_loop.config
    model = config.get_active_model()
    requested = model.thinking
    if agent_loop.committed_model is None:
        return EffectiveThinkingView(requested=requested)
    try:
        provider = config.get_provider_for_model(model)
    except ValueError:
        return EffectiveThinkingView(requested=requested)
    levels = get_thinking_levels(str(provider.backend), provider.api_style, model.name)
    if levels is None:
        if model.supported_thinking_levels is not None:
            return EffectiveThinkingView(requested=requested)
        # The provider cannot encode thinking at all, so only "off" resolves.
        disabled = requested == "off"
        return EffectiveThinkingView(
            requested=requested,
            semantic="off" if disabled else None,
            effective="off" if disabled else None,
            api_effort=None,
            known=disabled,
            matches_requested=True if disabled else None,
        )
    # One coercion rule, shared with launch admission: retained Anthropic
    # thinking forces a level even when the session asked for "off".
    from chartreux.core.agents.launch import _effective_thinking

    semantic = _effective_thinking(config, model, requested, agent_loop.messages)
    narrowed = narrow_thinking_levels(levels, model.supported_thinking_levels)
    try:
        if semantic != requested:
            # The retained-reasoning coercion must keep thinking enabled, so the
            # estimate matches the wire even when the semantic level is undeclared.
            resolution = resolve_enabled_thinking_level(narrowed, semantic)
        else:
            resolution = resolve_thinking_level(narrowed, semantic)
    except InvalidLaunchThinkingError:
        return EffectiveThinkingView(requested=requested)
    note_thinking_fallback(f"{model.provider}/{model.name}", resolution)
    return EffectiveThinkingView(
        requested=requested,
        semantic=semantic,
        effective=cast(ThinkingLevel, resolution.effective),
        api_effort=resolution.wire_value,
        known=True,
        matches_requested=(
            resolution.effective == requested
            and semantic == requested
            and (
                resolution.wire_value == requested
                or (requested == "off" and resolution.wire_value is None)
            )
        ),
    )


def project_config_view(
    config: ChartreuxConfigSchema,
    *,
    active_model_pinned: bool = False,
    effective_thinking: EffectiveThinkingView | None = None,
) -> ConfigView:
    routes, routes_error = _project_routes(config)
    return ConfigView(
        active_model=_project_model_config(config, config.get_active_model()),
        active_model_expression=config.active_model,
        allowed_models=list(config.allowed_models),
        active_model_pinned=active_model_pinned,
        effective_thinking=effective_thinking,
        # The configured default, never the active model: clients render it as
        # the "Default (currently X)" hint, which must stay stable while a pin
        # is in effect.
        default_model_alias=_project_default_model_alias(config),
        theme=config.theme,
        log_level=config.log_level,
        disable_welcome_banner_animation=config.disable_welcome_banner_animation,
        autocopy_to_clipboard=config.autocopy_to_clipboard,
        file_watcher_for_autocomplete=config.file_watcher_for_autocomplete,
        ask_confirmation_on_exit=config.ask_confirmation_on_exit,
        show_thinking_nodes=config.show_thinking_nodes,
        show_message_timestamps=config.show_message_timestamps,
        status_line=StatusLineConfigView(
            segments=list(config.status_line.segments),
            directory_style=config.status_line.directory_style,
            context_style=config.status_line.context_style,
            separator=config.status_line.separator,
        ),
        ascii_chrome=config.ascii_chrome,
        enable_notifications=config.enable_notifications,
        emit_program_status=config.emit_program_status,
        enable_system_trust_store=config.enable_system_trust_store,
        models=[
            _project_model_config(config, model)
            for model in config.available_models().values()
        ],
        validation_warnings=[
            *config.validation_warnings,
            *_web_search_validation_warnings(config),
        ],
        routes=routes,
        routes_error=routes_error,
    )


def _project_routes(
    config: ChartreuxConfigSchema,
) -> tuple[list[RouteView], str | None]:
    """The conversation's live purpose-route table, inspected side-effect-free.

    The projection is the wire form of the SAME inspection doctor renders, so
    the browser and the diagnostic can never disagree about a route. An
    inspection that cannot start reports why instead of fabricating rows.
    """
    from chartreux.core.dispatch.route_inspection import inspect_config_routes

    report = inspect_config_routes(config)
    return (
        [
            RouteView(
                purpose=route.purpose,
                requires_distinct_models=route.requires_distinct_models,
                resolved_distinct_models=route.resolved_distinct_models,
                reload_status=route.reload_status,
                reload_detail=route.reload_detail,
                launch_status=route.launch_status,
                launch_detail=route.launch_detail,
                targets=[
                    RouteTargetView(
                        seat=target.seat,
                        identity=target.identity,
                        tier=target.tier,
                        overlay=target.overlay,
                        binding=target.binding,
                        marker=target.marker,
                        preview=target.preview,
                        error=target.error,
                    )
                    for target in route.targets
                ],
            )
            for route in report.routes
        ],
        report.error,
    )


def _project_default_model_alias(config: ChartreuxConfigSchema) -> str:
    try:
        return config.resolve_default_model_alias()
    except ModelResolutionError:
        # The default is a display hint only; a valid explicit selection must
        # remain usable when the shipped default is unavailable.
        return "none"


def _web_search_validation_warnings(config: ChartreuxConfigSchema) -> list[str]:
    if config.enabled_tools and not name_matches("web_search", config.enabled_tools):
        return []
    if name_matches("web_search", config.disabled_tools):
        return []
    web_search_config = effective_web_search_config(config)
    if isinstance(web_search_config, SearchProviderDiagnostic):
        return [web_search_config.message]
    if web_search_config.permission is ToolPermission.NEVER:
        return []
    diagnostic = WebSearch.availability_diagnostic(config)
    return [diagnostic.message] if diagnostic is not None else []


def project_workdir(agent_loop: AgentLoop) -> str:
    return agent_loop.config.displayed_workdir or str(agent_loop.cwd)


def project_thinking_levels(
    config: ChartreuxConfigSchema, model: ModelConfig
) -> tuple[ThinkingLevel, ...]:
    """The levels a deployment declares, in vocabulary order.

    Clients offer exactly these levels, so a picker never lists a level the
    model would have to fall back from. The full vocabulary is the fallback
    only when the deployment cannot be resolved at all.
    """
    try:
        provider = config.get_provider_for_model(model)
    except ValueError:
        return THINKING_LEVELS
    levels = get_thinking_levels(str(provider.backend), provider.api_style, model.name)
    if levels is None:
        return ("off",)
    declared = declared_thinking_levels(levels, model.supported_thinking_levels)
    return cast(tuple[ThinkingLevel, ...], declared) or THINKING_LEVELS


def _project_model_config(
    config: ChartreuxConfigSchema, model: ModelConfig
) -> ModelConfigView:
    return ModelConfigView(
        name=model.name,
        alias=model.alias,
        thinking=model.thinking,
        supports_images=model.supports_images,
        display_name=format_model_display_name(model.provider, model.name),
        thinking_levels=list(project_thinking_levels(config, model)),
    )


def project_stats(agent_loop: AgentLoop) -> AgentStatsSnapshot:
    stats = agent_loop.stats
    return AgentStatsSnapshot(
        steps=stats.steps,
        session_prompt_tokens=stats.session_prompt_tokens,
        session_completion_tokens=stats.session_completion_tokens,
        session_cached_tokens=stats.session_cached_tokens,
        input_price_per_million=stats.input_price_per_million,
        output_price_per_million=stats.output_price_per_million,
        cached_input_price_per_million=stats.cached_input_price_per_million,
        known_cost_total=stats.known_cost_total,
        has_unknown_cost=stats.has_unknown_cost,
        tool_calls_agreed=stats.tool_calls_agreed,
        tool_calls_rejected=stats.tool_calls_rejected,
        tool_calls_failed=stats.tool_calls_failed,
        tool_calls_succeeded=stats.tool_calls_succeeded,
        context_tokens=stats.context_tokens,
        last_turn_prompt_tokens=stats.last_turn_prompt_tokens,
        last_turn_completion_tokens=stats.last_turn_completion_tokens,
        last_turn_cached_tokens=stats.last_turn_cached_tokens,
        last_turn_duration=stats.last_turn_duration,
        tokens_per_second=stats.tokens_per_second,
    )


def project_skill_summaries(skills: Iterable[SkillInfo]) -> list[SkillSummary]:
    return [
        SkillSummary.model_validate({
            "name": skill.name,
            "description": skill.description,
            "prompt": skill.prompt,
            "user_invocable": skill.user_invocable,
            "source": skill.source.value,
            "scope": skill.scope.value,
        })
        for skill in skills
    ]


def project_skills(agent_loop: AgentLoop) -> list[SkillSummary]:
    return project_skill_summaries(agent_loop.skill_manager.available_skills.values())


def project_installed_skills(agent_loop: AgentLoop) -> list[SkillSummary]:
    """Discovered local skills, excluding bundled skills."""
    return project_skill_summaries(
        info
        for info in agent_loop.skill_manager.available_skills.values()
        if info.source is SkillSource.LOCAL
    )


def project_tools(agent_loop: AgentLoop) -> list[ToolSummary]:
    custom_tool_names = agent_loop.tool_manager.custom_tool_names
    return [
        ToolSummary(name=name, is_custom=name in custom_tool_names)
        for name in agent_loop.tool_manager.available_tools
    ]


def project_mcp(
    agent_loop: AgentLoop, *, discovery_errors: Mapping[str, str] | None = None
) -> MCPState:
    tools = _project_mcp_tools(agent_loop)
    discovery_errors_set = set(discovery_errors) if discovery_errors else set()
    return MCPState(
        sources=[*_project_mcp_servers(agent_loop, tools, discovery_errors_set)],
        discovery_errors=dict(discovery_errors or {}),
    )


def _project_mcp_tools(agent_loop: AgentLoop) -> dict[str, list[MCPToolSummary]]:
    tools: dict[str, list[MCPToolSummary]] = {}
    available = agent_loop.tool_manager.available_tools
    for tool_name, tool_class in agent_loop.tool_manager.registered_tools.items():
        if not issubclass(tool_class, MCPTool):
            continue
        source_name = tool_class.get_server_name()
        if source_name is None:
            continue
        tools.setdefault(source_name, []).append(
            MCPToolSummary(
                name=tool_class.get_remote_name(),
                description=format_tool_display_description(
                    tool_class.description, source_name=source_name
                ),
                enabled=tool_name in available,
            )
        )
    return tools


def _project_mcp_servers(
    agent_loop: AgentLoop,
    tools: dict[str, list[MCPToolSummary]],
    discovery_errors: set[str],
) -> list[MCPSourceSummary]:
    registry = agent_loop.mcp_registry
    server_statuses = registry.status() if registry is not None else {}
    sources: list[MCPSourceSummary] = []
    for server in agent_loop.config.mcp_servers:
        if server.disabled:
            status = MCPSourceStatus.DISABLED
        elif server.name in discovery_errors:
            status = MCPSourceStatus.UNAVAILABLE
        else:
            match server_statuses.get(server.name):
                case AuthStatus.NEEDS_AUTH:
                    status = MCPSourceStatus.NEEDS_AUTH
                case AuthStatus.OK:
                    status = MCPSourceStatus.CONNECTED
                case _:
                    status = MCPSourceStatus.ENABLED
        sources.append(
            MCPSourceSummary(
                name=server.name,
                transport=server.transport,
                status=status,
                tools=sorted(tools.get(server.name, []), key=lambda tool: tool.name),
            )
        )
    return sources


def project_session_log(agent_loop: AgentLoop) -> SessionLogSummary:
    session_logger = agent_loop.session_logger
    session_dir = session_logger.session_dir if session_logger.persisted else None
    return SessionLogSummary(
        enabled=session_logger.enabled,
        session_id=session_logger.session_id,
        persisted=session_logger.persisted,
        path=str(session_dir) if session_dir is not None else None,
        title=session_logger.title,
        needs_initial_auto_title=session_logger.needs_initial_auto_title(),
    )


def committed_model_recovery_issue(agent_loop: AgentLoop) -> ConfigIssue | None:
    """A session resumed after its committed deployment left the catalog.

    The resume deliberately loads the transcript without the unresolvable
    committed identity, so the pending state is derivable: the stored launch
    envelope still commits a model while the loop no longer carries one. The
    issue clears once a newly chosen model is committed by the next turn.
    """
    metadata = agent_loop.session_logger.session_metadata
    envelope = metadata.launch_config if metadata is not None else None
    if (
        not isinstance(envelope, LaunchMetadataV2)
        or agent_loop.committed_model is not None
    ):
        return None
    identity = envelope.committed_model
    return ConfigIssue(
        file=COMMITTED_MODEL_RECOVERY_ISSUE_FILE,
        message=(
            f"The model committed to this session ({identity.provider}/"
            f"{identity.wire_name}) is no longer available. Pick a model to "
            "continue this session."
        ),
    )


def project_diagnostics(agent_loop: AgentLoop) -> tuple[list[ConfigIssue], int]:
    issues = [
        *(_project_issue(issue) for issue in agent_loop.hook_config_issues),
        *(_project_issue(issue) for issue in agent_loop.skill_manager.config_issues),
    ]
    if (recovery := committed_model_recovery_issue(agent_loop)) is not None:
        issues.append(recovery)
    return issues, agent_loop.hooks_count


def project_debug_logs(logs: PaginatedLogs) -> DebugLogPage:
    return DebugLogPage(
        entries=[
            DebugLogEntry(
                id=hashlib.sha1(
                    entry.raw_line.encode("utf-8"), usedforsecurity=False
                ).hexdigest(),
                timestamp=entry.timestamp,
                ppid=entry.ppid,
                pid=entry.pid,
                level=entry.level,
                message=entry.message,
                raw_line=entry.raw_line,
            )
            for entry in logs.entries
        ],
        has_more=logs.has_more,
        cursor=logs.cursor,
    )


def project_history(agent_loop: AgentLoop) -> list[PublicHistoryEntry]:
    """The visible conversation, reconstructed from disk when a prefix is sealed.

    Without authoritative archive state the projection walks the live
    message list exactly as before. With it, the sealed prefix streams from
    the transcript through the shared archive-read interface one record at
    a time and is interleaved slot-for-slot with the live list, so trimming
    the RAM copies never hides or discards an old turn. Display keeps the
    complete conversation — nothing is paged out of it or lazily loaded.
    """
    return project_message_history(
        agent_loop.session_id,
        history_messages(agent_loop),
        agent_loop.session_logger.session_metadata,
    )


def history_messages(agent_loop: AgentLoop) -> Iterable[LLMMessage]:
    """The message view the visible conversation is reconstructed from.

    A sealed prefix is read back from the persisted transcript so the
    originals, not the payload-trimmed RAM copies, drive the projection. The
    prefix is verified against its descriptor BEFORE any of it is projected,
    and a prefix that cannot be verified degrades to the live view: display
    never shows bytes the archive authority does not vouch for, and the
    fallback loses no turn the live copies still carry.
    """
    source = archive_transcript_source(agent_loop)
    if source is None:
        return agent_loop.messages
    try:
        verify_archived_prefix_on_disk(source.descriptor, source.path)
    except ArchiveIntegrityError as exc:
        _warn_unreadable_prefix(exc)
        return agent_loop.messages
    return _archived_then_live(agent_loop.messages, source)


_MISSING = object()


def _archived_then_live(
    messages: Iterable[LLMMessage], source: ArchiveTranscriptSource
) -> Iterator[LLMMessage]:
    """Interleave the sealed disk prefix with the live tail, slot for slot.

    The descriptor's ``archived_count`` counts the persisted non-system
    records, so every non-system position before it is replaced by the
    streamed original and every position after it yields the live message
    unchanged. Sealed turns the live list no longer holds are still yielded
    (the disk is the record of what the conversation contained), and the
    closing drain forces the prefix's hash and count verification.
    """
    archived = iter_archived_messages(source)
    remaining = source.descriptor.archived_count
    readable = True
    for message in messages:
        if message.role is Role.system:
            yield message
            continue
        if remaining and readable:
            try:
                hydrated = next(archived, _MISSING)
            except ArchiveIntegrityError as exc:
                _warn_unreadable_prefix(exc)
                readable = False
                hydrated = _MISSING
            if hydrated is not _MISSING:
                remaining -= 1
                yield cast(LLMMessage, hydrated)
                continue
            readable = False
        yield message
    if not readable:
        return
    try:
        for leftover in archived:
            if remaining == 0:
                continue
            remaining -= 1
            yield leftover
    except ArchiveIntegrityError as exc:
        _warn_unreadable_prefix(exc)


def _warn_unreadable_prefix(exc: ArchiveIntegrityError) -> None:
    logger.warning(
        "Sealed transcript prefix is unreadable; projecting the live view: %s", exc
    )


def project_message_content(
    text: str | None,
    images: Sequence[CoreImageAttachment] | None,
    resources: Sequence[UserResource] | None = None,
) -> list[ContentBlock]:
    content: list[ContentBlock] = []
    if text:
        content.append(TextContentBlock(text=text))
    content.extend(
        ImageContentBlock(
            attachment=ImageAttachment.model_validate(image.model_dump(mode="json"))
        )
        for image in images or []
    )
    content.extend(
        ResourceContentBlock(resource=resource) for resource in resources or []
    )
    return content


def project_message_history(
    session_id: str, messages: Iterable[LLMMessage], metadata: SessionMetadata | None
) -> list[PublicHistoryEntry]:
    timestamp = now_ms()
    entries: list[PublicHistoryEntry] = []
    effect_indices: dict[str, int] = {}
    # Harness-injected notices — delivered completion notifications, the P2b
    # sweep's synthesized backstop notifications, and recovery notices — all
    # arrive as injected messages here and project uniformly: only shell
    # effects gain a history entry, text-only injected notices project none.
    child_sessions = (
        {link.tool_call_id: link.session_id for link in metadata.child_sessions}
        if metadata is not None
        else {}
    )
    # First: the worktree is created before the session has a single message.
    if metadata is not None and metadata.created_worktree is not None:
        entries.append(_history_worktree(session_id, metadata.created_worktree))
    for index, message in enumerate(messages):
        _project_stored_message(
            session_id,
            message,
            index,
            timestamp + index,
            entries,
            effect_indices,
            child_sessions,
        )
    return entries


def _project_stored_message(
    session_id: str,
    message: LLMMessage,
    index: int,
    created_at: int,
    entries: list[PublicHistoryEntry],
    effect_indices: dict[str, int],
    child_sessions: dict[str, str],
) -> None:
    if message.role is Role.system:
        return
    if message.context_boundary == "compaction":
        _append_compaction_history(session_id, message, index, created_at, entries)
        return
    if message.injected:
        _append_injected_history(session_id, message, entries)
        return
    match message.role:
        case Role.user:
            entries.append(
                _history_user_message(session_id, message, index, created_at)
            )
        case Role.assistant:
            _append_assistant_history(
                session_id,
                message,
                index,
                created_at,
                entries,
                effect_indices,
                child_sessions,
            )
        case Role.tool:
            _apply_tool_history(
                session_id,
                message,
                index,
                created_at,
                entries,
                effect_indices,
                child_sessions,
            )


def _history_worktree(session_id: str, worktree: WorktreeContext) -> PublicEffectEntry:
    restored = WorktreeEffect.restored(worktree)
    return PublicEffectEntry(
        **_history_fields(session_id, worktree.entry_id, worktree.created_at),
        title="worktree",
        detail=restored.detail,
        state=restored.state,
    )


def _append_injected_history(
    session_id: str, message: LLMMessage, entries: list[PublicHistoryEntry]
) -> None:
    shell = message.manual_shell
    if shell is None:
        return
    entries.append(
        PublicEffectEntry(
            **_history_fields(session_id, shell.operation_id, shell.created_at),
            title="shell",
            detail=shell_effect_detail(shell.command),
            state=restored_shell_effect_state(shell),
        )
    )


def _append_compaction_history(
    session_id: str,
    message: LLMMessage,
    index: int,
    created_at: int,
    entries: list[PublicHistoryEntry],
) -> None:
    message_id = message.message_id or f"history:{index}:compaction"
    entries.append(
        PublicCheckpointEntry(
            **_history_fields(
                session_id, f"checkpoint:compaction:{message_id}", created_at
            ),
            kind="compaction",
            message="Context compacted",
            details={},
        )
    )


def _history_user_message(
    session_id: str, message: LLMMessage, index: int, created_at: int
) -> PublicMessageEntry:
    return PublicMessageEntry(
        **_history_fields(session_id, history_message_id(message, index), created_at),
        role="user",
        posted_at=message.posted_at,
        content=project_message_content(
            message.input_text if message.input_text is not None else message.content,
            message.images,
            message.resources,
        ),
        source="harness",
        user_display_content=message.user_display_content,
    )


def _append_assistant_history(
    session_id: str,
    message: LLMMessage,
    index: int,
    created_at: int,
    entries: list[PublicHistoryEntry],
    effect_indices: dict[str, int],
    child_sessions: dict[str, str],
) -> None:
    if message.reasoning_content:
        entries.append(
            PublicReasoningEntry(
                **_history_fields(
                    session_id,
                    message.reasoning_message_id or f"history:{index}:reasoning",
                    created_at,
                ),
                text=message.reasoning_content,
                event_at=message.reasoning_at,
            )
        )
    if message.content:
        entries.append(
            PublicMessageEntry(
                **_history_fields(
                    session_id,
                    message.message_id or f"history:{index}:assistant",
                    created_at,
                ),
                role="assistant",
                posted_at=message.posted_at,
                turn_duration_ms=(
                    message.turn_duration * 1000
                    if message.turn_duration is not None
                    else None
                ),
                content=[TextContentBlock(text=message.content)],
                source="harness",
            )
        )
    for tool_call in message.tool_calls or []:
        tool_call_id = tool_call.id or f"history:{index}:effect"
        effect_indices[tool_call_id] = len(entries)
        entries.append(
            _history_effect(
                session_id,
                tool_call_id,
                tool_call.function.name or "unknown",
                tool_call.function.arguments,
                created_at,
                event_at=tool_call.started_at,
                presentation=tool_call.presentation,
                child_session_id=child_sessions.get(tool_call_id),
            )
        )


def _apply_tool_history(
    session_id: str,
    message: LLMMessage,
    index: int,
    created_at: int,
    entries: list[PublicHistoryEntry],
    effect_indices: dict[str, int],
    child_sessions: dict[str, str],
) -> None:
    tool_call_id = message.tool_call_id or ""
    effect_index = effect_indices.get(tool_call_id)
    if effect_index is None:
        entries.append(
            _history_effect(
                session_id,
                tool_call_id or f"history:{index}:effect",
                message.name or "tool",
                None,
                created_at,
                result_message=message,
                child_session_id=child_sessions.get(tool_call_id),
            )
        )
        return
    effect = entries[effect_index]
    if not isinstance(effect, PublicEffectEntry):
        return
    entries[effect_index] = effect.model_copy(
        update={"state": _persisted_effect_state(effect, message)}
    )


def history_message_id(message: LLMMessage, index: int) -> str:
    return message.message_id or f"history:{index}:{message.role.value}"


def history_user_message_index(agent_loop: AgentLoop, entry_id: str) -> int | None:
    for index, message in enumerate(agent_loop.messages):
        if message.role is not Role.user or message.injected:
            continue
        if history_message_id(message, index) == entry_id:
            return index
    return None


class _ConfigIssue(Protocol):
    file: Path
    message: str


def _project_issue(issue: _ConfigIssue) -> ConfigIssue:
    return ConfigIssue(file=str(issue.file), message=issue.message)


class _HistoryFields(TypedDict):
    id: str
    session_id: str
    turn_id: str | None
    created_at: int
    updated_at: int
    generation_status: PublicEntryGenerationStatus
    related_entry_id: str | None


def _history_fields(session_id: str, entry_id: str, created_at: int) -> _HistoryFields:
    return {
        "id": entry_id,
        "session_id": session_id,
        "turn_id": None,
        "created_at": created_at,
        "updated_at": created_at,
        "generation_status": PublicEntryGenerationStatus.COMPLETED,
        "related_entry_id": None,
    }


def _history_effect(
    session_id: str,
    entry_id: str,
    tool_name: str,
    raw_arguments: str | None,
    created_at: int,
    *,
    event_at: int | None = None,
    result_message: LLMMessage | None = None,
    presentation: ToolCallPresentation | None = None,
    child_session_id: str | None = None,
) -> PublicEffectEntry:
    arguments = _parse_arguments(raw_arguments)
    if presentation is not None:
        detail = project_effect_detail(tool_name, arguments, presentation)
    else:
        summary = _generic_call_summary(tool_name, arguments)
        display = EffectCallDisplay(
            summary=summary,
            verb="Running",
            message=summary,
            settled_verb="Ran",
            settled_message=summary,
            status_text=f"Running {tool_name}",
        )
        detail = GenericEffectDetail(
            tool_name=tool_name, input=arguments, display=display
        )
    if isinstance(detail, SubagentEffectDetail):
        detail = detail.model_copy(update={"child_session_id": child_session_id})
    entry = PublicEffectEntry(
        **_history_fields(session_id, entry_id, created_at),
        title=tool_name,
        detail=detail,
        state=CompletedEffectState(
            output=None,
            output_text="",
            display=EffectResultDisplay(success=True, message=f"{tool_name} completed"),
        ),
        event_at=event_at,
    )
    return entry.model_copy(
        update={"state": _persisted_effect_state(entry, result_message)}
    )


def _persisted_effect_state(
    effect: PublicEffectEntry, message: LLMMessage | None
) -> EffectState:
    if message is None:
        reason = "Tool did not complete before the session ended"
        return CancelledEffectState(
            reason=reason,
            output_text="",
            display=EffectResultDisplay(success=False, message=reason),
        )
    text = TaggedText.from_string(message.content or "")
    persisted = message.tool_result
    duration_ms = (
        persisted.duration * 1000
        if persisted is not None and persisted.duration is not None
        else None
    )
    display_message = text.message or f"{effect.detail.tool_name} completed"
    if text.tag == CANCELLATION_TAG:
        return CancelledEffectState(
            reason=display_message,
            output_text=text.message,
            duration_ms=duration_ms,
            display=EffectResultDisplay(success=False, message=display_message),
        )
    if text.tag == TOOL_ERROR_TAG:
        return FailedEffectState(
            error=PublicError(message=display_message),
            output_text=text.message,
            duration_ms=duration_ms,
            display=EffectResultDisplay(success=False, message=display_message),
        )
    if persisted is not None:
        presentation = persisted.presentation
        display = (
            presentation.display
            if presentation is not None
            else EffectResultDisplay(success=True, message=display_message)
        )
        if persisted.cancelled:
            return CancelledEffectState(
                reason=display.message,
                output_text=text.message,
                duration_ms=duration_ms,
                display=display,
            )
        kind = presentation.kind if presentation is not None else effect.detail.kind
        output = (
            presentation.projected_output
            if presentation is not None and presentation.projected_output is not None
            else persisted.output
        )
        return CompletedEffectState(
            output=project_effect_output_value(kind, output),
            output_text=text.message,
            duration_ms=duration_ms,
            display=display,
        )
    return CompletedEffectState(
        output=None,
        output_text=text.message,
        display=EffectResultDisplay(success=True, message=display_message),
    )


def _parse_arguments(value: str | None) -> JsonValue:
    if value is None:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return value
    return parsed


def _generic_call_summary(tool_name: str, arguments: JsonValue) -> str:
    if not isinstance(arguments, dict):
        return tool_name
    rendered = ", ".join(
        f"{key}={value!r}" for key, value in list(arguments.items())[:3]
    )
    return f"{tool_name}({rendered})"
