from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Protocol, TypedDict

from pydantic import JsonValue

from chartreux.app_server._session_model import active_model_is_pinned
from chartreux.app_server._shell import restored_shell_effect_state, shell_effect_detail
from chartreux.app_server._tool_projection import (
    project_effect_detail,
    project_effect_output_value,
)
from chartreux.app_server._utils import now_ms
from chartreux.app_server._worktree_effects import WorktreeEffect
from chartreux.app_server.config import ConfigView, ModelConfigView
from chartreux.app_server.models import (
    AgentStatsSnapshot,
    AgentSummary,
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
from chartreux.core.agents import AgentProfile
from chartreux.core.config import ChartreuxConfigSchema, ModelConfig
from chartreux.core.llm_models import (
    ImageAttachment as CoreImageAttachment,
    LLMMessage,
    Role,
)
from chartreux.core.log_reader import PaginatedLogs
from chartreux.core.model_catalog.resolver import ModelResolutionError
from chartreux.core.session_types import SessionMetadata, WorktreeContext
from chartreux.core.skills.models import SkillInfo, SkillSource
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
from chartreux.user_content import UserResource
from chartreux.utils.mcp import format_tool_display_description
from chartreux.utils.tool_presentation import ToolCallPresentation


def project_config(agent_loop: AgentLoop) -> ConfigView:
    return project_config_view(
        agent_loop.config,
        active_model_pinned=active_model_is_pinned(agent_loop.config_orchestrator),
    )


def project_config_view(
    config: ChartreuxConfigSchema, *, active_model_pinned: bool = False
) -> ConfigView:
    return ConfigView(
        active_model=_project_model_config(config.get_active_model()),
        active_model_pinned=active_model_pinned,
        # The configured default, never the active model: clients render it as
        # the "Default (currently X)" hint, which must stay stable while a pin
        # is in effect.
        default_model_alias=_project_default_model_alias(config),
        theme=config.theme,
        log_level=config.log_level,
        disable_welcome_banner_animation=config.disable_welcome_banner_animation,
        show_greeting=config.show_greeting,
        autocopy_to_clipboard=config.autocopy_to_clipboard,
        file_watcher_for_autocomplete=config.file_watcher_for_autocomplete,
        ask_confirmation_on_exit=config.ask_confirmation_on_exit,
        show_thinking_nodes=config.show_thinking_nodes,
        enable_notifications=config.enable_notifications,
        enable_system_trust_store=config.enable_system_trust_store,
        models=[
            _project_model_config(model) for model in config.available_models().values()
        ],
        validation_warnings=[
            *config.validation_warnings,
            *_web_search_validation_warnings(config),
        ],
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


def _project_model_config(model: ModelConfig) -> ModelConfigView:
    return ModelConfigView(
        name=model.name,
        alias=model.alias,
        thinking=model.thinking,
        supports_images=model.supports_images,
        display_name=format_model_display_name(model.provider, model.name),
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


def project_agent_summaries(
    active: AgentProfile, available: Iterable[AgentProfile]
) -> tuple[AgentSummary, list[AgentSummary]]:
    return _project_agent(active), [_project_agent(profile) for profile in available]


def project_agents(agent_loop: AgentLoop) -> tuple[None, list[AgentSummary]]:
    return None, [
        _project_agent(profile)
        for profile in agent_loop.agent_manager.available_agents.values()
    ]


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


def project_diagnostics(agent_loop: AgentLoop) -> tuple[list[ConfigIssue], int]:
    issues = [
        *(_project_issue(issue) for issue in agent_loop.hook_config_issues),
        *(_project_issue(issue) for issue in agent_loop.skill_manager.config_issues),
    ]
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
    return project_message_history(
        agent_loop.session_id,
        agent_loop.messages,
        agent_loop.session_logger.session_metadata,
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
    session_id: str, messages: Sequence[LLMMessage], metadata: SessionMetadata | None
) -> list[PublicHistoryEntry]:
    timestamp = now_ms()
    entries: list[PublicHistoryEntry] = []
    effect_indices: dict[str, int] = {}
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


def _project_agent(profile: AgentProfile) -> AgentSummary:
    return AgentSummary(
        name=profile.name,
        display_name=profile.display_name,
        description=profile.description,
        safety=profile.safety,
        agent_type=profile.agent_type,
    )


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
    display_message = text.message or f"{effect.detail.tool_name} completed"
    if text.tag == CANCELLATION_TAG:
        return CancelledEffectState(
            reason=display_message,
            output_text=text.message,
            display=EffectResultDisplay(success=False, message=display_message),
        )
    if text.tag == TOOL_ERROR_TAG:
        return FailedEffectState(
            error=PublicError(message=display_message),
            output_text=text.message,
            display=EffectResultDisplay(success=False, message=display_message),
        )
    persisted = message.tool_result
    if persisted is not None:
        presentation = persisted.presentation
        display = (
            presentation.display
            if presentation is not None
            else EffectResultDisplay(success=True, message=display_message)
        )
        duration_ms = (persisted.duration or 0.0) * 1000
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
