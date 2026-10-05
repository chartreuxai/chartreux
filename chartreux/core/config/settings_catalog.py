"""Curated settings surface, not an automatically generated schema editor.

ProjectContextConfig and SessionLoggingConfig are BaseSettings: environment variables
may own effective leaf values without appearing in layer-dict provenance. In that
case the displayed origin may say "default" despite an environment override.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, cast

from pydantic import JsonValue
import tomli_w

from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
from chartreux.core.config.models import (
    ProjectContextConfig,
    SessionLoggingConfig,
    StatusLineConfig,
)


@dataclass(frozen=True)
class SettingDescriptor:
    path: str
    label: str
    description: str
    kind: Literal["bool", "enum", "int", "float", "str", "list", "link", "deferred"]
    group: str
    choices: tuple[str, ...] = ()
    item_kind: Literal["path", "pattern"] | None = None
    control: Literal["checklist", "toggle_inventory", "status_line"] | None = None
    inventory: Literal["tools", "skills", "agents"] | None = None
    minimum: int | float | None = None
    exclusive_minimum: bool = False
    empty: str = "Empty text is a saved value; use Remove User Override for Not Set."
    risk: Literal["routine", "needs-confirmation"] = "routine"
    timing: Literal["after-save", "next-turn", "next-conversation", "next-launch"] = (
        "after-save"
    )
    timing_verified: bool = False
    command: str | None = None

    def validate(self, value: JsonValue) -> None:
        validate_setting_value(
            self.path,
            self.kind,
            self.choices,
            self.minimum,
            self.exclusive_minimum,
            value,
        )


def validate_setting_value(
    path: str,
    kind: Literal["bool", "enum", "int", "float", "str", "list", "link", "deferred"],
    choices: tuple[str, ...],
    minimum: int | float | None,
    exclusive_minimum: bool,
    value: JsonValue,
) -> None:
    """Validate a setting value independently of its wire representation."""
    if path == "status_line.segments":
        StatusLineConfig.model_validate({"segments": value})
        return
    if kind == "bool" and type(value) is bool:
        return
    if kind == "str" and isinstance(value, str):
        return
    if (
        kind == "list"
        and isinstance(value, list)
        and all(isinstance(item, str) for item in value)
    ):
        return
    if kind == "enum" and isinstance(value, str) and value in choices:
        return
    if kind == "int" and isinstance(value, int) and not isinstance(value, bool):
        numeric = value
    elif (
        kind == "float"
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    ):
        numeric = value
    else:
        raise ValueError(f"Invalid value for {path}")
    if not math.isfinite(numeric) or (
        minimum is not None
        and (numeric <= minimum if exclusive_minimum else numeric < minimum)
    ):
        raise ValueError(f"Invalid value for {path}")


def _field(
    path: str,
    label: str,
    description: str,
    kind: Literal["bool", "int", "float", "str"],
    group: str,
    **kwargs: object,
) -> SettingDescriptor:
    return SettingDescriptor(path, label, description, kind, group, **kwargs)  # type: ignore[arg-type]


EDITABLE_SETTINGS: tuple[SettingDescriptor, ...] = (
    _field(
        "show_greeting",
        "Show Greeting",
        "Show the startup greeting for Mistral providers, at most once per day.",
        "bool",
        "Interface",
    ),
    _field(
        "autocopy_to_clipboard",
        "Auto-Copy",
        "Copy selected text to the clipboard automatically.",
        "bool",
        "Interface",
    ),
    _field(
        "ask_confirmation_on_exit",
        "Confirm Exit",
        "Ask before closing the conversation.",
        "bool",
        "Interface",
    ),
    _field(
        "file_watcher_for_autocomplete",
        "Watch Files for Completion",
        "Update file completion when project files change.",
        "bool",
        "Interface",
    ),
    _field(
        "disable_welcome_banner_animation",
        "Disable Welcome Animation",
        "Show the welcome banner without animation.",
        "bool",
        "Interface",
    ),
    _field(
        "context_warnings",
        "Context Warnings",
        "Show context usage warnings.",
        "bool",
        "Interface",
    ),
    _field(
        "show_thinking_nodes",
        "Show Thinking Nodes",
        "Display thinking nodes in conversation output.",
        "bool",
        "Interface",
    ),
    _field(
        "ascii_chrome",
        "ASCII Chrome",
        "Use ASCII markers and navigation glyphs in the interface.",
        "bool",
        "Interface",
    ),
    _field(
        "displayed_workdir",
        "Displayed Workdir",
        "Workdir label displayed in the UI; empty uses the default display.",
        "str",
        "Interface",
        empty="Empty uses the default display; Remove User Override restores inheritance.",
    ),
    _field(
        "enable_notifications",
        "Notifications",
        "Enable desktop notifications.",
        "bool",
        "Interface",
    ),
    SettingDescriptor(
        "status_line.segments",
        "Status Line Segments",
        "Ordered segments: directory, pid, model, context, git-branch, spend-today, "
        "spend-week, spend-month. Directory and context are required; no duplicates. "
        "Spend segments show recorded USD spend across all projects for the local "
        "calendar day, Monday-start week, or month (Today/Week/Month). The Usage "
        "browser's Current project filter never changes this scope. "
        "+ marks partly unknown cost; "
        "Unknown means nothing priced; — means unavailable, not zero spend.",
        "list",
        "Interface",
        empty="Directory and context are required; an empty list cannot be saved.",
    ),
    SettingDescriptor(
        "status_line.directory_style",
        "Status Line Directory Style",
        "Show the directory name or full path.",
        "enum",
        "Interface",
        choices=("name", "path"),
    ),
    SettingDescriptor(
        "status_line.context_style",
        "Status Line Context Style",
        "Show context tokens alone or with the percentage of the compaction threshold.",
        "enum",
        "Interface",
        choices=("tokens", "tokens-percent"),
    ),
    SettingDescriptor(
        "status_line.separator",
        "Status Line Separator",
        "Separate status line segments with spaces or pipes.",
        "enum",
        "Interface",
        choices=("space", "pipe"),
    ),
    _field(
        "show_message_timestamps",
        "Show Message Timestamps",
        "Show posting times, whole-turn totals, and settled tool durations; capture continues when off.",
        "bool",
        "Interface",
    ),
    _field(
        "include_commit_signature",
        "Commit Signature",
        "Include the configured commit signature in prompts.",
        "bool",
        "Prompts & Compaction",
    ),
    _field(
        "include_model_info",
        "Model Info",
        "Include model information in prompts.",
        "bool",
        "Prompts & Compaction",
    ),
    _field(
        "include_project_context",
        "Project Context",
        "Include project context in prompts.",
        "bool",
        "Prompts & Compaction",
    ),
    _field(
        "include_prompt_detail",
        "Prompt Detail",
        "Include additional detail in prompts.",
        "bool",
        "Prompts & Compaction",
    ),
    _field(
        "raise_on_compaction_failure",
        "Raise on Compaction Failure",
        "Surface a compaction error instead of continuing silently.",
        "bool",
        "Prompts & Compaction",
    ),
    _field(
        "auto_compact_threshold",
        "Fallback Compaction Threshold",
        "Fallback token count for models without their own threshold; 0 disables automatic compaction.",
        "int",
        "Prompts & Compaction",
    ),
    _field(
        "project_context.default_commit_count",
        "Default Commit Count",
        "Number of recent commits included in project context; 0 requests none.",
        "int",
        "Project Context",
    ),
    _field(
        "project_context.timeout_seconds",
        "Project Context Timeout",
        "Maximum seconds to gather project context (capped at 10 seconds); 0 allows no time for gathering.",
        "float",
        "Project Context",
    ),
    _field(
        "subagents.idle_ttl_seconds",
        "Idle Agent Lifetime",
        "Seconds before an idle subagent is retired; 0 disables TTL eviction.",
        "int",
        "Subagents",
        minimum=0,
    ),
    _field(
        "subagents.max_idle_agents",
        "Maximum Idle Agents",
        "Maximum idle subagents retained; 0 disables the count cap.",
        "int",
        "Subagents",
        minimum=0,
    ),
    _field(
        "session_logging.enabled",
        "Session Logging",
        "Save conversation history to disk.",
        "bool",
        "Session History",
    ),
    _field(
        "session_logging.save_dir",
        "Session Log Directory",
        "Directory for saved sessions; empty selects the default log directory.",
        "str",
        "Session History",
        empty="Empty selects the default directory; Remove User Override restores inheritance.",
    ),
    _field(
        "session_logging.session_prefix",
        "Session Filename Prefix",
        "Prefix used for session log filenames.",
        "str",
        "Session History",
    ),
    _field(
        "session_logging.generate_titles",
        "Generate Titles",
        "Generate background session titles; off uses the first-message preview.",
        "bool",
        "Session History",
    ),
    _field(
        "api_timeout",
        "API Request Timeout",
        "Timeout for API requests in seconds.",
        "float",
        "Network",
    ),
    _field(
        "api_connect_timeout",
        "API Connect Timeout",
        "Connection timeout in seconds.",
        "float",
        "Network",
    ),
    _field(
        "api_write_timeout",
        "API Write Timeout",
        "Write timeout in seconds.",
        "float",
        "Network",
    ),
    _field(
        "api_pool_timeout",
        "API Pool Timeout",
        "Connection-pool timeout in seconds.",
        "float",
        "Network",
    ),
    _field(
        "api_retry_max_elapsed_time",
        "API Retry Budget",
        "Maximum elapsed seconds for retries.",
        "float",
        "Network",
    ),
    _field(
        "enable_system_trust_store",
        "System Trust Store",
        "Trust operating-system certificate authorities for API connections.",
        "bool",
        "Network",
        risk="needs-confirmation",
    ),
    SettingDescriptor(
        "system_prompt_id",
        "System Prompt",
        "Prompt used for conversations.",
        "enum",
        "Prompts & Compaction",
    ),
    SettingDescriptor(
        "compaction_prompt_id",
        "Compaction Prompt",
        "Prompt used to compact context.",
        "enum",
        "Prompts & Compaction",
    ),
    *(
        SettingDescriptor(
            path,
            label,
            description,
            "list",
            "Advanced & Tools",
            item_kind=cast(Literal["path", "pattern"], item_kind),
            empty=empty,
        )
        for path, label, description, item_kind, empty in (
            (
                "agent_paths",
                "Agent Paths",
                "Additional agent directories (paths).",
                "path",
                "Empty uses built-in agent locations.",
            ),
            (
                "skill_paths",
                "Skill Paths",
                "Additional skill directories (paths).",
                "path",
                "Empty uses built-in skill locations.",
            ),
            (
                "enabled_agents",
                "Enabled Agents",
                "Allowed agent names, globs, or re: patterns.",
                "pattern",
                "Empty allows all agents (subject to disabled_agents).",
            ),
            (
                "disabled_agents",
                "Disabled Agents",
                "Blocked agent names, globs, or re: patterns.",
                "pattern",
                "Empty blocks none; ignored when enabled_agents is nonempty.",
            ),
            (
                "enabled_skills",
                "Enabled Skills",
                "Allowed skill names, globs, or re: patterns.",
                "pattern",
                "Empty allows all skills (subject to disabled_skills).",
            ),
            (
                "disabled_skills",
                "Disabled Skills",
                "Blocked skill names, globs, or re: patterns.",
                "pattern",
                "Empty blocks none; ignored when enabled_skills is nonempty.",
            ),
            (
                "tool_paths",
                "Tool Paths",
                "Additional tool directories or files (paths).",
                "path",
                "Empty uses built-in tool locations.",
            ),
            (
                "enabled_tools",
                "Enabled Tools",
                "Allowed tool names, globs, or re: patterns.",
                "pattern",
                "Empty allows all tools (subject to disabled_tools).",
            ),
            (
                "disabled_tools",
                "Disabled Tools",
                "Blocked tool names, globs, or re: patterns, including enabled_tools matches.",
                "pattern",
                "Empty blocks none; applies even when enabled_tools is nonempty.",
            ),
        )
    ),
)

STATUS_LINE_PATHS = tuple(
    item.path for item in EDITABLE_SETTINGS if item.path.startswith("status_line.")
)
STATUS_LINE_SETTING = SettingDescriptor(
    "status_line",
    "Status line",
    "Choose and reorder status line segments, directory and context styles, and the "
    "separator. Directory and context are required. Spend segments show recorded "
    "USD spend across all projects for the local calendar day, Monday-start week, "
    "or month (Today/Week/Month). The Usage browser's Current project filter never "
    "changes this scope. + marks partly unknown cost; Unknown means nothing priced; "
    "— means unavailable, not zero spend.",
    "list",
    "Interface",
    control="status_line",
)

VISIBLE_SETTINGS: tuple[SettingDescriptor, ...] = (
    *(
        STATUS_LINE_SETTING if item.path == "status_line.segments" else item
        for item in EDITABLE_SETTINGS
        if not item.path.startswith(("enabled_", "disabled_"))
        and (item.path not in STATUS_LINE_PATHS or item.path == "status_line.segments")
    ),
    *(
        SettingDescriptor(
            f"inventory_{category}",
            category.title(),
            f"All {category} are on by default; uncheck to disable. "
            f"Nonempty enabled_{category} enables only matches (allow-only); "
            + (
                "disabled_tools still blocks matching tools."
                if category == "tools"
                else f"disabled_{category} is ignored then."
            ),
            "list",
            "Advanced & Tools",
            control="toggle_inventory",
            inventory=cast(Literal["tools", "skills", "agents"], category),
        )
        for category in ("agents", "skills", "tools")
    ),
)

DEFERRED_SETTINGS: tuple[SettingDescriptor, ...] = ()

LINK_SETTINGS: tuple[SettingDescriptor, ...] = tuple(
    SettingDescriptor(
        path,
        label,
        f"Edit with {command}.",
        "link",
        "Advanced & Tools",
        command=command,
    )
    for path, label, command in (
        ("theme", "Theme", "/theme"),
        ("log_level", "Log level", "/log-level"),
        ("mcp_servers", "MCP servers", "/mcp"),
        ("models/providers", "Provider Settings", "/providers"),
        ("tools/web_search", "Web Search", "/web-search"),
        ("proxy", "Proxy setup", "/proxy-setup"),
    )
)

EXCLUDED_SETTINGS = frozenset({
    "authorized_roots_by_project",
    "credential_env_passthrough",
    "session_logging.permission_repair_dir",
    "tools.<name>.allowlist",
    "tools.<name>.denylist",
    "tools.<name>.sensitive_patterns",
})

EDITABLE_BY_PATH = {item.path: item for item in EDITABLE_SETTINGS}


def render_initial_user_config() -> str:
    """Render editable schema defaults as a commented user-layer TOML file."""

    # Supply every nested BaseSettings field explicitly so environment values
    # cannot leak into the bootstrap file through schema default factories.
    def built_in_fields(
        schema: type[ProjectContextConfig] | type[SessionLoggingConfig],
    ) -> dict[str, object]:
        return {
            key: field.get_default(call_default_factory=True)
            for key, field in schema.model_fields.items()
        }

    defaults = ChartreuxConfigSchema.model_construct(
        project_context=ProjectContextConfig.model_validate(
            built_in_fields(ProjectContextConfig)
        ),
        session_logging=SessionLoggingConfig.model_validate(
            built_in_fields(SessionLoggingConfig)
        ),
    ).model_dump(mode="json")
    lines = [
        "# Chartreux user configuration.",
        "#",
        "# These are the built-in defaults for editable settings. Edit values here",
        "# to customize them. This file pins today's defaults explicitly; later",
        "# built-in default changes will not apply until you edit this file.",
        "# Configuration is layered, lowest to highest precedence:",
        "# built-in defaults, this file, a trusted project `.chartreux/config.toml`,",
        "# `CHARTREUX_` environment variables, the active agent profile, and runtime",
        "# overrides. Higher layers can override values in this file.",
    ]

    # TOML root keys must precede the first table. Keep sections in first-seen
    # registry order and fields in registry order within each section.
    root_groups = dict.fromkeys(
        item.group for item in EDITABLE_SETTINGS if "." not in item.path
    )
    for section in root_groups:
        lines.extend(("", f"# {section}"))
        for item in EDITABLE_SETTINGS:
            if "." in item.path or item.group != section:
                continue
            value = defaults[item.path]
            lines.extend((
                f"# {item.description}",
                tomli_w.dumps({item.path: value}).strip(),
            ))

    tables = dict.fromkeys(
        item.path.partition(".")[0] for item in EDITABLE_SETTINGS if "." in item.path
    )
    for table in tables:
        entries = [
            item for item in EDITABLE_SETTINGS if item.path.startswith(f"{table}.")
        ]
        lines.extend(("", f"# {entries[0].group}", f"[{table}]"))
        for item in entries:
            key = item.path.partition(".")[2]
            value = defaults[table][key]
            lines.extend((f"# {item.description}", tomli_w.dumps({key: value}).strip()))
    return "\n".join((*lines, ""))
