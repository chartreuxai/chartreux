from __future__ import annotations

from acp.schema import (
    ContentToolCallContent,
    Implementation,
    SessionConfigOptionSelect,
    SessionConfigSelectOption,
    TextContentBlock,
    ToolCallProgress,
    ToolCallStart,
)

from chartreux.app_server.config import THINKING_LEVELS, ConfigView, ProxySettingsView


def is_jetbrains_client(client_info: Implementation | None) -> bool:
    return bool(client_info and client_info.name.startswith("JetBrains."))


def build_model_config(config: ConfigView) -> SessionConfigOptionSelect:
    return SessionConfigOptionSelect(
        id="model",
        name="Model",
        current_value=config.active_model.alias,
        category="model",
        type="select",
        options=[
            SessionConfigSelectOption(
                value=model.alias, name=model.display_name, description=model.name
            )
            for model in config.models
        ],
    )


def make_thinking_response(config: ConfigView) -> SessionConfigOptionSelect:
    return SessionConfigOptionSelect(
        id="thinking",
        name="Thinking",
        current_value=config.active_model.thinking,
        category="thinking",
        type="select",
        options=[
            SessionConfigSelectOption(value=level, name=level.capitalize())
            for level in THINKING_LEVELS
        ],
    )


def compact_start_update(tool_call_id: str) -> ToolCallStart:
    return ToolCallStart(
        session_update="tool_call",
        tool_call_id=tool_call_id,
        title="Compacting conversation history...",
        kind="other",
        status="in_progress",
        content=[
            ContentToolCallContent(
                type="content",
                content=TextContentBlock(
                    type="text",
                    text=(
                        "Automatic context management, no approval required. "
                        "This may take some time..."
                    ),
                ),
            )
        ],
    )


def compact_end_update(tool_call_id: str, message: str) -> ToolCallProgress:
    return ToolCallProgress(
        session_update="tool_call_update",
        tool_call_id=tool_call_id,
        title="Compacted conversation history",
        status="completed",
        content=[
            ContentToolCallContent(
                type="content", content=TextContentBlock(type="text", text=message)
            )
        ],
    )


def compact_error_update(tool_call_id: str, message: str) -> ToolCallProgress:
    return ToolCallProgress(
        session_update="tool_call_update",
        tool_call_id=tool_call_id,
        title="Compaction failed",
        status="failed",
        raw_output=message,
    )


def get_proxy_help_text(settings: ProxySettingsView) -> str:
    lines = [
        "## Proxy Configuration",
        "",
        "Configure proxy and SSL settings for HTTP requests.",
        "",
        "### Usage:",
        "- `/proxy-setup` - Show this help and current settings",
        "- `/proxy-setup KEY value` - Set an environment variable",
        "- `/proxy-setup KEY` - Remove an environment variable",
        "",
        "### Supported Variables:",
        *[
            f"- `{key}`: {description}"
            for key, description in settings.descriptions.items()
        ],
        "",
        "### Current Settings:",
    ]
    configured = [
        f"- `{key}={value}`" for key, value in settings.values.items() if value
    ]
    lines.extend(configured or ["- (none configured)"])
    return "\n".join(lines)
