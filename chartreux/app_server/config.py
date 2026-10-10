from __future__ import annotations

from typing import Literal

from pydantic import Field

from chartreux.app_server._model import ProtocolModel
from chartreux.config_values import (
    THINKING_LEVELS as THINKING_LEVELS,
    ThinkingLevel as ThinkingLevel,
)


class ModelConfigView(ProtocolModel):
    name: str
    alias: str
    thinking: ThinkingLevel
    supports_images: bool
    display_name: str


class ProxySettingsView(ProtocolModel):
    values: dict[str, str | None]
    descriptions: dict[str, str]


class StatusLineConfigView(ProtocolModel):
    segments: list[str] = Field(default_factory=lambda: ["directory", "pid", "context"])
    directory_style: Literal["name", "path"] = "name"
    context_style: Literal["tokens", "tokens-percent"] = "tokens-percent"
    separator: Literal["space", "pipe"] = "pipe"


class ConfigView(ProtocolModel):
    active_model: ModelConfigView
    active_model_expression: str = ""
    allowed_models: list[str] = Field(default_factory=list)
    # Whether the user has pinned a specific model, vs. the "default" (unpinned)
    active_model_pinned: bool
    default_model_alias: str
    theme: str
    log_level: str | None
    disable_welcome_banner_animation: bool
    autocopy_to_clipboard: bool
    file_watcher_for_autocomplete: bool
    ask_confirmation_on_exit: bool
    show_thinking_nodes: bool
    show_message_timestamps: bool = True
    status_line: StatusLineConfigView = Field(default_factory=StatusLineConfigView)
    ascii_chrome: bool = False
    enable_notifications: bool
    enable_system_trust_store: bool = False
    dispatch_mode: Literal["standalone", "orchestrated"] | None = None
    # Distinct canonical identities bound by the session's dispatch policy.
    # None when no policy is attached (for example before a session starts);
    # an attached policy that bound no slots projects an empty list, which is
    # a live empty roster and must not be confused with "no policy".
    bound_roster: list[str] | None = None
    models: list[ModelConfigView]
    validation_warnings: list[str]

    def model_display_name(self, alias: str) -> str:
        """User-facing name for a configured alias, or the alias when unknown."""
        return next(
            (model.display_name for model in self.models if model.alias == alias), alias
        )

    @property
    def default_model_display_name(self) -> str:
        return self.model_display_name(self.default_model_alias)
