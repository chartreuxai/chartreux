from __future__ import annotations

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
    show_greeting: bool
    autocopy_to_clipboard: bool
    file_watcher_for_autocomplete: bool
    ask_confirmation_on_exit: bool
    show_thinking_nodes: bool
    enable_notifications: bool
    enable_system_trust_store: bool = False
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
