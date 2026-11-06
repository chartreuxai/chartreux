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
    # The levels this deployment declares, in vocabulary order. Empty when the
    # projection could not resolve them; clients then offer the full vocabulary.
    thinking_levels: list[ThinkingLevel] = Field(default_factory=list)


class ProxySettingsView(ProtocolModel):
    values: dict[str, str | None]
    descriptions: dict[str, str]


class StatusLineConfigView(ProtocolModel):
    segments: list[str] = Field(default_factory=lambda: ["directory", "pid", "context"])
    directory_style: Literal["name", "path"] = "name"
    context_style: Literal["tokens", "tokens-percent"] = "tokens-percent"
    separator: Literal["space", "pipe"] = "pipe"


class RouteTargetView(ProtocolModel):
    """One purpose-route target as the conversation's live table carries it.

    ``preview`` is the model a launch would resolve right now, or ``None``
    with ``error`` when the binding cannot resolve. It is always a preview,
    never a launch guarantee (a launch-time marker resolves again at launch).
    """

    seat: str | None = None
    identity: str
    tier: str
    overlay: str | None = None
    binding: str
    marker: str | None = None
    preview: str | None = None
    error: str | None = None


class RouteView(ProtocolModel):
    """One purpose route of the live ``[orchestration.routes]`` table.

    ``requires_distinct_models`` and ``resolved_distinct_models`` are reported
    side by side; ``resolved_distinct_models`` is ``None`` when any target
    could not resolve, so the count is unknown rather than claimed.
    ``reload_status`` and ``launch_status`` are separate facts: a route that
    binds only launch-time markers passes reload validation while its
    launch-time diversity can still fail.
    """

    purpose: str
    requires_distinct_models: int
    resolved_distinct_models: int | None = None
    reload_status: str
    reload_detail: str | None = None
    launch_status: str
    launch_detail: str | None = None
    targets: list[RouteTargetView] = Field(default_factory=list)


class EffectiveThinkingView(ProtocolModel):
    """The next request's thinking estimate on the committed deployment.

    ``requested`` is the level the session selected, ``semantic`` the level
    after launch-time semantic coercion (retained Anthropic thinking keeps
    thinking on), ``effective`` the declared level nearest the semantic one,
    and ``api_effort`` the provider's wire value. ``known`` is false when the
    estimate cannot be resolved, and clients must then show ``unknown`` rather
    than imply the requested level is effective. ``matches_requested`` is the
    same comparison made once server-side, including the nearest-level
    fallback. The estimate describes the next request only: an in-flight call
    keeps the encoding it started with, and failover may pick another
    deployment.
    """

    requested: ThinkingLevel
    semantic: ThinkingLevel | None = None
    effective: ThinkingLevel | None = None
    api_effort: str | None = None
    known: bool = False
    matches_requested: bool | None = None


class ConfigView(ProtocolModel):
    active_model: ModelConfigView
    active_model_expression: str = ""
    allowed_models: list[str] = Field(default_factory=list)
    # Whether the user has pinned a specific model, vs. the "default" (unpinned)
    active_model_pinned: bool
    # The next request's thinking estimate. ``None`` when the projection has
    # no conversation context to resolve (a sessionless configuration read);
    # clients render no annotation then.
    effective_thinking: EffectiveThinkingView | None = None
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
    emit_program_status: bool = True
    enable_system_trust_store: bool = False
    models: list[ModelConfigView]
    validation_warnings: list[str]
    # The conversation's live purpose-route table, inspected against the
    # configuration's own catalog. Empty with ``routes_error`` set when the
    # table cannot be inspected at all.
    routes: list[RouteView] = Field(default_factory=list)
    routes_error: str | None = None

    def model_display_name(self, alias: str) -> str:
        """User-facing name for a configured alias, or the alias when unknown."""
        return next(
            (model.display_name for model in self.models if model.alias == alias), alias
        )

    @property
    def default_model_display_name(self) -> str:
        return self.model_display_name(self.default_model_alias)
