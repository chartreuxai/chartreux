"""Configuration schema, proxy, fields/settings introspection, and config write wire models."""

from __future__ import annotations

from enum import StrEnum, auto
from typing import Literal

from pydantic import Field, JsonValue, model_validator

from chartreux.app_server._model import ProtocolModel
from chartreux.app_server.config import ConfigView, ProxySettingsView
from chartreux.app_server.models import ConfigIssue
from chartreux.app_server.protocol.common import NonNegativeStrictInt
from chartreux.app_server.protocol.runtime import RuntimeMutationResponse
from chartreux.core.config.settings_catalog import (
    STATUS_LINE_PATHS as STATUS_LINE_PATHS,
    TASK_IDENTITY_DENYLIST_PATH as TASK_IDENTITY_DENYLIST_PATH,
    validate_setting_value,
)


class ConfigSchemaReadParams(ProtocolModel):
    pass


class ConfigSchemaReadResponse(ProtocolModel):
    config_schema_version: str
    config_schema: dict[str, JsonValue] = Field(alias="schema")


class ConfigReloadParams(ProtocolModel):
    session_id: str
    reload_runtime: bool = True


class ConfigProxyReadParams(ProtocolModel):
    session_id: str


class ConfigProxyReadResponse(ProtocolModel):
    settings: ProxySettingsView


class ConfigProxyWriteParams(ProtocolModel):
    session_id: str
    changes: dict[str, str | None]


class SessionSettingsUpdateParams(ProtocolModel):
    session_id: str
    max_turns: NonNegativeStrictInt | None = None
    max_tokens: NonNegativeStrictInt | None = None

    @model_validator(mode="after")
    def require_update(self) -> SessionSettingsUpdateParams:
        if self.max_turns is None and self.max_tokens is None:
            raise ValueError("At least one session setting must be provided")
        return self


class ConfigMutationResponse(RuntimeMutationResponse):
    stripped_history_images: int = 0
    launch_metadata_persisted: bool = True


class ConfigFieldKind(StrEnum):
    BOOL = auto()
    ENUM = auto()
    INT = auto()
    FLOAT = auto()
    STR = auto()
    LIST = auto()
    COMPLEX = auto()


class ConfigLayerValueWire(ProtocolModel):
    layer: str
    value: JsonValue = None


class ConfigFieldWire(ProtocolModel):
    name: str
    kind: ConfigFieldKind
    description: str
    value: JsonValue = None
    path: str
    popular: bool = False
    enum_choices: list[str] = Field(default_factory=list)
    value_labels: dict[str, str] = Field(default_factory=dict)
    layer_values: list[ConfigLayerValueWire] = Field(default_factory=list)
    writable_targets: list[str] = Field(default_factory=list)

    @property
    def origin(self) -> str:
        return self.layer_values[0].layer if self.layer_values else "default"


class ConfigFieldsReadParams(ProtocolModel):
    session_id: str


class ConfigFieldsReadResponse(ProtocolModel):
    fields: list[ConfigFieldWire]
    targets: list[str]
    revisions: dict[str, str] = Field(default_factory=dict)


class SettingsReadParams(ProtocolModel):
    session_id: str


class SettingLeafWire(ProtocolModel):
    path: str
    effective_value: JsonValue
    origin: str
    saved_explicit: bool
    saved_value: JsonValue = None


class WebSearchSettingsWire(ProtocolModel):
    fields: list[SettingLeafWire]
    invalid_fields: list[str] = Field(default_factory=list)
    readiness: Literal["ready", "missing_key", "invalid"]
    readiness_message: str | None = None
    credential_env_var: str | None = None
    default_credential_env_vars: dict[str, str | None] = Field(default_factory=dict)


type InventoryCategory = Literal["tools", "skills", "identities"]


class SettingDescriptorWire(ProtocolModel):
    path: str
    label: str
    description: str
    kind: Literal["bool", "enum", "int", "float", "str", "list", "link", "deferred"]
    group: str
    choices: tuple[str, ...] = ()
    item_kind: Literal["path", "pattern"] | None = None
    control: (
        Literal["checklist", "toggle_inventory", "status_line", "shell_permissions"]
        | None
    ) = None
    inventory: InventoryCategory | None = None
    minimum: int | float | None = None
    exclusive_minimum: bool = False
    empty: str = "Empty text is a saved value; use Remove User Override for Not Set."
    risk: Literal["routine", "needs-confirmation"] = "routine"
    timing: Literal["after-save", "next-turn", "next-conversation", "next-launch"] = (
        "after-save"
    )
    timing_verified: bool = False
    command: str | None = None

    def validate_value(self, value: JsonValue) -> None:
        validate_setting_value(
            self.path,
            self.kind,
            self.choices,
            self.minimum,
            self.exclusive_minimum,
            value,
        )


class InventoryItemStateWire(ProtocolModel):
    effective: bool
    default_effective: bool
    pattern_driven: bool


class SettingsReadResponse(ProtocolModel):
    fields: list[SettingLeafWire]
    web_search: WebSearchSettingsWire | None = None
    catalog: list[SettingDescriptorWire] = Field(default_factory=list)
    backing_settings: dict[str, SettingDescriptorWire] = Field(default_factory=dict)
    inventories: dict[InventoryCategory, list[str]] = Field(default_factory=dict)
    inventory_states: dict[InventoryCategory, dict[str, InventoryItemStateWire]] = (
        Field(default_factory=dict)
    )
    user_layer: str | None = None
    user_revision: str | None = None

    @property
    def view_only(self) -> bool:
        return self.user_revision is None


class ConfigWriteOpWire(ProtocolModel):
    op: Literal["set", "remove"]
    path: str
    value: JsonValue = None
    target_layer: str | None = None


class ConfigWriteParams(ProtocolModel):
    session_id: str
    ops: list[ConfigWriteOpWire]
    reason: str = "config write"
    reload_runtime: bool = False
    target: Literal["session", "user", "project"] = "session"
    expected_revision: str | None = None


class ConfigWriteResponse(ConfigMutationResponse):
    rejected: bool = False
    failures: list[str] = Field(default_factory=list)
    target: Literal["session", "user", "project"] = "session"
    persistence: Literal["not_saved", "saved", "durability_uncertain"] = "not_saved"
    application: Literal["unchanged", "applied", "failed"] = "unchanged"
    revision: str | None = None
    fields: list[ConfigFieldWire] = Field(default_factory=list)
    saved_values: dict[str, JsonValue] = Field(default_factory=dict)


class ConfigReadParams(ProtocolModel):
    session_id: str | None = None
    cwd: str | None = None


class ConfigReadResponse(ProtocolModel):
    config: ConfigView
    startup_issue: ConfigIssue | None = None
    stripped_history_images: int = 0
    skills_count: int = 0
    hooks_count: int = 0
    mcp_servers_total: int = 0
    mcp_servers_enabled: int = 0
