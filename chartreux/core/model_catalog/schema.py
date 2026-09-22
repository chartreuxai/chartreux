"""Validated, immutable definitions for the model catalog."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
import re
from types import MappingProxyType
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from chartreux.core.llm.thinking_levels import (
    ANTHROPIC_THINKING_LEVELS,
    GLM_5_3_THINKING_LEVELS,
    MISTRAL_THINKING_LEVELS,
    OPENAI_RESPONSES_THINKING_LEVELS,
    OPENAI_THINKING_LEVELS,
)

_ENV_VAR_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_THINKING_LEVELS = frozenset().union(
    MISTRAL_THINKING_LEVELS,
    OPENAI_THINKING_LEVELS,
    OPENAI_RESPONSES_THINKING_LEVELS,
    ANTHROPIC_THINKING_LEVELS,
    GLM_5_3_THINKING_LEVELS,
)


def _mutable_catalog_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _mutable_catalog_value(value.model_dump(mode="python"))
    if isinstance(value, Mapping):
        return {key: _mutable_catalog_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_mutable_catalog_value(item) for item in value)
    if isinstance(value, list):
        return [_mutable_catalog_value(item) for item in value]
    return value


class _FrozenCatalogModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> _FrozenCatalogModel:
        """Catalog values are immutable and safely shared by copied snapshots."""
        return self


class ProviderDefinition(_FrozenCatalogModel):
    api_base: str
    api_key_env_var: str = ""
    api_style: Literal["openai", "openai-responses", "anthropic"] = "openai"
    backend: str = "generic"
    reasoning_field_name: str = "reasoning_content"
    emits_finish_reason: bool = True
    # Whether Responses-compatible providers accept images in function-call output.
    # Set False to project tool-result images into a synthetic user turn instead.
    supports_tool_result_images: bool = True
    extra_headers: Mapping[str, str] = Field(
        default_factory=lambda: MappingProxyType({})
    )
    disabled: bool = False

    @field_validator("api_base")
    @classmethod
    def api_base_is_url(cls, value: str) -> str:
        parsed = urlsplit(value)
        if not value or parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("api_base must be a non-empty HTTP(S) URL")
        return value

    @field_validator("api_key_env_var")
    @classmethod
    def api_key_env_var_is_valid(cls, value: str) -> str:
        if value and not _ENV_VAR_NAME.fullmatch(value):
            raise ValueError(
                "api_key_env_var must be a valid environment-variable name"
            )
        return value

    @field_validator("extra_headers")
    @classmethod
    def freeze_extra_headers(cls, value: Mapping[str, str]) -> Mapping[str, str]:
        return MappingProxyType(dict(value))


class Prices(_FrozenCatalogModel):
    """Token prices; ``None`` means unknown and ``0.0`` means explicitly free."""

    input: float | None = None
    output: float | None = None
    cached_input: float | None = None

    @field_validator("input", "output", "cached_input")
    @classmethod
    def prices_are_finite_and_nonnegative(cls, value: float | None) -> float | None:
        if value is not None and (not isfinite(value) or value < 0):
            raise ValueError("prices must be finite and non-negative")
        return value


class DeploymentDefinition(_FrozenCatalogModel):
    provider: str
    name: str
    prices: Prices = Field(default_factory=Prices)
    supports_images: bool = False
    supported_thinking_levels: tuple[str, ...] | None = None
    auto_compact_threshold: float | None = None
    disabled: bool = False

    @field_validator("provider", "name")
    @classmethod
    def no_reserved_at(cls, value: str) -> str:
        if "@" in value:
            raise ValueError("'@' is reserved in deployment names and provider IDs")
        return value

    @field_validator("supported_thinking_levels")
    @classmethod
    def supported_thinking_levels_are_known(
        cls, value: tuple[str, ...] | None
    ) -> tuple[str, ...] | None:
        if value is not None and any(level not in _THINKING_LEVELS for level in value):
            raise ValueError(
                "supported_thinking_levels contains an unknown thinking level"
            )
        return value

    @field_validator("auto_compact_threshold")
    @classmethod
    def auto_compact_threshold_is_positive(cls, value: float | None) -> float | None:
        if value is not None and (not isfinite(value) or value <= 0):
            raise ValueError("auto_compact_threshold must be finite and positive")
        return value


class BaseModelDefinition(_FrozenCatalogModel):
    thinking: str = "medium"
    temperature: float | None = None
    deployments: tuple[DeploymentDefinition, ...] = Field(min_length=1)
    disabled: bool = False

    @field_validator("thinking")
    @classmethod
    def thinking_is_known(cls, value: str) -> str:
        if value not in _THINKING_LEVELS:
            raise ValueError("thinking must be a known thinking level")
        return value

    @model_validator(mode="after")
    def unique_provider_deployments(self) -> BaseModelDefinition:
        providers = [deployment.provider for deployment in self.deployments]
        if len(providers) != len(set(providers)):
            raise ValueError("Only one deployment per (base, provider) is allowed")
        return self


class RoleDefinition(_FrozenCatalogModel):
    """An ordered, documented priority list of canonical model names."""

    description: str = ""
    models: tuple[str, ...] = Field(min_length=1)


class ModelCatalog(_FrozenCatalogModel):
    providers: Mapping[str, ProviderDefinition]
    models: Mapping[str, BaseModelDefinition]
    roles: Mapping[str, RoleDefinition] = Field(
        default_factory=lambda: MappingProxyType({})
    )

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        """Serialize immutable containers as ordinary data for loaders and writers."""
        kwargs["mode"] = "python"
        kwargs.setdefault("warnings", False)
        return _mutable_catalog_value(super().model_dump(**kwargs))

    def model_dump_json(
        self, *, indent: int | None = None, ensure_ascii: bool = False, **kwargs: Any
    ) -> str:
        """Serialize immutable containers through the plain-data dump path."""
        from pydantic_core import to_json

        return to_json(
            self.model_dump(**kwargs), indent=indent, ensure_ascii=ensure_ascii
        ).decode()

    @field_validator("providers")
    @classmethod
    def provider_ids_are_qualified(
        cls, value: Mapping[str, ProviderDefinition]
    ) -> Mapping[str, ProviderDefinition]:
        for provider_id in value:
            if "/" not in provider_id:
                raise ValueError("Provider IDs must contain '/'")
            if "@" in provider_id:
                raise ValueError("'@' is reserved in provider IDs")
        return MappingProxyType(dict(value))

    @field_validator("models")
    @classmethod
    def model_names_without_at(
        cls, value: Mapping[str, BaseModelDefinition]
    ) -> Mapping[str, BaseModelDefinition]:
        if any("@" in name for name in value):
            raise ValueError("'@' is reserved in model names")
        return MappingProxyType(dict(value))

    @field_validator("roles")
    @classmethod
    def nonempty_unique_roles(
        cls, value: Mapping[str, RoleDefinition]
    ) -> Mapping[str, RoleDefinition]:
        for name, definition in value.items():
            members = definition.models
            if "@" in name:
                raise ValueError("'@' is reserved in role names")
            if len(members) != len(set(members)):
                raise ValueError("Duplicate role members are not allowed")
            if any("@" in member for member in members):
                raise ValueError("'@' is reserved in role members")
        return MappingProxyType(dict(value))

    @model_validator(mode="after")
    def references_exist(self) -> ModelCatalog:
        for base_name, definition in self.models.items():
            for deployment in definition.deployments:
                if deployment.provider not in self.providers:
                    raise ValueError(
                        f"Model {base_name!r} references unknown provider "
                        f"{deployment.provider!r}"
                    )
        for role_name, definition in self.roles.items():
            unknown = set(definition.models) - set(self.models)
            if unknown:
                raise ValueError(
                    f"Role {role_name!r} references unknown models: {sorted(unknown)!r}"
                )
        return self
