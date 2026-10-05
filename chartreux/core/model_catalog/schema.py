"""Validated, immutable definitions for the model catalog."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
import re
from types import MappingProxyType
from typing import Any, Literal
from unicodedata import category
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

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

    @field_serializer("extra_headers")
    @classmethod
    def serialize_extra_headers(cls, value: Mapping[str, str]) -> dict[str, str]:
        return dict(value)


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


def valid_provider_name(value: str) -> str:
    """Validate a persisted provider identity without changing its spelling."""
    name = value.strip()
    if (
        not name
        or any(char in name for char in "/@")
        or any(category(char) == "Cc" for char in name)
    ):
        raise ValueError(
            "Provider name must be non-empty and contain no '/', '@', or control characters"
        )
    return name


class DeploymentDefinition(_FrozenCatalogModel):
    provider: str
    name: str
    prices: Prices = Field(default_factory=Prices)
    supports_images: bool = False
    supported_thinking_levels: tuple[str, ...] | None = None
    auto_compact_threshold: int | None = None
    disabled: bool = False

    @field_validator("provider")
    @classmethod
    def provider_name_is_valid(cls, value: str) -> str:
        return valid_provider_name(value)

    @field_validator("name")
    @classmethod
    def no_reserved_at(cls, value: str) -> str:
        if "@" in value:
            raise ValueError("'@' is reserved in deployment names")
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

    @field_validator("auto_compact_threshold", mode="before")
    @classmethod
    def auto_compact_threshold_is_positive_integer(cls, value: Any) -> int | None:
        if value is None:
            return None
        error = "auto_compact_threshold must be a positive whole integer"
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(error)
        if isinstance(value, float) and (not isfinite(value) or not value.is_integer()):
            raise ValueError(error)
        if value <= 0:
            raise ValueError(error)
        return int(value)


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
            raise ValueError(
                "Only one existing provider deployment per model is allowed"
            )
        return self


class RoleDefinition(_FrozenCatalogModel):
    """One named default preset selecting a canonical model and thinking level."""

    description: str = ""
    model: str = Field(min_length=1)
    thinking: str

    @model_validator(mode="before")
    @classmethod
    def reject_legacy_members(cls, value: Any) -> Any:
        if isinstance(value, Mapping) and "models" in value:
            raise ValueError(
                "Role 'models' lists are no longer supported; set one 'model' "
                "and one 'thinking' level for this preset"
            )
        return value

    @field_validator("model")
    @classmethod
    def canonical_model_name(cls, value: str) -> str:
        if "@" in value or not value.strip():
            raise ValueError("role model must be a non-empty canonical model name")
        return value

    @field_validator("thinking")
    @classmethod
    def thinking_is_known(cls, value: str) -> str:
        if value not in _THINKING_LEVELS:
            raise ValueError("role thinking must be a known thinking level")
        return value


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
    def provider_names_are_valid(
        cls, value: Mapping[str, ProviderDefinition]
    ) -> Mapping[str, ProviderDefinition]:
        providers: dict[str, ProviderDefinition] = {}
        original_names: dict[str, str] = {}
        for raw_name, definition in value.items():
            name = valid_provider_name(raw_name)
            if name in providers:
                raise ValueError(
                    f"Provider name collision: {original_names[name]!r} and {raw_name!r}"
                )
            providers[name] = definition
            original_names[name] = raw_name
        return MappingProxyType(providers)

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
        for name in value:
            if "@" in name:
                raise ValueError("'@' is reserved in role names")
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
        # A saved provider/model draft may temporarily leave a preset pointing
        # at an unavailable model. Finish checks runnability after setup.
        return self
