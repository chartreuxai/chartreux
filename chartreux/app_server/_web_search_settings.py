"""Safe, leaf-only settings projection for the web search editor."""

from __future__ import annotations

import re
from typing import Any, Literal
from urllib.parse import urlsplit

from chartreux.app_server.protocol import SettingLeafWire, WebSearchSettingsWire
from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
from chartreux.core.tools.builtins.web_search import (
    ResolvedSearchProvider,
    SearchProviderDiagnostic,
    WebSearchConfig,
    effective_web_search_config,
    resolve_web_search_provider,
)

_FIELDS = (
    "permission",
    "provider",
    "api_key_env_var",
    "base_url",
    "timeout",
    "max_results",
    "model",
)
_FIELD_LABELS = {
    "permission": "permission",
    "provider": "provider",
    "api_key_env_var": "credential variable",
    "base_url": "base URL",
    "timeout": "timeout",
    "max_results": "maximum results",
    "model": "model",
}
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_INVALID = "[invalid]"
_MAX_ENV_LENGTH = 256
_MAX_FIELD_LENGTH = 512


def _safe_env_name(value: object) -> str | None:
    return (
        value
        if isinstance(value, str)
        and len(value) <= _MAX_ENV_LENGTH
        and _ENV_NAME.fullmatch(value)
        else None
    )


def _safe_leaf_value(name: str, value: Any) -> Any:
    """Never reflect nested malformed data or URL-embedded credentials."""
    if name in {"timeout", "max_results"}:
        safe = value if type(value) is int else _INVALID
    elif name in {"api_key_env_var", "base_url"} and value is None:
        safe = None
    elif not isinstance(value, str) or len(value) > _MAX_FIELD_LENGTH:
        safe = _INVALID
    elif name == "api_key_env_var" and value and not _safe_env_name(value):
        safe = _INVALID
    elif name == "base_url" and value:
        try:
            parsed = urlsplit(value)
        except ValueError:
            safe = _INVALID
        else:
            safe = (
                "[redacted]"
                if parsed.username or parsed.password or parsed.query or parsed.fragment
                else value
            )
    else:
        safe = value
    return safe


def _project_fields(
    values: dict[str, Any],
    defaults: dict[str, Any],
    layers: list[tuple[str, dict[str, Any]]],
    user_layer: str | None,
    user_unavailable: bool,
    fallback: bool,
) -> tuple[list[SettingLeafWire], list[str]]:
    fields: list[SettingLeafWire] = []
    invalid_fields: set[str] = set()
    for name in _FIELDS:
        path = f"tools.web_search.{name}"
        leaf_values: list[tuple[str, Any]] = []
        for layer_name, data in reversed(layers):
            search = _search_values(data)
            if name in search:
                safe = _safe_leaf_value(name, search[name])
                if (
                    safe != search[name]
                    and layer_name == user_layer
                    and not user_unavailable
                ):
                    invalid_fields.add(path)
                leaf_values.append((layer_name, safe))
        effective_value = values.get(name, defaults[name])
        safe_effective = _safe_leaf_value(name, effective_value)
        if safe_effective != effective_value:
            invalid_fields.add(path)
        saved = next(
            (value for layer_name, value in leaf_values if layer_name == user_layer),
            None,
        )
        saved_explicit = (
            any(layer_name == user_layer for layer_name, _ in leaf_values)
            and not user_unavailable
        )
        fields.append(
            SettingLeafWire(
                path=path,
                effective_value=safe_effective,
                origin=(
                    "live config"
                    if fallback
                    else leaf_values[0][0]
                    if leaf_values
                    else "default"
                ),
                saved_explicit=saved_explicit,
                saved_value=saved if saved_explicit else None,
            )
        )
    return fields, sorted(invalid_fields)


def _search_values(data: dict[str, Any]) -> dict[str, Any]:
    tools = data.get("tools")
    search = tools.get("web_search") if isinstance(tools, dict) else None
    return search if isinstance(search, dict) else {}


def _default_credential_env_vars(
    config: ChartreuxConfigSchema,
) -> dict[str, str | None]:
    try:
        mistral = config.get_mistral_provider()
    except (AttributeError, ValueError):
        mistral = None
    mistral_env = (
        _safe_env_name(mistral.api_key_env_var)
        if mistral is not None
        else "MISTRAL_API_KEY"
    )
    return {
        "auto": mistral_env,
        "mistral": mistral_env,
        "exa": "EXA_API_KEY",
        "brave": "BRAVE_SEARCH_API_KEY",
        "duckduckgo": None,
    }


def project_web_search_settings(
    config: ChartreuxConfigSchema,
    layers: list[tuple[str, dict[str, Any]]],
    *,
    user_layer: str | None,
    user_unavailable: bool,
    fallback: bool,
) -> WebSearchSettingsWire:
    """Project only supported values and names, never resolved key material."""
    defaults = WebSearchConfig().model_dump(mode="json")
    effective = effective_web_search_config(config)
    if isinstance(effective, WebSearchConfig):
        values = effective.model_dump(mode="json")
    else:
        values = {**defaults, **_search_values(config.model_dump(mode="json"))}
    fields, invalid_fields = _project_fields(
        values, defaults, layers, user_layer, user_unavailable, fallback
    )

    resolved = resolve_web_search_provider(effective, config)
    readiness: Literal["ready", "missing_key", "invalid"] = "ready"
    message: str | None = None
    if isinstance(resolved, SearchProviderDiagnostic):
        if (
            resolved.env_var
            and resolved.config_key == "tools.web_search.api_key_env_var"
        ):
            readiness = "missing_key"
            message = "Credential is missing for the selected web search provider."
        else:
            readiness = "invalid"
            # Validation diagnostics can echo malformed values, including URLs.
            message = "Web search configuration is invalid. Review the saved settings."
            diagnostic_key = resolved.config_key
            if diagnostic_key and diagnostic_key.startswith("tools.web_search."):
                field_name = diagnostic_key.removeprefix("tools.web_search.")
                if field_name in _FIELD_LABELS:
                    invalid_fields = sorted({*invalid_fields, diagnostic_key})
                    message = (
                        f"Invalid web search {_FIELD_LABELS[field_name]}; "
                        "review that setting."
                    )

    env_vars = _default_credential_env_vars(config)
    provider = values.get("provider")
    override_env = values.get("api_key_env_var")
    credential_env = None
    if isinstance(provider, str) and provider != "duckduckgo":
        credential_env = (
            override_env if _safe_env_name(override_env) else env_vars.get(provider)
        )
    if (
        isinstance(resolved, ResolvedSearchProvider)
        and resolved.provider == "duckduckgo"
    ):
        credential_env = None
    if readiness == "missing_key" and credential_env is None:
        readiness = "invalid"
        message = "Credential variable name is invalid. Review the provider settings."
    return WebSearchSettingsWire(
        fields=fields,
        invalid_fields=invalid_fields,
        readiness=readiness,
        readiness_message=message,
        credential_env_var=credential_env,
        default_credential_env_vars=env_vars,
    )
