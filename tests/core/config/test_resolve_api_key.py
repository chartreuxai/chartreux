from __future__ import annotations

import pytest

from chartreux.utils.api_keys import (
    ApiKeyOrigin,
    ApiKeySource,
    resolve_api_key,
    resolve_api_key_with_origin,
)


def test_resolve_returns_env_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUSTOM_API_KEY", "env-key")

    assert resolve_api_key("CUSTOM_API_KEY") == "env-key"
    assert resolve_api_key_with_origin("CUSTOM_API_KEY") == (
        "env-key",
        ApiKeyOrigin(ApiKeySource.ENVIRONMENT, "CUSTOM_API_KEY"),
    )


def test_resolve_returns_none_for_empty_env_key() -> None:
    assert resolve_api_key("") is None
    assert resolve_api_key_with_origin("") is None


def test_resolve_returns_none_when_env_var_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUSTOM_API_KEY", raising=False)

    assert resolve_api_key("CUSTOM_API_KEY") is None
    assert resolve_api_key_with_origin("CUSTOM_API_KEY") is None
