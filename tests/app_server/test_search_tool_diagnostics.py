from __future__ import annotations

import pytest

from chartreux.app_server._projection import project_config_view
from tests.conftest import build_test_vibe_config


def test_web_search_invalid_provider_diagnostic_is_bounded() -> None:
    config = build_test_vibe_config(tools={"web_search": {"provider": "x" * 100_000}})

    warnings = project_config_view(config).validation_warnings

    assert len(warnings) == 1
    assert len(warnings[0]) <= 500
    assert "provider" in warnings[0]
    assert warnings[0].endswith("...")


@pytest.mark.parametrize(
    ("tools", "missing_key"),
    [
        ({"web_search": {"provider": "auto"}}, "MISTRAL_API_KEY"),
        ({"web_search": {"provider": "exa"}}, "EXA_API_KEY"),
    ],
)
def test_web_search_missing_credentials_are_actionable_and_current(
    monkeypatch: pytest.MonkeyPatch, tools: dict[str, dict[str, str]], missing_key: str
) -> None:
    monkeypatch.delenv(missing_key, raising=False)
    config = build_test_vibe_config(tools=tools)

    missing = project_config_view(config).validation_warnings

    assert len(missing) == 1
    assert missing_key in missing[0]
    assert "tools.web_search.api_key_env_var" in missing[0]
    assert "mock" not in missing[0]
    monkeypatch.setenv(missing_key, "restored-secret")
    assert project_config_view(config).validation_warnings == []


@pytest.mark.parametrize(
    ("disabled_tools", "tools"),
    [
        (["web_search"], {"web_search": {"provider": "auto"}}),
        ([], {"web_search": {"provider": "exa", "permission": "never"}}),
    ],
)
def test_disabled_web_search_does_not_warn_for_missing_credentials(
    monkeypatch: pytest.MonkeyPatch,
    disabled_tools: list[str],
    tools: dict[str, dict[str, str]],
) -> None:
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    config = build_test_vibe_config(disabled_tools=disabled_tools, tools=tools)

    assert project_config_view(config).validation_warnings == []


def test_web_search_missing_credentials_do_not_warn_when_not_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    config = build_test_vibe_config(
        enabled_tools=["bash"], tools={"web_search": {"provider": "auto"}}
    )

    assert project_config_view(config).validation_warnings == []
