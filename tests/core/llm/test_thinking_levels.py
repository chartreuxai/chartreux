from __future__ import annotations

import logging

import pytest

from chartreux.config_values import THINKING_LEVELS
from chartreux.core.config import ModelConfig, ProviderConfig
from chartreux.core.llm.backend.base import apply_reasoning_effort
from chartreux.core.llm.backend.generic import _declared_request_thinking
from chartreux.core.llm.thinking_levels import (
    ANTHROPIC_THINKING_LEVELS,
    GLM_5_3_THINKING_LEVELS,
    MISTRAL_LARGE_4_THINKING_LEVELS,
    MISTRAL_THINKING_LEVELS,
    THINKING_LEVEL_ORDER,
    declared_thinking_levels,
    get_thinking_levels,
    get_thinking_wire_value,
    narrow_thinking_levels,
    note_thinking_fallback,
    reset_thinking_fallback_notices,
    resolve_enabled_thinking_level,
    resolve_thinking_level,
)
from chartreux.core.llm_models import LLMMessage, Role
from chartreux.core.subagents import InvalidLaunchThinkingError


def test_public_vocabulary_is_one_ordered_tuple() -> None:
    assert THINKING_LEVEL_ORDER == ("off", "low", "medium", "high", "xhigh", "max")
    assert THINKING_LEVELS == THINKING_LEVEL_ORDER
    assert "xhigh" in THINKING_LEVEL_ORDER


def test_declared_level_sets_match_the_documented_model_capabilities() -> None:
    assert declared_thinking_levels(MISTRAL_LARGE_4_THINKING_LEVELS) == (
        "low",
        "medium",
        "high",
    )
    assert declared_thinking_levels(GLM_5_3_THINKING_LEVELS) == ("low", "high", "max")
    assert (
        get_thinking_levels("mistral", None, "mistral-large-4")
        is MISTRAL_LARGE_4_THINKING_LEVELS
    )
    for model_name in (
        "zai-glm-5-3",
        "zai-glm-5-3-flash",
        "zai-glm-5",
        "zai-glm-latest",
    ):
        assert (
            get_thinking_levels("mistral", None, model_name) is GLM_5_3_THINKING_LEVELS
        )


def test_narrow_thinking_levels_preserves_wire_values() -> None:
    levels = MISTRAL_LARGE_4_THINKING_LEVELS
    assert narrow_thinking_levels(levels, None) is levels
    assert narrow_thinking_levels(levels, ["low", "high"]) == {
        "low": "low",
        "high": "high",
    }
    assert narrow_thinking_levels({}, ["low"]) == {}
    with pytest.raises(InvalidLaunchThinkingError):
        resolve_thinking_level(narrow_thinking_levels({}, ["low"]), "low")


def test_resolve_enabled_thinking_level_excludes_off() -> None:
    narrowed = narrow_thinking_levels(ANTHROPIC_THINKING_LEVELS, ["off", "max"])
    forced = resolve_enabled_thinking_level(narrowed)
    assert forced.effective == "max"
    assert forced.wire_value == "max"
    assert (
        resolve_enabled_thinking_level(ANTHROPIC_THINKING_LEVELS).effective == "medium"
    )
    with pytest.raises(InvalidLaunchThinkingError):
        resolve_enabled_thinking_level({"off": None})


def test_declared_off_only_rejects_retained_anthropic_reasoning() -> None:
    provider = ProviderConfig(
        name="anthropic",
        api_base="https://api.anthropic.com",
        api_key_env_var="",
        api_style="anthropic",
    )
    model = ModelConfig(
        name="claude-sonnet-4-20250514",
        provider="anthropic",
        alias="claude",
        thinking="off",
        supported_thinking_levels=["off"],
    )
    levels = get_thinking_levels("generic", "anthropic", model.name)
    assert levels is not None
    narrowed = narrow_thinking_levels(levels, model.supported_thinking_levels)
    history = [
        LLMMessage(role=Role.user, content="Hello"),
        LLMMessage(
            role=Role.assistant,
            content="Answer",
            reasoning_payloads=[{"type": "thinking", "thinking": "considering"}],
        ),
    ]

    assert _declared_request_thinking(model, provider, history[:1], narrowed) == "off"
    with pytest.raises(InvalidLaunchThinkingError):
        _declared_request_thinking(model, provider, history, narrowed)


def test_unknown_mistral_model_uses_provider_table() -> None:
    assert (
        get_thinking_levels("mistral", None, "zai-glm-future")
        is MISTRAL_THINKING_LEVELS
    )


@pytest.mark.parametrize(
    ("requested", "effective", "wire"),
    [
        ("low", "low", "low"),
        ("medium", "medium", "medium"),
        ("high", "high", "high"),
        ("off", "low", "low"),
        ("xhigh", "high", "high"),
        ("max", "high", "high"),
    ],
)
def test_mistral_large_4_resolves_within_its_declared_levels(
    requested: str, effective: str, wire: str
) -> None:
    resolution = resolve_thinking_level(MISTRAL_LARGE_4_THINKING_LEVELS, requested)

    assert (resolution.requested, resolution.effective) == (requested, effective)
    assert resolution.wire_value == wire
    assert resolution.fell_back is (requested != effective)


@pytest.mark.parametrize(
    ("requested", "effective", "wire"),
    [
        ("low", "low", "low"),
        ("high", "high", "high"),
        ("max", "max", "max"),
        ("off", "low", "low"),
        ("medium", "high", "high"),
        ("xhigh", "max", "max"),
    ],
)
def test_glm_5_3_resolves_within_its_declared_levels(
    requested: str, effective: str, wire: str
) -> None:
    resolution = resolve_thinking_level(GLM_5_3_THINKING_LEVELS, requested)

    assert (resolution.requested, resolution.effective) == (requested, effective)
    assert resolution.wire_value == wire


def test_equal_distance_fallback_breaks_upward_toward_more_thinking() -> None:
    levels = {"low": "low", "high": "high"}

    assert resolve_thinking_level(levels, "medium").effective == "high"


def test_unknown_level_names_stay_typed_errors() -> None:
    with pytest.raises(InvalidLaunchThinkingError, match="Unknown thinking level"):
        resolve_thinking_level(MISTRAL_LARGE_4_THINKING_LEVELS, "absurd")
    with pytest.raises(InvalidLaunchThinkingError, match="declares no thinking levels"):
        resolve_thinking_level({}, "max")


def test_encoder_reads_the_live_table(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(MISTRAL_THINKING_LEVELS, "low", "minimal")
    payload: dict[str, str] = {}

    apply_reasoning_effort(payload, "low", MISTRAL_THINKING_LEVELS)

    assert payload["reasoning_effort"] == "minimal"


def test_encoder_falls_back_instead_of_raising_for_an_undeclared_level() -> None:
    assert get_thinking_wire_value(GLM_5_3_THINKING_LEVELS, "off") == "low"
    assert get_thinking_wire_value(MISTRAL_LARGE_4_THINKING_LEVELS, "off") == "low"
    assert get_thinking_wire_value(MISTRAL_LARGE_4_THINKING_LEVELS, "xhigh") == "high"


def test_one_notice_per_new_deployment_requested_effective(
    caplog: pytest.LogCaptureFixture,
) -> None:
    reset_thinking_fallback_notices()
    fallback = resolve_thinking_level(GLM_5_3_THINKING_LEVELS, "medium")
    exact = resolve_thinking_level(GLM_5_3_THINKING_LEVELS, "high")

    with caplog.at_level(logging.WARNING, logger="vibe"):
        assert note_thinking_fallback("mistral/zai-glm-5-3", fallback) is True
        assert note_thinking_fallback("mistral/zai-glm-5-3", fallback) is False
        assert note_thinking_fallback("mistral/zai-glm-5-3", exact) is False
        assert note_thinking_fallback("mistral/other", fallback) is True

    assert len(caplog.records) == 2
    assert all(
        "nearest level 'high'" in record.getMessage() for record in caplog.records
    )
    reset_thinking_fallback_notices()
