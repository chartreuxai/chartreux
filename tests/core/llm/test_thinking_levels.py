from __future__ import annotations

import pytest

from chartreux.core.llm.backend.base import apply_reasoning_effort
from chartreux.core.llm.thinking_levels import (
    GLM_5_2_THINKING_LEVELS,
    GLM_5_3_THINKING_LEVELS,
    MISTRAL_THINKING_LEVELS,
    get_thinking_levels,
)
from chartreux.core.subagents import InvalidLaunchThinkingError


@pytest.mark.parametrize("model_name", ["glm-5-2", "zai-glm-5-2"])
def test_glm_5_2_exact_aliases_share_mapping(model_name: str) -> None:
    assert get_thinking_levels("mistral", None, model_name) is GLM_5_2_THINKING_LEVELS
    assert GLM_5_2_THINKING_LEVELS == {
        "off": "none",
        "low": "high",
        "medium": "high",
        "high": "high",
        "max": "max",
    }


@pytest.mark.parametrize("model_name", ["zai-glm-5-3", "zai-glm-5", "zai-glm-latest"])
def test_glm_5_3_exact_aliases_cannot_disable_thinking(model_name: str) -> None:
    assert get_thinking_levels("mistral", None, model_name) is GLM_5_3_THINKING_LEVELS
    assert "off" not in GLM_5_3_THINKING_LEVELS


def test_unknown_mistral_model_uses_provider_table() -> None:
    assert (
        get_thinking_levels("mistral", None, "zai-glm-future")
        is MISTRAL_THINKING_LEVELS
    )


def test_encoder_reads_the_live_table(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(MISTRAL_THINKING_LEVELS, "low", "minimal")
    payload: dict[str, str] = {}

    apply_reasoning_effort(payload, "low", MISTRAL_THINKING_LEVELS)

    assert payload["reasoning_effort"] == "minimal"


def test_model_entry_rejection_is_a_typed_error() -> None:
    levels = get_thinking_levels("mistral", None, "zai-glm-5-3")
    assert levels is not None

    with pytest.raises(InvalidLaunchThinkingError, match="cannot disable thinking"):
        apply_reasoning_effort({}, "off", levels)
