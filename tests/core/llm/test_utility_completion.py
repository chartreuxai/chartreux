from __future__ import annotations

from chartreux.core.llm.utility_completion import (
    is_fast_utility_model,
    select_utility_model,
)
from tests.conftest import build_test_vibe_config


def test_utility_completion_uses_shipped_active_model() -> None:
    model, provider = select_utility_model(build_test_vibe_config())
    assert model.alias == "glm-5-2"
    assert model.name == "glm-5-2"
    assert provider.name == "mistral/default"


def test_shipped_default_is_not_fast_utility_model() -> None:
    assert not is_fast_utility_model(build_test_vibe_config())
