from __future__ import annotations

from pydantic import ValidationError
import pytest

from chartreux.core.config import MissingAPIKeyError
from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
from chartreux.core.prompts import UtilityPrompt
from tests.conftest import build_test_vibe_config


def test_unpinned_configuration_uses_shipped_default() -> None:
    config = build_test_vibe_config()
    assert config.active_model == ""
    assert config.get_active_model().alias == "glm-5-3"


def test_default_compaction_prompt_is_compact() -> None:
    config = build_test_vibe_config()
    assert config.compaction_prompt_id == "compact"
    assert config.compaction_prompt == UtilityPrompt.COMPACT.read()


def test_catalog_backed_unknown_active_model_fails_at_resolution() -> None:
    config = build_test_vibe_config(active_model="private")
    with pytest.raises(ValueError, match="Unknown model"):
        config.get_active_model()


def test_catalog_backed_compaction_model_is_resolved() -> None:
    config = build_test_vibe_config(
        compaction_model="glm-5-3", thinking_overrides={"glm-5-3": "low"}
    )
    assert config.get_compaction_model().thinking == "low"


def test_schema_still_validates_scalar_config_fields() -> None:
    assert ChartreuxConfigSchema(log_level="debug").log_level == "DEBUG"
    with pytest.raises(ValidationError):
        ChartreuxConfigSchema(log_level="verbose")


def test_api_key_readiness_is_separate_from_schema_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    with pytest.raises(MissingAPIKeyError):
        build_test_vibe_config().require_active_provider_api_key()
