"""v0.8.0 A-P2: the [safety] section, model/prompt resolution, and the
prompt pin for the bundled SMART classifier prompt.
"""

from __future__ import annotations

import hashlib

from pydantic import ValidationError
import pytest

from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
from chartreux.core.config.models import (
    DEFAULT_SMART_APPROVE_CONTEXT_CHARS,
    DEFAULT_SMART_APPROVE_MAX_CALLS,
    DEFAULT_SMART_APPROVE_MAX_SPEND_USD,
    DEFAULT_SMART_APPROVE_PROMPT_ID,
    SafetySectionConfig,
)
from chartreux.core.model_catalog.resolver import ModelResolutionError
from chartreux.core.prompts import UtilityPrompt
from chartreux.core.usage import UsagePurpose
from tests.conftest import build_test_vibe_config

#: The amendment Q5 FINAL prompt digest (panel-computed, never self-pinned).
FINAL_PROMPT_SHA256 = "6d83f49fa8f840dcc023cc5d5dabb74354fd6da5042c297fcbcaecc2c33ec930"
FINAL_PROMPT_BYTES = 2153


class TestSafetySectionDefaults:
    def test_defaults(self) -> None:
        section = SafetySectionConfig()
        assert section.default_level == "safe"
        assert section.smart_approve_model == ""
        assert section.smart_approve_prompt_id == DEFAULT_SMART_APPROVE_PROMPT_ID
        assert section.smart_approve_max_calls == DEFAULT_SMART_APPROVE_MAX_CALLS == 200
        assert (
            section.smart_approve_max_spend_usd
            == (DEFAULT_SMART_APPROVE_MAX_SPEND_USD)
            == 0.25
        )
        assert (
            section.smart_approve_context_chars
            == (DEFAULT_SMART_APPROVE_CONTEXT_CHARS)
            == 3000
        )

    def test_registered_on_the_schema_with_deep_merge(self) -> None:
        config = build_test_vibe_config()
        assert isinstance(config.safety, SafetySectionConfig)
        merged = build_test_vibe_config(
            safety={"smart_approve_max_calls": 5, "default_level": "smart"}
        )
        assert merged.safety.smart_approve_max_calls == 5
        assert merged.safety.default_level == "smart"
        # Unspecified keys keep their defaults under deep merge.
        assert merged.safety.smart_approve_context_chars == 3000

    def test_extra_keys_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            SafetySectionConfig.model_validate({"unknown_key": 1})

    def test_default_level_accepts_safe_and_smart(self) -> None:
        assert (
            SafetySectionConfig.model_validate({"default_level": "safe"}).default_level
            == "safe"
        )
        assert (
            SafetySectionConfig.model_validate({"default_level": "smart"}).default_level
            == "smart"
        )

    def test_default_level_rejects_dangerous_with_pointed_message(self) -> None:
        with pytest.raises(ValidationError, match="ceremony-only"):
            SafetySectionConfig.model_validate({"default_level": "dangerous"})
        with pytest.raises(ValidationError):
            SafetySectionConfig.model_validate({"default_level": "everything"})

    def test_cap_bounds(self) -> None:
        with pytest.raises(ValidationError):
            SafetySectionConfig.model_validate({"smart_approve_max_calls": 0})
        with pytest.raises(ValidationError):
            SafetySectionConfig.model_validate({"smart_approve_max_calls": 1.5})
        with pytest.raises(ValidationError):
            SafetySectionConfig.model_validate({"smart_approve_max_spend_usd": -0.01})
        with pytest.raises(ValidationError):
            SafetySectionConfig.model_validate({"smart_approve_max_spend_usd": True})
        with pytest.raises(ValidationError):
            SafetySectionConfig.model_validate({"smart_approve_context_chars": -1})
        with pytest.raises(ValidationError):
            SafetySectionConfig.model_validate({"smart_approve_context_chars": "3000"})
        # 0 is legal for the context budget (packet-only, option A).
        assert (
            SafetySectionConfig.model_validate({
                "smart_approve_context_chars": 0
            }).smart_approve_context_chars
            == 0
        )


class TestSmartApproveModelResolution:
    def test_empty_key_falls_back_to_active_model_with_thinking_off(self) -> None:
        config = build_test_vibe_config()
        active = config.get_active_model()
        resolved = config.get_smart_approve_model()
        assert resolved.name == active.name
        # The classifier is pinned to thinking OFF regardless of overrides.
        assert resolved.thinking == "off"

    def test_thinking_overrides_do_not_apply(self) -> None:
        config = build_test_vibe_config(
            thinking_overrides={"mistral-large-4": "high"},
            safety={"smart_approve_model": "mistral-large-4"},
        )
        assert config.get_active_model().thinking == "high"
        assert config.get_smart_approve_model().thinking == "off"

    def test_explicit_model_resolves_at_use_time(self) -> None:
        config = build_test_vibe_config(safety={"smart_approve_model": "glm-5-3"})
        resolved = config.get_smart_approve_model()
        assert resolved.alias == "glm-5-3"
        assert resolved.thinking == "off"

    def test_unknown_model_raises_model_resolution_error(self) -> None:
        config = build_test_vibe_config(
            safety={"smart_approve_model": "mistral-small-4"}
        )
        with pytest.raises(ModelResolutionError):
            config.get_smart_approve_model()


class TestSmartApprovePrompt:
    def test_prompt_property_loads_the_bundled_prompt(self) -> None:
        config = build_test_vibe_config()
        prompt = config.smart_approve_prompt
        assert "SMART-mode approval classifier" in prompt
        assert prompt.strip() == (
            UtilityPrompt.SMART_APPROVE_CLASSIFIER.path.read_text(
                encoding="utf-8"
            ).strip()
        )

    def test_unknown_prompt_id_raises(self) -> None:
        from chartreux.core.policy.resources import ResourceResolutionError
        from chartreux.core.prompts import MissingPromptFileError

        config = build_test_vibe_config(
            safety={"smart_approve_prompt_id": "no-such-prompt"}
        )
        with pytest.raises((MissingPromptFileError, ResourceResolutionError)):
            _ = config.smart_approve_prompt


class TestPromptPin:
    """The bundled prompt is the amendment Q5 FINAL text VERBATIM: the digest
    below was computed by the design panel from the design text, never by the
    executor; any paraphrase or byte drift fails.
    """

    def test_bundled_prompt_hashes_to_the_final_digest(self) -> None:
        raw = UtilityPrompt.SMART_APPROVE_CLASSIFIER.path.read_bytes()
        assert len(raw) == FINAL_PROMPT_BYTES
        assert hashlib.sha256(raw).hexdigest() == FINAL_PROMPT_SHA256

    def test_prompt_spot_assertions(self) -> None:
        text = UtilityPrompt.SMART_APPROVE_CLASSIFIER.path.read_text(encoding="utf-8")
        assert "Shell-grammar refusals never reach you" in text
        assert "presumed an accident" in text
        assert "your own JSON" in text
        assert '{"verdict": "uphold", "rationale": "example rationale"}' in text


class TestUsagePurpose:
    def test_smart_approve_purpose_exists(self) -> None:
        assert UsagePurpose.SMART_APPROVE == "smart-approve"


def test_schema_accepts_safety_section_via_model_validate() -> None:
    config = ChartreuxConfigSchema.model_validate({
        "safety": {"default_level": "smart", "smart_approve_max_spend_usd": 0.10}
    })
    assert config.safety.default_level == "smart"
    assert config.safety.smart_approve_max_spend_usd == 0.10
