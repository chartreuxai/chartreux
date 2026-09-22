from __future__ import annotations

import json

from chartreux.ui.providers.contracts import (
    DiscoveryError,
    DiscoveryItem,
    DiscoveryResult,
    ModelEdits,
    ModelSelectionDraft,
    OptionalEdit,
    ProviderDraft,
    ProviderManagementDraft,
    active_model_expression_is_valid,
    serialize_draft,
)


def test_draft_preserves_untouched_and_cleared_optional_edits_when_serialized() -> None:
    draft = ProviderManagementDraft(
        provider=ProviderDraft(
            preset="generic-openai",
            provider_id="example/default",
            name="Example",
            api_base="https://example.test/v1",
            api_style="openai",
            api_key_env_var="EXAMPLE_API_KEY",
            key="test-key",
        ),
        selections=(
            ModelSelectionDraft(
                "example-model",
                "example-model",
                ModelEdits(
                    input_price=OptionalEdit(),
                    role_memberships=OptionalEdit.set(("preferred",)),
                ),
            ),
        ),
    )

    serialized = serialize_draft(draft)
    selections = serialized["selections"]
    assert isinstance(selections, list)
    selection = selections[0]
    assert isinstance(selection, dict)
    edits = selection["edits"]
    assert isinstance(edits, dict)

    assert edits["input_price"] == {"state": "untouched", "value": None}
    assert edits["role_memberships"] == {"state": "set", "value": ("preferred",)}
    provider = serialized["provider"]
    assert isinstance(provider, dict)
    assert "key" not in provider
    assert (
        json.loads(json.dumps(serialized))["provider"]["provider_id"]
        == "example/default"
    )


def test_discovery_contract_types_construct_cleanly() -> None:
    result = DiscoveryResult((DiscoveryItem("wire", "Wire"),), ("one page",))
    error = DiscoveryError("auth_rejected", "Credentials were rejected")

    assert result.models[0].display_label == "Wire"
    assert error.code == "auth_rejected"


def test_active_model_contract_accepts_only_v01_expression_shape() -> None:
    assert active_model_expression_is_valid("canonical")
    assert active_model_expression_is_valid("canonical")
    assert active_model_expression_is_valid("@preferred")
    assert not active_model_expression_is_valid("provider/default/wire")
    assert not active_model_expression_is_valid("@")
