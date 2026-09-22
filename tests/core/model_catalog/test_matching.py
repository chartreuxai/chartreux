from __future__ import annotations

from chartreux.core.model_catalog.defaults import SHIPPED_CATALOG
from chartreux.core.model_catalog.matching import match_discovered_model
from chartreux.core.model_catalog.schema import ModelCatalog


def test_existing_deployment_reuses_its_base() -> None:
    outcome = match_discovered_model(SHIPPED_CATALOG, "mistral/default", "zai-glm-5-3")

    assert outcome.kind == "existing"
    assert outcome.existing_base == "glm-5-3"
    assert outcome.matches[0].deployment.name == "zai-glm-5-3"


def test_former_alias_wire_names_become_new_model_proposals() -> None:
    for wire_name in ("zai-glm-5", "zai-glm-latest"):
        outcome = match_discovered_model(SHIPPED_CATALOG, "new/default", wire_name)

        assert outcome.kind == "new_model"
        assert outcome.proposed is not None
        assert outcome.proposed.base_name == wire_name


def test_new_model_uses_conservative_defaults_and_off_when_allowed() -> None:
    outcome = match_discovered_model(
        SHIPPED_CATALOG, "mistral/default", "future-mistral-model"
    )

    assert outcome.kind == "new_model"
    assert outcome.proposed is not None
    assert outcome.proposed.definition.thinking == "off"
    assert outcome.proposed.deployment.prices.input is None
    assert outcome.proposed.deployment.prices.output is None
    assert outcome.proposed.deployment.prices.cached_input is None
    assert not outcome.proposed.deployment.supports_images
    assert outcome.proposed.deployment.supported_thinking_levels is None


def test_exact_known_model_without_off_uses_an_admitted_default() -> None:
    catalog = ModelCatalog.model_validate({
        "providers": {
            "mistral/default": {
                "api_base": "https://api.mistral.ai/v1",
                "backend": "mistral",
            }
        },
        "models": {},
    })

    outcome = match_discovered_model(catalog, "mistral/default", "zai-glm-5")

    assert outcome.kind == "new_model"
    assert outcome.proposed is not None
    assert outcome.proposed.definition.thinking == "medium"


def test_occupied_provider_slot_is_not_appendable() -> None:
    catalog = ModelCatalog.model_validate({
        "providers": {"test/default": {"api_base": "https://test.example"}},
        "models": {
            "known": {"deployments": [{"provider": "test/default", "name": "old-wire"}]}
        },
    })

    outcome = match_discovered_model(catalog, "test/default", "known")

    assert outcome.kind == "occupied_slot"
    assert outcome.provider_slot_occupied
    assert outcome.proposed_base_exists


def test_base_on_another_provider_offers_deployment_addition() -> None:
    outcome = match_discovered_model(SHIPPED_CATALOG, "codex/local", "glm-5-3")

    assert outcome.kind == "base_exists_other_provider"
    assert outcome.existing_base == "glm-5-3"
    assert outcome.proposed_base_exists
    assert not outcome.provider_slot_occupied


def test_multiple_deployments_with_same_wire_are_explicitly_ambiguous() -> None:
    catalog = ModelCatalog.model_validate({
        "providers": {"test/default": {"api_base": "https://test.example"}},
        "models": {
            "first": {
                "deployments": [{"provider": "test/default", "name": "same-wire"}]
            },
            "second": {
                "deployments": [{"provider": "test/default", "name": "same-wire"}]
            },
        },
    })

    outcome = match_discovered_model(catalog, "test/default", "same-wire")

    assert outcome.kind == "multiple_matches"
    assert {match.base_name for match in outcome.matches} == {"first", "second"}


def test_zero_match_creates_a_proposal() -> None:
    outcome = match_discovered_model(SHIPPED_CATALOG, "codex/local", "brand-new")

    assert outcome.kind == "new_model"
    assert outcome.proposed is not None
    assert outcome.proposed.base_name == "brand-new"
