"""Workbench draft tests: effective-state comparison and atomic catalog changes."""

from __future__ import annotations

from dataclasses import replace
from typing import cast

import pytest

from chartreux.core.model_catalog.contracts import (
    CatalogChanges,
    DiscoveryItem,
    DiscoveryResult,
    ModelEdits,
    OptionalEdit,
)
from chartreux.core.model_catalog.loader import CatalogSnapshot, merge_catalog_overlay
from chartreux.core.model_catalog.schema import ModelCatalog, ProviderDefinition
from chartreux.ui.providers.management_state import (
    ManagementState,
    PendingModel,
    credential_status,
)


def state() -> ManagementState:
    catalog = ModelCatalog.model_validate({
        "providers": {
            "one": {"api_base": "https://one.test", "api_key_env_var": "ONE"},
            "two": {"api_base": "https://two.test"},
        },
        "models": {
            "a": {
                "temperature": 0.5,
                "deployments": [
                    {
                        "provider": "one",
                        "name": "wire-a",
                        "prices": {"input": 1.0, "output": 2.0},
                        "supports_images": True,
                        "auto_compact_threshold": 0.75,
                    },
                    {"provider": "two", "name": "other-a"},
                ],
            },
            "b": {"deployments": [{"provider": "one", "name": "wire-b"}]},
            "c": {"deployments": [{"provider": "two", "name": "wire-c"}]},
        },
        "roles": {
            "orchestrator": {"model": "a", "thinking": "medium"},
            "single": {"model": "b", "thinking": "high"},
            "cross": {"model": "c", "thinking": "low"},
        },
    })
    return ManagementState.from_snapshot(CatalogSnapshot(catalog, "effective"), "one")


def errors(draft: ManagementState, expression: str | None = "c") -> str:
    return " ".join(draft.validate(expression).errors)


def deployment_patch(batch: CatalogChanges, name: str) -> dict[str, object]:
    return cast("list[dict[str, object]]", batch.models[name]["deployments"])[0]


def test_effective_snapshot_and_connection_round_trip() -> None:
    draft = state()
    assert not draft.dirty
    draft.toggle("a", False)
    draft.toggle("a", True)
    draft.set_edits("a", ModelEdits(input_price=OptionalEdit.set(1.0)))
    assert not draft.dirty
    draft.connection = replace(draft.connection, api_base="https://changed.test")
    assert draft.dirty
    assert draft.changes("c").provider == {"api_base": "https://changed.test"}


def test_two_new_providers_share_one_pending_catalog_batch() -> None:
    draft = state()
    draft.stage_provider("alpha", ProviderDefinition(api_base="https://alpha.test/v1"))
    draft.select("alpha-model")
    draft.stage_provider("beta", ProviderDefinition(api_base="https://beta.test/v1"))
    draft.select("beta-model")

    assert draft.dirty
    assert {"alpha", "beta"} <= draft.catalog.providers.keys()
    batch = draft.changes("c")
    assert batch.provider_patches["alpha"]["api_base"] == "https://alpha.test/v1"
    assert batch.provider_patches["beta"]["api_base"] == "https://beta.test/v1"
    assert batch.models["alpha-model"]["deployments"] == [
        {"provider": "alpha", "name": "alpha-model"}
    ]
    assert batch.models["beta-model"]["deployments"] == [
        {"provider": "beta", "name": "beta-model"}
    ]


def test_disable_and_reenable_preserves_metadata() -> None:
    draft = state()
    draft.toggle("a", False)
    change = draft.changes("c")
    assert change.models["a"]["deployments"] == [
        {"provider": "one", "name": "wire-a", "disabled": True}
    ]

    def effective_patch(catalog: ModelCatalog, patch: object) -> ModelCatalog:
        deployments = [dep.model_dump() for dep in catalog.models["a"].deployments]
        assert isinstance(patch, list)
        deployments[0].update(patch[0])
        return merge_catalog_overlay(
            catalog, {"models": {"a": {"deployments": deployments}}}
        )

    disabled = effective_patch(draft.catalog, change.models["a"]["deployments"])
    resumed = ManagementState.from_snapshot(
        CatalogSnapshot(disabled, "disabled"), "one"
    )
    resumed.toggle("a", True)
    reenable = resumed.changes("c")
    assert deployment_patch(reenable, "a")["disabled"] is False
    restored = effective_patch(disabled, reenable.models["a"]["deployments"])
    assert restored.models["a"].deployments == draft.catalog.models["a"].deployments


def test_prices_and_preset_patch_keep_other_provider() -> None:
    draft = state()
    draft.set_role_preset("cross", "a", "high")
    draft.set_edits(
        "a",
        ModelEdits(
            input_price=OptionalEdit.cleared(), output_price=OptionalEdit.set(0.0)
        ),
    )
    batch = draft.changes("c")
    assert batch.roles == {"cross": {"model": "a", "thinking": "high"}}
    assert deployment_patch(batch, "a")["prices"] == {"output": 0.0}


def test_metadata_edits_patch_canonical_and_deployment_fields_and_clear_nullable() -> (
    None
):
    draft = state()
    draft.set_edits(
        "a",
        ModelEdits(
            thinking=OptionalEdit.set("high"),
            temperature=OptionalEdit.cleared(),
            supports_images=OptionalEdit.set(False),
            auto_compact_threshold=OptionalEdit.cleared(),
        ),
    )
    # A second provider view's untouched canonical fields cannot erase the edit.
    draft.for_provider("two")
    draft.set_edits("a", ModelEdits(supports_images=OptionalEdit.set(True)))
    batch = draft.changes("c")
    assert batch.models["a"]["thinking"] == "high"
    assert batch.models["a"]["temperature"] is None
    assert deployment_patch(batch, "a")["supports_images"] is False
    assert deployment_patch(batch, "a")["auto_compact_threshold"] is None


def test_canonical_edits_follow_latest_provider_edit_chronology() -> None:
    draft = state()
    draft.set_edits("a", ModelEdits(thinking=OptionalEdit.set("high")))
    draft.for_provider("two")
    assert draft.edits.get("a", ModelEdits()).thinking == OptionalEdit.set("high")
    draft.set_edits("a", ModelEdits(thinking=OptionalEdit.set("low")))
    draft.for_provider("one")
    draft.set_edits("a", ModelEdits(thinking=OptionalEdit.set("high")))

    assert draft.edits["a"].thinking == OptionalEdit.set("high")
    draft.for_provider("two")
    assert draft.edits["a"].thinking == OptionalEdit.set("high")
    assert draft.changes("c").models["a"]["thinking"] == "high"


@pytest.mark.parametrize("pending_is_latest", [False, True])
def test_pending_and_configured_canonical_edits_follow_chronology(
    pending_is_latest: bool,
) -> None:
    draft = state()
    draft.for_provider("two")
    draft.select("novel")
    draft.pending["novel"] = replace(
        draft.pending["novel"], canonical_name="b", decision="add_existing"
    )
    pending_edits = ModelEdits(temperature=OptionalEdit.set(0.9))
    configured_edits = ModelEdits(temperature=OptionalEdit.set(0.2))
    if pending_is_latest:
        draft.for_provider("one")
        draft.set_edits("b", configured_edits)
        draft.for_provider("two")
        draft.set_pending_edits("novel", pending_edits)
        expected = 0.9
    else:
        draft.set_pending_edits("novel", pending_edits)
        draft.for_provider("one")
        draft.set_edits("b", configured_edits)
        expected = 0.2

    draft.for_provider("two")
    assert draft.pending["novel"].edits.temperature == OptionalEdit.set(expected)
    draft.for_provider("one")
    assert draft.edits["b"].temperature == OptionalEdit.set(expected)
    assert draft.changes("c").models["b"]["temperature"] == expected


def test_disabled_new_pending_model_drops_its_canonical_edit() -> None:
    draft = state()
    draft.select("novel")
    draft.set_pending_edits("novel", ModelEdits(thinking=OptionalEdit.set("high")))
    draft.pending["novel"] = replace(draft.pending["novel"], enabled=False)

    assert "novel" not in draft.changes("c").models


@pytest.mark.parametrize("field", ["thinking", "supports_images"])
def test_nonnullable_metadata_cannot_be_cleared(field: str) -> None:
    draft = state()
    draft.set_edits("a", ModelEdits(**{field: OptionalEdit.cleared()}))
    assert f"{field} cannot be cleared" in errors(draft)


def test_preset_edit_requires_known_role_and_valid_pair() -> None:
    draft = state()
    with pytest.raises(ValueError, match="Unknown preset"):
        draft.set_role_preset("unknown", "a", "medium")
    with pytest.raises(ValueError, match="canonical model"):
        draft.set_role_preset("single", "@orchestrator", "high")
    with pytest.raises(ValueError, match="Unknown thinking level"):
        draft.set_role_preset("single", "b", "absurd")


def test_provider_save_allows_incomplete_preset_and_finish_guides_repair() -> None:
    draft = state()
    draft.set_role_preset("single", "pending", "high")
    assert draft.validate(completion=False).errors == ()
    assert draft.changes().roles == {"single": {"model": "pending", "thinking": "high"}}
    finish = draft.validate(completion=True)
    assert finish.unresolved_roles == ("single",)
    assert any(
        "Preset single" in error and "pending" in error for error in finish.errors
    )


def test_finish_checks_preset_credential_and_selected_thinking() -> None:
    draft = state()
    missing = draft.validate(credential_resolver=lambda _env: None)
    assert "single" in missing.unusable_roles
    assert any("credential ONE" in error for error in missing.errors)
    draft.set_role_preset("single", "c", "high")
    assert (
        "single"
        not in draft.validate(credential_resolver=lambda _env: None).unusable_roles
    )
    draft.set_role_preset("single", "c", "off")
    assert draft.validate(completion=False).errors == ()


def test_collisions_and_duplicate_pending() -> None:
    draft = state()
    draft.select("c")  # base exists on other provider
    assert "Resolve the collision" in errors(draft)
    draft.pending["c"] = replace(draft.pending["c"], decision="add_existing")
    assert not errors(draft)
    draft.pending["c"] = replace(
        draft.pending["c"], decision="separate", canonical_name="separate"
    )
    assert not errors(draft)
    draft.select("wire-b", canonical_name="duplicate")  # exact existing toggles instead
    assert "wire-b" not in draft.pending
    draft.select("new-1")
    draft.select("new-2", canonical_name="new-1")
    assert "Duplicate pending" in errors(draft)
    draft.pending.clear()
    draft.select("wire-x", canonical_name="a")  # occupied provider slot
    assert "occupied" in errors(draft)
    draft.pending["wire-x"] = replace(draft.pending["wire-x"], canonical_name="new-x")
    assert not errors(draft)


def test_multiple_exact_matches_rejected() -> None:
    draft = state()
    catalog = draft.catalog.model_dump()
    catalog["models"]["b"]["deployments"][0]["name"] = "wire-a"
    draft.snapshot = CatalogSnapshot(ModelCatalog.model_validate(catalog), "duplicate")
    draft.pending["wire-a"] = PendingModel("wire-a", "new", True)
    assert "already matches" in errors(draft)


def test_single_batch_pending_new_and_preset_edit() -> None:
    draft = state()
    draft.connection = replace(draft.connection, api_base="https://new.test")
    draft.select("novel")
    draft.set_role_preset("cross", "novel", "high")
    batch = draft.changes("c")
    assert batch.provider["api_base"] == "https://new.test"
    assert deployment_patch(batch, "novel")["name"] == "novel"
    assert batch.roles == {"cross": {"model": "novel", "thinking": "high"}}


def test_catalog_draft_persists_across_provider_views() -> None:
    draft = state()
    draft.connection = replace(draft.connection, api_base="https://edited.test")
    draft.toggle("a", False)
    draft.begin_discovery()
    draft.for_provider("two")
    draft.toggle("c", False)
    assert draft.discovery_generation == 0
    draft.begin_discovery()
    draft.set_edits("a", ModelEdits(input_price=OptionalEdit.set(3.0)))
    draft.for_provider("one")
    assert draft.connection.api_base == "https://edited.test"
    assert draft.enabled["a"] is False
    assert draft.discovery_generation == 1
    batch = draft.changes("b")
    assert batch.provider_patches == {"one": {"api_base": "https://edited.test"}}
    assert {
        dep["provider"]
        for dep in cast("list[dict[str, object]]", batch.models["a"]["deployments"])
    } == {"one", "two"}
    assert deployment_patch(batch, "c")["provider"] == "two"


def test_preset_edits_persist_across_provider_views() -> None:
    draft = state()
    draft.set_role_preset("single", "c", "medium")
    draft.for_provider("two")
    assert draft.preset("single") == ("c", "medium")
    assert draft.changes("c").roles == {"single": {"model": "c", "thinking": "medium"}}
    assert draft.dirty


def test_readiness_is_deferred_for_structural_save() -> None:
    draft = state()
    missing = lambda _env: None
    management = draft.validate("c", credential_resolver=missing)
    assert not management.unresolved_roles
    assert "single" in management.unusable_roles
    assert any("single" in error for error in management.errors)
    assert (
        draft.validate("c", credential_resolver=missing, completion=False).errors == ()
    )
    assert draft.changes("c", credential_resolver=missing).roles is None


def test_shipped_onboarding_requires_resolvable_credential() -> None:
    from chartreux.core.model_catalog.defaults import SHIPPED_CATALOG

    draft = ManagementState.from_snapshot(
        CatalogSnapshot(SHIPPED_CATALOG, "shipped"), "mistral"
    )
    management = draft.validate("glm-5-3", credential_resolver=lambda _: None)
    assert management.unusable_roles == tuple(SHIPPED_CATALOG.roles)
    assert not management.unresolved_roles
    assert len(management.errors) == len(SHIPPED_CATALOG.roles)
    onboarding = draft.validate(
        "glm-5-3", mode="onboarding", credential_resolver=lambda _: None
    )
    assert len(onboarding.errors) == len(SHIPPED_CATALOG.roles)
    assert not draft.validate(
        "glm-5-3", mode="onboarding", credential_resolver=lambda _: "key"
    ).errors


def test_preset_edit_survives_other_provider_price_edit() -> None:
    draft = state()
    draft.set_role_preset("cross", "a", "high")
    draft.for_provider("two")
    draft.set_edits("a", ModelEdits(input_price=OptionalEdit.set(3.0)))
    assert draft.changes("c").roles == {"cross": {"model": "a", "thinking": "high"}}


def test_discovery_generation_and_union() -> None:
    draft = state()
    provider_id, first = draft.begin_discovery()
    provider_id, second = draft.begin_discovery(provider_id)
    result = DiscoveryResult((DiscoveryItem("new"),))
    assert not draft.accept_discovery(provider_id, first, result)
    assert draft.accept_discovery(provider_id, second, result)
    assert ("a", "wire-a", True, False) in draft.model_rows()
    assert ("new", "new", False, True) in draft.model_rows()
    draft.select("new")
    assert ("new", "new", True, True) in draft.model_rows()
    assert draft.begin_discovery() == ("one", second + 1)
    assert draft.discovery is None


def test_discovery_result_stays_with_captured_provider_after_switch() -> None:
    draft = state()
    provider_id, generation = draft.begin_discovery()
    draft.for_provider("two")
    result = DiscoveryResult((DiscoveryItem("one-only"),))
    assert draft.accept_discovery(provider_id, generation, result)
    assert draft.discovery is None
    draft.for_provider("one")
    assert draft.discovery == result


def test_credential_status_labels() -> None:
    assert credential_status("ONE", lambda _: "secret") == "Key Set"
    assert credential_status("", lambda _: "secret") == "No Authentication"
    assert credential_status("ONE", lambda _: None) == "Key Required"


def test_finish_reports_unsupported_thinking_for_one_preset() -> None:
    draft = state()
    catalog_data = draft.catalog.model_dump()
    catalog_data["models"]["c"]["deployments"][0]["supported_thinking_levels"] = ["low"]
    draft.snapshot = CatalogSnapshot(
        ModelCatalog.model_validate(catalog_data), "limited"
    )
    draft.set_role_preset("cross", "c", "high")
    assert draft.validate(completion=False).errors == ()
    result = draft.validate(completion=True, credential_resolver=lambda _: "key")
    assert result.unusable_roles == ("cross",)
    assert any("thinking high is unsupported" in error for error in result.errors)
