from __future__ import annotations

from dataclasses import replace

from pydantic import ValidationError
import pytest

from chartreux.core.agents.models import BUILTIN_SUBAGENTS
from chartreux.core.dispatch.presets import STANDALONE_PRESET
from chartreux.core.dispatch.session import (
    BoundDispatchPolicy,
    SessionPolicyError,
    bind_policy,
    bind_snapshot_policy,
    resume_policy,
)
from chartreux.core.model_catalog.defaults import SHIPPED_CATALOG
from chartreux.core.model_catalog.loader import CatalogSnapshot
from chartreux.core.model_catalog.resolver import ModelResolver
from chartreux.core.model_catalog.schema import ModelCatalog, RoleDefinition
from tests.conftest import build_test_vibe_config


def config_for_policy():
    return build_test_vibe_config(
        catalog_snapshot=CatalogSnapshot(SHIPPED_CATALOG, "original")
    )


def test_snapshot_round_trip_is_resolved_and_immutable():
    bound = bind_policy(config_for_policy())
    assert bound.bindings
    restored = BoundDispatchPolicy.model_validate_json(bound.model_dump_json())
    assert restored == bound
    assert restored.roster == bound.roster
    with pytest.raises(TypeError):
        restored.bindings["new"] = next(iter(bound.bindings.values()))  # type: ignore[index]
    with pytest.raises(ValidationError):
        restored.version = 2  # type: ignore[assignment]


def test_bound_canonical_identities_counts_wire_names_and_thinking_once():
    # One canonical model served through two providers under different wire
    # names, bound by every slot at different thinking levels, is one identity
    # (ADR 0018-G.4: a single-model roster is one canonical model).
    catalog = ModelCatalog.model_validate({
        "providers": {
            "one": {"api_base": "https://one.test"},
            "two": {"api_base": "https://two.test"},
        },
        "models": {
            "a": {
                "deployments": [
                    {"provider": "one", "name": "wire-one"},
                    {"provider": "two", "name": "wire-two"},
                ]
            }
        },
        "roles": {
            "worker": {"model": "a", "thinking": "medium"},
            "scout": {"model": "a", "thinking": "low"},
            "heavy": {"model": "a", "thinking": "high"},
        },
    })
    bound = bind_snapshot_policy(CatalogSnapshot(catalog, "wires"))
    assert len(bound.bindings) == len(STANDALONE_PRESET.slots)
    assert bound.bound_canonical_identities == frozenset({"a"})


def test_bound_canonical_identities_distinguishes_canonical_models():
    config = config_for_policy()
    bound = bind_policy(config)
    # The shipped roster binds glm-5-3 through several slots and
    # mistral-large-4 through the rest: exactly two identities.
    assert bound.bound_canonical_identities == frozenset({"glm-5-3", "mistral-large-4"})


def test_bound_canonical_identities_excludes_unbound_slots():
    snapshot = CatalogSnapshot(SHIPPED_CATALOG, "original")
    models = dict(snapshot.catalog.models)
    models.pop("mistral-large-4")
    snapshot = replace(
        snapshot, catalog=snapshot.catalog.model_copy(update={"models": models})
    )
    bound = bind_snapshot_policy(snapshot)
    # @heavy no longer resolves: those slots stay unbound and out of the
    # identity set, exactly as identity_for skips them.
    assert bound.bound_canonical_identities == frozenset({"glm-5-3"})
    assert bound.identity_for("@heavy") is None
    assert all(identity.base_model == "glm-5-3" for identity in bound.bindings.values())


def test_bind_snapshot_policy_resolves_the_runtime_roster():
    config = config_for_policy()
    bound = bind_policy(config)
    resolved = bind_snapshot_policy(
        config.catalog_snapshot,
        allowed_models=config.allowed_models,
        auto_compact_threshold=config.auto_compact_threshold,
    )
    assert resolved.bindings == bound.bindings
    assert resolved.diagnostics == bound.diagnostics
    assert resolved.bound_canonical_identities == bound.bound_canonical_identities


def test_catalog_edit_does_not_rebind_resumed_slots():
    config = config_for_policy()
    bound = bind_policy(config)
    snapshot = config.catalog_snapshot
    roles = dict(snapshot.catalog.roles)
    roles["worker"] = RoleDefinition(model="glm-5-3", thinking="off")
    config.attach_catalog_snapshot(
        replace(
            snapshot,
            catalog=snapshot.catalog.model_copy(update={"roles": roles}),
            dispatch=STANDALONE_PRESET,
        )
    )
    restored = resume_policy(config, bound.model_dump(mode="json"), BUILTIN_SUBAGENTS)
    assert restored.policy == bound.policy
    assert restored.bindings == bound.bindings
    fresh = bind_policy(config)
    assert fresh.policy == STANDALONE_PRESET
    assert fresh.bindings != bound.bindings


def test_legacy_session_resolves_live_with_visible_note():
    config = config_for_policy()
    bound = resume_policy(config, None, BUILTIN_SUBAGENTS)
    config.attach_dispatch_policy(bound)
    assert any("Legacy session" in note for note in config.validation_warnings)
    from chartreux.app_server._projection import project_config_view

    assert any(
        "Legacy session" in note
        for note in project_config_view(config).validation_warnings
    )


def test_unsupported_snapshot_version_fails_closed():
    saved = bind_policy(config_for_policy()).model_dump(mode="json")
    saved["version"] = 2
    with pytest.raises(SessionPolicyError, match="Unsupported"):
        resume_policy(config_for_policy(), saved, BUILTIN_SUBAGENTS)


@pytest.mark.parametrize("removed", ["role", "profile"])
def test_removed_bound_reference_is_explicit_error(removed):
    config = config_for_policy()
    saved = bind_policy(config).model_dump(mode="json")
    profiles = dict(BUILTIN_SUBAGENTS)
    slot = next(iter(BoundDispatchPolicy.model_validate(saved).policy.slots.values()))
    if removed == "role":
        snapshot = config.catalog_snapshot
        roles = dict(snapshot.catalog.roles)
        roles.pop(slot.role[1:])
        config.attach_catalog_snapshot(
            replace(
                snapshot, catalog=snapshot.catalog.model_copy(update={"roles": roles})
            )
        )
    else:
        profiles.pop(slot.profile)
    with pytest.raises(SessionPolicyError, match="removed"):
        resume_policy(config, saved, profiles)


def test_pre_upgrade_policy_with_removed_roles_fails_resume_without_substitution():
    # A policy saved before the roster rename binds the removed tier roles:
    # resume fails explicitly (ADR 0018-H forbids substitution), while the
    # committed model identities are role-free and still resolve.
    config = config_for_policy()
    legacy_roles = (
        "@small",
        "@medium",
        "@medium",
        "@large",
        "@medium",
        "@large",
        "@large",
        "@medium",
    )
    slots = {
        name: slot.model_copy(update={"role": role})
        for (name, slot), role in zip(
            STANDALONE_PRESET.slots.items(), legacy_roles, strict=True
        )
    }
    saved = bind_policy(config).model_dump(mode="json")
    saved["policy"] = STANDALONE_PRESET.model_copy(update={"slots": slots}).model_dump(
        mode="json"
    )
    with pytest.raises(SessionPolicyError, match="Bound dispatch role removed"):
        resume_policy(config, saved, BUILTIN_SUBAGENTS)
    resolver = ModelResolver(CatalogSnapshot(SHIPPED_CATALOG, "committed"))
    for identity in BoundDispatchPolicy.model_validate(saved).bindings.values():
        assert resolver.resolve_committed(identity).base_model


def test_deployment_unavailable_degrades_without_substitution():
    config = config_for_policy()
    bound = bind_policy(config)
    snapshot = config.catalog_snapshot
    config.attach_catalog_snapshot(
        replace(snapshot, catalog=snapshot.catalog.model_copy(update={"models": {}}))
    )
    restored = resume_policy(config, bound.model_dump(mode="json"), BUILTIN_SUBAGENTS)
    assert restored.render_policy.mode.value == "standalone"
    assert restored.bindings == bound.bindings
    config.attach_dispatch_policy(restored)
    for name in bound.bindings:
        assert any(f"slot {name!r}:" in note for note in config.validation_warnings)
    assert not restored.roster.bindings


def test_resume_repair_clears_degradation_diagnostics(monkeypatch):
    from chartreux.app_server._projection import project_config_view
    from chartreux.core.dispatch.renderer import render_routing_region

    monkeypatch.setattr(
        "chartreux.utils.api_keys.resolve_api_key", lambda _: "test-key"
    )
    config = config_for_policy()
    snapshot = config.catalog_snapshot
    models = dict(snapshot.catalog.models)
    models["secondary"] = models["glm-5-3"]
    roles = dict(snapshot.catalog.roles)
    roles["scout"] = RoleDefinition(model="secondary", thinking="low")
    healthy = replace(
        snapshot,
        catalog=snapshot.catalog.model_copy(update={"models": models, "roles": roles}),
    )
    config.attach_catalog_snapshot(healthy)
    bound = bind_policy(config)
    assert not bound.roster.single_model

    broken_models = dict(models)
    broken_models.pop("secondary")
    config.attach_catalog_snapshot(
        replace(
            healthy,
            catalog=healthy.catalog.model_copy(update={"models": broken_models}),
        )
    )
    degraded = resume_policy(config, bound.model_dump(mode="json"), BUILTIN_SUBAGENTS)
    config.attach_dispatch_policy(degraded)
    assert "mechanical" in degraded.roster.failures
    assert any(
        "slot 'mechanical':" in note
        for note in project_config_view(config).validation_warnings
    )
    assert "unavailable" in render_routing_region(
        degraded.render_policy, degraded.roster
    )

    config.attach_catalog_snapshot(healthy)
    repaired = resume_policy(
        config, degraded.model_dump(mode="json"), BUILTIN_SUBAGENTS
    )
    config.attach_dispatch_policy(repaired)
    assert repaired.bindings == bound.bindings
    assert repaired.roster == bound.roster
    assert repaired.diagnostics == ()
    assert project_config_view(config).validation_warnings == []
    assert render_routing_region(
        repaired.render_policy, repaired.roster
    ) == render_routing_region(bound.render_policy, bound.roster)
    assert bind_policy(config).diagnostics == ()


@pytest.mark.asyncio
async def test_session_logger_loader_round_trip_binding(tmp_path):
    from chartreux.core.session.session_loader import SessionLoader
    from tests.conftest import build_test_agent_loop

    config = config_for_policy()
    config.session_logging.enabled = True
    config.session_logging.save_dir = str(tmp_path)
    agent = build_test_agent_loop(config=config)
    await agent._save_messages(allow_empty=True)
    assert agent.session_logger.session_dir is not None
    metadata = SessionLoader.load_metadata(agent.session_logger.session_dir)
    assert metadata.dispatch_policy == agent.bound_dispatch_policy.model_dump(
        mode="json"
    )
    resumed = resume_policy(config, metadata.dispatch_policy, BUILTIN_SUBAGENTS)
    assert resumed.bindings == agent.bound_dispatch_policy.bindings
