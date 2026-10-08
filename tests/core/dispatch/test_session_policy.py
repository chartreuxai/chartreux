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
    resume_policy,
)
from chartreux.core.model_catalog.defaults import SHIPPED_CATALOG
from chartreux.core.model_catalog.loader import CatalogSnapshot
from chartreux.core.model_catalog.schema import RoleDefinition
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


def test_catalog_edit_does_not_rebind_resumed_slots():
    config = config_for_policy()
    bound = bind_policy(config)
    snapshot = config.catalog_snapshot
    roles = dict(snapshot.catalog.roles)
    roles["medium"] = RoleDefinition(model="glm-5-3", thinking="off")
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
    roles["small"] = RoleDefinition(model="secondary", thinking="low")
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
