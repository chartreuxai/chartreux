"""The frozen-bundle session policy surface (the legacy bind's successor).

Package 16b retired the legacy session-policy bind (``BoundDispatchPolicy``,
``bind_snapshot_policy``, ``bind_policy``, ``resume_policy``): the successor is
the frozen-bundle surface package 10 shipped in
``chartreux.core.dispatch.session``. These tests pin that surface: the
validate/activate round trip, the legacy-record rejection, the corrupt-record
rejection at load, the identity binding, the fresh-bind validation, the
pending-change report, the session-logger round trip, and the identity
registry every built-in binds through.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import ValidationError
import pytest

from chartreux.core.agents.identities import (
    UnknownIdentityError,
    builtin_identities,
    builtin_identity_names,
    get_builtin_identity,
)
from chartreux.core.config import SessionLoggingConfig
from chartreux.core.config.harness_files import get_harness_files_manager
from chartreux.core.dispatch.session import (
    SessionPolicyError,
    bind_policy_bundle,
    bound_identity_reference,
    pending_policy_changes,
    resume_policy_bundle,
    validate_stored_policy_bundle,
)
from chartreux.core.model_catalog.defaults import SHIPPED_CATALOG
from chartreux.core.model_catalog.loader import CatalogSnapshot
from chartreux.core.policy.composition import ExecutionContext
from chartreux.core.session.session_loader import SessionLoader
from chartreux.core.session_types import POLICY_STORAGE_VERSION, SessionMetadata
from chartreux.core.skills.manager import SkillManager
from tests.conftest import build_test_agent_loop, build_test_vibe_config

ORCHESTRATED_SELECTION = {
    "identity": "orchestrator",
    "workflow": "ceremony",
    "orchestration_strategy": "builtin:default",
}


def _config(selection: dict[str, str] | None = None):
    return build_test_vibe_config(
        agent=selection or ORCHESTRATED_SELECTION
    ).attach_catalog_snapshot(CatalogSnapshot(SHIPPED_CATALOG, "original"))


def _bind(config):
    return bind_policy_bundle(
        config,
        execution_context=ExecutionContext.INTERACTIVE_MAIN,
        skill_manager=SkillManager(lambda: config),
        scratchpad_dir=None,
        cwd=Path.cwd(),
        harness_files=get_harness_files_manager(),
        role_instructions=None,
    )


def _metadata(
    bundle,
    *,
    session_id: str = "s1",
    storage_version: Literal[2] | None = POLICY_STORAGE_VERSION,
) -> SessionMetadata:
    return SessionMetadata(
        session_id=session_id,
        start_time="2026-01-01T00:00:00Z",
        end_time=None,
        git_commit=None,
        git_branch=None,
        username="tester",
        environment={},
        policy_bundle=bundle,
        policy_storage_version=storage_version,
    )


def test_validate_and_resume_round_trip_preserves_hash_and_identity() -> None:
    bundle, activated = _bind(_config())
    # The storage boundary revalidates the persisted payload structurally and
    # carries the same frozen-policy hash.
    reparsed = validate_stored_policy_bundle(bundle.model_dump(mode="json"))
    assert reparsed.manifest.frozen_policy_hash == activated.frozen_policy_hash
    # Resume activates the persisted record: same frozen hash, same identity
    # binding, served from the carried content.
    resumed = resume_policy_bundle(_metadata(bundle.model_dump(mode="json")))
    assert resumed.frozen_policy_hash == activated.frozen_policy_hash
    assert bound_identity_reference(resumed) == bound_identity_reference(activated)
    assert resumed.policy_text == activated.policy_text


def test_legacy_record_without_bundle_fails_with_start_a_new_session() -> None:
    legacy = SessionMetadata(
        session_id="legacy",
        start_time="2026-01-01T00:00:00Z",
        end_time=None,
        git_commit=None,
        git_branch=None,
        username="tester",
        environment={},
    )
    with pytest.raises(SessionPolicyError, match="Start a new session"):
        resume_policy_bundle(legacy)


def test_mismatched_storage_version_fails_with_start_a_new_session() -> None:
    bundle, _activated = _bind(_config())
    metadata = _metadata(bundle.model_dump(mode="json"), storage_version=None)
    with pytest.raises(SessionPolicyError, match="older harness format"):
        resume_policy_bundle(metadata)


def test_tampered_bundle_fails_resume_without_substitution() -> None:
    bundle, _activated = _bind(_config())
    payload = bundle.model_dump(mode="json")
    frozen_key = next(
        key
        for key, value in payload["frozen"].items()
        if isinstance(value, str) and len(value) > 100
    )
    payload["frozen"][frozen_key] += " tampered."
    with pytest.raises(SessionPolicyError, match="unreadable persisted policy bundle"):
        resume_policy_bundle(_metadata(payload))


def test_corrupt_bundle_fails_at_metadata_load() -> None:
    with pytest.raises(ValidationError, match="unsupported bundle format version"):
        SessionMetadata.model_validate({
            "session_id": "corrupt",
            "start_time": "2026-01-01T00:00:00Z",
            "end_time": None,
            "git_commit": None,
            "git_branch": None,
            "username": "tester",
            "environment": {},
            "policy_bundle": {"version": 99},
            "policy_storage_version": POLICY_STORAGE_VERSION,
        })


def test_bound_identity_reference_is_the_validated_selection() -> None:
    _bundle, activated = _bind(_config())
    assert bound_identity_reference(activated) == "orchestrator"


def test_fresh_bind_validates_the_identity_selection() -> None:
    # A bare custom identity with no user prompt file passes config schema
    # validation but fails the NEW binding's identity validation, closed.
    config = _config({**ORCHESTRATED_SELECTION, "identity": "no-such-custom"})
    with pytest.raises(UnknownIdentityError, match="no-such-custom"):
        _bind(config)


def test_pending_policy_changes_reports_never_applies() -> None:
    bundle, activated = _bind(_config())
    harness_files = get_harness_files_manager()
    # The bound selections match the saved state: nothing is pending.
    assert (
        pending_policy_changes(_config(), activated, harness_files=harness_files) == ()
    )
    # A saved selection change is REPORTED against the bound bundle and never
    # applied to the conversation.
    changed = _config({**ORCHESTRATED_SELECTION, "identity": "coder"})
    notes = pending_policy_changes(changed, activated, harness_files=harness_files)
    assert len(notes) == 1
    assert "identity" in notes[0]
    assert "'orchestrator'" in notes[0]
    assert "'coder'" in notes[0]
    assert "new conversations only" in notes[0]


def test_identity_registry_is_the_builtin_subagent_surface() -> None:
    # BUILTIN_SUBAGENTS' successor: the identity registry's five built-ins.
    assert builtin_identity_names() == (
        "advisor",
        "coder",
        "orchestrator",
        "researcher",
        "reviewer",
    )
    assert len(builtin_identities()) == 5
    for name in builtin_identity_names():
        identity = get_builtin_identity(name)
        assert identity is not None
        assert identity.id == name


@pytest.mark.parametrize("identity", builtin_identity_names())
def test_every_builtin_identity_binds_a_valid_bundle(identity: str) -> None:
    config = _config({**ORCHESTRATED_SELECTION, "identity": identity})
    bundle, activated = _bind(config)
    assert bound_identity_reference(activated) == identity
    reparsed = validate_stored_policy_bundle(bundle.model_dump(mode="json"))
    assert reparsed.manifest.frozen_policy_hash == activated.frozen_policy_hash
    resumed = resume_policy_bundle(_metadata(bundle.model_dump(mode="json")))
    assert resumed.frozen_policy_hash == activated.frozen_policy_hash
    assert bound_identity_reference(resumed) == identity


@pytest.mark.asyncio
async def test_session_logger_round_trip_persists_the_bound_bundle(tmp_path) -> None:
    config = build_test_vibe_config(
        agent=ORCHESTRATED_SELECTION,
        session_logging=SessionLoggingConfig(enabled=True, save_dir=str(tmp_path)),
    ).attach_catalog_snapshot(CatalogSnapshot(SHIPPED_CATALOG, "original"))
    agent = build_test_agent_loop(config=config)
    try:
        bound = agent._bound_policy
        assert bound is not None
        await agent.persist_empty_session()
        session_dir = agent.session_logger.session_dir
        assert session_dir is not None
        metadata = SessionLoader.load_metadata(session_dir)
        assert metadata.policy_storage_version == POLICY_STORAGE_VERSION == 2
        assert metadata.policy_bundle is not None
        # The persisted record validates at the storage boundary and carries
        # the loop's bound frozen-policy hash and identity binding.
        reparsed = validate_stored_policy_bundle(metadata.policy_bundle)
        assert reparsed.manifest.frozen_policy_hash == bound.frozen_policy_hash
        resumed = resume_policy_bundle(metadata)
        assert resumed.frozen_policy_hash == bound.frozen_policy_hash
        assert bound_identity_reference(resumed) == bound_identity_reference(bound)
    finally:
        await agent.aclose()
