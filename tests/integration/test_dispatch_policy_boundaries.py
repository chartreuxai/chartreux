"""End-to-end boundaries for the retired dispatch axis and its successors.

Package 16b retired the dispatch policy bind: a legacy ``[dispatch]`` table
in ``models.toml`` is reported loudly with successor guidance and never
merged, delegated routing runs through ``[orchestration.routes.*]``, and the
session's frozen policy record is the package-10 bundle. These tests pin the
integration boundaries: the loud report through the config load, the
route-keyed task description, concrete launch resolution against an
attached catalog, the agent-bar projection, and the bundle storage
boundary.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from textual.app import App, ComposeResult

from chartreux.app_server.protocol import AgentSummaryModel
from chartreux.cli.textual_ui.widgets.agent_bar import AgentBar
from chartreux.core.agents.launch import resolve_launch
from chartreux.core.config import ChartreuxConfigSchema, SessionLoggingConfig
from chartreux.core.dispatch.renderer import task_description_for_config
from chartreux.core.dispatch.session import SessionPolicyError, resume_policy_bundle
from chartreux.core.model_catalog.loader import load_catalog
from chartreux.core.session.session_loader import SessionLoader
from chartreux.core.session_types import POLICY_STORAGE_VERSION
from chartreux.core.subagents import LaunchConfig, derive_agent_name
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.stubs.fake_config_orchestrator import FakeConfigOrchestrator


def _config(path: Path) -> ChartreuxConfigSchema:
    return build_test_vibe_config().attach_catalog_snapshot(load_catalog(path))


def test_legacy_dispatch_overlay_is_reported_and_concrete_launch_resolves(
    tmp_path: Path,
):
    path = tmp_path / "models.toml"
    path.write_text(
        '[roles.scout]\nthinking="medium"\n\n[dispatch]\nmode="standalone"\n'
    )
    config = _config(path)
    snapshot = config.catalog_snapshot
    assert snapshot is not None
    # The retired table is reported loudly with successor guidance and
    # carries no dispatch state at all.
    assert not hasattr(snapshot, "dispatch")
    assert len(snapshot.dispatch_diagnostics) == 1
    report = snapshot.dispatch_diagnostics[0]
    assert "[dispatch]" in report and "retired" in report
    assert "[orchestration.routes.<purpose>]" in report
    assert str(path) in report
    assert report in config.validation_warnings
    # The catalog beside the retired table loads normally.
    assert snapshot.catalog.roles["scout"].thinking == "medium"
    assert "### Launch targets" in task_description_for_config(config)

    # A concrete launch resolves against the attached catalog: the committed
    # model is the catalog's scout deployment at the catalog revision.
    parent = FakeConfigOrchestrator(config)
    candidate = resolve_launch(
        identity="coder",
        config=LaunchConfig(model="@scout"),
        parent_orchestrator=parent,
        tool_inventory={},
        authorized_tool_names=frozenset(),
    )
    assert candidate.identity == "coder"
    assert candidate.committed_model.catalog_revision == snapshot.revision
    assert candidate.effective_thinking == "medium"
    override = resolve_launch(
        identity="coder",
        config=LaunchConfig(model="@scout", thinking="high"),
        parent_orchestrator=parent,
        tool_inventory={},
        authorized_tool_names=frozenset(),
    )
    assert override.committed_model.base_model == candidate.committed_model.base_model
    assert override.effective_thinking == "high"


def test_catalog_edit_is_served_next_load_and_the_task_description_is_stable(
    tmp_path: Path,
):
    path = tmp_path / "models.toml"
    path.write_text(
        '[roles.scout]\nthinking="low"\n\n[dispatch]\nmode="orchestrated"\n'
    )
    config = _config(path)
    original_task = task_description_for_config(config)
    original = config.catalog_snapshot
    assert original is not None

    # A saved catalog edit applies to the next load: the retired table's
    # report is stable (it never echoes content), the catalog change is
    # served, and the route-keyed task description is stable because the
    # session model did not change.
    edited = '[roles.scout]\nthinking="high"\n\n[dispatch]\nmode="standalone"\n'
    path.write_text(edited)
    config.attach_catalog_snapshot(load_catalog(path))
    snapshot = config.catalog_snapshot
    assert snapshot is not None
    assert snapshot.dispatch_diagnostics == original.dispatch_diagnostics
    assert snapshot.catalog.roles["scout"].thinking == "high"
    assert snapshot.revision != original.revision
    assert task_description_for_config(config) == original_task
    # The load never rewrites the user's file.
    assert path.read_text() == edited


def test_invalid_dispatch_table_is_reported_loudly_and_never_repaired(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    path = tmp_path / "models.toml"
    original = '[dispatch]\nmode="invalid"\n'
    path.write_text(original)
    with caplog.at_level(logging.WARNING, logger="vibe"):
        config = _config(path)
    snapshot = config.catalog_snapshot
    assert snapshot is not None
    # The retired table is reported, never merged, never fatal: the snapshot
    # carries no dispatch state and the catalog serves normally.
    assert not hasattr(snapshot, "dispatch")
    assert len(snapshot.dispatch_diagnostics) == 1
    report = snapshot.dispatch_diagnostics[0]
    assert "[dispatch]" in report and "retired" in report
    assert report in config.validation_warnings
    # The loud report is logged at startup and the file is never repaired.
    assert report in {record.getMessage() for record in caplog.records}
    assert path.read_text() == original
    assert "### Launch targets" in task_description_for_config(config)


def test_legacy_roles_only_overlay_remains_usable(tmp_path: Path):
    path = tmp_path / "models.toml"
    path.write_text('[roles.main]\nthinking="medium"\n')
    config = _config(path)
    snapshot = config.catalog_snapshot
    assert snapshot is not None
    assert snapshot.dispatch_diagnostics == ()
    assert not hasattr(snapshot, "dispatch")
    assert snapshot.catalog.roles["main"].thinking == "medium"
    assert config.get_active_model().thinking == "medium"
    assert "### Launch targets" in task_description_for_config(config)


@pytest.mark.asyncio
async def test_launch_agent_bar_identity_hides_dispatch_internals(tmp_path: Path):
    path = tmp_path / "models.toml"
    path.write_text('[dispatch]\nmode="standalone"\n')
    config = _config(path)
    candidate = resolve_launch(
        identity="coder",
        config=LaunchConfig(model="@scout"),
        parent_orchestrator=FakeConfigOrchestrator(config),
        tool_inventory={},
        authorized_tool_names=frozenset(),
    )
    agent = AgentSummaryModel(
        agent_id="agent-1",
        identity=candidate.identity,
        availability="idle",
        base_model=candidate.committed_model.base_model,
        effective_model=candidate.effective_model.name,
        active_provider=candidate.effective_model.provider,
        effective_thinking=candidate.effective_thinking,
        name=derive_agent_name(
            agent_id="agent-1",
            identity=candidate.identity,
            identity_display_label=None,
            base_model=candidate.committed_model.base_model,
            effective_model=candidate.effective_model.name,
        ),
    )

    class Host(App[None]):
        def compose(self) -> ComposeResult:
            yield AgentBar()

    async with Host().run_test(size=(140, 24)) as pilot:
        bar = pilot.app.query_one(AgentBar)
        bar.update_agents((agent,))
        bar.open_browser()
        details = bar._agent_details(agent)
        metadata = bar.full_metadata(agent)
        assert f"Identity: {candidate.identity}" in metadata
        assert f"{agent.active_provider}/{agent.effective_model}" in details
        # The retired dispatch vocabulary never leaks into the projection.
        for internal in (
            "@scout",
            "dispatch",
            "slot",
            "preset",
            "roster",
            "implementor",
        ):
            assert internal not in details
            assert internal not in metadata


@pytest.mark.asyncio
async def test_persisted_policy_record_round_trips_through_the_bundle_boundary(
    tmp_path: Path,
):
    # Package 10: the persisted policy record of a conversation is its bound
    # bundle — the storage boundary activates it back to the same frozen
    # policy hash, and a legacy record (no bundle) rejects actionably.
    config = build_test_vibe_config(
        session_logging=SessionLoggingConfig(enabled=True, save_dir=str(tmp_path))
    )
    loop = build_test_agent_loop(config=config)
    try:
        bound = loop._bound_policy
        assert bound is not None
        await loop.persist_empty_session()
        session_dir = loop.session_logger.session_dir
        assert session_dir is not None
        metadata = SessionLoader.load_metadata(session_dir)
        assert metadata.policy_storage_version == POLICY_STORAGE_VERSION == 2
        assert metadata.policy_bundle is not None
        resumed = resume_policy_bundle(metadata)
        assert resumed.frozen_policy_hash == bound.frozen_policy_hash
        assert resumed.manifest.identity.selected == bound.manifest.identity.selected
        legacy = metadata.model_copy(
            update={"policy_bundle": None, "policy_storage_version": None}
        )
        with pytest.raises(SessionPolicyError, match="Start a new session"):
            resume_policy_bundle(legacy)
    finally:
        await loop.aclose()
