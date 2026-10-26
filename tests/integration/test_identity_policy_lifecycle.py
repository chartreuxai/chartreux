"""Package 10 lifecycle integration: the bundle is the persisted policy record.

Design §6/§10: a conversation's frozen policy bundle is captured once at
startup, persisted inline in the session record, and served by every prompt
refresh — refresh, reload, retask, and resume never reread the frozen policy
from disk. These tests exercise that lifecycle end to end (startup bind,
reload reporting, the five MCP refresh paths, retask, in-process and
fresh-runtime resume after source deletion, /clear, prepare failure, legacy
records, and the version pins), independently of the package-11 entrypoint
cutover.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
import json
from pathlib import Path
from typing import Literal
from unittest.mock import AsyncMock, MagicMock

import pytest

from chartreux.app_server._runtime import AgentRuntimeFactory, _RootRuntimeBlueprint
from chartreux.app_server._session_backend_port import (
    MCPAuthorizationRef,
    MCPAuthorizationRequired,
    MCPAuthorizationResult,
    MCPAuthorizationSnapshot,
    ResolvedMCPCatalog,
    ResolvedMCPServerConfig,
)
from chartreux.app_server._sessions import SessionRuntime, SessionRuntimeRegistry
from chartreux.app_server.client import AppServerClient
from chartreux.app_server.protocol import ClientCapabilities, ClientInfo, SessionOptions
from chartreux.app_server.transport import memory_transport_pair
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.agents.identities import UnknownIdentityError
from chartreux.core.agents.launch import (
    FrozenPersona,
    LaunchAuthorityInputs,
    LaunchCandidate,
)
from chartreux.core.config import ChartreuxConfigSchema, MCPStdio, SessionLoggingConfig
from chartreux.core.config.harness_files import HarnessFilesManager
from chartreux.core.dispatch.renderer import task_description_for_config
from chartreux.core.dispatch.route_models import PackageTier, PurposeRoute, RouteTarget
from chartreux.core.dispatch.routes import RouteResolutionError, shipped_routes
from chartreux.core.dispatch.session import SessionPolicyError, resume_policy_bundle
from chartreux.core.events import RouteTableAppliedEvent
from chartreux.core.hooks.models import HookConfigResult
from chartreux.core.policy.bundle import BUNDLE_FORMAT_VERSION, BundleStoreError
from chartreux.core.session.session_loader import SessionLoader
from chartreux.core.session_types import (
    POLICY_STORAGE_VERSION,
    LaunchMetadataV1,
    LaunchMetadataV2,
    LaunchPersonaV1,
    LaunchPersonaV2,
)
from chartreux.core.subagents import InvalidLaunchConfigError, LaunchConfig, TaskArgs
from chartreux.core.tools.base import InvokeContext, ToolPermission
from chartreux.core.tools.builtins.read_file import ReadFile, ReadFileArgs
from chartreux.core.tools.manager import NoSuchToolError, ToolManager
from chartreux.utils.cache_store import FileSystemCacheStore
from tests.agent_loop.test_accepted_source_snapshot import make_orchestrator
from tests.app_server._subagent_test_support import (
    BlockingBackend,
    _background_result,
    _retention_registry,
)
from tests.conftest import (
    build_test_agent_loop,
    build_test_vibe_config,
    set_agent_config,
    stub_config_reload,
)
from tests.stubs.app_server import (
    attach_test_app_server_session,
    build_test_app_server,
    legacy_backend,
)
from tests.stubs.fake_backend import FakeBackend
from tests.stubs.fake_config_orchestrator import FakeConfigOrchestrator


@pytest.mark.asyncio
async def test_zero_config_startup_binds_shipped_coder_minimal_none(
    tmp_path: Path,
) -> None:
    # Design criterion 1, first clause: the shipped coder/minimal/none
    # selection works with ZERO configuration — a bare config binds at
    # startup and persists as the conversation's policy record.
    config = build_test_vibe_config(
        session_logging=SessionLoggingConfig(enabled=True, save_dir=str(tmp_path))
    )
    assert config.agent.identity == "coder"
    assert config.agent.workflow == "minimal"
    assert config.agent.orchestration_strategy == "none"
    loop = build_test_agent_loop(config=config)
    try:
        bound = loop._bound_policy
        assert bound is not None
        assert bound.manifest.identity.selected == "coder"
        assert bound.manifest.workflow.selected == "minimal"
        assert bound.manifest.strategy.selected == "none"
        assert "You are the coder" in (loop.messages[0].content or "")
        await loop.persist_empty_session()
        session_dir = loop.session_logger.session_dir
        assert session_dir is not None
        metadata = SessionLoader.load_metadata(session_dir)
        assert metadata.policy_storage_version == POLICY_STORAGE_VERSION == 2
        assert metadata.policy_bundle is not None
    finally:
        await loop.aclose()


@pytest.mark.asyncio
async def test_startup_binds_once_and_refresh_serves_the_frozen_bundle() -> None:
    # Startup binds ONCE: the first render captures the bundle, and a later
    # refresh serves it even after a live selection edit (the invariant: no
    # live reread of frozen policy during prompt refresh).
    loop = build_test_agent_loop()
    try:
        bound = loop._bound_policy
        assert bound is not None
        initial_prompt = loop.messages[0].content
        binding = loop._identity_binding()

        loop.config.agent.identity = "reviewer"
        await loop.refresh_system_prompt()

        assert loop._bound_policy is bound
        assert loop._identity_binding() == binding
        assert loop._bound_identity == "coder"
        assert loop.messages[0].content == initial_prompt
    finally:
        await loop.aclose()


@pytest.mark.asyncio
async def test_reload_retains_frozen_bytes_and_reports_pending_selection() -> None:
    # Design §10: a config reload mid-conversation does not re-posture — the
    # frozen bytes are retained and the saved [agent] selection change is
    # reported as pending, applying to new conversations only.
    loop = build_test_agent_loop()
    try:
        bound = loop._bound_policy
        assert bound is not None
        frozen_prompt = loop.messages[0].content

        set_agent_config(loop, build_test_vibe_config(agent={"identity": "reviewer"}))
        await loop.reload_with_initial_messages()

        assert loop._bound_policy is bound
        assert loop.messages[0].content == frozen_prompt
        assert any(
            "identity selection change" in note for note in loop.pending_policy_changes
        )
    finally:
        await loop.aclose()


@pytest.mark.asyncio
async def test_reload_reports_pending_identity_resource_content_change(
    config_dir: Path,
) -> None:
    # The same reporting contract for resource CONTENT: the bound identity's
    # prompt file is edited on disk mid-conversation; the reload reports the
    # pending content change and keeps the frozen bytes.
    prompt_file = config_dir / "prompts" / "lifecycle-probe.md"
    prompt_file.parent.mkdir(parents=True, exist_ok=True)
    prompt_file.write_text(
        "# Lifecycle probe\n\nYou are the lifecycle probe persona.\n", encoding="utf-8"
    )
    loop = build_test_agent_loop(
        config=build_test_vibe_config(agent={"identity": "user:lifecycle-probe"})
    )
    try:
        bound = loop._bound_policy
        assert bound is not None
        frozen_prompt = loop.messages[0].content
        assert "lifecycle probe persona" in (frozen_prompt or "")

        prompt_file.write_text(
            "# Lifecycle probe\n\nYou are the EDITED probe persona.\n", encoding="utf-8"
        )
        await loop.reload_with_initial_messages()

        assert loop._bound_policy is bound
        assert loop.messages[0].content == frozen_prompt
        assert any(
            "identity resource content change" in note
            for note in loop.pending_policy_changes
        )
    finally:
        await loop.aclose()


@pytest.mark.asyncio
async def test_reload_rebuilds_skill_eligibility_from_the_frozen_policy() -> None:
    # Round-2 Ab6: reload rebuilds the SkillManager around the RELOADED
    # (pending) config, but eligibility must be rebuilt from the SAME
    # frozen policy source the loop reports — a pending ceremony
    # selection applies to new conversations only and never widens the
    # bound conversation's eligibility.
    loop = build_test_agent_loop()
    try:
        assert loop.skill_manager.policy_facts.workflow == "minimal"
        assert "main-review" not in loop.skill_manager.available_skills
        assert not loop.skill_manager.skill_eligibility("main-review").eligible

        set_agent_config(loop, build_test_vibe_config(agent={"workflow": "ceremony"}))
        await loop.reload_with_initial_messages()

        # The loop reports the pending selection for new conversations...
        assert any(
            "workflow selection change" in note for note in loop.pending_policy_changes
        )
        # ...and the rebuilt SkillManager stays on the frozen policy: the
        # ceremony-bearing skill stays withheld, the catalog still lists it
        # (filtering is not pruning), and the diagnostic names the frozen
        # workflow.
        assert loop.skill_manager.policy_facts.workflow == "minimal"
        assert "main-review" not in loop.skill_manager.available_skills
        assert not loop.skill_manager.skill_eligibility("main-review").eligible
        assert "main-review" in loop.skill_manager.settings_inventory
        diagnostic = loop.skill_manager.incompatibility_diagnostic("main-review")
        assert diagnostic is not None and "minimal" in diagnostic
    finally:
        await loop.aclose()


def _research_route_rebound_to_coder() -> dict[str, PurposeRoute]:
    """The final-release-review EF1 repro: research researcher -> coder."""
    return _research_route_rebound_to("coder")


def _research_route_rebound_to(identity: str) -> dict[str, PurposeRoute]:
    """Rebind the research route to ``identity``; the rest stays shipped."""
    routes = dict(shipped_routes())
    routes["research"] = PurposeRoute(
        targets=(
            RouteTarget(
                model="$session", identity=identity, tier=PackageTier.SUBSTANTIVE
            ),
        )
    )
    return routes


def _admission_registry(
    loop: AgentLoop,
) -> tuple[SessionRuntimeRegistry, SessionRuntime]:
    """A minimal root registry so a test can drive launch admission.

    ``ChildLaunchAdmission.resolve_candidate`` is the production admission
    seam (pure resolution over the parent runtime): driving it pins the
    route SOURCE admission consumes, not just the resolution function.
    """
    registry = SessionRuntimeRegistry(AsyncMock(), AsyncMock(), lambda _: 0)
    root = registry._build_child_runtime(loop)
    registry.bind_root(root)
    return registry, root


@pytest.mark.asyncio
async def test_reload_applies_the_edited_route_to_subsequent_launches_and_reports_it() -> (
    None
):
    # ADR 0022 (P1-i-a), checks 2 and 4: purpose routes are bindings, not
    # frozen content. A validated route edit — the reviewer's offline
    # repro: research researcher -> coder — APPLIES at the reload commit:
    # launch admission resolves the edited table for every subsequent
    # launch, the applied report enumerates the edit old -> new per purpose,
    # and the conversation-visible notice is queued for publication. The
    # route edit is no longer reported as pending, while the identity
    # selection change (frozen content) still is.
    loop = build_test_agent_loop()
    registry, root = _admission_registry(loop)
    try:
        bound = loop._bound_policy
        assert bound is not None
        before = registry._launch_admission.resolve_candidate(
            root, TaskArgs(task="inspect", purpose="research")
        )
        assert before.identity == "researcher"
        assert loop.live_purpose_routes["research"].targets[0].identity == "researcher"

        set_agent_config(
            loop,
            build_test_vibe_config(
                agent={"identity": "reviewer"},
                orchestration={"routes": _research_route_rebound_to_coder()},
            ),
        )
        await loop.reload_with_initial_messages()

        # The frozen bundle (and its persisted route snapshot) is untouched
        # by the reload — it stays the integrity and diff anchor...
        assert loop._bound_policy is bound
        # ...the live table carries the edit...
        assert (
            loop.config.purpose_routes.routes["research"].targets[0].identity == "coder"
        )
        # ...admission resolves the EDITED route for subsequent launches...
        after = registry._launch_admission.resolve_candidate(
            root, TaskArgs(task="inspect", purpose="research")
        )
        assert after.identity == "coder"
        assert loop.live_purpose_routes["research"].targets[0].identity == "coder"
        # ...the applied report enumerates the edit old -> new...
        changes = loop.applied_route_changes
        assert [change.purpose for change in changes] == ["research"]
        assert "researcher" in changes[0].before
        assert "coder" in changes[0].after
        # ...the conversation-visible notice is queued for publication...
        queued: list[object] = []
        while not loop._out_of_band_events.empty():
            queued.append(loop._out_of_band_events.get_nowait())
        route_events = [
            event for event in queued if isinstance(event, RouteTableAppliedEvent)
        ]
        assert len(route_events) == 1
        assert route_events[0].session_id == loop.session_id
        assert route_events[0].changes == changes
        # ...the identity selection change is still reported as pending
        # (frozen content)...
        assert any(
            "identity selection change" in note for note in loop.pending_policy_changes
        )
        # ...and the route edit is NOT pending: the "applies to new
        # conversations only" route text is gone.
        assert not any(
            "[orchestration.routes]" in note for note in loop.pending_policy_changes
        )
    finally:
        await loop.aclose()


@pytest.mark.asyncio
async def test_in_flight_child_keeps_its_binding_while_a_new_launch_adopts_the_edit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # ADR 0022 checks 1 and 2: admission is the binding point. A launch
    # admitted BEFORE the route edit keeps its committed model and identity
    # (the T1-pinned record), while a fresh launch after the reload resolves
    # the edited target (the T2 record) — the distinguishing pair in one
    # conversation.
    backends: list[BlockingBackend] = []

    def create_backend(*, provider, **_kwargs):
        backend = BlockingBackend()
        backends.append(backend)
        return backend

    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", create_backend
    )

    def launch_config() -> ChartreuxConfigSchema:
        return build_test_vibe_config(
            enabled_tools=["task"], tools={"task": {"permission": "always"}}
        )

    parent = build_test_agent_loop(config=launch_config(), backend=FakeBackend())
    registry = _retention_registry()
    root = registry._build_child_runtime(parent)
    root.turns._projector = MagicMock()
    root.turns.link_subagent = AsyncMock()
    registry.bind_root(root)
    context = InvokeContext(tool_call_id="route-edit", session_id=parent.session_id)
    try:
        # T1: admitted on the shipped research route (researcher).
        first = await _background_result(
            registry,
            TaskArgs(task="first", purpose="research", background=True, retain=True),
            context,
        )
        assert first.agent_id is not None
        record = registry._agent_records[first.agent_id]
        assert record.identity == "researcher"
        assert record.base_model is not None

        # Wait until the child's turn is blocked inside a backend call.
        # The child may create more than one backend over its lifetime
        # (the turn's attempt backend), so poll the created set; from
        # that point the committed model is stable and the pin below is
        # exact.
        async def wait_blocked() -> BlockingBackend:
            for _ in range(500):
                for backend in backends:
                    if backend.started.is_set():
                        return backend
                await asyncio.sleep(0.01)
            raise TimeoutError("child never reached a backend call")

        blocked = await asyncio.wait_for(wait_blocked(), timeout=5)
        assert record.runtime.agent_loop.backend is blocked
        t1_model = record.runtime.agent_loop.committed_model
        assert t1_model is not None
        assert t1_model.base_model == record.base_model

        # The route edit + reload lands while T1 is in flight.
        edited = build_test_vibe_config(
            enabled_tools=["task"],
            tools={"task": {"permission": "always"}},
            orchestration={"routes": _research_route_rebound_to_coder()},
        )
        set_agent_config(parent, edited)
        await parent.reload_with_initial_messages()

        # Check 1: the in-flight child's record keeps its committed model
        # and identity across the edit.
        assert registry._agent_records[first.agent_id] is record
        assert record.identity == "researcher"
        assert record.base_model == t1_model.base_model
        assert record.runtime.agent_loop.committed_model is t1_model
        assert record.runtime.agent_loop._bound_identity == "researcher"

        # Check 2: a fresh launch after the reload resolves the edited
        # target — the T2 record in the same conversation.
        second = await _background_result(
            registry,
            TaskArgs(task="second", purpose="research", background=True, retain=True),
            context,
        )
        assert second.agent_id is not None
        assert second.agent_id != first.agent_id
        t2 = registry._agent_records[second.agent_id]
        assert t2.identity == "coder"
        assert t2.base_model == t1_model.base_model
        assert t2.runtime.agent_loop.committed_model is not None
        assert t2.runtime.agent_loop._bound_identity == "coder"
    finally:
        for backend in backends:
            backend.release.set()
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_clear_binds_the_edited_route_for_the_new_conversation(
    tmp_path: Path,
) -> None:
    # /clear binds a FRESH bundle from the current selections (design §10),
    # and the new conversation's admission resolves the edited route: the
    # live table carries the edit from the start.
    config = build_test_vibe_config(
        session_logging=SessionLoggingConfig(enabled=True, save_dir=str(tmp_path))
    )
    loop = build_test_agent_loop(config=config)
    registry, root = _admission_registry(loop)
    try:
        first_bundle = loop._bound_policy
        assert first_bundle is not None
        # The OLD conversation launches on its live (shipped) route...
        before = registry._launch_admission.resolve_candidate(
            root, TaskArgs(task="inspect", purpose="research")
        )
        assert before.identity == "researcher"
        assert loop.live_purpose_routes["research"].targets[0].identity == "researcher"

        set_agent_config(
            loop,
            build_test_vibe_config(
                orchestration={"routes": _research_route_rebound_to_coder()},
                session_logging=SessionLoggingConfig(
                    enabled=True, save_dir=str(tmp_path)
                ),
            ),
        )
        await loop.clear_history()

        # ...and /clear REBINDS the conversation: the new conversation's
        # live route table is the edited one, and its admission resolves it.
        assert loop._bound_policy is not first_bundle
        assert loop._bound_policy is not None
        assert loop.live_purpose_routes["research"].targets[0].identity == "coder"
        candidate = registry._launch_admission.resolve_candidate(
            root, TaskArgs(task="inspect", purpose="research")
        )
        assert candidate.identity == "coder"
    finally:
        await loop.aclose()


@pytest.mark.asyncio
async def test_resume_adopts_the_current_saved_route_table(tmp_path: Path) -> None:
    # ADR 0022 resume half (check 5): the persisted bundle round-trips the
    # route snapshot as the integrity and diff anchor, but a resumed
    # conversation's ADMISSION resolves the current saved table — the table
    # the source loop's orchestrator reloaded at resume — not the persisted
    # snapshot.
    config = build_test_vibe_config(
        orchestration={"routes": _research_route_rebound_to_coder()},
        session_logging=SessionLoggingConfig(enabled=True, save_dir=str(tmp_path)),
    )
    saved = build_test_agent_loop(config=config)
    await saved.persist_empty_session()
    saved_session_id = saved.session_id
    await saved.aclose()

    # The source loop's current saved state carries the SHIPPED research
    # route (researcher) again.
    source = build_test_agent_loop(
        config=build_test_vibe_config(
            session_logging=SessionLoggingConfig(enabled=True, save_dir=str(tmp_path))
        )
    )
    registry, root = _admission_registry(source)
    try:
        await AgentRuntimeFactory().resume_root(source, saved_session_id)
        await source.wait_until_ready()
        assert source.session_id == saved_session_id

        # The persisted bundle round-trips the frozen route snapshot (the
        # diff anchor)...
        resumed = source._bound_policy
        assert resumed is not None
        research_entry = next(
            entry
            for entry in resumed.manifest.route_snapshot
            if entry.purpose == "research"
        )
        assert research_entry.targets[0].identity == "coder"
        # ...but the resumed conversation's live route table is the current
        # saved one...
        assert (
            source.live_purpose_routes["research"].targets[0].identity == "researcher"
        )
        # ...and admission launches on the current saved route, not the
        # persisted snapshot.
        candidate = registry._launch_admission.resolve_candidate(
            root, TaskArgs(task="inspect", purpose="research")
        )
        assert candidate.identity == "researcher"
    finally:
        await source.aclose()


def _served_task_description(loop: AgentLoop) -> str:
    """The model-facing task-tool description the loop's manager serves."""
    specs = {spec.name: spec for spec in loop.tool_manager.available_tool_specs()}
    return specs["task"].description


def _launch_target_row(description: str, purpose: str) -> str:
    """The rendered launch-target row a task-tool description advertises."""
    rows = [
        line for line in description.splitlines() if line.startswith(f"| `{purpose}` |")
    ]
    assert len(rows) == 1
    return rows[0]


@pytest.mark.asyncio
async def test_reload_serves_the_edited_route_table_in_the_task_tool_description() -> (
    None
):
    # ADR 0022 check 3 (no divergence): the model-facing task-tool
    # description is rebuilt from the SAME live route table launch admission
    # resolves against, in the same synchronous reload commit — the served
    # launch-target table equals the admission table after the reload.
    loop = build_test_agent_loop()
    registry, root = _admission_registry(loop)
    try:
        before = registry._launch_admission.resolve_candidate(
            root, TaskArgs(task="inspect", purpose="research")
        )
        assert before.identity == "researcher"
        initial_row = _launch_target_row(_served_task_description(loop), "research")
        assert "`researcher`" in initial_row

        set_agent_config(
            loop,
            build_test_vibe_config(
                orchestration={"routes": _research_route_rebound_to_coder()}
            ),
        )
        await loop.reload_with_initial_messages()

        # Admission launches the EDITED route...
        after = registry._launch_admission.resolve_candidate(
            root, TaskArgs(task="inspect", purpose="research")
        )
        assert after.identity == "coder"
        # ...and the model-facing description advertises the SAME edited
        # route: the reload re-rendered it from the live table.
        served_row = _launch_target_row(_served_task_description(loop), "research")
        assert "`coder`" in served_row
        assert served_row != initial_row
    finally:
        await loop.aclose()


@pytest.mark.asyncio
async def test_resume_serves_the_current_saved_route_table_in_the_task_tool_description(
    tmp_path: Path,
) -> None:
    # ADR 0022 check 3 at the resume boundary: after a disk resume the
    # model-facing task-tool description renders from the CURRENT SAVED
    # route table — the same source launch admission resolves against — so
    # the served launch-target table can never diverge from admission
    # post-resume. The persisted snapshot stays the integrity and diff
    # anchor (the resume notice diffs against it), never the served table.
    config = build_test_vibe_config(
        orchestration={"routes": _research_route_rebound_to_coder()},
        session_logging=SessionLoggingConfig(enabled=True, save_dir=str(tmp_path)),
    )
    saved = build_test_agent_loop(config=config)
    persisted_row = _launch_target_row(_served_task_description(saved), "research")
    assert "`coder`" in persisted_row
    await saved.persist_empty_session()
    saved_session_id = saved.session_id
    await saved.aclose()

    # The source loop's current saved state carries the SHIPPED research
    # route (researcher) again.
    source = build_test_agent_loop(
        config=build_test_vibe_config(
            session_logging=SessionLoggingConfig(enabled=True, save_dir=str(tmp_path))
        )
    )
    registry, root = _admission_registry(source)
    try:
        await AgentRuntimeFactory().resume_root(source, saved_session_id)
        await source.wait_until_ready()
        assert source.session_id == saved_session_id

        # Admission launches the CURRENT SAVED route (the resumed
        # conversation adopts the live table)...
        candidate = registry._launch_admission.resolve_candidate(
            root, TaskArgs(task="inspect", purpose="research")
        )
        assert candidate.identity == "researcher"
        # ...and the model-facing description advertises the SAME live
        # route: no divergence post-resume (P1-i-a's check-3 pattern at the
        # resume boundary).
        served_row = _launch_target_row(_served_task_description(source), "research")
        assert "`researcher`" in served_row
        assert served_row != persisted_row
        # Control: the served row equals the live config's own render.
        live_row = _launch_target_row(
            task_description_for_config(source.config), "research"
        )
        assert served_row == live_row
    finally:
        await source.aclose()


@pytest.mark.asyncio
async def test_resume_reports_the_route_diff_against_the_persisted_snapshot(
    tmp_path: Path,
) -> None:
    # ADR 0022 check 5's notice half: a resumed conversation adopts the
    # CURRENT SAVED route table, and the resume notice reports the diff
    # against the persisted bundle's bind-time snapshot through the same
    # applied-notice channel the reload commit publishes through.
    config = build_test_vibe_config(
        orchestration={"routes": _research_route_rebound_to_coder()},
        session_logging=SessionLoggingConfig(enabled=True, save_dir=str(tmp_path)),
    )
    saved = build_test_agent_loop(config=config)
    await saved.persist_empty_session()
    saved_session_id = saved.session_id
    await saved.aclose()

    # The source loop's current saved state carries the SHIPPED research
    # route (researcher) again.
    source = build_test_agent_loop(
        config=build_test_vibe_config(
            session_logging=SessionLoggingConfig(enabled=True, save_dir=str(tmp_path))
        )
    )
    try:
        await AgentRuntimeFactory().resume_root(source, saved_session_id)
        await source.wait_until_ready()
        assert source.session_id == saved_session_id

        # The resumed conversation adopts the CURRENT SAVED route...
        assert (
            source.live_purpose_routes["research"].targets[0].identity == "researcher"
        )
        # ...and the resume notice reports the diff against the persisted
        # bundle's bind-time snapshot (coder -> researcher) through the same
        # applied-notice channel the reload commit publishes through.
        queued: list[object] = []
        while not source._out_of_band_events.empty():
            queued.append(source._out_of_band_events.get_nowait())
        route_events = [
            event for event in queued if isinstance(event, RouteTableAppliedEvent)
        ]
        assert len(route_events) == 1
        assert route_events[0].session_id == source.session_id
        changes = route_events[0].changes
        assert [change.purpose for change in changes] == ["research"]
        assert "coder" in changes[0].before
        assert "researcher" in changes[0].after
        # The loop's applied report carries the same diff.
        assert source.applied_route_changes == changes
    finally:
        await source.aclose()


@pytest.mark.asyncio
async def test_reload_notices_diff_against_the_previous_live_table() -> None:
    # R7 (ADR 0022 notice fix): the reload's applied notice diffs against
    # the previous LIVE table, not the frozen bind-time snapshot. Bind at
    # the shipped table (A); A -> B reports "A -> B"; B -> C reports
    # "B -> C"; a reload that leaves the live table unchanged reports
    # nothing; a revert to A reports "C -> A". The frozen snapshot stays
    # the integrity anchor and the resume notice's diff base.
    loop = build_test_agent_loop()
    try:
        bound = loop._bound_policy
        assert bound is not None
        assert loop._last_seen_purpose_routes is None

        async def reload_with(
            routes: dict[str, PurposeRoute],
        ) -> tuple[RouteTableAppliedEvent, ...]:
            set_agent_config(
                loop, build_test_vibe_config(orchestration={"routes": routes})
            )
            await loop.reload_with_initial_messages()
            queued: list[object] = []
            while not loop._out_of_band_events.empty():
                queued.append(loop._out_of_band_events.get_nowait())
            return tuple(
                event for event in queued if isinstance(event, RouteTableAppliedEvent)
            )

        # A -> B: the first reload diffs against the bind-time table.
        events = await reload_with(_research_route_rebound_to("coder"))
        assert len(events) == 1
        changes = events[0].changes
        assert [change.purpose for change in changes] == ["research"]
        assert "researcher" in changes[0].before
        assert "coder" in changes[0].after
        assert loop.applied_route_changes == changes
        assert loop._last_seen_purpose_routes == dict(loop.config.purpose_routes.routes)

        # B -> C: the notice diffs against the previous LIVE table.
        events = await reload_with(_research_route_rebound_to("reviewer"))
        assert len(events) == 1
        changes = events[0].changes
        assert [change.purpose for change in changes] == ["research"]
        assert "coder" in changes[0].before
        assert "reviewer" in changes[0].after

        # A reload that leaves the live table unchanged reports nothing.
        events = await reload_with(_research_route_rebound_to("reviewer"))
        assert events == ()
        assert loop.applied_route_changes == ()

        # C -> A (revert): the notice reports the revert against the
        # previous live table, never the bind-time snapshot.
        events = await reload_with(dict(shipped_routes()))
        assert len(events) == 1
        changes = events[0].changes
        assert [change.purpose for change in changes] == ["research"]
        assert "reviewer" in changes[0].before
        assert "researcher" in changes[0].after

        # The frozen bundle (and its persisted route snapshot) is untouched
        # by every reload — it stays the integrity anchor.
        assert loop._bound_policy is bound
    finally:
        await loop.aclose()


@pytest.mark.asyncio
async def test_reload_notice_diffs_against_the_live_table_when_the_last_seen_table_is_stale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Round-3 F7: the last-seen live table goes stale when the live table is
    # swapped without a consumer commit (what refresh_config or a raw
    # orchestrator reload does). A reload through the orchestrator path
    # commits BEFORE the orchestrator accepts the candidate, so the live
    # table at commit time is still the swapped one — the notice must diff
    # against THAT table, never the stale last-seen one.
    loop = build_test_agent_loop()
    try:
        bound = loop._bound_policy
        assert bound is not None

        def drain() -> list[RouteTableAppliedEvent]:
            queued: list[object] = []
            while not loop._out_of_band_events.empty():
                queued.append(loop._out_of_band_events.get_nowait())
            return [
                event for event in queued if isinstance(event, RouteTableAppliedEvent)
            ]

        # A -> B through the install-then-catch-up path: the notice diffs
        # against the last-seen live table (the pinned R7 semantics).
        set_agent_config(
            loop,
            build_test_vibe_config(
                orchestration={"routes": _research_route_rebound_to("coder")}
            ),
        )
        await loop.reload_with_initial_messages()
        events = drain()
        assert len(events) == 1
        changes = events[0].changes
        assert [change.purpose for change in changes] == ["research"]
        assert "coder" in changes[0].after
        assert loop._last_seen_purpose_routes == dict(loop.config.purpose_routes.routes)

        # B -> C behind the back: the live table is swapped with no consumer
        # commit, so the last-seen table stays at B while the live table is C.
        set_agent_config(
            loop,
            build_test_vibe_config(
                orchestration={"routes": _research_route_rebound_to("reviewer")}
            ),
        )
        assert loop._last_seen_purpose_routes != dict(loop.config.purpose_routes.routes)

        # C -> D through the orchestrator path: _commit_reload runs before the
        # orchestrator accepts the candidate, so the live table at commit time
        # is still C — the notice must report C -> D, not the stale B -> D.
        candidate = build_test_vibe_config(
            orchestration={"routes": _research_route_rebound_to("advisor")}
        )
        stub_config_reload(monkeypatch, candidate)
        await loop.reload_with_initial_messages(reload_config=True)
        events = drain()
        assert len(events) == 1
        changes = events[0].changes
        assert [change.purpose for change in changes] == ["research"]
        assert "reviewer" in changes[0].before
        assert "advisor" in changes[0].after
        assert loop.applied_route_changes == changes
        assert loop._last_seen_purpose_routes == dict(loop.config.purpose_routes.routes)

        # The frozen bundle (and its persisted route snapshot) is untouched —
        # it stays the integrity anchor.
        assert loop._bound_policy is bound
    finally:
        await loop.aclose()


@pytest.mark.asyncio
async def test_resume_seeds_the_last_seen_live_table_for_later_reload_notices(
    tmp_path: Path,
) -> None:
    # R7 (ADR 0022 notice fix, resume half): resume adopts the current
    # saved table and the resume notice still diffs against the persisted
    # bind-time snapshot (P1-i-b unchanged); the last-seen LIVE table
    # starts at the adopted table, so the next reload's notice diffs
    # against the adopted live table — never the persisted snapshot.
    config = build_test_vibe_config(
        orchestration={"routes": _research_route_rebound_to_coder()},
        session_logging=SessionLoggingConfig(enabled=True, save_dir=str(tmp_path)),
    )
    saved = build_test_agent_loop(config=config)
    await saved.persist_empty_session()
    saved_session_id = saved.session_id
    await saved.aclose()

    # The source loop's current saved state carries the SHIPPED research
    # route (researcher) again.
    source = build_test_agent_loop(
        config=build_test_vibe_config(
            session_logging=SessionLoggingConfig(enabled=True, save_dir=str(tmp_path))
        )
    )
    try:
        await AgentRuntimeFactory().resume_root(source, saved_session_id)
        await source.wait_until_ready()
        assert source.session_id == saved_session_id

        # The resume notice reports the diff against the persisted
        # bundle's bind-time snapshot (coder -> researcher), unchanged...
        queued: list[object] = []
        while not source._out_of_band_events.empty():
            queued.append(source._out_of_band_events.get_nowait())
        route_events = [
            event for event in queued if isinstance(event, RouteTableAppliedEvent)
        ]
        assert len(route_events) == 1
        resume_changes = route_events[0].changes
        assert [change.purpose for change in resume_changes] == ["research"]
        assert "coder" in resume_changes[0].before
        assert "researcher" in resume_changes[0].after
        # ...and the last-seen live table is the ADOPTED table.
        assert source._last_seen_purpose_routes == dict(
            source.config.purpose_routes.routes
        )

        # A reload after the resume diffs against the adopted live table
        # (researcher), not the persisted snapshot (coder).
        set_agent_config(
            source,
            build_test_vibe_config(
                orchestration={"routes": _research_route_rebound_to("reviewer")}
            ),
        )
        await source.reload_with_initial_messages()
        queued = []
        while not source._out_of_band_events.empty():
            queued.append(source._out_of_band_events.get_nowait())
        route_events = [
            event for event in queued if isinstance(event, RouteTableAppliedEvent)
        ]
        assert len(route_events) == 1
        changes = route_events[0].changes
        assert [change.purpose for change in changes] == ["research"]
        assert "researcher" in changes[0].before
        assert "reviewer" in changes[0].after
    finally:
        await source.aclose()


@pytest.mark.asyncio
async def test_construction_with_a_bound_bundle_serves_the_live_route_table() -> None:
    # ADR 0022: a loop constructed WITH a bound policy bundle (a
    # reconstructed conversation) serves the task-tool description from the
    # LIVE config's route table — the same source its admission resolves
    # against — never the bundle's frozen snapshot, which stays the diff
    # anchor.
    source = build_test_agent_loop(
        config=build_test_vibe_config(
            orchestration={"routes": _research_route_rebound_to_coder()}
        )
    )
    try:
        bundle = source._bound_policy
        assert bundle is not None
        live_row = _launch_target_row(_served_task_description(source), "research")
        assert "`coder`" in live_row
    finally:
        await source.aclose()

    loop = build_test_agent_loop(config=build_test_vibe_config(), bound_policy=bundle)
    try:
        assert loop._bound_policy is bundle
        # The live config carries the SHIPPED research route...
        assert (
            loop.config.purpose_routes.routes["research"].targets[0].identity
            == "researcher"
        )
        # ...and the served description follows the LIVE table, matching
        # admission — the bundle's snapshot is only the diff anchor.
        served_row = _launch_target_row(_served_task_description(loop), "research")
        assert "`researcher`" in served_row
        # Control: a fresh construction (no bound bundle) IS the current
        # selections — it serves the live config's table.
        fresh = build_test_agent_loop(config=build_test_vibe_config())
        try:
            assert "`researcher`" in _launch_target_row(
                _served_task_description(fresh), "research"
            )
        finally:
            await fresh.aclose()
    finally:
        await loop.aclose()


@pytest.mark.asyncio
async def test_retask_serves_the_live_route_table_in_the_task_tool_description() -> (
    None
):
    # ADR 0022: the retained-retask rebuild of the ToolManager (triggered
    # by a tool-override change) serves the retask target config's LIVE
    # route table — the same source admission resolves against once the
    # retask's orchestrator becomes the loop's live config.
    loop = build_test_agent_loop(
        config=build_test_vibe_config(), parent_authority_revision_getter=lambda: 0
    )
    registry, root = _admission_registry(loop)
    try:
        initial_row = _launch_target_row(_served_task_description(loop), "research")
        assert "`researcher`" in initial_row
        assert loop.committed_model is not None

        # The retask target config — built from the parent's current config
        # — carries the edited route.
        edited = build_test_vibe_config(
            orchestration={"routes": _research_route_rebound_to_coder()}
        )
        set_agent_config(loop, edited)
        target = loop.config_orchestrator.copy(config=edited)
        candidate = LaunchCandidate(
            identity="coder",
            config_inputs={},
            orchestrator=target,
            # The tool-override change forces the ToolManager rebuild whose
            # description this test pins.
            semantic_overrides=LaunchConfig(thinking="high", disabled_tools=["todo"]),
            persona=FrozenPersona(None),
            effective_model=target.config.get_active_model(),
            committed_model=loop.committed_model,
            effective_thinking=target.config.get_active_model().thinking,
            authority_inputs=LaunchAuthorityInputs(frozenset()),
        )
        prepared = await loop.prepare_launch_reconfiguration(
            candidate,
            expected_session_generation=loop._session_generation,
            expected_parent_authority_revision=0,
        )
        loop.publish_launch_reconfiguration(prepared)

        # The retask's live config carries the edit, so admission resolves
        # the EDITED route...
        after = registry._launch_admission.resolve_candidate(
            root, TaskArgs(task="inspect", purpose="research")
        )
        assert after.identity == "coder"
        # ...and the rebuilt manager advertises the SAME edited route.
        served_row = _launch_target_row(_served_task_description(loop), "research")
        assert "`coder`" in served_row
        assert served_row != initial_row
    finally:
        await loop.aclose()


@pytest.mark.asyncio
async def test_policy_replacement_serves_the_live_route_table(tmp_path: Path) -> None:
    # ADR 0022: the policy-replacement rebuild of the ToolManager serves
    # the conversation's LIVE route table — the same source admission
    # resolves against — never the staged config's routes diverging from
    # the live ones. The policy-replacement staging needs the real
    # layer-backed orchestrator (the fake lacks the mutation lock), so a
    # first loop binds the SHIPPED bundle (the conversation's frozen policy
    # record) and the test conversation is reconstructed around it while
    # the saved config carries the edited route.
    shipped_settings = tmp_path / "shipped-settings.toml"
    shipped_settings.write_text(
        '[tools.read_file]\ndenylist = ["A"]\n', encoding="utf-8"
    )
    shipped_orchestrator = await make_orchestrator(shipped_settings)
    await shipped_orchestrator.set_field("/tools/read_file/denylist", ["other"])
    source = AgentLoop(
        config_orchestrator=shipped_orchestrator, cwd=tmp_path, backend=FakeBackend()
    )
    try:
        bundle = source._bound_policy
        assert bundle is not None
        assert bundle.manifest.route_snapshot
    finally:
        await source.aclose()

    settings = tmp_path / "settings.toml"
    settings.write_text(
        '[tools.read_file]\ndenylist = ["A"]\n\n'
        "[orchestration.routes.research]\n"
        'targets = [{ model = "$session", identity = "coder", '
        'tier = "substantive" }]\n',
        encoding="utf-8",
    )
    orchestrator = await make_orchestrator(settings)
    await orchestrator.set_field("/tools/read_file/denylist", ["other"])
    loop = AgentLoop(
        config_orchestrator=orchestrator,
        cwd=tmp_path,
        backend=FakeBackend(),
        bound_policy=bundle,
    )
    registry, root = _admission_registry(loop)
    try:
        # The live config carries the edited route (ADR 0022: the edit
        # applies to this conversation's subsequent launches)...
        assert (
            loop.config.purpose_routes.routes["research"].targets[0].identity == "coder"
        )
        # ...and the served description advertises it, matching admission.
        live_row = _launch_target_row(_served_task_description(loop), "research")
        assert "`coder`" in live_row

        prepared = await loop._prepare_policy_replacement(
            source="user-toml",
            tools={"read_file": {"denylist": ["B"]}},
            expected_token=loop.config_orchestrator.accepted_token,
        )
        loop._commit_policy_replacement(prepared)

        # Admission launches the LIVE (edited) route...
        after = registry._launch_admission.resolve_candidate(
            root, TaskArgs(task="inspect", purpose="research")
        )
        assert after.identity == "coder"
        # ...and the rebuilt manager advertises the SAME route.
        served_row = _launch_target_row(_served_task_description(loop), "research")
        assert served_row == live_row
    finally:
        await loop.aclose()


@pytest.mark.asyncio
async def test_clear_rebinds_the_task_tool_description_to_the_new_conversation(
    tmp_path: Path,
) -> None:
    # DF1 sweep: /clear binds a FRESH bundle from the current selections,
    # so the rebuilt task-tool description follows the NEW conversation's
    # route table (the edited route) — the same table its admission
    # resolves against.
    config = build_test_vibe_config(
        session_logging=SessionLoggingConfig(enabled=True, save_dir=str(tmp_path))
    )
    loop = build_test_agent_loop(config=config)
    registry, root = _admission_registry(loop)
    try:
        frozen_row = _launch_target_row(_served_task_description(loop), "research")
        assert "`researcher`" in frozen_row

        set_agent_config(
            loop,
            build_test_vibe_config(
                orchestration={"routes": _research_route_rebound_to_coder()},
                session_logging=SessionLoggingConfig(
                    enabled=True, save_dir=str(tmp_path)
                ),
            ),
        )
        await loop.clear_history()

        # The NEW conversation launches the edited route...
        candidate = registry._launch_admission.resolve_candidate(
            root, TaskArgs(task="inspect", purpose="research")
        )
        assert candidate.identity == "coder"
        # ...and its model-facing description advertises the SAME route.
        served_row = _launch_target_row(_served_task_description(loop), "research")
        assert "`coder`" in served_row
    finally:
        await loop.aclose()


@pytest.mark.asyncio
async def test_subagent_rewind_serves_the_live_route_table(tmp_path: Path) -> None:
    # ADR 0022 (rewind half): a subagent rewind keeps the CHILD's own bound
    # bundle for the frozen identity content, but the served route table is
    # live: the rewound conversation's admission resolves the child's live
    # (edited) route, and its model-facing description advertises the SAME
    # route. (A root clear binds a FRESH bundle from the current
    # selections, so its live config and fresh bundle agree by
    # construction; the rewind is the divergent reset path.)
    config = build_test_vibe_config(
        session_logging=SessionLoggingConfig(enabled=True, save_dir=str(tmp_path))
    )
    parent = build_test_agent_loop(config=config)
    loop = build_test_agent_loop(
        config=config,
        is_subagent=True,
        parent_session_id=parent.session_id,
        inherited_workspace=parent.tool_manager.workspace,
    )
    registry, root = _admission_registry(loop)
    try:
        assert loop.is_subagent
        await parent.aclose()
        initial_row = _launch_target_row(_served_task_description(loop), "research")
        assert "`researcher`" in initial_row

        # The child reloads a saved route edit: its live config carries the
        # edit, so the edit applies to the child's subsequent launches.
        set_agent_config(
            loop,
            build_test_vibe_config(
                orchestration={"routes": _research_route_rebound_to_coder()},
                session_logging=SessionLoggingConfig(
                    enabled=True, save_dir=str(tmp_path)
                ),
            ),
        )
        await loop.reload_with_initial_messages()
        assert (
            loop.config.purpose_routes.routes["research"].targets[0].identity == "coder"
        )

        # Rewind (keep the parent): the child's OWN bundle is kept for the
        # frozen identity content, but the served route table is live.
        await loop._reset_session()

        # The rewound conversation's admission resolves the child's LIVE
        # (edited) route...
        candidate = registry._launch_admission.resolve_candidate(
            root, TaskArgs(task="inspect", purpose="research")
        )
        assert candidate.identity == "coder"
        # ...and its model-facing description advertises the SAME route.
        served_row = _launch_target_row(_served_task_description(loop), "research")
        assert "`coder`" in served_row
        assert served_row != initial_row
    finally:
        await loop.aclose()


class _StaticProvider:
    """An MCP authorization provider returning one canned result."""

    def __init__(self, result: MCPAuthorizationResult) -> None:
        self._result = result

    async def resolve(self, reference: MCPAuthorizationRef) -> MCPAuthorizationResult:
        return self._result

    async def reject(
        self,
        reference: MCPAuthorizationRef,
        *,
        observed_connection_revision: str,
        reason: Literal["http_unauthorized", "mcp_unauthorized"],
    ) -> MCPAuthorizationResult:
        return self._result


def _review_deep_single_model_routes() -> dict[str, PurposeRoute]:
    """review.deep with all three seats on ONE explicit model (invalid).

    The route declares ``requires_distinct_models = 3`` but binds every
    seat to the same literal model, so the diversity requirement is
    statically unmet — exactly the edited table the reload must reject.
    """
    routes = dict(shipped_routes())
    routes["review.deep"] = PurposeRoute(
        targets=(
            RouteTarget(
                model="glm-5-3",
                identity="reviewer",
                tier=PackageTier.SUBSTANTIVE,
                seat="analytical",
            ),
            RouteTarget(
                model="glm-5-3",
                identity="reviewer",
                tier=PackageTier.SUBSTANTIVE,
                seat="peer",
            ),
            RouteTarget(
                model="glm-5-3",
                identity="coder",
                tier=PackageTier.SUBSTANTIVE,
                seat="execution",
            ),
        ),
        requires_distinct_models=3,
    )
    return routes


@pytest.mark.asyncio
async def test_invalid_edited_route_table_rejects_the_reload_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # ADR 0022 check 6: an invalid edited table rejects the WHOLE reload,
    # fail-closed — requires_distinct_models is validated at reload, not
    # only at launch. The prior table stays effective, no applied notice is
    # published, and in-flight children are untouched.
    backends: list[BlockingBackend] = []

    def create_backend(*, provider, **_kwargs):
        backend = BlockingBackend()
        backends.append(backend)
        return backend

    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", create_backend
    )
    parent = build_test_agent_loop(
        config=build_test_vibe_config(
            enabled_tools=["task"], tools={"task": {"permission": "always"}}
        ),
        backend=FakeBackend(),
    )
    registry = _retention_registry()
    root = registry._build_child_runtime(parent)
    root.turns._projector = MagicMock()
    root.turns.link_subagent = AsyncMock()
    registry.bind_root(root)
    context = InvokeContext(tool_call_id="fail-closed", session_id=parent.session_id)
    try:
        # An in-flight child admitted on the shipped research route.
        first = await _background_result(
            registry,
            TaskArgs(task="first", purpose="research", background=True, retain=True),
            context,
        )
        assert first.agent_id is not None
        record = registry._agent_records[first.agent_id]
        assert record.identity == "researcher"
        t1_model = record.runtime.agent_loop.committed_model
        assert t1_model is not None

        # The edited table violates review.deep's declared diversity: all
        # three seats bind one explicit model while the route requires 3
        # distinct canonical models. The reload candidate resolves to this
        # table, and the reload-time validation rejects it.
        invalid = build_test_vibe_config(
            enabled_tools=["task"],
            tools={"task": {"permission": "always"}},
            orchestration={"routes": _review_deep_single_model_routes()},
        )
        stub_config_reload(monkeypatch, invalid)
        with pytest.raises(RouteResolutionError, match="distinct canonical models"):
            await parent.reload_with_initial_messages(reload_config=True)

        # The reload rejected the whole table: the prior table stays
        # effective...
        assert (
            parent.live_purpose_routes["research"].targets[0].identity == "researcher"
        )
        candidate = registry._launch_admission.resolve_candidate(
            root, TaskArgs(task="inspect", purpose="research")
        )
        assert candidate.identity == "researcher"
        # ...the served description still advertises the prior table...
        served_row = _launch_target_row(_served_task_description(parent), "research")
        assert "`researcher`" in served_row
        # ...no applied notice was published...
        assert parent.applied_route_changes == ()
        assert parent._out_of_band_events.empty()
        # ...and the in-flight child is untouched.
        assert registry._agent_records[first.agent_id] is record
        assert record.identity == "researcher"
        assert record.runtime.agent_loop.committed_model is t1_model
        assert record.runtime.agent_loop._bound_identity == "researcher"
    finally:
        for backend in backends:
            backend.release.set()
        await registry.drain_children()
        await parent.aclose()
    """An MCP authorization provider returning one canned result."""

    def __init__(self, result: MCPAuthorizationResult) -> None:
        self._result = result

    async def resolve(self, reference: MCPAuthorizationRef) -> MCPAuthorizationResult:
        return self._result

    async def reject(
        self,
        reference: MCPAuthorizationRef,
        *,
        observed_connection_revision: str,
        reason: Literal["http_unauthorized", "mcp_unauthorized"],
    ) -> MCPAuthorizationResult:
        return self._result


def _resolved_server(*, disabled: bool = False) -> ResolvedMCPServerConfig:
    return ResolvedMCPServerConfig(
        name="search",
        transport="stdio",
        url=None,
        command="fake-cmd",
        args=(),
        cwd=None,
        env={},
        authorization=MCPAuthorizationRef(
            server_name="search",
            server_fingerprint="fp",
            kind="none",
            descriptor_revision="rev-1",
        ),
        prompt=None,
        startup_timeout_s=30.0,
        tool_timeout_s=60.0,
        disabled=disabled,
        disabled_tools=frozenset(),
    )


@pytest.mark.asyncio
async def test_all_five_mcp_refresh_paths_serve_the_frozen_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Every MCP config-refresh path in the loop re-renders the system prompt
    # through refresh_system_prompt — each must serve the captured bundle,
    # never a live reread. The five paths: reconfigure_mcp convergence,
    # suspend_mcp, and the three authorization_changed branches (disabled,
    # still-required, resolved).
    config = build_test_vibe_config(
        mcp_servers=[MCPStdio(name="search", transport="stdio", command="fake-cmd")]
    )
    loop = build_test_agent_loop(config=config)
    client_transport, server_transport = memory_transport_pair()
    server = build_test_app_server(loop, server_transport)
    client = AppServerClient(client_transport, run_peer=server.serve)
    session = await attach_test_app_server_session(client)
    backend = legacy_backend(server)
    try:
        bound = loop._bound_policy
        assert bound is not None
        frozen_prompt = loop.messages[0].content

        refresh_calls = 0
        original_refresh = loop.refresh_system_prompt

        async def counting_refresh() -> None:
            nonlocal refresh_calls
            refresh_calls += 1
            await original_refresh()

        monkeypatch.setattr(loop, "refresh_system_prompt", counting_refresh)
        loop.tool_manager.reconfigure_mcp_async = AsyncMock()

        # Path 1: reconfigure_mcp convergence.
        await backend.reconfigure_mcp(
            ResolvedMCPCatalog(revision="r1", servers=(_resolved_server(),)),
            force_remote_discovery=False,
        )
        assert refresh_calls == 1

        # Path 2: suspend_mcp.
        await backend.suspend_mcp(name="search", tool_name=None, reason="disable")
        assert refresh_calls == 2

        # Path 3: authorization_changed on a disabled server.
        backend.mcp_catalog = ResolvedMCPCatalog(
            revision="r2", servers=(_resolved_server(disabled=True),)
        )
        backend.mcp_authorization_provider = _StaticProvider(
            MCPAuthorizationRequired(reason="missing", descriptor_revision="rev-1")
        )
        await backend.authorization_changed(name="search", descriptor_revision="rev-1")
        assert refresh_calls == 3

        # Path 4: authorization_changed with authorization still required.
        backend.mcp_catalog = ResolvedMCPCatalog(
            revision="r3", servers=(_resolved_server(),)
        )
        backend.mcp_authorization_provider = _StaticProvider(
            MCPAuthorizationRequired(reason="missing", descriptor_revision="rev-1")
        )
        await backend.authorization_changed(name="search", descriptor_revision="rev-1")
        assert refresh_calls == 4

        # Path 5: authorization_changed with authorization resolved.
        backend.mcp_catalog = ResolvedMCPCatalog(
            revision="r4", servers=(_resolved_server(),)
        )
        backend.mcp_authorization_provider = _StaticProvider(
            MCPAuthorizationSnapshot(
                headers={}, connection_revision="c1", descriptor_revision="rev-1"
            )
        )
        await backend.authorization_changed(name="search", descriptor_revision="rev-1")
        assert refresh_calls == 5

        # Every path served the frozen bundle: same bound policy, same prompt.
        assert loop._bound_policy is bound
        assert loop.messages[0].content == frozen_prompt
    finally:
        await session.close()
        await loop.aclose()


@pytest.mark.asyncio
async def test_retask_keeps_the_bound_identity_revision() -> None:
    # A retask reconfigures model/thinking/tools but cannot replace the
    # persona: the bound identity revision is the conversation's, not the
    # retask's.
    loop = build_test_agent_loop(
        config=build_test_vibe_config(), parent_authority_revision_getter=lambda: 0
    )
    try:
        bound = loop._bound_policy
        assert bound is not None
        binding = loop._identity_binding()
        frozen_prompt = loop.messages[0].content
        assert loop.committed_model is not None

        target = loop.config_orchestrator.copy()
        candidate = LaunchCandidate(
            identity="coder",
            config_inputs={},
            orchestrator=target,
            semantic_overrides=LaunchConfig(thinking="high"),
            persona=FrozenPersona(None),
            effective_model=target.config.get_active_model(),
            committed_model=loop.committed_model,
            effective_thinking=target.config.get_active_model().thinking,
            authority_inputs=LaunchAuthorityInputs(frozenset()),
        )
        prepared = await loop.prepare_launch_reconfiguration(
            candidate,
            expected_session_generation=loop._session_generation,
            expected_parent_authority_revision=0,
        )
        # The retask renders the frozen bundle: the persona cannot be replaced.
        assert prepared.consumers.system_prompt == frozen_prompt
        loop.publish_launch_reconfiguration(prepared)

        assert loop._bound_policy is bound
        assert loop._identity_binding() == binding
        assert loop._bound_identity == "coder"
        assert loop.launch_overrides is not None
        assert loop.launch_overrides.thinking == "high"
        assert loop.messages[0].content == frozen_prompt
    finally:
        await loop.aclose()


@pytest.mark.asyncio
async def test_in_process_resume_serves_persisted_content_after_source_deletion(
    config_dir: Path, tmp_path: Path
) -> None:
    # Resume preserves the bound bundle: the persisted record carries the
    # identity's actual content, so the conversation resumes even when the
    # resource file is gone.
    prompt_file = config_dir / "prompts" / "lifecycle-resume.md"
    prompt_file.parent.mkdir(parents=True, exist_ok=True)
    prompt_file.write_text(
        "# Resume probe\n\nYou are the resume probe persona.\n", encoding="utf-8"
    )
    config = build_test_vibe_config(
        agent={"identity": "user:lifecycle-resume"},
        session_logging=SessionLoggingConfig(enabled=True, save_dir=str(tmp_path)),
    )
    saved = build_test_agent_loop(config=config)
    original_prompt = saved.messages[0].content
    assert "resume probe persona" in (original_prompt or "")
    await saved.persist_empty_session()
    saved_session_id = saved.session_id
    await saved.aclose()

    # The source loop's own NEW binding validates while the resource exists.
    source = build_test_agent_loop(config=config)
    try:
        prompt_file.unlink()
        await AgentRuntimeFactory().resume_root(source, saved_session_id)
        await source.wait_until_ready()
        assert source.session_id == saved_session_id
        assert source._bound_identity == "user:lifecycle-resume"
        assert source.messages[0].content == original_prompt
        # The frozen policy is never reread from disk: a refresh after the
        # deletion still serves the persisted content.
        await source.refresh_system_prompt()
        assert source.messages[0].content == original_prompt
    finally:
        await source.aclose()


@pytest.mark.asyncio
async def test_fresh_runtime_resume_serves_persisted_content_after_source_deletion(
    config_dir: Path, tmp_path: Path
) -> None:
    # The fresh-runtime resume constructs the loop AT RESUME TIME around the
    # persisted bundle: the deleted resource must not fail construction —
    # identity validation applies to NEW bindings only.
    prompt_file = config_dir / "prompts" / "lifecycle-blueprint.md"
    prompt_file.parent.mkdir(parents=True, exist_ok=True)
    prompt_file.write_text(
        "# Blueprint probe\n\nYou are the blueprint probe persona.\n", encoding="utf-8"
    )
    config = build_test_vibe_config(
        agent={"identity": "user:lifecycle-blueprint"},
        session_logging=SessionLoggingConfig(enabled=True, save_dir=str(tmp_path)),
    )
    saved = build_test_agent_loop(config=config)
    original_prompt = saved.messages[0].content
    assert "blueprint probe persona" in (original_prompt or "")
    await saved.persist_empty_session()
    saved_session_id = saved.session_id
    await saved.aclose()

    prompt_file.unlink()
    blueprint = _RootRuntimeBlueprint(
        config_orchestrator=FakeConfigOrchestrator(config),
        harness_files=HarnessFilesManager(),
        options=SessionOptions(cwd=str(tmp_path)),
        client_info=ClientInfo(name="lifecycle", version="0"),
        client_capabilities=ClientCapabilities(),
        hook_config_result=HookConfigResult(hooks=[], issues=[]),
        cache_store=FileSystemCacheStore(tmp_path / "cache.toml"),
    )
    replacement = await AgentRuntimeFactory().resume_blueprint(
        blueprint, saved_session_id
    )
    try:
        await replacement.wait_until_ready()
        assert replacement.session_id == saved_session_id
        assert replacement._bound_identity == "user:lifecycle-blueprint"
        assert replacement.messages[0].content == original_prompt
    finally:
        await replacement.aclose()


@pytest.mark.asyncio
async def test_in_place_resume_rebinds_effective_tool_authority(tmp_path: Path) -> None:
    # Round-2 Ab1 (resume half): an in-place resume rebinds the loop's
    # identity from the persisted bundle AND the ToolManager's bound
    # identity in the SAME transaction — resuming a reviewer conversation
    # into a coder-bound loop must not keep coder authority (coder-only
    # tools stay unavailable personally AND undelegable).
    config = build_test_vibe_config(
        agent={"identity": "reviewer"},
        session_logging=SessionLoggingConfig(enabled=True, save_dir=str(tmp_path)),
    )
    saved = build_test_agent_loop(config=config)
    await saved.persist_empty_session()
    saved_session_id = saved.session_id
    await saved.aclose()

    # The source loop binds coder (the current selection).
    source = build_test_agent_loop(
        config=build_test_vibe_config(
            session_logging=SessionLoggingConfig(enabled=True, save_dir=str(tmp_path))
        )
    )
    try:
        assert source._bound_identity == "coder"
        # Warm every authority surface under the coder binding first.
        assert "edit" in source.tool_manager.available_tools
        assert "edit" in source.tool_manager.delegable_tools
        assert "edit" in {
            spec.name for spec in source.tool_manager.available_tool_specs()
        }
        source.tool_manager.get("edit")

        await AgentRuntimeFactory().resume_root(source, saved_session_id)
        await source.wait_until_ready()

        assert source.session_id == saved_session_id
        assert source._bound_identity == "reviewer"
        assert source.tool_manager._bound_identity == "reviewer"
        # Effective authority follows the rebind: coder-only (mutating)
        # tools are unavailable personally AND undelegable, on warm caches
        # included; the read-only set survives.
        assert "edit" not in source.tool_manager.available_tools
        assert "edit" not in source.tool_manager.delegable_tools
        assert "edit" not in {
            spec.name for spec in source.tool_manager.available_tool_specs()
        }
        with pytest.raises(NoSuchToolError):
            source.tool_manager.get("edit")
        assert "read_file" in source.tool_manager.available_tools
    finally:
        await source.aclose()


@pytest.mark.asyncio
async def test_clear_binds_a_fresh_bundle_from_current_selections(
    tmp_path: Path,
) -> None:
    # /clear binds FRESH resources: a new bundle from the current selections,
    # a new session — never a continuation of the old conversation's policy.
    config = build_test_vibe_config(
        session_logging=SessionLoggingConfig(enabled=True, save_dir=str(tmp_path))
    )
    loop = build_test_agent_loop(config=config)
    try:
        first_bundle = loop._bound_policy
        assert first_bundle is not None
        first_session_id = loop.session_id
        old_prompt = loop.messages[0].content
        assert "You are the coder" in (old_prompt or "")

        loop.config.agent.identity = "reviewer"
        await loop.clear_history()

        assert loop.session_id != first_session_id
        assert loop._bound_policy is not first_bundle
        assert loop._bound_policy is not None
        assert loop._bound_policy.manifest.identity.selected == "reviewer"
        assert loop._bound_identity == "reviewer"
        new_prompt = loop.messages[0].content or ""
        assert "You are the reviewer" in new_prompt
        assert "You are the coder" not in new_prompt
    finally:
        await loop.aclose()


@pytest.mark.asyncio
async def test_clear_prepare_failure_preserves_the_old_generation(
    tmp_path: Path,
) -> None:
    # A prepare failure/cancellation preserves the OLD generation: the fresh
    # /clear bind validates the NEW selection, and its failure leaves the
    # loop bound to the original session — no mixed generation, no
    # half-bound state.
    config = build_test_vibe_config(
        session_logging=SessionLoggingConfig(enabled=True, save_dir=str(tmp_path))
    )
    loop = build_test_agent_loop(config=config)
    try:
        old_bundle = loop._bound_policy
        assert old_bundle is not None
        old_session_id = loop.session_id
        old_prompt = loop.messages[0].content

        loop.config.agent.identity = "user:missing"
        with pytest.raises(UnknownIdentityError):
            await loop.clear_history()

        assert loop.session_id == old_session_id
        assert loop._bound_policy is old_bundle
        assert loop.messages[0].content == old_prompt
    finally:
        await loop.aclose()


@pytest.mark.asyncio
async def test_clear_rebinds_effective_tool_authority(tmp_path: Path) -> None:
    # Round-2 Ab1 (clear half): /clear rebinds the loop's identity AND the
    # ToolManager's bound identity in the SAME transaction — a coder
    # cleared into reviewer loses coder-only tools personally AND as a
    # delegation ceiling (no authority widening relative to the new
    # identity, on warm caches included).
    config = build_test_vibe_config(
        session_logging=SessionLoggingConfig(enabled=True, save_dir=str(tmp_path))
    )
    loop = build_test_agent_loop(config=config)
    try:
        assert loop._bound_identity == "coder"
        # Warm every authority surface under the coder binding first.
        assert "edit" in loop.tool_manager.available_tools
        assert "edit" in loop.tool_manager.delegable_tools
        assert "edit" in {
            spec.name for spec in loop.tool_manager.available_tool_specs()
        }
        loop.tool_manager.get("edit")

        loop.config.agent.identity = "reviewer"
        await loop.clear_history()

        assert loop._bound_identity == "reviewer"
        assert loop.tool_manager._bound_identity == "reviewer"
        # Effective authority follows the rebind: coder-only (mutating)
        # tools are unavailable personally AND undelegable; the read-only
        # set survives.
        assert "edit" not in loop.tool_manager.available_tools
        assert "edit" not in loop.tool_manager.delegable_tools
        assert "edit" not in {
            spec.name for spec in loop.tool_manager.available_tool_specs()
        }
        with pytest.raises(NoSuchToolError):
            loop.tool_manager.get("edit")
        assert "read_file" in loop.tool_manager.available_tools
    finally:
        await loop.aclose()


@pytest.mark.asyncio
async def test_construction_with_a_bound_bundle_reads_the_bundle_not_the_pending_config() -> (
    None
):
    # Round-2 delta D1 (Ab6 residual): the construction-time SkillManager
    # snapshotted the LIVE config, so a loop constructed WITH a bound
    # policy bundle while the saved config carried a pending selection
    # reported the pending selection's eligibility. Eligibility must
    # derive from the bound bundle's manifest selections — the same frozen
    # source the reload path reads — never the pending saved config (a
    # pending selection applies to new conversations only).
    source = build_test_agent_loop()
    try:
        bundle = source._bound_policy
        assert bundle is not None
        assert bundle.manifest.workflow.selected == "minimal"
    finally:
        await source.aclose()

    loop = build_test_agent_loop(
        config=build_test_vibe_config(agent={"workflow": "ceremony"}),
        bound_policy=bundle,
    )
    try:
        assert loop._bound_policy is bundle
        assert loop.skill_manager.policy_facts.workflow == "minimal"
        assert "main-review" not in loop.skill_manager.available_skills
        assert not loop.skill_manager.skill_eligibility("main-review").eligible
        # Filtering is not pruning: the catalog still lists the skill.
        assert "main-review" in loop.skill_manager.settings_inventory
        prompt = loop.messages[0].content or ""
        assert "Workflow: minimal" in prompt
        assert "Opening gate" not in prompt

        # Control: a FRESH construction (no bound bundle) IS the current
        # selections — the ceremony config reads as ceremony eligibility.
        fresh = build_test_agent_loop(
            config=build_test_vibe_config(agent={"workflow": "ceremony"})
        )
        try:
            assert fresh._bound_policy is not bundle
            assert fresh.skill_manager.policy_facts.workflow == "ceremony"
            assert fresh.skill_manager.skill_eligibility("main-review").eligible
        finally:
            await fresh.aclose()
    finally:
        await loop.aclose()


@pytest.mark.asyncio
async def test_clear_rebinds_skill_eligibility_to_the_new_conversation(
    tmp_path: Path,
) -> None:
    # Round-2 delta D1 (Ab6 residual): /clear retained the previous
    # conversation's SkillManager, so clearing a ceremony conversation into
    # minimal left the manager on ceremony eligibility. The reset must
    # rebuild the manager on the NEW conversation's frozen selections, in
    # the same transaction as the identity rebinding.
    config = build_test_vibe_config(
        agent={"workflow": "ceremony"},
        session_logging=SessionLoggingConfig(enabled=True, save_dir=str(tmp_path)),
    )
    loop = build_test_agent_loop(config=config)
    try:
        assert loop.skill_manager.policy_facts.workflow == "ceremony"
        assert loop.skill_manager.skill_eligibility("main-review").eligible
        assert "main-review" in loop.skill_manager.available_skills

        loop.config.agent.workflow = "minimal"
        await loop.clear_history()

        # The new conversation is minimal: the ceremony-bearing skill is
        # withheld (filtering is not pruning — the catalog still lists it,
        # and the diagnostic names the frozen workflow).
        assert loop._bound_policy is not None
        assert loop._bound_policy.manifest.workflow.selected == "minimal"
        assert loop.skill_manager.policy_facts.workflow == "minimal"
        assert "main-review" not in loop.skill_manager.available_skills
        assert not loop.skill_manager.skill_eligibility("main-review").eligible
        assert "main-review" in loop.skill_manager.settings_inventory
        diagnostic = loop.skill_manager.incompatibility_diagnostic("main-review")
        assert diagnostic is not None and "minimal" in diagnostic
        prompt = loop.messages[0].content or ""
        assert "Workflow: minimal" in prompt
        assert "Opening gate" not in prompt

        # The reverse direction pins the same contract: clearing minimal
        # into ceremony makes the ceremony-bearing skill eligible.
        loop.config.agent.workflow = "ceremony"
        await loop.clear_history()

        assert loop._bound_policy is not None
        assert loop._bound_policy.manifest.workflow.selected == "ceremony"
        assert loop.skill_manager.policy_facts.workflow == "ceremony"
        assert loop.skill_manager.skill_eligibility("main-review").eligible
        assert "main-review" in loop.skill_manager.available_skills
        prompt = loop.messages[0].content or ""
        assert "Opening gate" in prompt
    finally:
        await loop.aclose()


@pytest.mark.asyncio
async def test_in_place_resume_rebinds_skill_eligibility_to_the_resumed_bundle(
    tmp_path: Path,
) -> None:
    # Round-2 delta D1 (the same Ab6 residual on the resume half of the Ab1
    # transaction): an in-place resume rebinds the loop onto the persisted
    # bundle; the SkillManager must rebind with it, so eligibility reads the
    # RESUMED conversation's frozen selections, never the source loop's.
    saved = build_test_agent_loop(
        config=build_test_vibe_config(
            agent={"workflow": "ceremony"},
            session_logging=SessionLoggingConfig(enabled=True, save_dir=str(tmp_path)),
        )
    )
    await saved.persist_empty_session()
    saved_session_id = saved.session_id
    await saved.aclose()

    source = build_test_agent_loop(
        config=build_test_vibe_config(
            session_logging=SessionLoggingConfig(enabled=True, save_dir=str(tmp_path))
        )
    )
    try:
        assert source.skill_manager.policy_facts.workflow == "minimal"
        assert not source.skill_manager.skill_eligibility("main-review").eligible

        await AgentRuntimeFactory().resume_root(source, saved_session_id)
        await source.wait_until_ready()

        assert source.session_id == saved_session_id
        assert source._bound_policy is not None
        assert source._bound_policy.manifest.workflow.selected == "ceremony"
        assert source.skill_manager.policy_facts.workflow == "ceremony"
        assert source.skill_manager.skill_eligibility("main-review").eligible
        assert "main-review" in source.skill_manager.available_skills
        prompt = source.messages[0].content or ""
        assert "Opening gate" in prompt
    finally:
        await source.aclose()


@pytest.mark.asyncio
async def test_clear_at_bundle_store_capacity_fails_before_any_state_change(
    tmp_path: Path,
) -> None:
    # Round-2 Ab7: the bundle store's capacity is settled during
    # PREPARATION — at capacity the reset fails BEFORE the session id,
    # generation, or resumability state changes, and the prior generation
    # remains fully resumable (design §6: no mixed generation).
    config = build_test_vibe_config(
        session_logging=SessionLoggingConfig(enabled=True, save_dir=str(tmp_path))
    )
    loop = build_test_agent_loop(config=config)
    try:
        store = loop._bundle_store
        bundle = store.get(loop.session_id)
        assert bundle is not None
        # Fill the store to its shipped capacity (64 entries) with
        # resumable conversations; the startup bundle is entry one.
        while len(store) < 64:
            filler = f"filler-{len(store):03d}"
            store.put(filler, bundle)
            store.set_resumable(filler, True)
        assert len(store) == 64

        old_session_id = loop.session_id
        old_generation = loop._session_generation
        old_bundle = loop._bound_policy
        assert old_bundle is not None
        old_prompt = loop.messages[0].content
        assert old_session_id in store._resumable

        with pytest.raises(BundleStoreError, match="bundle store is full"):
            await loop.clear_history()

        # The reset failed before any durable state change: session id,
        # generation, binding, and prompt are untouched...
        assert loop.session_id == old_session_id
        assert loop._session_generation == old_generation
        assert loop._bound_policy is old_bundle
        assert loop._bound_identity == "coder"
        assert loop.messages[0].content == old_prompt
        # ...and the prior generation remains fully resumable.
        assert store.get(old_session_id) is not None
        assert old_session_id in store._resumable
        assert store.activate(old_session_id).policy_text == old_bundle.policy_text
    finally:
        await loop.aclose()


@pytest.mark.asyncio
async def test_legacy_record_rejects_actionably(tmp_path: Path) -> None:
    # LEGACY resume: a record persisted by the old format carries no frozen
    # policy bundle and rejects with an actionable "start a new session" —
    # there is no migration.
    config = build_test_vibe_config(
        session_logging=SessionLoggingConfig(enabled=True, save_dir=str(tmp_path))
    )
    saved = build_test_agent_loop(config=config)
    await saved.persist_empty_session()
    saved_session_id = saved.session_id
    session_dir = saved.session_logger.session_dir
    assert session_dir is not None
    await saved.aclose()

    meta_path = session_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.pop("policy_bundle", None)
    meta.pop("policy_storage_version", None)
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    metadata = SessionLoader.load_metadata(session_dir)
    with pytest.raises(SessionPolicyError, match="Start a new session"):
        resume_policy_bundle(metadata)

    source = build_test_agent_loop(config=config)
    try:
        with pytest.raises(SessionPolicyError, match="Start a new session"):
            await AgentRuntimeFactory().resume_root(source, saved_session_id)
        # The failed resume leaves the source bound to its own session.
        assert source.session_id != saved_session_id
    finally:
        await source.aclose()


@pytest.mark.asyncio
async def test_launch_persona_and_policy_storage_versions_change_explicitly(
    tmp_path: Path,
) -> None:
    # The launch persona and the policy storage change VERSIONS explicitly:
    # the persisted envelope carries a V2 persona keyed on the identity
    # revision, and the record carries policy_storage_version 2.
    assert POLICY_STORAGE_VERSION == 2
    assert BUNDLE_FORMAT_VERSION == 1

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
        assert metadata.policy_storage_version == POLICY_STORAGE_VERSION
        envelope = metadata.launch_config
        assert isinstance(envelope, LaunchMetadataV2)
        persona = envelope.persona
        assert isinstance(persona, LaunchPersonaV2)
        assert persona.version == 2
        assert persona.identity == "coder"
        assert persona.identity_revision == bound.manifest.identity_revision
        assert loop._identity_binding() == (persona.identity, persona.identity_revision)
    finally:
        await loop.aclose()

    # The V1 envelope shape still parses: legacy records remain readable.
    legacy = LaunchMetadataV1.model_validate({
        "version": 1,
        "profile": "worker",
        "overrides": {},
        "persona": {"system_prompt_id": "cli", "instructions": None},
    })
    assert isinstance(legacy.persona, LaunchPersonaV1)


_CUSTOM_READ_FILE_SOURCE = '''\
from collections.abc import AsyncGenerator

from pydantic import BaseModel

from chartreux.core.tools.base import BaseTool, BaseToolConfig, BaseToolState


class ReadFileArgs(BaseModel):
    file_path: str = ""


class ReadFileResult(BaseModel):
    pass


class ReadFile(BaseTool[ReadFileArgs, ReadFileResult, BaseToolConfig, BaseToolState]):
    """Custom variant, always available, wins selection over the builtin."""

    selection_priority = 1

    @classmethod
    def is_available(cls, config=None) -> bool:
        return True

    async def run(
        self, args: ReadFileArgs, ctx: object = None
    ) -> AsyncGenerator[ReadFileResult, None]:
        yield ReadFileResult()
'''


def _corner_manager(
    identity: str,
    scratch: Path,
    tools_dir: Path,
    *,
    parent: ToolManager | None,
    custom: bool,
) -> tuple[ToolManager, Callable[[str], None]]:
    kwargs: dict = {"agent": {"identity": identity}}
    if custom:
        kwargs["tool_paths"] = [tools_dir]
    resolved = build_test_vibe_config(**kwargs)

    def flip(new_identity: str) -> None:
        resolved.agent.identity = new_identity

    manager = ToolManager(
        lambda: resolved,
        cwd=scratch,
        defer_mcp=True,
        accepted_token_getter=lambda: "accepted",
        parent_authority_getter=(lambda: parent) if parent is not None else None,
    )
    return manager, flip


def test_delegated_instance_cache_is_keyed_by_caller_class_flip_corner(
    tmp_path: Path,
) -> None:
    """S2 final warning / package 3b amendment: the delegated-instance cache
    is keyed by (tool name, caller implementation), mirroring the delegation
    selection cache.

    The flip corner: a parent's delegated policy instance for a name is built
    with the caller's class baked into its liveness closure. An unsanctioned
    in-place parent identity flip un-admits that caller's class while a
    different caller's class stays admitted. The name-keyed cache served the
    first caller's stale instance to the second (a persistent over-narrow
    NEVER); the (name, class)-keyed cache never does.
    """
    scratch = tmp_path / "corner"
    scratch.mkdir()
    tools_dir = scratch / "tools"
    tools_dir.mkdir()
    (tools_dir / "read_file.py").write_text(_CUSTOM_READ_FILE_SOURCE, encoding="utf-8")
    target = scratch / "notes.txt"
    target.write_text("original\n", encoding="utf-8")
    args = ReadFileArgs(file_path=str(target))

    # Coder parent (full delegation ceiling), builtin-only registry.
    parent, flip_parent = _corner_manager(
        "coder", scratch, tools_dir, parent=None, custom=False
    )
    parent_variant = parent._select_available_variant(
        "read_file", parent._all_tools["read_file"]
    )
    assert parent_variant is ReadFile

    # A coder child whose registry selects the custom read_file variant.
    child_custom, _ = _corner_manager(
        "coder", scratch, tools_dir, parent=parent, custom=True
    )
    custom_cls = child_custom._select_available_variant(
        "read_file", child_custom._all_tools["read_file"]
    )
    assert custom_cls is not None and custom_cls is not ReadFile

    # A coder child with only the builtin variant.
    child_builtin, _ = _corner_manager(
        "coder", scratch, tools_dir, parent=parent, custom=False
    )
    assert (
        child_builtin._select_available_variant(
            "read_file", child_builtin._all_tools["read_file"]
        )
        is ReadFile
    )

    # Under the coder parent the custom implementation is admitted, and the
    # delegated instance is cached under (name, caller class) — built from
    # the parent's own variant.
    assert child_custom._parent_permission("read_file", args, custom_cls) is None
    baked = parent._delegated_instances.get(("read_file", custom_cls))
    assert baked is not None
    assert type(baked) is ReadFile

    # The unsanctioned in-place parent flip retires the custom caller's
    # admission but not the builtin's (the parent's own variant).
    flip_parent("reviewer")
    denied = child_custom._parent_permission("read_file", args, custom_cls)
    assert denied is not None
    assert denied.permission is ToolPermission.NEVER
    assert denied.reason == "Tool is unavailable under parent authority"

    # The FIX: the builtin caller no longer reuses the custom-keyed cached
    # instance — it resolves a fresh one and is served (the old name-keyed
    # cache gave this admitted caller a persistent over-narrow NEVER).
    assert child_builtin._parent_permission("read_file", args, ReadFile) is None
    assert child_builtin._parent_permission("read_file", args, ReadFile) is None

    # The stale custom-keyed instance stays retired and never serves.
    assert not baked._authority_is_current()

    # reset_all (the loop reload path) reconciles the caches.
    parent.reset_all()
    assert ("read_file", custom_cls) not in parent._delegated_instances
    assert child_builtin._parent_permission("read_file", args, ReadFile) is None


@pytest.mark.asyncio
async def test_resume_child_uses_the_persisted_identity_binding(tmp_path: Path) -> None:
    # Follow-up 3 (peer round-3 W5): the launch envelope persists the child's
    # resolved identity reference and revision, so resume re-reads the
    # binding instead of re-deriving identity from the live configuration —
    # an identity edited between sessions must not re-posture a resumed child.
    config = build_test_vibe_config(
        session_logging=SessionLoggingConfig(enabled=True, save_dir=str(tmp_path))
    )
    parent = build_test_agent_loop(config=config)
    factory = AgentRuntimeFactory()
    child = await factory.create_child(parent, "coder")
    await child.persist_empty_session()
    # The child binds lazily; the save backstop forces the startup capture so
    # the persisted record always carries the frozen bundle.
    bound = child._bound_policy
    assert bound is not None
    child_dir = child.session_logger.session_dir
    assert child_dir is not None
    child_id = child.session_id

    envelope = SessionLoader.load_metadata(child_dir).launch_config
    assert isinstance(envelope, LaunchMetadataV2)
    persona = envelope.persona
    assert isinstance(persona, LaunchPersonaV2)
    assert persona.identity == child._bound_identity
    assert persona.identity == bound.manifest.identity.selected
    assert persona.identity_revision == child._identity_binding()[1]
    assert persona.identity_revision == bound.manifest.identity_revision
    await child.aclose()

    # The parent identity is edited between sessions; the resumed child keeps
    # the persisted binding.
    parent.config.agent.identity = "reviewer"
    try:
        resumed = await factory.resume_child(parent, "coder", child_id, child_dir)
        try:
            await resumed.wait_until_ready()
            assert resumed._bound_identity == persona.identity
            assert resumed._bound_policy is not None
            assert resumed._bound_policy.manifest.identity.selected == persona.identity
        finally:
            await resumed.aclose()
    finally:
        await parent.aclose()


@pytest.mark.asyncio
async def test_resume_child_rejects_persona_bundle_mismatch(tmp_path: Path) -> None:
    # Follow-up 3, the rejection half: a stored persona that disagrees with
    # the persisted policy bundle is an incoherent record and rejects at
    # resume admission.
    config = build_test_vibe_config(
        session_logging=SessionLoggingConfig(enabled=True, save_dir=str(tmp_path))
    )
    parent = build_test_agent_loop(config=config)
    factory = AgentRuntimeFactory()
    child = await factory.create_child(parent, "coder")
    await child.persist_empty_session()
    child_dir = child.session_logger.session_dir
    assert child_dir is not None
    child_id = child.session_id
    await child.aclose()

    meta_path = child_dir / "meta.json"
    raw = json.loads(meta_path.read_text(encoding="utf-8"))
    raw["launch_config"]["persona"]["identity"] = "reviewer"
    meta_path.write_text(json.dumps(raw), encoding="utf-8")

    try:
        with pytest.raises(InvalidLaunchConfigError) as raised:
            await factory.resume_child(parent, "coder", child_id, child_dir)
        assert raised.value.field == "launch_config.persona"
    finally:
        await parent.aclose()
