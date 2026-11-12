"""Cross-process policy-tree transactions and their crash recovery (WP5 [5D]).

The four phases are driven once, over the node port: the in-process cut and
the real child-process adapter run the same transaction code. A crash is
injected at each named recovery point by stopping between phases and letting a
fresh transaction object adjudicate the journal.
"""

from __future__ import annotations

import inspect
import logging
import os
from pathlib import Path
import sys
from typing import Any
from unittest.mock import AsyncMock

import pytest

from chartreux.app_server._execution import SessionExecutionConflict
from chartreux.app_server._policy_tree import (
    FilePolicyTreeJournal,
    InProcessPolicyNode,
    InProcessPolicyTree,
    ManifestPolicyTreeJournal,
    PolicyChange,
    PolicyNodeRevision,
    PolicyNodeService,
    PolicyRevisionStaleError,
    PolicyTreePhase,
    PolicyTreeRecoveryDecision,
    PolicyTreeRetiredPendingError,
    PolicyTreeTransaction,
    ProcessPolicyNode,
    ProcessPolicyTree,
    policy_tree_transport,
)
from chartreux.app_server._sessions import SessionRuntime, SessionRuntimeRegistry
from chartreux.app_server._subagent_launch import ProcessChildSession
from chartreux.app_server.agent_wire import (
    POLICY_COMMIT_METHOD,
    POLICY_PREPARE_METHOD,
    POLICY_RESERVE_METHOD,
    POLICY_RETIRE_METHOD,
    POLICY_REVISION_METHOD,
)
from chartreux.app_server.child_agent_process import ChildAgentProcess
from chartreux.app_server.process_bootstrap import spawn_process
from chartreux.app_server.registry_manifest import RegistryManifest
from chartreux.app_server.subagent_ports import SubagentSpawnSpec
from chartreux.app_server.transport import (
    IN_PROCESS_TRANSPORT,
    STDIO_TRANSPORT,
    TRANSPORT_ENV_VAR,
    TransportSelection,
)
from chartreux.core.agent_loop import AgentLoop
from tests.agent_loop.test_policy_replacement import make_loop
from tests.differential import egress

pytestmark = pytest.mark.asyncio

_REPO_ROOT = Path(__file__).resolve().parents[2]

_STDIO_SELECTION = TransportSelection(
    kind=STDIO_TRANSPORT, selector=STDIO_TRANSPORT, source="environment"
)
_IN_PROCESS_SELECTION = TransportSelection(
    kind=IN_PROCESS_TRANSPORT, selector=IN_PROCESS_TRANSPORT, source="default"
)


def change(*patterns: str) -> PolicyChange:
    return PolicyChange(
        source="user-toml", tools={"read_file": {"denylist": list(patterns)}}
    )


def journal(tmp_path: Path, transaction_id: str) -> FilePolicyTreeJournal:
    return FilePolicyTreeJournal(tmp_path / "journal", transaction_id)


def denylist(loop: AgentLoop) -> set[str]:
    return set(loop.tool_manager.get_tool_config("read_file").denylist)


async def close(*loops: AgentLoop) -> None:
    for loop in loops:
        await loop.aclose()


async def test_transaction_runs_all_four_phases_in_process(tmp_path: Path) -> None:
    root, child = await make_loop(tmp_path), await make_loop(tmp_path)
    try:
        transaction = PolicyTreeTransaction(
            "tx-in-process",
            nodes=[
                InProcessPolicyNode("root", root),
                InProcessPolicyNode("child", child),
            ],
            journal=journal(tmp_path, "tx-in-process"),
            change=change("B"),
        )
        record = await transaction.run()
        assert record.phase is PolicyTreePhase.RETIRED
        assert transaction.require_settled().phase is PolicyTreePhase.RETIRED
        for loop in (root, child):
            assert "B" in denylist(loop)
            # The [3C-1] revision lock advances with the commit.
            assert loop._dangerous_command_grants.current_policy_revision == 1
    finally:
        await close(root, child)


async def test_in_process_escape_runs_the_same_transaction_code(tmp_path: Path) -> None:
    root = await make_loop(tmp_path)
    try:
        assert isinstance(policy_tree_transport({}), InProcessPolicyTree)
        assert isinstance(
            policy_tree_transport({TRANSPORT_ENV_VAR: "in_process"}),
            InProcessPolicyTree,
        )
        assert isinstance(
            policy_tree_transport({TRANSPORT_ENV_VAR: "stdio"}), ProcessPolicyTree
        )
        transport = policy_tree_transport({TRANSPORT_ENV_VAR: "in_process"})
        transaction = PolicyTreeTransaction(
            "tx-escape",
            nodes=[transport.node(root, node_id="root")],
            journal=journal(tmp_path, "tx-escape"),
            change=change("B"),
        )
        assert (await transaction.run()).phase is PolicyTreePhase.RETIRED
        assert "B" in denylist(root)
    finally:
        await close(root)


async def test_stale_revision_commit_is_refused_typed(tmp_path: Path) -> None:
    root = await make_loop(tmp_path)
    try:
        node = InProcessPolicyNode("root", root)
        transaction = PolicyTreeTransaction(
            "tx-stale",
            nodes=[node],
            journal=journal(tmp_path, "tx-stale"),
            change=change("B"),
        )
        await transaction.reserve()
        await transaction.prepare()
        manager = root.tool_manager
        # A revocation/policy change lands between prepare and commit: the
        # [3C-1] lock no longer matches the reserved revision.
        root.advance_dangerous_command_policy_revision()
        with pytest.raises(PolicyRevisionStaleError, match="stale"):
            await transaction.commit()
        assert root.tool_manager is manager
        assert "B" not in denylist(root)
        assert transaction.recover().phase is PolicyTreePhase.PREPARED
        await transaction.abort()
        assert transaction.recover().phase is PolicyTreePhase.ABORTED
    finally:
        await close(root)


async def test_stale_tree_revision_commit_is_refused_typed(tmp_path: Path) -> None:
    root = await make_loop(tmp_path)
    try:
        transaction = PolicyTreeTransaction(
            "tx-tree-stale",
            nodes=[InProcessPolicyNode("root", root)],
            journal=journal(tmp_path, "tx-tree-stale"),
            change=change("B"),
        )
        await transaction.reserve()
        await transaction.prepare()
        with pytest.raises(PolicyRevisionStaleError, match="tree revision is stale"):
            await transaction.commit(expected_revision="another-tree-revision")
        assert "B" not in denylist(root)
        await transaction.abort()
    finally:
        await close(root)


async def test_crash_before_prepare_is_recovered_by_reprepare(tmp_path: Path) -> None:
    root = await make_loop(tmp_path)
    try:
        transaction = PolicyTreeTransaction(
            "tx-reserve-crash",
            nodes=[InProcessPolicyNode("root", root)],
            journal=journal(tmp_path, "tx-reserve-crash"),
            change=change("B"),
        )
        await transaction.reserve()
        # Crash: prepare never ran, and the transaction object is gone. A
        # fresh transaction adjudicates the journal and completes the unit.
        resumed = PolicyTreeTransaction(
            "tx-reserve-crash",
            nodes=[InProcessPolicyNode("root", root)],
            journal=journal(tmp_path, "tx-reserve-crash"),
            change=change("B"),
        )
        recovery = resumed.recover()
        assert recovery.phase is PolicyTreePhase.RESERVED
        assert recovery.decision is PolicyTreeRecoveryDecision.REPREPARE_OR_ABORT
        assert "B" not in denylist(root)
        assert (await resumed.reprepare()).phase is PolicyTreePhase.RETIRED
        assert "B" in denylist(root)
    finally:
        await close(root)


async def test_reserved_unprepared_tree_can_abort_and_reuse_the_node(
    tmp_path: Path,
) -> None:
    root = await make_loop(tmp_path)
    try:
        transaction = PolicyTreeTransaction(
            "tx-abort",
            nodes=[InProcessPolicyNode("root", root)],
            journal=journal(tmp_path, "tx-abort"),
            change=change("B"),
        )
        await transaction.reserve()
        resumed = PolicyTreeTransaction(
            "tx-abort",
            nodes=[InProcessPolicyNode("root", root)],
            journal=journal(tmp_path, "tx-abort"),
            change=change("B"),
        )
        assert (await resumed.abort()).phase is PolicyTreePhase.ABORTED
        assert "B" not in denylist(root)
        # The node is free again: a later transaction reaches retirement.
        later = PolicyTreeTransaction(
            "tx-after-abort",
            nodes=[InProcessPolicyNode("root", root)],
            journal=journal(tmp_path, "tx-after-abort"),
            change=change("C"),
        )
        assert (await later.run()).phase is PolicyTreePhase.RETIRED
        assert "C" in denylist(root)
    finally:
        await close(root)


async def test_crash_after_commit_is_retired_pending_and_reads_fail_closed(
    tmp_path: Path,
) -> None:
    root = await make_loop(tmp_path)
    try:
        transaction = PolicyTreeTransaction(
            "tx-commit-crash",
            nodes=[InProcessPolicyNode("root", root)],
            journal=journal(tmp_path, "tx-commit-crash"),
            change=change("B"),
        )
        await transaction.reserve()
        await transaction.prepare()
        await transaction.commit()
        # Crash before retire: the new policy is published, the previous
        # authority is still retired-pending.
        resumed = PolicyTreeTransaction(
            "tx-commit-crash",
            nodes=[InProcessPolicyNode("root", root)],
            journal=journal(tmp_path, "tx-commit-crash"),
            change=change("B"),
        )
        recovery = resumed.recover()
        assert recovery.phase is PolicyTreePhase.COMMITTED
        assert recovery.decision is PolicyTreeRecoveryDecision.RETIRED_PENDING
        assert "B" in denylist(root)
        with pytest.raises(PolicyTreeRetiredPendingError, match="retired-pending"):
            resumed.require_settled()
        assert (await resumed.resume_retire()).phase is PolicyTreePhase.RETIRED
        assert resumed.require_settled().phase is PolicyTreePhase.RETIRED
        # Retirement completion is idempotent.
        assert (await resumed.resume_retire()).phase is PolicyTreePhase.RETIRED
    finally:
        await close(root)


async def test_manifest_journal_is_durable_and_adds_no_artifact(tmp_path: Path) -> None:
    session = tmp_path / "sessions" / "root"
    session.mkdir(parents=True)
    manifest = RegistryManifest(session)
    manifest.initialize()
    journal_ids = ("tx-manifest",)
    assert manifest.policy_transaction(journal_ids[0]) is None
    manifest.record_policy_transaction(journal_ids[0], {"phase": "reserved"})
    # A reloaded manifest reconciles the surviving record across a restart:
    # initialize() rewrites the file and keeps the policy rows.
    reloaded = RegistryManifest(session)
    reloaded.initialize()
    assert reloaded.policy_transaction(journal_ids[0]) == {"phase": "reserved"}
    files = {
        str(path.relative_to(session.parent))
        for path in session.parent.rglob("*")
        if path.is_file()
    }
    assert files == {"root.registry.json", "root/.writer-fence.lock"}


async def test_manifest_journal_carries_a_transaction(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    session = tmp_path / "sessions" / "root"
    session.mkdir(parents=True)
    manifest = RegistryManifest(session)
    manifest.initialize()
    root = await make_loop(work)
    try:
        transaction = PolicyTreeTransaction(
            "tx-manifest",
            nodes=[InProcessPolicyNode("root", root)],
            journal=ManifestPolicyTreeJournal(manifest, "tx-manifest"),
            change=change("B"),
        )
        assert (await transaction.run()).phase is PolicyTreePhase.RETIRED
        assert "B" in denylist(root)
        assert RegistryManifest(session).policy_transaction("tx-manifest") is not None
    finally:
        await close(root)


_CHILD_SCRIPT = """
import asyncio
from pathlib import Path

from chartreux.app_server.child_agent import serve_child_agent
from chartreux.app_server.process_bootstrap import bootstrap_requested
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.config import ChartreuxConfigSchema
from chartreux.core.config.harness_files import init_harness_files_manager
from chartreux.core.config.layers.default import DefaultConfigLayer
from chartreux.core.config.layers.overrides import OverridesLayer
from chartreux.core.config.layers.user import UserConfigLayer
from chartreux.core.config.orchestrator import ConfigOrchestrator
from tests.stubs.fake_backend import FakeBackend


class _Factory:
    def __init__(self, loop):
        self._loop = loop

    async def open(self, params):
        return self._loop

    async def close(self):
        pass


async def main():
    init_harness_files_manager("user", "project")
    root = Path(__WORK_DIR__)
    settings = root / "settings.toml"
    settings.write_text('[tools.read_file]\\ndenylist = ["A"]\\n')
    session = OverridesLayer(
        data={
            "tools": {
                "read_file": {
                    "permission": "always",
                    "denylist": [],
                    "sensitive_patterns": [],
                }
            }
        }
    )
    orchestrator = await ConfigOrchestrator.create(
        schema=ChartreuxConfigSchema,
        layers=[
            DefaultConfigLayer(schema=ChartreuxConfigSchema),
            UserConfigLayer(path=settings),
            session,
        ],
        default_layer_resolver=lambda: session,
    )
    loop = AgentLoop(
        config_orchestrator=orchestrator, cwd=root, backend=FakeBackend()
    )
    await serve_child_agent(factory=_Factory(loop), announce_ready=bootstrap_requested())


asyncio.run(main())
"""


def _child_command(tmp_path: Path) -> list[str]:
    work = tmp_path / "child-work"
    work.mkdir(exist_ok=True)
    script = _CHILD_SCRIPT.replace("__WORK_DIR__", repr(str(work)))
    return [sys.executable, "-c", script]


def _child_env(guard_dir: Path) -> dict[str, str]:
    env = dict(os.environ)
    env.update(egress.child_environment(egress.MODE_ALL))
    env[TRANSPORT_ENV_VAR] = "stdio"
    paths = [str(_REPO_ROOT), str(guard_dir)]
    inherited = env.get("PYTHONPATH")
    if inherited:
        paths.append(inherited)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _spec() -> SubagentSpawnSpec:
    return SubagentSpawnSpec(agent_id="agent-1", parent_session_id="root-session-1")


@pytest.mark.timeout(120)
async def test_process_node_commits_across_the_process_boundary(tmp_path: Path) -> None:
    with egress.bootstrap_directory() as guard_dir:
        spawned = await spawn_process(
            _child_command(tmp_path),
            env=_child_env(guard_dir),
            cwd=str(_REPO_ROOT),
            ready_timeout=30,
        )
        process = ChildAgentProcess(spawned, request_timeout=60)
        try:
            await process.open_subagent(_spec())
            node = ProcessPolicyNode("agent-1", process)
            transaction = PolicyTreeTransaction(
                "tx-process",
                nodes=[node],
                journal=journal(tmp_path, "tx-process"),
                change=change("B"),
            )
            assert (await transaction.run()).phase is PolicyTreePhase.RETIRED
            # The commit really landed in the child: its policy lock advanced.
            assert (await node.revision()).policy_revision == 1
        finally:
            await process.close()


@pytest.mark.timeout(120)
async def test_process_node_refuses_a_stale_commit_typed(tmp_path: Path) -> None:
    with egress.bootstrap_directory() as guard_dir:
        spawned = await spawn_process(
            _child_command(tmp_path),
            env=_child_env(guard_dir),
            cwd=str(_REPO_ROOT),
            ready_timeout=30,
        )
        process = ChildAgentProcess(spawned, request_timeout=60)
        try:
            await process.open_subagent(_spec())
            node = ProcessPolicyNode("agent-1", process)
            transaction = PolicyTreeTransaction(
                "tx-process-stale",
                nodes=[node],
                journal=journal(tmp_path, "tx-process-stale"),
                change=change("B"),
            )
            reserved = await transaction.reserve()
            handle = await node.prepare(
                "tx-process-stale", change("B"), reserved.nodes[0].revision
            )
            stale = PolicyNodeRevision(
                tree_revision=reserved.nodes[0].revision.tree_revision,
                policy_revision=reserved.nodes[0].revision.policy_revision + 1,
            )
            with pytest.raises(PolicyRevisionStaleError):
                await node.commit("tx-process-stale", handle, stale)
            assert (await node.revision()).policy_revision == 0
            await transaction.abort()
        finally:
            await process.close()


# ---------------------------------------------------------------------------
# Package 20b: the production call site — a stdio session's replacement drives
# each process-backed descendant through its own wire.
# ---------------------------------------------------------------------------


async def _ignore(*_args: Any, **_kwargs: Any) -> None:
    return None


class _WirePolicyClient:
    """The coordinator-side client of one child's own policy authority.

    The child's ``PolicyNodeService`` answers every frame, so this double
    reproduces the process boundary without a second interpreter: the phases
    and their refusals are the shipped ones.
    """

    def __init__(self, loop: AgentLoop) -> None:
        self.service = PolicyNodeService(loop)
        self.methods: list[str] = []

    async def policy_request(
        self, method: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        self.methods.append(method)
        return await self.service.handle(method, params)


class _ProcessSpawnedChild:
    """A spawned child handle whose process the test declares reachable or not."""

    def __init__(self, wire: _WirePolicyClient, *, alive: bool) -> None:
        self.agent_id = "coder-process"
        self.session_id = "child-process-session"
        self.child_session_dir = None
        self.root_session_dir = None
        self.completion_source: Any = None
        self.wire_client = wire
        self._alive = alive

    @property
    def alive(self) -> bool:
        return self._alive

    async def close(self) -> None:
        self._alive = False

    def reconcile_death(self) -> Any:
        return None


async def _stdio_tree(
    tmp_path: Path, *, child_alive: bool = True
) -> tuple[SessionRuntimeRegistry, SessionRuntime, _WirePolicyClient, list[AgentLoop]]:
    """A stdio-selected registry whose child lives behind its own wire."""
    root_dir = tmp_path / "root"
    child_dir = tmp_path / "child"
    root_dir.mkdir()
    child_dir.mkdir()
    root_loop = await make_loop(root_dir)
    child_loop = await make_loop(child_dir)
    manifest_dir = tmp_path / "manifest"
    manifest_dir.mkdir()
    manifest = RegistryManifest(manifest_dir)
    manifest.initialize()
    registry = SessionRuntimeRegistry(
        _ignore, _ignore, lambda _session_id: 0, transport_selection=_STDIO_SELECTION
    )
    root = registry._build_child_runtime(root_loop)
    registry.bind_root(root)
    registry._runtime_factory.registry_manifests[root_loop] = manifest
    wire = _WirePolicyClient(child_loop)
    spawned = _ProcessSpawnedChild(wire, alive=child_alive)
    registry._process_children[spawned.session_id] = ProcessChildSession(
        agent_id=spawned.agent_id,
        identity="coder",
        session_id=spawned.session_id,
        attempt_id="attempt-policy-1",
        spawned=spawned,
    )
    return registry, root, wire, [root_loop, child_loop]


async def _replace(registry: SessionRuntimeRegistry, root: SessionRuntime) -> str:
    return await registry.replace_policy(
        session_id=root.agent_loop.session_id,
        source="user-toml",
        tools={"read_file": {"denylist": ["B"]}},
        expected_revision=registry.policy_revision,
    )


async def test_policy_tree_transport_requires_the_resolved_selection() -> None:
    process_topology = policy_tree_transport(_STDIO_SELECTION)
    in_process_topology = policy_tree_transport(_IN_PROCESS_SELECTION)
    assert isinstance(process_topology, ProcessPolicyTree)
    assert isinstance(in_process_topology, InProcessPolicyTree)
    # No environment default: the topology is the session's resolved choice,
    # and an uninjected call must fail rather than answer from an empty environ.
    parameter = inspect.signature(policy_tree_transport).parameters["selection"]
    assert parameter.default is inspect.Parameter.empty


async def test_a_registry_without_a_selection_keeps_the_in_process_topology(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An ambient selector must not claim a topology this session never chose:
    # only the explicit session selection reaches the seam.
    monkeypatch.setenv(TRANSPORT_ENV_VAR, STDIO_TRANSPORT)
    registry = SessionRuntimeRegistry(AsyncMock(), AsyncMock(), lambda _session_id: 0)
    assert isinstance(registry.policy_transport(), InProcessPolicyTree)


async def test_replace_policy_drives_a_process_child_through_its_wire(
    tmp_path: Path,
) -> None:
    registry, root, wire, loops = await _stdio_tree(tmp_path)
    try:
        await _replace(registry, root)
        # The change is published in the process that owns each authority.
        assert "B" in denylist(root.agent_loop)
        assert "B" in denylist(loops[1])
        assert loops[1]._dangerous_command_grants.current_policy_revision == 1
        # Every phase crossed the child's own wire, in transaction order.
        assert wire.methods == [
            POLICY_REVISION_METHOD,
            POLICY_RESERVE_METHOD,
            POLICY_PREPARE_METHOD,
            POLICY_COMMIT_METHOD,
            POLICY_RETIRE_METHOD,
        ]
    finally:
        await close(*loops)


async def test_replace_policy_refuses_an_unreachable_process_child(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    registry, root, wire, loops = await _stdio_tree(tmp_path, child_alive=False)
    before = [denylist(loop) for loop in loops]
    try:
        with caplog.at_level(logging.WARNING):
            with pytest.raises(SessionExecutionConflict, match="cannot reach"):
                await _replace(registry, root)
        # Refused BEFORE the first reserve: nothing was prepared, published or
        # driven over the dead child's wire.
        assert "unreachable" in caplog.text
        assert wire.methods == []
        assert [denylist(loop) for loop in loops] == before
        assert all("B" not in patterns for patterns in before)
    finally:
        await close(*loops)


async def test_replace_policy_refuses_a_roots_change_across_processes(
    tmp_path: Path,
) -> None:
    registry, root, wire, loops = await _stdio_tree(tmp_path)
    before = [denylist(loop) for loop in loops]
    try:
        with pytest.raises(SessionExecutionConflict, match="in-process"):
            await registry.replace_policy(
                session_id=root.agent_loop.session_id,
                source="user-toml",
                tools={"read_file": {"denylist": ["B"]}},
                expected_revision=registry.policy_revision,
                roots={},
            )
        assert wire.methods == []
        assert [denylist(loop) for loop in loops] == before
    finally:
        await close(*loops)
