"""WP5 [5C]: child-owned transcripts, durable markers, coordinator
reconciliation, and the live paging proxy.

The crash-injection cells inject death before the start marker, after the
start marker, and after the terminal save, and assert the coordinator
classifies each case as never-executed / started-outcome-unknown / completed
with no replay in any of them.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from uuid import UUID

import pytest

from chartreux.app_server import _runtime
from chartreux.app_server._agent_transcript import read_live_agent_transcript
from chartreux.app_server._connection_protocol import ClientCapabilities, ClientInfo
from chartreux.app_server._sessions import SessionRuntimeRegistry
from chartreux.app_server._subagent_launch import ProcessChildPublication
from chartreux.app_server._subagent_recovery import (
    AGENT_TRANSCRIPT_RECONCILE_METHOD,
    ChildTranscriptReconciler,
)
from chartreux.app_server._subagent_transcript import (
    AGENT_CONFIG_PAYLOAD_KEY,
    CHILD_RESULT_FILENAME,
    CHILD_START_MARKER_FILENAME,
    CHILD_TOMBSTONE_FILENAME,
    CLIENT_CAPABILITIES_PAYLOAD_KEY,
    CLIENT_INFO_PAYLOAD_KEY,
    INHERITED_MODE_RESTRICTIONS_PAYLOAD_KEY,
    INHERITED_RESTRICTIONS_PAYLOAD_KEY,
    INHERITED_ROOT_GRANTS_PAYLOAD_KEY,
    INHERITED_WORKSPACE_PAYLOAD_KEY,
    ChildResultRecord,
    ChildStartMarker,
    ChildTombstoneRecord,
    ChildTranscriptError,
    ChildTranscriptOutcome,
    ChildTranscriptPathError,
    ChildTranscriptPayloadError,
    ChildTranscriptStore,
    child_result_digest,
    child_transcript_dir_name,
    child_transcript_dirs,
    child_transcript_logging,
    decode_inherited_root_grants,
    decode_inherited_workspace,
    decode_source_restrictions,
    encode_inherited_workspace,
    encode_source_restrictions,
)
from chartreux.app_server.agent_wire import (
    AGENT_TURN_ENDED_METHOD,
    SUBAGENT_OPEN_METHOD,
    TURN_OUTCOME_COMPLETED,
    UnsupportedAgentRequestError,
)
from chartreux.app_server.child_agent import ChildAgentServer, HarnessChildLoopFactory
from chartreux.app_server.child_agent_process import ChildAgentProcess
from chartreux.app_server.process_bootstrap import SpawnedProcess
from chartreux.app_server.protocol import AgentConfig
from chartreux.app_server.registry_manifest import (
    AgentRoute,
    LiveAgentExitedError,
    RegistryManifest,
)
from chartreux.app_server.root_agent import AgentChildHost
from chartreux.app_server.subagent_ports import (
    CHILD_HANDOFF_PAYLOAD_KEY,
    SubagentSpawnSpec,
    child_spawn_spec,
    spawn_spec_to_wire,
)
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.agent_loop._loop_types import AgentRuntimePolicy
from chartreux.core.config._dangerous_command_authority import (
    ProjectDangerousCommandGrant,
    ProjectRemoteExecGrant,
)
from chartreux.core.config._restrictions import (
    PolicySourceIdentity,
    SourceRestrictions,
    ToolRestriction,
)
from chartreux.core.config._root_authority import ProjectRootAuthority
from chartreux.core.config.harness_files import HarnessFilesManager
from chartreux.core.config.models import SessionLoggingConfig, SubagentsConfig
from chartreux.core.dangerous_commands import RemoteDestination
from chartreux.core.hooks.models import HookConfigResult
from chartreux.core.llm_models import LLMMessage, Role
from chartreux.core.session.session_logger import SessionLogger
from chartreux.core.subagents import TaskArgs
from chartreux.core.tools.base import InvokeContext
from chartreux.core.workspace import Workspace
from chartreux.utils.cache_store import FileSystemCacheStore
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend
from tests.stubs.fake_config_orchestrator import FakeConfigOrchestrator

_ROOT_SESSION_ID = "root-session-1"


def _root(tmp_path: Path) -> Path:
    root = tmp_path / "sessions" / _ROOT_SESSION_ID
    root.mkdir(parents=True)
    return root


def _child_dir(root: Path, name: str) -> Path:
    child_dir = root / "agents" / name
    child_dir.mkdir(parents=True)
    return child_dir


def _store(
    root: Path, name: str = "coder-py_20260101_000000_abcdef12"
) -> ChildTranscriptStore:
    return ChildTranscriptStore(_child_dir(root, name), root_session_dir=root)


def _start(turn_id: str = "turn-1") -> ChildStartMarker:
    return ChildStartMarker(
        agent_id="coder:py",
        session_id="child-session-1",
        run_id=turn_id,
        started_at="2026-01-01T00:00:00+00:00",
    )


def _result(turn_id: str = "turn-1") -> ChildResultRecord:
    return ChildResultRecord(
        agent_id="coder:py",
        session_id="child-session-1",
        run_id=turn_id,
        result_id=f"coder:py:{turn_id}",
        digest=child_result_digest(
            agent_id="coder:py",
            session_id="child-session-1",
            run_id=turn_id,
            outcome=TURN_OUTCOME_COMPLETED,
            error=None,
        ),
        outcome=TURN_OUTCOME_COMPLETED,
        completed_at="2026-01-01T00:00:05+00:00",
    )


# ---------------------------------------------------------------------------
# Layout: child transcripts are subdirectories of the root session dir
# ---------------------------------------------------------------------------


def test_child_transcript_dir_is_a_subdirectory_of_the_root_session(
    tmp_path: Path,
) -> None:
    root = _root(tmp_path)
    config = child_transcript_logging(root, "coder-py")
    assert config.enabled is True
    assert config.session_prefix == "coder-py"
    assert Path(config.save_dir) == root / "agents"

    logger_instance = SessionLogger(config, "child-session-1")
    session_dir = logger_instance.session_dir
    assert session_dir is not None
    assert session_dir.parent == root / "agents"
    assert session_dir.name.startswith("coder-py_")
    session_dir.mkdir(parents=True)
    assert child_transcript_dirs(root) == [session_dir]


def test_child_transcript_dir_name_preserves_the_2c_layout() -> None:
    name = child_transcript_dir_name(
        "coder-py", "abcdef1234567890", timestamp="20260101_000000"
    )
    assert name.startswith("coder-py_20260101_000000_")


def test_a_later_child_spawn_gets_a_fresh_directory(tmp_path: Path) -> None:
    root = _root(tmp_path)
    config = child_transcript_logging(root, "coder-py")
    first = SessionLogger(config, "1a2b3c4d-session").session_dir
    second = SessionLogger(config, "9f8e7d6c-session").session_dir
    assert first is not None and second is not None
    # A later child process is fresh: a new session id yields a new transcript
    # directory, never the earlier child's record.
    assert first != second


# ---------------------------------------------------------------------------
# Marker and result durability
# ---------------------------------------------------------------------------


def test_start_marker_and_terminal_result_are_durable_and_atomic(
    tmp_path: Path,
) -> None:
    store = _store(_root(tmp_path))
    store.record_start(_start())
    assert store.start_marker_path.is_file()
    assert store.start_marker_path.read_text().startswith("{")
    # Atomic write leaves no temporary sibling behind.
    assert sorted(path.name for path in store.child_dir.iterdir()) == [
        CHILD_START_MARKER_FILENAME
    ]

    store.record_result(_result())
    assert sorted(path.name for path in store.child_dir.iterdir()) == sorted([
        CHILD_START_MARKER_FILENAME,
        CHILD_RESULT_FILENAME,
    ])
    adjudication = store.adjudicate()
    assert adjudication.outcome is ChildTranscriptOutcome.COMPLETED
    assert adjudication.result_id == "coder:py:turn-1"
    assert adjudication.digest == _result().digest


def test_record_result_replaces_the_previous_run_record(tmp_path: Path) -> None:
    store = _store(_root(tmp_path))
    store.record_start(_start("turn-1"))
    store.record_result(_result("turn-1"))
    store.record_start(_start("turn-2"))
    # A second run's start marker makes the newer run the in-flight one.
    assert store.adjudicate().outcome is ChildTranscriptOutcome.STARTED_OUTCOME_UNKNOWN
    store.record_result(_result("turn-2"))
    adjudication = store.adjudicate()
    assert adjudication.outcome is ChildTranscriptOutcome.COMPLETED
    assert adjudication.result_id == "coder:py:turn-2"


# ---------------------------------------------------------------------------
# Crash injection: never executed / outcome unknown / completed
# ---------------------------------------------------------------------------


def test_child_death_before_the_start_marker_is_never_executed(tmp_path: Path) -> None:
    root = _root(tmp_path)
    child_dir = _child_dir(root, "coder-py_20260101_000000_abcdef12")

    reconciliation = ChildTranscriptReconciler(root).reconcile(
        child_dir, agent_id="coder:py"
    )

    assert reconciliation.outcome is ChildTranscriptOutcome.NEVER_EXECUTED
    assert reconciliation.result_id is None
    assert reconciliation.denial is not None
    assert "never executed" in reconciliation.denial
    assert "replayed" in reconciliation.denial
    assert list(child_dir.iterdir()) == []


def test_child_death_after_the_start_marker_is_outcome_unknown_and_not_replayed(
    tmp_path: Path,
) -> None:
    root = _root(tmp_path)
    store = _store(root)
    store.record_start(_start())
    before = store.start_marker_path.read_bytes()

    reconciliation = ChildTranscriptReconciler(root).reconcile(
        store.child_dir, agent_id="coder:py"
    )

    assert reconciliation.outcome is ChildTranscriptOutcome.STARTED_OUTCOME_UNKNOWN
    assert reconciliation.run_id == "turn-1"
    assert reconciliation.result_id is None
    assert reconciliation.denial is not None
    assert "outcome is unknown" in reconciliation.denial
    assert "will not be replayed" in reconciliation.denial
    # No replay: the transcript is untouched and no result record is invented.
    assert store.start_marker_path.read_bytes() == before
    assert not store.result_path.exists()


def test_child_death_after_the_terminal_save_reconciles_completed(
    tmp_path: Path,
) -> None:
    root = _root(tmp_path)
    store = _store(root)
    store.record_start(_start())
    store.record_result(_result())

    reconciliation = ChildTranscriptReconciler(root).reconcile(
        store.child_dir, agent_id="coder:py"
    )

    assert reconciliation.outcome is ChildTranscriptOutcome.COMPLETED
    assert reconciliation.result_id == "coder:py:turn-1"
    assert reconciliation.digest == _result().digest
    assert reconciliation.has_unconsumed_result is True
    assert reconciliation.denial is None
    message = reconciliation.coordinator_message()
    assert message["method"] == AGENT_TRANSCRIPT_RECONCILE_METHOD
    params = cast("dict[str, Any]", message["params"])
    assert params["outcome"] == ChildTranscriptOutcome.COMPLETED.value
    assert params["result_id"] == "coder:py:turn-1"
    assert params["digest"] == _result().digest


def test_reconciliation_scan_reports_each_child_once(tmp_path: Path) -> None:
    root = _root(tmp_path)
    completed = _store(root, "coder-py_20260101_000000_aaaaaaaa")
    completed.record_start(_start())
    completed.record_result(_result())
    crashed = _store(root, "coder-rust_20260101_000000_bbbbbbbb")
    crashed.record_start(
        ChildStartMarker(
            agent_id="coder:rust",
            session_id="child-session-2",
            run_id="turn-9",
            started_at="2026-01-01T00:00:00+00:00",
        )
    )
    idle = _child_dir(root, "coder-go_20260101_000000_cccccccc")

    first = ChildTranscriptReconciler(root).scan()
    outcomes = {item.child_dir.name: item.outcome for item in first}
    assert outcomes == {
        "coder-py_20260101_000000_aaaaaaaa": ChildTranscriptOutcome.COMPLETED,
        "coder-rust_20260101_000000_bbbbbbbb": (
            ChildTranscriptOutcome.STARTED_OUTCOME_UNKNOWN
        ),
        "coder-go_20260101_000000_cccccccc": ChildTranscriptOutcome.NEVER_EXECUTED,
    }
    assert idle.is_dir()

    # A consumed result id is an idempotent receipt: it is never handed over
    # a second time.
    replayed = ChildTranscriptReconciler(root).scan(
        consumed_result_ids={"coder:py:turn-1"}
    )
    by_dir = {item.child_dir.name: item for item in replayed}
    assert by_dir["coder-py_20260101_000000_aaaaaaaa"].unconsumed is False
    assert by_dir["coder-py_20260101_000000_aaaaaaaa"].result_id == "coder:py:turn-1"


def test_unreadable_child_transcript_is_surfaced_not_concealed(tmp_path: Path) -> None:
    root = _root(tmp_path)
    store = _store(root)
    store.record_start(_start())
    store.result_path.write_text("{torn")

    reconciliation = ChildTranscriptReconciler(root).reconcile(
        store.child_dir, agent_id="coder:py"
    )

    assert reconciliation.outcome is ChildTranscriptOutcome.UNREADABLE
    assert reconciliation.denial is not None
    assert "unreadable" in reconciliation.denial
    assert "skipped" in reconciliation.denial
    assert [item.outcome for item in ChildTranscriptReconciler(root).scan()] == [
        ChildTranscriptOutcome.UNREADABLE
    ]


def test_child_transcript_paths_are_confined_to_the_agents_subtree(
    tmp_path: Path,
) -> None:
    root = _root(tmp_path)
    outside = tmp_path / "elsewhere" / "coder-py_20260101_000000_abcdef12"
    outside.mkdir(parents=True)
    with pytest.raises(ChildTranscriptPathError):
        ChildTranscriptStore(outside, root_session_dir=root)
    with pytest.raises(ChildTranscriptPathError):
        ChildTranscriptStore(root / "agents", root_session_dir=root)
    assert not outside.joinpath(CHILD_START_MARKER_FILENAME).exists()


# ---------------------------------------------------------------------------
# Tombstones: the durable retraction the reconciliation scan honors
# ---------------------------------------------------------------------------


def _tombstone(
    *, attempt_id: str | None = "attempt-0123456789abcdef"
) -> ChildTombstoneRecord:
    return ChildTombstoneRecord(
        session_id="child-session-1",
        reason="publication-rejected",
        created_at="2026-01-01T00:00:06+00:00",
        attempt_id=attempt_id,
        agent_id="coder:py",
    )


def test_tombstone_record_is_durable_atomic_and_idempotent(tmp_path: Path) -> None:
    store = _store(_root(tmp_path))
    by_session = ChildTombstoneRecord(
        session_id="child-session-1",
        reason="publication-rejected",
        created_at="2026-01-01T00:00:06+00:00",
    )
    store.record_tombstone(by_session)
    assert sorted(path.name for path in store.child_dir.iterdir()) == [
        CHILD_TOMBSTONE_FILENAME
    ]
    assert store.read_tombstone() == by_session

    # A retraction by session id upgrades once the attempt id is known.
    store.record_tombstone(_tombstone())
    recorded = store.read_tombstone()
    assert recorded == _tombstone()
    assert recorded is not None and recorded.attempt_id == "attempt-0123456789abcdef"

    # A repeated retraction of the same attempt keeps the record byte for byte,
    # and a later id-less retraction never downgrades it.
    before = store.tombstone_path.read_bytes()
    store.record_tombstone(_tombstone())
    assert store.tombstone_path.read_bytes() == before
    store.record_tombstone(by_session)
    assert store.tombstone_path.read_bytes() == before


def test_adjudication_honors_a_tombstone_over_every_marker(tmp_path: Path) -> None:
    store = _store(_root(tmp_path))
    store.record_start(_start())
    store.record_result(_result())
    assert store.adjudicate().outcome is ChildTranscriptOutcome.COMPLETED

    store.record_tombstone(_tombstone())
    adjudication = store.adjudicate()
    assert adjudication.outcome is ChildTranscriptOutcome.TOMBSTONED
    assert adjudication.tombstoned is True
    assert adjudication.tombstone == _tombstone()
    assert adjudication.attempt_id == "attempt-0123456789abcdef"
    assert adjudication.result is None
    assert adjudication.result_id is None


def test_a_malformed_tombstone_is_surfaced_not_ignored(tmp_path: Path) -> None:
    root = _root(tmp_path)
    store = _store(root)
    store.record_start(_start())
    store.tombstone_path.write_text("{torn")

    adjudication = store.adjudicate()
    assert adjudication.outcome is ChildTranscriptOutcome.UNREADABLE
    assert adjudication.detail is not None
    assert CHILD_TOMBSTONE_FILENAME in adjudication.detail
    with pytest.raises(ChildTranscriptError):
        store.read_tombstone()
    reconciliation = ChildTranscriptReconciler(root).reconcile(
        store.child_dir, agent_id="coder:py"
    )
    assert reconciliation.outcome is ChildTranscriptOutcome.UNREADABLE


def test_scan_filters_tombstoned_transcripts_and_stays_idempotent(
    tmp_path: Path,
) -> None:
    root = _root(tmp_path)
    completed = _store(root, "coder-py_20260101_000000_aaaaaaaa")
    completed.record_start(_start())
    completed.record_result(_result())
    retracted = _store(root, "coder-rust_20260101_000000_bbbbbbbb")
    retracted.record_start(_start())
    retracted.record_result(_result())
    retracted.record_tombstone(_tombstone())

    reconciler = ChildTranscriptReconciler(root)
    scanned = reconciler.scan()
    assert [item.child_dir.name for item in scanned] == [
        "coder-py_20260101_000000_aaaaaaaa"
    ]
    assert scanned[0].has_unconsumed_result is True
    # The retracted transcript is never handed over, on this scan or a later one.
    assert reconciler.scan() == reconciler.scan()
    reconciliation = reconciler.reconcile(retracted.child_dir)
    assert reconciliation.outcome is ChildTranscriptOutcome.TOMBSTONED
    assert reconciliation.has_unconsumed_result is False
    assert reconciliation.denial is not None
    assert "retracted" in reconciliation.denial


# ---------------------------------------------------------------------------
# Live paging proxy through the registry manifest
# ---------------------------------------------------------------------------


def _page_reader(agent_id: str, messages: list[LLMMessage]) -> Any:
    async def read(bound_agent_id: str, before: str | None, limit: int) -> Any:
        assert bound_agent_id == agent_id
        return await asyncio.to_thread(
            read_live_agent_transcript, messages, before=before, limit=limit
        )

    return read


def _messages(*texts: str) -> list[LLMMessage]:
    return [
        LLMMessage(role=Role.user, message_id=f"message-{index}", content=text)
        for index, text in enumerate(texts, start=1)
    ]


@pytest.mark.asyncio
async def test_live_child_pages_are_proxied_through_the_manifest(
    tmp_path: Path,
) -> None:
    root = _root(tmp_path)
    manifest = RegistryManifest(root)
    manifest.initialize()
    manifest.register_agent(
        "coder:py",
        AgentRoute("connection-1", "incarnation-1"),
        _page_reader("coder:py", _messages("working", "done")),
    )

    page = await manifest.read_live_page("coder:py", limit=10)
    assert [entry.display_text for entry in page.entries or []] == ["working", "done"]

    # A confirmed exit retires the row: paging fails typed, and there is no
    # coordinator disk fallback.
    manifest.remove_agent("coder:py")
    with pytest.raises(LiveAgentExitedError) as exc_info:
        await manifest.read_live_page("coder:py", limit=10)
    assert "exited" in str(exc_info.value)
    assert "session reload" in str(exc_info.value)


class _QuietTransport:
    """A transport whose peer never sends anything (a live, idle child)."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.closed = False

    async def messages(self) -> AsyncIterator[dict[str, Any]]:
        await asyncio.Event().wait()
        yield {}  # pragma: no cover - never reached

    async def send(self, message: dict[str, Any]) -> None:
        self.sent.append(message)

    async def close(self) -> None:
        self.closed = True


async def _open_child_server(
    root: Path, *, session_id: str = "child-session-1"
) -> tuple[ChildAgentServer, _QuietTransport]:
    session_dir = root / "agents" / child_transcript_dir_name("coder-py", session_id)
    session_dir.mkdir(parents=True)
    loop = SimpleNamespace(
        session_id=session_id,
        messages=_messages("hello", "world"),
        session_logger=SimpleNamespace(session_dir=session_dir),
    )

    class _Factory:
        async def open(self, params: Any) -> Any:
            return loop

        async def close(self) -> None:
            pass

    transport = _QuietTransport()
    server = ChildAgentServer(cast("Any", transport), factory=cast("Any", _Factory()))
    spec = child_spawn_spec(
        agent_id="coder:py",
        parent_session_id=_ROOT_SESSION_ID,
        root_session_dir=root,
        payload={},
    )
    result = await server._handle_request(
        SUBAGENT_OPEN_METHOD, spawn_spec_to_wire(spec)
    )
    assert result["agent_id"] == "coder:py"
    assert result["session_id"] == session_id
    assert result["session_dir"] == str(session_dir)
    return server, transport


@pytest.mark.asyncio
async def test_child_process_serves_pages_by_agent_and_session_id(
    tmp_path: Path,
) -> None:
    root = _root(tmp_path)
    server, _transport = await _open_child_server(root)

    by_agent = await server._read_page({"agent_id": "coder:py", "limit": 10})
    by_session = await server._read_page({"agent_id": "child-session-1", "limit": 10})
    assert [entry["displayText"] for entry in by_agent["entries"]] == ["hello", "world"]
    assert by_agent["entries"] == by_session["entries"]

    with pytest.raises(UnsupportedAgentRequestError):
        await server._read_page({"agent_id": "someone-else", "limit": 10})


@pytest.mark.asyncio
async def test_child_turn_writes_its_markers_before_emitting_the_completion(
    tmp_path: Path,
) -> None:
    root = _root(tmp_path)
    server, transport = await _open_child_server(root)

    class _ExecutionPort:
        def run_turn(self, request: Any) -> AsyncIterator[Any]:
            return self._stream()

        async def _stream(self) -> AsyncIterator[Any]:
            if False:  # pragma: no cover - keeps the generator shape
                yield None
            return

    loop = cast("Any", server)._loop
    loop.execution_port = _ExecutionPort()

    await server._run_turn(
        "turn-1", {"message": "hi", "turn_options": {"turn_id": "turn-1"}}
    )

    session_dir = Path(loop.session_logger.session_dir)
    store = ChildTranscriptStore(session_dir, root_session_dir=root)
    adjudication = store.adjudicate()
    assert adjudication.outcome is ChildTranscriptOutcome.COMPLETED
    assert adjudication.result_id == "coder:py:turn-1"
    assert adjudication.digest == child_result_digest(
        agent_id="coder:py",
        session_id="child-session-1",
        run_id="turn-1",
        outcome=TURN_OUTCOME_COMPLETED,
        error=None,
    )
    ended = [
        message
        for message in transport.sent
        if message.get("method") == AGENT_TURN_ENDED_METHOD
    ]
    assert ended and ended[-1]["params"]["outcome"] == TURN_OUTCOME_COMPLETED


# ---------------------------------------------------------------------------
# A later child process is a fresh incarnation, never a resume
# ---------------------------------------------------------------------------


def _spawned_process(transport: _QuietTransport) -> SpawnedProcess:
    return cast(
        SpawnedProcess,
        SimpleNamespace(transport=transport, process=None, retire=lambda: None),
    )


@pytest.mark.asyncio
async def test_each_child_process_spawn_is_a_new_incarnation(tmp_path: Path) -> None:
    root = _root(tmp_path)
    manifest = RegistryManifest(root)
    manifest.initialize()

    first = ChildAgentProcess(_spawned_process(_QuietTransport()))
    second = ChildAgentProcess(_spawned_process(_QuietTransport()))
    try:
        first.register_live_route(manifest, "coder:py")
        first_incarnation = first.incarnation_id
        assert manifest.is_live("coder:py")
        second.register_live_route(manifest, "coder:py")
        assert second.incarnation_id != first_incarnation
        assert manifest.is_live("coder:py")
    finally:
        await first.close()
        await second.close()
    assert not manifest.is_live("coder:py")


def test_reconciliation_message_is_addressed_to_the_root(tmp_path: Path) -> None:
    root = _root(tmp_path)
    store = _store(root)
    store.record_start(_start())
    store.record_result(_result())
    message = (
        ChildTranscriptReconciler(root)
        .reconcile(store.child_dir, agent_id="coder:py")
        .coordinator_message()
    )
    assert message["method"] == "agent/transcripts/reconcile"
    params = cast("dict[str, Any]", message["params"])
    assert params["transcript_dir"] == str(store.child_dir)
    assert params["unconsumed"] is True


# ---------------------------------------------------------------------------
# The child factory binds its loop to the root session's agents/ subtree
# ---------------------------------------------------------------------------


def _empty_policy_payload() -> dict[str, Any]:
    """The F11 parent-policy keys carrying a minimal (empty) policy."""
    return {
        AGENT_CONFIG_PAYLOAD_KEY: AgentConfig().model_dump(mode="json"),
        INHERITED_RESTRICTIONS_PAYLOAD_KEY: [],
        INHERITED_MODE_RESTRICTIONS_PAYLOAD_KEY: [],
        INHERITED_ROOT_GRANTS_PAYLOAD_KEY: [],
    }


@pytest.mark.asyncio
async def test_child_factory_binds_the_transcript_to_the_root_agents_subtree(
    tmp_path: Path,
) -> None:
    root = _root(tmp_path)
    requests: list[Any] = []
    persisted: list[str | None] = []

    async def _persist_empty_session(*, attempt_id: str | None = None) -> None:
        persisted.append(attempt_id)

    loop = SimpleNamespace(
        session_id="child-session-1",
        session_logger=SimpleNamespace(session_dir=None),
        messages=[],
        persist_empty_session=_persist_empty_session,
    )

    class _Process:
        async def open_root(self, request: Any) -> Any:
            requests.append(request)
            return loop

        async def close(self) -> None:
            pass

    factory = object.__new__(HarnessChildLoopFactory)
    factory._process = cast(Any, _Process())
    spec = child_spawn_spec(
        agent_id="coder:py",
        parent_session_id=_ROOT_SESSION_ID,
        root_session_dir=root,
        agent_name="coder-py",
        payload={
            "client_info": {"name": "tests", "version": "1.0"},
            **_empty_policy_payload(),
        },
        attempt_id="attempt-child-1",
    )

    assert await factory.open(spawn_spec_to_wire(spec)) is loop
    # The attempt id lands in the child's FIRST durable write, so a torn
    # persist-to-link window stays attributable to its publication attempt.
    assert persisted == ["attempt-child-1"]
    request = requests[0]
    assert request.is_subagent is True
    assert request.parent_session_id == _ROOT_SESSION_ID
    assert request.session_logging is not None
    assert Path(request.session_logging.save_dir) == root / "agents"
    assert request.session_logging.session_prefix == "coder-py"

    # Fail closed: a spec without the root session dir never opens a loop
    # whose transcript would land outside the root's agents/ subtree.
    with pytest.raises(UnsupportedAgentRequestError):
        await factory.open(spawn_spec_to_wire(SubagentSpawnSpec(agent_id="coder:py")))
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_child_factory_decodes_the_parent_workspace_and_stays_loud_on_legacy(
    tmp_path: Path,
) -> None:
    """The child's open hands the decoded workspace to its loop request.

    The E2 spawn fix: a subagent's loop construction is illegal without the
    parent workspace, so the child's factory decodes the payload value into
    the open request. A legacy payload WITHOUT the key decodes to None and
    the loud open failure stays exactly where the loop constructor raises
    it -- there is no fallback workspace.
    """
    root = _root(tmp_path)
    requests: list[Any] = []

    class _Process:
        async def open_root(self, request: Any) -> Any:
            requests.append(request)
            return SimpleNamespace(
                session_id="child-session-1",
                session_logger=SimpleNamespace(session_dir=None),
                messages=[],
                persist_empty_session=_persist_empty_session,
            )

    async def _persist_empty_session(*, attempt_id: str | None = None) -> None:
        return None

    factory = object.__new__(HarnessChildLoopFactory)
    factory._process = cast(Any, _Process())
    workspace = Workspace(
        cwd=Path("/project/root"),
        authorized_roots=(Path("/project/root"), Path("/other")),
    )

    spec = child_spawn_spec(
        agent_id="coder:py",
        parent_session_id=_ROOT_SESSION_ID,
        root_session_dir=root,
        agent_name="coder-py",
        payload={
            "client_info": {"name": "tests", "version": "1.0"},
            INHERITED_WORKSPACE_PAYLOAD_KEY: encode_inherited_workspace(workspace),
            **_empty_policy_payload(),
        },
        attempt_id="attempt-workspace-1",
    )
    assert await factory.open(spawn_spec_to_wire(spec)) is not None
    request = requests[0]
    assert request.is_subagent is True
    assert request.inherited_workspace == workspace

    legacy_spec = child_spawn_spec(
        agent_id="coder:py",
        parent_session_id=_ROOT_SESSION_ID,
        root_session_dir=root,
        agent_name="coder-py",
        payload={
            "client_info": {"name": "tests", "version": "1.0"},
            **_empty_policy_payload(),
        },
        attempt_id="attempt-workspace-2",
    )
    # The workspace-legacy shape (a pre-F6 parent) has no workspace key: the
    # decoded request carries None, which is the loud subagent open failure
    # an old-version peer deserves -- never a fallback. (F11 keys present:
    # their own loud shape is pinned a test further down.)
    assert await factory.open(spawn_spec_to_wire(legacy_spec)) is not None
    legacy_request = requests[1]
    assert legacy_request.is_subagent is True
    assert legacy_request.inherited_workspace is None


# ---------------------------------------------------------------------------
# A process spawn's payload carries the client identity the child validates
# ---------------------------------------------------------------------------

_SOURCE_RESTRICTION = SourceRestrictions(
    layer_name="project policy",
    locator="AGENTS.md",
    kind="source",
    content_fingerprint="fp-project-policy",
    store_fingerprint="store-fp-project-policy",
    tools=(
        ToolRestriction(
            tool_name="bash",
            denied=True,
            denylist=("rm -rf /",),
            sensitive_patterns=("sk-live-",),
        ),
        ToolRestriction(
            tool_name="write",
            denied=False,
            denylist=(),
            sensitive_patterns=("bearer ",),
        ),
    ),
    identity=PolicySourceIdentity(
        owner=UUID("12345678-1234-5678-1234-567812345678"),
        source=UUID("87654321-4321-8765-4321-876543218765"),
    ),
    authorized_roots=(
        ProjectRootAuthority(project=Path("/project"), roots=(Path("/project"),)),
    ),
    dangerous_command_grants=(
        ProjectDangerousCommandGrant(
            project=Path("/project"), operations=("git.push",)
        ),
    ),
    remote_exec_grants=(
        ProjectRemoteExecGrant(
            project=Path("/project"),
            destinations=(RemoteDestination(user="ci", host="runner", port=2222),),
        ),
    ),
    replaces_roots=True,
)

_MODE_RESTRICTION = SourceRestrictions(
    layer_name="session mode",
    locator="review",
    kind="mode",
    content_fingerprint="fp-review-mode",
    store_fingerprint=None,
    tools=(
        ToolRestriction(
            tool_name="file_write", denied=True, denylist=(), sensitive_patterns=()
        ),
    ),
    identity=None,
    authorized_roots=(),
    dangerous_command_grants=(),
    remote_exec_grants=(),
    replaces_roots=False,
)

_CLIENT_INFO = ClientInfo(name="chartreux-tests", version="1.0")
_CLIENT_CAPABILITIES = ClientCapabilities(callback_kinds=["user_input"])


class _SpawnAdapter:
    """Records the spawn spec, the payload carrier of a process launch."""

    agent_id = "recorded"

    def __init__(self) -> None:
        self.specs: list[Any] = []

    async def spawn(self, spec: Any) -> Any:
        self.specs.append(spec)
        return SimpleNamespace(
            agent_id=spec.agent_id, session_id="child-session-1", child_session_dir=None
        )


def _publish_root(tmp_path: Path) -> AgentLoop:
    return build_test_agent_loop(
        config=build_test_vibe_config(
            session_logging=SessionLoggingConfig(
                enabled=True,
                save_dir=str(tmp_path / "sessions"),
                session_prefix="session",
            )
        ),
        backend=FakeBackend([[mock_llm_chunk(content="ack")]]),
    )


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_root_publish_payload_carries_the_validated_client_identity(
    tmp_path: Path,
) -> None:
    """The root agent's own spawn payload names the client the child validates.

    The child's ``HarnessChildLoopFactory.open`` validates ``client_info``
    from the payload (it required name/version), so a payload the host builds
    without the identity fails the child's open deterministically. The host
    built with the validated identity of this root's open carries it, and
    the child's model round-trips it.
    """
    loop = _publish_root(tmp_path)
    port = _SpawnAdapter()
    host = AgentChildHost(
        loop,
        spawn=port,
        client_info=_CLIENT_INFO,
        client_capabilities=_CLIENT_CAPABILITIES,
    )
    try:
        handle = await host.publish(
            TaskArgs(task="Inspect the project", identity="coder"),
            InvokeContext(tool_call_id="task-1", session_id=loop.session_id),
            attempt_id="attempt-spawn-1",
            retained=False,
        )
    finally:
        await loop.aclose()

    assert handle.session_id == "child-session-1"
    payload = port.specs[0].payload
    info = payload[CLIENT_INFO_PAYLOAD_KEY]
    assert info["name"] == "chartreux-tests"
    assert info["version"] == "1.0"
    assert ClientInfo.model_validate(info) == _CLIENT_INFO
    assert payload[CLIENT_CAPABILITIES_PAYLOAD_KEY] == _CLIENT_CAPABILITIES.model_dump(
        mode="json"
    )


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_root_publish_payload_without_identity_keeps_the_legacy_shape(
    tmp_path: Path,
) -> None:
    """The host holds no identity: the payload keeps its pre-F4 shape.

    This lenience is disclosed, not endorsed: a production root always
    supplies the validated identity (``RootLoopFactory.open``), so the
    legacy shape exists only for hosts built outside that open, whose
    children use factories that do not validate client_info.
    """
    loop = _publish_root(tmp_path)
    port = _SpawnAdapter()
    host = AgentChildHost(loop, spawn=port)
    try:
        handle = await host.publish(
            TaskArgs(task="Inspect the project", identity="coder"),
            InvokeContext(tool_call_id="task-1", session_id=loop.session_id),
            attempt_id="attempt-spawn-2",
            retained=False,
        )
    finally:
        await loop.aclose()

    assert handle.session_id == "child-session-1"
    assert CLIENT_INFO_PAYLOAD_KEY not in port.specs[0].payload
    assert CLIENT_CAPABILITIES_PAYLOAD_KEY not in port.specs[0].payload


class _HandoffLoop:
    """The parent loop shape the coordinator's handoff payload reads."""

    def __init__(self, content: str | None, workspace: Workspace) -> None:
        self._content = content
        self.cwd = workspace.cwd
        self.tool_manager = SimpleNamespace(workspace=workspace)
        self.runtime_policy = AgentRuntimePolicy(
            max_turns=12,
            max_price=None,
            max_tokens=None,
            max_session_tokens=None,
            run_max_price=1.25,
            run_max_session_tokens=2048,
            enable_streaming=True,
            launch_context=None,
            headless=False,
            hook_config_result=None,
            cache_store=None,  # type: ignore[arg-type]
            inherited_restrictions=(_SOURCE_RESTRICTION,),
            inherited_mode_restrictions=(_MODE_RESTRICTION,),
            inherited_workspace=workspace,
        )
        self.config_orchestrator = SimpleNamespace(restrictions=())
        self._dispatch_root_grants = (Path("/granted/one"),)

    def file_state_handoff_message(self) -> Any:
        return None if self._content is None else SimpleNamespace(content=self._content)


class _Identity:
    """The backend-services identity subset a coordinator spawn consumes."""

    def client_info(self) -> ClientInfo:
        return _CLIENT_INFO

    def client_capabilities(self) -> ClientCapabilities:
        return _CLIENT_CAPABILITIES


@pytest.mark.asyncio
async def test_process_handoff_payload_carries_the_provisioned_client_identity() -> (
    None
):
    """The coordinator's publisher merges identity and workspace it must.

    The view provisions the publisher together with its client-identity
    services (``SpawnClientIdentity``, the backend-services subset): the
    payload carries both identity keys whenever the identity is provisioned,
    with or without a working-set handoff, and -- unlike the MAY-member
    identity -- the parent workspace ALWAYS, because the child's subagent
    loop construction is illegal without it.
    """
    identity = _Identity()
    workspace = Workspace(
        cwd=Path("/project/root"),
        authorized_roots=(Path("/project/root"),),
        ceiling=Workspace(cwd=Path("/project"), authorized_roots=(Path("/project"),)),
    )

    def _publisher(with_identity: bool) -> ProcessChildPublication:
        return ProcessChildPublication(
            admission=cast("Any", SimpleNamespace()),
            spawn=cast("Any", SimpleNamespace()),
            rollback_wait=0.1,
            identity=identity if with_identity else None,
        )

    def _attempt(content: str | None) -> Any:
        return SimpleNamespace(
            parent=SimpleNamespace(agent_loop=_HandoffLoop(content, workspace))
        )

    handed_over = _publisher(True)._handoff_payload(_attempt("working set"))
    assert handed_over[CHILD_HANDOFF_PAYLOAD_KEY] == "working set"
    info = handed_over[CLIENT_INFO_PAYLOAD_KEY]
    assert info["name"] == "chartreux-tests"
    assert ClientInfo.model_validate(info) == _CLIENT_INFO
    assert handed_over[
        CLIENT_CAPABILITIES_PAYLOAD_KEY
    ] == _CLIENT_CAPABILITIES.model_dump(mode="json")
    assert (
        decode_inherited_workspace(handed_over[INHERITED_WORKSPACE_PAYLOAD_KEY])
        == workspace
    )
    # F11: the parent's policy rides the coordinator's payload the same
    # unconditional way -- decoded values equal what the fake loop holds.
    assert decode_source_restrictions(
        handed_over[INHERITED_RESTRICTIONS_PAYLOAD_KEY]
    ) == (_SOURCE_RESTRICTION,)
    assert decode_source_restrictions(
        handed_over[INHERITED_MODE_RESTRICTIONS_PAYLOAD_KEY]
    ) == (_MODE_RESTRICTION,)
    assert decode_inherited_root_grants(
        handed_over[INHERITED_ROOT_GRANTS_PAYLOAD_KEY]
    ) == (Path("/granted/one"),)
    assert AgentConfig.model_validate(
        handed_over[AGENT_CONFIG_PAYLOAD_KEY]
    ) == AgentConfig(
        cwd=str(workspace.cwd),
        max_turns=12,
        run_max_price=1.25,
        run_max_session_tokens=2048,
    )

    identity_only = _publisher(True)._handoff_payload(_attempt(None))
    assert CHILD_HANDOFF_PAYLOAD_KEY not in identity_only
    assert identity_only[CLIENT_INFO_PAYLOAD_KEY]["name"] == "chartreux-tests"
    assert INHERITED_WORKSPACE_PAYLOAD_KEY in identity_only
    assert AGENT_CONFIG_PAYLOAD_KEY in identity_only

    legacy = _publisher(False)._handoff_payload(_attempt("working set"))
    assert set(legacy) == {
        CHILD_HANDOFF_PAYLOAD_KEY,
        INHERITED_WORKSPACE_PAYLOAD_KEY,
        AGENT_CONFIG_PAYLOAD_KEY,
        INHERITED_RESTRICTIONS_PAYLOAD_KEY,
        INHERITED_MODE_RESTRICTIONS_PAYLOAD_KEY,
        INHERITED_ROOT_GRANTS_PAYLOAD_KEY,
    }
    assert decode_inherited_workspace(legacy[INHERITED_WORKSPACE_PAYLOAD_KEY]) == (
        workspace
    )


def test_inherited_workspace_codec_round_trips_every_path_exactly() -> None:
    """The strict codec loses nothing: cwd, roots and a recursive ceiling."""
    workspace = Workspace(
        cwd=Path("/project/root"),
        authorized_roots=(Path("/project/root"), Path("/other")),
        ceiling=Workspace(cwd=Path("/project"), authorized_roots=(Path("/project"),)),
    )
    encoded = encode_inherited_workspace(workspace)
    # JSON-wireable: the payload crosses the spawn frame as plain JSON.
    assert set(encoded) == {"cwd", "authorized_roots", "ceiling"}
    assert json.loads(json.dumps(encoded)) == encoded
    assert decode_inherited_workspace(encoded) == workspace

    flat = Workspace(cwd=Path("/r"), authorized_roots=(Path("/r"),))
    assert decode_inherited_workspace(encode_inherited_workspace(flat)) == flat


@pytest.mark.parametrize(
    "payload",
    [
        "not-an-object",
        {},
        {"cwd": "/r", "authorized_roots": ["/r"]},
        {"cwd": "/r", "authorized_roots": [], "ceiling": None, "extra": 1},
        {"cwd": "", "authorized_roots": [], "ceiling": None},
        {"cwd": "/r", "authorized_roots": "/r", "ceiling": None},
        {"cwd": "/r", "authorized_roots": ["/r", 3], "ceiling": None},
        {"cwd": "/r", "authorized_roots": ["/r"], "ceiling": "not-a-ceiling"},
    ],
)
def test_inherited_workspace_codec_refuses_every_malformed_payload(
    payload: Any,
) -> None:
    """No repair, no fallback: a malformed value is a typed payload error."""
    with pytest.raises(ChildTranscriptPayloadError):
        decode_inherited_workspace(payload)


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_root_publish_payload_carries_the_parent_workspace(
    tmp_path: Path,
) -> None:
    """The root host's spawn payload names the workspace the child inherits.

    The in-process route hands the parent workspace over through
    ``child_runtime_policy``; the process route's counterpart is this payload
    value, so it must round-trip the exact workspace this root's tool
    manager holds (cwd, roots and ceiling), with no identity-key coupling:
    the workspace rides every fresh spawn.
    """
    loop = _publish_root(tmp_path)
    port = _SpawnAdapter()
    host = AgentChildHost(loop, spawn=port)
    try:
        handle = await host.publish(
            TaskArgs(task="Inspect the project", identity="coder"),
            InvokeContext(tool_call_id="task-1", session_id=loop.session_id),
            attempt_id="attempt-spawn-workspace",
            retained=False,
        )
    finally:
        await loop.aclose()

    assert handle.session_id == "child-session-1"
    payload = port.specs[0].payload
    assert INHERITED_WORKSPACE_PAYLOAD_KEY in payload
    assert decode_inherited_workspace(payload[INHERITED_WORKSPACE_PAYLOAD_KEY]) == (
        loop.tool_manager.workspace
    )


# ---------------------------------------------------------------------------
# F11: a process child inherits the parent's policy (strict code + loud gate)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_child_factory_decodes_the_parent_policy_and_stays_loud_on_legacy(
    tmp_path: Path,
) -> None:
    """The child's open hands the decoded parent policy to its loop request.

    F11's strict stance, matching F6's workspace stance: a payload a pre-F11
    peer would have opened with an EMPTY policy (no restrictions, no grants,
    no budgets, no agent config) answers the typed payload error at the
    child's open -- a restricted or budget-capped parent can never spawn an
    unrestricted, unbudgeted child.
    """
    root = _root(tmp_path)
    requests: list[Any] = []

    class _Process:
        async def open_root(self, request: Any) -> Any:
            requests.append(request)
            return SimpleNamespace(
                session_id="child-session-1",
                session_logger=SimpleNamespace(session_dir=None),
                messages=[],
                persist_empty_session=_persist_empty_session,
            )

    async def _persist_empty_session(*, attempt_id: str | None = None) -> None:
        return None

    factory = object.__new__(HarnessChildLoopFactory)
    factory._process = cast(Any, _Process())
    workspace = Workspace(
        cwd=Path("/project/root"), authorized_roots=(Path("/project"),)
    )
    policy_payload = {
        AGENT_CONFIG_PAYLOAD_KEY: AgentConfig(
            cwd="/project/root",
            max_turns=12,
            run_max_price=1.5,
            run_max_session_tokens=2048,
        ).model_dump(mode="json"),
        INHERITED_RESTRICTIONS_PAYLOAD_KEY: encode_source_restrictions((
            _SOURCE_RESTRICTION,
        )),
        INHERITED_MODE_RESTRICTIONS_PAYLOAD_KEY: encode_source_restrictions((
            _MODE_RESTRICTION,
        )),
        INHERITED_ROOT_GRANTS_PAYLOAD_KEY: ["/granted/one", "/granted/two"],
    }

    spec = child_spawn_spec(
        agent_id="coder:py",
        parent_session_id=_ROOT_SESSION_ID,
        root_session_dir=root,
        agent_name="coder-py",
        payload={
            "client_info": {"name": "tests", "version": "1.0"},
            INHERITED_WORKSPACE_PAYLOAD_KEY: encode_inherited_workspace(workspace),
            **policy_payload,
        },
        attempt_id="attempt-policy-1",
    )
    assert await factory.open(spawn_spec_to_wire(spec)) is not None
    request = requests[0]
    assert request.options == AgentConfig(
        cwd="/project/root",
        max_turns=12,
        run_max_price=1.5,
        run_max_session_tokens=2048,
    )
    assert request.inherited_restrictions == (_SOURCE_RESTRICTION,)
    assert request.inherited_mode_restrictions == (_MODE_RESTRICTION,)
    assert request.inherited_root_grants == (Path("/granted/one"), Path("/granted/two"))

    # The F11 legacy shape: a pre-F11 payload answers the typed open failure
    # naming every missing policy key -- never a silent empty policy.
    legacy_spec = child_spawn_spec(
        agent_id="coder:py",
        parent_session_id=_ROOT_SESSION_ID,
        root_session_dir=root,
        agent_name="coder-py",
        payload={
            "client_info": {"name": "tests", "version": "1.0"},
            INHERITED_WORKSPACE_PAYLOAD_KEY: encode_inherited_workspace(workspace),
        },
        attempt_id="attempt-policy-2",
    )
    with pytest.raises(ChildTranscriptPayloadError, match="parent-policy keys"):
        await factory.open(spawn_spec_to_wire(legacy_spec))


def test_source_restrictions_codec_round_trips_every_field() -> None:
    """The strict codec loses nothing a child's policy needs."""
    encoded = encode_source_restrictions((_SOURCE_RESTRICTION, _MODE_RESTRICTION))
    # JSON-wireable: the payload crosses the spawn frame as plain JSON.
    assert json.loads(json.dumps(encoded)) == encoded
    assert decode_source_restrictions(encoded) == (
        _SOURCE_RESTRICTION,
        _MODE_RESTRICTION,
    )
    assert decode_source_restrictions([]) == ()


@pytest.mark.parametrize(
    "payload",
    [
        "not-a-list",
        {},
        ["not-an-object"],
        [{}],
        [{**encode_source_restrictions((_SOURCE_RESTRICTION,))[0], "extra": 1}],
        [{**encode_source_restrictions((_SOURCE_RESTRICTION,))[0], "kind": "weird"}],
        [{**encode_source_restrictions((_SOURCE_RESTRICTION,))[0], "tools": "x"}],
        [
            {
                **encode_source_restrictions((_SOURCE_RESTRICTION,))[0],
                "tools": [{"tool_name": "bash"}],
            }
        ],
        [
            {
                **encode_source_restrictions((_SOURCE_RESTRICTION,))[0],
                "identity": {"owner": "not-a-uuid"},
            }
        ],
        [
            {
                **encode_source_restrictions((_SOURCE_RESTRICTION,))[0],
                "authorized_roots": [{"project": "/p", "roots": []}],
            }
        ],
        [
            {
                **encode_source_restrictions((_SOURCE_RESTRICTION,))[0],
                "dangerous_command_grants": [{"project": "/p"}],
            }
        ],
        [
            {
                **encode_source_restrictions((_SOURCE_RESTRICTION,))[0],
                "remote_exec_grants": [
                    {"project": "/p", "destinations": [{"user": "ci"}]}
                ],
            }
        ],
        [
            {
                **encode_source_restrictions((_SOURCE_RESTRICTION,))[0],
                "remote_exec_grants": [
                    {
                        "project": "/p",
                        "destinations": [{"user": "ci", "host": "h", "port": "22"}],
                    }
                ],
            }
        ],
        [
            {
                **encode_source_restrictions((_SOURCE_RESTRICTION,))[0],
                "content_fingerprint": 17,
            }
        ],
        [
            {
                **encode_source_restrictions((_SOURCE_RESTRICTION,))[0],
                "replaces_roots": "yes",
            }
        ],
    ],
)
def test_source_restrictions_codec_refuses_every_malformed_payload(
    payload: Any,
) -> None:
    """No repair, no fallback: a malformed value is a typed payload error."""
    with pytest.raises(ChildTranscriptPayloadError):
        decode_source_restrictions(payload)


def test_inherited_root_grants_codec_round_trips_and_refuses_malformed() -> None:
    """Grants ride as plain path strings, or the decode fails typed."""
    assert decode_inherited_root_grants(["/granted/one", "/granted/two"]) == (
        Path("/granted/one"),
        Path("/granted/two"),
    )
    for malformed in ("not-a-list", "{}", [1], [""], {"root": "/r"}, None):
        with pytest.raises(ChildTranscriptPayloadError):
            decode_inherited_root_grants(malformed)


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_root_publish_payload_carries_the_parent_policy(tmp_path: Path) -> None:
    """A restricted, budget-capped parent's spawn payload carries its policy.

    The payload values decode to exactly the restrictions, mode restrictions,
    dispatch root grants and budget values the in-process route's
    ``child_runtime_policy`` would hand an in-process child of this loop --
    the parent's policy, never an empty one.
    """
    loop = build_test_agent_loop(
        config=build_test_vibe_config(
            session_logging=SessionLoggingConfig(
                enabled=True,
                save_dir=str(tmp_path / "sessions"),
                session_prefix="session",
            )
        ),
        backend=FakeBackend([[mock_llm_chunk(content="ack")]]),
        inherited_restrictions=(_SOURCE_RESTRICTION,),
        inherited_mode_restrictions=(_MODE_RESTRICTION,),
        run_max_price=1.25,
        run_max_session_tokens=3456,
    )
    port = _SpawnAdapter()
    host = AgentChildHost(loop, spawn=port)
    try:
        handle = await host.publish(
            TaskArgs(task="Inspect the project", identity="coder"),
            InvokeContext(tool_call_id="task-1", session_id=loop.session_id),
            attempt_id="attempt-spawn-policy",
            retained=False,
        )
    finally:
        await loop.aclose()

    assert handle.session_id == "child-session-1"
    payload = port.specs[0].payload
    expected = loop.child_runtime_policy
    assert (
        decode_source_restrictions(payload[INHERITED_RESTRICTIONS_PAYLOAD_KEY])
        == expected.inherited_restrictions
    )
    assert (
        decode_source_restrictions(payload[INHERITED_MODE_RESTRICTIONS_PAYLOAD_KEY])
        == expected.inherited_mode_restrictions
    )
    assert (
        decode_inherited_root_grants(payload[INHERITED_ROOT_GRANTS_PAYLOAD_KEY])
        == expected.inherited_root_grants
    )
    options = AgentConfig.model_validate(payload[AGENT_CONFIG_PAYLOAD_KEY])
    assert options.run_max_price == expected.run_max_price
    assert options.run_max_session_tokens == expected.run_max_session_tokens
    assert options.cwd == str(loop.cwd)
    assert json.loads(json.dumps(payload)) == payload


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_the_subagent_blueprint_builds_the_child_policy_from_the_spawn_shape(
    tmp_path: Path,
) -> None:
    """The process child's BUILT policy carries the spawn shape field by field.

    The blueprint is the process child's loop constructor: the inherited
    restrictions, mode restrictions and root grants land in the built
    policy, and the run-budget default arms the same way ``_create_like``
    arms an in-process child (the explicit value wins, the [subagents]
    config default fills a None).
    """
    workspace = Workspace(
        cwd=Path("/project/root"), authorized_roots=(Path("/project"),)
    )
    blueprint = _runtime._RootRuntimeBlueprint(
        config_orchestrator=FakeConfigOrchestrator(
            build_test_vibe_config(
                session_logging=SessionLoggingConfig(
                    enabled=True, save_dir=str(tmp_path)
                ),
                subagents=SubagentsConfig(
                    run_max_price=0.75, run_max_session_tokens=2048
                ),
            )
        ),
        harness_files=HarnessFilesManager(),
        options=AgentConfig(cwd="/project/root", run_max_price=2.0),
        client_info=ClientInfo(name="tests", version="1.0"),
        client_capabilities=ClientCapabilities(),
        hook_config_result=HookConfigResult(hooks=[], issues=[]),
        cache_store=FileSystemCacheStore(tmp_path / "cache.toml"),
    )
    loop = blueprint.build(
        parent_session_id=_ROOT_SESSION_ID,
        is_subagent=True,
        inherited_workspace=workspace,
        inherited_restrictions=(_SOURCE_RESTRICTION,),
        inherited_mode_restrictions=(_MODE_RESTRICTION,),
        inherited_root_grants=(Path("/granted/one"),),
    )
    try:
        policy = loop.runtime_policy
        assert policy.inherited_restrictions == (_SOURCE_RESTRICTION,)
        assert policy.inherited_mode_restrictions == (_MODE_RESTRICTION,)
        assert policy.inherited_workspace == workspace
        assert policy.enable_streaming is False
        assert loop._dispatch_root_grants == (Path("/granted/one"),)
        # B2c, process-spawn counterpart: the explicit parent value wins, the
        # [subagents] config default fills the unset one (both middleware
        # inputs arm: RunBudgetMiddleware reads these exact loop fields).
        assert policy.run_max_price == 2.0
        assert policy.run_max_session_tokens == 2048
    finally:
        await loop.aclose()

    # The ROOT shape stays untouched: no [subagents] fill (the config default
    # stays out of a root's session budgets) and no inherited policy.
    fresh = blueprint.build(parent_session_id=None)
    try:
        assert fresh.runtime_policy.run_max_price == 2.0
        assert fresh.runtime_policy.run_max_session_tokens is None
        assert fresh.runtime_policy.inherited_restrictions == ()
        assert fresh.runtime_policy.inherited_mode_restrictions == ()
    finally:
        await fresh.aclose()


# ---------------------------------------------------------------------------
# The root records a child result receipt exactly once
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_child_result_receipts_are_idempotent_and_handed_over_once(
    tmp_path: Path,
) -> None:
    root = _root(tmp_path)
    store = _store(root)
    store.record_start(_start())
    store.record_result(_result())

    async def _notify_child(method: str, params: Any) -> None:
        return None

    async def _deliver_callback(callback_id: str, params: Any) -> None:
        return None

    registry = SessionRuntimeRegistry(
        notify_child=cast("Any", _notify_child),
        deliver_callback=cast("Any", _deliver_callback),
        event_watermark=lambda _session_id: 0,
    )
    registry._root = cast(
        "Any",
        SimpleNamespace(
            agent_loop=SimpleNamespace(
                session_id=_ROOT_SESSION_ID,
                _session_generation=1,
                session_logger=SimpleNamespace(session_dir=root),
            )
        ),
    )
    try:
        assert await registry.record_child_result_receipt("coder:py", "coder:py:turn-1")
        assert not await registry.record_child_result_receipt(
            "coder:py", "coder:py:turn-1"
        )
        scanned = await registry.scan_child_transcripts()
        assert [item.unconsumed for item in scanned] == [False]
        assert scanned[0].result_id == "coder:py:turn-1"
        assert (
            await registry.reconcile_child_transcript("coder:py", store.child_dir)
        ).outcome is ChildTranscriptOutcome.COMPLETED
    finally:
        await registry.close()
