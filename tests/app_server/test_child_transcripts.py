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
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from chartreux.app_server._agent_transcript import read_live_agent_transcript
from chartreux.app_server._sessions import SessionRuntimeRegistry
from chartreux.app_server._subagent_recovery import (
    AGENT_TRANSCRIPT_RECONCILE_METHOD,
    ChildTranscriptReconciler,
)
from chartreux.app_server._subagent_transcript import (
    CHILD_RESULT_FILENAME,
    CHILD_START_MARKER_FILENAME,
    ChildResultRecord,
    ChildStartMarker,
    ChildTranscriptOutcome,
    ChildTranscriptPathError,
    ChildTranscriptStore,
    child_result_digest,
    child_transcript_dir_name,
    child_transcript_dirs,
    child_transcript_logging,
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
from chartreux.app_server.registry_manifest import (
    AgentRoute,
    LiveAgentExitedError,
    RegistryManifest,
)
from chartreux.app_server.subagent_ports import (
    SubagentSpawnSpec,
    child_spawn_spec,
    spawn_spec_to_wire,
)
from chartreux.core.llm_models import LLMMessage, Role
from chartreux.core.session.session_logger import SessionLogger

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


@pytest.mark.asyncio
async def test_child_factory_binds_the_transcript_to_the_root_agents_subtree(
    tmp_path: Path,
) -> None:
    root = _root(tmp_path)
    requests: list[Any] = []
    loop = SimpleNamespace(
        session_id="child-session-1", session_logger=SimpleNamespace(session_dir=None)
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
        payload={"client_info": {"name": "tests", "version": "1.0"}},
    )

    assert await factory.open(spawn_spec_to_wire(spec)) is loop
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
