"""Root death, session switch, orphan reaping, and fresh respawn (WP7 [7C]).

The lifecycle rules under test: root-process death is typed (pipe EOF or a
missed liveness heartbeat on the root pipe), a session switch closes the old
tree completely before it opens the new one, a child whose parent session is
gone is reaped and reconciled from its durable markers but NEVER replayed, and
a respawned child is a fresh incarnation that never resumes the earlier one.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import suppress
from dataclasses import replace
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from chartreux.app_server._child_runtime_lifecycle import (
    TreeNotReapedError,
    TreeShutdownReport,
    TreeShutdownStatus,
)
from chartreux.app_server._dispatch import RequestFailure
from chartreux.app_server._session_runtime_impl import ProcessRootReplace
from chartreux.app_server._sessions import SessionRuntimeRegistry
from chartreux.app_server._subagent_transcript import (
    ChildStartMarker,
    ChildTranscriptOutcome,
    ChildTranscriptStore,
    child_transcript_dir_name,
    child_transcripts_root,
)
from chartreux.app_server.child_agent_process import ProcessSubagentSpawn
from chartreux.app_server.process_bootstrap import (
    DEFAULT_SHUTDOWN_TIMEOUT,
    ReadyFrame,
    SpawnedProcess,
    encode_ready_frame,
    spawn_process,
)
from chartreux.app_server.protocol import ProtocolErrorCode
from chartreux.app_server.root_agent_process import (
    DEFAULT_ROOT_HEARTBEAT_INTERVAL,
    RootAgentExitedError,
    RootAgentOpTimeoutError,
    RootAgentProcess,
    RootFailureSignature,
)
from chartreux.app_server.subagent_ports import (
    SubagentAdmissionClosedError,
    SubagentSpawnDeniedError,
    child_spawn_spec,
)
from chartreux.core.hooks.config import HookConfig
from chartreux.core.hooks.executor import HookExecutor
from chartreux.core.hooks.models import HookType, PostAgentInvocation
import chartreux.core.job_launcher as job_launcher
from chartreux.core.subagents import TaskArgs
from chartreux.core.tools.base import InvokeContext
from chartreux.core.utils import async_subprocess as job_processes
from chartreux.core.utils.async_subprocess import (
    PROCESS_GROUP_INCONCLUSIVE,
    PROCESS_GROUP_REAPED,
    PROCESS_GROUP_STALE_REFUSED,
    RecordedJobGroup,
    settle_recorded_job_group,
)
from tests.app_server._subagent_test_support import (
    _ManualClock,
    _reaper_registry,
    _record,
)

_READY_LITERAL = encode_ready_frame().decode().strip()

_SLEEPING_CHILD = (
    "import sys, time\n"
    f"sys.stdout.write({_READY_LITERAL!r} + chr(10))\n"
    "sys.stdout.flush()\n"
    "time.sleep(60)\n"
)


class _JumpClock:
    """Every read after the first advances by ``step`` (an expired bound)."""

    def __init__(self, step: float) -> None:
        self.step = step
        self.value = 0.0
        self.calls = 0

    def __call__(self) -> float:
        self.calls += 1
        if self.calls > 1:
            self.value += self.step
        return self.value


class _ProcessDouble:
    """A process handle double whose liveness the test controls."""

    stdin = None
    stdout = None

    def __init__(self, exit_code: int | None) -> None:
        self.exit_code = exit_code
        self.signals: list[str] = []

    def poll(self) -> int | None:
        return self.exit_code

    def terminate(self) -> None:
        self.signals.append("terminate")

    def kill(self) -> None:
        self.signals.append("kill")


class _SilentTransport:
    """A transport whose message stream never ends: only the heartbeat can tell."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.closed = False
        self._never = asyncio.Event()

    async def send(self, message: dict[str, Any]) -> None:
        self.sent.append(message)

    async def messages(self) -> AsyncIterator[dict[str, Any]]:
        await self._never.wait()
        if False:  # pragma: no cover - an async generator must contain a yield
            yield {}

    async def close(self) -> None:
        self.closed = True


class _EndingTransport(_SilentTransport):
    """A transport whose message stream ends at once while the process lives."""

    async def messages(self) -> AsyncIterator[dict[str, Any]]:
        if False:  # pragma: no cover - an async generator must contain a yield
            yield {}


class _PushTransport(_SilentTransport):
    """A transport the test feeds, so a reply can interrupt a timeout streak."""

    def __init__(self) -> None:
        super().__init__()
        self._frames: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

    def push(self, message: dict[str, Any]) -> None:
        self._frames.put_nowait(message)

    async def messages(self) -> AsyncIterator[dict[str, Any]]:
        while True:
            frame = await self._frames.get()
            if frame is None:
                return
            yield frame


def _dead_root(
    *, exit_code: int | None, heartbeat_interval: float = 0.01
) -> RootAgentProcess:
    transport = _SilentTransport()
    spawned = SpawnedProcess(
        process=cast("subprocess.Popen[bytes]", _ProcessDouble(exit_code)),
        transport=transport,  # type: ignore[arg-type] - transport double
        ready=ReadyFrame(protocol_version=1),
    )
    return RootAgentProcess(spawned, heartbeat_interval=heartbeat_interval)


def _reap(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        process.kill()
        process.wait(timeout=5)


class _ClosingHandle:
    """A spawned-subagent double that records when it is reaped."""

    agent_id = "agent-1"
    completion_source: Any = None

    def __init__(self, order: list[str]) -> None:
        self._order = order

    async def close(self) -> None:
        self._order.append("close")


class _ReplaceRecorder:
    """A root process double that records the rebind it was asked for."""

    def __init__(self, order: list[str], reply: object) -> None:
        self._order = order
        self._reply = reply
        self.calls: list[tuple[str | None, bool]] = []

    async def replace(
        self, *, session_id: str | None = None, continue_latest: bool = False
    ) -> Any:
        self._order.append("rebind")
        self.calls.append((session_id, continue_latest))
        return self._reply


def _orphan_child_dir(root_session_dir: Path) -> Path:
    child_dir = child_transcripts_root(root_session_dir) / child_transcript_dir_name(
        "coder", "session-agent-1", timestamp="20260101_000000"
    )
    child_dir.mkdir(parents=True)
    ChildTranscriptStore(child_dir, root_session_dir=root_session_dir).record_start(
        ChildStartMarker(
            agent_id="agent-1",
            session_id="session-agent-1",
            run_id="run-1",
            started_at="2026-01-01T00:00:00+00:00",
        )
    )
    return child_dir


def test_the_root_heartbeat_is_the_planned_two_seconds() -> None:
    assert DEFAULT_ROOT_HEARTBEAT_INTERVAL == 2.0


@pytest.mark.asyncio
async def test_a_replace_reaps_the_old_cohort_before_it_rebinds() -> None:
    order: list[str] = []
    reply = object()
    process = _ReplaceRecorder(order, reply)

    async def reap() -> object:
        order.append("reap")
        return object()

    transition = ProcessRootReplace(
        process=cast("RootAgentProcess", process), reap=reap
    )

    assert await transition.replace(session_id="session-7") is reply
    assert order == ["reap", "rebind"]
    assert process.calls == [("session-7", False)]


@pytest.mark.asyncio
async def test_a_cohort_that_could_not_be_reaped_never_sends_the_rebind() -> None:
    """The failed reaping ends the transition BEFORE a frame is sent."""
    order: list[str] = []
    process = _ReplaceRecorder(order, object())

    async def reap() -> object:
        raise TreeNotReapedError(
            TreeShutdownReport(status=TreeShutdownStatus.TIMED_OUT)
        )

    transition = ProcessRootReplace(
        process=cast("RootAgentProcess", process), reap=reap
    )

    with pytest.raises(RequestFailure) as refusal:
        await transition.replace(continue_latest=True)

    assert refusal.value.code is ProtocolErrorCode.CONFLICT
    assert "Retry the cohort reaper" in str(refusal.value)
    assert "shutdown" in str(refusal.value)
    assert order == []
    assert process.calls == []


@pytest.mark.asyncio
async def test_a_root_pipe_eof_is_a_typed_root_death() -> None:
    spawned = await spawn_process([sys.executable, "-c", _SLEEPING_CHILD])
    process = RootAgentProcess(spawned, heartbeat_interval=0.05)
    try:
        assert process.alive
        spawned.process.kill()
        spawned.process.wait(timeout=5)
        death = await asyncio.wait_for(process.wait_for_death(), timeout=10)
        assert process.alive is False
        assert death.exit_code is not None
        with pytest.raises(RootAgentExitedError):
            await process.read_page("agent-1")
    finally:
        await process.close()
        _reap(spawned.process)


@pytest.mark.asyncio
async def test_a_missed_heartbeat_detects_a_dead_root_whose_pipe_stays_open() -> None:
    process = _dead_root(exit_code=9)
    try:
        death = await asyncio.wait_for(process.wait_for_death(), timeout=5)
        assert death.exit_code == 9
        assert death.reason == "the root agent process exited"
        assert process.alive is False
    finally:
        await process.close()


@pytest.mark.asyncio
async def test_a_request_in_flight_fails_typed_when_the_root_dies() -> None:
    process = _dead_root(exit_code=None)
    try:
        request = asyncio.create_task(process.read_page("agent-1"))
        await asyncio.sleep(0)
        # The root process is gone while its pipe stays open: the heartbeat
        # (not EOF) is what declares death here.
        cast("_ProcessDouble", process.process).exit_code = 3
        await asyncio.wait_for(process.wait_for_death(), timeout=5)
        with pytest.raises(RootAgentExitedError):
            await request
    finally:
        await process.close()


def _controllable_root(
    transport: _SilentTransport, *, exit_code: int | None, timeout: float = 0.05
) -> RootAgentProcess:
    """A root whose transport the test feeds and whose process state it sets."""
    spawned = SpawnedProcess(
        process=cast("subprocess.Popen[bytes]", _ProcessDouble(exit_code)),
        transport=transport,  # type: ignore[arg-type] - transport double
        ready=ReadyFrame(protocol_version=1),
    )
    return RootAgentProcess(spawned, heartbeat_interval=0.01, request_timeout=timeout)


@pytest.mark.asyncio
async def test_the_death_watcher_names_its_death_receipt_signature() -> None:
    process = _dead_root(exit_code=9)
    try:
        death = await asyncio.wait_for(process.wait_for_death(), timeout=5)
    finally:
        await process.close()

    assert death.signature is RootFailureSignature.DEATH_RECEIPT
    assert death.diagnostic == "death-receipt: the root agent process exited"
    assert any(
        "death-receipt" in diagnostic.message() for diagnostic in process.diagnostics
    )


@pytest.mark.asyncio
async def test_a_pipe_that_closes_with_the_root_alive_names_its_own_signature() -> None:
    process = _controllable_root(_EndingTransport(), exit_code=None)
    try:
        death = await asyncio.wait_for(process.wait_for_death(), timeout=5)
    finally:
        await process.close()

    # The process never exited: the immediate pipe signature is what a user
    # must see, never the poll-interval death receipt.
    assert death.signature is RootFailureSignature.PIPE_EOF
    assert death.diagnostic.startswith("pipe-eof-while-alive:")


@pytest.mark.asyncio
async def test_a_single_wedged_op_names_the_op_deadline_signature() -> None:
    process = _controllable_root(_SilentTransport(), exit_code=None)
    try:
        with pytest.raises(RootAgentOpTimeoutError) as refusal:
            await process.read_page("agent-1")
    finally:
        await process.close()

    assert refusal.value.signature is RootFailureSignature.OP_DEADLINE
    assert refusal.value.occurrences == 1
    assert str(refusal.value).startswith("op-deadline-timeout:")


@pytest.mark.asyncio
async def test_a_repeated_timeout_names_the_idle_wedge_and_a_reply_ends_it() -> None:
    transport = _PushTransport()
    process = _controllable_root(transport, exit_code=None)
    try:
        with pytest.raises(RootAgentOpTimeoutError) as first:
            await process.read_page("agent-1")
        assert first.value.signature is RootFailureSignature.OP_DEADLINE

        # Nothing answered between the two timeouts: the wedge is now the
        # observed signature rather than a single slow operation.
        with pytest.raises(RootAgentOpTimeoutError) as second:
            await process.read_page("agent-2")
        assert second.value.signature is RootFailureSignature.IDLE_WEDGE
        assert second.value.occurrences == 2

        # Any answer proves the root still serves and restarts the streak.
        answering = asyncio.create_task(process.read_page("agent-3"))
        await asyncio.sleep(0.01)
        transport.push({"jsonrpc": "2.0", "id": process.last_request_id, "result": {}})
        with suppress(Exception):
            await answering

        with pytest.raises(RootAgentOpTimeoutError) as third:
            await process.read_page("agent-4")
        assert third.value.signature is RootFailureSignature.OP_DEADLINE
        assert third.value.occurrences == 1
    finally:
        await process.close()


@pytest.mark.asyncio
async def test_a_deliberate_close_is_never_reported_as_root_death() -> None:
    spawned = await spawn_process([sys.executable, "-c", _SLEEPING_CHILD])
    process = RootAgentProcess(spawned, heartbeat_interval=0.05)
    try:
        await process.close()
        assert process.death is None
        assert process.alive is False
    finally:
        _reap(spawned.process)


@pytest.mark.asyncio
async def test_root_death_disposes_of_the_whole_child_tree() -> None:
    clock = _ManualClock(0)
    registry, _root = _reaper_registry(clock)
    record = _record("agent-1", clock=clock)
    registry._agent_records[record.agent_id] = record
    registry._children[record.session_id] = record.runtime

    report = await registry.dispose_tree_after_root_death()

    assert report.complete
    assert report.drain is not None
    assert report.drain.complete
    cast("AsyncMock", record.runtime.close).assert_awaited()
    assert not registry._children


@pytest.mark.asyncio
async def test_a_session_switch_closes_the_old_tree_before_it_reopens_admission() -> (
    None
):
    clock = _ManualClock(0)
    registry, _root = _reaper_registry(clock)
    record = _record("agent-1", clock=clock)
    registry._agent_records[record.agent_id] = record
    registry._children[record.session_id] = record.runtime
    entered = asyncio.Event()
    gate = asyncio.Event()

    async def gated_close() -> None:
        entered.set()
        await gate.wait()

    record.runtime.close = AsyncMock(side_effect=gated_close)
    switch = asyncio.create_task(registry.close_tree_for_switch())
    try:
        await asyncio.wait_for(entered.wait(), timeout=1)
        # The old tree is still closing: no new child may start beside it.
        with pytest.raises(SubagentAdmissionClosedError):
            registry._reserve_admission(
                TaskArgs(task="late work", identity="coder", background=True),
                InvokeContext(tool_call_id="late", session_id="root"),
            )
    finally:
        gate.set()
        report = await asyncio.wait_for(switch, timeout=2)

    assert report.complete
    cast("AsyncMock", record.runtime.close).assert_awaited()
    assert registry._admission_closed


@pytest.mark.asyncio
async def test_a_tree_that_could_not_be_reaped_refuses_a_switch() -> None:
    registry = SessionRuntimeRegistry(
        AsyncMock(),
        AsyncMock(),
        lambda _session_id: 0,
        clock=_JumpClock(100.0),
        shutdown_timeout=DEFAULT_SHUTDOWN_TIMEOUT,
    )

    with pytest.raises(TreeNotReapedError) as switch_error:
        await registry.close_tree_for_switch()
    with pytest.raises(TreeNotReapedError) as death_error:
        await registry.dispose_tree_after_root_death()

    assert switch_error.value.report.timed_out
    assert death_error.value.report.timed_out
    assert "no new tree starts before the old one is reaped" in str(switch_error.value)


@pytest.mark.asyncio
async def test_an_orphaned_child_is_reaped_and_never_replayed(tmp_path: Path) -> None:
    clock = _ManualClock(0)
    registry, root = _reaper_registry(clock)
    root_session_dir = tmp_path / "session-root"
    root_session_dir.mkdir()
    root.agent_loop.session_logger.session_dir = root_session_dir
    child_dir = _orphan_child_dir(root_session_dir)
    record = _record("agent-1", clock=clock)
    record.runtime.agent_loop.session_id = "session-agent-1"
    record.runtime.agent_loop.parent_session_id = "old-root"
    record.runtime.agent_loop.session_logger.session_dir = child_dir
    registry._agent_records[record.agent_id] = record
    registry._children[record.session_id] = record.runtime

    report = await registry.reap_orphaned_children()

    assert report.complete
    assert report.reaped == 1
    assert report.reconciled == 1
    orphan = report.orphans[0]
    assert orphan.agent_id == "agent-1"
    assert orphan.reconciliation is not None
    assert orphan.reconciliation.child_dir == child_dir
    assert (
        orphan.reconciliation.outcome is ChildTranscriptOutcome.STARTED_OUTCOME_UNKNOWN
    )
    assert orphan.reconciliation.denial is not None
    assert "will not be replayed" in orphan.reconciliation.denial
    cast("AsyncMock", record.runtime.close).assert_awaited_once()
    assert not registry._children
    assert not registry._agent_records


@pytest.mark.asyncio
async def test_a_child_of_the_live_session_is_not_an_orphan() -> None:
    clock = _ManualClock(0)
    registry, _root = _reaper_registry(clock)
    record = _record("agent-1", clock=clock)
    record.runtime.agent_loop.parent_session_id = "root"
    registry._agent_records[record.agent_id] = record
    registry._children[record.session_id] = record.runtime

    report = await registry.reap_orphaned_children()

    assert report.orphans == ()
    assert registry._children[record.session_id] is record.runtime
    cast("AsyncMock", record.runtime.close).assert_not_awaited()


@pytest.mark.asyncio
async def test_an_orphan_is_reconciled_under_the_session_that_owned_it(
    tmp_path: Path,
) -> None:
    """A switch leaves the orphan's transcript under the OLD root session."""
    clock = _ManualClock(0)
    registry, root = _reaper_registry(clock)
    old_root_session_dir = tmp_path / "session-old"
    old_root_session_dir.mkdir()
    new_root_session_dir = tmp_path / "session-new"
    new_root_session_dir.mkdir()
    root.agent_loop.session_logger.session_dir = new_root_session_dir
    child_dir = _orphan_child_dir(old_root_session_dir)
    record = _record("agent-1", clock=clock)
    record.runtime.agent_loop.session_id = "session-agent-1"
    record.runtime.agent_loop.parent_session_id = "old-root"
    record.runtime.agent_loop.session_logger.session_dir = child_dir
    registry._agent_records[record.agent_id] = record
    registry._children[record.session_id] = record.runtime

    report = await registry.reap_orphaned_children()

    assert report.complete
    assert report.reconciled == 1
    orphan = report.orphans[0]
    assert orphan.reconciliation is not None
    assert orphan.reconciliation.child_dir == child_dir
    assert (
        orphan.reconciliation.outcome is ChildTranscriptOutcome.STARTED_OUTCOME_UNKNOWN
    )


@pytest.mark.asyncio
async def test_a_respawn_reaps_the_previous_child_before_it_spawns(
    tmp_path: Path,
) -> None:
    port = ProcessSubagentSpawn(env={})
    order: list[str] = []
    spawned = object()

    async def fake_spawn(spec: object) -> object:
        order.append("spawn")
        return spawned

    port.spawn = fake_spawn  # type: ignore[method-assign]
    spec = child_spawn_spec(
        agent_id="agent-1", parent_session_id="root", root_session_dir=tmp_path
    )

    result = await port.respawn(spec, previous=_ClosingHandle(order))

    assert result is spawned
    assert order == ["close", "spawn"]


@pytest.mark.asyncio
async def test_a_respawn_never_resumes_a_named_child_session(tmp_path: Path) -> None:
    port = ProcessSubagentSpawn(env={})
    port.spawn = AsyncMock()  # type: ignore[method-assign]
    spec = replace(
        child_spawn_spec(
            agent_id="agent-1", parent_session_id="root", root_session_dir=tmp_path
        ),
        session_dir=str(tmp_path / "child-session"),
    )

    with pytest.raises(SubagentSpawnDeniedError, match="never resumed"):
        await port.respawn(spec, previous=_ClosingHandle([]))

    port.spawn.assert_not_awaited()


# ---------------------------------------------------------------------------
# Package 23: the background-job launch and cleanup primitives.
#
# A job's leader guarantee is kernel-enforced: the launcher records its pid and
# /proc start time durably BEFORE it arms PR_SET_PDEATHSIG, re-checks its parent
# before it execs, and a death cleanup reconstructs that record from disk and
# re-establishes the ownership predicate before EACH signal it sends -- an
# un-re-establishable predicate is a typed INCONCLUSIVE outcome, never a signal.
# ---------------------------------------------------------------------------

_JOB_TARGET = (
    "import os, pathlib, sys, time\n"
    "pathlib.Path(sys.argv[1]).write_text(str(os.getpid()))\n"
    "time.sleep(60)\n"
)


def _launch_plan(tmp_path: Path, *, session_id: str = "session-jobs") -> Any:
    return job_launcher.plan_job_launch(
        jobs_dir=tmp_path / "session" / "jobs",
        session_id=session_id,
        parent_pid=os.getpid(),
    )


def _wait_for(predicate: Any, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("the condition never held")


def _stop_leader(pid: int | None) -> None:
    if pid is None:
        return
    with suppress(ProcessLookupError, PermissionError):
        os.killpg(pid, signal.SIGKILL)


def _recorded_group(**overrides: Any) -> RecordedJobGroup:
    record = job_launcher.JobGroupRecord(
        job_id="job-1", pid=123, start_time=100, session_id="session-jobs"
    )
    return RecordedJobGroup(replace(record, **overrides))


def test_the_launch_argv_wraps_the_target_and_creates_the_jobs_directory(
    tmp_path: Path,
) -> None:
    plan = _launch_plan(tmp_path)

    argv = plan.wrap(["chosen-executor", "--flag"])

    assert argv[:2] == [sys.executable, str(Path(job_launcher.__file__).resolve())]
    assert argv[2] == str(plan.jobs_dir)
    assert argv[3].startswith("job-")
    assert (argv[4], argv[5], argv[6]) == (str(os.getpid()), "session-jobs", "--")
    assert argv[7:] == ["chosen-executor", "--flag"]
    assert plan.jobs_dir.is_dir()
    assert plan.owner_marker == {job_launcher.JOB_OWNER_MARKER_ENV: "session-jobs"}


def test_the_launcher_records_its_leader_before_the_exec(tmp_path: Path) -> None:
    plan = _launch_plan(tmp_path)
    marker = tmp_path / "leader-pid"
    argv = plan.argv_for("job-1", [sys.executable, "-c", _JOB_TARGET, str(marker)])

    process = subprocess.Popen(argv, start_new_session=True)
    record = None
    try:
        _wait_for(marker.exists)
        record = job_launcher.read_job_group_record(plan.record_path("job-1"))
        assert record is not None
        assert (record.job_id, record.session_id) == ("job-1", "session-jobs")
        # The record names the launcher, which the exec did not replace: the
        # leader identity it carries still describes the running job.
        assert record.pid == process.pid
        assert int(marker.read_text()) == process.pid
        identity = job_processes._process_identity(record.pid)
        assert identity is not None and identity[0] == record.start_time
    finally:
        _stop_leader(process.pid)
        process.wait()


def test_the_launcher_refuses_an_already_dead_root_before_the_exec(
    tmp_path: Path,
) -> None:
    jobs_dir = tmp_path / "jobs"
    marker = tmp_path / "leader-pid"
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait()

    completed = subprocess.run(
        [
            sys.executable,
            str(Path(job_launcher.__file__).resolve()),
            str(jobs_dir),
            "job-dead-root",
            str(gone.pid),
            "session-jobs",
            "--",
            sys.executable,
            "-c",
            _JOB_TARGET,
            str(marker),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == job_launcher.EXIT_PARENT_GONE
    assert "the root already died" in completed.stderr
    # Nothing was exec'd -- and the record that precedes the arming step is
    # already durable, which is what lets ownership be reconstructed later.
    assert not marker.exists()
    assert (
        job_launcher.read_job_group_record(jobs_dir / "job-dead-root.pgid") is not None
    )


def test_the_leader_dies_with_the_root_that_spawned_it(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    marker = tmp_path / "leader-pid"
    helper = "\n".join([
        "import os, subprocess, sys, time",
        "shim, jobs, job_id, session, marker, target = sys.argv[1:7]",
        "subprocess.Popen([sys.executable, shim, jobs, job_id, "
        "str(os.getpid()), session, '--', sys.executable, '-c', target, marker], "
        "start_new_session=True)",
        "time.sleep(60)",
    ])
    root = subprocess.Popen([
        sys.executable,
        "-c",
        helper,
        str(Path(job_launcher.__file__).resolve()),
        str(jobs_dir),
        "job-death",
        "session-jobs",
        str(marker),
        _JOB_TARGET,
    ])
    leader: int | None = None
    try:
        _wait_for(marker.exists)
        record = job_launcher.read_job_group_record(jobs_dir / "job-death.pgid")
        assert record is not None
        leader = record.pid
        assert job_processes._process_identity(leader) is not None

        root.kill()
        root.wait()

        # PR_SET_PDEATHSIG is the kernel's: the leader goes with its root.
        _wait_for(
            lambda: job_processes._process_identity(cast("int", leader)) is None,
            timeout=3.0,
        )
    finally:
        root.kill()
        root.wait()
        _stop_leader(leader)


def test_a_logging_disabled_session_refuses_job_admission() -> None:
    with pytest.raises(job_launcher.JobLaunchRefused) as caught:
        job_launcher.plan_job_launch(
            jobs_dir=None, session_id="session-jobs", parent_pid=os.getpid()
        )

    assert caught.value.reason == job_launcher.JOB_LAUNCH_LOGGING_DISABLED
    assert "session_logging.enabled" in str(caught.value)


def test_a_host_without_a_parent_death_signal_refuses_job_admission(
    tmp_path: Path,
) -> None:
    with pytest.raises(job_launcher.JobLaunchRefused) as caught:
        job_launcher.plan_job_launch(
            jobs_dir=tmp_path / "jobs",
            session_id="session-jobs",
            parent_pid=os.getpid(),
            platform="darwin",
        )

    assert caught.value.reason == job_launcher.JOB_LAUNCH_PLATFORM_UNSUPPORTED
    assert not (tmp_path / "jobs").exists()


def test_the_shim_aborts_a_launch_whose_jobs_directory_is_unusable(
    tmp_path: Path,
) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")

    refused = job_launcher.main([
        str(blocker / "jobs"),
        "job-1",
        str(os.getpid()),
        "session-jobs",
        "--",
        "true",
    ])

    assert refused == job_launcher.EXIT_JOBS_DIR_UNUSABLE


def test_the_shim_aborts_a_launch_whose_record_cannot_be_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unwritable(*args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(job_launcher, "write_job_group_record", unwritable)

    refused = job_launcher.main([
        str(tmp_path / "jobs"),
        "job-1",
        str(os.getpid()),
        "session-jobs",
        "--",
        "true",
    ])

    assert refused == job_launcher.EXIT_RECORD_UNWRITABLE


def test_the_shim_refuses_a_non_linux_host_before_it_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        job_launcher, "supports_parent_death_signal", lambda platform=None: False
    )

    refused = job_launcher.main([
        str(tmp_path / "jobs"),
        "job-1",
        str(os.getpid()),
        "session-jobs",
        "--",
        "true",
    ])

    assert refused == job_launcher.EXIT_PLATFORM_UNSUPPORTED
    assert not (tmp_path / "jobs").exists()


@pytest.mark.asyncio
async def test_a_launched_hook_keeps_its_stdin_pipe_and_records_its_group(
    tmp_path: Path,
) -> None:
    plan = _launch_plan(tmp_path)
    executor = HookExecutor(cwd=tmp_path, launch=plan)
    hook = HookConfig(
        name="echo-stdin",
        type=HookType.POST_AGENT,
        command=f'{sys.executable} -c "import sys; sys.stdout.write(sys.stdin.read())"',
        timeout=30.0,
    )

    result = await executor.run(
        hook,
        PostAgentInvocation(
            session_id="hook-session", transcript_path="", cwd=str(tmp_path)
        ),
    )

    # The launcher only prefixes the argv and execs into it, so the hook still
    # receives the invocation on the stdin pipe it was given.
    assert result.exit_code == 0
    assert '"session_id":"hook-session"' in result.stdout
    records = list(plan.jobs_dir.glob(f"*{job_launcher.JOB_GROUP_RECORD_SUFFIX}"))
    assert len(records) == 1
    record = job_launcher.read_job_group_record(records[0])
    assert record is not None
    assert record.session_id == "session-jobs"


@pytest.mark.asyncio
async def test_a_group_whose_recorded_identity_is_stale_is_never_signaled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    group = _recorded_group()
    monkeypatch.setattr(
        job_processes, "_process_identity", lambda pid: (999, 123, 123, "S")
    )
    monkeypatch.setattr(os, "listdir", lambda path: ["123"])
    signals: list[int] = []
    monkeypatch.setattr(os, "killpg", lambda pgid, sig: signals.append(sig))

    settlement = await settle_recorded_job_group(group, grace=0, kill_timeout=0)

    assert settlement.outcome == PROCESS_GROUP_STALE_REFUSED
    assert not settlement.reaped
    assert signals == []


@pytest.mark.asyncio
async def test_the_ownership_predicate_is_reestablished_before_the_kill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    group = _recorded_group()
    state = {"identity": (100, 123, 123, "S")}
    monkeypatch.setattr(
        job_processes, "_process_identity", lambda pid: state["identity"]
    )
    monkeypatch.setattr(os, "listdir", lambda path: ["123"])
    signals: list[int] = []

    def record_signal(pgid: int, sig: int) -> None:
        signals.append(sig)
        # The group is recycled right after the term signal: the escalation
        # must re-establish ownership before it sends anything else.
        state["identity"] = (999, 123, 123, "S")

    monkeypatch.setattr(os, "killpg", record_signal)

    settlement = await settle_recorded_job_group(group, grace=0, kill_timeout=0)

    assert signals == [signal.SIGTERM]
    assert settlement.outcome == PROCESS_GROUP_STALE_REFUSED
    assert "kill signal" in settlement.detail


@pytest.mark.asyncio
async def test_an_unestablishable_predicate_settles_inconclusive_without_a_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    group = _recorded_group()
    identities: dict[int, tuple[int, int, int, str]] = {456: (101, 123, 123, "S")}
    monkeypatch.setattr(job_processes, "_process_identity", identities.get)
    monkeypatch.setattr(os, "listdir", lambda path: ["456"])
    monkeypatch.setattr(job_processes, "_process_environ", lambda pid: None)
    monkeypatch.setattr(
        os, "killpg", lambda *args: pytest.fail("an unproven group was signaled")
    )

    settlement = await settle_recorded_job_group(group, grace=0, kill_timeout=0)

    assert settlement.outcome == PROCESS_GROUP_INCONCLUSIVE
    assert not settlement.reaped
    assert "could not be established" in settlement.detail


@pytest.mark.asyncio
async def test_a_leaderless_group_is_covered_by_the_owner_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    group = _recorded_group()
    identities: dict[int, tuple[int, int, int, str]] = {456: (101, 123, 123, "S")}
    monkeypatch.setattr(job_processes, "_process_identity", identities.get)
    monkeypatch.setattr(os, "listdir", lambda path: list(map(str, identities)))
    monkeypatch.setattr(
        job_processes,
        "_process_environ",
        lambda pid: {job_launcher.JOB_OWNER_MARKER_ENV: "session-jobs"},
    )
    signals: list[int] = []

    def stop(pgid: int, sig: int) -> None:
        signals.append(sig)
        identities.clear()

    monkeypatch.setattr(os, "killpg", stop)

    settlement = await settle_recorded_job_group(group, grace=0, kill_timeout=0)

    assert signals == [signal.SIGTERM]
    assert settlement.outcome == PROCESS_GROUP_REAPED


@pytest.mark.asyncio
async def test_an_empty_group_is_reaped_without_a_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    group = _recorded_group()
    monkeypatch.setattr(job_processes, "_process_identity", lambda pid: None)
    monkeypatch.setattr(os, "listdir", lambda path: [])
    monkeypatch.setattr(
        os, "killpg", lambda *args: pytest.fail("an empty group was signaled")
    )

    settlement = await settle_recorded_job_group(group, grace=0, kill_timeout=0)

    assert settlement.outcome == PROCESS_GROUP_REAPED
    assert settlement.reaped
    assert "no live member" in settlement.detail
