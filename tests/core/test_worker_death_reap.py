"""C4 chunk (iii) tests: session-observed worker death -> worker-scoped reap.

The worker-death trigger places the dead worker's structurally enumerated
invocation dirs through the worker-scoped gates (S1a-S1c pinned): the
enclosing live session allocation is never a candidate, only the candidate
subtree is removed, and any ambiguity retains. No legacy/unmarked tree is
ever involved — fixtures are probe-created trees under a tmp temp root.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Any, cast

import pytest

from chartreux.app_server.process_bootstrap import (
    PROCESS_PROTOCOL_VERSION,
    ReadyFrame,
    SpawnedProcess,
)
from chartreux.app_server.root_agent_process import (
    RootAgentProcess,
    RootFailureSignature,
)
from chartreux.core import tmp_ownership as to
from chartreux.core.background_jobs import (
    BackgroundJobRegistry,
    BashStartArgs,
    enqueue_worker_death_reap,
    reap_worker_invocation_dirs,
)
from chartreux.core.tmp_ownership import IdentityTuple, Ledger, SpawnLedgerBinding


@pytest.fixture(autouse=True)
def _clear_ambient_binding():
    to.set_ambient_binding(None)
    yield
    to.set_ambient_binding(None)


@pytest.fixture
def parent(tmp_path: Path) -> Path:
    return to.managed_parent(tmp_path)


def _spawned_pid() -> int:
    handle = subprocess.Popen(["true"])
    handle.wait()
    return handle.pid


def _dead_identity(pid: int | None = None) -> IdentityTuple:
    pid = _spawned_pid() if pid is None else pid
    return IdentityTuple(pid=pid, pgid=pid, starttime=0, boot_id=to.boot_id() or "")


def _init(parent: Path, owner: IdentityTuple | None = None) -> to.Allocation:
    return to.init_allocation(parent, to.KIND_SCRATCHPAD, owner=owner)


def _register_worker_spawn(
    alloc: to.Allocation, worker: IdentityTuple, invocation_dir: Path, *, pid: int
) -> None:
    """Record one worker-attributed spawn with an invocation-dir binding."""
    worker_binding = SpawnLedgerBinding(alloc, owner=worker)
    handle = worker_binding.begin_spawn(invocation_dir=str(invocation_dir))
    worker_binding.complete_spawn(handle, pid, _dead_identity(pid))


class TestReapWorkerInvocationDirs:
    def test_dead_worker_reaps_only_the_candidate_subtree(self, parent: Path) -> None:
        # S1a: the enclosing session owner is LIVE and an out-of-scope live
        # sibling exists — both are OUT of the worker-scoped inputs.
        alloc = _init(parent, owner=to.current_identity())
        worker = _dead_identity()
        inv = alloc.path / "worker-inv"
        inv.mkdir()
        (inv / "artifact.txt").write_text("x")
        witness = alloc.path / "witness"
        witness.mkdir()
        pid = _spawned_pid()
        _register_worker_spawn(alloc, worker, inv, pid=pid)
        sibling = subprocess.Popen(["sleep", "30"])
        try:
            sibling_binding = SpawnLedgerBinding(
                alloc, owner=to.process_identity_tuple(sibling.pid)
            )
            sib_handle = sibling_binding.begin_spawn()
            sibling_binding.complete_spawn(
                sib_handle, sibling.pid, to.process_identity_tuple(sibling.pid)
            )
            binding = SpawnLedgerBinding(alloc)
            results = reap_worker_invocation_dirs(binding, worker, [str(inv)])
            assert any(r.reaped for r in results), results
            assert not inv.exists()
            # The enclosing allocation and the witness are untouched.
            assert alloc.path.is_dir()
            assert witness.is_dir()
        finally:
            sibling.kill()
            sibling.wait()

    def test_dirty_session_ledger_retains(self, parent: Path) -> None:
        # S1b: a dirty session ledger has missing-entry risk — retain even a
        # dead worker's candidate.
        alloc = _init(parent, owner=to.current_identity())
        worker = _dead_identity()
        inv = alloc.path / "worker-inv"
        inv.mkdir()
        _register_worker_spawn(alloc, worker, inv, pid=_spawned_pid())
        Ledger(alloc).mark_dirty()
        results = reap_worker_invocation_dirs(
            SpawnLedgerBinding(alloc), worker, [str(inv)]
        )
        assert not any(r.reaped for r in results)
        assert "retained-ledger-dirty" in results[0].reasons
        assert inv.is_dir()

    def test_worker_led_group_survivor_retains_then_reaps(self, parent: Path) -> None:
        # S1c: the leader-group sweep covers only groups LED by the worker or
        # its recorded spawns.
        alloc = _init(parent, owner=to.current_identity())
        worker = _dead_identity()
        inv = alloc.path / "worker-inv"
        inv.mkdir()
        leader = subprocess.Popen(
            ["sh", "-c", "sleep 30 & echo started > marker; wait"],
            start_new_session=True,
            cwd=inv,
        )
        identity = to.process_identity_tuple(leader.pid)
        assert identity is not None
        worker_binding = SpawnLedgerBinding(alloc, owner=worker)
        handle = worker_binding.begin_spawn(
            is_leader_group=True, invocation_dir=str(inv)
        )
        worker_binding.complete_spawn(handle, leader.pid, identity)
        marker = inv / "marker"
        deadline = time.time() + 2
        while not marker.exists() and time.time() < deadline:
            time.sleep(0.02)
        # The recorded leader itself is dead, but its group still has the
        # surviving grandchild: the leader-group sweep retains.
        os.kill(leader.pid, signal.SIGKILL)
        leader.wait()
        binding = SpawnLedgerBinding(alloc)
        try:
            results = reap_worker_invocation_dirs(binding, worker, [str(inv)])
            assert not any(r.reaped for r in results)
            assert "retained-leader-group-nonempty" in results[0].reasons
            assert inv.is_dir()
        finally:
            try:
                os.killpg(identity.pgid, signal.SIGKILL)
            except OSError:
                pass
        time.sleep(0.1)
        results = reap_worker_invocation_dirs(binding, worker, [str(inv)])
        assert any(r.reaped for r in results), results
        assert not inv.exists()
        assert alloc.path.is_dir()

    def test_live_worker_retains(self, parent: Path) -> None:
        alloc = _init(parent, owner=to.current_identity())
        inv = alloc.path / "worker-inv"
        inv.mkdir()
        results = reap_worker_invocation_dirs(
            SpawnLedgerBinding(alloc), to.current_identity(), [str(inv)]
        )
        assert not any(r.reaped for r in results)
        assert "retained-worker-live" in results[0].reasons
        assert inv.is_dir()

    def test_uncompleted_worker_intent_is_unresolved(self, parent: Path) -> None:
        alloc = _init(parent, owner=to.current_identity())
        worker = _dead_identity()
        inv = alloc.path / "worker-inv"
        inv.mkdir()
        SpawnLedgerBinding(alloc, owner=worker).begin_spawn(invocation_dir=str(inv))
        results = reap_worker_invocation_dirs(
            SpawnLedgerBinding(alloc), worker, [str(inv)]
        )
        assert not any(r.reaped for r in results)
        assert "UNRESOLVED-INTENT" in results[0].reasons
        assert inv.is_dir()

    def test_containment_ref_in_candidate_dir_retains(self, parent: Path) -> None:
        alloc = _init(parent, owner=to.current_identity())
        worker = _dead_identity()
        inv = alloc.path / "worker-inv"
        inv.mkdir()
        user = subprocess.Popen(["sleep", "30"], cwd=inv)
        try:
            results = reap_worker_invocation_dirs(
                SpawnLedgerBinding(alloc), worker, [str(inv)]
            )
            assert not any(r.reaped for r in results)
            assert "retained-containment-ref" in results[0].reasons
            assert inv.is_dir()
        finally:
            user.kill()
            user.wait()

    def test_candidate_outside_allocation_or_the_allocation_itself_is_ignored(
        self, parent: Path, tmp_path: Path
    ) -> None:
        alloc = _init(parent, owner=to.current_identity())
        worker = _dead_identity()
        outside = tmp_path / "outside"
        outside.mkdir()
        results = reap_worker_invocation_dirs(
            SpawnLedgerBinding(alloc), worker, [str(outside), str(alloc.path)]
        )
        assert results == []
        assert outside.is_dir()
        assert alloc.path.is_dir()


class TestEnqueueWorkerDeathReap:
    def test_dead_worker_pid_from_ledger_reaps_recorded_invocation_dir(
        self, parent: Path
    ) -> None:
        alloc = _init(parent, owner=to.current_identity())
        to.set_ambient_binding(SpawnLedgerBinding(alloc))
        pid = _spawned_pid()
        worker = _dead_identity(pid)
        inv = alloc.path / "worker-inv"
        inv.mkdir()
        (inv / "artifact.txt").write_text("x")
        _register_worker_spawn(alloc, worker, inv, pid=pid)
        # The session observes the worker's death by pid (the completion entry
        # recorded the full tuple at spawn).
        results = enqueue_worker_death_reap(pid)
        assert any(r.reaped for r in results), results
        assert not inv.exists()
        assert alloc.path.is_dir()

    def test_no_binding_retains_silently(self, parent: Path) -> None:
        alloc = _init(parent, owner=to.current_identity())
        worker = _dead_identity()
        inv = alloc.path / "worker-inv"
        inv.mkdir()
        _register_worker_spawn(alloc, worker, inv, pid=_spawned_pid())
        assert enqueue_worker_death_reap(_spawned_pid()) == []
        assert inv.is_dir()

    def test_unknown_pid_without_records_retains(self, parent: Path) -> None:
        alloc = _init(parent, owner=to.current_identity())
        to.set_ambient_binding(SpawnLedgerBinding(alloc))
        inv = alloc.path / "worker-inv"
        inv.mkdir()
        results = enqueue_worker_death_reap(_spawned_pid())
        assert results == []
        assert inv.is_dir()


class _DeadProcess:
    def __init__(self, pid: int) -> None:
        self.pid = pid

    def poll(self) -> int | None:
        return 0


class _SilentTransport:
    async def send(self, message: dict[str, Any]) -> None:
        pass

    async def messages(self):
        await asyncio.Event().wait()
        if False:  # pragma: no cover - an async generator must contain a yield
            yield {}

    async def close(self) -> None:
        pass


class TestRootAgentProcessDeathTrigger:
    @pytest.mark.asyncio
    async def test_declare_death_enqueues_worker_scoped_reap(
        self, parent: Path
    ) -> None:
        alloc = _init(parent, owner=to.current_identity())
        to.set_ambient_binding(SpawnLedgerBinding(alloc))
        pid = _spawned_pid()
        worker = _dead_identity(pid)
        inv = alloc.path / "worker-inv"
        inv.mkdir()
        (inv / "artifact.txt").write_text("x")
        _register_worker_spawn(alloc, worker, inv, pid=pid)
        spawned = SpawnedProcess(
            process=cast("subprocess.Popen[bytes]", _DeadProcess(pid)),
            transport=cast("Any", _SilentTransport()),
            ready=ReadyFrame(
                protocol_version=PROCESS_PROTOCOL_VERSION, capabilities=frozenset()
            ),
        )
        process = RootAgentProcess(spawned, heartbeat_interval=30.0)
        try:
            process._declare_death(
                "the root agent process exited",
                signature=RootFailureSignature.DEATH_RECEIPT,
            )
            assert process.death is not None
            # Session-observed worker death reaped the worker's invocation
            # dir through the worker-scoped gates; the session allocation and
            # its other contents are untouched.
            assert not inv.exists()
            assert alloc.path.is_dir()
        finally:
            await process.close()

    @pytest.mark.asyncio
    async def test_declare_death_without_binding_is_a_noop(self, parent: Path) -> None:
        spawned = SpawnedProcess(
            process=cast("subprocess.Popen[bytes]", _DeadProcess(_spawned_pid())),
            transport=cast("Any", _SilentTransport()),
            ready=ReadyFrame(
                protocol_version=PROCESS_PROTOCOL_VERSION, capabilities=frozenset()
            ),
        )
        process = RootAgentProcess(spawned, heartbeat_interval=30.0)
        try:
            process._declare_death(
                "the root agent process exited",
                signature=RootFailureSignature.DEATH_RECEIPT,
            )
            assert process.death is not None
        finally:
            await process.close()


class TestBackgroundJobSettleTrigger:
    @pytest.mark.asyncio
    async def test_settle_reaps_the_dead_jobs_recorded_invocation_dir(
        self, parent: Path
    ) -> None:
        alloc = _init(parent, owner=to.current_identity())
        binding = SpawnLedgerBinding(alloc)
        to.set_ambient_binding(binding)
        worker = _dead_identity()
        inv = alloc.path / "job-inv"
        inv.mkdir()
        (inv / "out.txt").write_text("done")
        registry = BackgroundJobRegistry()
        port = registry.root_port()
        try:
            job_id = port.reserve(BashStartArgs(command="true"))
            with registry._lock:
                job = registry._jobs[job_id]
                job.tmp_ledger_binding = binding
                job.run_contract = {
                    "schema": 1,
                    "job_id": job_id,
                    "context": {
                        "alloc_id": alloc.alloc_id,
                        "ledger_path": str(alloc.ledger_path),
                    },
                    "invocation_dirs": [str(inv)],
                    "worker": worker.to_json(),
                    "command_redacted_summary": "true",
                    "created_epoch": time.time(),
                }
            registry.commit(job_id)
            registry.capture(job_id, b"done", final=True)
            registry.observe_leader(job_id, 0)
            registry.settle(job_id)
            # The managed job's death observation placed its recorded
            # invocation dir through the worker-scoped gates.
            assert not inv.exists()
            assert alloc.path.is_dir()
        finally:
            await registry.aclose()

    @pytest.mark.asyncio
    async def test_settle_retains_when_the_session_ledger_is_dirty(
        self, parent: Path
    ) -> None:
        alloc = _init(parent, owner=to.current_identity())
        binding = SpawnLedgerBinding(alloc)
        to.set_ambient_binding(binding)
        worker = _dead_identity()
        inv = alloc.path / "job-inv"
        inv.mkdir()
        Ledger(alloc).mark_dirty()
        registry = BackgroundJobRegistry()
        port = registry.root_port()
        try:
            job_id = port.reserve(BashStartArgs(command="true"))
            with registry._lock:
                job = registry._jobs[job_id]
                job.tmp_ledger_binding = binding
                job.run_contract = {
                    "schema": 1,
                    "job_id": job_id,
                    "context": {
                        "alloc_id": alloc.alloc_id,
                        "ledger_path": str(alloc.ledger_path),
                    },
                    "invocation_dirs": [str(inv)],
                    "worker": worker.to_json(),
                    "command_redacted_summary": "true",
                    "created_epoch": time.time(),
                }
            registry.commit(job_id)
            registry.capture(job_id, b"done", final=True)
            registry.observe_leader(job_id, 0)
            registry.settle(job_id)
            assert inv.is_dir()
            assert alloc.path.is_dir()
        finally:
            await registry.aclose()
