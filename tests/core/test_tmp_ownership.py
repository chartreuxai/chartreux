from __future__ import annotations

import asyncio
import errno
import fcntl
import inspect
import json
import multiprocessing
import os
from pathlib import Path
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time

import pytest

from chartreux.app_server.process_bootstrap import spawn_process
from chartreux.app_server.transport import terminate_process
from chartreux.core import tmp_ownership as to
from chartreux.core.hooks.executor import HookExecutor
from chartreux.core.hooks.models import HookConfig, HookType, PostAgentInvocation
from chartreux.core.tmp_ownership import (
    INDETERMINATE,
    LIVE,
    IdentityTuple,
    Ledger,
    SpawnLedgerBinding,
    SpawnRefusedError,
)
from chartreux.core.tools.base import BaseToolState, InvokeContext
from chartreux.core.tools.builtins.grep import Grep, GrepArgs, GrepToolConfig
from chartreux.core.utils.async_subprocess import spawn_registered_process
from chartreux.core.utils.shell import spawn_argv_command, spawn_shell_command


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


def _dead_identity() -> IdentityTuple:
    pid = _spawned_pid()
    return IdentityTuple(pid=pid, pgid=pid, starttime=0, boot_id=to.boot_id() or "")


def _init(parent: Path, owner: IdentityTuple | None = None) -> to.Allocation:
    return to.init_allocation(parent, to.KIND_SCRATCHPAD, owner=owner)


class TestAdjudication:
    def test_live_owner_tuple_matches(self) -> None:
        identity = to.current_identity()
        verdict = to.adjudicate(identity.to_json())
        assert verdict.verdict == LIVE
        assert verdict.live

    def test_dead_pid_gone(self) -> None:
        verdict = to.adjudicate(_dead_identity().to_json())
        assert verdict.dead
        assert verdict.reason == "pid-gone"

    def test_pid_reuse_starttime_mismatch_is_dead(self) -> None:
        identity = to.current_identity()
        reused = IdentityTuple(
            pid=identity.pid,
            pgid=identity.pgid,
            starttime=identity.starttime + 1,
            boot_id=identity.boot_id,
        )
        assert to.adjudicate(reused.to_json()).dead

    def test_boot_change_is_dead(self) -> None:
        identity = to.current_identity()
        other_boot = IdentityTuple(
            pid=identity.pid,
            pgid=identity.pgid,
            starttime=identity.starttime,
            boot_id="0" * 36,
        )
        verdict = to.adjudicate(other_boot.to_json())
        assert verdict.dead
        assert verdict.reason == "boot-change"

    def test_zombie_is_conclusively_dead(self, monkeypatch) -> None:
        identity = to.current_identity()
        monkeypatch.setattr(
            to,
            "read_proc_stat",
            lambda pid: to.ProcStat(
                state="Z",
                pgrp=identity.pgid,
                session=identity.pgid,
                starttime=identity.starttime,
            ),
        )
        assert to.adjudicate(identity.to_json()).dead

    def test_sig0_eperm_is_indeterminate(self, monkeypatch) -> None:
        identity = to.current_identity()
        monkeypatch.setattr(
            to,
            "read_proc_stat",
            lambda pid: (_ for _ in ()).throw(to.ProcUnreadable("x")),
        )

        def fake_kill(pid, sig):
            raise PermissionError(errno.EPERM, "operation not permitted")

        monkeypatch.setattr(os, "kill", fake_kill)
        verdict = to.adjudicate(identity.to_json())
        assert verdict.verdict == INDETERMINATE

    def test_malformed_record_is_indeterminate(self) -> None:
        assert to.adjudicate(None).verdict == INDETERMINATE
        assert to.adjudicate({"pid": "not-an-int"}).verdict == INDETERMINATE
        assert to.adjudicate({"pid": 1}).verdict == INDETERMINATE

    def test_no_tuple_live_pid_is_indeterminate(self) -> None:
        identity = to.current_identity()
        verdict = to.adjudicate({"pid": identity.pid})
        assert verdict.verdict == INDETERMINATE
        assert verdict.reason == "no-tuple-live"

    def test_no_tuple_gone_pid_is_dead(self) -> None:
        assert to.adjudicate({"pid": _spawned_pid()}).dead


class TestAllocationInit:
    def test_init_publishes_record_ledger_and_flag(self, parent: Path) -> None:
        alloc = _init(parent)
        assert to.is_managed_name(alloc.name)
        assert alloc.path.is_dir()
        record = to.load_ownership_record(parent, alloc.name)
        assert record is not None
        assert record.schema == 1
        assert record.kind == to.KIND_SCRATCHPAD
        assert record.owner == to.current_identity()
        assert alloc.ledger_path.exists()
        assert alloc.ledger_lock_path.exists()
        assert alloc.flag_clean_path.exists()
        assert not alloc.flag_dirty_path.exists()

    def test_pytest_run_root_prefix(self, parent: Path) -> None:
        alloc = to.init_allocation(parent, to.KIND_PYTEST_RUN_ROOT)
        assert alloc.name.startswith(to.PYTEST_RUN_PREFIX)

    def test_concurrent_creators_get_distinct_recorded_allocations(
        self, parent: Path
    ) -> None:
        results: list[to.Allocation] = []
        lock = threading.Lock()

        def create() -> None:
            alloc = _init(parent)
            with lock:
                results.append(alloc)

        threads = [threading.Thread(target=create) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert len({alloc.alloc_id for alloc in results}) == 8
        for alloc in results:
            assert to.load_ownership_record(parent, alloc.name) is not None
            assert alloc.flag_clean_path.exists()

    def test_allocation_from_path_roundtrip(self, parent: Path) -> None:
        alloc = _init(parent)
        again = to.allocation_from_path(alloc.path)
        assert again == alloc


class TestLedgerAndEnospc:
    def test_append_entries_roundtrip(self, parent: Path) -> None:
        alloc = _init(parent)
        ledger = Ledger(alloc)
        binding = SpawnLedgerBinding(alloc)
        handle = binding.begin_spawn(is_leader_group=True)
        assert binding.complete_spawn(handle, os.getpid(), to.current_identity())
        kinds = [entry["kind"] for entry in ledger.entries()]
        assert kinds == ["intent", "completion"]
        completion = ledger.entries()[1]
        assert completion["pid"] == os.getpid()
        assert completion["no_tuple"] is False
        assert completion["is_leader_group"] is True

    def test_retraction_settles_intent(self, parent: Path) -> None:
        alloc = _init(parent)
        binding = SpawnLedgerBinding(alloc)
        handle = binding.begin_spawn()
        binding.retract_spawn(handle)
        kinds = [entry["kind"] for entry in Ledger(alloc).entries()]
        assert kinds == ["intent", "retraction"]

    def test_intent_append_fails_flag_flips_spawn_proceeds(
        self, parent: Path, monkeypatch
    ) -> None:
        alloc = _init(parent)
        binding = SpawnLedgerBinding(alloc)

        def enospc(fd: int, data: bytes) -> int:
            raise OSError(errno.ENOSPC, "no space left on device")

        monkeypatch.setattr(to, "_os_write", enospc)
        handle = binding.begin_spawn()
        assert handle is not None
        assert Ledger(alloc).is_dirty()
        assert Ledger(alloc).entries() == []

    def test_intent_append_and_flag_both_fail_refuses_spawn(
        self, parent: Path, monkeypatch
    ) -> None:
        alloc = _init(parent)
        binding = SpawnLedgerBinding(alloc)

        def enospc(fd: int, data: bytes) -> int:
            raise OSError(errno.ENOSPC, "no space left on device")

        def fail_rename(src, dst) -> None:
            raise OSError(errno.ENOSPC, "no space left on device")

        monkeypatch.setattr(to, "_os_write", enospc)
        monkeypatch.setattr(to, "_os_rename", fail_rename)
        with pytest.raises(SpawnRefusedError):
            binding.begin_spawn()
        assert Ledger(alloc).entries() == []
        assert not alloc.path.joinpath(to.LEDGER_DIRNAME, to.FLAG_DIRTY).exists()

    def test_completion_append_failure_flips_dirty_and_stays_uncompleted(
        self, parent: Path, monkeypatch
    ) -> None:
        alloc = _init(parent)
        binding = SpawnLedgerBinding(alloc)
        handle = binding.begin_spawn()

        def enospc(fd: int, data: bytes) -> int:
            raise OSError(errno.ENOSPC, "no space left on device")

        monkeypatch.setattr(to, "_os_write", enospc)
        assert (
            binding.complete_spawn(handle, os.getpid(), to.current_identity()) is False
        )
        assert Ledger(alloc).is_dirty()
        kinds = [entry["kind"] for entry in Ledger(alloc).entries()]
        assert kinds == ["intent"]

    def test_resync_clears_dirty_on_single_appender_ledger(self, parent: Path) -> None:
        alloc = _init(parent)
        binding = SpawnLedgerBinding(alloc)
        Ledger(alloc).mark_dirty()
        assert Ledger(alloc).is_dirty()
        assert binding.resync() is True
        assert not Ledger(alloc).is_dirty()
        assert alloc.flag_clean_path.exists()

    def test_resync_on_foreign_ledger_keeps_dirty(self, parent: Path) -> None:
        alloc = _init(parent)
        binding = SpawnLedgerBinding(alloc)
        foreign = IdentityTuple(pid=1, pgid=1, starttime=1, boot_id=to.boot_id() or "")
        Ledger(alloc).append({
            "v": 1,
            "kind": "intent",
            "spawn_id": "x",
            "appender": foreign.to_json(),
        })
        Ledger(alloc).mark_dirty()
        assert binding.resync() is False
        assert Ledger(alloc).is_dirty()
        kinds = [entry["kind"] for entry in Ledger(alloc).entries()]
        assert kinds == ["intent", "resync"]

    def test_mark_dirty_never_creates_a_new_inode(
        self, parent: Path, monkeypatch
    ) -> None:
        alloc = _init(parent)
        ledger = Ledger(alloc)
        clean_inode = alloc.flag_clean_path.stat().st_ino
        ledger.mark_dirty()
        assert alloc.flag_dirty_path.stat().st_ino == clean_inode
        assert not alloc.flag_clean_path.exists()

        def fail_rename(src, dst) -> None:
            raise OSError(errno.ENOSPC, "no space left on device")

        monkeypatch.setattr(to, "_os_rename", fail_rename)
        with pytest.raises(OSError):
            ledger.clear_dirty()
        assert ledger.is_dirty()


class TestReapGates:
    def test_live_owner_retains(self, parent: Path) -> None:
        alloc = _init(parent, owner=to.current_identity())
        result = to.evaluate_allocation_gates(alloc)
        assert not result.reapable
        assert "retained-live" in result.reasons

    def test_dead_owner_clean_ledger_is_reapable(self, parent: Path) -> None:
        alloc = _init(parent, owner=_dead_identity())
        result = to.evaluate_allocation_gates(alloc)
        assert result.reapable, result.reasons

    def test_torn_init_managed_name_without_record_retains(self, parent: Path) -> None:
        (parent / "alloc-deadbeefcafe").mkdir()
        alloc = to.allocation_from_path(parent / "alloc-deadbeefcafe")
        result = to.evaluate_allocation_gates(alloc)
        assert not result.reapable
        assert "torn-init" in result.reasons

    def test_malformed_record_retains(self, parent: Path) -> None:
        alloc = _init(parent, owner=_dead_identity())
        alloc.owner_record_path.write_text("{not json")
        result = to.evaluate_allocation_gates(alloc)
        assert not result.reapable
        assert "torn-init" in result.reasons

    def test_symlink_substitution_retains(self, parent: Path) -> None:
        alloc = _init(parent, owner=_dead_identity())
        innocent = parent / "innocent"
        innocent.mkdir()
        shutil.rmtree(alloc.path)
        alloc.path.symlink_to(innocent)
        result = to.evaluate_allocation_gates(alloc)
        assert not result.reapable
        assert "retained-symlink-substitution" in result.reasons
        assert innocent.exists()

    def test_surviving_registered_spawn_retains_then_reaps(self, parent: Path) -> None:
        alloc = _init(parent, owner=_dead_identity())
        binding = SpawnLedgerBinding(alloc)
        child = subprocess.Popen(["sleep", "30"])
        try:
            handle = binding.begin_spawn()
            identity = to.process_identity_tuple(child.pid)
            assert identity is not None
            assert binding.complete_spawn(handle, child.pid, identity)
            result = to.evaluate_allocation_gates(alloc)
            assert not result.reapable
            assert "retained-registered-spawn" in result.reasons
        finally:
            child.kill()
            child.wait()
        result = to.evaluate_allocation_gates(alloc)
        assert result.reapable, result.reasons

    def test_leader_group_grandchild_retains_until_group_empties(
        self, parent: Path
    ) -> None:
        alloc = _init(parent, owner=_dead_identity())
        binding = SpawnLedgerBinding(alloc)
        leader = subprocess.Popen(
            ["sh", "-c", "sleep 30 & echo started > marker; wait"],
            start_new_session=True,
            cwd=alloc.path,
        )
        identity = to.process_identity_tuple(leader.pid)
        assert identity is not None
        handle = binding.begin_spawn(is_leader_group=True)
        assert binding.complete_spawn(handle, leader.pid, identity)
        marker = alloc.path / "marker"
        deadline = time.time() + 2
        while not marker.exists() and time.time() < deadline:
            time.sleep(0.02)
        assert marker.exists()
        os.kill(leader.pid, signal.SIGKILL)
        leader.wait()
        result = to.evaluate_allocation_gates(alloc)
        assert not result.reapable
        assert "retained-leader-group-nonempty" in result.reasons
        os.killpg(identity.pgid, signal.SIGKILL)
        time.sleep(0.1)
        result = to.evaluate_allocation_gates(alloc)
        assert result.reapable, result.reasons

    def test_containment_ref_retains_until_user_exits(self, parent: Path) -> None:
        alloc = _init(parent, owner=_dead_identity())
        user = subprocess.Popen(["sleep", "30"], cwd=alloc.path)
        try:
            result = to.evaluate_allocation_gates(alloc)
            assert not result.reapable
            assert "retained-containment-ref" in result.reasons
        finally:
            user.kill()
            user.wait()
        result = to.evaluate_allocation_gates(alloc)
        assert result.reapable, result.reasons

    def test_uncompleted_intent_retains(self, parent: Path) -> None:
        alloc = _init(parent, owner=_dead_identity())
        binding = SpawnLedgerBinding(alloc)
        binding.begin_spawn()
        result = to.evaluate_allocation_gates(alloc)
        assert not result.reapable
        assert "retained-uncompleted-intent" in result.reasons

    def test_orphaned_intent_dead_spawner_is_unresolved_intent(
        self, parent: Path
    ) -> None:
        alloc = _init(parent, owner=_dead_identity())
        binding = SpawnLedgerBinding(alloc, owner=_dead_identity())
        binding.begin_spawn()
        result = to.evaluate_allocation_gates(alloc)
        assert not result.reapable
        assert "UNRESOLVED-INTENT" in result.reasons

    def test_ledger_dirty_retains(self, parent: Path) -> None:
        alloc = _init(parent, owner=_dead_identity())
        Ledger(alloc).mark_dirty()
        result = to.evaluate_allocation_gates(alloc)
        assert not result.reapable
        assert "retained-ledger-dirty" in result.reasons

    def test_missing_ledger_retains(self, parent: Path) -> None:
        alloc = _init(parent, owner=_dead_identity())
        alloc.ledger_path.unlink()
        result = to.evaluate_allocation_gates(alloc)
        assert not result.reapable
        assert "retained-ledger-missing" in result.reasons

    def test_no_tuple_completion_live_retains_dead_reaps(self, parent: Path) -> None:
        alloc = _init(parent, owner=_dead_identity())
        binding = SpawnLedgerBinding(alloc)
        child = subprocess.Popen(["sleep", "30"])
        try:
            handle = binding.begin_spawn()
            assert binding.complete_spawn(handle, child.pid, None)
            result = to.evaluate_allocation_gates(alloc)
            assert not result.reapable
            assert any(
                reason.startswith("retained-registered-spawn-indeterminate")
                for reason in result.reasons
            )
        finally:
            child.kill()
            child.wait()
        result = to.evaluate_allocation_gates(alloc)
        assert result.reapable, result.reasons

    def test_owner_self_exclusion_allows_live_owner_reap(self, parent: Path) -> None:
        alloc = _init(parent, owner=to.current_identity())
        result = to.evaluate_allocation_gates(
            alloc, self_exclusion=to.current_identity()
        )
        assert result.reapable, result.reasons


class TestAmbientBinding:
    def test_ambient_binding_roundtrip(self, parent: Path) -> None:
        assert to.ambient_binding() is None
        alloc = _init(parent)
        binding = SpawnLedgerBinding(alloc)
        to.set_ambient_binding(binding)
        assert to.ambient_binding() is binding
        assert to.binding_from_context(None) is binding
        to.set_ambient_binding(None)
        assert to.ambient_binding() is None

    def test_context_field_wins_over_ambient(self, parent: Path) -> None:
        alloc = _init(parent)
        ambient = SpawnLedgerBinding(alloc)
        other = SpawnLedgerBinding(_init(parent))
        to.set_ambient_binding(ambient)

        class Ctx:
            tmp_ledger_binding = other

        assert to.binding_from_context(Ctx()) is other
        assert to.binding_from_context(object()) is ambient


class TestSpawnLedgerEntryShape:
    def test_entries_carry_invocation_dir_and_spawner(self, parent: Path) -> None:
        alloc = _init(parent)
        binding = SpawnLedgerBinding(alloc, invocation_dir=str(alloc.path))
        handle = binding.begin_spawn(
            spawn_id="fixed", is_leader_group=True, invocation_dir="/inv/dir"
        )
        binding.complete_spawn(handle, os.getpid(), to.current_identity())
        entries = Ledger(alloc).entries()
        intent, completion = entries
        assert intent["spawn_id"] == "fixed"
        assert intent["spawner"]["pid"] == os.getpid()
        assert intent["invocation_dir"] == "/inv/dir"
        assert completion["invocation_dir"] == "/inv/dir"
        assert completion["boot_id"] == to.boot_id()
        json.dumps(entries)


class TestQuarantineReap:
    def test_reap_removes_tree_and_every_record(self, parent: Path) -> None:
        alloc = _init(parent, owner=_dead_identity())
        (alloc.path / "work").mkdir()
        (alloc.path / "work" / "f.txt").write_text("x")
        result = to.reap_allocation(alloc)
        assert result.reaped, result.reasons
        assert not alloc.path.exists()
        assert not alloc.quarantine_path.exists()
        assert not alloc.qrec_path.exists()
        assert not alloc.qlease_path.exists()
        assert not alloc.owner_record_path.exists()

    def test_reap_retains_live_owner_untouched(self, parent: Path) -> None:
        alloc = _init(parent, owner=to.current_identity())
        result = to.reap_allocation(alloc)
        assert not result.reaped
        assert "retained-live" in result.reasons
        assert alloc.path.exists()
        assert not alloc.qrec_path.exists()

    def test_reaper_side_enospc_aborts_with_allocation_untouched(
        self, parent: Path, monkeypatch
    ) -> None:
        alloc = _init(parent, owner=_dead_identity())
        (alloc.path / "f.txt").write_text("x")

        def enospc(path, payload):
            raise OSError(errno.ENOSPC, "no space left on device")

        monkeypatch.setattr(to, "_write_json_atomic", enospc)
        result = to.reap_allocation(alloc)
        assert not result.reaped
        assert "reaper-side-abort" in result.reasons
        assert alloc.path.exists()
        assert (alloc.path / "f.txt").exists()
        assert not alloc.qrec_path.exists()
        assert not alloc.quarantine_path.exists()

    def test_rename_failure_leaves_dangling_record_resume_completes(
        self, parent: Path, monkeypatch
    ) -> None:
        alloc = _init(parent, owner=_dead_identity())
        (alloc.path / "f.txt").write_text("x")
        calls = {"n": 0}

        def fail_once(src, dst) -> None:
            calls["n"] += 1
            if calls["n"] == 1:
                raise OSError(errno.ENOSPC, "no space left on device")
            to.os.rename(src, dst)

        monkeypatch.setattr(to, "_os_rename", fail_once)
        result = to.reap_allocation(alloc)
        assert not result.reaped
        assert "quarantine-residue" in result.reasons
        assert alloc.path.exists()
        assert alloc.qrec_path.exists()
        monkeypatch.undo()
        results = to.resume_quarantine_records(parent)
        assert any(r.reaped for r in results), results
        assert not alloc.path.exists()
        assert not alloc.qrec_path.exists()
        assert not alloc.owner_record_path.exists()

    def test_dangling_record_resume_completes(self, parent: Path) -> None:
        alloc = _init(parent, owner=_dead_identity())
        (alloc.path / "f.txt").write_text("x")
        deleter = to.current_identity()
        to._write_json_atomic(
            alloc.qrec_path, to._qrec_payload(alloc, deleter, "authorized")
        )
        assert alloc.path.exists()
        results = to.resume_quarantine_records(parent, deleter=deleter)
        assert results[0].reaped, results
        assert not alloc.path.exists()
        assert not alloc.qrec_path.exists()

    def test_resume_discards_record_on_symlink_substitution(self, parent: Path) -> None:
        alloc = _init(parent, owner=_dead_identity())
        innocent = parent / "innocent"
        innocent.mkdir()
        (innocent / "keep.txt").write_text("keep")
        deleter = to.current_identity()
        to._write_json_atomic(
            alloc.qrec_path, to._qrec_payload(alloc, deleter, "authorized")
        )
        shutil.rmtree(alloc.path)
        alloc.path.symlink_to(innocent)
        results = to.resume_quarantine_records(parent, deleter=deleter)
        assert results[0].reasons == ("quarantine-record-discarded",)
        assert not alloc.qrec_path.exists()
        assert (innocent / "keep.txt").read_text() == "keep"
        assert alloc.path.is_symlink()

    def test_mid_rmtree_residue_resume_completes(self, parent: Path) -> None:
        alloc = _init(parent, owner=_dead_identity())
        (alloc.path / "f.txt").write_text("x")
        deleter = to.current_identity()
        os.rename(alloc.path, alloc.quarantine_path)
        to._write_json_atomic(
            alloc.qrec_path, to._qrec_payload(alloc, deleter, "renamed")
        )
        results = to.resume_quarantine_records(parent, deleter=deleter)
        assert results[0].reaped, results
        assert not alloc.quarantine_path.exists()
        assert not alloc.qrec_path.exists()
        assert not alloc.owner_record_path.exists()

    def test_tree_gone_record_is_cleared_idempotently(self, parent: Path) -> None:
        alloc = _init(parent, owner=_dead_identity())
        shutil.rmtree(alloc.path)
        deleter = to.current_identity()
        to._write_json_atomic(
            alloc.qrec_path, to._qrec_payload(alloc, deleter, "tree-gone")
        )
        results = to.resume_quarantine_records(parent, deleter=deleter)
        assert results[0].reasons == ("quarantine-residue-cleared",)
        assert not alloc.qrec_path.exists()
        assert to.resume_quarantine_records(parent, deleter=deleter) == []


class TestDeletionLease:
    def test_exclusive_under_live_holder(self, parent: Path) -> None:
        alloc = _init(parent)
        holder = to.current_identity()
        lease = to.acquire_deletion_lease(alloc.qlease_path, holder)
        assert lease is not None
        assert to.acquire_deletion_lease(alloc.qlease_path, _dead_identity()) is None
        lease.release()
        # The released-but-live holder's record still blocks takeover.
        assert to.acquire_deletion_lease(alloc.qlease_path, _dead_identity()) is None
        alloc.qlease_path.write_text(
            json.dumps({
                "schema": 1,
                "holder": _dead_identity().to_json(),
                "epoch": 0.0,
            })
        )
        taken = to.acquire_deletion_lease(alloc.qlease_path, to.current_identity())
        assert taken is not None
        taken.release()

    def test_takeover_after_recorded_holder_conclusively_dead(
        self, parent: Path
    ) -> None:
        alloc = _init(parent)
        dead = _dead_identity()
        alloc.qlease_path.write_text(
            json.dumps({"schema": 1, "holder": dead.to_json(), "epoch": 0.0})
        )
        lease = to.acquire_deletion_lease(alloc.qlease_path, to.current_identity())
        assert lease is not None
        lease.release()

    def test_no_tuple_live_record_blocks_takeover(self, parent: Path) -> None:
        alloc = _init(parent)
        alloc.qlease_path.write_text(
            json.dumps({"schema": 1, "holder": {"pid": os.getpid()}, "epoch": 0.0})
        )
        assert (
            to.acquire_deletion_lease(alloc.qlease_path, to.current_identity()) is None
        )

    def test_live_match_record_blocks_takeover(self, parent: Path) -> None:
        alloc = _init(parent)
        alloc.qlease_path.write_text(
            json.dumps({
                "schema": 1,
                "holder": to.current_identity().to_json(),
                "epoch": 0.0,
            })
        )
        assert to.acquire_deletion_lease(alloc.qlease_path, _dead_identity()) is None


class TestParentLockBlocking:
    """R4 pins: non-blocking lock-with-skip. A non-blocking acquisition failure
    fails closed as ``retained-lock-busy``; the blocking default and every
    pre-existing call shape stay byte-compatible.
    """

    def test_nb_parent_lock_raises_busy_when_held(self, parent: Path) -> None:
        with to.parent_lock(parent):
            with pytest.raises(to.ParentLockBusy):
                with to.parent_lock(parent, blocking=False):
                    pass

    def test_nb_reap_allocation_fails_closed_and_retains_untouched(
        self, parent: Path
    ) -> None:
        alloc = _init(parent, owner=_dead_identity())
        (alloc.path / "f.txt").write_text("x")
        with to.parent_lock(parent):
            result = to.reap_allocation(alloc, blocking=False)
        assert not result.reaped
        assert result.reasons == ("retained-lock-busy",)
        assert alloc.path.exists()
        assert (alloc.path / "f.txt").exists()
        assert alloc.owner_record_path.exists()
        assert not alloc.qrec_path.exists()
        assert not alloc.quarantine_path.exists()

    def test_nb_resume_quarantine_records_reports_retained_lock_busy(
        self, parent: Path
    ) -> None:
        alloc = _init(parent, owner=_dead_identity())
        deleter = to.current_identity()
        to._write_json_atomic(
            alloc.qrec_path, to._qrec_payload(alloc, deleter, "authorized")
        )
        with to.parent_lock(parent):
            results = to.resume_quarantine_records(parent, blocking=False)
        assert len(results) == 1
        assert not results[0].reaped
        assert results[0].reasons == ("retained-lock-busy",)
        assert alloc.qrec_path.exists()
        assert alloc.path.exists()

    def test_blocking_defaults_true_and_existing_call_shapes_work(
        self, parent: Path
    ) -> None:
        for fn in (to.parent_lock, to.reap_allocation, to.resume_quarantine_records):
            param = inspect.signature(fn).parameters["blocking"]
            assert param.kind is inspect.Parameter.KEYWORD_ONLY
            assert param.default is True
        # The named pre-R4 call shapes keep working unchanged:
        # tmp_residue_command.py:383 -> reap_allocation(allocation_from_path(path)).
        alloc = _init(parent, owner=_dead_identity())
        result = to.reap_allocation(to.allocation_from_path(alloc.path))
        assert result.reaped, result.reasons
        assert not alloc.path.exists()
        # The pre-R4 sweep shape: resume with no blocking kwarg at all.
        assert to.resume_quarantine_records(parent) == []

    def test_blocking_default_waits_for_a_held_lock(self, parent: Path) -> None:
        lock_path = parent / to.PARENT_LOCK_FILENAME
        holder_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        fcntl.flock(holder_fd, fcntl.LOCK_EX)

        def release() -> None:
            time.sleep(0.3)
            fcntl.flock(holder_fd, fcntl.LOCK_UN)

        releaser = threading.Thread(target=release)
        releaser.start()
        try:
            started = time.monotonic()
            with to.parent_lock(parent):
                waited = time.monotonic() - started
            assert waited >= 0.2
        finally:
            releaser.join()
            os.close(holder_fd)


class TestWorkerScopedGates:
    def test_dead_worker_with_live_session_owner_and_sibling_reaps(
        self, parent: Path
    ) -> None:
        # The enclosing session owner is LIVE: out of the worker-scoped inputs.
        alloc = _init(parent, owner=to.current_identity())
        worker = _dead_identity()
        worker_binding = SpawnLedgerBinding(alloc, owner=worker)
        dead_child = _spawned_pid()
        handle = worker_binding.begin_spawn(invocation_dir=str(alloc.path))
        worker_binding.complete_spawn(
            handle,
            dead_child,
            IdentityTuple(
                pid=dead_child, pgid=dead_child, starttime=0, boot_id=to.boot_id() or ""
            ),
        )
        # An out-of-scope live sibling spawn: not the worker's entry.
        sibling = subprocess.Popen(["sleep", "30"])
        try:
            sibling_binding = SpawnLedgerBinding(
                alloc, owner=to.process_identity_tuple(sibling.pid)
            )
            sib_handle = sibling_binding.begin_spawn()
            sibling_binding.complete_spawn(
                sib_handle, sibling.pid, to.process_identity_tuple(sibling.pid)
            )
            inv = alloc.path / "inv"
            inv.mkdir()
            result = to.evaluate_worker_scoped_gates(alloc, worker, [inv])
            assert result.reapable, result.reasons
        finally:
            sibling.kill()
            sibling.wait()

    def test_dirty_session_ledger_retains_dead_workers_candidate(
        self, parent: Path
    ) -> None:
        alloc = _init(parent, owner=to.current_identity())
        worker = _dead_identity()
        Ledger(alloc).mark_dirty()
        result = to.evaluate_worker_scoped_gates(alloc, worker, [alloc.path])
        assert not result.reapable
        assert "retained-ledger-dirty" in result.reasons

    def test_worker_led_group_survivor_retains(self, parent: Path) -> None:
        alloc = _init(parent, owner=to.current_identity())
        worker = _dead_identity()
        worker_binding = SpawnLedgerBinding(alloc, owner=worker)
        leader = subprocess.Popen(
            ["sh", "-c", "sleep 30 & echo started > marker; wait"],
            start_new_session=True,
            cwd=alloc.path,
        )
        identity = to.process_identity_tuple(leader.pid)
        assert identity is not None
        handle = worker_binding.begin_spawn(is_leader_group=True)
        worker_binding.complete_spawn(handle, leader.pid, identity)
        marker = alloc.path / "marker"
        deadline = time.time() + 2
        while not marker.exists() and time.time() < deadline:
            time.sleep(0.02)
        os.kill(leader.pid, signal.SIGKILL)
        leader.wait()
        try:
            result = to.evaluate_worker_scoped_gates(alloc, worker, [alloc.path])
            assert not result.reapable
            assert "retained-leader-group-nonempty" in result.reasons
        finally:
            os.killpg(identity.pgid, signal.SIGKILL)
        time.sleep(0.1)
        result = to.evaluate_worker_scoped_gates(alloc, worker, [alloc.path])
        assert result.reapable, result.reasons

    def test_uncompleted_worker_intent_is_unresolved(self, parent: Path) -> None:
        alloc = _init(parent, owner=to.current_identity())
        worker = _dead_identity()
        SpawnLedgerBinding(alloc, owner=worker).begin_spawn()
        result = to.evaluate_worker_scoped_gates(alloc, worker, [alloc.path])
        assert not result.reapable
        assert "UNRESOLVED-INTENT" in result.reasons

    def test_live_worker_retains(self, parent: Path) -> None:
        alloc = _init(parent, owner=to.current_identity())
        result = to.evaluate_worker_scoped_gates(
            alloc, to.current_identity(), [alloc.path]
        )
        assert not result.reapable
        assert "retained-worker-live" in result.reasons

    def test_worker_containment_ref_in_candidate_dir_retains(
        self, parent: Path
    ) -> None:
        alloc = _init(parent, owner=to.current_identity())
        worker = _dead_identity()
        inv = alloc.path / "inv"
        inv.mkdir()
        user = subprocess.Popen(["sleep", "30"], cwd=inv)
        try:
            result = to.evaluate_worker_scoped_gates(alloc, worker, [inv])
            assert not result.reapable
            assert "retained-containment-ref" in result.reasons
        finally:
            user.kill()
            user.wait()
        result = to.evaluate_worker_scoped_gates(alloc, worker, [inv])
        assert result.reapable, result.reasons


def _enospc_patches(monkeypatch) -> None:
    def enospc_write(fd: int, data: bytes) -> int:
        raise OSError(errno.ENOSPC, "no space left on device")

    def enospc_rename(src, dst) -> None:
        raise OSError(errno.ENOSPC, "no space left on device")

    monkeypatch.setattr(to, "_os_write", enospc_write)
    monkeypatch.setattr(to, "_os_rename", enospc_rename)


class TestSpawnSiteRegistration:
    @pytest.mark.asyncio
    async def test_shell_spawn_registers_intent_and_completion(
        self, parent: Path
    ) -> None:
        alloc = _init(parent)
        binding = SpawnLedgerBinding(alloc)
        to.set_ambient_binding(binding)
        proc = await spawn_shell_command("true")
        await proc.wait()
        entries = Ledger(alloc).entries()
        assert [entry["kind"] for entry in entries] == ["intent", "completion"]
        completion = entries[1]
        assert completion["pid"] == proc.pid
        assert completion["is_leader_group"] is True
        assert completion["no_tuple"] is False
        assert completion["boot_id"] == to.boot_id()

    @pytest.mark.asyncio
    async def test_argv_spawn_registers_with_explicit_ledger(
        self, parent: Path
    ) -> None:
        alloc = _init(parent)
        binding = SpawnLedgerBinding(alloc)
        proc = await spawn_argv_command(["true"], ledger=binding)
        await proc.wait()
        assert [e["kind"] for e in Ledger(alloc).entries()] == ["intent", "completion"]

    @pytest.mark.asyncio
    async def test_unbound_spawn_registers_nothing(self, parent: Path) -> None:
        alloc = _init(parent)
        proc = await spawn_argv_command(["true"])
        await proc.wait()
        assert Ledger(alloc).entries() == []
        assert to.ambient_binding() is None

    @pytest.mark.asyncio
    async def test_shell_spawn_refusal_is_typed_and_records_nothing(
        self, parent: Path, monkeypatch
    ) -> None:
        alloc = _init(parent)
        to.set_ambient_binding(SpawnLedgerBinding(alloc))
        _enospc_patches(monkeypatch)
        with pytest.raises(SpawnRefusedError):
            await spawn_shell_command("true")
        assert Ledger(alloc).entries() == []
        assert not Ledger(alloc).is_dirty()

    @pytest.mark.asyncio
    async def test_spawn_failure_retracts_intent(
        self, parent: Path, monkeypatch
    ) -> None:
        alloc = _init(parent)
        to.set_ambient_binding(SpawnLedgerBinding(alloc))

        async def failing_factory(*args, **kwargs):
            raise OSError("spawn failed")

        monkeypatch.setattr(asyncio, "create_subprocess_shell", failing_factory)
        with pytest.raises(OSError, match="spawn failed"):
            await spawn_shell_command("true")
        assert [e["kind"] for e in Ledger(alloc).entries()] == ["intent", "retraction"]

    @pytest.mark.asyncio
    async def test_registered_process_seam_with_ledger_param(
        self, parent: Path
    ) -> None:
        alloc = _init(parent)
        binding = SpawnLedgerBinding(alloc)
        proc = await spawn_registered_process(
            asyncio.create_subprocess_exec("true"), ledger=binding
        )
        await proc.wait()
        assert [e["kind"] for e in Ledger(alloc).entries()] == ["intent", "completion"]

    @pytest.mark.asyncio
    async def test_hook_executor_registers_and_typed_refusal(
        self, parent: Path, monkeypatch
    ) -> None:
        alloc = _init(parent)
        binding = SpawnLedgerBinding(alloc)
        hook = HookConfig(name="h", type=HookType.POST_AGENT, command="true")
        invocation = PostAgentInvocation(session_id="s", transcript_path="", cwd=".")
        result = await HookExecutor(ledger=binding).run(hook, invocation)
        assert result.exit_code == 0
        entries = Ledger(alloc).entries()
        assert [e["kind"] for e in entries] == ["intent", "completion"]
        assert entries[1]["is_leader_group"] is True

        other = _init(parent)
        _enospc_patches(monkeypatch)
        refused = await HookExecutor(ledger=SpawnLedgerBinding(other)).run(
            hook, invocation
        )
        assert refused.exit_code == 1
        assert "Spawn refused" in refused.stderr
        assert Ledger(other).entries() == []

    @pytest.mark.asyncio
    async def test_process_bootstrap_spawn_registers(self, parent: Path) -> None:
        alloc = _init(parent)
        binding = SpawnLedgerBinding(alloc)
        child = (
            "import sys, time; "
            'sys.stdout.write(\'{"chartreuxBootstrap":"ready","protocolVersion":2}\\n\'); '
            "sys.stdout.flush(); time.sleep(5)"
        )
        spawned = await spawn_process(
            [sys.executable, "-c", child], ledger=binding, ready_timeout=5.0
        )
        try:
            entries = Ledger(alloc).entries()
            assert [e["kind"] for e in entries] == ["intent", "completion"]
            completion = entries[1]
            assert completion["pid"] == spawned.process.pid
            assert completion["is_leader_group"] is False
            assert completion["no_tuple"] is False
        finally:
            await asyncio.to_thread(terminate_process, spawned.process)
            spawned.retire()

    @pytest.mark.asyncio
    async def test_grep_tool_registers_via_invoke_context_binding(
        self, parent: Path, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "f.txt").write_text("needle\n")
        alloc = _init(parent)
        binding = SpawnLedgerBinding(alloc)
        tool = Grep(config_getter=lambda: GrepToolConfig(), state=BaseToolState())
        ctx = InvokeContext(tool_call_id="t", tmp_ledger_binding=binding)
        async for _event in tool.run(GrepArgs(pattern="needle", path="."), ctx):
            pass
        entries = Ledger(alloc).entries()
        kinds = [e["kind"] for e in entries]
        assert kinds and set(kinds) <= {"intent", "completion"}
        completions = [e for e in entries if e["kind"] == "completion"]
        assert completions
        assert all(e["is_leader_group"] is False for e in completions)

    def test_invoke_context_delivers_the_binding(self, parent: Path) -> None:
        alloc = _init(parent)
        binding = SpawnLedgerBinding(alloc)
        ctx = InvokeContext(tool_call_id="t", tmp_ledger_binding=binding)
        assert ctx.tmp_ledger_binding is binding
        assert to.binding_from_context(ctx) is binding


class TestInitAllocationBlocking:
    """W3 pins: the init/allocation path is bounded under a held parent lock.
    ``blocking=True`` stays the default (byte-compatible for every non-startup
    caller, e.g. pytest_chartreux.py:286 and tmp_residue_command.py:59);
    ``blocking=False`` raises ParentLockBusy before anything is created.
    """

    def test_blocking_kwarg_default_true_and_existing_call_shapes_work(
        self, parent: Path
    ) -> None:
        params = inspect.signature(to.init_allocation).parameters
        assert list(params) == ["parent", "kind", "owner", "alloc_id", "blocking"]
        assert params["blocking"].kind is inspect.Parameter.KEYWORD_ONLY
        assert params["blocking"].default is True
        # The named non-startup call shapes keep working unchanged:
        # pytest_chartreux.py:286 -> init_allocation(parent, KIND_PYTEST_RUN_ROOT).
        run_root = to.init_allocation(parent, to.KIND_PYTEST_RUN_ROOT)
        assert run_root.name.startswith(to.PYTEST_RUN_PREFIX)
        # tmp_residue_command.py:59 -> init_allocation(parent, kind, owner=...).
        alloc = to.init_allocation(parent, to.KIND_SCRATCHPAD, owner=_dead_identity())
        assert to.load_ownership_record(parent, alloc.name) is not None

    def test_nb_init_under_held_lock_raises_busy_and_creates_nothing(
        self, parent: Path
    ) -> None:
        with to.parent_lock(parent):
            before = {p.name for p in parent.iterdir()}
            with pytest.raises(to.ParentLockBusy):
                to.init_allocation(parent, to.KIND_SCRATCHPAD, blocking=False)
        assert {p.name for p in parent.iterdir()} == before
        assert not list(parent.glob(f"{to.ALLOC_PREFIX}*"))
        assert not list(parent.glob(f"*{to.OWNER_RECORD_SUFFIX}"))

    def test_blocking_init_waits_for_a_held_lock(self, parent: Path) -> None:
        lock_path = parent / to.PARENT_LOCK_FILENAME
        holder_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        fcntl.flock(holder_fd, fcntl.LOCK_EX)

        def release() -> None:
            time.sleep(0.3)
            fcntl.flock(holder_fd, fcntl.LOCK_UN)

        releaser = threading.Thread(target=release)
        releaser.start()
        try:
            started = time.monotonic()
            alloc = to.init_allocation(parent, to.KIND_SCRATCHPAD)
            waited = time.monotonic() - started
            assert waited >= 0.2
            assert alloc.path.is_dir()
            assert to.load_ownership_record(parent, alloc.name) is not None
        finally:
            releaser.join()
            os.close(holder_fd)


class TestNonBlockingInitRetry:
    """W10 pins at the ownership layer: the in-session retry of a busy-lock
    init reuses the non-blocking ``init_allocation`` path, so repeated
    attempts under a held lock never create partial state, and the first
    attempt after the lock is released publishes exactly one complete
    allocation under the same single parent lock (lock ordering unchanged).
    """

    def test_repeated_nb_attempts_under_held_lock_create_nothing(
        self, parent: Path
    ) -> None:
        with to.parent_lock(parent):
            before = {p.name for p in parent.iterdir()}
            for _ in range(5):
                with pytest.raises(to.ParentLockBusy):
                    to.init_allocation(parent, to.KIND_SCRATCHPAD, blocking=False)
            assert {p.name for p in parent.iterdir()} == before
        assert not list(parent.glob(f"{to.ALLOC_PREFIX}*"))
        assert not list(parent.glob(f"*{to.OWNER_RECORD_SUFFIX}"))
        # Lock released: one non-blocking attempt publishes the whole
        # allocation (dir, ledger, lock, pre-established flag, record).
        alloc = to.init_allocation(parent, to.KIND_SCRATCHPAD, blocking=False)
        assert alloc.path.is_dir()
        assert alloc.ledger_dir.is_dir()
        assert alloc.ledger_path.exists()
        assert alloc.ledger_lock_path.exists()
        assert alloc.flag_clean_path.exists()
        assert not alloc.flag_dirty_path.exists()
        assert to.load_ownership_record(parent, alloc.name) is not None
        allocs = [
            p for p in parent.iterdir() if p.is_dir() and to.is_managed_name(p.name)
        ]
        assert allocs == [alloc.path]


class TestMultiProcessBarriers:
    """W3 pins: deterministic controlled multi-process barriers (forked
    children rendezvous on mp Events — no sleep-based timing): overlapping
    reapers and allocator under one held lock, one crashed reaper recovered
    by a later process, live/indeterminate allocations retained, and a stale
    clean allocation reclaimed.
    """

    @staticmethod
    def _start(ctx, target):
        proc = ctx.Process(target=target, daemon=True)
        proc.start()
        return proc

    @staticmethod
    def _reap_all(proc) -> None:
        proc.join(8)
        if proc.is_alive():
            proc.terminate()
            proc.join(8)

    def test_overlapping_reapers_and_allocator_under_one_held_lock(
        self, parent: Path
    ) -> None:
        ctx = multiprocessing.get_context("fork")
        stale = to.init_allocation(parent, to.KIND_SCRATCHPAD, owner=_dead_identity())
        (stale.path / "f.txt").write_text("x")

        held = ctx.Event()
        release = ctx.Event()
        results = ctx.Queue()

        def holder() -> None:
            with to.parent_lock(parent):
                held.set()
                release.wait(8)

        def reaper() -> None:
            assert held.wait(8)
            result = to.reap_allocation(
                to.allocation_from_path(stale.path), blocking=False
            )
            results.put(("reaper", result.reaped, list(result.reasons)))

        def sweep() -> None:
            assert held.wait(8)
            from chartreux.core.scratchpad import reap_stale_allocations

            out = reap_stale_allocations(parent.parent)
            results.put(("sweep", [(r.reaped, list(r.reasons)) for r in out]))

        def allocator() -> None:
            assert held.wait(8)
            try:
                alloc = to.init_allocation(parent, to.KIND_SCRATCHPAD, blocking=False)
                results.put(("allocator", "ok", alloc.name))
            except to.ParentLockBusy:
                results.put(("allocator", "busy", None))

        procs = [self._start(ctx, holder)]
        procs += [self._start(ctx, reaper) for _ in range(2)]
        procs += [self._start(ctx, sweep)]
        procs += [self._start(ctx, allocator) for _ in range(2)]
        try:
            assert held.wait(8)
            got = []
            deadline = time.monotonic() + 8
            while len(got) < 5 and time.monotonic() < deadline:
                try:
                    got.append(results.get(timeout=8))
                except queue.Empty:
                    break
        finally:
            release.set()
            for proc in procs:
                self._reap_all(proc)
        assert len(got) == 5, got
        reapers = [g for g in got if g[0] == "reaper"]
        sweeps = [g for g in got if g[0] == "sweep"]
        allocators = [g for g in got if g[0] == "allocator"]
        assert len(reapers) == 2 and len(sweeps) == 1 and len(allocators) == 2
        # Every overlapping reaper skipped and retained the entry (fail-closed).
        for _, reaped, reasons in reapers:
            assert not reaped
            assert reasons == ["retained-lock-busy"]
        # The full sweep skipped the stale entry the same way.
        assert len(sweeps[0][1]) == 1
        assert not sweeps[0][1][0][0]
        assert "retained-lock-busy" in sweeps[0][1][0][1]
        # Both allocators failed closed with the typed busy error, creating
        # nothing partial.
        assert sorted(allocators) == [("allocator", "busy", None)] * 2
        assert {p.name for p in parent.iterdir()} == {
            to.PARENT_LOCK_FILENAME,
            stale.name,
            stale.name + to.OWNER_RECORD_SUFFIX,
        }
        assert stale.path.exists()
        assert (stale.path / "f.txt").exists()
        assert stale.owner_record_path.exists()
        assert not list(parent.glob(f"*{to.QREC_SUFFIX}"))
        # Lock released: the later sweep reclaims the stale clean allocation
        # (no double delete — every record goes exactly once) and the
        # allocator retry succeeds, so the retry cadence is unchanged.
        result = to.reap_allocation(to.allocation_from_path(stale.path))
        assert result.reaped, result.reasons
        assert not stale.path.exists()
        assert not stale.qrec_path.exists()
        assert not stale.qlease_path.exists()
        assert not stale.quarantine_path.exists()
        assert not stale.owner_record_path.exists()
        retry = to.init_allocation(parent, to.KIND_SCRATCHPAD, blocking=False)
        assert retry.path.is_dir()
        assert to.load_ownership_record(parent, retry.name) is not None

    def test_crashed_reaper_mid_transaction_is_recovered_by_a_later_process(
        self, parent: Path
    ) -> None:
        ctx = multiprocessing.get_context("fork")
        alloc = to.init_allocation(parent, to.KIND_SCRATCHPAD, owner=_dead_identity())
        (alloc.path / "f.txt").write_text("x")
        residue_written = ctx.Event()
        results = ctx.Queue()

        def crashed_reaper() -> None:
            # Die (SIGKILL) between d0 (qrec written) and d1 (rename): the
            # kernel releases the holder's flock and leaves a resumable
            # residue — exactly a crashed reaper mid-transaction.
            original = to._write_json_atomic

            def die_after_record(path, payload):
                original(path, payload)
                residue_written.set()
                os.kill(os.getpid(), signal.SIGKILL)

            to._write_json_atomic = die_after_record
            to.reap_allocation(alloc)

        def later_resume() -> None:
            out = to.resume_quarantine_records(parent, blocking=False)
            results.put([(r.reaped, list(r.reasons)) for r in out])

        child = self._start(ctx, crashed_reaper)
        try:
            assert residue_written.wait(8)
            child.join(8)
        finally:
            self._reap_all(child)
        assert child.exitcode == -signal.SIGKILL
        # Residue after the crash: the authorization record is durable and
        # the candidate tree is still in place (the crash preceded d1).
        assert alloc.qrec_path.exists()
        assert alloc.path.exists()
        assert (alloc.path / "f.txt").exists()
        # The kernel released the crashed holder's flock: NB acquisition works.
        with to.parent_lock(parent, blocking=False):
            pass
        # A LATER process completes the interrupted transaction.
        later = self._start(ctx, later_resume)
        try:
            out = results.get(timeout=8)
            later.join(8)
        finally:
            self._reap_all(later)
        assert later.exitcode == 0
        assert out and out[0][0], out
        assert not alloc.path.exists()
        assert not alloc.qrec_path.exists()
        assert not alloc.qlease_path.exists()
        assert not alloc.quarantine_path.exists()
        assert not alloc.owner_record_path.exists()

    def test_live_and_indeterminate_allocations_retained_stale_reclaimed(
        self, parent: Path
    ) -> None:
        ctx = multiprocessing.get_context("fork")
        live = subprocess.Popen(["sleep", "60"], start_new_session=True)
        live_owner = to.process_identity_tuple(live.pid)
        assert live_owner is not None
        live_alloc = to.init_allocation(parent, to.KIND_SCRATCHPAD, owner=live_owner)
        dead = _dead_identity()
        ind_alloc = to.init_allocation(parent, to.KIND_SCRATCHPAD, owner=dead)
        ind_binding = SpawnLedgerBinding(ind_alloc, owner=dead)
        survivor = subprocess.Popen(["sleep", "60"], start_new_session=True)
        stale = to.init_allocation(parent, to.KIND_SCRATCHPAD, owner=_dead_identity())
        results = ctx.Queue()

        def run_sweep() -> list:
            go = ctx.Event()

            def sweep_child() -> None:
                assert go.wait(8)
                from chartreux.core.scratchpad import reap_stale_allocations

                out = reap_stale_allocations(parent.parent)
                results.put([(r.reaped, list(r.reasons)) for r in out])

            proc = self._start(ctx, sweep_child)
            try:
                go.set()
                return results.get(timeout=8)
            finally:
                self._reap_all(proc)

        try:
            handle = ind_binding.begin_spawn(
                is_leader_group=True, invocation_dir=str(ind_alloc.path)
            )
            # NO-TUPLE completion of a LIVE pid: indeterminate, fails closed.
            assert ind_binding.complete_spawn(handle, survivor.pid, None)
            # The sweep iterates sorted(parent.iterdir()) at sweep time, so the
            # entry snapshot is taken just before the sweep runs.
            entries = sorted(
                p.name
                for p in parent.iterdir()
                if to.is_managed_name(p.name) and p.is_dir()
            )
            assert entries == sorted([live_alloc.name, ind_alloc.name, stale.name])
            first = run_sweep()
            by_name = dict(zip(entries, first, strict=True))
            # Live owner: retained. Indeterminate registered spawn: retained.
            assert not by_name[live_alloc.name][0]
            assert "retained-live" in by_name[live_alloc.name][1]
            assert not by_name[ind_alloc.name][0]
            assert any(
                reason.startswith("retained-registered-spawn-indeterminate")
                for reason in by_name[ind_alloc.name][1]
            )
            # Stale clean allocation (dead owner, clean ledger): reclaimed.
            assert by_name[stale.name][0]
            assert by_name[stale.name][1] == [f"reaped-{stale.alloc_id}"]
            assert live_alloc.path.exists()
            assert ind_alloc.path.exists()
            assert not stale.path.exists()
            assert not stale.owner_record_path.exists()

            # Both survivors exit (reaped pids are conclusively dead): the
            # later sweep reclaims the remaining two allocations.
            live.kill()
            live.wait()
            survivor.kill()
            survivor.wait()
            entries2 = sorted(
                p.name
                for p in parent.iterdir()
                if to.is_managed_name(p.name) and p.is_dir()
            )
            assert entries2 == sorted([live_alloc.name, ind_alloc.name])
            second = run_sweep()
            by_name2 = dict(zip(entries2, second, strict=True))
            assert by_name2[live_alloc.name][0]
            assert by_name2[ind_alloc.name][0]
            assert not live_alloc.path.exists()
            assert not ind_alloc.path.exists()
            assert not live_alloc.owner_record_path.exists()
            assert not ind_alloc.owner_record_path.exists()
        finally:
            for proc in (live, survivor):
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
