from __future__ import annotations

import inspect
import multiprocessing
import os
from pathlib import Path
import subprocess
import time

import pytest

from chartreux.core import tmp_ownership as to
from chartreux.core.agent_loop._loop import AgentLoop
from chartreux.core.scratchpad import (
    SCRATCHPAD_INIT_MAX_ATTEMPTS,
    ScratchpadInitRetry,
    bind_scratchpad,
    cleanup_scratchpad,
    init_scratchpad,
    is_scratchpad_display_path,
    is_scratchpad_path,
    reap_stale_allocations,
    unbind_scratchpad,
)
from chartreux.core.tmp_ownership import (
    SpawnLedgerBinding,
    allocation_from_path,
    load_ownership_record,
    managed_parent,
)


@pytest.fixture(autouse=True)
def _clear_ambient_binding():
    to.set_ambient_binding(None)
    yield
    to.set_ambient_binding(None)


def _dead_identity() -> to.IdentityTuple:
    handle = subprocess.Popen(["true"])
    handle.wait()
    return to.IdentityTuple(
        pid=handle.pid, pgid=handle.pid, starttime=0, boot_id=to.boot_id() or ""
    )


class TestInitScratchpad:
    def test_creates_managed_marked_allocation(self, tmp_path: Path) -> None:
        result = init_scratchpad("test-session", temp_root=tmp_path)
        assert result is not None
        assert result.is_dir()
        parent = managed_parent(tmp_path)
        assert result.parent == parent
        assert result.name.startswith("alloc-")
        alloc = allocation_from_path(result)
        record = load_ownership_record(parent, alloc.name)
        assert record is not None
        assert record.kind == to.KIND_SCRATCHPAD
        assert record.owner == to.current_identity()
        assert alloc.ledger_path.exists()
        assert alloc.flag_clean_path.exists()

    def test_different_sessions_get_different_dirs(self, tmp_path: Path) -> None:
        first = init_scratchpad("session-1", temp_root=tmp_path)
        second = init_scratchpad("session-2", temp_root=tmp_path)
        assert first != second

    def test_cleanup_reaps_clean_allocation(self, tmp_path: Path) -> None:
        path = init_scratchpad("test-session", temp_root=tmp_path)
        assert path is not None
        assert cleanup_scratchpad(path) is True
        assert not path.exists()
        assert not allocation_from_path(path).owner_record_path.exists()

    def test_cleanup_none_is_noop(self) -> None:
        assert cleanup_scratchpad(None) is False

    def test_cleanup_refuses_legacy_unmarked_tree(self, tmp_path: Path) -> None:
        legacy = tmp_path / "chartreux-scratchpad-legacy-abc"
        legacy.mkdir()
        (legacy / "keep.txt").write_text("x")
        assert cleanup_scratchpad(legacy) is False
        assert (legacy / "keep.txt").exists()


class TestGatedGracefulClose:
    def test_close_retains_whole_allocation_on_surviving_registered_spawn(
        self, tmp_path: Path
    ) -> None:
        path = init_scratchpad("test-session", temp_root=tmp_path)
        assert path is not None
        alloc = allocation_from_path(path)
        binding = SpawnLedgerBinding(alloc)
        proc = subprocess.Popen(["sleep", "60"], start_new_session=True)
        try:
            handle = binding.begin_spawn(is_leader_group=True, invocation_dir=str(path))
            binding.complete_spawn(
                handle, proc.pid, to.process_identity_tuple(proc.pid)
            )
            (path / "work.txt").write_text("wip")
            assert cleanup_scratchpad(path) is False
            assert path.exists()
            assert (path / "work.txt").exists()
            assert alloc.owner_record_path.exists()
        finally:
            proc.kill()
            proc.wait()
        # Survivor exited: the same gated close now reaps (residue zero).
        assert cleanup_scratchpad(path) is True
        assert not path.exists()
        assert not alloc.owner_record_path.exists()


class TestStartupReaper:
    def test_reaper_collects_dead_owner_after_survivor_exit(
        self, tmp_path: Path
    ) -> None:
        parent = managed_parent(tmp_path)
        dead = _dead_identity()
        alloc = to.init_allocation(parent, to.KIND_SCRATCHPAD, owner=dead)
        binding = SpawnLedgerBinding(alloc, owner=dead)
        proc = subprocess.Popen(["sleep", "60"], start_new_session=True)
        try:
            handle = binding.begin_spawn(
                is_leader_group=True, invocation_dir=str(alloc.path)
            )
            binding.complete_spawn(
                handle, proc.pid, to.process_identity_tuple(proc.pid)
            )
            results = reap_stale_allocations(tmp_path)
            assert not any(r.reaped for r in results)
            assert alloc.path.exists()
        finally:
            proc.kill()
            proc.wait()
        results = reap_stale_allocations(tmp_path)
        assert any(r.reaped for r in results)
        assert not alloc.path.exists()
        assert not alloc.owner_record_path.exists()

    def test_reaper_retains_live_owner(self, tmp_path: Path) -> None:
        path = init_scratchpad("live-session", temp_root=tmp_path)
        assert path is not None
        results = reap_stale_allocations(tmp_path)
        assert not any(r.reaped for r in results)
        assert path.exists()

    def test_reaper_never_touches_unmarked_trees(self, tmp_path: Path) -> None:
        legacy = tmp_path / "chartreux-scratchpad-legacy-abc"
        legacy.mkdir()
        (legacy / "keep.txt").write_text("x")
        reap_stale_allocations(tmp_path)
        assert (legacy / "keep.txt").exists()
        pofp = tmp_path / "pytest-of-pav"
        pofp.mkdir()
        reap_stale_allocations(tmp_path)
        assert pofp.exists()


class TestSweepLockContention:
    """R4 pins: the reaper sweep takes the machine-global parent lock
    non-blocking per entry — a busy lock skips and retains the entry
    (retained-lock-busy), and the next sweep retries it unchanged.
    """

    def test_sweep_signature_unchanged_and_skips_but_retains_under_contention(
        self, tmp_path: Path
    ) -> None:
        # Byte-compatible no-arg callers (pytest_chartreux.py:283, _loop.py:645).
        params = inspect.signature(reap_stale_allocations).parameters
        assert list(params) == ["temp_root"]
        assert params["temp_root"].default is None

        parent = managed_parent(tmp_path)
        dead = _dead_identity()
        alloc = to.init_allocation(parent, to.KIND_SCRATCHPAD, owner=dead)
        (alloc.path / "f.txt").write_text("x")
        with to.parent_lock(parent):
            results = reap_stale_allocations(tmp_path)
        assert results
        assert not any(r.reaped for r in results)
        assert all("retained-lock-busy" in r.reasons for r in results)
        assert alloc.path.exists()
        assert (alloc.path / "f.txt").exists()
        assert alloc.owner_record_path.exists()

    def test_sweep_retries_retained_entries_on_the_next_run(
        self, tmp_path: Path
    ) -> None:
        parent = managed_parent(tmp_path)
        dead = _dead_identity()
        alloc = to.init_allocation(parent, to.KIND_SCRATCHPAD, owner=dead)
        (alloc.path / "f.txt").write_text("x")
        deleter = to.current_identity()
        to._write_json_atomic(
            alloc.qrec_path, to._qrec_payload(alloc, deleter, "authorized")
        )
        with to.parent_lock(parent):
            busy = reap_stale_allocations(tmp_path)
        assert not any(r.reaped for r in busy)
        assert all("retained-lock-busy" in r.reasons for r in busy)
        assert alloc.path.exists()
        assert alloc.qrec_path.exists()
        # Lock released: the very next sweep retries and completes — the busy
        # skip consumed no state, so the per-boot retry cadence is unchanged.
        results = reap_stale_allocations(tmp_path)
        assert any(r.reaped for r in results)
        assert not alloc.path.exists()
        assert not alloc.qrec_path.exists()
        assert not alloc.owner_record_path.exists()


class TestStartupInitBounded:
    """W3 pins: the startup init allocation path is bounded under a held
    parent lock. A lock held across BOTH startup steps (the sweep and the
    scratchpad init, _loop.py:642-651) no longer blocks the boot: the sweep
    reports retained-lock-busy and the init fails closed to degrade-to-None,
    so the startup completes without queueing behind the holder; the next
    boot retries and succeeds.
    """

    def test_init_scratchpad_signature_unchanged(self) -> None:
        # Byte-compatible callers (_loop.py:650 and friends).
        params = inspect.signature(init_scratchpad).parameters
        assert list(params) == ["session_id", "temp_root"]
        assert params["temp_root"].kind is inspect.Parameter.KEYWORD_ONLY
        assert params["temp_root"].default is None

    def test_init_under_held_lock_degrades_to_none_and_creates_nothing(
        self, tmp_path: Path
    ) -> None:
        parent = managed_parent(tmp_path)
        with to.parent_lock(parent):
            before = {p.name for p in parent.iterdir()}
            assert init_scratchpad("s", temp_root=tmp_path) is None
        assert {p.name for p in parent.iterdir()} == before
        assert not list(parent.glob("alloc-*"))
        assert not list(parent.glob("*.owner.json"))
        # Lock released: the very next init succeeds unchanged.
        assert init_scratchpad("s", temp_root=tmp_path) is not None

    def test_startup_under_one_held_lock_is_bounded_and_degrades(
        self, tmp_path: Path
    ) -> None:
        ctx = multiprocessing.get_context("fork")
        parent = managed_parent(tmp_path)
        stale = to.init_allocation(parent, to.KIND_SCRATCHPAD, owner=_dead_identity())
        (stale.path / "f.txt").write_text("x")

        held = ctx.Event()
        release = ctx.Event()
        results = ctx.Queue()

        def holder() -> None:
            with to.parent_lock(parent):
                held.set()
                release.wait(8)

        def startup_child() -> None:
            # The _loop.py startup sequence in a separate process, while
            # another process holds the machine-global parent lock.
            assert held.wait(8)
            t0 = time.monotonic()
            sweep = reap_stale_allocations(tmp_path)
            t1 = time.monotonic()
            sp = init_scratchpad("bench-session", temp_root=tmp_path)
            t2 = time.monotonic()
            results.put({
                "sweep_wall": t1 - t0,
                "init_wall": t2 - t1,
                "scratchpad": str(sp) if sp is not None else None,
                "sweep": [(r.reaped, list(r.reasons)) for r in sweep],
            })

        procs = [
            ctx.Process(target=holder, daemon=True),
            ctx.Process(target=startup_child, daemon=True),
        ]
        for proc in procs:
            proc.start()
        try:
            assert held.wait(8)
            # The startup result arrives WHILE the lock is still held.
            result = results.get(timeout=8)
        finally:
            release.set()
            for proc in procs:
                proc.join(8)
                if proc.is_alive():
                    proc.terminate()
                    proc.join(8)
        assert result["scratchpad"] is None
        assert result["init_wall"] < 5.0
        assert result["sweep_wall"] < 5.0
        assert result["sweep"]
        assert all(
            not reaped and "retained-lock-busy" in reasons
            for reaped, reasons in result["sweep"]
        )
        # The degraded init created nothing partial.
        assert {p.name for p in parent.iterdir()} == {
            to.PARENT_LOCK_FILENAME,
            stale.name,
            stale.name + to.OWNER_RECORD_SUFFIX,
        }
        assert stale.path.exists()
        assert (stale.path / "f.txt").exists()
        # Lock released: the next boot's startup path succeeds and reaps.
        assert init_scratchpad("retry-session", temp_root=tmp_path) is not None
        sweep = reap_stale_allocations(tmp_path)
        assert any(r.reaped for r in sweep)
        assert not stale.path.exists()
        assert not stale.owner_record_path.exists()


def _managed_allocs(parent: Path) -> list[Path]:
    return [p for p in parent.iterdir() if p.is_dir() and to.is_managed_name(p.name)]


class TestInSessionRetry:
    """W10 pins: a boot init that degraded under a held parent lock is not
    permanent for the session. The bounded ScratchpadInitRetry retries the
    same non-blocking, fail-closed init in-session: a session that degraded
    at boot recovers its scratchpad once the lock is released (same process,
    deterministic rendezvous — no sleep timing), and after the attempt budget
    is spent it stays degraded and never spins.
    """

    def test_retry_budget_defaults(self) -> None:
        assert SCRATCHPAD_INIT_MAX_ATTEMPTS >= 2
        retry = ScratchpadInitRetry("s")
        assert retry.max_attempts == SCRATCHPAD_INIT_MAX_ATTEMPTS
        assert retry.attempts == 0
        assert not retry.exhausted
        retry.attempts = retry.max_attempts
        assert retry.exhausted

    def test_degraded_boot_recovers_in_same_process_after_lock_release(
        self, tmp_path: Path
    ) -> None:
        parent = managed_parent(tmp_path)
        with to.parent_lock(parent):
            # Boot under a held lock: degrade-to-None, nothing created, and
            # the armed retry (boot attempt counts as the first) still
            # degrades while the lock is held.
            assert init_scratchpad("s", temp_root=tmp_path) is None
            retry = ScratchpadInitRetry("s", temp_root=tmp_path, attempts=1)
            assert not retry.exhausted
            assert retry.attempt() is None
            assert retry.attempts == 2
            assert _managed_allocs(parent) == []
        # Lock released: the next in-session retry recovers, same process,
        # publishing exactly one complete allocation.
        recovered = retry.attempt()
        assert recovered is not None
        assert recovered.is_dir()
        assert recovered.parent == parent
        alloc = allocation_from_path(recovered)
        record = load_ownership_record(parent, alloc.name)
        assert record is not None
        assert record.kind == to.KIND_SCRATCHPAD
        assert alloc.ledger_path.exists()
        assert alloc.ledger_lock_path.exists()
        assert alloc.flag_clean_path.exists()
        assert _managed_allocs(parent) == [recovered]

    def test_retry_budget_is_bounded_and_never_spins(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        parent = managed_parent(tmp_path)
        calls = 0

        def counting_init(session_id: str, *, temp_root=None):  # type: ignore[no-untyped-def]
            nonlocal calls
            calls += 1
            return init_scratchpad(session_id, temp_root=temp_root)

        monkeypatch.setattr("chartreux.core.scratchpad.init_scratchpad", counting_init)
        retry = ScratchpadInitRetry("s", temp_root=tmp_path, max_attempts=3, attempts=1)
        with to.parent_lock(parent):
            # Budget: attempts 2 and 3 fire and degrade; every further call
            # is a no-op that does not even reach init_scratchpad.
            assert retry.attempt() is None
            assert retry.attempt() is None
            assert retry.exhausted
            for _ in range(5):
                assert retry.attempt() is None
            assert calls == 2
            assert retry.attempts == 3
            assert _managed_allocs(parent) == []
        # Even after the lock is released, an exhausted retry stays degraded.
        assert retry.attempt() is None
        assert calls == 2
        assert _managed_allocs(parent) == []

    def test_boot_degrade_recovers_in_session_after_holder_release(
        self, tmp_path: Path
    ) -> None:
        ctx = multiprocessing.get_context("fork")
        parent = managed_parent(tmp_path)
        held = ctx.Event()
        release = ctx.Event()
        released = ctx.Event()
        results = ctx.Queue()

        def holder() -> None:
            with to.parent_lock(parent):
                held.set()
                release.wait(8)
            released.set()

        def session_child() -> None:
            # The _loop.py boot sequence in a separate process while another
            # process holds the machine-global parent lock.
            assert held.wait(8)
            t0 = time.monotonic()
            boot = init_scratchpad("bench-session", temp_root=tmp_path)
            t1 = time.monotonic()
            results.put({
                "boot": str(boot) if boot is not None else None,
                "boot_wall": t1 - t0,
            })
            # The per-turn in-session retry, still under the held lock.
            retry = ScratchpadInitRetry("bench-session", temp_root=tmp_path, attempts=1)
            assert retry.attempt() is None
            # The holder released the lock (rendezvous, no sleep timing): the
            # next turn's retry recovers within the SAME process.
            assert released.wait(8)
            t2 = time.monotonic()
            recovered = retry.attempt()
            t3 = time.monotonic()
            results.put({
                "recovered": str(recovered) if recovered is not None else None,
                "retry_wall": t3 - t2,
                "attempts": retry.attempts,
            })

        procs = [
            ctx.Process(target=holder, daemon=True),
            ctx.Process(target=session_child, daemon=True),
        ]
        for proc in procs:
            proc.start()
        try:
            assert held.wait(8)
            boot_result = results.get(timeout=8)
            release.set()
            retry_result = results.get(timeout=8)
        finally:
            release.set()
            for proc in procs:
                proc.join(8)
                if proc.is_alive():
                    proc.terminate()
                    proc.join(8)
        # Boot degraded, bounded, while the lock was still held ...
        assert boot_result["boot"] is None
        assert boot_result["boot_wall"] < 5.0
        # ... and the same process recovered once the lock was released.
        assert retry_result["recovered"] is not None
        assert retry_result["retry_wall"] < 5.0
        assert retry_result["attempts"] == 3
        recovered = Path(retry_result["recovered"])
        assert recovered.is_dir()
        assert _managed_allocs(parent) == [recovered]
        assert (
            load_ownership_record(parent, allocation_from_path(recovered).name)
            is not None
        )

    def test_loop_ensure_scratchpad_recovers_and_rebinds(self, tmp_path: Path) -> None:
        # The _loop.py wiring: a boot-degraded loop (retry armed, no
        # scratchpad) spends one bounded non-blocking attempt per turn via
        # _ensure_scratchpad and re-points the live consumers on recovery.

        class _ToolManager:
            def __init__(self) -> None:
                self.calls: list[Path] = []

            def set_scratchpad_dir(self, scratchpad_dir: Path | None) -> None:
                self.calls.append(scratchpad_dir)  # type: ignore[arg-type]

        loop = AgentLoop.__new__(AgentLoop)
        loop.scratchpad_dir = None
        loop._scratchpad_retry = ScratchpadInitRetry(
            "s", temp_root=tmp_path, attempts=1
        )
        tool_manager = _ToolManager()
        loop.tool_manager = tool_manager  # type: ignore[assignment]
        parent = managed_parent(tmp_path)
        with to.parent_lock(parent):
            loop._ensure_scratchpad()
            assert loop.scratchpad_dir is None
            assert tool_manager.calls == []
            assert to.ambient_binding() is None
            assert _managed_allocs(parent) == []
        loop._ensure_scratchpad()
        assert loop.scratchpad_dir is not None
        assert tool_manager.calls == [loop.scratchpad_dir]
        binding = to.ambient_binding()
        assert binding is not None
        assert binding.allocation.path == loop.scratchpad_dir
        # Recovered: later turns do not spend further attempts or re-init.
        attempts = loop._scratchpad_retry.attempts
        loop._ensure_scratchpad()
        assert loop._scratchpad_retry.attempts == attempts
        assert _managed_allocs(parent) == [loop.scratchpad_dir]


class TestBinding:
    def test_bind_sets_ambient_binding(self, tmp_path: Path) -> None:
        path = init_scratchpad("test-session", temp_root=tmp_path)
        assert path is not None
        binding = bind_scratchpad(path)
        assert binding is not None
        assert binding.allocation.path == path
        assert to.ambient_binding() is binding
        unbind_scratchpad()
        assert to.ambient_binding() is None

    def test_bind_refuses_unmarked_path(self, tmp_path: Path) -> None:
        legacy = tmp_path / "chartreux-scratchpad-x"
        legacy.mkdir()
        assert bind_scratchpad(legacy) is None
        assert to.ambient_binding() is None
        assert bind_scratchpad(None) is None


class TestIsScratchpadPath:
    def test_false_when_not_initialized(self) -> None:
        assert not is_scratchpad_path("/tmp/anything", scratchpad_dir=None)

    def test_true_for_file_inside(self, tmp_path: Path) -> None:
        sp = init_scratchpad("test-session", temp_root=tmp_path)
        assert sp is not None
        assert is_scratchpad_path(str(sp / "file.txt"), scratchpad_dir=sp)

    def test_true_for_nested_file(self, tmp_path: Path) -> None:
        sp = init_scratchpad("test-session", temp_root=tmp_path)
        assert sp is not None
        assert is_scratchpad_path(str(sp / "subdir" / "file.txt"), scratchpad_dir=sp)

    def test_true_for_dir_itself(self, tmp_path: Path) -> None:
        sp = init_scratchpad("test-session", temp_root=tmp_path)
        assert sp is not None
        assert is_scratchpad_path(str(sp), scratchpad_dir=sp)

    def test_other_session_scratchpad_is_not_trusted(self, tmp_path: Path) -> None:
        sp1 = init_scratchpad("session-1", temp_root=tmp_path)
        sp2 = init_scratchpad("session-2", temp_root=tmp_path)
        assert sp1 is not None and sp2 is not None
        assert is_scratchpad_path(str(sp1 / "file.txt"), scratchpad_dir=sp1)
        assert not is_scratchpad_path(str(sp2 / "file.txt"), scratchpad_dir=sp1)

    def test_false_for_outside_path(self, tmp_path: Path) -> None:
        sp = init_scratchpad("test-session", temp_root=tmp_path)
        assert sp is not None
        assert not is_scratchpad_path("/etc/passwd", scratchpad_dir=sp)

    def test_false_for_traversal_attack(self, tmp_path: Path) -> None:
        sp = init_scratchpad("test-session", temp_root=tmp_path)
        assert sp is not None
        traversal = str(sp / ".." / ".." / ".." / "etc" / "passwd")
        assert not is_scratchpad_path(traversal, scratchpad_dir=sp)

    def test_false_for_sibling_directory(self, tmp_path: Path) -> None:
        sp = init_scratchpad("test-session", temp_root=tmp_path)
        assert sp is not None
        sibling = str(sp.parent / "other-dir" / "file.txt")
        assert not is_scratchpad_path(sibling, scratchpad_dir=sp)


class TestScratchpadDisplayPaths:
    def test_recognizes_current_and_legacy_scratchpad_prefixes(self):
        assert is_scratchpad_display_path("/tmp/chartreux-scratchpad-abc/file.txt")
        assert is_scratchpad_display_path("/tmp/vibe-scratchpad-abc/file.txt")

    def test_recognizes_managed_parent_paths(self):
        managed = f"/tmp/chartreux-managed-{os.getuid()}"
        assert is_scratchpad_display_path(f"{managed}/alloc-abc123/file.txt")
        assert is_scratchpad_display_path(
            f"{managed}/pytest-run-abc123/popen-gw0/test_x0"
        )
        assert not is_scratchpad_display_path("/tmp/pytest-of-pav/pytest-8/test_x0")
