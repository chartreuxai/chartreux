from __future__ import annotations

import asyncio
import math
import os
from pathlib import Path
from typing import Any
import warnings

import pytest

_WORKER_COUNT_ENV_VAR = "PYTEST_XDIST_AUTO_NUM_WORKERS"


def _available_cpu_count() -> int:
    sched_getaffinity = getattr(os, "sched_getaffinity", None)
    if sched_getaffinity is not None:
        try:
            return len(sched_getaffinity(0))
        except OSError:
            pass
    return os.cpu_count() or 1


# Runner-specific signals, not the generic ``CI`` variable. ``CI`` is routinely
# exported in developer shells (and inherited by every child process), so keying
# off it would unthrottle local runs and saturate the machine. Dedicated CI
# runners set one of these instead.
_CI_RUNNER_ENV_VARS = ("BUILDKITE", "GITHUB_ACTIONS")


def _is_ci() -> bool:
    return any(
        os.environ.get(var, "").lower() in {"1", "true", "yes"}
        for var in _CI_RUNNER_ENV_VARS
    )


def _default_worker_count(cpu_count: int) -> int:
    return max(1, math.ceil(cpu_count / 2) - 1)


@pytest.hookimpl(tryfirst=True)
def pytest_xdist_auto_num_workers() -> int:
    configured_worker_count = os.environ.get(_WORKER_COUNT_ENV_VAR)
    if configured_worker_count is not None:
        try:
            worker_count = int(configured_worker_count)
        except ValueError:
            worker_count = 0

        if worker_count > 0:
            return worker_count

        warnings.warn(
            f"{_WORKER_COUNT_ENV_VAR} must be a positive integer; using the default",
            stacklevel=2,
        )

    cpu_count = _available_cpu_count()
    # Only throttle interactive/local runs to keep developer machines usable.
    # CI runs the full suite and needs every available core for parallelism.
    if _is_ci():
        return cpu_count
    return _default_worker_count(cpu_count)


# --- Managed pytest run root (point-3 wiring, C4 chunk i) -------------------
#
# Every controller allocates a marked per-controller run root
# ``<temproot>/chartreux-managed-<uid>/pytest-run-<rand>`` and presets it as
# the factory basetemp, so no ``pytest-of-<user>`` numbered root is ever
# created. xdist workers receive their controller's root as an explicit
# ``popen-gwN`` basetemp (xdist workermanage) and only bind to it. All native
# pytest deletion routes (the green whole-basetemp rmtree at sessionfinish,
# the per-test fixture-dir teardown, the explicit-basetemp rm_rf) route through
# the C3 live-user gates with the owner self-excluded, so no deletion happens
# under a live attributable user while plain green runs still clean natively.


def _allocation_containing(target: Path, to: Any) -> Any | None:
    """The managed allocation containing ``target`` (the target may be the
    allocation root itself), or None when the target is outside the managed
    parent (explicit user destinations stay user-owned).
    """
    parent = to.managed_parent().resolve()
    current = Path(os.path.realpath(target))
    while current != parent:
        if current.parent == parent and to.is_managed_name(current.name):
            return to.allocation_from_path(current)
        if current.parent == current:
            return None
        current = current.parent
    return None


def _completion_record(entry: dict[str, Any]) -> dict[str, Any]:
    record: dict[str, Any] = {"pid": entry.get("pid")}
    if entry.get("no_tuple"):
        return record
    record["pgid"] = entry.get("pgid")
    record["starttime"] = entry.get("starttime")
    record["boot_id"] = entry.get("boot_id")
    return record


def _dir_gate_allows(alloc: Any, target: Path, me: Any, to: Any) -> bool:
    """The per-test teardown gate: ledger + recorded leader groups scoped to
    entries attributed (by invocation dir) to the finished dir. No
    full-containment sweep per test — allocation-level gates cover the root.
    """
    ledger = to.Ledger(alloc)
    entries = ledger.entries()
    target_str = str(target)

    def _inside(path: str) -> bool:
        return path == target_str or path.startswith(target_str + os.sep)

    mine = [
        entry
        for entry in entries
        if isinstance(entry.get("invocation_dir"), str)
        and _inside(entry["invocation_dir"])
    ]
    settled = {
        entry.get("spawn_id")
        for entry in entries
        if entry.get("kind") in {"completion", "retraction"}
    }
    for entry in mine:
        if entry.get("kind") == "intent" and entry.get("spawn_id") not in settled:
            return False
    dead_pids: set[int] = set()
    for entry in mine:
        if entry.get("kind") != "completion":
            continue
        record = _completion_record(entry)
        if record.get("pid") == me.pid:
            continue
        verdict = to.adjudicate(record)
        if verdict.live or not verdict.dead:
            return False
        if isinstance(record.get("pid"), int):
            dead_pids.add(record["pid"])
    for entry in mine:
        pgid = entry.get("pgid")
        if entry.get("is_leader_group") and isinstance(pgid, int):
            if to.leader_group_nonempty(
                pgid, exclude_pid=me.pid, dead_pids=frozenset(dead_pids)
            ):
                return False
    return True


def _install_gated_deletion(config: pytest.Config, alloc: Any, to: Any) -> None:
    """Route the native pytest deletion routes through the live-user gates."""
    import _pytest.tmpdir as tmpdir_module

    original_rmtree = tmpdir_module.rmtree
    original_rm_rf = tmpdir_module.rm_rf
    me = to.current_identity()

    def _retain_ambiguous(target: Path, exc: BaseException) -> None:
        # Retain-on-ambiguity: a deletion path that fails — including a
        # non-OSError raised by a test's globally monkeypatched fs call
        # (os.scandir/Path.mkdir/Path.open, e.g. pytest.fail's BaseException
        # outcome) inside the gate or the native rmtree — never fails
        # teardown; the tree is retained and reported instead.
        warnings.warn(
            f"chartreux retained managed path {target}: "
            f"deletion path failed ({exc!r}); retaining on ambiguity",
            stacklevel=2,
        )

    def _swallowable(exc: BaseException) -> bool:
        # Control-flow exceptions are never deletion-path failures.
        return not isinstance(
            exc, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)
        )

    def _gated_rmtree(path: Any, **kwargs: Any) -> Any:
        target = Path(path)
        try:
            owner_alloc = _allocation_containing(target, to)
            if owner_alloc is None:
                return original_rmtree(path, **kwargs)
            if target == owner_alloc.path:
                # Green whole-basetemp rmtree at sessionfinish: the full
                # allocation gate set with the live owner self-excluded.
                with to.parent_lock(owner_alloc.parent):
                    gates = to.evaluate_allocation_gates(owner_alloc, self_exclusion=me)
                if gates.reapable:
                    return original_rmtree(path, **kwargs)
                warnings.warn(
                    f"chartreux retained managed run root {target}: {gates.reasons}",
                    stacklevel=2,
                )
                return None
            # Per-test fixture-dir teardown: scoped gate only.
            if _dir_gate_allows(owner_alloc, target, me, to):
                return original_rmtree(path, **kwargs)
            warnings.warn(f"chartreux retained managed test dir {target}", stacklevel=2)
            return None
        except BaseException as exc:
            if not _swallowable(exc):
                raise
            _retain_ambiguous(target, exc)
            return None

    def _gated_rm_rf(path: Any) -> Any:
        target = Path(path)
        try:
            owner_alloc = _allocation_containing(target, to)
            if owner_alloc is None:
                return original_rm_rf(path)
            # Explicit basetemp rm_rf on first use: scoped gate plus a
            # containment sweep of the target itself (once per worker, not
            # per test).
            if not _dir_gate_allows(owner_alloc, target, me, to):
                warnings.warn(
                    f"chartreux retained explicit managed basetemp {target}",
                    stacklevel=2,
                )
                return None
            refs = to.containment_refs(target, exclude=me)
            if refs is None or refs:
                warnings.warn(
                    f"chartreux retained explicit managed basetemp {target}",
                    stacklevel=2,
                )
                return None
            return original_rm_rf(path)
        except BaseException as exc:
            if not _swallowable(exc):
                raise
            _retain_ambiguous(target, exc)
            return None

    tmpdir_module.rmtree = _gated_rmtree  # type: ignore[assignment]
    tmpdir_module.rm_rf = _gated_rm_rf  # type: ignore[assignment]

    def _restore() -> None:
        tmpdir_module.rmtree = original_rmtree  # type: ignore[assignment]
        tmpdir_module.rm_rf = original_rm_rf  # type: ignore[assignment]

    config.add_cleanup(_restore)


@pytest.hookimpl(trylast=True)
def pytest_configure(config: pytest.Config) -> None:
    """Preset the managed per-controller run root as the basetemp.

    ``trylast`` is load-bearing (the pinned pytest 9.1.1 ordering fact): a
    ``-p`` plugin's plain ``pytest_configure`` runs BEFORE the core tmpdir
    plugin attaches ``config._tmp_path_factory``, so the preset happens here —
    after the factory exists and before any first temp allocation. An explicit
    ``--basetemp`` always takes precedence and is never preset over.
    """
    try:
        from chartreux.core import tmp_ownership as to
        from chartreux.core.scratchpad import reap_stale_allocations
    except ImportError:
        # Repo-local plugin (ADJ-3): unpackaged installs never load it.
        return
    factory = getattr(config, "_tmp_path_factory", None)
    if factory is None:
        return
    given = config.option.basetemp
    if given is not None:
        # Explicit basetemp (xdist workers receive their controller's root as
        # a popen-gwN subdir): bind this process's spawns to the containing
        # managed allocation's ledger and gate its deletion routes, but never
        # allocate a second run root.
        alloc = _allocation_containing(Path(given), to)
        if alloc is not None:
            basetemp = Path(os.path.realpath(given))
            to.set_ambient_binding(
                to.SpawnLedgerBinding(alloc, invocation_dir=str(basetemp))
            )
            _install_gated_deletion(config, alloc, to)
        return
    # Invocation-time reaper before allocating the direct pytest run root.
    reap_stale_allocations()
    parent = to.managed_parent()
    try:
        alloc = to.init_allocation(parent, to.KIND_PYTEST_RUN_ROOT)
    except OSError:
        warnings.warn(
            "chartreux managed pytest run root unavailable; "
            "falling back to the default temp root",
            stacklevel=2,
        )
        return
    # The run root directory already exists (init_allocation created it);
    # preset _basetemp while keeping _given_basetemp unset so the native
    # sessionfinish cleanup stays live.
    factory._basetemp = alloc.path
    binding = to.SpawnLedgerBinding(alloc)
    to.set_ambient_binding(binding)
    me = to.current_identity()
    try:
        handle = binding.begin_spawn(
            spawn_id=f"controller-{os.getpid()}", is_leader_group=me.pgid == me.pid
        )
        binding.complete_spawn(handle, me.pid, me)
    except (OSError, to.SpawnRefusedError) as exc:
        warnings.warn(
            f"chartreux could not register the pytest controller: {exc}", stacklevel=2
        )
    _install_gated_deletion(config, alloc, to)
