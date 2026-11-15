from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import tempfile
import threading

from chartreux.core import tmp_ownership
from chartreux.core.tmp_ownership import (
    KIND_SCRATCHPAD,
    MANAGED_PARENT_PREFIX,
    Allocation,
    IdentityTuple,
    ParentLockBusy,
    ReapResult,
    SpawnLedgerBinding,
    allocation_from_path,
    current_identity,
    init_allocation,
    is_managed_name,
    managed_parent,
    parent_lock,
    reap_allocation,
    resume_quarantine_records,
    set_ambient_binding,
)
from chartreux.observability.logging import logger
from chartreux.utils.session_id import shorten_session_id

SCRATCHPAD_PREFIX = "chartreux-scratchpad-"
LEGACY_SCRATCHPAD_PREFIX = "vibe-scratchpad-"

SCRATCHPAD_INIT_MAX_ATTEMPTS = 8
"""Bounded in-session attempt budget for a boot-degraded scratchpad init.

The boot attempt consumes the first attempt, so a session that degraded at
boot gets at most ``SCRATCHPAD_INIT_MAX_ATTEMPTS - 1`` further attempts.
"""


def init_scratchpad(
    session_id: str, *, temp_root: Path | str | None = None
) -> Path | None:
    """Create the session's managed scratchpad allocation.

    The scratchpad is a marked allocation under the per-user managed parent
    (``<temproot>/chartreux-managed-<uid>/alloc-<rand>``) with a parent-beside
    ownership record and an in-tree spawn ledger, so the startup reaper can
    collect it once this session is conclusively dead. Initialization keeps
    the degrade-to-None behavior on any OSError (incl. ENOSPC).

    The init takes the parent lock NON-BLOCKING: a lock held across the
    startup steps (a concurrent reaper or allocator) fails closed with the
    same ``retained-lock-busy`` semantics as the sweep — the session starts
    without a scratchpad instead of queueing behind the holder, so the
    startup path is bounded under a held parent lock.
    """
    try:
        parent = managed_parent(
            temp_root if temp_root is not None else tempfile.gettempdir()
        )
        alloc = init_allocation(parent, KIND_SCRATCHPAD, blocking=False)
        logger.debug(
            "Scratchpad initialized at %s for session %s",
            alloc.path,
            shorten_session_id(session_id),
        )
        return alloc.path
    except ParentLockBusy:
        logger.warning(
            "Scratchpad init skipped: parent lock busy (retained-lock-busy); "
            "session %s starts without a scratchpad",
            shorten_session_id(session_id),
        )
        return None
    except OSError:
        logger.warning("Failed to create scratchpad directory")
        return None


@dataclass
class ScratchpadInitRetry:
    """Bounded in-session retry of a scratchpad init that degraded at boot.

    W10: a boot init that returned None (parent lock busy or OSError) is not
    permanent for the session. The session retries the init lazily — one
    NON-BLOCKING attempt per call — at most ``max_attempts`` times in total,
    boot attempt included. Every attempt is the same fail-closed
    :func:`init_scratchpad` path (non-blocking parent lock, degrade-to-None,
    nothing partial created under the lock), so the session never queues
    behind the holder and the blocking startup path is never reintroduced.
    Once the budget is spent, :meth:`attempt` is a no-op returning None: the
    session stays degraded instead of spinning.

    W11: the budget is also spent by the ``AgentLoop.scratchpad_dir``
    accessor on construction-time reads while degraded, so a read that
    happens before any turn still gets one bounded non-blocking retry. The
    spend is serialized by an internal lock, so concurrent readers can never
    race two successful attempts into two allocations.
    """

    session_id: str
    temp_root: Path | str | None = None
    max_attempts: int = SCRATCHPAD_INIT_MAX_ATTEMPTS
    attempts: int = 0
    _attempt_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False, compare=False
    )

    @property
    def exhausted(self) -> bool:
        """Whether the attempt budget is spent (stay degraded, never spin)."""
        return self.attempts >= self.max_attempts

    def attempt(self) -> Path | None:
        """Run one non-blocking init attempt.

        Returns the new scratchpad path on success, None while the parent
        lock is still busy or the budget is already spent (no call is made
        once exhausted). The budget check and the init run under one lock,
        so concurrent callers spend distinct attempts and at most one of
        them can observe a successful allocation.
        """
        with self._attempt_lock:
            if self.exhausted:
                return None
            self.attempts += 1
            path = init_scratchpad(self.session_id, temp_root=self.temp_root)
            if path is not None:
                logger.info(
                    "Scratchpad init recovered in-session after %d attempts for session %s",
                    self.attempts,
                    shorten_session_id(self.session_id),
                )
            return path


def _gated_close(alloc: Allocation, path: Path, me: IdentityTuple) -> bool:
    """The gated close transaction, split so no long operation holds the
    parent lock: the cheap gates (1..6) under the lock, the /proc containment
    scan outside it, then the cheap gates again under the lock together with
    the record-first quarantine transaction (d0..d2), and finally d3..d5
    (the rmtree) after the lock is released (W11 hold-shortening).
    """
    try:
        with parent_lock(alloc.parent):
            cheap = tmp_ownership._evaluate_cheap_gates(alloc, me)
            if cheap is not None:
                logger.info("Scratchpad %s retained on close: %s", path, cheap.reasons)
                return False
    except OSError:
        logger.warning("Failed to clean up scratchpad at %s", path)
        return False
    containment = tmp_ownership._gate_containment(alloc, me)
    if containment is not None:
        logger.info("Scratchpad %s retained on close: %s", path, containment.reasons)
        return False
    try:
        with parent_lock(alloc.parent):
            cheap = tmp_ownership._evaluate_cheap_gates(alloc, me)
            if cheap is not None:
                logger.info("Scratchpad %s retained on close: %s", path, cheap.reasons)
                return False
            # The C3 record-first quarantine transaction, reused for the
            # self-excluding close (reap_allocation has no self-exclusion).
            outcome = tmp_ownership._quarantine_transaction(
                alloc, deleter=me, record_written=False, defer_rmtree=True
            )
        if isinstance(outcome, tmp_ownership._PendingDeletion):
            outcome = tmp_ownership._delete_quarantine(outcome)
    except OSError:
        logger.warning("Failed to clean up scratchpad at %s", path)
        return False
    if outcome.reaped:
        logger.debug("Scratchpad reaped at %s", path)
    else:
        logger.info("Scratchpad %s cleanup deferred: %s", path, outcome.reasons)
    return outcome.reaped


def cleanup_scratchpad(scratchpad_dir: Path | None) -> bool:
    """Gated graceful close of one scratchpad allocation.

    The whole allocation is reaped only when every live-user gate passes with
    the closing owner self-excluded (no surviving registered spawn, no
    nonempty recorded leader group, no live containment reference, clean
    ledger); otherwise the entire marked allocation is retained for the
    startup reaper, which collects it after the survivor exits. Unmarked
    (legacy) trees are never deleted here — they are preview-only.
    """
    if scratchpad_dir is None:
        return False
    path = Path(scratchpad_dir)
    alloc = allocation_from_path(path)
    if not is_managed_name(alloc.name):
        logger.warning(
            "Refusing to clean up unmarked path %s (legacy trees are preview-only)",
            path,
        )
        return False
    return _gated_close(alloc, path, current_identity())


def reap_stale_allocations(temp_root: Path | str | None = None) -> list[ReapResult]:
    """The startup reaper: resume dangling quarantine records, then gate-reap
    every managed allocation under the parent.

    Runs at top-level session start and before a managed pytest run root is
    allocated. Live, indeterminate, and unmarked trees are retained
    (fail-closed); only conclusively-dead allocations with clean ledgers and
    no live users are removed. Never touches ``pytest-of-<user>`` roots or
    legacy unmarked trees.

    The sweep takes the machine-global parent lock NON-BLOCKING per entry:
    a busy lock (a concurrent reaper or allocator holds it) skips that entry
    and reports it ``retained-lock-busy`` — the entry is retained untouched
    and retried by the next sweep, so the retry cadence is unchanged and a
    boot under contention never queues behind the lock.
    """
    parent = managed_parent(
        temp_root if temp_root is not None else tempfile.gettempdir()
    )
    results = resume_quarantine_records(parent, blocking=False)
    for entry in sorted(parent.iterdir()):
        if not is_managed_name(entry.name) or not entry.is_dir():
            continue
        try:
            results.append(reap_allocation(allocation_from_path(entry), blocking=False))
        except OSError as exc:
            logger.warning("Startup reaper skipped %s: %s", entry, exc)
    return results


def bind_scratchpad(scratchpad_dir: Path | None) -> SpawnLedgerBinding | None:
    """Install the session allocation's ledger binding as this process's
    ambient binding, so context-less spawn sites append to the session ledger.
    """
    if scratchpad_dir is None:
        return None
    alloc = allocation_from_path(Path(scratchpad_dir))
    if not is_managed_name(alloc.name):
        return None
    binding = SpawnLedgerBinding(alloc)
    set_ambient_binding(binding)
    return binding


def unbind_scratchpad() -> None:
    """Drop the ambient ledger binding (graceful close)."""
    set_ambient_binding(None)


def is_scratchpad_path(path_str: str, *, scratchpad_dir: Path | None) -> bool:
    """Return whether a path is inside this runtime's scratchpad capability."""
    if scratchpad_dir is None:
        return False
    try:
        resolved = Path(path_str).expanduser().resolve()
        return _is_subpath(resolved, scratchpad_dir.resolve())
    except (ValueError, OSError):
        return False


def is_scratchpad_display_path(path_str: str) -> bool:
    try:
        path = Path(path_str).expanduser().resolve()
    except (ValueError, OSError):
        return False
    return any(
        part.startswith((
            SCRATCHPAD_PREFIX,
            LEGACY_SCRATCHPAD_PREFIX,
            MANAGED_PARENT_PREFIX,
        ))
        for part in path.parts
    )


def _is_subpath(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False
