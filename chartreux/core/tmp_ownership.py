"""Managed /tmp allocation ownership, durable spawn ledger, and gated reaping (v1).

Point-3 protocol of the v0.7.9 tmp-inode design. Every managed allocation
(session scratchpad, pytest run root) lives under a private per-user parent
``<temproot>/chartreux-managed-<uid>/`` and carries a parent-beside, versioned
ownership record published atomically under the parent allocation lock, so no
observer ever sees a partially initialized allocation. Reaping is fail-closed:
an allocation is removed only when the owner and every registered spawn are
conclusively dead, every recorded leader group is empty, the ledger is present
and not dirty, and no live reaper-readable process references the tree.

Spawns are registered record-before-spawn: the owning process appends an
INTENT entry to the allocation's ledger before the fork and completes it with
the child's full identity tuple (pid, pgid, /proc stat-22 starttime, boot id)
captured synchronously in the spawn frame; a spawn failure retracts the
intent. Any ledger-append failure flips the pre-established ledger-dirty flag
(a same-directory rename of an inode created at init — creation is what fails
under ENOSPC); if the intent append and the flag rename both fail before the
fork, the spawn is refused with a typed error and no child ever starts.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import errno
import fcntl
import json
import os
from pathlib import Path
import secrets
import shutil
import stat as stat_module
import tempfile
import threading
import time
from typing import Any

__all__ = [
    "DEAD",
    "INDETERMINATE",
    "LIVE",
    "Adjudication",
    "Allocation",
    "DeletionLease",
    "GateResult",
    "IdentityTuple",
    "Ledger",
    "OwnershipRecord",
    "ParentLockBusy",
    "ProcStat",
    "ProcUnreadable",
    "ReapResult",
    "SpawnLedgerBinding",
    "SpawnRefusedError",
    "acquire_deletion_lease",
    "adjudicate",
    "allocation_from_path",
    "ambient_binding",
    "binding_from_context",
    "boot_id",
    "containment_refs",
    "current_identity",
    "evaluate_allocation_gates",
    "evaluate_worker_scoped_gates",
    "init_allocation",
    "is_managed_name",
    "managed_parent",
    "parent_lock",
    "process_identity_tuple",
    "read_proc_stat",
    "reap_allocation",
    "resume_quarantine_records",
    "set_ambient_binding",
]

SCHEMA_VERSION = 1
MANAGED_PARENT_PREFIX = "chartreux-managed-"
ALLOC_PREFIX = "alloc-"
PYTEST_RUN_PREFIX = "pytest-run-"
PARENT_LOCK_FILENAME = ".alloc.lock"
OWNER_RECORD_SUFFIX = ".owner.json"
QREC_SUFFIX = ".qrec"
QLEASE_SUFFIX = ".qlease"
QUARANTINED_SUFFIX = ".quarantined"
LEDGER_DIRNAME = "ledger"
LEDGER_FILENAME = "ledger.jsonl"
LEDGER_LOCK_FILENAME = ".lock"
FLAG_CLEAN = ".flag-clean"
FLAG_DIRTY = ".flag-dirty"
KIND_SCRATCHPAD = "scratchpad"
KIND_PYTEST_RUN_ROOT = "pytest-run-root"

DEAD = "dead"
LIVE = "live"
INDETERMINATE = "indeterminate"


# Filesystem seams the ENOSPC matrix drives: the spawn-side append and the
# dirty-flag rename are the two operations whose failure the protocol must
# classify, so both route through module-level functions tests can fail.
def _os_write(fd: int, data: bytes) -> int:
    return os.write(fd, data)


def _os_rename(src: Path, dst: Path) -> None:
    os.rename(src, dst)


_boot_id_cache: str | None = None
_boot_id_resolved = False


def boot_id() -> str | None:
    """The boot identity from /proc, cached: it cannot change within a boot."""
    global _boot_id_cache, _boot_id_resolved
    if _boot_id_resolved:
        return _boot_id_cache
    try:
        with open("/proc/sys/kernel/random/boot_id") as stream:
            _boot_id_cache = stream.read().strip()
    except OSError:
        _boot_id_cache = None
    _boot_id_resolved = True
    return _boot_id_cache


def btime() -> int | None:
    """The boot time from /proc/stat, or None when unreadable."""
    try:
        with open("/proc/stat") as stream:
            for line in stream:
                if line.startswith("btime "):
                    return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        return None
    return None


@dataclass(frozen=True, slots=True)
class ProcStat:
    """The identity-bearing /proc/<pid>/stat fields (3, 4, 5, 22)."""

    state: str
    pgrp: int
    session: int
    starttime: int


class ProcUnreadable(Exception):
    """The pid exists but its stat is unreadable or malformed."""


def read_proc_stat(pid: int) -> ProcStat | None:
    """Read /proc/<pid>/stat.

    ``None`` means the pid is gone (FileNotFoundError/ProcessLookupError);
    :class:`ProcUnreadable` means it is present but cannot be read or parsed,
    which is inaccessibility, never proof of death.
    """
    try:
        with open(f"/proc/{pid}/stat", "rb") as stream:
            raw = stream.read()
    except (FileNotFoundError, ProcessLookupError):
        return None
    except OSError as exc:
        raise ProcUnreadable(str(exc)) from exc
    try:
        fields = raw.decode("utf-8", "surrogateescape").rsplit(")", 1)[1].split()
        return ProcStat(
            state=fields[0],
            pgrp=int(fields[2]),
            session=int(fields[3]),
            starttime=int(fields[19]),
        )
    except (ValueError, IndexError) as exc:
        raise ProcUnreadable("malformed /proc stat") from exc


@dataclass(frozen=True, slots=True)
class IdentityTuple:
    """The full process identity: pid, pgid, starttime (stat field 22), boot id."""

    pid: int
    pgid: int
    starttime: int
    boot_id: str

    def to_json(self) -> dict[str, Any]:
        return {
            "pid": self.pid,
            "pgid": self.pgid,
            "starttime": self.starttime,
            "boot_id": self.boot_id,
        }

    @classmethod
    def from_json(cls, data: Any) -> IdentityTuple | None:
        if not isinstance(data, Mapping):
            return None
        pid = data.get("pid")
        pgid = data.get("pgid")
        starttime = data.get("starttime")
        boot = data.get("boot_id")
        if not isinstance(pid, int) or isinstance(pid, bool):
            return None
        if not isinstance(pgid, int) or isinstance(pgid, bool):
            return None
        if not isinstance(starttime, int) or isinstance(starttime, bool):
            return None
        if not isinstance(boot, str):
            return None
        return cls(pid=pid, pgid=pgid, starttime=starttime, boot_id=boot)


def process_identity_tuple(pid: int) -> IdentityTuple | None:
    """The full identity tuple of a live pid, or None when it cannot be read.

    Called synchronously in the spawn frame: a fast-exiting child stays
    readable as a zombie until its spawner reaps it, so an in-frame read is
    the only capture that never misses a started child.
    """
    try:
        stat = read_proc_stat(pid)
    except ProcUnreadable:
        return None
    if stat is None:
        return None
    return IdentityTuple(
        pid=pid, pgid=stat.pgrp, starttime=stat.starttime, boot_id=boot_id() or ""
    )


def current_identity(pid: int | None = None) -> IdentityTuple:
    """The calling process's own identity tuple (self /proc is always readable)."""
    pid = os.getpid() if pid is None else pid
    stat = read_proc_stat(pid)
    if stat is not None:
        return IdentityTuple(
            pid=pid, pgid=stat.pgrp, starttime=stat.starttime, boot_id=boot_id() or ""
        )
    try:
        pgid = os.getpgid(pid)
    except OSError:
        pgid = pid
    return IdentityTuple(pid=pid, pgid=pgid, starttime=0, boot_id=boot_id() or "")


@dataclass(frozen=True, slots=True)
class Adjudication:
    """What a recorded identity proves against live /proc."""

    verdict: str
    reason: str

    @property
    def dead(self) -> bool:
        return self.verdict == DEAD

    @property
    def live(self) -> bool:
        return self.verdict == LIVE


def _adjudicate_via_sig0(pid: int) -> Adjudication:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return Adjudication(DEAD, "pid-gone")
    except PermissionError:
        return Adjudication(INDETERMINATE, "sig0-eperm")
    except OSError:
        return Adjudication(INDETERMINATE, "sig0-error")
    return Adjudication(INDETERMINATE, "stat-unreadable")


def _adjudicate_stat(recorded: Mapping[str, Any], stat: ProcStat) -> Adjudication:
    if stat.state == "Z":
        return Adjudication(DEAD, "zombie")
    starttime = recorded.get("starttime")
    if starttime is None:
        # NO-TUPLE completion: a readable live pid is indeterminate; only
        # affirmative exit evidence (gone pid / zombie) proves death.
        return Adjudication(INDETERMINATE, "no-tuple-live")
    if not isinstance(starttime, int) or isinstance(starttime, bool):
        return Adjudication(INDETERMINATE, "malformed-record")
    if stat.starttime != starttime:
        return Adjudication(DEAD, "pid-reuse")
    return Adjudication(LIVE, "tuple-match")


def _adjudicate_pid(recorded: Mapping[str, Any], pid: int) -> Adjudication:
    boot = boot_id()
    if boot is None:
        return Adjudication(INDETERMINATE, "boot-id-unreadable")
    recorded_boot = recorded.get("boot_id")
    if isinstance(recorded_boot, str) and recorded_boot and recorded_boot != boot:
        return Adjudication(DEAD, "boot-change")
    try:
        stat = read_proc_stat(pid)
    except ProcUnreadable:
        return _adjudicate_via_sig0(pid)
    if stat is None:
        return _adjudicate_via_sig0(pid)
    return _adjudicate_stat(recorded, stat)


def adjudicate(recorded: Mapping[str, Any] | None) -> Adjudication:
    """Adjudicate one recorded identity dict against live /proc.

    Verdicts: ``dead`` (conclusive: pid gone, zombie, pid reuse by starttime,
    or boot change), ``live`` (tuple matches a non-zombie pid), or
    ``indeterminate`` (unreadable/malformed record, sig0 EPERM, NO-TUPLE with
    a readable pid). Indeterminate always fails closed toward retention.
    """
    if not isinstance(recorded, Mapping):
        return Adjudication(INDETERMINATE, "malformed-record")
    pid = recorded.get("pid")
    if not isinstance(pid, int) or isinstance(pid, bool):
        return Adjudication(INDETERMINATE, "malformed-record")
    return _adjudicate_pid(recorded, pid)


def is_managed_name(name: str) -> bool:
    """Whether a parent-level entry name is a managed allocation."""
    return name.startswith((ALLOC_PREFIX, PYTEST_RUN_PREFIX))


def managed_parent(temp_root: Path | str | None = None) -> Path:
    """The per-user managed parent on the existing temp filesystem (0700)."""
    root = Path(temp_root) if temp_root is not None else Path(tempfile.gettempdir())
    parent = root / f"{MANAGED_PARENT_PREFIX}{os.getuid()}"
    parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        os.chmod(parent, 0o700)
    except OSError:
        pass
    return parent


class ParentLockBusy(Exception):
    """A non-blocking parent-lock acquisition failed: another actor holds the
    lock, so the caller must skip and retain, never wait.
    """


@contextmanager
def parent_lock(parent: Path, *, blocking: bool = True) -> Iterator[None]:
    """The parent-level allocation lock (flock, exclusive) serializing init,
    publication, validation, and the quarantine transaction.

    ``blocking=False`` never waits: a busy lock raises :class:`ParentLockBusy`
    so the reaper sweep can skip and retain the entry (fail-closed) instead
    of queueing behind the holder. The default stays blocking, so every
    pre-existing caller is byte-compatible.
    """
    lock_path = parent / PARENT_LOCK_FILENAME
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        if blocking:
            fcntl.flock(fd, fcntl.LOCK_EX)
        else:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise ParentLockBusy(f"parent lock busy: {lock_path} ({exc})") from exc
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """tmp + fsync + rename publication; no observer sees a partial record."""
    tmp = path.with_name(path.name + ".tmp")
    data = json.dumps(dict(payload), separators=(",", ":")).encode()
    with open(tmp, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _ownership_scalars(
    data: Mapping[str, Any],
) -> tuple[str, str, float, int | None, str] | None:
    alloc_id = data.get("alloc_id")
    kind = data.get("kind")
    created = data.get("created_epoch")
    btime_value = data.get("btime")
    platform = data.get("platform")
    if not isinstance(alloc_id, str) or not isinstance(kind, str):
        return None
    if not isinstance(created, (int, float)) or isinstance(created, bool):
        return None
    if btime_value is not None and not isinstance(btime_value, int):
        return None
    if not isinstance(platform, str):
        return None
    return alloc_id, kind, float(created), btime_value, platform


def _ownership_record_from(
    data: Mapping[str, Any], owner: IdentityTuple
) -> OwnershipRecord | None:
    scalars = _ownership_scalars(data)
    if scalars is None:
        return None
    alloc_id, kind, created, btime_value, platform = scalars
    return OwnershipRecord(
        schema=SCHEMA_VERSION,
        alloc_id=alloc_id,
        kind=kind,
        owner=owner,
        created_epoch=created,
        btime=btime_value,
        platform=platform,
    )


@dataclass(frozen=True, slots=True)
class OwnershipRecord:
    """Ownership record v1, parent-beside the allocation tree."""

    schema: int
    alloc_id: str
    kind: str
    owner: IdentityTuple
    created_epoch: float
    btime: int | None
    platform: str

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "alloc_id": self.alloc_id,
            "kind": self.kind,
            "owner": self.owner.to_json(),
            "created_epoch": self.created_epoch,
            "btime": self.btime,
            "platform": self.platform,
        }

    @classmethod
    def from_json(cls, data: Any) -> OwnershipRecord | None:
        if not isinstance(data, Mapping) or data.get("schema") != SCHEMA_VERSION:
            return None
        owner = IdentityTuple.from_json(data.get("owner"))
        if owner is None:
            return None
        return _ownership_record_from(data, owner)


def load_ownership_record(parent: Path, name: str) -> OwnershipRecord | None:
    """The parent-beside ownership record, or None when missing/malformed."""
    path = parent / f"{name}{OWNER_RECORD_SUFFIX}"
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    try:
        return OwnershipRecord.from_json(json.loads(raw))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


@dataclass(frozen=True, slots=True)
class Allocation:
    """One managed allocation and all its derived parent-beside paths."""

    parent: Path
    name: str
    kind: str
    alloc_id: str

    @property
    def path(self) -> Path:
        return self.parent / self.name

    @property
    def owner_record_path(self) -> Path:
        return self.parent / f"{self.name}{OWNER_RECORD_SUFFIX}"

    @property
    def qrec_path(self) -> Path:
        return self.parent / f"{self.name}{QREC_SUFFIX}"

    @property
    def qlease_path(self) -> Path:
        return self.parent / f"{self.name}{QLEASE_SUFFIX}"

    @property
    def quarantine_path(self) -> Path:
        return self.parent / f"{self.name}{QUARANTINED_SUFFIX}"

    @property
    def ledger_dir(self) -> Path:
        return self.path / LEDGER_DIRNAME

    @property
    def ledger_path(self) -> Path:
        return self.ledger_dir / LEDGER_FILENAME

    @property
    def ledger_lock_path(self) -> Path:
        return self.ledger_dir / LEDGER_LOCK_FILENAME

    @property
    def flag_clean_path(self) -> Path:
        return self.ledger_dir / FLAG_CLEAN

    @property
    def flag_dirty_path(self) -> Path:
        return self.ledger_dir / FLAG_DIRTY


def new_alloc_id() -> str:
    return secrets.token_hex(6)


def allocation_from_path(path: Path) -> Allocation:
    """Reconstruct the Allocation handle for a managed path."""
    name = path.name
    kind = (
        KIND_PYTEST_RUN_ROOT if name.startswith(PYTEST_RUN_PREFIX) else KIND_SCRATCHPAD
    )
    alloc_id = name.rsplit("-", 1)[-1]
    return Allocation(parent=path.parent, name=name, kind=kind, alloc_id=alloc_id)


def init_allocation(
    parent: Path,
    kind: str,
    *,
    owner: IdentityTuple | None = None,
    alloc_id: str | None = None,
    blocking: bool = True,
) -> Allocation:
    """Create and publish one managed allocation under the parent lock.

    The directory, the ledger (with its lock and the PRE-ESTABLISHED dirty
    flag inode), and the parent-beside ownership record are all published
    inside the one lock, so a concurrent reaper never observes partial init.

    ``blocking=False`` never waits: a busy parent lock raises
    :class:`ParentLockBusy` before anything is created, so the startup init
    path can fail closed (degrade) instead of queueing behind the holder.
    The default stays blocking, so every pre-existing caller is
    byte-compatible.
    """
    prefix = PYTEST_RUN_PREFIX if kind == KIND_PYTEST_RUN_ROOT else ALLOC_PREFIX
    with parent_lock(parent, blocking=blocking):
        alloc_id = alloc_id or new_alloc_id()
        name = f"{prefix}{alloc_id}"
        alloc = Allocation(parent=parent, name=name, kind=kind, alloc_id=alloc_id)
        alloc.path.mkdir(mode=0o700)
        alloc.ledger_dir.mkdir(mode=0o700)
        alloc.ledger_path.touch()
        alloc.ledger_lock_path.touch()
        alloc.flag_clean_path.touch()
        record = OwnershipRecord(
            schema=SCHEMA_VERSION,
            alloc_id=alloc_id,
            kind=kind,
            owner=owner or current_identity(),
            created_epoch=time.time(),
            btime=btime(),
            platform="linux",
        )
        _write_json_atomic(alloc.owner_record_path, record.to_json())
    return alloc


class Ledger:
    """The allocation's JSONL registration ledger (O_APPEND, fsynced lines,
    serialized under the ledger's own flock).
    """

    def __init__(self, allocation: Allocation) -> None:
        self.allocation = allocation

    @property
    def path(self) -> Path:
        return self.allocation.ledger_path

    @property
    def lock_path(self) -> Path:
        return self.allocation.ledger_lock_path

    def append(self, entry: Mapping[str, Any]) -> None:
        """Append one entry; raises OSError (incl. ENOSPC) on any failure."""
        line = (json.dumps(dict(entry), separators=(",", ":")) + "\n").encode()
        fd = os.open(self.lock_path, os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            try:
                wfd = os.open(self.path, os.O_WRONLY | os.O_APPEND)
                try:
                    written = 0
                    while written < len(line):
                        written += _os_write(wfd, line[written:])
                    os.fsync(wfd)
                finally:
                    os.close(wfd)
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    def entries(self) -> list[dict[str, Any]]:
        """All well-formed entries; torn or malformed lines are skipped."""
        try:
            raw = self.path.read_bytes()
        except OSError:
            return []
        out: list[dict[str, Any]] = []
        for line in raw.splitlines():
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if isinstance(value, dict):
                out.append(value)
        return out

    def is_dirty(self) -> bool:
        return self.allocation.flag_dirty_path.exists()

    def mark_dirty(self) -> None:
        """Flip the pre-established flag clean -> dirty (a rename, never a create)."""
        if self.allocation.flag_dirty_path.exists():
            return
        if not self.allocation.flag_clean_path.exists():
            raise OSError(errno.ENOENT, "pre-established dirty flag missing")
        _os_rename(self.allocation.flag_clean_path, self.allocation.flag_dirty_path)

    def clear_dirty(self) -> None:
        """The single-appender RESYNC clear: dirty -> clean (a rename back)."""
        if self.allocation.flag_clean_path.exists():
            return
        _os_rename(self.allocation.flag_dirty_path, self.allocation.flag_clean_path)


class SpawnRefusedError(RuntimeError):
    """Typed pre-fork refusal: the intent append failed AND the pre-established
    dirty flag could not be flipped either, so the spawn is refused before the
    fork and no child ever starts.
    """


@dataclass(frozen=True, slots=True)
class SpawnHandle:
    """One in-flight registered spawn (an intent entry)."""

    spawn_id: str
    is_leader_group: bool
    invocation_dir: str | None


class SpawnLedgerBinding:
    """The ledger binding an invoking context carries: every spawn site appends
    to the ledger of the allocation bound to its context (never the command's
    cwd alone).
    """

    def __init__(
        self,
        allocation: Allocation,
        *,
        owner: IdentityTuple | None = None,
        invocation_dir: str | None = None,
    ) -> None:
        self.allocation = allocation
        self.ledger = Ledger(allocation)
        self.owner = owner or current_identity()
        self.invocation_dir = invocation_dir

    def _base_entry(self, kind: str, spawn_id: str) -> dict[str, Any]:
        return {
            "v": SCHEMA_VERSION,
            "epoch": time.time(),
            "kind": kind,
            "spawn_id": spawn_id,
            "appender": self.owner.to_json(),
        }

    def begin_spawn(
        self,
        *,
        spawn_id: str | None = None,
        is_leader_group: bool = False,
        invocation_dir: str | None = None,
    ) -> SpawnHandle:
        """Record the INTENT before the fork.

        Raises :class:`SpawnRefusedError` when the append fails and the dirty
        flag cannot be flipped either (pre-fork, so refusal is possible). When
        only the append fails, the flag flips and the spawn proceeds
        recorded-dirty — the ledger still tells the truth.
        """
        handle = SpawnHandle(
            spawn_id=spawn_id or secrets.token_hex(8),
            is_leader_group=is_leader_group,
            invocation_dir=invocation_dir or self.invocation_dir,
        )
        entry = self._base_entry("intent", handle.spawn_id)
        entry["spawner"] = self.owner.to_json()
        entry["is_leader_group"] = handle.is_leader_group
        entry["invocation_dir"] = handle.invocation_dir
        try:
            self.ledger.append(entry)
        except OSError:
            try:
                self.ledger.mark_dirty()
            except OSError as flag_error:
                raise SpawnRefusedError(
                    f"spawn refused before fork: ledger append failed "
                    f"({flag_error}) and the dirty flag could not be flipped; "
                    f"no child was started"
                ) from flag_error
        return handle

    def complete_spawn(
        self, handle: SpawnHandle, pid: int, identity: IdentityTuple | None
    ) -> bool:
        """Complete the intent with the child's full tuple, captured in-frame.

        A failed read is a NO-TUPLE completion (child provably started). A
        failed append post-spawn cannot refuse a started child: the flag
        flips and the entry stays provisionally uncompleted — the completion
        is never re-appended (the next ledger interaction lands at most a
        RESYNC annotation), so the uncompleted intent keeps every reap gate
        retaining and reporting the allocation for the remainder of the boot
        (INDETERMINATE ⇒ retain; over-retention is the safe direction).
        """
        entry = self._base_entry("completion", handle.spawn_id)
        entry["pid"] = pid
        if identity is not None:
            entry["pgid"] = identity.pgid
            entry["starttime"] = identity.starttime
            entry["boot_id"] = identity.boot_id
            entry["no_tuple"] = False
        else:
            entry["no_tuple"] = True
        entry["is_leader_group"] = handle.is_leader_group
        entry["invocation_dir"] = handle.invocation_dir
        try:
            self.ledger.append(entry)
        except OSError:
            try:
                self.ledger.mark_dirty()
            except OSError:
                pass
            return False
        return True

    def retract_spawn(self, handle: SpawnHandle) -> None:
        """Retract the intent after a spawn failure (no child ever started)."""
        entry = self._base_entry("retraction", handle.spawn_id)
        try:
            self.ledger.append(entry)
        except OSError:
            try:
                self.ledger.mark_dirty()
            except OSError:
                pass

    def resync(self) -> bool:
        """The single-appender RESYNC clear.

        Only when every entry's appender is the owner's own may the flag be
        un-flipped; on a shared/foreign ledger the resync lands as an advisory
        annotation and the flag stays dirty for the remainder of the boot.
        """
        entries = self.ledger.entries()
        owner_json = self.owner.to_json()
        single_appender = all(
            entry.get("appender") == owner_json
            for entry in entries
            if "appender" in entry
        ) and all("appender" in entry for entry in entries)
        entry = self._base_entry("resync", secrets.token_hex(8))
        try:
            self.ledger.append(entry)
        except OSError:
            return False
        if not single_appender:
            return False
        try:
            self.ledger.clear_dirty()
        except OSError:
            return False
        return True


_AMBIENT_BINDING: SpawnLedgerBinding | None = None
_AMBIENT_LOCK = threading.Lock()


def set_ambient_binding(binding: SpawnLedgerBinding | None) -> None:
    """Install this process's own allocation binding (the invoking-context
    binding for spawn sites that carry no context object).
    """
    global _AMBIENT_BINDING
    with _AMBIENT_LOCK:
        _AMBIENT_BINDING = binding


def ambient_binding() -> SpawnLedgerBinding | None:
    with _AMBIENT_LOCK:
        return _AMBIENT_BINDING


def binding_from_context(ctx: Any) -> SpawnLedgerBinding | None:
    """The ledger binding of an invoking context: the context's delivered field
    wins; otherwise the process's ambient binding.
    """
    binding = getattr(ctx, "tmp_ledger_binding", None)
    if binding is not None:
        return binding
    return ambient_binding()


@dataclass(frozen=True, slots=True)
class GateResult:
    """The reap-gate verdict: reapable, plus the §9 report vocabulary reasons."""

    reapable: bool
    reasons: tuple[str, ...]


def _path_inside(target: str, root: str) -> bool:
    return target == root or target.startswith(root + os.sep)


def _pid_refs_inside(pid: int, root: str) -> bool:
    for link in ("cwd", "root"):
        try:
            target = os.readlink(f"/proc/{pid}/{link}")
        except (FileNotFoundError, ProcessLookupError):
            return False
        except OSError:
            continue
        if _path_inside(target, root):
            return True
    try:
        fd_names = os.listdir(f"/proc/{pid}/fd")
    except OSError:
        return False
    for fd_name in fd_names:
        try:
            target = os.readlink(f"/proc/{pid}/fd/{fd_name}")
        except OSError:
            continue
        if _path_inside(target, root):
            return True
    return False


def containment_refs(
    candidate: Path, *, exclude: IdentityTuple | None = None
) -> list[int] | None:
    """Live reaper-readable pids whose cwd, root, or an open fd resolves inside
    the candidate. ``None`` means the sweep itself was indeterminate.

    Entries unreadable to the reaper are out of the detection input (the
    accepted residual); a pid vanishing mid-sweep is benign.
    """
    try:
        root = os.path.realpath(candidate)
        names = os.listdir("/proc")
    except OSError:
        return None
    refs: list[int] = []
    for name in names:
        if not name.isdecimal():
            continue
        pid = int(name)
        if exclude is not None and pid == exclude.pid:
            continue
        if _pid_refs_inside(pid, root):
            refs.append(pid)
    return refs


def leader_group_nonempty(
    pgid: int,
    *,
    exclude_pid: int | None = None,
    dead_pids: frozenset[int] = frozenset(),
) -> bool:
    """Numeric leader-group probe: killpg(pgid, 0) ESRCH means empty, EPERM
    means treat-alive (fail closed); otherwise scan members, discounting
    zombies, the excluded pid, and recorded identities already proven dead.
    """
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    try:
        names = os.listdir("/proc")
    except OSError:
        return True
    for name in names:
        if not name.isdecimal():
            continue
        pid = int(name)
        if pid == exclude_pid or pid in dead_pids:
            continue
        try:
            member = read_proc_stat(pid)
        except ProcUnreadable:
            continue
        if member is not None and member.pgrp == pgid and member.state != "Z":
            return True
    return False


def _gate_name_and_record(alloc: Allocation) -> GateResult | None:
    if not is_managed_name(alloc.name):
        return GateResult(False, ("not-managed-name",))
    if load_ownership_record(alloc.parent, alloc.name) is None:
        return GateResult(False, ("torn-init",))
    return None


def _gate_owner(
    alloc: Allocation, self_exclusion: IdentityTuple | None
) -> GateResult | None:
    record = load_ownership_record(alloc.parent, alloc.name)
    assert record is not None
    if self_exclusion is not None and record.owner == self_exclusion:
        return None
    verdict = adjudicate(record.owner.to_json())
    if verdict.live:
        return GateResult(False, ("retained-live",))
    if not verdict.dead:
        return GateResult(False, ("retained-owner-indeterminate", verdict.reason))
    return None


def _gate_symlink(alloc: Allocation) -> GateResult | None:
    try:
        info = os.lstat(alloc.path)
    except FileNotFoundError:
        return GateResult(False, ("candidate-gone",))
    if stat_module.S_ISLNK(info.st_mode):
        return GateResult(False, ("retained-symlink-substitution",))
    return None


def _uncompleted_intents(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    settled = {
        entry.get("spawn_id")
        for entry in entries
        if entry.get("kind") in {"completion", "retraction"}
    }
    return [
        entry
        for entry in entries
        if entry.get("kind") == "intent" and entry.get("spawn_id") not in settled
    ]


def _gate_ledger(ledger: Ledger) -> GateResult | None:
    if not ledger.path.exists():
        return GateResult(False, ("retained-ledger-missing",))
    if ledger.is_dirty():
        return GateResult(False, ("retained-ledger-dirty",))
    reasons: list[str] = []
    for entry in _uncompleted_intents(ledger.entries()):
        spawner = entry.get("spawner")
        verdict = adjudicate(spawner) if isinstance(spawner, Mapping) else None
        if verdict is not None and verdict.dead:
            reasons.append("UNRESOLVED-INTENT")
        else:
            reasons.append("retained-uncompleted-intent")
    if reasons:
        return GateResult(False, tuple(reasons))
    return None


def _completion_record(entry: Mapping[str, Any]) -> dict[str, Any]:
    record = {"pid": entry.get("pid")}
    if entry.get("no_tuple"):
        return record
    record["pgid"] = entry.get("pgid")
    record["starttime"] = entry.get("starttime")
    record["boot_id"] = entry.get("boot_id")
    return record


def _gate_completions(
    entries: list[dict[str, Any]], self_exclusion: IdentityTuple | None
) -> GateResult | None:
    for entry in entries:
        if entry.get("kind") != "completion":
            continue
        record = _completion_record(entry)
        if self_exclusion is not None and record.get("pid") == self_exclusion.pid:
            continue
        verdict = adjudicate(record)
        if verdict.live:
            return GateResult(False, ("retained-registered-spawn",))
        if not verdict.dead:
            return GateResult(
                False, ("retained-registered-spawn-indeterminate", verdict.reason)
            )
    return None


def _dead_completion_pids(entries: list[dict[str, Any]]) -> frozenset[int]:
    """The pids of completions already adjudicated conclusively dead."""
    dead: set[int] = set()
    for entry in entries:
        if entry.get("kind") != "completion":
            continue
        pid = entry.get("pid")
        if not isinstance(pid, int) or isinstance(pid, bool):
            continue
        if adjudicate(_completion_record(entry)).dead:
            dead.add(pid)
    return frozenset(dead)


def _gate_leader_groups(
    entries: list[dict[str, Any]], self_exclusion: IdentityTuple | None
) -> GateResult | None:
    dead_pids = _dead_completion_pids(entries)
    exclude_pid = self_exclusion.pid if self_exclusion is not None else None
    for entry in entries:
        if not entry.get("is_leader_group"):
            continue
        pgid = entry.get("pgid")
        if not isinstance(pgid, int) or isinstance(pgid, bool):
            continue
        if leader_group_nonempty(pgid, exclude_pid=exclude_pid, dead_pids=dead_pids):
            return GateResult(False, ("retained-leader-group-nonempty",))
    return None


def _gate_containment(
    alloc: Allocation, self_exclusion: IdentityTuple | None
) -> GateResult | None:
    refs = containment_refs(alloc.path, exclude=self_exclusion)
    if refs is None:
        return GateResult(False, ("retained-containment-indeterminate",))
    if refs:
        return GateResult(False, ("retained-containment-ref",))
    return None


def _evaluate_cheap_gates(
    alloc: Allocation, self_exclusion: IdentityTuple | None
) -> GateResult | None:
    """Gates 1..6 of the pinned reap set (everything but the /proc containment
    scan), in the pinned order. Returns the first failing :class:`GateResult`,
    or None when every cheap gate passes.
    """
    ledger = Ledger(alloc)
    gates = (
        lambda: _gate_name_and_record(alloc),
        lambda: _gate_owner(alloc, self_exclusion),
        lambda: _gate_symlink(alloc),
        lambda: _gate_ledger(ledger),
        lambda: _gate_completions(ledger.entries(), self_exclusion),
        lambda: _gate_leader_groups(ledger.entries(), self_exclusion),
    )
    for gate in gates:
        result = gate()
        if result is not None:
            return result
    return None


def evaluate_allocation_gates(
    alloc: Allocation, *, self_exclusion: IdentityTuple | None = None
) -> GateResult:
    """The allocation-wide reap gate set, in the pinned order, all evaluated
    under the parent lock by the caller.

    ``self_exclusion`` discounts the reaper's own identity in the owner check,
    ledger membership, leader groups, and containment sweep (the pytest
    run-root case: the live owner cleans its own root natively).

    The locked reap path (:func:`reap_allocation`) evaluates the same set in
    two phases so the /proc containment scan — the one gate that can take
    tens of milliseconds — never runs under the parent lock (W11); this
    entry point keeps the full set in one call for direct callers.
    """
    cheap = _evaluate_cheap_gates(alloc, self_exclusion)
    if cheap is not None:
        return cheap
    containment = _gate_containment(alloc, self_exclusion)
    if containment is not None:
        return containment
    return GateResult(True, (f"reaped-{alloc.alloc_id}",))


# --- C3b: record-first quarantine transaction, deletion lease, resume --------


@dataclass(frozen=True, slots=True)
class ReapResult:
    """The outcome of one gated reap attempt."""

    reaped: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _PendingDeletion:
    """A quarantine transaction whose d0..d2 completed under the parent lock.

    The physical d3..d5 (rmtree, tree-gone mark, record cleanup) are run by
    the caller via :func:`_delete_quarantine` AFTER the parent lock is
    released, so deleting a big tree never holds the machine-global lock.
    The deletion lease taken at d2 still serializes deleters, and a crash
    before d3 leaves exactly the resumable residue the next sweep finishes.
    """

    alloc: Allocation
    lease: DeletionLease


def _unlink_quietly(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


def _qrec_payload(
    alloc: Allocation, deleter: IdentityTuple, state: str
) -> dict[str, Any]:
    record = load_ownership_record(alloc.parent, alloc.name)
    validated = record.owner.to_json() if record is not None else None
    return {
        "schema": SCHEMA_VERSION,
        "alloc_id": alloc.alloc_id,
        "candidate": str(alloc.path),
        "quarantine": str(alloc.quarantine_path),
        "validated_identity": validated,
        "state": state,
        "epoch": time.time(),
        "deleter": deleter.to_json(),
    }


def _mark_qrec_state(alloc: Allocation, state: str) -> None:
    try:
        raw = alloc.qrec_path.read_bytes()
        payload = json.loads(raw)
        payload["state"] = state
        _write_json_atomic(alloc.qrec_path, payload)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        # The state is advisory: resume re-derives it from the filesystem.
        pass


class DeletionLease:
    """An exclusive flock on the lease file plus the holder's identity record."""

    def __init__(self, fd: int, path: Path, holder: IdentityTuple) -> None:
        self._fd = fd
        self.path = path
        self.holder = holder

    def release(self) -> None:
        try:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            os.close(self._fd)


def _read_lease_holder(fd: int) -> Mapping[str, Any] | None:
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        raw = os.read(fd, 4096)
    except OSError:
        return None
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if isinstance(value, Mapping) and isinstance(value.get("holder"), Mapping):
        return value["holder"]
    return None


def acquire_deletion_lease(path: Path, holder: IdentityTuple) -> DeletionLease | None:
    """Acquire the exclusive deletion lease, or None when blocked.

    A second simultaneous deleter is blocked while the holder is live. After
    the holder's hard death the kernel has released the flock, so the lease is
    re-acquirable — but the recorded holder identity must adjudicate
    CONCLUSIVE DEAD before takeover (a stale reuse-stamp adjudicates dead, not
    live); a NO-TUPLE or live-match record retains.
    """
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        return None
    record = _read_lease_holder(fd)
    if record is not None and not adjudicate(record).dead:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
        return None
    payload = json.dumps({
        "schema": SCHEMA_VERSION,
        "holder": holder.to_json(),
        "epoch": time.time(),
    }).encode()
    os.lseek(fd, 0, os.SEEK_SET)
    os.ftruncate(fd, 0)
    os.write(fd, payload)
    os.fsync(fd)
    return DeletionLease(fd, path, holder)


def _quarantine_transaction(
    alloc: Allocation,
    *,
    deleter: IdentityTuple,
    record_written: bool,
    defer_rmtree: bool = False,
) -> ReapResult | _PendingDeletion:
    """The d0..d5 record-first quarantine sequence.

    d0 writes the parent-beside authorization record FIRST (a reaper-side
    failure here aborts with the allocation untouched); d1 renames the
    candidate into quarantine; d2 takes the deletion lease; d3 removes the
    tree; d4 marks the record tree-gone; d5 deletes the lease and the record
    LAST. A crash at any point leaves a resumable residue, never an
    unvalidated deletion.

    d0..d2 run under the caller's parent lock. With ``defer_rmtree`` the
    physical d3..d5 are NOT run under the lock: the caller receives the
    :class:`_PendingDeletion` and runs :func:`_delete_quarantine` after
    releasing the parent lock, so removing a big tree never holds the
    machine-global lock (a crash before d3 leaves exactly the resumable
    residue the next sweep's resume finishes, so deferral is safe). The
    default keeps the whole sequence under the lock, byte-compatible with
    pre-W11 callers.
    """
    if not record_written:
        try:
            _write_json_atomic(
                alloc.qrec_path, _qrec_payload(alloc, deleter, "authorized")
            )
        except OSError as exc:
            return ReapResult(False, ("reaper-side-abort", str(exc)))
    try:
        _os_rename(alloc.path, alloc.quarantine_path)
    except OSError as exc:
        return ReapResult(False, ("quarantine-residue", f"rename-failed: {exc}"))
    _mark_qrec_state(alloc, "renamed")
    lease = acquire_deletion_lease(alloc.qlease_path, deleter)
    if lease is None:
        return ReapResult(False, ("quarantine-residue", "lease-blocked"))
    pending = _PendingDeletion(alloc=alloc, lease=lease)
    if defer_rmtree:
        return pending
    return _delete_quarantine(pending)


def _delete_quarantine(pending: _PendingDeletion) -> ReapResult:
    """d3..d5 of the quarantine transaction: rmtree, tree-gone mark, cleanup.

    Runs OUTSIDE the parent lock when the transaction was deferred; the
    deletion lease taken at d2 still serializes deleters, and a crash here
    leaves the same resumable residue the next sweep's resume finishes.
    """
    alloc = pending.alloc
    lease = pending.lease
    try:
        shutil.rmtree(alloc.quarantine_path)
    except OSError as exc:
        lease.release()
        return ReapResult(False, ("quarantine-residue", f"rmtree-failed: {exc}"))
    _mark_qrec_state(alloc, "tree-gone")
    lease.release()
    _unlink_quietly(alloc.qlease_path)
    _unlink_quietly(alloc.qrec_path)
    _unlink_quietly(alloc.owner_record_path)
    return ReapResult(True, (f"reaped-{alloc.alloc_id}",))


def _locked_reap_transaction(
    alloc: Allocation, deleter: IdentityTuple, blocking: bool
) -> ReapResult | _PendingDeletion:
    """Under the parent lock: re-run the cheap gates (pinned order) and start
    the record-first transaction. Returns a retain ReapResult, or the pending
    deletion for the caller to finish after the lock is released.
    """
    with parent_lock(alloc.parent, blocking=blocking):
        cheap = _evaluate_cheap_gates(alloc, None)
        if cheap is not None:
            return ReapResult(False, cheap.reasons)
        return _quarantine_transaction(
            alloc, deleter=deleter, record_written=False, defer_rmtree=True
        )


def reap_allocation(
    alloc: Allocation, *, deleter: IdentityTuple | None = None, blocking: bool = True
) -> ReapResult:
    """Validate, then run the record-first quarantine transaction.

    The cheap gates (1..6, pinned order) run under the parent lock; the /proc
    containment scan — the one gate that can take tens of milliseconds — runs
    OUTSIDE the lock, and the physical rmtree and record cleanup (d3..d5) run
    after the lock is released again. Neither a big tree nor a busy /proc
    therefore ever holds the machine-global lock and starves non-blocking
    boot inits (W11). The cheap gates are re-run under the lock immediately
    before the transaction, so the reap decision is still made under the lock
    on fresh state.

    ``blocking=False`` never waits on the lock: a busy parent lock fails
    closed as ``retained-lock-busy`` — the allocation is retained untouched
    and the next sweep retries, so the retry cadence is unchanged.
    """
    deleter = deleter or current_identity()
    try:
        with parent_lock(alloc.parent, blocking=blocking):
            cheap = _evaluate_cheap_gates(alloc, None)
            if cheap is not None:
                return ReapResult(False, cheap.reasons)
    except ParentLockBusy:
        return ReapResult(False, ("retained-lock-busy",))
    # The /proc containment scan outside the parent lock (W11 hold-shortening).
    containment = _gate_containment(alloc, None)
    if containment is not None:
        return ReapResult(False, containment.reasons)
    try:
        outcome = _locked_reap_transaction(alloc, deleter, blocking)
    except ParentLockBusy:
        return ReapResult(False, ("retained-lock-busy",))
    if isinstance(outcome, _PendingDeletion):
        # d3..d5 outside the parent lock.
        return _delete_quarantine(outcome)
    return outcome


def _revalidate_candidate(alloc: Allocation) -> bool:
    """Re-check the still-unrenamed candidate's identity under the lock:
    a committed-identity mismatch or a symlink substitution discards the
    record — an unvalidated tree is never renamed.
    """
    if load_ownership_record(alloc.parent, alloc.name) is None:
        return False
    try:
        info = os.lstat(alloc.path)
    except OSError:
        return False
    if stat_module.S_ISLNK(info.st_mode):
        return False
    record = load_ownership_record(alloc.parent, alloc.name)
    assert record is not None
    return adjudicate(record.owner.to_json()).dead


def _load_qrec(qrec_path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(qrec_path.read_bytes())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _resume_decision(alloc: Allocation, payload: Mapping[str, Any]) -> str:
    """clear | discard | complete | finish — what the dangling record resolves to."""
    if payload.get("state") == "tree-gone":
        return "clear"
    try:
        candidate_info = os.lstat(alloc.path)
    except OSError:
        candidate_info = None
    if candidate_info is not None and stat_module.S_ISLNK(candidate_info.st_mode):
        # Symlink substitution: discard the record, never rename an
        # unvalidated tree; the innocent target stays intact.
        return "discard"
    if candidate_info is not None:
        return "complete" if _revalidate_candidate(alloc) else "discard"
    if alloc.quarantine_path.exists():
        # Mid-rmtree residue: finish the deletion from the parent-beside record.
        return "finish"
    return "clear"


def _finish_quarantine(
    alloc: Allocation, qrec_path: Path, deleter: IdentityTuple
) -> ReapResult | _PendingDeletion:
    """d2 of the finish-residue path: take the deletion lease under the lock.

    The physical d3..d5 are run by the caller via :func:`_delete_quarantine`
    after the parent lock is released (the rmtree must not hold the lock).
    """
    lease = acquire_deletion_lease(alloc.qlease_path, deleter)
    if lease is None:
        return ReapResult(False, ("quarantine-residue", "lease-blocked"))
    return _PendingDeletion(alloc=alloc, lease=lease)


def _apply_resume(
    alloc: Allocation, qrec_path: Path, deleter: IdentityTuple, decision: str
) -> ReapResult | _PendingDeletion:
    if decision in {"clear", "discard"}:
        _unlink_quietly(alloc.qlease_path)
        _unlink_quietly(qrec_path)
        reason = (
            "quarantine-residue-cleared"
            if decision == "clear"
            else "quarantine-record-discarded"
        )
        return ReapResult(False, (reason,))
    if decision == "complete":
        return _quarantine_transaction(
            alloc, deleter=deleter, record_written=True, defer_rmtree=True
        )
    return _finish_quarantine(alloc, qrec_path, deleter)


def _resume_one(
    qrec_path: Path, deleter: IdentityTuple
) -> ReapResult | _PendingDeletion:
    name = qrec_path.name[: -len(QREC_SUFFIX)]
    alloc = allocation_from_path(qrec_path.parent / name)
    payload = _load_qrec(qrec_path)
    if payload is None:
        return ReapResult(False, ("quarantine-residue", "malformed-record"))
    return _apply_resume(alloc, qrec_path, deleter, _resume_decision(alloc, payload))


def resume_quarantine_records(
    parent: Path, *, deleter: IdentityTuple | None = None, blocking: bool = True
) -> list[ReapResult]:
    """Complete or discard every dangling quarantine record under the lock.

    d0..d2 of each record run under the parent lock; when a record resolves
    to a physical deletion, its d3..d5 run AFTER the lock is released, so the
    rmtree of a big quarantine tree never holds the machine-global lock.

    ``blocking=False`` skips a record while the parent lock is busy and
    reports it as ``retained-lock-busy`` (fail-closed; the next sweep retries).
    """
    deleter = deleter or current_identity()
    results: list[ReapResult] = []
    for qrec_path in sorted(parent.glob(f"*{QREC_SUFFIX}")):
        pending: _PendingDeletion | None = None
        try:
            with parent_lock(parent, blocking=blocking):
                outcome = _resume_one(qrec_path, deleter)
                if isinstance(outcome, _PendingDeletion):
                    pending = outcome
                else:
                    results.append(outcome)
        except ParentLockBusy:
            results.append(ReapResult(False, ("retained-lock-busy",)))
            continue
        if pending is not None:
            # d3..d5 outside the parent lock (W11 hold-shortening).
            results.append(_delete_quarantine(pending))
    return results


# --- C3b: the worker-scoped reap gate variant (no trigger wiring yet) --------


def _worker_entries(
    entries: list[dict[str, Any]], worker_json: Mapping[str, Any]
) -> list[dict[str, Any]]:
    return [
        entry
        for entry in entries
        if entry.get("appender") == worker_json or entry.get("spawner") == worker_json
    ]


def _gate_worker_liveness(worker: IdentityTuple) -> GateResult | None:
    verdict = adjudicate(worker.to_json())
    if verdict.live:
        return GateResult(False, ("retained-worker-live",))
    if not verdict.dead:
        return GateResult(False, ("retained-worker-indeterminate", verdict.reason))
    return None


def _gate_worker_ledger(alloc: Allocation, worker: IdentityTuple) -> GateResult | None:
    ledger = Ledger(alloc)
    if not ledger.path.exists():
        return GateResult(False, ("retained-ledger-missing",))
    # A dirty session ledger has missing-entry risk: retain even a dead
    # worker's candidate for the remainder of the boot.
    if ledger.is_dirty():
        return GateResult(False, ("retained-ledger-dirty",))
    entries = ledger.entries()
    worker_json = worker.to_json()
    mine = _worker_entries(entries, worker_json)
    settled = {
        entry.get("spawn_id")
        for entry in entries
        if entry.get("kind") in {"completion", "retraction"}
    }
    uncompleted = [
        entry
        for entry in mine
        if entry.get("kind") == "intent" and entry.get("spawn_id") not in settled
    ]
    if uncompleted:
        return GateResult(False, ("UNRESOLVED-INTENT",))
    for entry in mine:
        if entry.get("kind") != "completion":
            continue
        verdict = adjudicate(_completion_record(entry))
        if verdict.live:
            return GateResult(False, ("retained-registered-spawn",))
        if not verdict.dead:
            return GateResult(
                False, ("retained-registered-spawn-indeterminate", verdict.reason)
            )
    return None


def _gate_worker_groups(alloc: Allocation, worker: IdentityTuple) -> GateResult | None:
    ledger = Ledger(alloc)
    entries = ledger.entries()
    mine = _worker_entries(entries, worker.to_json())
    dead_pids = _dead_completion_pids(mine)
    for entry in mine:
        if not entry.get("is_leader_group"):
            continue
        pgid = entry.get("pgid")
        if not isinstance(pgid, int) or isinstance(pgid, bool):
            continue
        if leader_group_nonempty(pgid, dead_pids=dead_pids):
            return GateResult(False, ("retained-leader-group-nonempty",))
    return None


def _gate_worker_containment(candidate_dirs: Sequence[Path]) -> GateResult | None:
    for directory in candidate_dirs:
        refs = containment_refs(directory)
        if refs is None:
            return GateResult(False, ("retained-containment-indeterminate",))
        if refs:
            return GateResult(False, ("retained-containment-ref",))
    return None


def evaluate_worker_scoped_gates(
    alloc: Allocation, worker: IdentityTuple, candidate_dirs: Sequence[Path]
) -> GateResult:
    """The worker-scoped reap gates, evaluated ENTIRELY on the worker-scoped
    record set.

    The candidate is never the enclosing live session's allocation: the
    session owner may be live, and sibling spawns, their groups, and the
    ledger at large are OUT of this reap's inputs. Candidates enumerate
    structurally from the run contract's recorded invocation dirs.
    """
    if not is_managed_name(alloc.name):
        return GateResult(False, ("not-managed-name",))
    if load_ownership_record(alloc.parent, alloc.name) is None:
        return GateResult(False, ("torn-init",))
    gates = (
        lambda: _gate_worker_liveness(worker),
        lambda: _gate_worker_ledger(alloc, worker),
        lambda: _gate_worker_groups(alloc, worker),
        lambda: _gate_worker_containment(candidate_dirs),
    )
    for gate in gates:
        result = gate()
        if result is not None:
            return result
    return GateResult(True, ("reaped-worker-subtree",))
