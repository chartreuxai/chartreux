from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
from threading import Lock, RLock, local
from types import MappingProxyType
from typing import Any, BinaryIO, ClassVar
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from chartreux.core.paths import WORKTREES_DIR
from chartreux.core.utils.time import utc_now
from chartreux.observability.logging import logger

# Claims live beside the repo buckets rather than inside them for two reasons.
# A file inside the worktree would show up in `git status --untracked-files=all`
# and permanently fail the cleanup check, and a file inside the bucket
# directory would make `git worktree add` reject the directory the atomic mkdir
# claim just reserved. A bucket is always "<name>-<12 hex>", so this leading-dot
# name cannot collide with one.
CLAIMS_DIR_NAME = ".claims"
RECORD_FILENAME = "record.json"
HOLDERS_DIR_NAME = "holders"
_BUCKET_AND_NAME_PARTS = 2
_STARTING_HOLDER = ".starting"
_HOLDER_REGISTRY_FILENAME = ".holder-registry"
_PRUNE_LOCK_FILENAME = ".prune"
_HOLDER_FORMAT_VERSION = 1
_PROCESS_ID = os.getpid()
_REGISTRY_MUTEX = Lock()
_PRUNE_MUTEX = RLock()
_PRUNE_STATE = local()


class HolderKind(StrEnum):
    STARTING = "starting"
    PENDING_ATTACHMENT = "pending_attachment"
    SESSION = "session"
    CLI = "cli"


class AcquireOutcome(StrEnum):
    SUCCESS = "success"
    CONFLICT = "conflict"
    STALE = "stale"
    UNMANAGED = "unmanaged"
    ERROR = "error"


@dataclass(frozen=True)
class AcquireResult:
    outcome: AcquireOutcome
    token: OwnershipToken | None = None
    error: OSError | None = None

    SUCCESS: ClassVar[AcquireOutcome] = AcquireOutcome.SUCCESS
    CONFLICT: ClassVar[AcquireOutcome] = AcquireOutcome.CONFLICT
    STALE: ClassVar[AcquireOutcome] = AcquireOutcome.STALE
    UNMANAGED: ClassVar[AcquireOutcome] = AcquireOutcome.UNMANAGED
    ERROR: ClassVar[AcquireOutcome] = AcquireOutcome.ERROR


class ReleaseResult(StrEnum):
    RELEASED = "released"
    ALREADY_RELEASED = "already_released"


@dataclass(eq=False)
class OwnershipToken:
    claim_identity: tuple[str, str]
    claim_generation: str
    acquisition_id: str
    holder_kind: HolderKind
    holder_id: str
    _process_id: int = field(repr=False)
    _consumed: bool = field(default=False, repr=False)

    @property
    def consumed(self) -> bool:
        return self._consumed

    def __reduce__(self) -> str | tuple[Any, ...]:
        raise TypeError("OwnershipToken is process-bound and cannot be serialized")


@dataclass(frozen=True)
class HolderSnapshot:
    outcome: AcquireOutcome
    holders: Mapping[str, int] = field(default_factory=lambda: MappingProxyType({}))
    starting: bool = False
    error: OSError | None = None


@dataclass
class _HeldFile:
    file: BinaryIO
    generation: str
    kind: HolderKind
    count: int


_HELD_FILES: dict[Path, _HeldFile] = {}
_ACTIVE_TOKENS: dict[str, OwnershipToken] = {}
_LEGACY_TOKENS: dict[tuple[WorktreeClaim, str], list[OwnershipToken]] = {}


def _reset_after_fork() -> None:
    # Fork-based embedding is unsupported: inherited flock descriptors still
    # share an open-file description with the parent, so either process can
    # unlock the other's hold. Resetting only prevents the child from using
    # stale registries, mutex state, and the parent's process identity.
    global _PROCESS_ID, _REGISTRY_MUTEX, _PRUNE_MUTEX, _PRUNE_STATE
    _PROCESS_ID = os.getpid()
    _REGISTRY_MUTEX = Lock()
    _PRUNE_MUTEX = RLock()
    _PRUNE_STATE = local()
    _HELD_FILES.clear()
    _ACTIVE_TOKENS.clear()
    _LEGACY_TOKENS.clear()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_after_fork)


class WorktreeRecordError(Exception): ...


class WorktreeRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    version: int = 1
    name: str
    branch: str
    repo_root: Path
    # None between the mkdir claim and a completed `git worktree add`. Such a
    # record describes a reservation, not yet a worktree.
    base_commit: str | None = None
    branch_created: bool
    claimed_at: datetime

    @classmethod
    def new(
        cls, *, name: str, branch: str, repo_root: Path, branch_created: bool
    ) -> WorktreeRecord:
        return cls(
            name=name,
            branch=branch,
            repo_root=repo_root,
            branch_created=branch_created,
            claimed_at=utc_now(),
        )


def managed_bucket_name(repo_root: Path, common_git_dir: Path) -> str:
    repo_hash = hashlib.sha256(str(common_git_dir).encode()).hexdigest()[:12]
    return f"{repo_root.name}-{repo_hash}"


def _claims_root() -> Path:
    return WORKTREES_DIR.path.resolve() / CLAIMS_DIR_NAME


# Identifies one managed worktree. Bucket and name are a pair that is meaningless
# apart and indistinguishable as bare strings, so they travel as one value: a
# transposed argument would otherwise read and delete a plausible wrong path
# without any type error.
@dataclass(frozen=True, kw_only=True)
class WorktreeClaim:
    bucket: str
    name: str

    @classmethod
    def locate(cls, path: Path) -> WorktreeClaim | None:
        managed_root = WORKTREES_DIR.path.resolve()
        try:
            relative = path.resolve().relative_to(managed_root)
        except (OSError, ValueError):
            return None
        parts = relative.parts
        if len(parts) < _BUCKET_AND_NAME_PARTS or parts[0] == CLAIMS_DIR_NAME:
            return None
        return cls(bucket=parts[0], name=parts[1])

    # Only claimed names, never a listing of the bucket itself: a directory there
    # with no claim is either a live mkdir reservation or something the user
    # made, and neither is the sweep's to touch.
    @classmethod
    def in_bucket(cls, bucket: str) -> tuple[WorktreeClaim, ...]:
        # iterdir() is lazy, so the tuple must be built inside the try: a missing
        # bucket raises on first iteration, not on the call.
        try:
            return tuple(
                cls(bucket=bucket, name=entry.name)
                for entry in (_claims_root() / bucket).iterdir()
            )
        except OSError:
            return ()

    @property
    def directory(self) -> Path:
        return _claims_root() / self.bucket / self.name

    def write(self, record: WorktreeRecord) -> None:
        directory = self.directory
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / RECORD_FILENAME
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".json.tmp",
                dir=str(directory),
                delete=False,
                encoding="utf-8",
            ) as handle:
                temporary = Path(handle.name)
                handle.write(record.model_dump_json(indent=2))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def read(self) -> WorktreeRecord | None:
        target = self.directory / RECORD_FILENAME
        try:
            raw = target.read_text(encoding="utf-8")
        except OSError:
            return None
        try:
            return WorktreeRecord.model_validate_json(raw)
        except ValueError:
            # Fail closed: an unreadable record means the worktree is not treated
            # as Chartreux-owned, so nothing is ever deleted on the strength of one.
            # The file stays put because it may be the only remaining breadcrumb.
            logger.warning("Ignoring unreadable worktree record at %s", target)
            return None

    def delete(self) -> None:
        (self.directory / RECORD_FILENAME).unlink(missing_ok=True)
        self._discard_empty_directories()

    def _discard_empty_directories(self) -> None:
        directory = self.directory
        # rmdir, never rmtree: a surviving holder means a live session, and losing
        # its marker would let the next release delete the worktree underneath it.
        for parent in (directory / HOLDERS_DIR_NAME, directory, directory.parent):
            try:
                parent.rmdir()
            except FileNotFoundError:
                continue
            except OSError:
                return

    # A holder is an empty file named for a session that is currently working in
    # the worktree. Sessions run in separate app-server processes with nothing
    # shared between them, so presence of the marker is the only liveness signal
    # available without a lock. A hard-killed process leaves its marker behind
    # and the worktree is kept forever, which is the safe direction to fail in.
    def _holder_path(self, session_id: str) -> Path:
        holders = self.directory / HOLDERS_DIR_NAME
        # A session id names a file, so anything that could climb out of the
        # holders directory is rejected outright rather than sanitised into
        # something that still unlinks the wrong path.
        candidate = (holders / session_id).resolve()
        if candidate.parent != holders.resolve() or not session_id:
            raise WorktreeRecordError(f"Unusable worktree holder id {session_id!r}.")
        return candidate

    def add_holder(self, session_id: str) -> None:
        generation = _claim_generation(self)
        if generation is None:
            raise WorktreeRecordError("Cannot hold an unmanaged worktree claim.")
        result = acquire_holder(
            self, session_id, kind=HolderKind.SESSION, expected_generation=generation
        )
        if result.outcome is not AcquireOutcome.SUCCESS or result.token is None:
            raise WorktreeRecordError(
                f"Unable to acquire worktree holder {session_id!r}: {result.outcome.value}."
            )
        with _REGISTRY_MUTEX:
            _LEGACY_TOKENS.setdefault((self, session_id), []).append(result.token)

    def remove_holder(self, session_id: str) -> None:
        self._holder_path(session_id)
        with _REGISTRY_MUTEX:
            tokens = _LEGACY_TOKENS.get((self, session_id))
            token = tokens.pop() if tokens else None
            if tokens == []:
                del _LEGACY_TOKENS[(self, session_id)]
        if token is not None:
            release_holder(token)
        # Preserve old callers' idempotence without ever unlinking a marker that
        # this process did not acquire.
        if not (self.directory / RECORD_FILENAME).exists():
            self._discard_empty_directories()

    def holders(self) -> frozenset[str]:
        snapshot = inspect_holders(self)
        if snapshot.outcome is AcquireOutcome.ERROR:
            raise WorktreeRecordError("Unable to inspect worktree holders.")
        return frozenset(snapshot.holders)


def _claim_generation(claim: WorktreeClaim) -> str | None:
    record = claim.read()
    return record.claimed_at.isoformat() if record is not None else None


@contextmanager
def _file_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    file = path.open("a+b")
    try:
        fcntl.flock(file.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(file.fileno(), fcntl.LOCK_UN)
        finally:
            file.close()


@contextmanager
def worktree_prune_lock() -> Iterator[None]:
    """Serialize attachment publication with pruning and destructive removal."""
    with _PRUNE_MUTEX:
        depth = getattr(_PRUNE_STATE, "depth", 0)
        if depth:
            _PRUNE_STATE.depth = depth + 1
            try:
                yield
            finally:
                _PRUNE_STATE.depth -= 1
            return
        _PRUNE_STATE.depth = 1
        try:
            with _file_lock(_claims_root() / _PRUNE_LOCK_FILENAME):
                yield
        finally:
            _PRUNE_STATE.depth = 0


@contextmanager
def _ordered_registry_locks() -> Iterator[None]:
    root = _claims_root()
    with worktree_prune_lock():
        with _file_lock(root / _HOLDER_REGISTRY_FILENAME):
            yield


def _holder_filename(holder_id: str, kind: HolderKind) -> str:
    return _STARTING_HOLDER if kind is HolderKind.STARTING else holder_id


def _try_lock(file: BinaryIO) -> bool:
    try:
        fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        if exc.errno in {errno.EACCES, errno.EAGAIN}:
            return False
        raise
    return True


def _write_holder(file: BinaryIO, *, generation: str, kind: HolderKind) -> None:
    payload = {
        "holder_version": _HOLDER_FORMAT_VERSION,
        "generation": generation,
        "kind": kind.value,
        "process_id": _PROCESS_ID,
    }
    file.seek(0)
    file.truncate()
    file.write((json.dumps(payload, sort_keys=True) + "\n").encode())
    file.flush()
    os.fsync(file.fileno())


def _is_new_format(file: BinaryIO) -> bool:
    file.seek(0)
    try:
        payload = json.loads(file.read().decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return False
    return (
        isinstance(payload, dict)
        and payload.get("holder_version") == _HOLDER_FORMAT_VERSION
    )


def acquire_holder(  # noqa: PLR0911
    claim: WorktreeClaim, holder_id: str, *, kind: HolderKind, expected_generation: str
) -> AcquireResult:
    if not isinstance(kind, HolderKind) or not isinstance(expected_generation, str):
        raise WorktreeRecordError("Malformed holder acquisition request.")
    filename = _holder_filename(holder_id, kind)
    try:
        path = claim._holder_path(filename)
        with _ordered_registry_locks():
            generation = _claim_generation(claim)
            if generation is None:
                return AcquireResult(AcquireOutcome.UNMANAGED)
            if generation != expected_generation:
                return AcquireResult(AcquireOutcome.STALE)
            path.parent.mkdir(parents=True, exist_ok=True)
            with _REGISTRY_MUTEX:
                held = _HELD_FILES.get(path)
                if held is not None:
                    if kind not in {HolderKind.SESSION, HolderKind.CLI}:
                        return AcquireResult(AcquireOutcome.CONFLICT)
                    if held.generation != generation:
                        return AcquireResult(AcquireOutcome.STALE)
                    held.count += 1
                    token = _new_token(claim, holder_id, kind, generation)
                    _ACTIVE_TOKENS[token.acquisition_id] = token
                    return AcquireResult(AcquireOutcome.SUCCESS, token)
            created = False
            try:
                descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o666)
                created = True
            except FileExistsError:
                descriptor = os.open(path, os.O_RDWR)
            file = os.fdopen(descriptor, "r+b")
            try:
                if not _try_lock(file):
                    file.close()
                    return AcquireResult(AcquireOutcome.CONFLICT)
                # Existing empty/unknown files are legacy markers. They are
                # deliberately never reclaimed because their owner's death
                # cannot be proven. Only an exclusively created empty file is
                # ours to initialize with the current holder format.
                if not created and not _is_new_format(file):
                    fcntl.flock(file.fileno(), fcntl.LOCK_UN)
                    file.close()
                    return AcquireResult(AcquireOutcome.CONFLICT)
                _write_holder(file, generation=generation, kind=kind)
                with _REGISTRY_MUTEX:
                    token = _new_token(claim, holder_id, kind, generation)
                    _HELD_FILES[path] = _HeldFile(file, generation, kind, 1)
                    _ACTIVE_TOKENS[token.acquisition_id] = token
                return AcquireResult(AcquireOutcome.SUCCESS, token)
            except BaseException:
                if not file.closed:
                    file.close()
                raise
    except OSError as exc:
        return AcquireResult(AcquireOutcome.ERROR, error=exc)


def _new_token(
    claim: WorktreeClaim, holder_id: str, kind: HolderKind, generation: str
) -> OwnershipToken:
    return OwnershipToken(
        claim_identity=(claim.bucket, claim.name),
        claim_generation=generation,
        acquisition_id=uuid4().hex,
        holder_kind=kind,
        holder_id=holder_id,
        _process_id=_PROCESS_ID,
    )


def release_holder(token: OwnershipToken) -> ReleaseResult:
    if not isinstance(token, OwnershipToken) or token._process_id != os.getpid():
        raise WorktreeRecordError("Malformed or foreign-process ownership token.")
    claim = WorktreeClaim(bucket=token.claim_identity[0], name=token.claim_identity[1])
    path = claim._holder_path(_holder_filename(token.holder_id, token.holder_kind))
    with _ordered_registry_locks():
        with _REGISTRY_MUTEX:
            if token._consumed:
                return ReleaseResult.ALREADY_RELEASED
            if _ACTIVE_TOKENS.get(token.acquisition_id) is not token:
                raise WorktreeRecordError("Unknown ownership token.")
            token._consumed = True
            del _ACTIVE_TOKENS[token.acquisition_id]
            held = _HELD_FILES.get(path)
            if held is None or held.generation != token.claim_generation:
                raise WorktreeRecordError(
                    "Ownership token no longer matches its claim."
                )
            held.count -= 1
            if held.count:
                return ReleaseResult.RELEASED
            del _HELD_FILES[path]
            file = held.file
        try:
            fcntl.flock(file.fileno(), fcntl.LOCK_UN)
        finally:
            file.close()
        # Registry serialization and the generation check prevent deleting a
        # replacement holder between unlock and unlink.
        current_generation = _claim_generation(claim)
        if current_generation in {None, token.claim_generation}:
            path.unlink(missing_ok=True)
        if not (claim.directory / RECORD_FILENAME).exists():
            claim._discard_empty_directories()
    return ReleaseResult.RELEASED


def _record_live_entry(
    entry: Path, local: Mapping[Path, _HeldFile], counts: dict[str, int]
) -> bool:
    held = local.get(entry)
    if held is not None:
        if entry.name != _STARTING_HOLDER:
            counts[entry.name] = held.count
        return entry.name == _STARTING_HOLDER
    with entry.open("r+b") as file:
        if not _try_lock(file):
            if entry.name != _STARTING_HOLDER:
                counts[entry.name] = 1
            return entry.name == _STARTING_HOLDER
        try:
            if _is_new_format(file):
                entry.unlink()
                return False
            # A legacy marker has no lock protocol. Conservatively report it live.
            if entry.name != _STARTING_HOLDER:
                counts[entry.name] = 1
            return entry.name == _STARTING_HOLDER
        finally:
            fcntl.flock(file.fileno(), fcntl.LOCK_UN)


def inspect_holders(claim: WorktreeClaim) -> HolderSnapshot:
    counts: dict[str, int] = {}
    starting = False
    try:
        with _ordered_registry_locks():
            directory = claim.directory / HOLDERS_DIR_NAME
            try:
                entries = tuple(directory.iterdir())
            except FileNotFoundError:
                return HolderSnapshot(AcquireOutcome.SUCCESS)
            with _REGISTRY_MUTEX:
                local = dict(_HELD_FILES)
            for entry in entries:
                starting = _record_live_entry(entry, local, counts) or starting
        return HolderSnapshot(
            AcquireOutcome.SUCCESS, MappingProxyType(counts), starting
        )
    except OSError as exc:
        return HolderSnapshot(AcquireOutcome.ERROR, error=exc)
