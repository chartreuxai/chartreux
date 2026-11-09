"""Plan episode arbitration: exclusive allocation, transition fence, sidecar.

A planning episode is the identity a plan handoff runs under. One directory
under the project's workspace ``tasks/`` holds it, and one JSON sidecar inside
that directory is the on-disk arbiter of its lifecycle state. Every sidecar
change goes through ONE serialized cross-process procedure — a kernel ``flock``
on ``.chartreux-plan-claim.lock`` — so a cancel, a claim, a re-bind, and a
cleanup can never interleave.

This module owns the declared-new-code pieces of that arbiter: the sidecar
read/verify/durable-write procedure and its state model, the transition fence,
the plan-mode durable write wrapper whose temporary siblings carry the
``.chartreux-plan-tmp-*`` prefix, the exclusive publication primitive for
consent snapshots, the PREPARE transaction (P1 claim, P2 raw-byte retention,
P2.5 confirm re-compare, P3 snapshot publication) that turns a draft into a
consented attempt, the COMMIT transaction (C1 digest dual-verify and the
durable attempt mark, C3's discovery-domain record and one-observation binding,
C5's terminal flip followed by the derived rename), the provenance names those
snapshots take, the direct-metas bounded executor scan, the crash-recovery
implementation the recovery rows are keyed on (the sidecar-first observation,
the fenced claim-release / restoration / re-offer / cancel legs and the
adjudication, the either-name locate, and the flip-agnostic completion rule),
the terminal ``/plan cancel`` transition, and the cleanup whitelist.

``chartreux/core/session/session_lease.py`` is the flock precedent this fence
mirrors. One deliberate difference: the lease unlinks its lock file on release,
while the episode fence file is NEVER unlinked (:class:`TransitionFence`).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import time
from typing import BinaryIO, NoReturn
import uuid

from chartreux.core.session_types import (
    LEGACY_UNVERIFIED_STATE,
    PlanModeRecord,
    PlanScanDomain,
)
from chartreux.utils.durable_io import fsync_directory

EPISODE_FENCE_FILENAME = ".chartreux-plan-claim.lock"
EPISODE_SIDECAR_FILENAME = ".chartreux-plan-episode.json"
PLAN_TEMP_PREFIX = ".chartreux-plan-tmp-"
EPISODE_SIDECAR_KIND = "chartreux-plan-episode"
EPISODE_SIDECAR_VERSION = 1
SESSION_META_FILENAME = "meta.json"

STATE_DRAFT = "draft"
STATE_ACCEPTING = "accepting"
STATE_ACCEPTED = "accepted"
STATE_CANCELLED = "cancelled"
STATES = frozenset({STATE_DRAFT, STATE_ACCEPTING, STATE_ACCEPTED, STATE_CANCELLED})
TERMINAL_STATES = frozenset({STATE_ACCEPTED, STATE_CANCELLED})

LOGGING_REQUIRED_MESSAGE = "plan mode requires session logging enabled"
DOMAIN_REFUSAL = "episode owned by a different logging domain — start-new"
UNRESOLVABLE_ARBITER_MESSAGE = (
    "unresolvable arbiter state — operator inspect, rebind, or start-new"
)
_POST_COMMIT_REFUSAL = (
    "the accept claim is post-commit (c1_verified is set); the crash-recovery "
    "rows govern it, not a release"
)
REPREVIEW_REQUIRED_MESSAGE = (
    "the plan document differs from the bytes last previewed — re-preview and "
    "re-consent before accepting"
)
SNAPSHOT_COLLISION_MESSAGE = (
    "a plan consent snapshot already occupies the live name; the existing file "
    "is never overwritten"
)
CARRIED_DIGEST_MISMATCH_MESSAGE = (
    "the plan handoff bytes do not match the episode's consent digest — refuse "
    "and re-consent"
)
SNAPSHOT_DIGEST_MISMATCH_MESSAGE = (
    "the plan consent snapshot does not match the episode's consent digest — "
    "refuse and re-consent"
)
EXECUTOR_EXISTS_MESSAGE = (
    "an executor session already records this plan episode — a second execution "
    "is refused; inspect the episode status"
)

SNAPSHOT_RENAMED = "renamed"
SNAPSHOT_RENAME_NOOP = "already-renamed"
SNAPSHOT_RENAME_PENDING = "rename-pending"

SEED_DOCUMENT_BEGIN = "---- BEGIN ACCEPTED PLAN DOCUMENT ----"
SEED_DOCUMENT_END = "---- END ACCEPTED PLAN DOCUMENT ----"


class PlanEpisodeError(Exception):
    """Base class for plan-episode arbitration failures."""


class PlanSidecarError(PlanEpisodeError):
    """The episode sidecar is missing, unreadable, or corrupt."""


class PlanTransitionError(PlanEpisodeError):
    """Base class for refused or failed episode state transitions."""


class PlanFenceBusyError(PlanTransitionError):
    """Another transition holds the episode's fence right now."""


class PlanEpisodeStateError(PlanTransitionError):
    """The episode is not in the state the transition requires."""


class PlanEpisodeTerminalError(PlanEpisodeStateError):
    """The episode reached a terminal state; only a new episode continues."""


class PlanEpisodeOwnerError(PlanTransitionError):
    """The acting session is not the episode's recorded owner."""


class PlanEpisodeBindError(PlanTransitionError):
    """A re-bind target is not inside the authorized roots."""


class PlanEpisodeWriteError(PlanTransitionError):
    """A durable sidecar write failed; its outcome is stated in the message."""


class PlanArbiterUnresolvableError(PlanTransitionError):
    """The sidecar holds neither the previous nor the intended state."""


class PlanLoggingDisabledError(PlanEpisodeError):
    """Plan mode was entered while session logging is disabled."""


class PlanEpisodeScanError(PlanEpisodeError):
    """The bounded executor-meta scan could not be trusted."""


class PlanDraftReadError(PlanEpisodeError):
    """The draft's raw bytes cannot be read for retention or re-comparison."""


class PlanDocumentChangedError(PlanTransitionError):
    """The draft differs from the previewed bytes; consent must be re-obtained."""


class PlanSnapshotCollisionError(PlanTransitionError):
    """A snapshot already occupies the live name; exclusive create refused."""


class PlanDigestMismatchError(PlanTransitionError):
    """C1 refused: the carried bytes or the snapshot is not the consented bytes."""


@dataclass(frozen=True, slots=True)
class LoggingDomain:
    """A session-logging root: the ``{save_dir, session_prefix}`` pair.

    The domain is compared verbatim. A process configured against a different
    logging root holds a different per-session lease for the same session id and
    therefore cannot act on the episode, so textual equality is the fail-closed
    reading: two differently spelled views of one directory refuse rather than
    silently agree.
    """

    save_dir: str
    session_prefix: str


@dataclass(frozen=True, slots=True)
class PlanEpisodeSidecar:
    """One episode's authoritative on-disk record."""

    episode_id: str
    draft_path: Path
    state: str = STATE_DRAFT
    kind: str = EPISODE_SIDECAR_KIND
    version: int = EPISODE_SIDECAR_VERSION
    claiming_session_id: str | None = None
    consent_sha256: str | None = None
    accepted_sha256: str | None = None
    executor_session_id: str | None = None
    owner_domain: LoggingDomain | None = None
    scan_domain: LoggingDomain | None = None
    attempt_generation: int = 0
    c1_verified: bool = False


@dataclass(frozen=True, slots=True)
class ExecutorScan:
    """The bounded executor-meta scan; ``executor_session_id`` None is a miss."""

    executor_session_id: str | None = None

    @property
    def found(self) -> bool:
        return self.executor_session_id is not None


@dataclass(frozen=True, slots=True)
class PreparedAccept:
    """The outcome of a completed PREPARE (P1-P3) transaction.

    ``retained_bytes`` are the previewed draft bytes the snapshot was published
    from and ``consent_sha256`` is their digest as the sidecar records it. The
    commit half carries the same bytes; the snapshot file is durable provenance
    and the crash-recovery carrier.
    """

    sidecar: PlanEpisodeSidecar
    snapshot_path: Path
    retained_bytes: bytes
    consent_sha256: str


@dataclass(frozen=True, slots=True)
class LoggingObservation:
    """ONE frozen logging-configuration observation.

    ``domain`` is the ``{save_dir, session_prefix}`` record C3 writes into the
    sidecar's ``scan_domain`` and ``session_dir`` is the destination of the
    executor's durable marker. Both derive from the SAME read, and the
    constructor that built them refuses when the directory does not fall inside
    the recorded domain — so the record and the marker's destination can never
    disagree (DR-68).
    """

    domain: LoggingDomain
    session_dir: Path


@dataclass(frozen=True, slots=True)
class CommittedEpisode:
    """The outcome of a completed COMMIT (C1-C5) transaction."""

    sidecar: PlanEpisodeSidecar
    rename_outcome: str


@dataclass(frozen=True, slots=True)
class CommitAttempt:
    """A COMMIT half in progress, holding the episode's transition fence.

    :func:`begin_commit` acquires the fence and runs C1 — the expected-state
    verify, the digest dual-check, and the durable ``c1_verified`` mark — before
    returning. Everything the commit changes afterwards goes through this
    handle, so every mutation stays inside the one held fence (DR-38). The
    caller MUST :meth:`close` it; the fence is held until then, through C5.
    """

    directory: Path
    episode_id: str
    session_id: str
    owner_domain: LoggingDomain
    snapshot_path: Path
    carried_bytes: bytes
    sidecar: PlanEpisodeSidecar
    _fence: TransitionFence = field(repr=False, compare=False)

    @property
    def attempt_generation(self) -> int:
        return self.sidecar.attempt_generation

    @property
    def consent_sha256(self) -> str | None:
        return self.sidecar.consent_sha256

    def record_scan_domain(self, domain: LoggingDomain) -> PlanEpisodeSidecar:
        """C3: re-record the discovery domain before the marker write.

        The record precedes the marker and is bound to the marker's destination
        by the single frozen observation it came from; ``owner_domain`` is
        immutable, so a discovery update can never delegitimize the owner.
        """
        sidecar = self._require_held_accepting()
        return _durable_transition(
            self.directory, sidecar, replace(sidecar, scan_domain=domain)
        )

    def record_executor_session(self, executor_session_id: str) -> PlanEpisodeSidecar:
        """C4: record the rotated executor session while the episode is accepting.

        The executor id exists only once the session replacement handed out the
        new id, so this write follows it and the terminal flip still owns the
        state change. Recording it BEFORE C5 is what lets an observer tell the
        launched executor apart from a PLANNING session that restored the same
        ``accepting`` episode: both carry the marker's shape, only one carries
        this id (see :func:`scan_sidecar_executor` and the
        ``expected_session_id`` predicate in :func:`scan_episode_executor`).
        """
        sidecar = self._require_held_accepting()
        return _durable_transition(
            self.directory,
            sidecar,
            replace(sidecar, executor_session_id=executor_session_id),
        )

    def finalize(self, *, executor_session_id: str) -> CommittedEpisode:
        """C5: the terminal flip, then the strictly-after derived rename.

        The flip is the LAST arbiter-visible state change and happens only once
        the executor session exists. The rename follows it and is DERIVED from
        it, so an ``accepting`` sidecar always means the rename has not run. A
        rename that cannot run is honest-tolerated: the flip is authoritative,
        both names are permanent provenance, and the completion rule completes
        the rename at the episode's next observation.
        """
        sidecar = self._require_held_accepting()
        accepted = replace(
            sidecar,
            state=STATE_ACCEPTED,
            accepted_sha256=sidecar.consent_sha256,
            executor_session_id=executor_session_id,
        )
        published = _durable_transition(self.directory, sidecar, accepted)
        try:
            outcome = rename_accepted_snapshot(
                self.directory,
                episode_id=published.episode_id,
                generation=published.attempt_generation,
            )
        except PlanTransitionError:
            outcome = SNAPSHOT_RENAME_PENDING
        return CommittedEpisode(sidecar=published, rename_outcome=outcome)

    def close(self) -> None:
        """Release the commit fence; safe to call more than once."""
        self._fence.close()

    def _require_held_accepting(self) -> PlanEpisodeSidecar:
        sidecar = read_sidecar_required(self.directory)
        if sidecar.episode_id != self.episode_id:
            raise PlanEpisodeStateError(
                f"the plan episode {sidecar.episode_id} does not match "
                f"{self.episode_id}"
            )
        if sidecar.state != STATE_ACCEPTING or not sidecar.c1_verified:
            raise PlanEpisodeStateError(
                f"plan episode {sidecar.episode_id} is not a held commit "
                f"(state {sidecar.state}, c1_verified {sidecar.c1_verified})"
            )
        require_episode_owner(
            sidecar, session_id=self.session_id, owner_domain=self.owner_domain
        )
        return sidecar


def new_episode_id() -> str:
    """Return a fresh, filename-safe episode identity."""
    return f"ep-{uuid.uuid4().hex}"


def fence_path(directory: Path) -> Path:
    return directory / EPISODE_FENCE_FILENAME


def sidecar_path(directory: Path) -> Path:
    return directory / EPISODE_SIDECAR_FILENAME


def accepted_snapshot_path(directory: Path, episode_id: str) -> Path:
    return directory / f"accepted-{episode_id}.md"


def provenance_snapshot_path(directory: Path, episode_id: str, generation: int) -> Path:
    return directory / f"provenance-{episode_id}-{generation}.md"


def is_plan_temp_name(name: str) -> bool:
    """True for a temporary sibling the plan-mode durable writer owns."""
    return name.startswith(PLAN_TEMP_PREFIX)


# --- Durable writes and exclusive publication -------------------------------


def durable_replace_plan(path: Path, content: bytes) -> None:
    """Publish ``path`` atomically with the plan-mode temporary-name convention.

    Barriers match ``durable_replace_stream``: the temporary sibling is created
    owner-only, synced, then renamed over ``path``, and the parent directory is
    synced after the rename. The temporary name carries the plan-mode prefix so
    cleanup can recognize it. A failure before the rename leaves ``path``
    exactly as it was; a failure after the rename is not a rollback.
    """
    temp_path = path.parent / f"{PLAN_TEMP_PREFIX}{os.getpid()}-{uuid.uuid4().hex}"
    try:
        descriptor = os.open(temp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
        fsync_directory(path.parent)
    finally:
        with suppress(FileNotFoundError):
            temp_path.unlink()


def _publish_exclusive(path: Path, content: bytes) -> None:
    """Create ``path`` exclusively, syncing its content before its parent."""
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        with suppress(OSError):
            path.unlink()
        raise
    fsync_directory(path.parent)


def publish_episode_snapshot(path: Path, content: bytes) -> None:
    """Publish a consent snapshot by EXCLUSIVE CREATE (DR-67).

    The snapshot's create is its publication: the content is fsynced BEFORE the
    parent directory, because a directory barrier alone does not make the bytes
    durable. There is no write-temp-then-rename path for snapshots; an existing
    path raises ``FileExistsError``, which is what keeps a retry's create clean.
    """
    _publish_exclusive(path, content)


# --- The transition fence ---------------------------------------------------


class TransitionFence:
    """One episode's cross-process transition fence.

    Acquisition is always non-blocking (``LOCK_EX|LOCK_NB``): a busy fence means
    exactly "that transition is running right now", so a site retries rather
    than blocks. Kernel ownership makes every failure mode benign — a crashed
    holder's fence is released when its last descriptor closes, so a stale fence
    cannot exist and there is no break-lock procedure at all.

    Release is unlock-ONLY and the file is NEVER unlinked: unlinking a held
    flock file would let a new opener lock a fresh inode while the old holder
    still holds the old one, silently breaking exclusion. A leftover fence file
    is harmless; the next acquire reuses it. The holder payload
    ``{holder_session_id, pid, acquired_at}`` is written under the fence after
    acquisition and is DIAGNOSTIC ONLY — every acquirer rewrites it, so it can
    never be a guard input.
    """

    def __init__(self, directory: Path) -> None:
        self._path = fence_path(directory)
        self._file: BinaryIO | None = None
        self._acquired = False

    @property
    def path(self) -> Path:
        return self._path

    def acquire(
        self,
        *,
        holder_session_id: str,
        retry_timeout: float = 0.0,
        retry_interval: float = 0.05,
    ) -> None:
        """Take the fence; a busy fence retries until the bounded timeout."""
        deadline = time.monotonic() + max(retry_timeout, 0.0)
        while True:
            try:
                self._try_acquire(holder_session_id)
                return
            except PlanFenceBusyError:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise
                time.sleep(min(retry_interval, remaining))

    def _open(self) -> BinaryIO:
        if self._file is None:
            descriptor = os.open(self._path, os.O_RDWR | os.O_CREAT, 0o600)
            self._file = os.fdopen(descriptor, "r+b")
        return self._file

    def _try_acquire(self, holder_session_id: str) -> None:
        file = self._open()
        try:
            _lock(file, blocking=False)
        except BlockingIOError as error:
            raise PlanFenceBusyError(
                f"plan episode transition fence is busy: {self._path}"
            ) from error
        try:
            payload = {
                "holder_session_id": holder_session_id,
                "pid": os.getpid(),
                "acquired_at": _timestamp(),
            }
            file.seek(0)
            file.truncate()
            file.write(
                json.dumps(
                    payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
                ).encode()
                + b"\n"
            )
            file.flush()
            os.fsync(file.fileno())
        except BaseException:
            # A fence that cannot publish its diagnostic is not held: undo the
            # lock so the caller sees the original failure, not a stuck fence.
            with suppress(OSError):
                _unlock(file)
            raise
        self._acquired = True

    def release(self) -> None:
        """Unlock; the descriptor and the file both stay (never unlinked)."""
        file = self._file
        if file is None or not self._acquired:
            return
        try:
            _unlock(file)
        finally:
            self._acquired = False

    def close(self) -> None:
        """Release and close the descriptor; the fence file itself remains."""
        try:
            self.release()
        finally:
            file = self._file
            self._file = None
            if file is not None:
                file.close()


def ensure_episode_fence(directory: Path) -> Path:
    """Open (creating if needed) the episode's fence file.

    Allocation opens the fence, so an ever-allocated episode directory always
    carries the file that keeps it from being rmdir-eligible. The file is never
    unlinked afterwards — the next acquire simply reuses it.
    """
    path = fence_path(directory)
    os.close(os.open(path, os.O_RDWR | os.O_CREAT, 0o600))
    return path


@contextmanager
def held_transition_fence(
    directory: Path, *, holder_session_id: str, retry_timeout: float = 0.0
) -> Iterator[TransitionFence]:
    """Hold one episode's fence for the duration of the block."""
    fence = TransitionFence(directory)
    fence.acquire(holder_session_id=holder_session_id, retry_timeout=retry_timeout)
    try:
        yield fence
    finally:
        fence.close()


def _lock(file: BinaryIO, *, blocking: bool) -> None:
    import fcntl

    operation = fcntl.LOCK_EX if blocking else fcntl.LOCK_EX | fcntl.LOCK_NB
    try:
        fcntl.flock(file.fileno(), operation)
    except OSError as error:
        raise BlockingIOError from error


def _unlock(file: BinaryIO) -> None:
    import fcntl

    fcntl.flock(file.fileno(), fcntl.LOCK_UN)


def _timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


# --- The sidecar ------------------------------------------------------------


def read_sidecar(directory: Path) -> PlanEpisodeSidecar | None:
    """Read the episode sidecar; ``None`` when the directory holds none."""
    path = sidecar_path(directory)
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return None
    except OSError as error:
        raise PlanSidecarError(
            f"plan episode sidecar is unreadable: {path}: {error}"
        ) from error
    try:
        payload = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as error:
        raise PlanSidecarError(
            f"plan episode sidecar is corrupt: {path}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise PlanSidecarError(f"plan episode sidecar is corrupt: {path}")
    return sidecar_from_payload(payload, path=path)


def read_sidecar_required(directory: Path) -> PlanEpisodeSidecar:
    """Read the episode sidecar, refusing a missing one."""
    sidecar = read_sidecar(directory)
    if sidecar is None:
        raise PlanSidecarError(
            f"plan episode sidecar is missing: {sidecar_path(directory)} — "
            "rebind or start-new"
        )
    return sidecar


def encode_sidecar(sidecar: PlanEpisodeSidecar) -> bytes:
    """Serialize the sidecar canonically, newline-terminated."""
    payload = {
        "kind": sidecar.kind,
        "version": sidecar.version,
        "episode_id": sidecar.episode_id,
        "draft_path": str(sidecar.draft_path),
        "state": sidecar.state,
        "claiming_session_id": sidecar.claiming_session_id,
        "consent_sha256": sidecar.consent_sha256,
        "accepted_sha256": sidecar.accepted_sha256,
        "executor_session_id": sidecar.executor_session_id,
        "owner_domain": _domain_payload(sidecar.owner_domain),
        "scan_domain": _domain_payload(sidecar.scan_domain),
        "attempt_generation": sidecar.attempt_generation,
        "c1_verified": sidecar.c1_verified,
    }
    return (
        json.dumps(
            payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode()
        + b"\n"
    )


def sidecar_from_payload(
    payload: dict[str, object], *, path: Path | None = None
) -> PlanEpisodeSidecar:
    """Build the sidecar from its JSON payload, refusing anything untrusted."""
    detail = f": {path}" if path is not None else ""
    if payload.get("kind") != EPISODE_SIDECAR_KIND:
        raise PlanSidecarError(f"plan episode sidecar has an unknown kind{detail}")
    if payload.get("version") != EPISODE_SIDECAR_VERSION:
        raise PlanSidecarError(
            f"plan episode sidecar has an unsupported version{detail}"
        )
    state = _payload_str(payload, "state", detail)
    if state not in STATES:
        raise PlanSidecarError(
            f"plan episode sidecar has an unknown state {state!r}{detail}"
        )
    generation = payload.get("attempt_generation", 0)
    if not isinstance(generation, int) or generation < 0:
        raise PlanSidecarError(
            f"plan episode sidecar has an invalid attempt generation{detail}"
        )
    verified = payload.get("c1_verified", False)
    if not isinstance(verified, bool):
        raise PlanSidecarError(
            f"plan episode sidecar has an invalid commit flag{detail}"
        )
    return PlanEpisodeSidecar(
        episode_id=_payload_str(payload, "episode_id", detail),
        draft_path=Path(_payload_str(payload, "draft_path", detail)),
        state=state,
        claiming_session_id=_payload_optional_str(
            payload, "claiming_session_id", detail
        ),
        consent_sha256=_payload_optional_str(payload, "consent_sha256", detail),
        accepted_sha256=_payload_optional_str(payload, "accepted_sha256", detail),
        executor_session_id=_payload_optional_str(
            payload, "executor_session_id", detail
        ),
        owner_domain=_payload_domain(payload, "owner_domain", detail),
        scan_domain=_payload_domain(payload, "scan_domain", detail),
        attempt_generation=generation,
        c1_verified=verified,
    )


def write_sidecar(directory: Path, sidecar: PlanEpisodeSidecar) -> None:
    """Durably replace the episode sidecar (temp, file fsync, rename, dir fsync)."""
    durable_replace_plan(sidecar_path(directory), encode_sidecar(sidecar))


def create_sidecar(directory: Path, sidecar: PlanEpisodeSidecar) -> None:
    """Claim the directory by creating its sidecar EXCLUSIVELY."""
    _publish_exclusive(sidecar_path(directory), encode_sidecar(sidecar))


def _payload_str(payload: dict[str, object], key: str, detail: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise PlanSidecarError(
            f"plan episode sidecar is corrupt: {key!r} is missing{detail}"
        )
    return value


def _payload_optional_str(
    payload: dict[str, object], key: str, detail: str
) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise PlanSidecarError(
            f"plan episode sidecar is corrupt: {key!r} is invalid{detail}"
        )
    return value


def _domain_payload(domain: LoggingDomain | None) -> dict[str, str] | None:
    if domain is None:
        return None
    return {"save_dir": domain.save_dir, "session_prefix": domain.session_prefix}


def _payload_domain(
    payload: dict[str, object], key: str, detail: str
) -> LoggingDomain | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise PlanSidecarError(
            f"plan episode sidecar is corrupt: {key!r} is invalid{detail}"
        )
    save_dir = value.get("save_dir")
    session_prefix = value.get("session_prefix")
    if not isinstance(save_dir, str) or not isinstance(session_prefix, str):
        raise PlanSidecarError(
            f"plan episode sidecar is corrupt: {key!r} is incomplete{detail}"
        )
    return LoggingDomain(save_dir=save_dir, session_prefix=session_prefix)


# --- Transitions ------------------------------------------------------------


def claim_accept(
    directory: Path, *, session_id: str, owner_domain: LoggingDomain
) -> PlanEpisodeSidecar:
    """P1: claim a draft episode for an accept attempt, durably.

    The episode's immutable ``owner_domain`` is recorded by the first claim and
    only ever compared afterwards; a claim from another logging domain refuses.
    The attempt generation increments and ``c1_verified`` clears, so a fresh
    attempt never inherits the previous attempt's commit flag. The previous
    attempt's consent is dropped too: its snapshot is gone, so keeping the
    digest would name bytes that no longer exist.
    """
    with held_transition_fence(directory, holder_session_id=session_id):
        sidecar = read_sidecar_required(directory)
        _refuse_terminal(sidecar)
        if sidecar.state != STATE_DRAFT:
            raise PlanEpisodeStateError(
                f"plan episode {sidecar.episode_id} is already {sidecar.state} — "
                "accept is refused; inspect the episode status"
            )
        recorded = sidecar.owner_domain
        if recorded is not None and recorded != owner_domain:
            raise PlanEpisodeOwnerError(DOMAIN_REFUSAL)
        intended = replace(
            sidecar,
            state=STATE_ACCEPTING,
            claiming_session_id=session_id,
            owner_domain=owner_domain,
            scan_domain=owner_domain,
            attempt_generation=sidecar.attempt_generation + 1,
            c1_verified=False,
            consent_sha256=None,
        )
        return _durable_transition(directory, sidecar, intended)


def release_claim(
    directory: Path, *, session_id: str, owner_domain: LoggingDomain
) -> PlanEpisodeSidecar:
    """Release a PRE-COMMIT accept claim and return the episode to draft.

    Only the owner may release: the sidecar's durable ``claiming_session_id``
    and its immutable ``owner_domain`` must both match the caller — the fence
    payload is diagnostics and proves nothing. A ``c1_verified`` claim is
    post-commit and is exited through the crash-recovery rows instead.

    The snapshot is unlinked BEFORE the durable return to draft, so a crash in
    the window leaves an accepting sidecar without a snapshot (a recognized
    recovery row) rather than a draft holding a stale snapshot that would block
    the next attempt's exclusive create.
    """
    with held_transition_fence(directory, holder_session_id=session_id):
        sidecar = read_sidecar_required(directory)
        _refuse_terminal(sidecar)
        if sidecar.state != STATE_ACCEPTING:
            raise PlanEpisodeStateError(
                f"plan episode {sidecar.episode_id} holds no accept claim "
                f"(state {sidecar.state})"
            )
        return _release_pre_commit_claim(
            directory, sidecar, session_id=session_id, owner_domain=owner_domain
        )


def _release_pre_commit_claim(
    directory: Path,
    sidecar: PlanEpisodeSidecar,
    *,
    session_id: str,
    owner_domain: LoggingDomain,
) -> PlanEpisodeSidecar:
    """The release body shared by release_claim and the cancel transition.

    The caller already holds the fence and has verified the accepting state.
    """
    require_episode_owner(sidecar, session_id=session_id, owner_domain=owner_domain)
    if sidecar.c1_verified:
        raise PlanEpisodeStateError(_POST_COMMIT_REFUSAL)
    unlink_accepted_snapshot(directory, sidecar.episode_id)
    return _durable_transition(directory, sidecar, _draft_view(sidecar))


def abort_accept(
    directory: Path, *, session_id: str, owner_domain: LoggingDomain
) -> PlanEpisodeSidecar:
    """/plan cancel and /clear during an accept attempt: cancel IS abort-accept.

    The claim is released, the snapshot unlinked in the same PRE-COMMIT scope,
    and the episode returns to draft with ``c1_verified`` cleared. Pins stay
    untouched and no cleanup runs: planning continues on the same episode.
    """
    return release_claim(directory, session_id=session_id, owner_domain=owner_domain)


def cancel_episode(
    directory: Path, *, session_id: str, owner_domain: LoggingDomain
) -> PlanEpisodeSidecar:
    """/plan cancel and /clear OUTSIDE the accept flow: the terminal cancel.

    The sidecar is the arbiter, so the cancel is a fenced, expected-state
    verified transition rather than a record edit: a draft episode is written
    ``cancelled`` durably (through the plan-mode temporary-name wrapper) and
    that state is TERMINAL — a new episode is the only continuation, so a
    cancelled episode can never be re-claimed, re-bound, or re-cancelled.

    Two legs, both decided by the state read under the fence:

    * ``accepting`` — the owner's cancel IS abort-accept (DR-17): the claim is
      released, the PRE-COMMIT snapshot unlinked, and the episode returns to
      draft so planning continues. A post-commit claim refuses (the fence is
      held by the commit and the crash-recovery rows govern it), and a claim
      held by another session or logging domain refuses.
    * ``draft`` — the terminal write. The recorded immutable ``owner_domain``
      (when a claim recorded one) must match the acting domain; a draft that
      was never claimed records none, so there is no owner to compare.

    ``accepted`` and ``cancelled`` refuse as terminal. The caller runs the
    cleanup whitelist separately; this transition never removes the episode's
    own arbiter entries.
    """
    with held_transition_fence(directory, holder_session_id=session_id):
        sidecar = read_sidecar_required(directory)
        _refuse_terminal(sidecar)
        if sidecar.state == STATE_ACCEPTING:
            return _release_pre_commit_claim(
                directory, sidecar, session_id=session_id, owner_domain=owner_domain
            )
        recorded = sidecar.owner_domain
        if recorded is not None and recorded != owner_domain:
            raise PlanEpisodeOwnerError(DOMAIN_REFUSAL)
        cancelled = replace(
            sidecar, state=STATE_CANCELLED, claiming_session_id=None, c1_verified=False
        )
        return _durable_transition(directory, sidecar, cancelled)


# --- The acceptance PREPARE transaction -------------------------------------


def sha256_bytes(content: bytes) -> str:
    """Digest RAW bytes; the bytes are the authority, never a decoded text."""
    return hashlib.sha256(content).hexdigest()


def _bound_resolution(path: Path) -> Path:
    """Resolve a bound document path, reporting a resolution failure as unreadable."""
    try:
        return Path(path).resolve()
    except OSError as error:
        raise PlanDraftReadError(
            f"cannot resolve the plan document: {path}: {error}"
        ) from error


def open_bound_document(path: Path) -> BinaryIO:
    """Open a bound document through a re-validated, dirfd-anchored read.

    A bound draft path is stored FULLY RESOLVED — a re-bind canonicalizes it
    with its symbolic links followed — so the stored path must still resolve to
    itself. A component swapped for a symbolic link after the binding makes the
    stored path name a different document, and that mismatch refuses instead of
    redirecting the read outside the authorized root.

    The open is anchored to the parent directory's descriptor (opened
    ``O_NOFOLLOW``, so a swapped parent refuses rather than being followed) and
    the leaf is opened relative to it ``O_NOFOLLOW``, so a leaf swapped for a
    symbolic link cannot redirect the bytes either. The binding is re-resolved
    AFTER the open, so a swap that lands while the read is being established
    refuses too.
    """
    target = Path(path)
    resolved = _bound_resolution(target)
    if resolved != target:
        raise PlanDraftReadError(
            f"the plan document moved after it was bound: {target} now resolves "
            f"to {resolved} — re-bind the document"
        )
    try:
        directory_descriptor = os.open(
            resolved.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        )
    except OSError as error:
        raise PlanDraftReadError(
            f"cannot open the plan document's directory: {resolved.parent}: {error}"
        ) from error
    try:
        try:
            descriptor = os.open(
                resolved.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_descriptor
            )
        except OSError as error:
            raise PlanDraftReadError(
                f"cannot read the plan document bytes: {resolved}: {error}"
            ) from error
        try:
            again = _bound_resolution(target)
        except PlanDraftReadError:
            os.close(descriptor)
            raise
        if again != resolved:
            os.close(descriptor)
            raise PlanDraftReadError(
                f"the plan document moved while it was being read: {target} "
                f"({resolved} -> {again}) — re-bind the document"
            )
    finally:
        os.close(directory_descriptor)
    return os.fdopen(descriptor, "rb")


def read_draft_bytes(path: Path) -> bytes:
    """Read the draft's RAW bytes for retention and re-comparison.

    The digest is over the file exactly as written. The shared document reader
    newline-normalizes through safe decoding, so it is never the digest source;
    the read itself is :func:`open_bound_document`'s anchored, re-validated one.
    """
    try:
        with open_bound_document(path) as stream:
            return stream.read()
    except PlanDraftReadError:
        raise
    except OSError as error:
        raise PlanDraftReadError(
            f"cannot read the plan document bytes: {path}: {error}"
        ) from error


def confirm_draft_unchanged(
    draft_path: Path, *, previewed_bytes: bytes, previewed_sha256: str
) -> bytes:
    """P2.5: re-read the draft and byte-compare it against the preview.

    Consent applies to the bytes last shown, so ANY difference — a different
    digest or a different byte string — refuses and demands a re-preview. The
    caller holds no fence here; that is why the re-read can be a whole user
    cycle long without blocking another transition.
    """
    reread = read_draft_bytes(draft_path)
    if reread != previewed_bytes or sha256_bytes(reread) != previewed_sha256:
        raise PlanDocumentChangedError(REPREVIEW_REQUIRED_MESSAGE)
    return reread


def publish_consent_snapshot(
    directory: Path,
    *,
    session_id: str,
    owner_domain: LoggingDomain,
    content: bytes,
    consent_sha256: str,
) -> PlanEpisodeSidecar:
    """P3: publish the retained bytes under the fence and record their digest.

    The snapshot is created EXCLUSIVELY from the retained bytes — never a
    re-read, never a write-temp-and-rename that would silently overwrite a
    survivor — and the sidecar's ``consent_sha256`` write is the same fenced
    transaction: a snapshot whose digest never reached the sidecar cannot be
    consented. A failed state write removes the just-created snapshot, so the
    PRE-COMMIT scope never leaves an unconsented file behind.
    """
    with held_transition_fence(directory, holder_session_id=session_id):
        sidecar = read_sidecar_required(directory)
        _refuse_terminal(sidecar)
        if sidecar.state != STATE_ACCEPTING:
            raise PlanEpisodeStateError(
                f"plan episode {sidecar.episode_id} holds no accept claim "
                f"(state {sidecar.state})"
            )
        require_episode_owner(sidecar, session_id=session_id, owner_domain=owner_domain)
        if sidecar.c1_verified:
            raise PlanEpisodeStateError(_POST_COMMIT_REFUSAL)
        snapshot = accepted_snapshot_path(directory, sidecar.episode_id)
        try:
            publish_episode_snapshot(snapshot, content)
        except FileExistsError as error:
            raise PlanSnapshotCollisionError(
                f"{SNAPSHOT_COLLISION_MESSAGE}: {snapshot}"
            ) from error
        except OSError as error:
            raise PlanTransitionError(
                f"cannot publish the plan consent snapshot: {snapshot}: {error}"
            ) from error
        intended = replace(sidecar, consent_sha256=consent_sha256)
        try:
            return _durable_transition(directory, sidecar, intended)
        except BaseException:
            unlink_accepted_snapshot(directory, sidecar.episode_id)
            raise


def prepare_accept(
    directory: Path,
    *,
    episode_id: str,
    session_id: str,
    owner_domain: LoggingDomain,
    previewed_bytes: bytes,
    previewed_sha256: str,
) -> PreparedAccept:
    """Run the PREPARE half (P1-P3) of a consent handoff.

    P1 claims the draft episode under the fence (owner domain recorded, attempt
    generation incremented, ``c1_verified`` cleared). P2 reads and RETAINS the
    draft's raw bytes; P2.5 re-reads and byte-compares them against the bytes
    the user previewed. P3 publishes the retained bytes exclusively and records
    the digest in the sidecar, each under its own fenced transition.

    Every observed exit after the claim releases it: the claim (guarded by the
    sidecar's claiming identity), the PRE-COMMIT snapshot, and the durable
    return to draft with ``c1_verified`` cleared happen before this raises. On
    success the claim is HELD for the commit half.
    """
    sidecar = claim_accept(directory, session_id=session_id, owner_domain=owner_domain)
    try:
        if sidecar.episode_id != episode_id:
            raise PlanEpisodeStateError(
                f"the plan episode {sidecar.episode_id} does not match {episode_id}"
            )
        retained = read_draft_bytes(sidecar.draft_path)
        retained_sha256 = sha256_bytes(retained)
        if retained != previewed_bytes or retained_sha256 != previewed_sha256:
            raise PlanDocumentChangedError(REPREVIEW_REQUIRED_MESSAGE)
        confirm_draft_unchanged(
            sidecar.draft_path,
            previewed_bytes=retained,
            previewed_sha256=retained_sha256,
        )
        published = publish_consent_snapshot(
            directory,
            session_id=session_id,
            owner_domain=owner_domain,
            content=retained,
            consent_sha256=retained_sha256,
        )
    except BaseException as error:
        try:
            release_claim(directory, session_id=session_id, owner_domain=owner_domain)
        except BaseException as release_error:
            raise release_error from error
        raise
    return PreparedAccept(
        sidecar=published,
        snapshot_path=accepted_snapshot_path(directory, published.episode_id),
        retained_bytes=retained,
        consent_sha256=retained_sha256,
    )


# --- The acceptance COMMIT transaction --------------------------------------


def _verify_snapshot_digest(path: Path, consent_sha256: str) -> None:
    """Cross-check the snapshot FILE against the consented digest.

    C1's second reading: the carried bytes are the authority, and this proves the
    durable provenance carrier holds the same bytes. A missing or unreadable
    snapshot is a mismatch — the handoff refuses rather than guessing.
    """
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise PlanDigestMismatchError(
            f"{SNAPSHOT_DIGEST_MISMATCH_MESSAGE}: {path}: {error}"
        ) from error
    if sha256_bytes(raw) != consent_sha256:
        raise PlanDigestMismatchError(f"{SNAPSHOT_DIGEST_MISMATCH_MESSAGE}: {path}")


def begin_commit(
    directory: Path,
    *,
    episode_id: str,
    session_id: str,
    owner_domain: LoggingDomain,
    carried_bytes: bytes,
    snapshot_path: Path,
    retry_timeout: float = 0.0,
) -> CommitAttempt:
    """Acquire the commit fence and run C1, returning the held attempt.

    The fence is taken non-blockingly; a busy fence retries for the bounded
    timeout and then surfaces as :class:`PlanFenceBusyError` ("that transition is
    running right now"). Once a running commit completes, the next attempt is
    refused by the expected-state verify below with the status surface.

    C1 runs BEFORE any destructive step: the sidecar's ``consent_sha256`` must
    equal the digest of the carried bytes — the one authority — and the snapshot
    file must hold the same bytes. Either mismatch refuses. On success the
    sidecar's attempt-scoped ``c1_verified`` flag is set DURABLY, which is what
    tells a later observer that this attempt's commit began. The returned
    attempt holds the fence; the caller must close it after C5.
    """
    fence = TransitionFence(directory)
    fence.acquire(holder_session_id=session_id, retry_timeout=max(retry_timeout, 0.0))
    try:
        sidecar = read_sidecar_required(directory)
        _refuse_terminal(sidecar)
        if sidecar.episode_id != episode_id:
            raise PlanEpisodeStateError(
                f"the plan episode {sidecar.episode_id} does not match {episode_id}"
            )
        if sidecar.state != STATE_ACCEPTING:
            raise PlanEpisodeStateError(
                f"plan episode {sidecar.episode_id} holds no accept claim "
                f"(state {sidecar.state})"
            )
        require_episode_owner(sidecar, session_id=session_id, owner_domain=owner_domain)
        consent_sha256 = sidecar.consent_sha256
        if consent_sha256 is None:
            raise PlanArbiterUnresolvableError(UNRESOLVABLE_ARBITER_MESSAGE)
        if sha256_bytes(carried_bytes) != consent_sha256:
            raise PlanDigestMismatchError(CARRIED_DIGEST_MISMATCH_MESSAGE)
        _verify_snapshot_digest(snapshot_path, consent_sha256)
        marked = _durable_transition(
            directory, sidecar, replace(sidecar, c1_verified=True)
        )
        return CommitAttempt(
            directory=directory,
            episode_id=marked.episode_id,
            session_id=session_id,
            owner_domain=owner_domain,
            snapshot_path=snapshot_path,
            carried_bytes=carried_bytes,
            sidecar=marked,
            _fence=fence,
        )
    except BaseException:
        fence.close()
        raise


def rename_accepted_snapshot(
    directory: Path, *, episode_id: str, generation: int
) -> str:
    """Rename the live-named snapshot to the attempt's provenance name.

    This is C5's derived follow-on and the flip-agnostic COMPLETION RULE: it runs
    whenever an accepted episode's live path still exists, so both observations
    after the terminal flip and the owner adjudication can complete it. The
    rename is IDEMPOTENT — an absent live path with the attempt's provenance file
    present is a verified NO-OP, because the desired end state already holds.
    Both names are permanent provenance; nothing here ever unlinks a snapshot.
    """
    live = accepted_snapshot_path(directory, episode_id)
    provenance = provenance_snapshot_path(directory, episode_id, generation)
    live_exists = live.is_file()
    provenance_exists = provenance.is_file()
    if not live_exists:
        if provenance_exists:
            return SNAPSHOT_RENAME_NOOP
        raise PlanTransitionError(
            "the plan consent snapshot is absent under both names: "
            f"{live} / {provenance}"
        )
    if provenance_exists:
        raise PlanTransitionError(
            f"the plan provenance name is already occupied: {provenance}"
        )
    try:
        os.replace(live, provenance)
    except OSError as error:
        raise PlanTransitionError(
            f"cannot rename the plan consent snapshot: {live} -> {provenance}: {error}"
        ) from error
    fsync_directory(directory)
    return SNAPSHOT_RENAMED


def observe_logging_domain(
    *, save_dir: str | Path | None, session_prefix: str | None, session_dir: Path | None
) -> LoggingObservation:
    """C3: derive the scan-domain record and the marker destination from ONE read.

    The logging configuration is observed exactly once: ``{save_dir,
    session_prefix}`` becomes the sidecar's ``scan_domain`` record and
    ``session_dir`` is where the executor's durable marker will land. The
    binding is verified here rather than assumed — a session directory outside
    the observed domain refuses loudly, because then the recorded domain could
    not find the marker it was recorded for (DR-68).
    """
    if save_dir is None or session_prefix is None or session_dir is None:
        raise PlanLoggingDisabledError(LOGGING_REQUIRED_MESSAGE)
    domain = LoggingDomain(save_dir=str(save_dir), session_prefix=session_prefix)
    observed = Path(session_dir)
    prefix = f"{domain.session_prefix}_"
    if observed.parent != Path(domain.save_dir) or not observed.name.startswith(prefix):
        raise PlanEpisodeWriteError(
            "the executor session directory does not derive from the observed "
            f"logging configuration: {observed} is not under "
            f"{domain.save_dir}{os.sep}{prefix}*"
        )
    return LoggingObservation(domain=domain, session_dir=observed)


def require_no_executor(
    domain: LoggingDomain, episode_id: str, *, excluding_session_id: str | None = None
) -> None:
    """The fenced pre-launch re-scan: refuse a marker that precedes the write.

    Any scan error fails closed — the caller refuses rather than reading an
    untrusted scan as "not executed" — and a discovered executor is refused
    loudly, never launched twice. ``excluding_session_id`` names the claiming
    source session, whose own record carries the same episode from P1: the
    committing session is never mistaken for a launched executor.
    """
    found = scan_episode_executor(
        domain, episode_id, excluding_session_id=excluding_session_id
    )
    if found.found:
        raise PlanEpisodeStateError(EXECUTOR_EXISTS_MESSAGE)


def build_handoff_context(
    *, content: bytes, episode_id: str, snapshot_path: Path, consent_sha256: str
) -> str:
    """C3 [3]: the executor's initial context — the consented bytes plus prose.

    The prose records the snapshot's provenance and digest as DOCUMENTATION
    only: the runtime digest verification is the enforcement, and a model-level
    hash check is supplemental. The document region is the carried bytes decoded
    WITHOUT newline normalization, so the injected text reproduces the snapshot
    exactly rather than a reader's rewritten view of it.
    """
    document = content.decode("utf-8", errors="replace")
    return (
        f"Execute the accepted plan for episode {episode_id}.\n"
        "\n"
        "The plan document below is reproduced verbatim from the consent "
        "snapshot: it is exactly the document that was reviewed and consented "
        "to. The provenance and digest lines are documentation only — the "
        "runtime already verified these bytes against the consent digest.\n"
        "\n"
        f"Snapshot: {snapshot_path}\n"
        f"Consent SHA-256: {consent_sha256}\n"
        f"{SEED_DOCUMENT_BEGIN}\n"
        f"{document}"
        f"\n{SEED_DOCUMENT_END}\n"
    )


def rebind_draft(
    directory: Path,
    *,
    draft_path: Path,
    authorized_roots: Iterable[Path],
    session_id: str,
    owner_domain: LoggingDomain | None = None,
) -> PlanEpisodeSidecar:
    """Atomically re-bind the episode's draft path under the fence.

    The target is resolved and re-validated UNDER the fence: its symbolic links
    are followed, it must lie inside one of the authorized roots, and a second
    resolution must agree with the first — a component swapped for a symbolic
    link between the two sends the second resolution somewhere else, and the
    re-bind refuses instead of writing a binding that could move later. The
    stored path is the canonical one, which is what the bound reads demand.

    A held claim refuses: abort-accept first, then re-bind. Re-binding judges no
    document content — the mandatory re-preview and re-consent gate the bytes
    afterwards.
    """
    with held_transition_fence(directory, holder_session_id=session_id):
        sidecar = read_sidecar_required(directory)
        _refuse_terminal(sidecar)
        if sidecar.state == STATE_ACCEPTING:
            raise PlanEpisodeStateError(
                "an accept claim is held — abort-accept first, then re-bind"
            )
        if sidecar.owner_domain is not None and sidecar.owner_domain != owner_domain:
            raise PlanEpisodeOwnerError(DOMAIN_REFUSAL)
        canonical = _authorized_resolution(draft_path, authorized_roots)
        return _durable_transition(
            directory, sidecar, replace(sidecar, draft_path=canonical)
        )


def clear_during_claim(
    directory: Path,
    *,
    session_id: str,
    owner_domain: LoggingDomain,
    clear_record: Callable[[], None],
) -> PlanEpisodeSidecar | None:
    """/clear while an accept claim is held: release the claim FIRST.

    The observed exit orders the abort-accept before the session record clears,
    so a silent accepting zombie can never survive a /clear; the episode resumes
    as draft. A claim held by another session refuses, and a post-commit claim
    refuses — both without clearing the record.
    """
    sidecar = read_sidecar(directory)
    if sidecar is None or sidecar.state != STATE_ACCEPTING:
        clear_record()
        return None
    released = release_claim(
        directory, session_id=session_id, owner_domain=owner_domain
    )
    clear_record()
    return released


def require_episode_owner(
    sidecar: PlanEpisodeSidecar, *, session_id: str, owner_domain: LoggingDomain
) -> None:
    """Verify the episode's durable owner record for an owner-only action.

    ``owner_domain`` is immutable for the episode's lifetime, so a discovery
    record update (``scan_domain``) can never delegitimize the source owner.
    """
    if sidecar.owner_domain is None:
        raise PlanArbiterUnresolvableError(UNRESOLVABLE_ARBITER_MESSAGE)
    if sidecar.owner_domain != owner_domain:
        raise PlanEpisodeOwnerError(DOMAIN_REFUSAL)
    if sidecar.claiming_session_id != session_id:
        holder = sidecar.claiming_session_id or "none"
        raise PlanEpisodeOwnerError(
            f"the plan episode claim is held by session {holder} — the owning "
            "session must act"
        )


def canonicalize_authorized_path(path: Path, authorized_roots: Iterable[Path]) -> Path:
    """Resolve ``path`` and require it to lie inside an authorized root."""
    try:
        target = Path(path).expanduser().resolve()
    except OSError as error:
        raise PlanEpisodeBindError(
            f"cannot resolve the plan document: {path}"
        ) from error
    for root in authorized_roots:
        try:
            resolved_root = Path(root).expanduser().resolve()
        except OSError:
            continue
        if target == resolved_root or target.is_relative_to(resolved_root):
            return target
    raise PlanEpisodeBindError(
        f"plan document is outside the authorized roots: {target}"
    )


def _authorized_resolution(path: Path, authorized_roots: Iterable[Path]) -> Path:
    """Resolve an authorized target TWICE; a path that moves between refuses.

    The first resolution is the containment check; the second runs under the
    same held fence and must land on the same canonical path. A component
    swapped for a symbolic link in the window between them sends the second
    resolution elsewhere — outside the authorized root included — and that
    mismatch refuses the re-bind rather than being written into the sidecar.
    """
    canonical = canonicalize_authorized_path(path, authorized_roots)
    revalidated = canonicalize_authorized_path(path, authorized_roots)
    if revalidated != canonical:
        raise PlanEpisodeBindError(
            f"the plan document moved while it was being validated: {path} "
            f"({canonical} -> {revalidated})"
        )
    return canonical


def unlink_accepted_snapshot(directory: Path, episode_id: str) -> bool:
    """Remove the live-named snapshot, directory-syncing the arbiter change."""
    path = accepted_snapshot_path(directory, episode_id)
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    except OSError as error:
        raise PlanTransitionError(
            f"cannot unlink the plan snapshot: {path}: {error}"
        ) from error
    fsync_directory(directory)
    return True


def _refuse_terminal(sidecar: PlanEpisodeSidecar) -> None:
    if sidecar.state in TERMINAL_STATES:
        raise PlanEpisodeTerminalError(
            f"plan episode {sidecar.episode_id} is {sidecar.state}; "
            "a new episode is the only continuation"
        )


def _durable_transition(
    directory: Path, previous: PlanEpisodeSidecar, intended: PlanEpisodeSidecar
) -> PlanEpisodeSidecar:
    try:
        write_sidecar(directory, intended)
    except OSError as error:
        _surface_write_failure(directory, previous, intended, error)
    return intended


def _surface_write_failure(
    directory: Path,
    previous: PlanEpisodeSidecar,
    intended: PlanEpisodeSidecar,
    error: OSError,
) -> NoReturn:
    """Refuse loudly; an error never proves the previous state remains (DR-56)."""
    try:
        observed = read_sidecar(directory)
    except PlanSidecarError:
        observed = None
    if observed == intended:
        raise PlanEpisodeWriteError(
            "the plan episode state write failed after publication; the new "
            "state is possibly published — inspect the episode before retrying"
        ) from error
    if observed == previous:
        raise PlanEpisodeWriteError(
            f"the plan episode state write failed: {error}"
        ) from error
    raise PlanArbiterUnresolvableError(UNRESOLVABLE_ARBITER_MESSAGE) from error


# --- The logging-disabled entry guard ---------------------------------------


def require_session_logging(enabled: bool) -> None:
    """Refuse plan mode when session logging is disabled.

    The per-session lease carrying "one live process per session id" is
    bypassed in that configuration, so plan mode — whose arbiter assumes exactly
    one owner process per session — refuses loudly instead of running unguarded.
    """
    if not enabled:
        raise PlanLoggingDisabledError(LOGGING_REQUIRED_MESSAGE)


def guard_plan_entry(enabled: bool) -> None:
    """The /plan entry guard; the invariant holds in every configuration."""
    require_session_logging(enabled)


def guard_plan_record_resume(enabled: bool) -> None:
    """The plan-record resume guard: the same invariant on the restore path."""
    require_session_logging(enabled)


# --- The bounded executor scan ----------------------------------------------


def scan_episode_executor(
    domain: LoggingDomain,
    episode_id: str,
    *,
    excluding_session_id: str | None = None,
    expected_session_id: str | None = None,
) -> ExecutorScan:
    """Scan the RECORDED logging domain's session metas for the executor marker.

    The metas are the source of truth, so the scan enumerates them directly —
    never the listing index, which is a display cache. The scan is bounded by
    the recorded domain, not by the recovery-time configuration. An unreadable
    domain, a missing or corrupt meta, or an ambiguous double match all fail
    closed: the caller must refuse rather than guess.

    A meta MATCHES as an executor only on the marker's own semantics (see
    :func:`_records_an_executor`): the episode id ALONE also appears in every
    planning session's own mirror — the departed planner, and a fork's copy —
    so matching on the id alone once mistook a predecessor for a launched
    executor. A planning mirror is a clean miss; an unreadable or corrupt meta,
    and several matching markers, still fail closed.

    ``expected_session_id`` is the executor the SIDECAR recorded at C4 (see
    :meth:`CommitAttempt.record_executor_session`). A restored ``accepting``
    planning record is marker-shaped too (it carries the episode's consent
    digest and snapshot name from the sidecar), so marker shape alone cannot
    tell a restored plan from the launch: when the sidecar names the executor,
    only that session's meta matches and every other marker-shaped mirror is a
    clean miss. Without a recorded executor (no sidecar, or a crash before C4)
    the caller passes ``None`` and the marker shape is the only available
    predicate — the documented, narrower fallback.

    C4 runs AFTER C3's marker write, so a recorded executor with no matching
    meta is not a miss: either the marker was never made durable or its meta
    is now absent, unreadable, or no longer marker-shaped. Execution may have
    begun, so that FAILS CLOSED (design §4.4) — the caller refuses and the
    status surface shows, rather than re-offering the handoff. The one
    legitimate miss is the observer's own exclusion: when the recorded
    executor IS ``excluding_session_id``, the match exists and is dropped.

    ``excluding_session_id`` drops one session from the result: the claiming
    source session's own record carries the episode from P1, so an observer
    that excludes itself never mistakes its own meta for a launched executor
    and never trips the ambiguity rule on that pair.
    """
    entries = _session_directories(Path(domain.save_dir), domain.session_prefix)
    recorded = [
        session_id
        for entry in entries
        if (
            session_id := _episode_marker_session_id(
                entry, episode_id, expected_session_id=expected_session_id
            )
        )
        is not None
    ]
    if expected_session_id is not None and not recorded:
        raise PlanEpisodeScanError(
            f"the plan episode sidecar records executor session "
            f"{expected_session_id} for episode {episode_id}, but no session meta "
            "matches it — the executor's meta is absent, unreadable, or not "
            "marker-shaped; fail closed"
        )
    matched = [
        session_id for session_id in recorded if session_id != excluding_session_id
    ]
    if len(matched) > 1:
        raise PlanEpisodeScanError(
            f"several executor sessions record plan episode {episode_id}: "
            f"{sorted(matched)}"
        )
    return ExecutorScan(matched[0] if matched else None)


def _session_directories(save_dir: Path, session_prefix: str) -> list[Path]:
    try:
        if not save_dir.is_dir():
            raise PlanEpisodeScanError(
                f"the recorded plan domain is unreadable: {save_dir}"
            )
        return sorted(
            entry for entry in save_dir.glob(f"{session_prefix}_*") if entry.is_dir()
        )
    except OSError as error:
        raise PlanEpisodeScanError(
            f"the recorded plan domain cannot be scanned: {save_dir}: {error}"
        ) from error


# The fields C3's marker write carries beside the episode id: the consented
# digest and the snapshot name the executor was launched with. A planning
# session's own mirror carries the episode id without them.
EXECUTOR_MARKER_FIELDS = ("consent_sha256", "snapshot_path")


def _episode_marker_session_id(
    directory: Path, episode_id: str, *, expected_session_id: str | None = None
) -> str | None:
    meta_path = directory / SESSION_META_FILENAME
    try:
        raw = meta_path.read_bytes()
    except OSError as error:
        raise PlanEpisodeScanError(
            f"cannot read the session meta {meta_path}: {error}"
        ) from error
    try:
        payload = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as error:
        raise PlanEpisodeScanError(
            f"the session meta is corrupt: {meta_path}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise PlanEpisodeScanError(f"the session meta is corrupt: {meta_path}")
    record = payload.get("plan_mode")
    if not isinstance(record, dict) or record.get("episode_id") != episode_id:
        return None
    if not _records_an_executor(record):
        return None
    session_id = payload.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise PlanEpisodeScanError(
            f"the session meta records no session id: {meta_path}"
        )
    if expected_session_id is not None and session_id != expected_session_id:
        return None
    return session_id


def _records_an_executor(record: dict[str, object]) -> bool:
    """Whether an episode-marked meta belongs to the EXECUTOR, not to planning.

    C3's marker write is the executor's first durable act, and it records the
    sidecar's ``accepting`` state beside the episode's consent digest and
    snapshot name. A PLANNING session's own mirror carries the same episode id
    under ITS lifecycle state instead — ``draft`` before C2 retires it,
    ``accepted`` after, and ``cancelled`` for a terminal cancel — so the
    episode id alone cannot identify an executor: once the sidecar was
    reclaimed, a departed planner matched and surfaced as "executed by
    <predecessor>".

    The lifecycle state is therefore the discriminator, and the marker fields
    prove the record came from the handoff. The marker write is one complete
    durable write (consent digest and snapshot name are set by construction at
    C3), so a record that is not marker-shaped is a PLANNING record — a clean
    miss — rather than an ambiguity the whole domain's scans must fail on.
    Nothing is guessed: an unreadable or corrupt meta still fails closed above,
    and several matching markers still fail closed in the caller.

    Shape alone is not enough once a PLANNING session restores an ``accepting``
    episode: ``record_from_sidecar`` rebuilds that record from the sidecar, so
    it carries the same state and marker fields as the launch. When the sidecar
    names the executor (:meth:`CommitAttempt.record_executor_session`), the scan
    narrows on that session id and the restored mirror is a clean miss; marker
    shape remains the predicate only while no executor id is recorded.
    """
    if record.get("state") != STATE_ACCEPTING:
        return False
    return any(
        isinstance(record.get(field), str) and record.get(field)
        for field in EXECUTOR_MARKER_FIELDS
    )


# --- Crash recovery (design §4.4) -------------------------------------------

RECOVERY_ROW_DRAFT = "draft"
RECOVERY_ROW_ACCEPTING_NO_SNAPSHOT = "accepting-no-snapshot"
RECOVERY_ROW_ACCEPTING_SNAPSHOT = "accepting-snapshot"
RECOVERY_ROW_POSSIBLY_EXECUTED = "possibly-executed"
RECOVERY_ROW_ACCEPTED = "accepted"
RECOVERY_ROW_CANCELLED = "cancelled"
RECOVERY_ROW_EXECUTED = "executed"
RECOVERY_ROW_UNRESOLVABLE = "unresolvable"
RECOVERY_ROW_LEGACY = "legacy-unverified"

AUTHORITY_SIDECAR = "sidecar"
AUTHORITY_RECORD = "record"
AUTHORITY_RECORD_TERMINAL = "record-terminal"

RESTORATION_FAILED_MESSAGE = (
    "the plan record could not be restored from the episode sidecar — the "
    "episode is unresolvable; rebind to a fresh episode or start-new"
)
POSSIBLY_EXECUTED_MESSAGE = (
    "an executor session already records this plan episode; cancelling would "
    "destroy the evidence of a possibly executed handoff — inspect the executor "
    "or mark the episode accepted"
)


class PlanEpisodePossiblyExecutedError(PlanTransitionError):
    """An executor marker refuses the cancel; the episode may already be running."""

    def __init__(self, message: str, *, executor_session_id: str | None) -> None:
        super().__init__(message)
        self.executor_session_id = executor_session_id


@dataclass(frozen=True, slots=True)
class EpisodeObservation:
    """One sidecar-first observation of an episode's recoverable state.

    ``row`` names the design's crash-table row the episode currently sits in.
    ``authority`` says what decided it: an existing sidecar always wins; without
    one, a terminal record stays display authority as history and a non-terminal
    record is resolved only by its P1-domain mirror scan.
    """

    row: str
    state: str | None
    episode_id: str | None
    authority: str
    executor_session_id: str | None = None
    snapshot_name: str | None = None
    owner: bool = False
    record_mismatch: bool = False
    rename_outcome: str | None = None

    @property
    def terminal(self) -> bool:
        """A terminal episode is history; only a new episode continues."""
        return self.row in {RECOVERY_ROW_ACCEPTED, RECOVERY_ROW_CANCELLED}

    @property
    def recoverable(self) -> bool:
        """An accepting episode can be re-offered or cancelled by its owner."""
        return self.row in {
            RECOVERY_ROW_ACCEPTING_NO_SNAPSHOT,
            RECOVERY_ROW_ACCEPTING_SNAPSHOT,
        }

    @property
    def possibly_executed(self) -> bool:
        """An episode-marked executor meta exists; never auto-retry."""
        return self.row in {RECOVERY_ROW_POSSIBLY_EXECUTED, RECOVERY_ROW_EXECUTED}


@dataclass(frozen=True, slots=True)
class RecoveryOffer:
    """Row 3's verified re-offer: the rebuilt handoff body with the claim held."""

    sidecar: PlanEpisodeSidecar
    snapshot_path: Path
    content: bytes


def scan_sidecar_executor(
    sidecar: PlanEpisodeSidecar, *, excluding_session_id: str | None = None
) -> ExecutorScan:
    """Scan the SIDECAR's recorded domain for this episode's executor marker.

    The sidecar is the arbiter and its ``scan_domain`` is the one mutable
    discovery record C3 re-records pre-marker — so it, not the recovery-time
    configuration, bounds the scan. No recorded domain means no bounded scan is
    possible, and that fails closed. ``excluding_session_id`` names the claiming
    source session, whose own record carries the same episode from P1: the
    observer is never mistaken for a launched executor.

    A sidecar that recorded its executor (C4) narrows the scan to that session,
    so a restored ``accepting`` plan's marker-shaped mirror is a clean miss
    rather than a second "executor". ``None`` there means no executor was
    recorded — the pre-C4 window — and the marker shape alone decides.
    """
    if sidecar.scan_domain is None:
        raise PlanEpisodeScanError(
            f"plan episode {sidecar.episode_id} records no scan domain — the "
            "executor scan is unbounded and refuses"
        )
    return scan_episode_executor(
        sidecar.scan_domain,
        sidecar.episode_id,
        excluding_session_id=excluding_session_id,
        expected_session_id=sidecar.executor_session_id,
    )


def scan_record_executor(
    record: PlanModeRecord, *, excluding_session_id: str | None = None
) -> ExecutorScan:
    """Scan a record's P1-domain MIRROR — the sidecar-loss display path.

    The record cannot be rewritten after the session id rotates, so it carries
    the P1 domain only; a clean miss falls to the unresolvable surface rather
    than being misclassified as executed. A record without a recorded domain
    fails closed exactly like a sidecar without one, and the record's OWN
    session is excluded for the same reason the sidecar scan excludes it.
    """
    if record.episode_id is None:
        raise PlanEpisodeScanError(
            "the plan record carries no episode identity — no executor scan exists"
        )
    if record.scan_domain is None:
        raise PlanEpisodeScanError(
            f"the plan record for episode {record.episode_id} records no scan "
            "domain — the executor scan is unbounded and refuses"
        )
    return scan_episode_executor(
        LoggingDomain(
            save_dir=record.scan_domain.save_dir,
            session_prefix=record.scan_domain.session_prefix,
        ),
        record.episode_id,
        excluding_session_id=excluding_session_id,
    )


def surviving_snapshot_name(directory: Path, sidecar: PlanEpisodeSidecar) -> str | None:
    """The episode's surviving snapshot name, by EITHER name (live first).

    The live name is checked first because it is the pre-completion state; the
    attempt's provenance name is the cancel-window survivor. Both are permanent
    provenance and neither is ever a cleanup target.
    """
    live = accepted_snapshot_path(directory, sidecar.episode_id)
    if live.exists():
        return live.name
    provenance = provenance_snapshot_path(
        directory, sidecar.episode_id, sidecar.attempt_generation
    )
    if provenance.exists():
        return provenance.name
    return None


def locate_episode_snapshot(
    directory: Path, sidecar: PlanEpisodeSidecar
) -> tuple[Path, bytes]:
    """Locate the surviving snapshot by EITHER name and verify its digest.

    The recovery path's declared carrier is the snapshot FILE (the previewed
    bytes died with the crashed process), so the located bytes must equal the
    sidecar's ``consent_sha256``. Absent under both names or mismatched fails
    closed: the caller refuses and re-consents rather than guessing.
    """
    consent_sha256 = sidecar.consent_sha256
    if consent_sha256 is None:
        raise PlanArbiterUnresolvableError(
            f"plan episode {sidecar.episode_id} records no consent digest — the "
            "surviving bytes cannot be verified; re-consent"
        )
    live = accepted_snapshot_path(directory, sidecar.episode_id)
    provenance = provenance_snapshot_path(
        directory, sidecar.episode_id, sidecar.attempt_generation
    )
    for candidate in (live, provenance):
        try:
            content = candidate.read_bytes()
        except FileNotFoundError:
            continue
        except OSError as error:
            raise PlanArbiterUnresolvableError(
                f"the surviving plan snapshot is unreadable: {candidate}: {error} "
                "— re-consent"
            ) from error
        if sha256_bytes(content) != consent_sha256:
            raise PlanDigestMismatchError(
                f"{SNAPSHOT_DIGEST_MISMATCH_MESSAGE}: {candidate}"
            )
        return candidate, content
    raise PlanArbiterUnresolvableError(
        f"{UNRESOLVABLE_ARBITER_MESSAGE}: the plan consent snapshot is absent "
        f"under both names ({live} / {provenance}) — re-consent"
    )


def complete_accepted_rename(directory: Path, sidecar: PlanEpisodeSidecar) -> str:
    """The flip-agnostic COMPLETION RULE, run at an observation surface.

    Whenever an ACCEPTED episode's live snapshot still exists, the derived rename
    runs as an idempotent follow-on — the status surface, cleanup, and the
    adjudication that flipped the sidecar all complete it. A completion that
    cannot run is honest-tolerated: the flip is authoritative, both names are
    permanent provenance, and an occupied live path is collision-harmless while
    the episode stays foreclosed.
    """
    if sidecar.state != STATE_ACCEPTED:
        raise PlanEpisodeStateError(
            f"plan episode {sidecar.episode_id} is {sidecar.state}, not accepted — "
            "the completion rule applies only after the terminal flip"
        )
    try:
        return rename_accepted_snapshot(
            directory,
            episode_id=sidecar.episode_id,
            generation=sidecar.attempt_generation,
        )
    except PlanTransitionError:
        return SNAPSHOT_RENAME_PENDING


def observe_episode(
    directory: Path,
    *,
    record: PlanModeRecord | None,
    session_id: str,
    owner_domain: LoggingDomain | None = None,
) -> EpisodeObservation:
    """Observe an episode sidecar-FIRST and classify its crash-recovery row.

    The sidecar is consulted first for every display and, when present, is the
    sole authority; its recorded domain bounds the executor scan. When it is
    missing, a terminal record remains valid display authority as history (it can
    never re-arm), and a non-terminal record is NOT assumed draft — its P1-domain
    mirror is scanned. Any scan error fails closed (it raises rather than reading
    an untrusted scan as a clean miss), and an accepted episode's live snapshot
    is completed by the rename rule at this observation.
    """
    sidecar = read_sidecar(directory)
    if sidecar is None:
        return _observe_without_sidecar(record, session_id=session_id)
    return _observe_from_sidecar(
        directory, sidecar, record, session_id=session_id, owner_domain=owner_domain
    )


def _observe_from_sidecar(
    directory: Path,
    sidecar: PlanEpisodeSidecar,
    record: PlanModeRecord | None,
    *,
    session_id: str,
    owner_domain: LoggingDomain | None,
) -> EpisodeObservation:
    executor_session_id: str | None = None
    rename_outcome: str | None = None
    if sidecar.state == STATE_ACCEPTING:
        scan = scan_sidecar_executor(sidecar, excluding_session_id=session_id)
        executor_session_id = scan.executor_session_id
        if scan.found:
            row = RECOVERY_ROW_POSSIBLY_EXECUTED
        elif surviving_snapshot_name(directory, sidecar) is not None:
            row = RECOVERY_ROW_ACCEPTING_SNAPSHOT
        else:
            row = RECOVERY_ROW_ACCEPTING_NO_SNAPSHOT
    elif sidecar.state == STATE_ACCEPTED:
        executor_session_id = sidecar.executor_session_id
        rename_outcome = complete_accepted_rename(directory, sidecar)
        row = RECOVERY_ROW_ACCEPTED
    elif sidecar.state == STATE_CANCELLED:
        row = RECOVERY_ROW_CANCELLED
    else:
        row = RECOVERY_ROW_DRAFT
    mismatch = (
        record is not None
        and record.episode_bound
        and record.episode_id != sidecar.episode_id
    )
    return EpisodeObservation(
        row=row,
        state=sidecar.state,
        episode_id=sidecar.episode_id,
        authority=AUTHORITY_SIDECAR,
        executor_session_id=executor_session_id,
        snapshot_name=surviving_snapshot_name(directory, sidecar),
        owner=_is_episode_owner(
            sidecar, session_id=session_id, owner_domain=owner_domain
        ),
        record_mismatch=mismatch,
        rename_outcome=rename_outcome,
    )


def _observe_without_sidecar(
    record: PlanModeRecord | None, *, session_id: str
) -> EpisodeObservation:
    if record is None or record.legacy_unverified:
        return EpisodeObservation(
            row=RECOVERY_ROW_LEGACY,
            state=LEGACY_UNVERIFIED_STATE if record is not None else None,
            episode_id=None,
            authority=AUTHORITY_RECORD,
        )
    state = record.display_state
    if state in TERMINAL_STATES:
        return EpisodeObservation(
            row=state,
            state=state,
            episode_id=record.episode_id,
            authority=AUTHORITY_RECORD_TERMINAL,
        )
    scan = scan_record_executor(record, excluding_session_id=session_id)
    if scan.found:
        return EpisodeObservation(
            row=RECOVERY_ROW_EXECUTED,
            state=state,
            episode_id=record.episode_id,
            authority=AUTHORITY_RECORD,
            executor_session_id=scan.executor_session_id,
        )
    return EpisodeObservation(
        row=RECOVERY_ROW_UNRESOLVABLE,
        state=state,
        episode_id=record.episode_id,
        authority=AUTHORITY_RECORD,
    )


def _is_episode_owner(
    sidecar: PlanEpisodeSidecar, *, session_id: str, owner_domain: LoggingDomain | None
) -> bool:
    """Whether the caller is the claim's actionable owner (DR-34/DR-54)."""
    if sidecar.claiming_session_id != session_id or sidecar.owner_domain is None:
        return False
    if owner_domain is not None and sidecar.owner_domain != owner_domain:
        return False
    return True


def record_from_sidecar(
    record: PlanModeRecord,
    sidecar: PlanEpisodeSidecar,
    *,
    directory: Path,
    state: str,
    snapshot_name: str | None,
) -> PlanModeRecord:
    """The source record's sidecar-derived view — the DR-22 restoration input.

    A C2 crash may leave the record at ``accepted`` while the sidecar still says
    ``accepting``; the sidecar is authoritative, so the record is rewritten from
    it (episode identity, digest, generation, the P1-domain mirror, and the
    surviving snapshot name) rather than trusted.
    """
    scan_domain = sidecar.scan_domain
    return record.model_copy(
        update={
            "episode_id": sidecar.episode_id,
            "sidecar_path": str(sidecar_path(directory)),
            "draft_path": str(sidecar.draft_path),
            "snapshot_path": snapshot_name,
            "state": state,
            "consent_sha256": sidecar.consent_sha256,
            "attempt_generation": sidecar.attempt_generation,
            "scan_domain": (
                None
                if scan_domain is None
                else PlanScanDomain(
                    save_dir=scan_domain.save_dir,
                    session_prefix=scan_domain.session_prefix,
                )
            ),
        }
    )


def episode_directory_for(record: PlanModeRecord) -> Path | None:
    """The episode directory a record names; ``None`` for a legacy record.

    The sidecar's own path is the direct authority; the draft path and (for an
    episode-bound copy) the plan document's directory are the fallbacks a
    partially-written record may hold. A record with no episode identity names
    no directory: it is legacy, and a legacy record has no sidecar to consult.
    """
    if record.sidecar_path:
        return Path(record.sidecar_path).parent
    if record.draft_path:
        return Path(record.draft_path).parent
    if record.episode_id is not None:
        return Path(record.plan_path).parent
    return None


@dataclass(frozen=True, slots=True)
class RestoredEpisode:
    """One restore site's sidecar-validated record and the row it came from."""

    record: PlanModeRecord
    observation: EpisodeObservation


def restore_episode_record(
    record: PlanModeRecord,
    *,
    session_id: str,
    owner_domain: LoggingDomain | None = None,
) -> RestoredEpisode | None:
    """Interpret a record at a RESTORE site, the sidecar consulted first.

    A fork's inherited copy, a rewind fork's record, and a compaction's
    re-injected binding are copies the source session wrote; the sidecar, never
    the copy, decides the lifecycle (DR-15/DR-24/DR-32). A present sidecar
    rebuilds the record from it — state, episode identity, digest, generation,
    the P1-domain mirror, and the surviving snapshot — so an ``accepted`` or
    ``cancelled`` episode restores as its terminal history and never as active
    planning, while an ``accepting`` one restores its accepting status. A
    missing sidecar leaves a TERMINAL record as valid display authority and
    classifies a non-terminal copy through the P1-domain mirror scan, never as a
    fresh draft. An accepted episode's live snapshot is completed by the rename
    rule at this observation, exactly as at every other observation surface.

    ``session_id`` is the session the record was WRITTEN under — the one whose
    own meta carries the P1 mirror — so the bounded scan never mistakes that
    mirror for a launched executor. ``None`` is returned when the record names
    no episode directory (a legacy record). An untrusted read or scan raises
    ``PlanEpisodeError``: the caller reports the failure rather than reading it
    as a clean miss.
    """
    directory = episode_directory_for(record)
    if directory is None:
        return None
    sidecar = read_sidecar(directory)
    if sidecar is None:
        return RestoredEpisode(
            record=record,
            observation=_observe_without_sidecar(record, session_id=session_id),
        )
    observation = _observe_from_sidecar(
        directory, sidecar, record, session_id=session_id, owner_domain=owner_domain
    )
    return RestoredEpisode(
        record=record_from_sidecar(
            record,
            sidecar,
            directory=directory,
            state=sidecar.state,
            snapshot_name=observation.snapshot_name,
        ),
        observation=observation,
    )


def restored_plan_record(
    record: PlanModeRecord,
    *,
    session_id: str,
    owner_domain: LoggingDomain | None = None,
) -> PlanModeRecord:
    """The record a restore site INSTALLS: sidecar-validated, else the copy.

    A fork's inherit and a rewind fork's re-install must never fail on the
    arbiter, so an untrusted read or scan keeps the departing copy: the restore
    surfaces observe the sidecar themselves and display the failure, and every
    transition (accept above all) re-validates the sidecar before it acts.
    """
    try:
        restored = restore_episode_record(
            record, session_id=session_id, owner_domain=owner_domain
        )
    except PlanEpisodeError:
        return record.model_copy(deep=True)
    if restored is None:
        return record.model_copy(deep=True)
    return restored.record.model_copy(deep=True)


def recover_release_claim(
    directory: Path, *, episode_id: str, session_id: str, owner_domain: LoggingDomain
) -> PlanEpisodeSidecar:
    """Row 2: release a crashed attempt that left no (or a partial) snapshot.

    The fence is acquired and ``{episode_id, state, claiming identity}`` are
    freshly re-verified inside it (DR-48). Only a PRE-COMMIT attempt's live-named
    file is unlinked; a ``c1_verified`` attempt's bytes are permanent provenance
    and are renamed instead. The episode returns to draft with ``c1_verified``
    cleared in the same durable write, so the next accept starts from scratch.
    """
    with held_transition_fence(directory, holder_session_id=session_id):
        sidecar = read_sidecar_required(directory)
        _require_recovery_claim(
            sidecar,
            episode_id=episode_id,
            session_id=session_id,
            owner_domain=owner_domain,
        )
        _retire_abandoned_snapshot(directory, sidecar)
        return _durable_transition(directory, sidecar, _draft_view(sidecar))


async def reopen_accepting_episode(
    directory: Path,
    *,
    episode_id: str,
    session_id: str,
    owner_domain: LoggingDomain,
    record: PlanModeRecord,
    write_record: Callable[[PlanModeRecord], Awaitable[None]],
) -> RecoveryOffer:
    """Row 3's re-offer leg: verify the claim, then rebuild the handoff body.

    Inside one held fence the episode is re-verified, the executor predicate is
    re-scanned (a discovered marker refuses the re-offer and moves the surface to
    the possibly-executed row), the surviving bytes are located by EITHER name
    and digest-verified against ``consent_sha256``, and the source record is
    restored from the sidecar. The claim STAYS held: the caller's confirm
    re-enters COMMIT, which acquires the fence itself and re-scans once more
    before the marker write.
    """
    with held_transition_fence(directory, holder_session_id=session_id):
        sidecar = read_sidecar_required(directory)
        _require_recovery_claim(
            sidecar,
            episode_id=episode_id,
            session_id=session_id,
            owner_domain=owner_domain,
        )
        scan = scan_sidecar_executor(sidecar, excluding_session_id=session_id)
        if scan.found:
            raise PlanEpisodePossiblyExecutedError(
                f"{POSSIBLY_EXECUTED_MESSAGE} (executor session "
                f"{scan.executor_session_id})",
                executor_session_id=scan.executor_session_id,
            )
        snapshot_path, content = locate_episode_snapshot(directory, sidecar)
        restored = record_from_sidecar(
            record,
            sidecar,
            directory=directory,
            state=STATE_ACCEPTING,
            snapshot_name=snapshot_path.name,
        )
        try:
            await write_record(restored)
        except Exception as error:
            raise PlanArbiterUnresolvableError(RESTORATION_FAILED_MESSAGE) from error
        return RecoveryOffer(
            sidecar=sidecar, snapshot_path=snapshot_path, content=content
        )


async def recover_cancel_accepting(
    directory: Path,
    *,
    episode_id: str,
    session_id: str,
    owner_domain: LoggingDomain,
    record: PlanModeRecord,
    write_record: Callable[[PlanModeRecord], Awaitable[None]],
) -> PlanEpisodeSidecar:
    """Row 3's cancel leg: the executor predicate is re-scanned FIRST (DR-57).

    A discovered executor marker REFUSES the cancel — the surface moves to the
    possibly-executed row — and only a clean scan proceeds: the record is
    returned to draft, the snapshot is unlinked (PRE-COMMIT) or renamed to the
    attempt's provenance name (post-C1), and the episode returns to draft with
    ``c1_verified`` cleared in the same durable state-write.
    """
    with held_transition_fence(directory, holder_session_id=session_id):
        sidecar = read_sidecar_required(directory)
        _require_recovery_claim(
            sidecar,
            episode_id=episode_id,
            session_id=session_id,
            owner_domain=owner_domain,
        )
        scan = scan_sidecar_executor(sidecar, excluding_session_id=session_id)
        if scan.found:
            raise PlanEpisodePossiblyExecutedError(
                f"{POSSIBLY_EXECUTED_MESSAGE} (executor session "
                f"{scan.executor_session_id})",
                executor_session_id=scan.executor_session_id,
            )
        intended = _draft_view(sidecar)
        restored = record_from_sidecar(
            record, intended, directory=directory, state=STATE_DRAFT, snapshot_name=None
        )
        try:
            await write_record(restored)
        except Exception as error:
            raise PlanArbiterUnresolvableError(RESTORATION_FAILED_MESSAGE) from error
        _retire_abandoned_snapshot(directory, sidecar)
        return _durable_transition(directory, sidecar, intended)


def adjudicate_possibly_executed(
    directory: Path, *, episode_id: str, session_id: str, owner_domain: LoggingDomain
) -> CommittedEpisode:
    """The possibly-executed row's mark-accepted: owner adjudication, then rename.

    The sidecar flips to accepted under the fence with the recorded executor
    session, and the flip-agnostic completion rule runs at this very observation
    (the adjudication is an accepted-flip site too). The flip forecloses the
    episode: its handoff can never be re-offered, re-accepted, or resumed — a
    fresh episode re-binding the surviving snapshot bytes is the remedy.
    """
    with held_transition_fence(directory, holder_session_id=session_id):
        sidecar = read_sidecar_required(directory)
        _require_recovery_claim(
            sidecar,
            episode_id=episode_id,
            session_id=session_id,
            owner_domain=owner_domain,
        )
        scan = scan_sidecar_executor(sidecar, excluding_session_id=session_id)
        if not scan.found:
            raise PlanEpisodeStateError(
                f"no executor session records plan episode {sidecar.episode_id} — "
                "the re-offer or cancel legs govern this attempt"
            )
        accepted = replace(
            sidecar,
            state=STATE_ACCEPTED,
            accepted_sha256=sidecar.consent_sha256,
            executor_session_id=scan.executor_session_id,
            c1_verified=True,
        )
        published = _durable_transition(directory, sidecar, accepted)
        outcome = complete_accepted_rename(directory, published)
        return CommittedEpisode(sidecar=published, rename_outcome=outcome)


def _require_recovery_claim(
    sidecar: PlanEpisodeSidecar,
    *,
    episode_id: str,
    session_id: str,
    owner_domain: LoggingDomain,
) -> None:
    """Every recovery mutation's fresh, fenced re-verification (DR-48)."""
    if sidecar.episode_id != episode_id:
        raise PlanEpisodeStateError(
            f"the plan episode {sidecar.episode_id} does not match {episode_id}"
        )
    if sidecar.state != STATE_ACCEPTING:
        raise PlanEpisodeStateError(
            f"plan episode {sidecar.episode_id} holds no accept claim "
            f"(state {sidecar.state})"
        )
    require_episode_owner(sidecar, session_id=session_id, owner_domain=owner_domain)


def _draft_view(sidecar: PlanEpisodeSidecar) -> PlanEpisodeSidecar:
    """The return-to-draft state: ``c1_verified`` cleared in the SAME write.

    A draft episode has no attempt in flight and no executor: the recorded
    executor id goes with the claim, so a later accept's scan can never inherit
    an earlier attempt's identity.
    """
    return replace(
        sidecar,
        state=STATE_DRAFT,
        claiming_session_id=None,
        c1_verified=False,
        consent_sha256=None,
        executor_session_id=None,
    )


def _retire_abandoned_snapshot(directory: Path, sidecar: PlanEpisodeSidecar) -> None:
    """Settle a cancelled attempt's bytes: PRE-COMMIT unlink, post-C1 rename.

    ``c1_verified`` is the attempt-scoped discriminator — never an inference. A
    PRE-COMMIT attempt's live name is never provenance, so it is unlinked; a
    post-C1 attempt's bytes are permanent provenance and are renamed to the
    attempt's provenance name (a surviving provenance name — the cancel-window
    state — needs nothing).
    """
    if sidecar.c1_verified:
        if accepted_snapshot_path(directory, sidecar.episode_id).exists():
            rename_accepted_snapshot(
                directory,
                episode_id=sidecar.episode_id,
                generation=sidecar.attempt_generation,
            )
        return
    unlink_accepted_snapshot(directory, sidecar.episode_id)


# --- Cleanup ----------------------------------------------------------------


def cleanup_episode_workspace(
    directory: Path,
    *,
    marker_name: str,
    holder_session_id: str = "plan-cleanup",
    retry_timeout: float = 0.0,
) -> bool:
    """Remove the plan-owned entries of a marker-claimed episode directory.

    The removal whitelist is the marker, the sidecar, PRE-COMMIT snapshots, and
    ``.chartreux-plan-tmp-*`` temporaries. The transition fence is NEVER removed
    (a leftover is harmless; unlinking a held flock file would silently break
    exclusion), and neither is any provenance-named or attempt-scoped file: a
    snapshot whose claim is post-commit is permanent provenance. Any other entry
    keeps the directory — nothing of uncertain ownership is deleted.

    Returns ``True`` only when the directory itself went away. Because the fence
    file is retained, an ever-allocated episode directory is never
    rmdir-eligible; the declared accumulation rule bounds that growth.
    """
    marker = directory / marker_name
    if marker.is_symlink() or not marker.is_file():
        return False
    sidecar = read_sidecar(directory)
    if sidecar is None:
        return _remove_owned_entries(directory, marker, sidecar=None)
    with held_transition_fence(
        directory, holder_session_id=holder_session_id, retry_timeout=retry_timeout
    ):
        current = read_sidecar_required(directory)
        if current.state == STATE_ACCEPTING:
            raise PlanEpisodeStateError(
                f"plan episode {current.episode_id} is accepting — a transition "
                "is in progress; retry after it completes"
            )
        return _remove_owned_entries(directory, marker, sidecar=current)


def _remove_owned_entries(
    directory: Path, marker: Path, *, sidecar: PlanEpisodeSidecar | None
) -> bool:
    if sidecar is not None:
        if sidecar.state == STATE_ACCEPTED:
            # The completion rule runs at this observation too: a pending rename
            # is completed before the whitelist is judged, and what remains
            # under either name is permanent provenance — never removed.
            complete_accepted_rename(directory, sidecar)
        elif not _snapshot_is_provenance(sidecar):
            unlink_accepted_snapshot(directory, sidecar.episode_id)
    _remove_plan_temporaries(directory)
    _unlink_quietly(sidecar_path(directory))
    _unlink_quietly(marker)
    fsync_directory(directory)
    try:
        directory.rmdir()
    except OSError:
        return False
    return True


def _snapshot_is_provenance(sidecar: PlanEpisodeSidecar) -> bool:
    """Decide whether the episode's snapshot is permanent provenance.

    A post-commit claim, or a recorded scan domain holding an executor marker,
    makes the bytes permanent. A scan that cannot be trusted keeps the snapshot:
    cleanup never deletes what it cannot classify.
    """
    if sidecar.c1_verified:
        return True
    if sidecar.scan_domain is None:
        return False
    try:
        return scan_episode_executor(sidecar.scan_domain, sidecar.episode_id).found
    except PlanEpisodeScanError:
        return True


def _remove_plan_temporaries(directory: Path) -> None:
    try:
        entries = list(directory.iterdir())
    except OSError as error:
        raise PlanTransitionError(
            f"cannot list the plan workspace {directory}: {error}"
        ) from error
    for entry in entries:
        if is_plan_temp_name(entry.name):
            _unlink_quietly(entry)


def _unlink_quietly(path: Path) -> None:
    with suppress(FileNotFoundError):
        path.unlink()


__all__ = [
    "AUTHORITY_RECORD",
    "AUTHORITY_RECORD_TERMINAL",
    "AUTHORITY_SIDECAR",
    "DOMAIN_REFUSAL",
    "EPISODE_FENCE_FILENAME",
    "EPISODE_SIDECAR_FILENAME",
    "EPISODE_SIDECAR_KIND",
    "EPISODE_SIDECAR_VERSION",
    "EXECUTOR_EXISTS_MESSAGE",
    "LOGGING_REQUIRED_MESSAGE",
    "PLAN_TEMP_PREFIX",
    "POSSIBLY_EXECUTED_MESSAGE",
    "RECOVERY_ROW_ACCEPTED",
    "RECOVERY_ROW_ACCEPTING_NO_SNAPSHOT",
    "RECOVERY_ROW_ACCEPTING_SNAPSHOT",
    "RECOVERY_ROW_CANCELLED",
    "RECOVERY_ROW_DRAFT",
    "RECOVERY_ROW_EXECUTED",
    "RECOVERY_ROW_LEGACY",
    "RECOVERY_ROW_POSSIBLY_EXECUTED",
    "RECOVERY_ROW_UNRESOLVABLE",
    "REPREVIEW_REQUIRED_MESSAGE",
    "RESTORATION_FAILED_MESSAGE",
    "SEED_DOCUMENT_BEGIN",
    "SEED_DOCUMENT_END",
    "SNAPSHOT_COLLISION_MESSAGE",
    "SNAPSHOT_RENAMED",
    "SNAPSHOT_RENAME_NOOP",
    "SNAPSHOT_RENAME_PENDING",
    "STATES",
    "STATE_ACCEPTED",
    "STATE_ACCEPTING",
    "STATE_CANCELLED",
    "STATE_DRAFT",
    "TERMINAL_STATES",
    "UNRESOLVABLE_ARBITER_MESSAGE",
    "CommitAttempt",
    "CommittedEpisode",
    "EpisodeObservation",
    "ExecutorScan",
    "LoggingDomain",
    "LoggingObservation",
    "PlanArbiterUnresolvableError",
    "PlanDigestMismatchError",
    "PlanDocumentChangedError",
    "PlanDraftReadError",
    "PlanEpisodeBindError",
    "PlanEpisodeError",
    "PlanEpisodeOwnerError",
    "PlanEpisodePossiblyExecutedError",
    "PlanEpisodeScanError",
    "PlanEpisodeStateError",
    "PlanEpisodeTerminalError",
    "PlanEpisodeWriteError",
    "PlanFenceBusyError",
    "PlanLoggingDisabledError",
    "PlanSidecarError",
    "PlanSnapshotCollisionError",
    "PlanTransitionError",
    "PreparedAccept",
    "RecoveryOffer",
    "RestoredEpisode",
    "TransitionFence",
    "abort_accept",
    "accepted_snapshot_path",
    "adjudicate_possibly_executed",
    "begin_commit",
    "build_handoff_context",
    "cancel_episode",
    "canonicalize_authorized_path",
    "claim_accept",
    "cleanup_episode_workspace",
    "clear_during_claim",
    "complete_accepted_rename",
    "confirm_draft_unchanged",
    "create_sidecar",
    "durable_replace_plan",
    "encode_sidecar",
    "ensure_episode_fence",
    "episode_directory_for",
    "fence_path",
    "guard_plan_entry",
    "guard_plan_record_resume",
    "held_transition_fence",
    "is_plan_temp_name",
    "locate_episode_snapshot",
    "new_episode_id",
    "observe_episode",
    "observe_logging_domain",
    "open_bound_document",
    "prepare_accept",
    "provenance_snapshot_path",
    "publish_consent_snapshot",
    "publish_episode_snapshot",
    "read_draft_bytes",
    "read_sidecar",
    "read_sidecar_required",
    "rebind_draft",
    "record_from_sidecar",
    "recover_cancel_accepting",
    "recover_release_claim",
    "release_claim",
    "reopen_accepting_episode",
    "require_episode_owner",
    "require_session_logging",
    "restore_episode_record",
    "restored_plan_record",
    "scan_episode_executor",
    "scan_record_executor",
    "scan_sidecar_executor",
    "sha256_bytes",
    "sidecar_from_payload",
    "sidecar_path",
    "surviving_snapshot_name",
    "unlink_accepted_snapshot",
    "write_sidecar",
]
