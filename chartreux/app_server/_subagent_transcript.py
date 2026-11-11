from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING, Any, Protocol

from chartreux.app_server._agent_transcript import archive_transcript_source
from chartreux.app_server._subagent_records import (
    AgentRecord,
    ResidentTranscriptReadSnapshot,
    StoredChildSession,
    TranscriptMessageView,
    TranscriptReadSnapshot,
    _AgentState,
    _AgentTombstone,
)
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.config.models import SessionLoggingConfig
from chartreux.core.session_types import ChildSessionLink
from chartreux.core.subagents import UnknownAgentError
from chartreux.utils.session_id import shorten_session_id

if TYPE_CHECKING:
    from chartreux.app_server._sessions import SessionRuntime


class TranscriptReadsView(Protocol):
    """Narrow typed view over the registry-owned transcript-read state.

    Dynamic access: every member is read through the view at call time, so
    replaceable scalars (root identity, generation) and tombstones are never
    snapshotted at construction — registry state mutated after the
    collaborator is built stays visible.
    """

    _root: SessionRuntime | None
    _children: dict[str, SessionRuntime]
    _child_links: dict[str, tuple[SessionRuntime, str]]
    _stored_children: dict[str, StoredChildSession]
    _readable_children: dict[str, SessionRuntime]
    _agent_records: dict[str, AgentRecord]
    _evicted_agents: dict[str, _AgentTombstone]
    _generation_identity: tuple[str, int] | None
    _pending_child_closes: dict[str, tuple[SessionRuntime, asyncio.Task[None]]]
    _detached_child_ids: set[str]
    _registry_lock: asyncio.Lock
    _clock: Callable[[], float]

    def _live_tombstone_locked(
        self, agent_id: str, now: float
    ) -> _AgentTombstone | None: ...


class TranscriptReads:
    """Authorized transcript reads for the registered child sessions.

    Narrow collaborator of the session runtime registry: all state is
    reached through the registry-owned typed view with dynamic access, the
    two-phase capture/validate authorization runs under the registry's own
    lock with the caller's await between the phases, and the lock-free
    synchronous methods stay lock-free. No state moves, no private locks,
    no new tasks or relay layers.
    """

    def __init__(self, view: TranscriptReadsView) -> None:
        self._view = view

    def resolve_child(
        self, session_id: str
    ) -> SessionRuntime | StoredChildSession | None:
        """Find a live child or a stable snapshot of its stored link."""
        if child := self._view._children.get(session_id):
            return child
        if child := self._view._readable_children.get(session_id):
            return child
        if stored := self._view._stored_children.get(session_id):
            return stored
        resolved = self._resolve_child_link(session_id)
        if resolved is None:
            return None
        parent_runtime, root_agent_loop, child_dir, link = resolved
        return StoredChildSession(
            parent_runtime=parent_runtime,
            root_session_id=root_agent_loop.session_id,
            root_generation=root_agent_loop._session_generation,
            child_dir=child_dir,
            link=link,
        )

    def stored_child_is_current(self, stored: StoredChildSession) -> bool:
        """Check that a stored read did not race root or child-link changes."""
        root = self._view._root
        if self._view._stored_children.get(stored.link.session_id) is stored:
            return (
                root is stored.parent_runtime
                and root is not None
                and root.agent_loop.session_id == stored.root_session_id
                and root.agent_loop._session_generation == stored.root_generation
                and stored.link.session_id not in self._view._children
            )
        if (
            root is not stored.parent_runtime
            or root is None
            or root.agent_loop.session_id != stored.root_session_id
            or root.agent_loop._session_generation != stored.root_generation
            or stored.link.session_id in self._view._children
        ):
            return False
        resolved = self._resolve_child_link(stored.link.session_id)
        return (
            resolved is not None
            and resolved[0] is stored.parent_runtime
            and resolved[2] == stored.child_dir
            and resolved[3] == stored.link
        )

    @staticmethod
    def _transcript_link_identity(
        link: ChildSessionLink | None,
    ) -> tuple[str, str, str, str | None] | None:
        if link is None:
            return None
        return (link.session_id, link.tool_call_id, link.agent, link.relative_path)

    @staticmethod
    def _transcript_link_for(
        root: SessionRuntime, child_session_id: str
    ) -> ChildSessionLink | None:
        metadata = root.agent_loop.session_logger.session_metadata
        if metadata is None:
            return None
        return next(
            (
                link
                for link in metadata.child_sessions
                if link.session_id == child_session_id
            ),
            None,
        )

    async def resolve_resident_transcript_read(
        self, agent_id: str
    ) -> ResidentTranscriptReadSnapshot | None:
        """Snapshot a running resident child's messages under its MessageList lock."""
        async with self._view._registry_lock:
            root = self._view._root
            if root is None:
                raise UnknownAgentError(f"Unknown agent: {agent_id}")
            identity = (root.agent_loop.session_id, root.agent_loop._session_generation)
            if self._view._generation_identity != identity:
                raise UnknownAgentError(f"Unknown agent: {agent_id}")
            record = self._view._agent_records.get(agent_id)
            if record is None:
                # Expired terminal metadata behaves as absent for transcript reads.
                if (
                    self._view._live_tombstone_locked(agent_id, self._view._clock())
                    is None
                ):
                    raise UnknownAgentError(f"Unknown agent: {agent_id}")
                return None
            if (
                record.root_generation != identity[1]
                or record.parent_session_id != identity[0]
                or record.state not in {_AgentState.RUNNING, _AgentState.FINALIZING}
                or self._view._children.get(record.session_id) is not record.runtime
            ):
                return None
            child_loop = record.runtime.agent_loop
            return ResidentTranscriptReadSnapshot(
                agent_id=agent_id,
                child_session_id=record.session_id,
                parent_identity=identity,
                parent_runtime_token=id(root),
                record_token=id(record),
                runtime_token=id(record.runtime),
                # The view carries the sealed-prefix source when the child's
                # list owns archive state, so paging reads the persisted
                # originals instead of the payload-trimmed RAM copies.
                messages=TranscriptMessageView(
                    list(child_loop.messages), archive_transcript_source(child_loop)
                ),
            )

    async def resident_transcript_read_is_current(
        self, snapshot: ResidentTranscriptReadSnapshot
    ) -> bool:
        """Check that a resident transcript snapshot remains authorized."""
        async with self._view._registry_lock:
            root = self._view._root
            if (
                root is None
                or id(root) != snapshot.parent_runtime_token
                or (root.agent_loop.session_id, root.agent_loop._session_generation)
                != snapshot.parent_identity
                or self._view._generation_identity != snapshot.parent_identity
            ):
                return False
            record = self._view._agent_records.get(snapshot.agent_id)
            return (
                record is not None
                and id(record) == snapshot.record_token
                and record.session_id == snapshot.child_session_id
                and record.parent_session_id == snapshot.parent_identity[0]
                and record.root_generation == snapshot.parent_identity[1]
                and record.state in {_AgentState.RUNNING, _AgentState.FINALIZING}
                and id(record.runtime) == snapshot.runtime_token
                and self._view._children.get(record.session_id) is record.runtime
            )

    async def resolve_transcript_read(self, agent_id: str) -> TranscriptReadSnapshot:
        """Capture parent-authorized disk-read inputs without touching child runtime state."""
        async with self._view._registry_lock:
            root = self._view._root
            if root is None:
                raise UnknownAgentError(f"Unknown agent: {agent_id}")
            identity = (root.agent_loop.session_id, root.agent_loop._session_generation)
            if self._view._generation_identity != identity:
                raise UnknownAgentError(f"Unknown agent: {agent_id}")
            record = self._view._agent_records.get(agent_id)
            tombstone = self._view._live_tombstone_locked(agent_id, self._view._clock())
            if record is not None:
                if (
                    record.root_generation != identity[1]
                    or record.parent_session_id != identity[0]
                ):
                    raise UnknownAgentError(f"Unknown agent: {agent_id}")
                child_session_id = record.session_id
            elif tombstone is not None:
                if tombstone.parent_identity != identity:
                    raise UnknownAgentError(f"Unknown agent: {agent_id}")
                child_session_id = tombstone.child_session_id
            else:
                raise UnknownAgentError(f"Unknown agent: {agent_id}")

            link = self._transcript_link_for(root, child_session_id)
            link_identity = self._transcript_link_identity(link)
            parent_dir = root.agent_loop.session_logger.session_dir
            if link is None or link.relative_path is None or parent_dir is None:
                return TranscriptReadSnapshot(
                    agent_id=agent_id,
                    child_session_id=child_session_id,
                    child_dir=None,
                    parent_dir=None,
                    parent_identity=identity,
                    parent_runtime_token=id(root),
                    link_identity=link_identity,
                )
            relative_path = Path(link.relative_path)
            if relative_path.is_absolute() or any(
                part in {"", ".", ".."} for part in relative_path.parts
            ):
                raise UnknownAgentError(f"Unknown agent: {agent_id}")
            child_dir = parent_dir / relative_path
            if not child_dir.is_relative_to(parent_dir):
                raise UnknownAgentError(f"Unknown agent: {agent_id}")
            return TranscriptReadSnapshot(
                agent_id=agent_id,
                child_session_id=child_session_id,
                child_dir=child_dir,
                parent_dir=parent_dir,
                parent_identity=identity,
                parent_runtime_token=id(root),
                link_identity=link_identity,
            )

    async def transcript_read_is_current(
        self, snapshot: TranscriptReadSnapshot
    ) -> bool:
        """Check whether an authorized transcript read remains bound to its parent link."""
        async with self._view._registry_lock:
            root = self._view._root
            if (
                root is None
                or id(root) != snapshot.parent_runtime_token
                or (root.agent_loop.session_id, root.agent_loop._session_generation)
                != snapshot.parent_identity
                or self._view._generation_identity != snapshot.parent_identity
            ):
                return False
            record = self._view._agent_records.get(snapshot.agent_id)
            tombstone = self._view._live_tombstone_locked(
                snapshot.agent_id, self._view._clock()
            )
            authorized = (
                record is not None
                and record.session_id == snapshot.child_session_id
                and record.parent_session_id == snapshot.parent_identity[0]
                and record.root_generation == snapshot.parent_identity[1]
            ) or (
                tombstone is not None
                and tombstone.child_session_id == snapshot.child_session_id
                and tombstone.parent_identity == snapshot.parent_identity
            )
            return (
                authorized
                and self._transcript_link_identity(
                    self._transcript_link_for(root, snapshot.child_session_id)
                )
                == snapshot.link_identity
            )

    def references_child(self, session_id: str) -> bool:
        return (
            session_id in self._view._children
            or session_id in self._view._pending_child_closes
            or session_id in self._view._detached_child_ids
            or self._resolve_child_link(session_id) is not None
        )

    def _resolve_child_link(
        self, session_id: str
    ) -> tuple[SessionRuntime, AgentLoop, Path, ChildSessionLink] | None:
        parent_runtime = self._view._root
        if parent_runtime is None:
            return None
        root_agent_loop = parent_runtime.agent_loop
        metadata = root_agent_loop.session_logger.session_metadata
        parent_dir = root_agent_loop.session_logger.session_dir
        if metadata is None or parent_dir is None:
            return None
        parent_root = parent_dir.resolve()
        link = next(
            (
                link
                for link in metadata.child_sessions
                if link.session_id == session_id and link.relative_path is not None
            ),
            None,
        )
        if link is None or link.relative_path is None:
            return None
        child_dir = (parent_dir / link.relative_path).resolve()
        if not child_dir.is_relative_to(parent_root) or not child_dir.is_dir():
            return None
        return parent_runtime, root_agent_loop, child_dir, link


# --- WP5 [5C]: child-owned transcripts, durable markers, reconciliation ------

CHILD_TRANSCRIPTS_DIRNAME = "agents"
"""The subdirectory of a root session dir that holds every child transcript."""

CHILD_START_MARKER_FILENAME = "child-start.json"
"""The durable start marker the child writes before it executes a run."""

CHILD_RESULT_FILENAME = "child-result.json"
"""The durable terminal result record the child writes save-before-emit."""

CHILD_MARKER_VERSION = 1
"""Schema version of the marker and result records."""

ROOT_SESSION_DIR_PAYLOAD_KEY = "root_session_dir"
"""Spawn-payload key carrying the root session dir the child transcript nests in."""

AGENT_NAME_PAYLOAD_KEY = "agent_name"
"""Spawn-payload key carrying the child's transcript folder prefix."""


class ChildTranscriptError(RuntimeError):
    """Base class for child-transcript durability and reconciliation failures."""


class ChildTranscriptPathError(ChildTranscriptError):
    """A child transcript path escaped the root session's ``agents/`` subtree."""


class ChildTranscriptOutcome(StrEnum):
    """How far a child got before its process died (design.md:196-205)."""

    NEVER_EXECUTED = "never_executed"
    STARTED_OUTCOME_UNKNOWN = "started_outcome_unknown"
    COMPLETED = "completed"
    UNREADABLE = "unreadable"


@dataclass(frozen=True, slots=True)
class ChildStartMarker:
    """The durable record written BEFORE the child executes a run.

    A start marker without a matching terminal result is "started, outcome
    unknown" after a crash: the crash may have landed before or during
    execution, so the run is never replayed.
    """

    agent_id: str
    session_id: str
    run_id: str
    started_at: str
    version: int = CHILD_MARKER_VERSION

    def to_payload(self) -> dict[str, object]:
        return {
            "version": self.version,
            "kind": "child-start",
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "run_id": self.run_id,
            "started_at": self.started_at,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> ChildStartMarker:
        """Rebuild one start marker, refusing a malformed record."""
        if payload.get("kind") != "child-start":
            raise ValueError("not a child start marker")
        return cls(
            agent_id=_required_marker_str(payload, "agent_id"),
            session_id=_required_marker_str(payload, "session_id"),
            run_id=_required_marker_str(payload, "run_id"),
            started_at=_required_marker_str(payload, "started_at"),
            version=_marker_version(payload),
        )


@dataclass(frozen=True, slots=True)
class ChildResultRecord:
    """The durable terminal result record: result id plus content digest.

    Written save-before-emit, so a completed child always leaves the record
    before its completion is announced. The coordinator hands an unconsumed
    record to the root by result id; the digest lets the root verify the
    record it adopts.
    """

    agent_id: str
    session_id: str
    run_id: str
    result_id: str
    digest: str
    outcome: str
    completed_at: str
    error: str | None = None
    version: int = CHILD_MARKER_VERSION

    def to_payload(self) -> dict[str, object]:
        return {
            "version": self.version,
            "kind": "child-result",
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "run_id": self.run_id,
            "result_id": self.result_id,
            "digest": self.digest,
            "outcome": self.outcome,
            "completed_at": self.completed_at,
            "error": self.error,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> ChildResultRecord:
        """Rebuild one terminal result record, refusing a malformed record."""
        if payload.get("kind") != "child-result":
            raise ValueError("not a child result record")
        error = payload.get("error")
        return cls(
            agent_id=_required_marker_str(payload, "agent_id"),
            session_id=_required_marker_str(payload, "session_id"),
            run_id=_required_marker_str(payload, "run_id"),
            result_id=_required_marker_str(payload, "result_id"),
            digest=_required_marker_str(payload, "digest"),
            outcome=_required_marker_str(payload, "outcome"),
            completed_at=_required_marker_str(payload, "completed_at"),
            error=error if isinstance(error, str) else None,
            version=_marker_version(payload),
        )


def child_result_digest(
    *, agent_id: str, session_id: str, run_id: str, outcome: str, error: str | None
) -> str:
    """The content digest of one child's terminal result."""
    payload = json.dumps(
        {
            "agent_id": agent_id,
            "session_id": session_id,
            "run_id": run_id,
            "outcome": outcome,
            "error": error,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


@dataclass(frozen=True, slots=True)
class ChildTranscriptAdjudication:
    """The marker-based state of one child transcript after its process died.

    ``NEVER_EXECUTED`` (no start marker) is distinct from
    ``STARTED_OUTCOME_UNKNOWN`` (start marker, no matching terminal result).
    Neither is ever a replay instruction: an ambiguous or unknown outcome is
    surfaced to the user, never re-executed (design.md:196-205).
    """

    child_dir: Path
    outcome: ChildTranscriptOutcome
    start_marker: ChildStartMarker | None = None
    result: ChildResultRecord | None = None
    detail: str = ""

    @property
    def result_id(self) -> str | None:
        return self.result.result_id if self.result is not None else None

    @property
    def digest(self) -> str | None:
        return self.result.digest if self.result is not None else None

    @property
    def run_id(self) -> str | None:
        if self.result is not None:
            return self.result.run_id
        if self.start_marker is not None:
            return self.start_marker.run_id
        return None


def child_transcript_logging(
    root_session_dir: Path, agent_name: str
) -> SessionLoggingConfig:
    """The child's own logging config: a fresh subdirectory of ``<root>/agents``.

    The child's SessionLogger creates ``<root>/agents/<agent_name>_<ts>_<id>``
    exactly as the in-process child path does ([2C] layout preserved), so the
    child process owns the only writer of its transcript.
    """
    if not agent_name:
        raise ValueError("a child transcript needs an agent name")
    return SessionLoggingConfig(
        enabled=True,
        save_dir=str(Path(root_session_dir) / CHILD_TRANSCRIPTS_DIRNAME),
        session_prefix=agent_name,
    )


def child_transcript_dir_name(
    agent_name: str, session_id: str, *, timestamp: str | None = None
) -> str:
    """The child transcript folder name: the [2C] ``<agent>_<ts>_<id>`` layout."""
    if not agent_name:
        raise ValueError("a child transcript needs an agent name")
    stamp = timestamp or datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return f"{agent_name}_{stamp}_{shorten_session_id(session_id)}"


def child_transcripts_root(root_session_dir: Path) -> Path:
    """The root session's ``agents/`` directory."""
    return Path(root_session_dir) / CHILD_TRANSCRIPTS_DIRNAME


def resolve_child_root_session_dir(session_dir: Path) -> Path | None:
    """The root session dir a child transcript nests in, when it nests in one."""
    parent = Path(session_dir).parent
    return parent.parent if parent.name == CHILD_TRANSCRIPTS_DIRNAME else None


def root_session_dir_from_spec(payload: dict[str, Any]) -> Path | None:
    """The root session dir a spawn spec carries, when it carries one."""
    value = payload.get(ROOT_SESSION_DIR_PAYLOAD_KEY)
    return Path(value) if isinstance(value, str) and value else None


def child_transcript_root_for(
    session_dir: Path | None, root_session_dir: Path | None
) -> Path | None:
    """Resolve the root session dir for a child transcript."""
    if root_session_dir is not None:
        return Path(root_session_dir)
    return None if session_dir is None else resolve_child_root_session_dir(session_dir)


def child_transcript_dirs(root_session_dir: Path) -> list[Path]:
    """Every child transcript dir under ``<root>/agents``, sorted, read-only."""
    root = child_transcripts_root(root_session_dir)
    try:
        entries = list(root.iterdir())
    except OSError:
        return []
    return sorted(
        entry for entry in entries if entry.is_dir() and not entry.name.startswith(".")
    )


def _required_marker_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"child transcript record carries no {key}")
    return value


def _marker_version(payload: dict[str, Any]) -> int:
    value = payload.get("version")
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("child transcript record carries no version")
    if value != CHILD_MARKER_VERSION:
        raise ValueError(f"unsupported child transcript record version {value}")
    return value


class ChildTranscriptStore:
    """The child-owned durable marker/result store for one transcript dir.

    The child process writes both records; the coordinator only ever reads
    them (read-only, path-confined). Every write is atomic — a same-directory
    temporary file, fsynced, then ``os.replace`` — so a crash leaves either
    the previous complete record or the new complete record, never a torn
    one. ``root_session_dir`` confines the target to the root session's
    ``agents/`` subtree.
    """

    def __init__(self, child_dir: Path, *, root_session_dir: Path) -> None:
        self.root_session_dir = Path(root_session_dir).resolve()
        self.child_dir = _confined_child_dir(child_dir, self.root_session_dir)
        self.start_marker_path = self.child_dir / CHILD_START_MARKER_FILENAME
        self.result_path = self.child_dir / CHILD_RESULT_FILENAME

    def record_start(self, marker: ChildStartMarker) -> None:
        """Durably record the start marker before the run executes."""
        self._ensure_dir()
        _atomic_write_json(self.start_marker_path, marker.to_payload())

    def record_result(self, result: ChildResultRecord) -> None:
        """Durably record the terminal result save-before-emit."""
        self._ensure_dir()
        _atomic_write_json(self.result_path, result.to_payload())

    def adjudicate(self) -> ChildTranscriptAdjudication:
        """Classify this child transcript from its markers, never replaying."""
        start_payload, start_error = _read_marker_payload(self.start_marker_path)
        result_payload, result_error = _read_marker_payload(self.result_path)
        errors = [error for error in (start_error, result_error) if error is not None]
        if errors:
            return ChildTranscriptAdjudication(
                child_dir=self.child_dir,
                outcome=ChildTranscriptOutcome.UNREADABLE,
                detail="; ".join(errors),
            )
        start = _parse_marker(ChildStartMarker.from_payload, start_payload)
        if isinstance(start, str):
            return _unreadable(self.child_dir, self.start_marker_path, start)
        result = _parse_marker(ChildResultRecord.from_payload, result_payload)
        if isinstance(result, str):
            return _unreadable(self.child_dir, self.result_path, result)
        if start is None and result is None:
            return ChildTranscriptAdjudication(
                child_dir=self.child_dir, outcome=ChildTranscriptOutcome.NEVER_EXECUTED
            )
        if result is not None and (start is None or start.run_id == result.run_id):
            return ChildTranscriptAdjudication(
                child_dir=self.child_dir,
                outcome=ChildTranscriptOutcome.COMPLETED,
                start_marker=start,
                result=result,
            )
        return ChildTranscriptAdjudication(
            child_dir=self.child_dir,
            outcome=ChildTranscriptOutcome.STARTED_OUTCOME_UNKNOWN,
            start_marker=start,
            result=result,
        )

    def _ensure_dir(self) -> None:
        self.child_dir.mkdir(parents=True, mode=0o700, exist_ok=True)


def _confined_child_dir(child_dir: Path, root_session_dir: Path) -> Path:
    agents_root = child_transcripts_root(root_session_dir).resolve()
    resolved = Path(child_dir).resolve()
    if resolved == agents_root or agents_root not in resolved.parents:
        raise ChildTranscriptPathError(
            f"child transcript {resolved} is outside {agents_root}"
        )
    return resolved


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(payload, temporary, sort_keys=True)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _read_marker_payload(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None, None
    except OSError as exc:
        return None, f"{path.name}: {type(exc).__name__}: {exc}"
    try:
        payload = json.loads(raw)
    except ValueError as exc:
        return None, f"{path.name}: unreadable record ({exc})"
    if not isinstance(payload, dict):
        return None, f"{path.name}: unreadable record (not an object)"
    return payload, None


def _parse_marker[T](
    factory: Callable[[dict[str, Any]], T], payload: dict[str, Any] | None
) -> T | str | None:
    if payload is None:
        return None
    try:
        return factory(payload)
    except ValueError as exc:
        return str(exc)


def _unreadable(
    child_dir: Path, path: Path, detail: str
) -> ChildTranscriptAdjudication:
    return ChildTranscriptAdjudication(
        child_dir=child_dir,
        outcome=ChildTranscriptOutcome.UNREADABLE,
        detail=f"{path.name}: {detail}",
    )


def read_child_transcript(
    child_dir: Path, *, root_session_dir: Path
) -> ChildTranscriptAdjudication:
    """Read one child transcript's markers without writing anything."""
    return ChildTranscriptStore(
        child_dir, root_session_dir=root_session_dir
    ).adjudicate()


__all__ = [
    "AGENT_NAME_PAYLOAD_KEY",
    "CHILD_MARKER_VERSION",
    "CHILD_RESULT_FILENAME",
    "CHILD_START_MARKER_FILENAME",
    "CHILD_TRANSCRIPTS_DIRNAME",
    "ROOT_SESSION_DIR_PAYLOAD_KEY",
    "ChildResultRecord",
    "ChildStartMarker",
    "ChildTranscriptAdjudication",
    "ChildTranscriptError",
    "ChildTranscriptOutcome",
    "ChildTranscriptPathError",
    "ChildTranscriptStore",
    "TranscriptReads",
    "TranscriptReadsView",
    "child_result_digest",
    "child_transcript_dir_name",
    "child_transcript_dirs",
    "child_transcript_logging",
    "child_transcript_root_for",
    "child_transcripts_root",
    "read_child_transcript",
    "resolve_child_root_session_dir",
    "root_session_dir_from_spec",
]
