from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from chartreux.app_server._subagent_records import (
    AgentRecord,
    ResidentTranscriptReadSnapshot,
    StoredChildSession,
    TranscriptReadSnapshot,
    _AgentState,
    _AgentTombstone,
)
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.session_types import ChildSessionLink
from chartreux.core.subagents import UnknownAgentError

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
            return ResidentTranscriptReadSnapshot(
                agent_id=agent_id,
                child_session_id=record.session_id,
                parent_identity=identity,
                parent_runtime_token=id(root),
                record_token=id(record),
                runtime_token=id(record.runtime),
                messages=list(record.runtime.agent_loop.messages),
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
