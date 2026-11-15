from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING, Any, Protocol
from uuid import UUID

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
from chartreux.core.config._dangerous_command_authority import (
    ProjectDangerousCommandGrant,
    ProjectRemoteExecGrant,
)
from chartreux.core.config._restrictions import (
    PolicySourceIdentity,
    SourceRestrictions,
    ToolRestriction,
)
from chartreux.core.config._root_authority import ProjectRootAuthority
from chartreux.core.config.models import SessionLoggingConfig
from chartreux.core.dangerous_commands import RemoteDestination
from chartreux.core.session.session_permissions import METADATA_FILENAME
from chartreux.core.session_types import ChildSessionLink
from chartreux.core.subagents import UnknownAgentError
from chartreux.core.workspace import Workspace
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

CHILD_TOMBSTONE_FILENAME = "child-tombstone.json"
"""The durable retraction record that retires a child publication attempt."""

CHILD_MARKER_VERSION = 1
"""Schema version of the marker and result records."""

ROOT_SESSION_DIR_PAYLOAD_KEY = "root_session_dir"
"""Spawn-payload key carrying the root session dir the child transcript nests in."""

AGENT_NAME_PAYLOAD_KEY = "agent_name"
"""Spawn-payload key carrying the child's transcript folder prefix."""

CLIENT_INFO_PAYLOAD_KEY = "client_info"
"""Spawn-payload key carrying the validated client identity the child re-opens with."""

CLIENT_CAPABILITIES_PAYLOAD_KEY = "client_capabilities"
"""Spawn-payload key carrying the client capabilities the child re-opens with."""

INHERITED_WORKSPACE_PAYLOAD_KEY = "inherited_workspace"
"""Spawn-payload key carrying the parent workspace a child loop must inherit."""

AGENT_CONFIG_PAYLOAD_KEY = "agent_config"
"""Spawn-payload key carrying the parent's live session options the child opens with."""

INHERITED_RESTRICTIONS_PAYLOAD_KEY = "inherited_restrictions"
"""Spawn-payload key carrying the accepted source restrictions a child inherits."""

INHERITED_MODE_RESTRICTIONS_PAYLOAD_KEY = "inherited_mode_restrictions"
"""Spawn-payload key carrying the parent's mode restrictions a child inherits."""

INHERITED_ROOT_GRANTS_PAYLOAD_KEY = "inherited_root_grants"
"""Spawn-payload key carrying the bare roots the parent was granted at dispatch."""


class ChildTranscriptError(RuntimeError):
    """Base class for child-transcript durability and reconciliation failures."""


class ChildTranscriptPathError(ChildTranscriptError):
    """A child transcript path escaped the root session's ``agents/`` subtree."""


class ChildTranscriptPayloadError(ChildTranscriptError):
    """A spawn-payload value failed its strict codec."""


def encode_inherited_workspace(workspace: Workspace) -> dict[str, Any]:
    """Encode the inherited workspace as the JSON-safe payload value.

    Every field is a plain path, so the encoding is exact string round-trips:
    the child re-opens with the same canonical cwd, roots and ceiling.
    """
    return {
        "cwd": str(workspace.cwd),
        "authorized_roots": [str(root) for root in workspace.authorized_roots],
        "ceiling": (
            None
            if workspace.ceiling is None
            else encode_inherited_workspace(workspace.ceiling)
        ),
    }


def decode_inherited_workspace(payload: object) -> Workspace:
    """Strictly decode one inherited-workspace payload value.

    No fallback and no repair: a malformed value raises the typed payload
    error instead of opening the child with a workspace it never had. A
    payload WITHOUT the key is not decoded here at all -- the child answers
    that shape with the loud ``inherited_workspace`` open failure an
    old-version peer deserves.
    """
    if not isinstance(payload, dict) or set(payload) != {
        "cwd",
        "authorized_roots",
        "ceiling",
    }:
        raise ChildTranscriptPayloadError(
            "the inherited_workspace payload must carry exactly cwd, "
            "authorized_roots and ceiling"
        )
    cwd = payload["cwd"]
    roots = payload["authorized_roots"]
    if not isinstance(cwd, str) or not cwd:
        raise ChildTranscriptPayloadError(
            "the inherited_workspace payload needs a cwd string"
        )
    if not isinstance(roots, list) or not all(
        isinstance(root, str) and root for root in roots
    ):
        raise ChildTranscriptPayloadError(
            "the inherited_workspace payload needs authorized_roots "
            "as a list of path strings"
        )
    ceiling = payload["ceiling"]
    return Workspace(
        cwd=Path(cwd),
        authorized_roots=tuple(Path(root) for root in roots),
        ceiling=None if ceiling is None else decode_inherited_workspace(ceiling),
    )


_SOURCE_RESTRICTION_KEYS = frozenset({
    "layer_name",
    "locator",
    "kind",
    "content_fingerprint",
    "store_fingerprint",
    "tools",
    "identity",
    "authorized_roots",
    "dangerous_command_grants",
    "remote_exec_grants",
    "replaces_roots",
})


def encode_source_restrictions(
    restrictions: tuple[SourceRestrictions, ...],
) -> list[dict[str, Any]]:
    """Encode accepted source restrictions as JSON-safe payload values."""
    return [_encode_source_restriction(restriction) for restriction in restrictions]


def _encode_source_restriction(restriction: SourceRestrictions) -> dict[str, Any]:
    identity = None
    if restriction.identity is not None:
        identity = {
            "owner": str(restriction.identity.owner),
            "source": str(restriction.identity.source),
        }
    return {
        "layer_name": restriction.layer_name,
        "locator": restriction.locator,
        "kind": restriction.kind,
        "content_fingerprint": restriction.content_fingerprint,
        "store_fingerprint": restriction.store_fingerprint,
        "tools": [
            {
                "tool_name": tool.tool_name,
                "denied": tool.denied,
                "denylist": list(tool.denylist),
                "sensitive_patterns": list(tool.sensitive_patterns),
            }
            for tool in restriction.tools
        ],
        "identity": identity,
        "authorized_roots": [
            {
                "project": str(authority.project),
                "roots": [str(root) for root in authority.roots],
            }
            for authority in restriction.authorized_roots
        ],
        "dangerous_command_grants": [
            {"project": str(grant.project), "operations": list(grant.operations)}
            for grant in restriction.dangerous_command_grants
        ],
        "remote_exec_grants": [
            {
                "project": str(grant.project),
                "destinations": [
                    {
                        "user": destination.user,
                        "host": destination.host,
                        "port": destination.port,
                    }
                    for destination in grant.destinations
                ],
            }
            for grant in restriction.remote_exec_grants
        ],
        "replaces_roots": restriction.replaces_roots,
    }


def decode_source_restrictions(payload: object) -> tuple[SourceRestrictions, ...]:
    """Strictly decode the inherited-restrictions payload value.

    No fallback and no repair: every field must carry exactly the shape the
    encoder wrote, so a malformed value raises the typed payload error instead
    of opening the child with a policy it never had.
    """
    if not isinstance(payload, list):
        raise ChildTranscriptPayloadError(
            "the inherited restrictions payload must be a list of source records"
        )
    return tuple(_decode_source_restriction(record) for record in payload)


def _decode_source_restriction(record: object) -> SourceRestrictions:
    if not isinstance(record, dict) or set(record) != _SOURCE_RESTRICTION_KEYS:
        raise ChildTranscriptPayloadError(
            "an inherited restriction must carry exactly the source record keys"
        )
    kind = record["kind"]
    if kind not in {"source", "mode"}:
        raise ChildTranscriptPayloadError(
            "an inherited restriction carries an unknown source kind"
        )
    tools = _decode_tool_restrictions(record["tools"])
    identity = _decode_policy_source_identity(record["identity"])
    authorized_roots = _decode_authorized_roots(record["authorized_roots"])
    dangerous_command_grants = _decode_dangerous_command_grants(
        record["dangerous_command_grants"]
    )
    remote_exec_grants = _decode_remote_exec_grants(record["remote_exec_grants"])
    store_fingerprint = record["store_fingerprint"]
    if store_fingerprint is not None and not isinstance(store_fingerprint, str):
        raise ChildTranscriptPayloadError(
            "an inherited restriction needs a store fingerprint string or null"
        )
    if not isinstance(record["replaces_roots"], bool):
        raise ChildTranscriptPayloadError(
            "an inherited restriction needs replaces_roots as a boolean"
        )
    for field in ("layer_name", "locator", "content_fingerprint"):
        if not isinstance(record[field], str):
            raise ChildTranscriptPayloadError(
                "an inherited restriction needs string provenance fields"
            )
    return SourceRestrictions(
        layer_name=record["layer_name"],
        locator=record["locator"],
        kind=kind,
        content_fingerprint=record["content_fingerprint"],
        store_fingerprint=store_fingerprint,
        tools=tools,
        identity=identity,
        authorized_roots=authorized_roots,
        dangerous_command_grants=dangerous_command_grants,
        remote_exec_grants=remote_exec_grants,
        replaces_roots=record["replaces_roots"],
    )


def _decode_tool_restrictions(payload: object) -> tuple[ToolRestriction, ...]:
    if not isinstance(payload, list):
        raise ChildTranscriptPayloadError(
            "inherited tool restrictions must be a list of tool records"
        )
    tools: list[ToolRestriction] = []
    for record in payload:
        if not isinstance(record, dict) or set(record) != {
            "tool_name",
            "denied",
            "denylist",
            "sensitive_patterns",
        }:
            raise ChildTranscriptPayloadError(
                "an inherited tool restriction must carry exactly its four keys"
            )
        tool_name, denied = record["tool_name"], record["denied"]
        denylist, sensitive = record["denylist"], record["sensitive_patterns"]
        if not isinstance(tool_name, str) or not tool_name:
            raise ChildTranscriptPayloadError(
                "an inherited tool restriction needs a tool name string"
            )
        if not isinstance(denied, bool):
            raise ChildTranscriptPayloadError(
                "an inherited tool restriction needs denied as a boolean"
            )
        for values, label in (
            (denylist, "denylist"),
            (sensitive, "sensitive_patterns"),
        ):
            if not isinstance(values, list) or not all(
                isinstance(value, str) for value in values
            ):
                raise ChildTranscriptPayloadError(
                    f"an inherited tool restriction needs {label} as a string list"
                )
        tools.append(
            ToolRestriction(
                tool_name=tool_name,
                denied=denied,
                denylist=tuple(denylist),
                sensitive_patterns=tuple(sensitive),
            )
        )
    return tuple(tools)


def _decode_policy_source_identity(payload: object) -> PolicySourceIdentity | None:
    if payload is None:
        return None
    if not isinstance(payload, dict) or set(payload) != {"owner", "source"}:
        raise ChildTranscriptPayloadError(
            "an inherited policy source identity must carry owner and source"
        )
    try:
        return PolicySourceIdentity(
            owner=UUID(payload["owner"]), source=UUID(payload["source"])
        )
    except (ValueError, AttributeError, TypeError) as exc:
        raise ChildTranscriptPayloadError(
            "an inherited policy source identity needs two UUID strings"
        ) from exc


def _decode_authorized_roots(payload: object) -> tuple[ProjectRootAuthority, ...]:
    if not isinstance(payload, list):
        raise ChildTranscriptPayloadError(
            "inherited authorized roots must be a list of authority records"
        )
    authorities: list[ProjectRootAuthority] = []
    for record in payload:
        if not isinstance(record, dict) or set(record) != {"project", "roots"}:
            raise ChildTranscriptPayloadError(
                "an inherited root authority must carry project and roots"
            )
        project, roots = record["project"], record["roots"]
        if not isinstance(project, str) or not project:
            raise ChildTranscriptPayloadError(
                "an inherited root authority needs a project string"
            )
        if (
            not isinstance(roots, list)
            or not roots
            or not all(isinstance(root, str) and root for root in roots)
        ):
            raise ChildTranscriptPayloadError(
                "an inherited root authority needs roots as path strings"
            )
        authorities.append(
            ProjectRootAuthority(
                project=Path(project), roots=tuple(Path(root) for root in roots)
            )
        )
    return tuple(authorities)


def _decode_dangerous_command_grants(
    payload: object,
) -> tuple[ProjectDangerousCommandGrant, ...]:
    if not isinstance(payload, list):
        raise ChildTranscriptPayloadError(
            "inherited dangerous-command grants must be a list of grant records"
        )
    grants: list[ProjectDangerousCommandGrant] = []
    for record in payload:
        if not isinstance(record, dict) or set(record) != {"project", "operations"}:
            raise ChildTranscriptPayloadError(
                "an inherited command grant must carry project and operations"
            )
        project, operations = record["project"], record["operations"]
        if not isinstance(project, str) or not project:
            raise ChildTranscriptPayloadError(
                "an inherited command grant needs a project string"
            )
        if not isinstance(operations, list) or not all(
            isinstance(operation, str) for operation in operations
        ):
            raise ChildTranscriptPayloadError(
                "an inherited command grant needs operations as a string list"
            )
        grants.append(
            ProjectDangerousCommandGrant(
                project=Path(project), operations=tuple(operations)
            )
        )
    return tuple(grants)


def _decode_remote_exec_grants(payload: object) -> tuple[ProjectRemoteExecGrant, ...]:
    if not isinstance(payload, list):
        raise ChildTranscriptPayloadError(
            "inherited remote-exec grants must be a list of grant records"
        )
    grants: list[ProjectRemoteExecGrant] = []
    for record in payload:
        if not isinstance(record, dict) or set(record) != {"project", "destinations"}:
            raise ChildTranscriptPayloadError(
                "an inherited remote-exec grant must carry project and destinations"
            )
        project, destinations = record["project"], record["destinations"]
        if not isinstance(project, str) or not project:
            raise ChildTranscriptPayloadError(
                "an inherited remote-exec grant needs a project string"
            )
        if not isinstance(destinations, list):
            raise ChildTranscriptPayloadError(
                "an inherited remote-exec grant needs destinations as a list"
            )
        parsed: list[RemoteDestination] = []
        for destination in destinations:
            if not isinstance(destination, dict) or set(destination) != {
                "user",
                "host",
                "port",
            }:
                raise ChildTranscriptPayloadError(
                    "an inherited remote destination must carry user, host and port"
                )
            user, host, port = (
                destination["user"],
                destination["host"],
                destination["port"],
            )
            if not isinstance(user, str) or not isinstance(host, str):
                raise ChildTranscriptPayloadError(
                    "an inherited remote destination needs user and host strings"
                )
            if isinstance(port, bool) or not isinstance(port, int):
                raise ChildTranscriptPayloadError(
                    "an inherited remote destination needs port as an integer"
                )
            parsed.append(RemoteDestination(user=user, host=host, port=port))
        grants.append(
            ProjectRemoteExecGrant(project=Path(project), destinations=tuple(parsed))
        )
    return tuple(grants)


def encode_inherited_root_grants(roots: tuple[Path, ...]) -> list[str]:
    """Encode the dispatch root grants as the JSON-safe payload value."""
    return [str(root) for root in roots]


def decode_inherited_root_grants(payload: object) -> tuple[Path, ...]:
    """Strictly decode the inherited root grants payload value."""
    if not isinstance(payload, list) or not all(
        isinstance(root, str) and root for root in payload
    ):
        raise ChildTranscriptPayloadError(
            "the inherited root grants payload needs a list of path strings"
        )
    return tuple(Path(root) for root in payload)


def encode_child_policy_payload(loop: AgentLoop) -> dict[str, Any]:
    """The parent-policy keys every process spawn must merge unconditionally.

    The process-route counterpart of the in-process route's
    ``child_runtime_policy`` handover: the parent loop's accepted source
    restrictions (its own inherited sources plus the live orchestrator's
    source-kind contributions, deduplicated), its mode restrictions, its
    dispatch root grants and its budget-bearing session options ride the
    spawn payload. Every value is read from the parent loop -- nothing is
    fabricated. ``parent_authority_getter`` and
    ``parent_authority_revision_getter`` are process-locals no payload can
    carry; a payload without these keys answers the child's loud open
    failure instead of opening with a policy it never had.
    """
    from chartreux.app_server.protocol import AgentConfig

    policy = loop.runtime_policy
    sources = loop.config_orchestrator.restrictions
    child_sources = tuple(
        dict.fromkeys(
            policy.inherited_restrictions
            + tuple(source for source in sources if source.kind == "source")
        )
    )
    return {
        AGENT_CONFIG_PAYLOAD_KEY: AgentConfig(
            cwd=str(loop.cwd),
            max_turns=policy.max_turns,
            max_price=policy.max_price,
            max_session_tokens=policy.max_session_tokens,
            run_max_price=policy.run_max_price,
            run_max_session_tokens=policy.run_max_session_tokens,
            headless=policy.headless,
        ).model_dump(mode="json"),
        INHERITED_RESTRICTIONS_PAYLOAD_KEY: encode_source_restrictions(child_sources),
        INHERITED_MODE_RESTRICTIONS_PAYLOAD_KEY: encode_source_restrictions(
            policy.inherited_mode_restrictions
        ),
        INHERITED_ROOT_GRANTS_PAYLOAD_KEY: encode_inherited_root_grants(
            loop._dispatch_root_grants
        ),
    }


def require_child_policy_payload_keys(payload: Mapping[str, Any]) -> None:
    """Refuse a spawn payload missing its parent-policy keys.

    The strict F11 stance, matching F6's workspace stance: a legacy payload a
    pre-F11 peer would have opened with an empty policy answers the typed
    payload error at the child's open, never a silent default.
    """
    missing = [
        key
        for key in (
            AGENT_CONFIG_PAYLOAD_KEY,
            INHERITED_RESTRICTIONS_PAYLOAD_KEY,
            INHERITED_MODE_RESTRICTIONS_PAYLOAD_KEY,
            INHERITED_ROOT_GRANTS_PAYLOAD_KEY,
        )
        if key not in payload
    ]
    if missing:
        raise ChildTranscriptPayloadError(
            "a child spawn payload is missing its parent-policy keys ("
            + ", ".join(missing)
            + ")"
        )


class ChildTranscriptOutcome(StrEnum):
    """How far a child got before its process died (design.md:196-205)."""

    NEVER_EXECUTED = "never_executed"
    STARTED_OUTCOME_UNKNOWN = "started_outcome_unknown"
    COMPLETED = "completed"
    TOMBSTONED = "tombstoned"
    UNREADABLE = "unreadable"


@dataclass(frozen=True, slots=True)
class ChildStartMarker:
    """The durable record written BEFORE the child executes a run.

    A start marker without a matching terminal result is "started, outcome
    unknown" after a crash: the crash may have landed before or during
    execution, so the run is never replayed. The run-time semantics are
    unchanged: the marker is written immediately before the run, never at
    publication (which would misclassify a never-run child). ``attempt_id``
    joins the payload when the publication attempt is known, so the marker
    bridges to the attempt that created the child.
    """

    agent_id: str
    session_id: str
    run_id: str
    started_at: str
    attempt_id: str | None = None
    version: int = CHILD_MARKER_VERSION

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "version": self.version,
            "kind": "child-start",
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "run_id": self.run_id,
            "started_at": self.started_at,
        }
        if self.attempt_id is not None:
            payload["attempt_id"] = self.attempt_id
        return payload

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> ChildStartMarker:
        """Rebuild one start marker, refusing a malformed record.

        ``attempt_id`` is tolerated in both directions: a marker written before
        the field existed loads without it, and an unknown value is ignored
        rather than failing the whole record.
        """
        if payload.get("kind") != "child-start":
            raise ValueError("not a child start marker")
        return cls(
            agent_id=_required_marker_str(payload, "agent_id"),
            session_id=_required_marker_str(payload, "session_id"),
            run_id=_required_marker_str(payload, "run_id"),
            started_at=_required_marker_str(payload, "started_at"),
            attempt_id=_optional_marker_str(payload, "attempt_id"),
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
    # The run's typed failure evidence (item 2 B.3): the derived envelope plus
    # the inputs it was derived from, written with the terminal record so a
    # crash between the record and the completion frame still leaves the
    # measured evidence on disk. Absent on completed runs and on records a
    # pre-rider child wrote.
    failure: dict[str, object] | None = None
    version: int = CHILD_MARKER_VERSION

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
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
        if self.failure is not None:
            payload["failure"] = self.failure
        return payload

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> ChildResultRecord:
        """Rebuild one terminal result record, refusing a malformed record.

        ``failure`` is tolerated in both directions: a record written before
        the field existed loads without it, and a malformed block is dropped
        rather than failing the whole record — the outcome is the record's
        contract, the envelope is additive evidence.
        """
        if payload.get("kind") != "child-result":
            raise ValueError("not a child result record")
        error = payload.get("error")
        failure = payload.get("failure")
        return cls(
            agent_id=_required_marker_str(payload, "agent_id"),
            session_id=_required_marker_str(payload, "session_id"),
            run_id=_required_marker_str(payload, "run_id"),
            result_id=_required_marker_str(payload, "result_id"),
            digest=_required_marker_str(payload, "digest"),
            outcome=_required_marker_str(payload, "outcome"),
            completed_at=_required_marker_str(payload, "completed_at"),
            error=error if isinstance(error, str) else None,
            failure=dict(failure) if isinstance(failure, dict) else None,
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
class ChildTombstoneRecord:
    """The durable retraction of one child publication attempt.

    Written by the teardown/rollback course when the persisted records of an
    abandoned attempt must never resurrect: a tombstoned transcript adjudicates
    as ``TOMBSTONED`` and the reconciliation scan never hands it to the root.
    The record carries the attempt id the publication's first durable write
    bridged, and it falls back to the pre-existing session id, so both a
    v0.7.8 attempt and a legacy record without an id are retractable.
    """

    session_id: str
    reason: str
    created_at: str
    attempt_id: str | None = None
    agent_id: str | None = None
    version: int = CHILD_MARKER_VERSION

    def to_payload(self) -> dict[str, object]:
        return {
            "version": self.version,
            "kind": "child-tombstone",
            "session_id": self.session_id,
            "attempt_id": self.attempt_id,
            "agent_id": self.agent_id,
            "reason": self.reason,
            "created_at": self.created_at,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> ChildTombstoneRecord:
        """Rebuild one tombstone record, refusing a malformed retraction."""
        if payload.get("kind") != "child-tombstone":
            raise ValueError("not a child tombstone record")
        return cls(
            session_id=_required_marker_str(payload, "session_id"),
            reason=_required_marker_str(payload, "reason"),
            created_at=_required_marker_str(payload, "created_at"),
            attempt_id=_optional_marker_str(payload, "attempt_id"),
            agent_id=_optional_marker_str(payload, "agent_id"),
            version=_marker_version(payload),
        )


@dataclass(frozen=True, slots=True)
class ChildTranscriptAdjudication:
    """The marker-based state of one child transcript after its process died.

    ``NEVER_EXECUTED`` (no start marker) is distinct from
    ``STARTED_OUTCOME_UNKNOWN`` (start marker, no matching terminal result).
    Neither is ever a replay instruction: an ambiguous or unknown outcome is
    surfaced to the user, never re-executed (design.md:196-205). A
    ``TOMBSTONED`` transcript is retracted: the attempt was abandoned, so no
    outcome is reported and nothing may be adopted from it.
    """

    child_dir: Path
    outcome: ChildTranscriptOutcome
    start_marker: ChildStartMarker | None = None
    result: ChildResultRecord | None = None
    tombstone: ChildTombstoneRecord | None = None
    attempt_id: str | None = None
    detail: str = ""

    @property
    def tombstoned(self) -> bool:
        return self.outcome is ChildTranscriptOutcome.TOMBSTONED

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


def _optional_marker_str(payload: dict[str, Any], key: str) -> str | None:
    """A tolerated optional field: absent or unknown values read as None."""
    value = payload.get(key)
    return value if isinstance(value, str) and value else None


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

    The tombstone is the one coordinator-authorized WRITE: a publication
    attempt that could not be completed is retracted with it, so its persisted
    records can never resurrect through the disk scan.
    """

    def __init__(self, child_dir: Path, *, root_session_dir: Path) -> None:
        self.root_session_dir = Path(root_session_dir).resolve()
        self.child_dir = _confined_child_dir(child_dir, self.root_session_dir)
        self.start_marker_path = self.child_dir / CHILD_START_MARKER_FILENAME
        self.result_path = self.child_dir / CHILD_RESULT_FILENAME
        self.tombstone_path = self.child_dir / CHILD_TOMBSTONE_FILENAME

    def record_start(self, marker: ChildStartMarker) -> None:
        """Durably record the start marker before the run executes."""
        self._ensure_dir()
        _atomic_write_json(self.start_marker_path, marker.to_payload())

    def record_result(self, result: ChildResultRecord) -> None:
        """Durably record the terminal result save-before-emit."""
        self._ensure_dir()
        _atomic_write_json(self.result_path, result.to_payload())

    def record_tombstone(self, tombstone: ChildTombstoneRecord) -> None:
        """Durably retract this transcript; a repeated retraction is a no-op.

        Idempotence is keyed on the retracted session and attempt: an attempt
        retracted twice keeps its original record byte for byte, a retraction
        by session id alone is upgraded once the attempt id is known, and a
        known attempt id is never downgraded.
        """
        try:
            existing = self.read_tombstone()
        except ChildTranscriptError:
            existing = None
        if existing is not None and existing.session_id == tombstone.session_id:
            if existing.attempt_id == tombstone.attempt_id:
                return
            if tombstone.attempt_id is None:
                return
        self._ensure_dir()
        _atomic_write_json(self.tombstone_path, tombstone.to_payload())

    def read_tombstone(self) -> ChildTombstoneRecord | None:
        """Read this transcript's retraction record, when one exists."""
        payload, error = _read_marker_payload(self.tombstone_path)
        if error is not None:
            raise ChildTranscriptError(error)
        if payload is None:
            return None
        try:
            return ChildTombstoneRecord.from_payload(payload)
        except ValueError as exc:
            raise ChildTranscriptError(str(exc)) from exc

    def adjudicate(self) -> ChildTranscriptAdjudication:
        """Classify this child transcript from its markers, never replaying."""
        retraction = self._retraction()
        if retraction is not None:
            return retraction
        return self._marker_verdict()

    def _marker_verdict(self) -> ChildTranscriptAdjudication:
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
        attempt_id = (start.attempt_id if start is not None else None) or (
            persisted_child_attempt_id(self.child_dir)
        )
        if start is None and result is None:
            return ChildTranscriptAdjudication(
                child_dir=self.child_dir,
                outcome=ChildTranscriptOutcome.NEVER_EXECUTED,
                attempt_id=attempt_id,
            )
        if result is not None and (start is None or start.run_id == result.run_id):
            return ChildTranscriptAdjudication(
                child_dir=self.child_dir,
                outcome=ChildTranscriptOutcome.COMPLETED,
                start_marker=start,
                result=result,
                attempt_id=attempt_id,
            )
        return ChildTranscriptAdjudication(
            child_dir=self.child_dir,
            outcome=ChildTranscriptOutcome.STARTED_OUTCOME_UNKNOWN,
            start_marker=start,
            result=result,
            attempt_id=attempt_id,
        )

    def _retraction(self) -> ChildTranscriptAdjudication | None:
        """The retraction verdict, when this transcript carries a tombstone.

        A retracted attempt reports no outcome at all: the tombstone wins over
        every marker, so a completed-looking record can never be adopted after
        its publication was abandoned.
        """
        payload, error = _read_marker_payload(self.tombstone_path)
        if error is not None:
            return _unreadable(self.child_dir, self.tombstone_path, error)
        tombstone = _parse_marker(ChildTombstoneRecord.from_payload, payload)
        if isinstance(tombstone, str):
            return _unreadable(self.child_dir, self.tombstone_path, tombstone)
        if tombstone is None:
            return None
        return ChildTranscriptAdjudication(
            child_dir=self.child_dir,
            outcome=ChildTranscriptOutcome.TOMBSTONED,
            tombstone=tombstone,
            attempt_id=tombstone.attempt_id,
            detail=f"{CHILD_TOMBSTONE_FILENAME}: {tombstone.reason}",
        )

    def _ensure_dir(self) -> None:
        self.child_dir.mkdir(parents=True, mode=0o700, exist_ok=True)


def persisted_child_attempt_id(child_dir: Path) -> str | None:
    """The publication attempt id the child's persisted first write carries.

    The publication transaction persists the empty child session BEFORE the
    parent link, so this bridge is what attributes a torn persist-to-link
    window to its attempt. A missing, torn, or pre-v0.7.8 record simply
    carries no id and retracts through its session id instead.
    """
    payload, error = _read_marker_payload(Path(child_dir) / METADATA_FILENAME)
    if error is not None or payload is None:
        return None
    value = payload.get("attempt_id")
    return value if isinstance(value, str) and value else None


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
    "CHILD_TOMBSTONE_FILENAME",
    "CHILD_TRANSCRIPTS_DIRNAME",
    "CLIENT_CAPABILITIES_PAYLOAD_KEY",
    "CLIENT_INFO_PAYLOAD_KEY",
    "INHERITED_WORKSPACE_PAYLOAD_KEY",
    "ROOT_SESSION_DIR_PAYLOAD_KEY",
    "ChildResultRecord",
    "ChildStartMarker",
    "ChildTombstoneRecord",
    "ChildTranscriptAdjudication",
    "ChildTranscriptError",
    "ChildTranscriptOutcome",
    "ChildTranscriptPathError",
    "ChildTranscriptPayloadError",
    "ChildTranscriptStore",
    "TranscriptReads",
    "TranscriptReadsView",
    "child_result_digest",
    "child_transcript_dir_name",
    "child_transcript_dirs",
    "child_transcript_logging",
    "child_transcript_root_for",
    "child_transcripts_root",
    "decode_inherited_workspace",
    "encode_inherited_workspace",
    "persisted_child_attempt_id",
    "read_child_transcript",
    "resolve_child_root_session_dir",
    "root_session_dir_from_spec",
]
