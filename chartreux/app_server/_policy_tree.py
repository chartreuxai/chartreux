"""Cross-process policy-tree transactions (WP5 [5D]).

The policy tree moves through four phases -- reserve, prepare, commit, retire
-- and the same protocol drives an in-process tree and a tree of subagent
processes. The process boundary is the node port: :class:`InProcessPolicyNode`
runs the node service in this process (the ``CHARTREUX_TRANSPORT=in_process``
escape), and :class:`ProcessPolicyNode` speaks the identical four phases over
the agent wire.

Every phase is journaled before the next one starts, so a crash is reconciled
from the record instead of guessed:

* a crash between reserve and prepare leaves the tree ``reserved`` --
  reserved but unprepared -- which :meth:`PolicyTreeTransaction.reprepare`
  completes, or :meth:`PolicyTreeTransaction.abort` releases;
* a crash after commit and before retire leaves the old tree ``committed`` --
  retired-pending -- and an authorized read fails closed
  (:class:`PolicyTreeRetiredPendingError`) until
  :meth:`PolicyTreeTransaction.resume_retire` finishes the retirement.

The commit gate is the [3C-1] policy-revision lock: each node captures its
revision at reserve, and commit refuses typed
(:class:`PolicyRevisionStaleError`) when the node's accepted revision, or its
dangerous-command policy revision, moved in between. A commit refused for a
stale revision publishes nothing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, replace
from enum import StrEnum
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Protocol
from uuid import uuid4

from chartreux.app_server.agent_wire import (
    POLICY_ABORT_METHOD,
    POLICY_COMMIT_METHOD,
    POLICY_PREPARE_METHOD,
    POLICY_RESERVE_METHOD,
    POLICY_RETIRE_METHOD,
    POLICY_REVISION_METHOD,
)
from chartreux.app_server.registry_manifest import RegistryManifest
from chartreux.app_server.transport import (
    IN_PROCESS_TRANSPORT,
    TransportSelection,
    resolve_transport_selection,
)
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.agent_loop._config_transactions import _PreparedPolicyReplacement


class PolicyTreeError(RuntimeError):
    """Base class for policy-tree transaction failures."""


class PolicyTreeBusyError(PolicyTreeError):
    """A node is already serving a different transaction."""


class PolicyNodePhaseError(PolicyTreeError):
    """A phase request arrived out of order."""


class PolicyRevisionStaleError(PolicyTreeError):
    """The transaction's accepted revision is no longer current; commit refuses."""


class PolicyTreeJournalError(PolicyTreeError):
    """The journal is missing, unreadable, or disagrees with the node set."""


class PolicyTreeRetiredPendingError(PolicyTreeError):
    """The new tree committed but the previous tree is still retired-pending.

    An authorized read fails closed here: until the retirement record is
    written, the tree the reader would observe is not settled, so the read is
    refused rather than served from a half-retired tree.
    """

    def __init__(self, transaction_id: str) -> None:
        self.transaction_id = transaction_id
        super().__init__(
            f"Policy tree transaction {transaction_id} is committed but its "
            "previous tree is retired-pending; reconcile before reading policy"
        )


class PolicyTreePhase(StrEnum):
    """The journaled phase of one policy-tree transaction."""

    RESERVED = "reserved"
    PREPARED = "prepared"
    COMMITTED = "committed"
    RETIRED = "retired"
    ABORTED = "aborted"


class PolicyTreeRecoveryDecision(StrEnum):
    """What a reader of a journaled transaction must do next."""

    COMPLETE = "complete"
    ABORTED = "aborted"
    REPREPARE_OR_ABORT = "reprepare_or_abort"
    RETIRED_PENDING = "retired_pending"


@dataclass(frozen=True, slots=True)
class PolicyNodeRevision:
    """One node's accepted revision: the tree token plus the policy lock."""

    tree_revision: str
    policy_revision: int

    def to_wire(self) -> dict[str, Any]:
        return {
            "tree_revision": self.tree_revision,
            "policy_revision": self.policy_revision,
        }

    @classmethod
    def from_wire(cls, payload: object) -> PolicyNodeRevision:
        if not isinstance(payload, Mapping):
            raise PolicyNodePhaseError("a policy revision frame carries no revision")
        tree_revision = payload.get("tree_revision")
        policy_revision = payload.get("policy_revision")
        if not isinstance(tree_revision, str) or not tree_revision:
            raise PolicyNodePhaseError(
                "a policy revision frame carries no tree revision"
            )
        if type(policy_revision) is not int or policy_revision < 0:
            raise PolicyNodePhaseError(
                "a policy revision frame carries no policy revision"
            )
        return cls(tree_revision=tree_revision, policy_revision=policy_revision)


@dataclass(frozen=True, slots=True)
class PolicyChange:
    """The secret-free, JSON-safe policy change one transaction applies."""

    source: str
    tools: dict[str, Any]
    roots: dict[str, list[str]] | None = None

    def to_wire(self) -> dict[str, Any]:
        return {"source": self.source, "tools": self.tools, "roots": self.roots}

    @classmethod
    def from_wire(cls, payload: object) -> PolicyChange:
        if not isinstance(payload, Mapping):
            raise PolicyNodePhaseError("a policy frame carries no change")
        source = payload.get("source")
        if not isinstance(source, str) or not source:
            raise PolicyNodePhaseError("a policy change carries no source")
        tools = payload.get("tools")
        if not isinstance(tools, Mapping):
            raise PolicyNodePhaseError("a policy change carries no tools mapping")
        roots_payload = payload.get("roots")
        roots: dict[str, list[str]] | None = None
        if roots_payload is not None:
            if not isinstance(roots_payload, Mapping):
                raise PolicyNodePhaseError("a policy change carries malformed roots")
            roots = {
                str(project): [str(root) for root in values]
                for project, values in roots_payload.items()
            }
        return cls(source=source, tools=dict(tools), roots=roots)


@dataclass(frozen=True, slots=True)
class PolicyNodeJournalEntry:
    """One node's durable record inside a transaction."""

    node_id: str
    revision: PolicyNodeRevision
    handle: str | None = None
    committed_revision: PolicyNodeRevision | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "revision": self.revision.to_wire(),
            "handle": self.handle,
            "committed_revision": (
                self.committed_revision.to_wire()
                if self.committed_revision is not None
                else None
            ),
        }

    @classmethod
    def from_payload(cls, payload: object) -> PolicyNodeJournalEntry:
        if not isinstance(payload, Mapping):
            raise PolicyTreeJournalError("a journal node entry is malformed")
        node_id = payload.get("node_id")
        if not isinstance(node_id, str) or not node_id:
            raise PolicyTreeJournalError("a journal node entry carries no node id")
        handle = payload.get("handle")
        committed = payload.get("committed_revision")
        return cls(
            node_id=node_id,
            revision=PolicyNodeRevision.from_wire(payload.get("revision")),
            handle=handle if isinstance(handle, str) and handle else None,
            committed_revision=(
                PolicyNodeRevision.from_wire(committed)
                if committed is not None
                else None
            ),
        )


@dataclass(frozen=True, slots=True)
class PolicyTreeRecord:
    """The durable state of one transaction, phase by phase."""

    transaction_id: str
    phase: PolicyTreePhase
    revision: str
    change: dict[str, Any]
    nodes: tuple[PolicyNodeJournalEntry, ...]

    def to_payload(self) -> dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "phase": self.phase.value,
            "revision": self.revision,
            "change": self.change,
            "nodes": [entry.to_payload() for entry in self.nodes],
        }

    @classmethod
    def from_payload(cls, payload: object) -> PolicyTreeRecord:
        if not isinstance(payload, Mapping):
            raise PolicyTreeJournalError("a policy decree carries no transaction")
        transaction_id = payload.get("transaction_id")
        phase = payload.get("phase")
        revision = payload.get("revision")
        change = payload.get("change")
        nodes = payload.get("nodes")
        if not isinstance(transaction_id, str) or not transaction_id:
            raise PolicyTreeJournalError("a policy decree carries no transaction id")
        if not isinstance(phase, str) or phase not in set(PolicyTreePhase):
            raise PolicyTreeJournalError("a policy decree carries no known phase")
        if not isinstance(revision, str) or not revision:
            raise PolicyTreeJournalError("a policy decree carries no revision")
        if not isinstance(change, Mapping):
            raise PolicyTreeJournalError("a policy decree carries no change")
        if not isinstance(nodes, Sequence) or isinstance(nodes, (str, bytes)):
            raise PolicyTreeJournalError("a policy decree carries no node records")
        return cls(
            transaction_id=transaction_id,
            phase=PolicyTreePhase(phase),
            revision=revision,
            change=dict(change),
            nodes=tuple(PolicyNodeJournalEntry.from_payload(entry) for entry in nodes),
        )


@dataclass(frozen=True, slots=True)
class PolicyTreeRecovery:
    """The adjudication of one journaled transaction after a crash."""

    transaction_id: str
    phase: PolicyTreePhase
    decision: PolicyTreeRecoveryDecision
    record: PolicyTreeRecord


class PolicyTreeJournal(Protocol):
    """Durable storage for one transaction's records."""

    def write(self, record: PolicyTreeRecord) -> None: ...

    def read(self) -> PolicyTreeRecord | None: ...


class FilePolicyTreeJournal:
    """A journal file beside the session state (tests and scratch runs).

    Production uses :class:`ManifestPolicyTreeJournal`, which stores the
    same records in the existing registry manifest, so the policy tree adds
    no new persistent artifact.
    """

    def __init__(self, directory: Path, transaction_id: str) -> None:
        self.path = directory / f"{transaction_id}.policy-tree.json"

    def write(self, record: PolicyTreeRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.path.parent, delete=False
            ) as temporary:
                temporary_path = Path(temporary.name)
                temporary.write(json.dumps(record.to_payload(), sort_keys=True) + "\n")
            os.replace(temporary_path, self.path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def read(self) -> PolicyTreeRecord | None:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError) as exc:
            raise PolicyTreeJournalError(
                f"Policy tree journal at {self.path} is unreadable"
            ) from exc
        return PolicyTreeRecord.from_payload(payload)


class ManifestPolicyTreeJournal:
    """A journal kept in the registry manifest (no new persistent artifact)."""

    def __init__(self, manifest: RegistryManifest, transaction_id: str) -> None:
        self._manifest = manifest
        self._transaction_id = transaction_id

    def write(self, record: PolicyTreeRecord) -> None:
        self._manifest.record_policy_transaction(
            self._transaction_id, record.to_payload()
        )

    def read(self) -> PolicyTreeRecord | None:
        payload = self._manifest.policy_transaction(self._transaction_id)
        return None if payload is None else PolicyTreeRecord.from_payload(payload)


class PolicyNodeService:
    """One agent loop's side of the four-phase transaction.

    The same service backs the in-process node and the child-process request
    handlers, so process boundaries are the only difference between the two
    topologies.
    """

    def __init__(self, loop: AgentLoop) -> None:
        self._loop = loop
        self._transaction_id: str | None = None
        self._reserved: PolicyNodeRevision | None = None
        self._prepared: _PreparedPolicyReplacement | None = None
        self._handle: str | None = None

    def revision(self) -> PolicyNodeRevision:
        """The node's current accepted revision: the transaction's commit gate."""
        orchestrator = self._loop.config_orchestrator
        return PolicyNodeRevision(
            tree_revision=(
                f"{self._loop.session_id}:{self._loop._session_generation}:"
                f"{orchestrator.accepted_token}"
            ),
            policy_revision=(
                self._loop._dangerous_command_grants.current_policy_revision
            ),
        )

    def reserve(self, transaction_id: str) -> PolicyNodeRevision:
        """Reserve this node for one transaction and capture its revision."""
        if self._transaction_id is not None and self._transaction_id != transaction_id:
            raise PolicyTreeBusyError(
                f"Policy node is reserved by transaction {self._transaction_id}"
            )
        self._loop._require_policy_idle()
        if self._transaction_id == transaction_id and self._reserved is not None:
            return self._reserved
        current = self.revision()
        self._transaction_id = transaction_id
        self._reserved = current
        return current

    async def prepare(
        self,
        transaction_id: str,
        change: PolicyChange,
        expected: PolicyNodeRevision | None = None,
    ) -> str:
        """Prepare locally; a re-prepare after a crash adopts the journaled revision."""
        if self._transaction_id is None:
            if expected is None:
                raise PolicyNodePhaseError(
                    "a reserved revision is required to adopt a policy transaction"
                )
            current = self.revision()
            if current != expected:
                raise PolicyRevisionStaleError(
                    f"Policy node {self._loop.session_id} moved past the reserved revision"
                )
            self._transaction_id = transaction_id
            self._reserved = expected
        if self._transaction_id != transaction_id:
            raise PolicyTreeBusyError(
                f"Policy node is reserved by transaction {self._transaction_id}"
            )
        if expected is not None and self._reserved != expected:
            raise PolicyRevisionStaleError(
                f"Policy node {self._loop.session_id} was reserved at another revision"
            )
        self._loop._require_policy_idle()
        token = self._loop.config_orchestrator.accepted_token
        prepared = (
            await self._loop._prepare_root_replacement(
                source=change.source, roots=change.roots or {}, expected_token=token
            )
            if change.roots is not None
            else await self._loop._prepare_policy_replacement(
                source=change.source, tools=change.tools, expected_token=token
            )
        )
        self._prepared = prepared
        self._handle = f"{transaction_id}:{uuid4().hex}"
        return self._handle

    def commit(
        self, transaction_id: str, handle: str, expected: PolicyNodeRevision
    ) -> PolicyNodeRevision:
        """Publish the prepared authority, gated by the policy-revision lock."""
        prepared = self._require_prepared(transaction_id, handle)
        current = self.revision()
        reserved = self._reserved
        if (
            reserved is None
            or expected != reserved
            or current.tree_revision != reserved.tree_revision
            or current.policy_revision != reserved.policy_revision
        ):
            raise PolicyRevisionStaleError(
                f"Policy commit refused for {self._loop.session_id}: "
                "policy revision is stale"
            )
        self._loop._validate_policy_replacement(prepared)
        self._loop._commit_policy_replacement(prepared, retire=False)
        self._loop.advance_dangerous_command_policy_revision()
        return self.revision()

    def retire(self, transaction_id: str, handle: str) -> None:
        """Retire the previous authority; idempotent for crash reconciliation."""
        if self._transaction_id is None and self._prepared is None:
            return
        prepared = self._require_prepared(transaction_id, handle)
        prepared.previous_manager._retire_authority()
        self._clear()

    def abort(self, transaction_id: str) -> None:
        """Release the reservation; nothing prepared is ever published."""
        if self._transaction_id is None:
            return
        if self._transaction_id != transaction_id:
            raise PolicyTreeBusyError(
                f"Policy node is reserved by transaction {self._transaction_id}"
            )
        self._clear()

    async def handle(self, method: str, params: Mapping[str, Any]) -> dict[str, Any]:
        """Serve one policy frame for a remote node."""
        if method == POLICY_REVISION_METHOD:
            return self.revision().to_wire()
        transaction_id = _required_str(params.get("transaction_id"), "transaction id")
        if method == POLICY_RESERVE_METHOD:
            return self.reserve(transaction_id).to_wire()
        if method == POLICY_PREPARE_METHOD:
            change = PolicyChange.from_wire(params.get("change"))
            expected_payload = params.get("expected")
            expected = (
                None
                if expected_payload is None
                else PolicyNodeRevision.from_wire(expected_payload)
            )
            return {"handle": await self.prepare(transaction_id, change, expected)}
        if method == POLICY_COMMIT_METHOD:
            return self.commit(
                transaction_id,
                _required_str(params.get("handle"), "handle"),
                PolicyNodeRevision.from_wire(params.get("expected")),
            ).to_wire()
        if method == POLICY_RETIRE_METHOD:
            self.retire(transaction_id, _required_str(params.get("handle"), "handle"))
            return {"ok": True}
        if method == POLICY_ABORT_METHOD:
            self.abort(transaction_id)
            return {"ok": True}
        raise PolicyNodePhaseError(f"unknown policy method {method!r}")

    def _require_prepared(
        self, transaction_id: str, handle: str
    ) -> _PreparedPolicyReplacement:
        if self._transaction_id != transaction_id:
            raise PolicyNodePhaseError(
                f"Policy node is serving {self._transaction_id}, not {transaction_id}"
            )
        if self._prepared is None or self._handle != handle:
            raise PolicyNodePhaseError("Policy node has no matching prepared authority")
        return self._prepared

    def _clear(self) -> None:
        self._transaction_id = None
        self._reserved = None
        self._prepared = None
        self._handle = None


class PolicyNodePort(Protocol):
    """The process boundary: one participant of a policy-tree transaction."""

    @property
    def node_id(self) -> str: ...

    async def revision(self) -> PolicyNodeRevision: ...

    async def reserve(self, transaction_id: str) -> PolicyNodeRevision: ...

    async def prepare(
        self, transaction_id: str, change: PolicyChange, expected: PolicyNodeRevision
    ) -> str: ...

    async def commit(
        self, transaction_id: str, handle: str, expected: PolicyNodeRevision
    ) -> PolicyNodeRevision: ...

    async def retire(self, transaction_id: str, handle: str) -> None: ...

    async def abort(self, transaction_id: str) -> None: ...


class PolicyWireClient(Protocol):
    """A client that can carry one policy frame to a remote node."""

    async def policy_request(
        self, method: str, params: dict[str, Any]
    ) -> dict[str, Any]: ...


class InProcessPolicyNode:
    """A node whose loop lives in this process (the in-process escape)."""

    def __init__(self, node_id: str, loop: AgentLoop) -> None:
        if not node_id:
            raise ValueError("a policy node carries an id")
        self._node_id = node_id
        self._service = PolicyNodeService(loop)

    @property
    def node_id(self) -> str:
        return self._node_id

    async def revision(self) -> PolicyNodeRevision:
        return self._service.revision()

    async def reserve(self, transaction_id: str) -> PolicyNodeRevision:
        return self._service.reserve(transaction_id)

    async def prepare(
        self, transaction_id: str, change: PolicyChange, expected: PolicyNodeRevision
    ) -> str:
        return await self._service.prepare(transaction_id, change, expected)

    async def commit(
        self, transaction_id: str, handle: str, expected: PolicyNodeRevision
    ) -> PolicyNodeRevision:
        return self._service.commit(transaction_id, handle, expected)

    async def retire(self, transaction_id: str, handle: str) -> None:
        self._service.retire(transaction_id, handle)

    async def abort(self, transaction_id: str) -> None:
        self._service.abort(transaction_id)


class ProcessPolicyNode:
    """A node served by another process over the agent wire."""

    def __init__(self, node_id: str, client: PolicyWireClient) -> None:
        if not node_id:
            raise ValueError("a policy node carries an id")
        self._node_id = node_id
        self._client = client

    @property
    def node_id(self) -> str:
        return self._node_id

    async def revision(self) -> PolicyNodeRevision:
        return PolicyNodeRevision.from_wire(
            await self._call(POLICY_REVISION_METHOD, {})
        )

    async def reserve(self, transaction_id: str) -> PolicyNodeRevision:
        return PolicyNodeRevision.from_wire(
            await self._call(POLICY_RESERVE_METHOD, {"transaction_id": transaction_id})
        )

    async def prepare(
        self, transaction_id: str, change: PolicyChange, expected: PolicyNodeRevision
    ) -> str:
        result = await self._call(
            POLICY_PREPARE_METHOD,
            {
                "transaction_id": transaction_id,
                "change": change.to_wire(),
                "expected": expected.to_wire(),
            },
        )
        handle = result.get("handle")
        if not isinstance(handle, str) or not handle:
            raise PolicyNodePhaseError("the remote policy node returned no handle")
        return handle

    async def commit(
        self, transaction_id: str, handle: str, expected: PolicyNodeRevision
    ) -> PolicyNodeRevision:
        return PolicyNodeRevision.from_wire(
            await self._call(
                POLICY_COMMIT_METHOD,
                {
                    "transaction_id": transaction_id,
                    "handle": handle,
                    "expected": expected.to_wire(),
                },
            )
        )

    async def retire(self, transaction_id: str, handle: str) -> None:
        await self._call(
            POLICY_RETIRE_METHOD, {"transaction_id": transaction_id, "handle": handle}
        )

    async def abort(self, transaction_id: str) -> None:
        await self._call(
            POLICY_ABORT_METHOD, {"transaction_id": transaction_id, "handle": None}
        )

    async def _call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        try:
            result = await self._client.policy_request(method, params)
        except PolicyTreeError:
            raise
        except Exception as exc:
            translated = translate_wire_error(exc)
            if translated is None:
                raise
            raise translated from exc
        if not isinstance(result, Mapping):
            raise PolicyNodePhaseError(
                f"The remote policy node answered {method} without a result"
            )
        return dict(result)


def translate_wire_error(exc: BaseException) -> PolicyTreeError | None:
    """Map a typed wire failure back onto this process's policy error."""
    code = str(getattr(exc, "code", type(exc).__name__))
    message = str(exc)
    match code:
        case PolicyRevisionStaleError.__name__:
            return PolicyRevisionStaleError(message)
        case PolicyTreeBusyError.__name__:
            return PolicyTreeBusyError(message)
        case PolicyNodePhaseError.__name__:
            return PolicyNodePhaseError(message)
        case PolicyTreeJournalError.__name__:
            return PolicyTreeJournalError(message)
        case _:
            return None


class PolicyTreeTransport(Protocol):
    """Selects which topology one policy-tree node belongs to."""

    def node(self, target: Any, *, node_id: str) -> PolicyNodePort: ...


class InProcessPolicyTree:
    """The in-process cut: every node's loop lives in this process."""

    def node(self, target: Any, *, node_id: str) -> PolicyNodePort:
        return InProcessPolicyNode(node_id, target)


class ProcessPolicyTree:
    """The process topology: every node is served over the agent wire."""

    def node(self, target: Any, *, node_id: str) -> PolicyNodePort:
        return ProcessPolicyNode(node_id, target)


def policy_tree_transport(
    selection: TransportSelection | Mapping[str, str],
) -> PolicyTreeTransport:
    """Select the topology for a policy-tree transaction.

    The production caller passes the SESSION-LEVEL resolved selection: the
    topology is already a property of the session, so no site re-resolves it
    from the environment — a default-empty environ would silently answer
    in-process for a stdio session, which is exactly the unwired state this
    seam exists to end. ``CHARTREUX_TRANSPORT=in_process`` returns the same
    in-process topology as the programmatic default; both run the same
    transaction code. A raw selector mapping stays accepted for the
    resolution tests, where no session selection exists yet.
    """
    resolved = (
        selection
        if isinstance(selection, TransportSelection)
        else resolve_transport_selection(environ=selection)
    )
    if resolved.kind == IN_PROCESS_TRANSPORT:
        return InProcessPolicyTree()
    return ProcessPolicyTree()


class PolicyTreeTransaction:
    """Drive one policy tree through reserve, prepare, commit and retire.

    Every phase is journaled before the next starts, so a transaction that
    crashed is reconstructed from the record with :meth:`recover` and resumed
    explicitly -- no phase is ever inferred from memory.
    """

    def __init__(
        self,
        transaction_id: str,
        *,
        nodes: Sequence[PolicyNodePort],
        journal: PolicyTreeJournal,
        change: PolicyChange,
    ) -> None:
        if not transaction_id:
            raise ValueError("a policy transaction carries an id")
        if not nodes:
            raise ValueError("a policy transaction carries at least one node")
        node_ids = [node.node_id for node in nodes]
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("policy transaction node ids must be unique")
        self.transaction_id = transaction_id
        self._nodes = tuple(nodes)
        self._journal = journal
        self._change = change

    @property
    def revision(self) -> str:
        """This transaction's accepted tree revision (from the journal)."""
        record = self._require_record()
        return record.revision

    async def run(self, *, expected_revision: str | None = None) -> PolicyTreeRecord:
        """Reserve, prepare, commit and retire one change in one call."""
        if self._journal.read() is not None:
            raise PolicyTreeJournalError(
                f"Transaction {self.transaction_id} is already journaled"
            )
        await self.reserve()
        await self.prepare()
        await self.commit(expected_revision=expected_revision)
        return await self.retire()

    async def reserve(self) -> PolicyTreeRecord:
        """Capture every node's revision, then reserve the tree durably."""
        if self._journal.read() is not None:
            raise PolicyTreeJournalError(
                f"Transaction {self.transaction_id} is already journaled"
            )
        entries = tuple([
            PolicyNodeJournalEntry(node.node_id, await node.revision())
            for node in self._nodes
        ])
        record = PolicyTreeRecord(
            transaction_id=self.transaction_id,
            phase=PolicyTreePhase.RESERVED,
            revision=_tree_revision(entries),
            change=self._change.to_wire(),
            nodes=entries,
        )
        self._journal.write(record)
        reserved: list[PolicyNodePort] = []
        try:
            for entry, node in self._node_pairs(record):
                observed = await node.reserve(self.transaction_id)
                if observed != entry.revision:
                    raise PolicyRevisionStaleError(
                        f"Policy reserve refused for {entry.node_id}: "
                        "the revision moved during reservation"
                    )
                reserved.append(node)
        except BaseException:
            for node in reserved:
                with suppress(PolicyTreeError):
                    await node.abort(self.transaction_id)
            self._journal.write(replace(record, phase=PolicyTreePhase.ABORTED))
            raise
        return record

    async def prepare(self) -> PolicyTreeRecord:
        """Prepare every node; a failure aborts the whole tree, publishing nothing."""
        record = self._require_record()
        if record.phase not in {PolicyTreePhase.RESERVED, PolicyTreePhase.PREPARED}:
            raise PolicyNodePhaseError(
                f"Cannot prepare a {record.phase.value} policy transaction"
            )
        handles = {
            entry.node_id: entry.handle
            for entry in record.nodes
            if entry.handle is not None
        }
        try:
            for entry, node in self._node_pairs(record):
                if entry.handle is not None:
                    continue
                handles[entry.node_id] = await node.prepare(
                    self.transaction_id, self._change, entry.revision
                )
        except BaseException:
            for _entry, node in self._node_pairs(record):
                with suppress(PolicyTreeError):
                    await node.abort(self.transaction_id)
            self._journal.write(
                replace(
                    record,
                    phase=PolicyTreePhase.ABORTED,
                    nodes=tuple(
                        replace(entry, handle=handles.get(entry.node_id))
                        for entry in record.nodes
                    ),
                )
            )
            raise
        record = replace(
            record,
            phase=PolicyTreePhase.PREPARED,
            nodes=tuple(
                replace(entry, handle=handles[entry.node_id]) for entry in record.nodes
            ),
        )
        self._journal.write(record)
        return record

    async def commit(self, *, expected_revision: str | None = None) -> PolicyTreeRecord:
        """Publish every node under the revision lock; stale commits refuse typed."""
        record = self._require_record()
        if record.phase is not PolicyTreePhase.PREPARED:
            raise PolicyNodePhaseError(
                f"Cannot commit a {record.phase.value} policy transaction"
            )
        if expected_revision is not None and expected_revision != record.revision:
            raise PolicyRevisionStaleError(
                "Policy commit refused: the accepted tree revision is stale"
            )
        committed: dict[str, PolicyNodeRevision] = {}
        try:
            for entry, node in self._node_pairs(record):
                if entry.handle is None:
                    raise PolicyTreeJournalError(
                        f"Policy node {entry.node_id} carries no prepared handle"
                    )
                committed[entry.node_id] = await node.commit(
                    self.transaction_id, entry.handle, entry.revision
                )
        except BaseException:
            if committed:
                # A partial commit is retired-pending, never rolled back: the
                # journal records it so reads fail closed until retirement.
                self._journal.write(
                    replace(
                        record,
                        phase=PolicyTreePhase.COMMITTED,
                        nodes=tuple(
                            replace(
                                entry, committed_revision=committed.get(entry.node_id)
                            )
                            for entry in record.nodes
                        ),
                    )
                )
            raise
        record = replace(
            record,
            phase=PolicyTreePhase.COMMITTED,
            nodes=tuple(
                replace(entry, committed_revision=committed[entry.node_id])
                for entry in record.nodes
            ),
        )
        self._journal.write(record)
        return record

    async def retire(self) -> PolicyTreeRecord:
        """Retire every node's previous authority; idempotent per node."""
        record = self._require_record()
        if record.phase is not PolicyTreePhase.COMMITTED:
            raise PolicyNodePhaseError(
                f"Cannot retire a {record.phase.value} policy transaction"
            )
        for entry, node in self._node_pairs(record):
            if entry.handle is not None:
                await node.retire(self.transaction_id, entry.handle)
        record = replace(record, phase=PolicyTreePhase.RETIRED)
        self._journal.write(record)
        return record

    async def abort(self) -> PolicyTreeRecord:
        """Release a reserved or prepared tree; a committed tree cannot abort."""
        record = self._require_record()
        if record.phase not in {PolicyTreePhase.RESERVED, PolicyTreePhase.PREPARED}:
            raise PolicyNodePhaseError(
                f"Cannot abort a {record.phase.value} policy transaction"
            )
        for _entry, node in self._node_pairs(record):
            with suppress(PolicyTreeError):
                await node.abort(self.transaction_id)
        record = replace(record, phase=PolicyTreePhase.ABORTED)
        self._journal.write(record)
        return record

    def recover(self) -> PolicyTreeRecovery:
        """Adjudicate a journaled transaction without touching any node."""
        record = self._require_record()
        return PolicyTreeRecovery(
            transaction_id=record.transaction_id,
            phase=record.phase,
            decision=_recovery_decision(record.phase),
            record=record,
        )

    async def reprepare(self) -> PolicyTreeRecord:
        """Recover a crashed transaction: re-prepare, then commit and retire."""
        record = self.recover()
        if record.decision is PolicyTreeRecoveryDecision.RETIRED_PENDING:
            return await self.resume_retire()
        if record.decision in {PolicyTreeRecoveryDecision.COMPLETE}:
            return record.record
        if record.decision is not PolicyTreeRecoveryDecision.REPREPARE_OR_ABORT:
            raise PolicyNodePhaseError(
                f"Cannot re-prepare a {record.phase.value} policy transaction"
            )
        if record.phase is PolicyTreePhase.RESERVED:
            await self.prepare()
        await self.commit()
        return await self.retire()

    async def resume_retire(self) -> PolicyTreeRecord:
        """Finish a committed transaction whose previous tree is retired-pending."""
        record = self._require_record()
        if record.phase is PolicyTreePhase.RETIRED:
            return record
        if record.phase is not PolicyTreePhase.COMMITTED:
            raise PolicyNodePhaseError(
                f"Cannot finish retirement of a {record.phase.value} transaction"
            )
        return await self.retire()

    def require_settled(self) -> PolicyTreeRecord:
        """Read the journal fail-closed: a retired-pending tree is never served."""
        record = self._require_record()
        if record.phase is PolicyTreePhase.COMMITTED:
            raise PolicyTreeRetiredPendingError(record.transaction_id)
        return record

    def _require_record(self) -> PolicyTreeRecord:
        record = self._journal.read()
        if record is None:
            raise PolicyTreeJournalError(
                f"Transaction {self.transaction_id} is not journaled"
            )
        if record.transaction_id != self.transaction_id:
            raise PolicyTreeJournalError("The journal holds another transaction")
        return record

    def _node_pairs(
        self, record: PolicyTreeRecord
    ) -> list[tuple[PolicyNodeJournalEntry, PolicyNodePort]]:
        nodes = {node.node_id: node for node in self._nodes}
        journaled = {entry.node_id for entry in record.nodes}
        if set(nodes) != journaled:
            raise PolicyTreeJournalError("The policy journal and the node set disagree")
        return [(entry, nodes[entry.node_id]) for entry in record.nodes]


def _tree_revision(entries: Sequence[PolicyNodeJournalEntry]) -> str:
    return "|".join(
        f"{entry.node_id}={entry.revision.tree_revision}@{entry.revision.policy_revision}"
        for entry in entries
    )


def _recovery_decision(phase: PolicyTreePhase) -> PolicyTreeRecoveryDecision:
    match phase:
        case PolicyTreePhase.RESERVED | PolicyTreePhase.PREPARED:
            return PolicyTreeRecoveryDecision.REPREPARE_OR_ABORT
        case PolicyTreePhase.COMMITTED:
            return PolicyTreeRecoveryDecision.RETIRED_PENDING
        case PolicyTreePhase.RETIRED:
            return PolicyTreeRecoveryDecision.COMPLETE
        case PolicyTreePhase.ABORTED:
            return PolicyTreeRecoveryDecision.ABORTED


def _required_str(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise PolicyNodePhaseError(f"a policy frame carries no {label}")
    return value


__all__ = [
    "FilePolicyTreeJournal",
    "InProcessPolicyNode",
    "InProcessPolicyTree",
    "ManifestPolicyTreeJournal",
    "PolicyChange",
    "PolicyNodeJournalEntry",
    "PolicyNodePhaseError",
    "PolicyNodePort",
    "PolicyNodeRevision",
    "PolicyNodeService",
    "PolicyRevisionStaleError",
    "PolicyTreeBusyError",
    "PolicyTreeError",
    "PolicyTreeJournal",
    "PolicyTreeJournalError",
    "PolicyTreePhase",
    "PolicyTreeRecord",
    "PolicyTreeRecovery",
    "PolicyTreeRecoveryDecision",
    "PolicyTreeRetiredPendingError",
    "PolicyTreeTransaction",
    "PolicyTreeTransport",
    "PolicyWireClient",
    "ProcessPolicyNode",
    "ProcessPolicyTree",
    "policy_tree_transport",
    "translate_wire_error",
]
