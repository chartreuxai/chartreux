from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Iterator
from contextlib import AbstractAsyncContextManager, contextmanager, suppress
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
import fnmatch
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast
from uuid import uuid4

from chartreux.app_server._subagent_records import AgentRecord
from chartreux.app_server._subagent_recovery import _effective_eligibility
from chartreux.app_server._subagent_transcript import ChildTombstoneRecord
from chartreux.app_server.subagent_ports import (
    CHILD_HANDOFF_PAYLOAD_KEY,
    CHILD_TEARDOWN_REASON,
    ChildTeardownOutcome,
    SpawnedSubagent,
    SubagentSpawnPort,
    child_result_store,
    child_spawn_spec,
)
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.agents.identity_launch import (
    parse_model_shorthand,
    resolve_identity_launch,
    validate_launch_purpose_binding,
)
from chartreux.core.agents.launch import FrozenPersona, LaunchCandidate, resolve_launch
from chartreux.core.agents.manager import (
    identity_denylist_keys,
    requested_identity_reference,
    resolve_identity_candidate,
)
from chartreux.core.compaction.context import select_model_context
from chartreux.core.dispatch.renderer import session_model_alias
from chartreux.core.dispatch.route_models import ReplacementCapability
from chartreux.core.model_catalog.resolver import ModelResolutionError, resolver_for
from chartreux.core.recovery_models import (
    FailureClass,
    FailureEnvelope,
    PackageTier,
    RecoveryEligibility,
    canonical_endpoint_identity,
)
from chartreux.core.recovery_override import (
    CooldownOverrideLedger,
    OverrideDenial,
    OverridePermit,
)
from chartreux.core.recovery_policy import (
    ReplacementCandidate,
    ReplacementDecision,
    ReplacementPolicyError,
    candidates_from_catalog,
    route_recovery_facts,
    validate_replacement_selection,
)
from chartreux.core.session.saved_sessions import delete_saved_session
from chartreux.core.session_types import ChildSessionLink, CommittedModelIdentity
from chartreux.core.subagents import (
    InvalidLaunchModelError,
    LaunchConfig,
    LaunchConfigError,
    ReconcileOutcome,
    RunStatus,
    TaskArgs,
)
from chartreux.core.tools.base import InvokeContext, ToolError
from chartreux.core.tools.manager import ToolManager
from chartreux.core.tools.models import ToolPermission, ToolPermissionError
from chartreux.observability.logging import logger

if TYPE_CHECKING:
    from chartreux.app_server._runtime import AgentRuntimeFactory
    from chartreux.app_server._sessions import SessionRuntime

_AGENT_ID_HIGH_WATER_MARK_KEY = "_app_server_agent_id_high_water_mark"

PUBLICATION_ROLLBACK_REASON = "publication-rolled-back"
"""Tombstone reason: the publication transaction rolled its own failure back."""

PUBLICATION_REJECTED_REASON = "publication-rejected"
"""Tombstone reason: a published child was rejected before it was accepted."""


def new_publication_attempt_id() -> str:
    """Mint the coordinator-side identity of one child publication attempt."""
    return f"attempt-{uuid4().hex}"


def retire_child_publication(
    child: AgentLoop, *, attempt_id: str | None, reason: str
) -> bool:
    """Durably retract an abandoned child publication; True when written.

    The tombstone is the retraction that outlives a lost delete: the child
    transcript scan reports the attempt as retracted (and filters it out), so
    its persisted records can never resurrect as an adoptable child. A child
    with no durable transcript (logging disabled) has nothing to retract.
    """
    store = child_result_store(child)
    if store is None:
        return False
    store.record_tombstone(
        ChildTombstoneRecord(
            session_id=child.session_id,
            reason=reason,
            created_at=datetime.now(UTC).isoformat(),
            attempt_id=attempt_id,
            agent_id=child.launch_identity,
        )
    )
    return True


def published_attempt_id(
    child: AgentLoop, parent: SessionRuntime, tool_call_id: str
) -> str | None:
    """The attempt id a published child was recorded under, when known.

    The child's own first write is the primary bridge; the parent link is the
    fallback, so a rollback retracts even when only one of the two survived.
    A pre-v0.7.8 record carries neither and retracts through its session id.
    """
    metadata = child.session_logger.session_metadata
    if metadata is not None and metadata.attempt_id is not None:
        return metadata.attempt_id
    parent_metadata = parent.agent_loop.session_logger.session_metadata
    if parent_metadata is None:
        return None
    return next(
        (
            link.attempt_id
            for link in parent_metadata.child_sessions
            if link.session_id == child.session_id and link.tool_call_id == tool_call_id
        ),
        None,
    )


class LaunchStateView(Protocol):
    """Narrow typed view over the registry-owned launch state.

    Dynamic access: every member is read through the view at call time, so
    replaceable scalars (root identity, generation, admission flags, the
    runtime factory) and the sequence counter are never snapshotted at
    construction — registry state mutated or rebound after the
    collaborator is built stays visible, and instance patches of the
    registry delegates still shadow.
    """

    _root: SessionRuntime | None
    _generation_identity: tuple[str, int] | None
    _admission_closed: bool
    _draining_children: bool
    _policy_reserved: bool
    _creating_children: int
    _children: dict[str, SessionRuntime]
    _child_links: dict[str, tuple[SessionRuntime, str]]
    _next_agent_seq: int
    _runtime_factory: AgentRuntimeFactory
    _agent_records: dict[str, AgentRecord]
    _package_replacements: dict[str, int]
    _registry_lock: asyncio.Lock
    # Item 1 child seam: the process-backed spawn adapter this run selected
    # (None keeps the in-process publication path byte for byte), and the
    # coordinator-side handles of the children that live in their own process.
    _subagent_spawn: SubagentSpawnPort | None
    _process_children: dict[str, ProcessChildSession]

    def _require_policy_unreserved(self) -> None: ...

    def _resolve_launch_candidate(
        self,
        parent: SessionRuntime,
        args: TaskArgs,
        *,
        runtime: SessionRuntime | None = None,
        identity: str | None = None,
    ) -> LaunchCandidate: ...

    def _build_child_runtime(self, child: AgentLoop) -> SessionRuntime: ...

    def _discard_child(self, child: AgentLoop) -> Awaitable[None]: ...

    def _child_publication(
        self, runtime: SessionRuntime
    ) -> AbstractAsyncContextManager[None]: ...

    def _invalidate_recovery_locked(self, record: AgentRecord | None) -> None: ...


@dataclass
class ChildPublicationAttempt:
    """Per-invocation state for one child publication (ASC area 1).

    Holds the parent runtime and captured identity, the resolved launch
    candidate and tool-call id, the coordinator-minted attempt id, the
    constructed child/runtime once they exist, and the transaction flags the
    rollback region reads. The registry-owned child maps stay canonical; the
    attempt carries only the publication transaction's local state.
    """

    parent: SessionRuntime
    identity: tuple[str, int]
    candidate: LaunchCandidate
    tool_call_id: str
    attempt_id: str
    child: AgentLoop | None = None
    runtime: SessionRuntime | None = None
    link_recorded: bool = False
    projection_started: bool = False
    persisted: bool = False

    async def rollback(self, view: LaunchStateView) -> None:
        """Remove every durable/runtime trace of an unaccepted publication.

        Replaces the local transaction flags of the former registry method
        without widening the cleanup region or changing exception
        suppression. A publication whose first durable write landed is
        retracted with a tombstone before removal: if the delete is lost or
        the process dies mid-rollback, the retraction still gates the child
        transcript scan. A publication that never persisted has nothing to
        retract and is deleted.
        """
        child = self.child
        runtime = self.runtime
        assert child is not None and runtime is not None
        session_id = child.session_id
        view._children.pop(session_id, None)
        view._child_links.pop(session_id, None)
        if self.projection_started:
            with suppress(Exception):
                await self.parent.turns.unlink_subagent(self.tool_call_id, session_id)
        if self.link_recorded:
            with suppress(Exception):
                await self.parent.agent_loop.forget_child_session(
                    session_id, self.tool_call_id
                )
        with suppress(Exception):
            await runtime.close()
        if self.persisted:
            with suppress(Exception):
                retire_child_publication(
                    child,
                    attempt_id=self.attempt_id,
                    reason=PUBLICATION_ROLLBACK_REASON,
                )
        with suppress(Exception):
            await delete_saved_session(session_id, child.config.session_logging)


class ChildPublicationStage(StrEnum):
    """The stage one aggregate child publication attempt reached.

    The aggregate outcome names the stage that failed, so a caller never has
    to infer it from a nested exception type. PERMIT is its own stage because
    a process-backed child cannot consume a C4 permit binding in v0.7.8: the
    request is refused there and the reservation is released, never stranded.
    """

    ADMISSION = "admission"
    HANDOFF = "handoff"
    PERMIT = "permit"
    SPAWN = "spawn"
    PERSISTENCE = "persistence"
    LINKAGE = "linkage"
    PROJECTION = "projection"


class ChildPublicationError(RuntimeError):
    """One process-backed child publication attempt's stage-named failure.

    ``retraction`` reports what the conservative teardown established after the
    failure: a reaped or tombstoned attempt, the typed INCONCLUSIVE class when
    the attempt could not be settled inside the bounded rollback wait, or None
    when nothing was spawned yet. A caller must never read this error as a
    claimed rollback.
    """

    def __init__(
        self,
        stage: ChildPublicationStage,
        attempt_id: str,
        *,
        retraction: ChildTeardownOutcome | None = None,
        detail: str = "",
    ) -> None:
        self.stage = stage
        self.attempt_id = attempt_id
        self.retraction = retraction
        message = (
            f"Child publication attempt {attempt_id} failed at stage {stage.value}"
        )
        if detail:
            message += f": {detail}"
        if retraction is not None:
            message += f" (teardown: {retraction.value})"
        super().__init__(message)


@dataclass(slots=True)
class ProcessChildSession:
    """A coordinator-side handle to a child whose loop lives in its own process.

    A RETAINED child keeps this handle — and therefore its process — across
    re-tasks; a non-retained child's handle is closed at its completion. The
    registry owns the map, so the handle is never the source of child truth.
    """

    agent_id: str
    identity: str
    session_id: str
    attempt_id: str
    spawned: SpawnedSubagent
    child_dir: str | None = None
    retained: bool = False
    closed: bool = False

    @property
    def alive(self) -> bool:
        return not self.closed and bool(getattr(self.spawned, "alive", True))

    async def close(self) -> None:
        """Reap this child's process; a repeated close is a no-op."""
        if self.closed:
            return
        self.closed = True
        with suppress(Exception):
            await self.spawned.close()


async def record_process_child_link(
    parent: AgentLoop,
    *,
    session_id: str,
    session_dir: str | None,
    tool_call_id: str,
    agent: str,
    attempt_id: str,
) -> ChildSessionLink:
    """Record the parent link of a child whose loop lives in another process.

    The in-process linkage writes the parent's transcript as well; this path
    changes no parent message, so it persists the link itself — the same
    append/rollback shape, keyed on the tool call so a repeated link is
    idempotent and a conflicting one fails loudly.
    """
    parent_dir = parent.session_logger.session_dir
    relative_path = (
        str(Path(session_dir).relative_to(parent_dir))
        if parent_dir is not None and session_dir is not None
        else None
    )
    link = ChildSessionLink(
        session_id=session_id,
        tool_call_id=tool_call_id,
        agent=agent,
        relative_path=relative_path,
        attempt_id=attempt_id,
    )
    metadata = parent.session_logger.session_metadata
    if metadata is None:
        return link
    existing = next(
        (item for item in metadata.child_sessions if item.tool_call_id == tool_call_id),
        None,
    )
    if existing is not None:
        if existing != link:
            raise RuntimeError(
                f"Tool call {tool_call_id} is already linked to a child session"
            )
        return existing
    metadata.child_sessions.append(link)
    try:
        await parent.session_logger.persist_child_sessions()
    except BaseException:
        metadata.child_sessions.remove(link)
        raise
    return link


class ProcessChildPublication:
    """The aggregate child publication transaction over the process boundary.

    One attempt runs as a SINGLE typed outcome keyed by the coordinator-minted
    attempt id: the admission fence, the working-set handoff, permit binding,
    persistence, parent linkage and projection happen in order, and a failure
    names the stage that failed. A failure after the spawn is settled
    CONSERVATIVELY — the coordinator never claims a rollback it cannot observe:
    the prepared child is cancelled, its atomic rollback gets a bounded wait,
    and the durable records it left are retracted (or the typed INCONCLUSIVE
    class is reported when the attempt cannot be settled inside that bound).

    A RETAINED child is not respawned by a re-task: the existing handle is
    reused, so its process survives across tasks; a non-retained child is
    spawned on every launch and exits at completion.
    """

    def __init__(
        self,
        admission: ChildLaunchAdmission,
        spawn: SubagentSpawnPort,
        *,
        rollback_wait: float,
    ) -> None:
        self._admission = admission
        self._spawn = spawn
        self._rollback_wait = rollback_wait

    async def publish(
        self,
        attempt: ChildPublicationAttempt,
        *,
        retained: bool,
        mint: OverrideMint | None = None,
        reuse: ProcessChildSession | None = None,
    ) -> ProcessChildSession:
        """Run one aggregate publication attempt and answer its handle."""
        stage = ChildPublicationStage.ADMISSION
        spawned: SpawnedSubagent | None = None
        try:
            self._admission.validate_admission(attempt.parent, attempt.identity)
            stage = ChildPublicationStage.HANDOFF
            agent_name = attempt.candidate.identity.replace(":", "-")
            if reuse is not None and reuse.alive:
                # A retained child keeps its process: the re-task only links
                # the new tool call to the process that already exists.
                await self._link(attempt, reuse, agent_name)
                return reuse
            root_session_dir = attempt.parent.agent_loop.session_logger.session_dir
            if root_session_dir is None:
                raise RuntimeError(
                    "a process-backed child needs the root session directory: "
                    "its transcript belongs under <root>/agents"
                )
            payload = self._handoff_payload(attempt)
            stage = ChildPublicationStage.PERMIT
            if mint is not None:
                raise RuntimeError(
                    "a process-backed child cannot consume a cooldown-override "
                    "permit in this release; the reservation was released"
                )
            spec = child_spawn_spec(
                agent_id=f"{agent_name}-{attempt.attempt_id}",
                parent_session_id=attempt.parent.agent_loop.session_id,
                root_session_dir=root_session_dir,
                agent_name=agent_name,
                payload=payload,
                attempt_id=attempt.attempt_id,
            )
            stage = ChildPublicationStage.SPAWN
            spawned = await self._spawn.spawn(spec)
            stage = ChildPublicationStage.PERSISTENCE
            session_id = getattr(spawned, "session_id", None)
            if not isinstance(session_id, str) or not session_id:
                raise RuntimeError(
                    "the spawned child announced no session id, so its first "
                    "durable record cannot be attributed to this attempt"
                )
            handle = ProcessChildSession(
                agent_id=getattr(spawned, "agent_id", spec.agent_id),
                identity=attempt.candidate.identity,
                session_id=session_id,
                attempt_id=attempt.attempt_id,
                spawned=spawned,
                child_dir=getattr(spawned, "child_session_dir", None),
                retained=retained,
            )
            attempt.persisted = True
            stage = ChildPublicationStage.LINKAGE
            await self._link(attempt, handle, agent_name)
            return handle
        except asyncio.CancelledError:
            if spawned is not None:
                await self._settle(attempt, spawned)
            raise
        except BaseException as exc:
            # ``_link`` raises its own stage-named error; the aggregate reply
            # must keep that stage instead of relabelling it as the caller's.
            failed = exc.stage if isinstance(exc, ChildPublicationError) else stage
            raise await self._fail(attempt, failed, exc, spawned=spawned) from exc

    async def _link(
        self, attempt: ChildPublicationAttempt, handle: ProcessChildSession, agent: str
    ) -> None:
        """Record the parent link and project it, naming the failing stage."""
        try:
            await record_process_child_link(
                attempt.parent.agent_loop,
                session_id=handle.session_id,
                session_dir=handle.child_dir,
                tool_call_id=attempt.tool_call_id,
                agent=agent,
                attempt_id=handle.attempt_id,
            )
            attempt.link_recorded = True
        except BaseException as exc:
            raise ChildPublicationError(
                ChildPublicationStage.LINKAGE,
                attempt.attempt_id,
                detail=f"{type(exc).__name__}: {exc}",
            ) from exc
        try:
            attempt.projection_started = True
            if attempt.parent.turns._projector is not None:
                await attempt.parent.turns.link_subagent(
                    attempt.tool_call_id, handle.session_id
                )
        except BaseException as exc:
            raise ChildPublicationError(
                ChildPublicationStage.PROJECTION,
                attempt.attempt_id,
                detail=f"{type(exc).__name__}: {exc}",
            ) from exc

    def _handoff_payload(self, attempt: ChildPublicationAttempt) -> dict[str, Any]:
        """The spawn handoff: the parent's rendered, redacted working set."""
        message = attempt.parent.agent_loop.file_state_handoff_message()
        if message is None or not message.content:
            return {}
        return {CHILD_HANDOFF_PAYLOAD_KEY: message.content}

    async def _fail(
        self,
        attempt: ChildPublicationAttempt,
        stage: ChildPublicationStage,
        exc: BaseException,
        *,
        spawned: SpawnedSubagent | None,
    ) -> ChildPublicationError:
        """Settle a failed attempt and return its stage-named aggregate error."""
        retraction = (
            await self._settle(attempt, spawned) if spawned is not None else None
        )
        return ChildPublicationError(
            stage,
            attempt.attempt_id,
            retraction=retraction,
            detail=f"{type(exc).__name__}: {exc}",
        )

    async def _settle(
        self, attempt: ChildPublicationAttempt, spawned: SpawnedSubagent
    ) -> ChildTeardownOutcome:
        """Retract this attempt's runtime and durable traces; never claim success.

        The link and projection this attempt already published are dropped
        first, so a retracted attempt leaves no adoptable parent-side trace;
        then the child is cancelled, given the bounded wait for its own atomic
        rollback, and its durable records are retracted.
        """
        session_id = getattr(spawned, "session_id", None)
        if attempt.projection_started and session_id is not None:
            with suppress(Exception):
                await attempt.parent.turns.unlink_subagent(
                    attempt.tool_call_id, session_id
                )
        if attempt.link_recorded and session_id is not None:
            with suppress(Exception):
                await attempt.parent.agent_loop.forget_child_session(
                    session_id, attempt.tool_call_id
                )
        attempt.link_recorded = False
        attempt.projection_started = False
        from chartreux.app_server.child_agent_process import (
            SpawnedAttempt,
            teardown_spawned_attempt,
        )

        report = await teardown_spawned_attempt(
            cast(SpawnedAttempt, spawned),
            attempt_id=attempt.attempt_id,
            reason=CHILD_TEARDOWN_REASON,
            wait_seconds=self._rollback_wait,
        )
        return report.outcome


@dataclass(frozen=True, slots=True)
class OverrideMint:
    """The C4 mint: one endpoint-bound permit plus its binding ledger."""

    permit: OverridePermit
    ledger: CooldownOverrideLedger
    endpoint: str


def bind_override_to_loop(loop: AgentLoop, mint: OverrideMint) -> None:
    """Bind the minted permit to the child's NEXT logical model call (C3).

    The coordinator slot is C3's binding channel: the child's first
    completion attempt enumerates its candidates THROUGH the permit (the
    cooled forced deployment is eligible), the forced candidate carries the
    binding into the gateway, the actual transport start consumes the
    permit, and the slot clears — exactly one forced attempt per mint.
    """
    loop._completion_coordinator._active_override = (mint.permit, mint.ledger)


def release_bound_override(loop: AgentLoop) -> None:
    """Release the loop's bound permit when its run never started (C4 net).

    A pre-start discard (rollback of the creation or of the admission) must
    not strand the endpoint's single in-flight slot: release the still-bound
    permit. Release is typed and idempotent — a permit already consumed at
    its transport start is no longer outstanding and release is a no-op —
    so the net is safe on every rollback path.
    """
    binding = loop._completion_coordinator._active_override
    if binding is None:
        return
    permit, ledger = binding
    loop._completion_coordinator._active_override = None
    ledger.release(permit)


def override_launch_diagnostics(loop: AgentLoop) -> dict[str, object] | None:
    """The bounded C4 un-stick diagnostics from the loop's bound permit.

    Read pre-start (before the child's first model call consumes the
    binding): the selected deployment, the cooldown exclusions the permit
    bypasses, the start allowance remaining after this attempt consumes, and
    the delay until the NEXT permitted start. None when the loop carries no
    override binding (ordinary and automatic launches never do — F6).
    """
    binding = loop._completion_coordinator._active_override
    if binding is None:
        return None
    permit, ledger = binding
    registry = loop.config_orchestrator.availability_registry
    # episode_status canonicalizes its input, so it needs the RAW api_base;
    # recover it from the catalog by matching the permit's canonical identity.
    snapshot = loop.config.catalog_snapshot
    raw_endpoint = (
        next(
            (
                provider.api_base
                for provider in snapshot.catalog.providers.values()
                if canonical_endpoint_identity(provider.api_base) == permit.endpoint
            ),
            None,
        )
        if snapshot is not None
        else None
    )
    status = ledger.episode_status(raw_endpoint)
    exclusions: list[str] = []
    committed = loop.committed_model
    if committed is not None:
        if registry.endpoint_correlation_active(permit.endpoint):
            exclusions.append("endpoint-cooldown")
        if not registry.is_available(committed.base_model, committed.provider):
            exclusions.append("deployment-cooldown")
    remaining = (status.starts_remaining - 1) if status is not None else 0
    next_start_delay: float | None = None
    if status is not None and remaining > 0:
        # The next start after this attempt's own: spacing since this
        # attempt's reserve and the pressure bound's window for start
        # number starts_used + 2.
        next_time = max(
            permit.issued_at + ledger.start_spacing_seconds,
            status.opened_at + (status.starts_used + 1) * status.start_spacing_seconds,
        )
        next_start_delay = max(0.0, next_time - ledger._clock())
    diagnostics: dict[str, object] = {
        "endpoint": permit.endpoint,
        "exclusions_bypassed": exclusions,
        "remaining_starts": max(0, remaining),
        "next_start_delay_seconds": next_start_delay,
    }
    if committed is not None:
        diagnostics["base_model"] = committed.base_model
        diagnostics["provider"] = committed.provider
        diagnostics["wire_name"] = committed.wire_name
    return diagnostics


def _override_denial_message(
    denial: OverrideDenial | None, ledger: CooldownOverrideLedger, endpoint: str | None
) -> str:
    """The bounded un-stick denial message, diagnostics included."""
    status = ledger.episode_status(endpoint) if endpoint is not None else None
    if denial is OverrideDenial.NO_ACTIVE_EPISODE or status is None:
        detail = (
            "no active endpoint episode: the un-stick override applies only "
            "within a measured cooldown episode"
        )
    elif denial is OverrideDenial.EPISODE_EXHAUSTED:
        detail = "the episode's forced-start budget is exhausted"
    elif denial is OverrideDenial.ATTEMPT_IN_FLIGHT:
        detail = "a forced attempt is already in flight for this endpoint"
    elif denial is OverrideDenial.SPACING_EXCEEDED:
        detail = "the forced-start spacing floor has not elapsed yet"
    else:
        detail = "the override ledger denied the reservation"
    message = f"Cooldown override denied for endpoint {endpoint!r}: {detail}."
    if status is not None:
        delay = status.next_start_delay_seconds(ledger._clock())
        message += f" Starts used {status.starts_used}/{status.max_starts}; " + (
            f"next permitted start in ~{delay:.0f}s."
            if delay is not None
            else "no further start is admitted in this episode."
        )
    return message


class ChildLaunchAdmission:
    """Launch admission, publication, and rollback for child sessions.

    Narrow collaborator of the session runtime registry (ASC area 1).
    Identity validation and candidate resolution (17.1a) are pure functions
    of the parent runtime and the launch arguments: they read no registry
    state, take no locks, and hold no state of their own. Sequence
    allocation and restoration, admission validation, the creation fence,
    registered-child publication, and created-child rollback (17.1b) run
    over the registry-owned typed view, which also supplies the runtime
    construction, child-publication context, and discard dependencies with
    call-time dynamic access. Canonical child maps, admission flags, and
    the sequence counter stay registry-owned (§6.7); only the existing
    registry locks are used; no private collaborator locks, tasks, queues,
    or relay layers. The registry keeps thin same-name delegates so
    existing instance patches and direct calls keep working.
    """

    def __init__(self, view: LaunchStateView) -> None:
        self._view = view

    def require_identity_allowed(self, parent: SessionRuntime, identity: str) -> None:
        """Apply effective Task admission to the RESOLVED launch identity.

        The denylist keys on the same identity surface the task tool enforces
        (design §12): a resolved identity reference — including route/shorthand
        selections and a retained instance's binding — checked after final
        resolution and before any child resource is constructed. A
        ``user:``-qualified custom identity also matches its bare name.
        """
        manager = parent.agent_loop.tool_manager
        if isinstance(manager, ToolManager) and "task" not in manager.available_tools:
            raise ToolPermissionError("Task tool is disabled")
        config = manager.get_tool_config("task")
        if (
            isinstance(manager, ToolManager)
            and config.permission == ToolPermission.NEVER
        ):
            raise ToolPermissionError("Task tool permission is never")
        if any(
            fnmatch.fnmatch(key, pattern)
            for key in identity_denylist_keys(identity)
            for pattern in config.denylist
        ):
            raise ToolPermissionError(f"Task denied for launch identity: {identity}")

    def resolve_candidate(
        self,
        parent: SessionRuntime,
        args: TaskArgs,
        *,
        runtime: SessionRuntime | None = None,
        identity: str | None = None,
        override_permit: OverridePermit | None = None,
        override_ledger: CooldownOverrideLedger | None = None,
    ) -> LaunchCandidate:
        """Resolve the launch candidate for a fresh or reused child.

        A FRESH launch (no retained runtime) resolves through the canonical
        grammar (package 8's resource-free resolution): the identity, the
        model expression, the effective thinking, and the authority axes
        come from ``resolve_identity_launch`` — the legacy profile path no
        longer selects them (11c cutover). A RETAINED launch keeps the
        committed-model/frozen-persona machinery: the record's identity
        revision and overlay stay immutable across the retask — the retained
        identity is the child loop's committed launch binding, falling back
        to the record's identity and the child's installed selection. The
        retained launch still goes through the SAME canonical resolution as
        a fresh launch for everything it selects: the top-level model field
        is the canonical model selection (the schema promotes a legacy
        nested config.model), and a supplied purpose/seat binding is
        validated through the route grammar — never copied from the previous
        run, never accepted unvalidated. The retained admission input is the
        child's EFFECTIVE MODEL CONTEXT (``select_model_context``), not its
        whole history: the sealed archive prefix sits before the newest
        compaction boundary, so its payload-trimmed RAM copies never reach
        the compatibility and thinking checks as if they were real history.

        ``override_permit``/``override_ledger`` (C4): the minted forced-
        attempt binding for an explicit ``override_cooldown`` task request,
        threaded into the model-assignment availability predicate so the
        cooled forced deployment stays assignable.
        """
        child = runtime.agent_loop if runtime is not None else None
        if child is None:
            candidate = resolve_identity_candidate(
                args,
                parent_orchestrator=parent.agent_loop.config_orchestrator,
                tool_inventory={
                    name: object()
                    for name in parent.agent_loop.tool_manager.registered_tools
                },
                # The admission ceiling is the parent's DELEGATION set (granted
                # configuration ∩ the parent identity's delegation posture ∩
                # every ancestor's delegation ceiling), not its personal
                # inventory: personal posture and delegation are independent
                # axes.
                authorized_tool_names=frozenset(
                    parent.agent_loop.tool_manager.delegable_tools
                ),
                session_model=parent.agent_loop.config.active_model,
                user_prompts_dirs=parent.agent_loop.harness_files.user_prompts_dirs,
                # Purpose routes resolve against the conversation's LIVE
                # route table — the in-memory table the reload swaps
                # atomically, the same source the rendered launch-target
                # table carries — so a validated route edit applies to
                # every launch admitted after the reload commit (ADR 0022).
                override_permit=override_permit,
                override_ledger=override_ledger,
            )
            claim = args.config
            if claim is not None and claim.predecessor_agent_id is not None:
                # A replacement claim carries the package's validated
                # effective tier end to end: admission judged the claim
                # against the claimed tier or the predecessor's established
                # tier, so the successor's record and loop carry that tier
                # instead of the direct-launch default.
                inherited_tier = claim.package_tier
                if inherited_tier is None:
                    predecessor = self._view._agent_records.get(
                        claim.predecessor_agent_id
                    )
                    if predecessor is not None:
                        inherited_tier = predecessor.package_tier
                if inherited_tier is not None:
                    candidate = replace(candidate, effective_tier=inherited_tier)
            return candidate
        retained_identity = (
            child.launch_identity or identity or child.config.agent.identity
        )
        retained_config = args.config
        if args.model is not None:
            # The top-level model field is the canonical selection on every
            # launch path: promote it over the retained semantic config (a
            # dual supply is rejected by the schema itself). The shorthand
            # grammar normalizes exactly like fresh resolution: the model
            # component selects the model, and the identity component is
            # checked against the retained record's committed identity by
            # the caller's retained-identity match.
            model_expression = args.model
            shorthand = parse_model_shorthand(args.model)
            if shorthand is not None:
                model_expression = shorthand.model
            retained_config = LaunchConfig.model_validate({
                **(
                    retained_config.model_dump(exclude_unset=True, mode="python")
                    if retained_config is not None
                    else {}
                ),
                "model": model_expression,
            })
        if args.purpose is not None:
            # The retained instance's committed identity/model are pinned by
            # its record, so the route's own defaults are validated for
            # availability only — never adopted — but an invalid purpose or
            # seat binding fails loudly instead of being accepted. The
            # binding validates against the conversation's LIVE route table,
            # so a purpose whose route was edited by a reload is validated
            # exactly as a fresh launch validates it (ADR 0022).
            validate_launch_purpose_binding(
                args.purpose,
                config=parent.agent_loop.config,
                seat=args.seat,
                session_model=parent.agent_loop.config.active_model,
            )
        return resolve_launch(
            identity=retained_identity,
            config=retained_config,
            parent_orchestrator=child.config_orchestrator,
            purpose=args.purpose,
            tool_inventory={
                name: object()
                for name in parent.agent_loop.tool_manager.registered_tools
            },
            # The admission ceiling is the parent's DELEGATION set (granted
            # configuration ∩ the parent identity's delegation posture ∩ every
            # ancestor's delegation ceiling), not its personal inventory:
            # personal posture and delegation are independent axes.
            authorized_tool_names=frozenset(
                parent.agent_loop.tool_manager.delegable_tools
            ),
            # Admission validates the EFFECTIVE model context: the messages
            # the continuation will actually send (everything from the newest
            # compaction boundary onward, plus the system prompts before it).
            # The archived history before that boundary is excluded because
            # M1 replaced exactly those RAM copies with payload-trimmed ones —
            # admission must never read that trimmed view as if it were the
            # child's real history.
            history=tuple(select_model_context(child.messages)),
            accumulated_overrides=child.launch_overrides,
            committed_model=child.committed_model,
            frozen_persona=(
                FrozenPersona(instructions=child.frozen_instructions)
                if child.frozen_instructions is not None
                else None
            ),
            # The retained instance's committed tier/capability ride the
            # retask's candidate, so the launch metadata the recovery
            # boundary reads keeps describing the package's true tier.
            effective_tier=child.launch_package_tier or PackageTier.SUBSTANTIVE,
            replacement_capability=(
                child.launch_replacement_capability or ReplacementCapability.NONE
            ),
            override_permit=override_permit,
            override_ledger=override_ledger,
        )

    def mint_override_permit(
        self,
        parent: SessionRuntime,
        args: TaskArgs,
        *,
        committed: CommittedModelIdentity | None = None,
    ) -> OverrideMint | None:
        """Mint the C1 permit for an explicit ``override_cooldown`` request.

        C4 activation: the strict task flag only REQUESTS the un-stick
        override — root launch admission mints the endpoint-bound permit
        from the ROOT ledger (shared by every orchestrator copy) only here,
        and only for the explicit request. The permit's endpoint comes from
        a pure pre-resolution of the launch selection — the explicitly named
        model, else the given committed binding (a retained retask), else
        the canonical session-default resolution — which is the same
        selection the deep validation makes once the permit bypasses the
        cooldown predicate. A ledger denial raises :class:`ToolError` with
        the bounded diagnostics and consumes NOTHING (a denied reserve
        mutates no counter, no slot, and no episode).

        Automatic paths never carry the flag (F6): only the task tool's
        explicit argument can request the override, and a subagent context
        cannot launch at all.
        """
        if not args.override_cooldown:
            return None
        config = parent.agent_loop.config
        expression: str | None = args.model
        if expression is not None:
            shorthand = parse_model_shorthand(expression)
            if shorthand is not None:
                expression = shorthand.model
        elif (
            args.config is not None
            and "model" in args.config.model_fields_set
            and args.config.model is not None
        ):
            # The schema's documented promotion: the legacy nested
            # config.model is the canonical top-level model selection.
            expression = args.config.model
        try:
            if expression is not None:
                selected = (
                    resolver_for(config)
                    .resolve(expression, allowed_models=config.allowed_models)
                    .identity
                )
            elif committed is not None:
                selected = committed
            else:
                selected = resolve_identity_launch(
                    args,
                    config=config,
                    authorized_tool_names=frozenset(
                        parent.agent_loop.tool_manager.delegable_tools
                    ),
                    session_model=parent.agent_loop.config.active_model,
                    user_prompts_dirs=parent.agent_loop.harness_files.user_prompts_dirs,
                ).committed_model
        except ModelResolutionError as exc:
            raise InvalidLaunchModelError("model", str(exc)) from exc
        snapshot = config.catalog_snapshot
        provider = (
            snapshot.catalog.providers.get(selected.provider)
            if snapshot is not None
            else None
        )
        # The raw api_base: the ledger canonicalizes internally, so the
        # reserve (and the denial diagnostics' status lookup) must receive
        # the URL, never the canonical identity.
        raw_endpoint = provider.api_base if provider is not None else None
        ledger = parent.agent_loop.config_orchestrator.cooldown_override_ledger
        outcome = ledger.reserve(raw_endpoint)
        if outcome.denial is not None or outcome.permit is None:
            raise ToolError(
                _override_denial_message(outcome.denial, ledger, raw_endpoint)
            )
        return OverrideMint(
            permit=outcome.permit, ledger=ledger, endpoint=outcome.permit.endpoint
        )

    def validate_admission(
        self, parent: SessionRuntime, identity: tuple[str, int]
    ) -> None:
        """Reject a launch whose admission window closed or identity changed."""
        live_identity = (
            parent.agent_loop.session_id,
            parent.agent_loop._session_generation,
        )
        admission_blocked = (
            self._view._admission_closed or self._view._draining_children
        )
        identity_changed = (
            self._view._root is not parent
            or self._view._generation_identity != identity
            or live_identity != identity
        )
        if admission_blocked or identity_changed or parent._closed is True:
            message = (
                "Agent admission is closed"
                if self._view._admission_closed
                else "Agent admission changed during child creation"
            )
            raise RuntimeError(message)

    @staticmethod
    def _claim_shape_guards(args: TaskArgs, claim: LaunchConfig) -> None:
        """Claim-shape guards: a NEW agent_id, an explicit model, an explicit package."""
        if args.agent_id is not None:
            raise LaunchConfigError(
                "config.predecessor_agent_id",
                "a replacement claim launches a NEW agent: agent_id must be None",
            )
        if args.model is None and claim.model is None:
            # The canonical top-level model field and the legacy nested
            # config.model are the same selection (the schema promotes the
            # nested form); a dual supply is a schema validation error.
            raise LaunchConfigError(
                "model",
                "a replacement claim must name the replacement model explicitly "
                "(the top-level model field or config.model)",
            )
        if claim.work_package_id is None:
            raise LaunchConfigError(
                "config.work_package_id",
                "a replacement claim must name its work package: the per-package "
                "replacement budget is keyed on work_package_id",
            )

    def _predecessor_envelope_locked(
        self, parent: SessionRuntime, predecessor_id: str
    ) -> tuple[AgentRecord, FailureEnvelope]:
        """The owned predecessor record and its latest failed run's envelope."""
        record = self._view._agent_records.get(predecessor_id)
        if record is None:
            raise ReplacementPolicyError(
                ["predecessor_unknown"],
                f"Replacement claim rejected: predecessor agent "
                f"{predecessor_id} is not a live record (closed/evicted "
                "agents cannot be replaced); re-dispatch the package fresh",
            )
        if record.parent_session_id != parent.agent_loop.session_id:
            raise ReplacementPolicyError(
                ["predecessor_not_owned"],
                f"Replacement claim rejected: predecessor agent "
                f"{predecessor_id} is not owned by this parent session",
            )
        # The failed run's envelope, latest-first over the current run
        # and the run history.
        failed_run = next(
            (
                run
                for run in (record.current_run, *reversed(record.run_history))
                if run is not None
                and run.status is RunStatus.FAILED
                and run.failure is not None
            ),
            None,
        )
        if failed_run is None or failed_run.failure is None:
            raise ReplacementPolicyError(
                ["predecessor_has_no_failure_envelope"],
                f"Replacement claim rejected: predecessor agent "
                f"{predecessor_id} has no failed run with a failure envelope",
            )
        envelope = failed_run.failure
        # Review R6: the envelope must come from the predecessor's LATEST
        # run, and that run must be terminal — a claim never launches a
        # replacement alongside a resumed successor or after a newer run
        # already displaced the failed one.
        if (
            failed_run.run_id != record.latest_run_id
            or record.current_run is not None
            and record.current_run.run_id != failed_run.run_id
        ):
            raise ReplacementPolicyError(
                ["predecessor_run_not_terminal"],
                f"Replacement claim rejected: predecessor agent "
                f"{predecessor_id} has a newer run than its failed run "
                f"{failed_run.run_id}; the failure is not the latest "
                "terminal state, so the package has no failed run to "
                "replace",
            )
        return record, envelope

    @staticmethod
    def _check_lineage_and_barrier_locked(
        record: AgentRecord, claim: LaunchConfig, envelope: FailureEnvelope
    ) -> None:
        """Lineage match-or-establish plus the reconciliation barrier (locked).

        A predecessor that already carries package lineage (it was itself a
        replacement) pins the claim to it; otherwise the claim establishes
        the lineage. The write-ownership transfer happens only after
        reconciliation — a suspended barrier that still requires
        reconcile_result blocks the claim; verified unrepeatable damage
        prohibits any repeat of the package. A typed completion-audit
        rejection is the one exempt prohibition: its PROHIBITED eligibility
        bars only a warm same-agent auto-resume, and the failure-time
        policy authorizes a cross-agent replacement for it (P11), so the
        claim proceeds to policy validation.
        """
        if (
            record.work_package_id is not None
            and record.work_package_id != claim.work_package_id
        ):
            raise ReplacementPolicyError(
                ["work_package_lineage_mismatch"],
                f"Replacement claim rejected: package "
                f"{claim.work_package_id} does not match the predecessor's "
                f"established package {record.work_package_id}",
            )
        if (
            record.package_tier is not None
            and claim.package_tier is not None
            and record.package_tier is not claim.package_tier
        ):
            raise ReplacementPolicyError(
                ["package_tier_lineage_mismatch"],
                f"Replacement claim rejected: tier {claim.package_tier.value} "
                f"does not match the predecessor's established tier "
                f"{record.package_tier.value}",
            )
        eligibility, _codes = _effective_eligibility(record, envelope)
        if eligibility is RecoveryEligibility.RECONCILE_REQUIRED:
            raise ReplacementPolicyError(
                ["reconcile_required_before_replacement"],
                f"Replacement claim rejected: predecessor agent "
                f"{record.agent_id} is suspended with a reconcile-required "
                "barrier; submit reconcile_result first — the transfer "
                "happens only after reconciliation",
            )
        rejection_escalation = (
            record.reconcile_outcome != ReconcileOutcome.PROHIBITED.value
            and envelope.failure.failure_class is FailureClass.COMPLETION_REJECTION
        )
        if eligibility is RecoveryEligibility.PROHIBITED and not rejection_escalation:
            raise ReplacementPolicyError(
                ["recovery_prohibited"],
                f"Replacement claim rejected: predecessor agent "
                f"{record.agent_id} has verified unrepeatable damage; the "
                "package is not repeatable on any model",
            )

    def _validate_claim_policy_locked(
        self,
        parent: SessionRuntime,
        record: AgentRecord,
        claim: LaunchConfig,
        envelope: FailureEnvelope,
        *,
        claim_model: str,
        model_field: str,
    ) -> ReplacementDecision:
        """Per-package budget + policy validation (locked); returns the decision.

        The budget spans the package lineage (keyed on work_package_id);
        exhaustion rejects with ``package_replacements_exhausted`` so the
        orchestrator parks the package. The policy validation resolves the
        claimed model (the canonical top-level model field, or the legacy
        nested config.model the schema promotes), builds the claimed
        candidate, and runs the full failure-time decision procedure —
        endpoint independence against the ATTEMPTED failed endpoint, tier
        capability, allowlist/price cap.
        """
        recovery = parent.agent_loop.config.recovery.subagents
        package_key = claim.work_package_id
        assert package_key is not None  # the claim-shape guard ran first
        used = self._view._package_replacements.get(package_key, 0)
        if used >= recovery.fallback_replacements_max_per_package:
            raise ReplacementPolicyError(
                ["package_replacements_exhausted"],
                f"Replacement claim rejected: package {package_key} has "
                f"used its budget of "
                f"{recovery.fallback_replacements_max_per_package} "
                "replacements; park the package and continue independent work",
            )
        config = parent.agent_loop.config
        try:
            resolver = resolver_for(config)
            resolved = resolver.resolve(claim_model)
        except ModelResolutionError as exc:
            raise InvalidLaunchModelError(model_field, str(exc)) from exc
        # The recovery boundary labels candidates by canonical base model,
        # so route preset references canonicalize through the catalog
        # mapping; a target's explicit replacement-capability marking rides
        # the claimed candidate and the candidate set — a separate axis from
        # the package tier, and absent markings keep the legacy tier-label
        # derivation (the shipped routes mark nothing). The facts derive
        # from the conversation's LIVE route table — the same source launch
        # admission resolves against (ADR 0022).
        mechanical_models, capability_markings = route_recovery_facts(
            config.purpose_routes.routes,
            session_model=session_model_alias(config),
            canonical_models=resolver.canonical_model_map(),
            catalog_prices=resolver.catalog_prices(),
        )
        claimed = ReplacementCandidate(
            base_model=resolved.base_model,
            provider=resolved.deployment.provider,
            wire_name=resolved.deployment.name,
            endpoint=canonical_endpoint_identity(resolved.provider.api_base),
            tier=(
                PackageTier.MECHANICAL
                if resolved.base_model in mechanical_models
                else PackageTier.SUBSTANTIVE
            ),
            capability=(
                capability_markings.get(resolved.base_model, ReplacementCapability.NONE)
                if capability_markings
                else None
            ),
            output_price_per_million=resolved.deployment.prices.output,
        )
        snapshot = config.catalog_snapshot
        assert snapshot is not None  # resolver_for raised otherwise
        candidates = candidates_from_catalog(
            snapshot.catalog,
            mechanical_base_models=mechanical_models,
            capabilities=(capability_markings or None),
        )
        effective_tier = (
            claim.package_tier
            if claim.package_tier is not None
            else (
                record.package_tier
                if record.package_tier is not None
                else PackageTier.SUBSTANTIVE
            )
        )
        decision = validate_replacement_selection(
            envelope=envelope,
            package_tier=effective_tier,
            claimed=claimed,
            candidates=candidates,
            config=recovery,
        )
        if decision.disposition != "selected":
            raise ReplacementPolicyError(
                decision.reason_codes,
                decision.notice
                or f"Replacement claim rejected for package {package_key}",
            )
        return decision

    async def validate_replacement_claim(
        self, parent: SessionRuntime, args: TaskArgs
    ) -> ReplacementDecision:
        """Validate a WP5 failure-time replacement claim at launch admission.

        The orchestrator claims a replacement by launching a NEW agent
        (``args.agent_id is None``): the replacement model is named by the
        canonical top-level ``model`` field (the legacy nested
        ``config.model`` is promoted to it by the schema), and ``config``
        carries the claim: ``predecessor_agent_id`` (the failed agent),
        ``work_package_id`` (the package identity the per-package budget is
        keyed on), and ``package_tier`` (mechanical/substantive; unknown
        defaults conservatively to substantive). There is NO dispatch-time
        substitution table and NO automatic cross-model fallback: the claim
        is validated HERE, against the predecessor's failed run's failure
        envelope, at failure time.

        Admission order (the write-ownership handoff): claim-shape guards →
        predecessor ownership + failed-run envelope → lineage
        match-or-establish → reconciliation barrier → per-package budget →
        policy validation (endpoint independence against the ATTEMPTED failed
        endpoint, tier capability, allowlist/price cap) → canonical identity
        resolution + Task admission. Only then is the old recovery
        invalidated and the package counter incremented — BEFORE the new
        run's write token is registered at accept_background, so the
        transfer happens only after reconciliation state is settled and the
        explicit handoff is admitted. Review finding 2: the identity
        resolution and admission COMPLETE before any consequential
        replacement-state change — an invalid or denied launch identity
        leaves zero side effects (recovery intact, budget unconsumed, no
        supersession). Budgets and recurrence counters span replacements:
        they are never reset.

        Policy rejections raise :class:`ReplacementPolicyError` (the
        orchestrator parks the package on those reason codes); claim-shape
        violations raise :class:`LaunchConfigError` /
        :class:`InvalidLaunchModelError`.
        """
        claim = args.config
        assert claim is not None and claim.predecessor_agent_id is not None
        self._claim_shape_guards(args, claim)
        # The claimed model is the canonical top-level model field, or the
        # legacy nested config.model the schema promotes — the same selection
        # the fresh launch resolves, so the policy validates what the new
        # agent will actually run.
        claim_model = args.model if args.model is not None else claim.model
        model_field = "model" if args.model is not None else "config.model"
        assert claim_model is not None  # the claim-shape guard ran first
        async with self._view._registry_lock:
            record, envelope = self._predecessor_envelope_locked(
                parent, claim.predecessor_agent_id
            )
            self._check_lineage_and_barrier_locked(record, claim, envelope)
            decision = self._validate_claim_policy_locked(
                parent,
                record,
                claim,
                envelope,
                claim_model=claim_model,
                model_field=model_field,
            )
            # Review R6: exclusive package ownership — a predecessor
            # already superseded (an earlier claim or busy replacement
            # admitted) can never ground a second claim: repeated claims
            # must not create multiple replacements. The check runs AFTER
            # the per-package budget so an exhausted package still reports
            # its exhaustion reason.
            if record.recovery_superseded or record.reserved_for_replacement:
                raise ReplacementPolicyError(
                    ["predecessor_already_superseded"],
                    f"Replacement claim rejected: predecessor agent "
                    f"{record.agent_id} was already superseded by an "
                    "admitted replacement; the package already has its "
                    "replacement",
                )
            # Review finding 2 (A2): canonical identity resolution and
            # admission COMPLETE before any consequential replacement-state
            # change. The invalidation and budget consumption below are the
            # consequential mutation; an invalid or denied launch identity
            # must fail BEFORE them — a rejected launch leaves zero side
            # effects (recovery intact, budget unconsumed, no supersession).
            # Resolution and admission are pure in the parent runtime and
            # the launch arguments (no registry state, no locks), so they
            # run safely under the registry lock, and they mirror the
            # fresh-launch admission in create_registered exactly.
            requested_identity = requested_identity_reference(args)
            if requested_identity is not None:
                self.require_identity_allowed(parent, requested_identity)
            # C4: an explicit override request on a replacement claim
            # resolves the claim's dry-run candidate THROUGH a minted permit
            # (the cooled forced deployment must pass the claim's identity
            # resolution); the dry-run reservation is released immediately —
            # the admitted launch mints its OWN permit at create_registered,
            # and a release never consumes the start budget.
            mint = self.mint_override_permit(parent, args)
            try:
                candidate = self.resolve_candidate(
                    parent,
                    args,
                    override_permit=mint.permit if mint is not None else None,
                    override_ledger=mint.ledger if mint is not None else None,
                )
            finally:
                if mint is not None:
                    mint.ledger.release(mint.permit)
            self.require_identity_allowed(
                parent, candidate.orchestrator.config.agent.identity
            )
            # Handoff ordering: the old recovery is disabled and the package
            # counter incremented BEFORE the new run's write token is
            # registered at accept_background — the write-ownership transfer
            # happens only after reconciliation state is settled and the
            # explicit claim is admitted. The predecessor is never
            # auto-restarted; its lineage, counters, and artifacts survive.
            package_key = claim.work_package_id
            assert package_key is not None  # the claim-shape guard ran first
            self._view._invalidate_recovery_locked(record)
            self._view._package_replacements[package_key] = (
                self._view._package_replacements.get(package_key, 0) + 1
            )
            logger.info(decision.notice)
            return decision

    @contextmanager
    def creation_fence(self) -> Iterator[None]:
        """Hold the child-creation fence (policy reservation + counter)."""
        self._view._require_policy_unreserved()
        self._view._creating_children += 1
        try:
            yield
        finally:
            self._view._creating_children -= 1

    def restore_sequence(self, root: SessionRuntime) -> None:
        """Restore the agent-id high-water mark from persisted metadata."""
        metadata = root.agent_loop.session_logger.session_metadata
        config = metadata.config if metadata is not None else None
        high_water_mark = (
            config.get(_AGENT_ID_HIGH_WATER_MARK_KEY) if config is not None else None
        )
        if isinstance(high_water_mark, int) and not isinstance(high_water_mark, bool):
            self._view._next_agent_seq = max(
                self._view._next_agent_seq, high_water_mark
            )

    async def issue_agent_id(self, parent: SessionRuntime) -> str:
        """Allocate the next agent id, persisting the high-water mark."""
        self._view._next_agent_seq += 1
        reserved_seq = self._view._next_agent_seq
        metadata = parent.agent_loop.session_logger.session_metadata
        if metadata is not None:
            config = dict(metadata.config or {})
            config[_AGENT_ID_HIGH_WATER_MARK_KEY] = reserved_seq
            metadata.config = config
            await parent.agent_loop.session_logger._persist_metadata_field(
                "config", config
            )
        return f"agent-{reserved_seq}"

    async def create_registered(
        self, parent: SessionRuntime, args: TaskArgs, ctx: InvokeContext
    ) -> tuple[SessionRuntime, str]:
        """Construct, register, and project one child, rolling back on failure.

        The second element is the child's committed launch identity reference.

        C4: an explicit ``override_cooldown`` task request mints its permit
        HERE, at root admission, before the candidate resolution (the deep
        validation must resolve the cooled forced deployment THROUGH the
        permit); the permit binds to the constructed child's first model
        call, and every failure before the run starts releases the
        reservation so the endpoint's in-flight slot never strands.
        """
        mint = self.mint_override_permit(parent, args)
        try:
            return await self._create_registered_minted(parent, args, ctx, mint)
        except BaseException:
            if mint is not None:
                # Pretransport release: the run never started, so the
                # reservation must not strand the in-flight slot.
                mint.ledger.release(mint.permit)
            raise

    async def create_process_registered(
        self,
        parent: SessionRuntime,
        args: TaskArgs,
        ctx: InvokeContext,
        *,
        attempt_id: str | None = None,
        reuse: ProcessChildSession | None = None,
        rollback_wait: float | None = None,
    ) -> ProcessChildSession:
        """Publish one child whose loop lives in its own process (item 1 seam 2).

        The aggregate transaction runs here, in the process that owns the
        children, while the child's loop lives behind its own pipe: admission
        fencing, the working-set handoff, persistence (the child's own first
        write, keyed by this attempt), parent linkage, and projection all
        execute as ONE attempt keyed by the coordinator-minted attempt id, and
        a failure answers a :class:`ChildPublicationError` naming the failed
        stage. The attempt id is caller-supplyable so a coordinator that mints
        it keeps the identity across a lost reply.

        ``reuse`` names a retained child's existing handle: a re-task of that
        child links the new tool call to the process it already has instead of
        spawning a second one.
        """
        from chartreux.app_server.child_agent_process import DEFAULT_CHILD_ROLLBACK_WAIT

        spawn = self._view._subagent_spawn
        resolved_attempt_id = attempt_id or new_publication_attempt_id()
        if spawn is None:
            raise ChildPublicationError(
                ChildPublicationStage.ADMISSION,
                resolved_attempt_id,
                detail="this run selects no process-backed spawn adapter",
            )
        mint = self.mint_override_permit(parent, args)
        try:
            candidate = (
                self.resolve_candidate(
                    parent,
                    args,
                    override_permit=mint.permit,
                    override_ledger=mint.ledger,
                )
                if mint is not None
                else self.resolve_candidate(parent, args)
            )
            self.require_identity_allowed(
                parent, candidate.orchestrator.config.agent.identity
            )
            attempt = ChildPublicationAttempt(
                parent=parent,
                identity=(
                    parent.agent_loop.session_id,
                    parent.agent_loop._session_generation,
                ),
                candidate=candidate,
                tool_call_id=ctx.tool_call_id,
                attempt_id=resolved_attempt_id,
            )
            retained = (
                bool(args.retain)
                if args.retain is not None
                else bool(reuse is not None and reuse.retained)
            )
            publisher = ProcessChildPublication(
                self,
                spawn,
                rollback_wait=(
                    DEFAULT_CHILD_ROLLBACK_WAIT
                    if rollback_wait is None
                    else rollback_wait
                ),
            )
            return await publisher.publish(
                attempt, retained=retained, mint=mint, reuse=reuse
            )
        except BaseException:
            if mint is not None:
                # Pretransport release: the run never started, so the
                # reservation must not strand the endpoint's in-flight slot.
                mint.ledger.release(mint.permit)
            raise

    async def _create_registered_minted(
        self,
        parent: SessionRuntime,
        args: TaskArgs,
        ctx: InvokeContext,
        mint: OverrideMint | None,
    ) -> tuple[SessionRuntime, str]:
        if mint is not None:
            # C4: the minted permit must reach the model-assignment
            # availability predicate, and the registry's thin delegate has a
            # fixed signature — the override path resolves through the
            # admission collaborator directly. The ordinary (no-mint) path
            # keeps the delegate so instance patches keep intercepting.
            candidate = self.resolve_candidate(
                parent, args, override_permit=mint.permit, override_ledger=mint.ledger
            )
        else:
            candidate = self._view._resolve_launch_candidate(parent, args)
        # The identity denylist applies AFTER final resolution (11c): a
        # route/shorthand/structured identity launch is checked against its
        # resolved identity — the same surface the task tool enforces — before
        # any child resource is constructed.
        self.require_identity_allowed(
            parent, candidate.orchestrator.config.agent.identity
        )
        attempt = ChildPublicationAttempt(
            parent=parent,
            identity=(
                parent.agent_loop.session_id,
                parent.agent_loop._session_generation,
            ),
            candidate=candidate,
            tool_call_id=ctx.tool_call_id,
            attempt_id=new_publication_attempt_id(),
        )
        self.validate_admission(parent, attempt.identity)
        with self.creation_fence():
            child = await self._view._runtime_factory.create_child(
                parent.agent_loop, candidate
            )
            # Spawn handoff, inside the creation fence: the child's first
            # message lists the working-set files the dispatcher hands it
            # (paths + last-seen revisions + current outlines), ahead of the
            # task brief. None when the parent has no working set.
            handoff = child.file_state_handoff_message()
            if handoff is not None:
                child.messages.append(handoff)
            if mint is not None:
                bind_override_to_loop(child, mint)
            attempt.child = child
        try:
            self.validate_admission(parent, attempt.identity)
        except BaseException:
            await self._view._discard_child(child)
            raise
        runtime = self._view._build_child_runtime(child)
        attempt.runtime = runtime
        async with self._view._child_publication(runtime):
            try:
                self.validate_admission(parent, attempt.identity)
            except BaseException:
                await self._view._discard_child(child)
                raise
            self._view._children[child.session_id] = runtime
            self._view._child_links[child.session_id] = (parent, ctx.tool_call_id)
            try:
                # The attempt id rides the FIRST durable write: a torn
                # persist-to-link window stays attributable to its attempt.
                await child.persist_empty_session(attempt_id=attempt.attempt_id)
                attempt.persisted = True
                self.validate_admission(parent, attempt.identity)
                await parent.agent_loop.record_child_session(
                    child,
                    ctx.tool_call_id,
                    candidate.identity,
                    attempt_id=attempt.attempt_id,
                )
                attempt.link_recorded = True
                self.validate_admission(parent, attempt.identity)
                attempt.projection_started = True
                if parent.turns._projector is not None:
                    await parent.turns.link_subagent(ctx.tool_call_id, child.session_id)
                self.validate_admission(parent, attempt.identity)
            except BaseException:
                await attempt.rollback(self._view)
                raise
            return runtime, candidate.identity

    async def rollback_created(
        self, parent: SessionRuntime, runtime: SessionRuntime, tool_call_id: str
    ) -> None:
        """Remove every durable/runtime trace of a child rejected before acceptance."""
        # C4: a rejected child never runs, so its bound override permit (if
        # any) is released pretransport — the in-flight slot must not strand.
        release_bound_override(runtime.agent_loop)
        child = runtime.agent_loop
        session_id = child.session_id
        attempt_id = published_attempt_id(child, parent, tool_call_id)
        self._view._children.pop(session_id, None)
        self._view._child_links.pop(session_id, None)
        with suppress(Exception):
            await parent.turns.unlink_subagent(tool_call_id, session_id)
        with suppress(Exception):
            await parent.agent_loop.forget_child_session(session_id, tool_call_id)
        with suppress(Exception):
            await runtime.close()
        # The accepted-then-rejected child is retracted before its records are
        # removed, so a lost delete cannot leave an adoptable transcript.
        with suppress(Exception):
            retire_child_publication(
                child, attempt_id=attempt_id, reason=PUBLICATION_REJECTED_REASON
            )
        with suppress(Exception):
            await delete_saved_session(session_id, child.config.session_logging)
