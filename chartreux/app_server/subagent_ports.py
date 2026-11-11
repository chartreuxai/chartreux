"""Ports for spawning subagents and observing their completion (WP5 [5A]).

The root agent owns admission and the coordinator owns the physical spawn and
reap of subagent runtimes (design.md:89-95); these protocols are the boundary
between them:

* :class:`SubagentSpawnPort` physically spawns one subagent runtime and
  returns a :class:`SpawnedSubagent` handle. Two adapters implement it. The
  in-process adapter — the default under ``CHARTREUX_TRANSPORT=in_process``,
  zero behavior change — builds the child in this process through an injected
  factory. The process-backed adapter spawns a child process via
  :func:`chartreux.app_server.process_bootstrap.spawn_process` and drives it
  over the agent wire.

* :class:`SubagentCompletionSource` streams one spawned subagent's run: its
  events in production order, ending with exactly one terminal
  :class:`SubagentCompletion` frame. The coordinator's event feed ([5B])
  consumes this source.

WP5 [5B] wires these ports into the coordinator's subagent management:
:class:`SubagentAdmission` enforces the capacity cap and the maximum depth
with an atomic slot reservation and typed denials (never silent queueing),
:class:`SubagentEventFeed` relays one run's frames into the coordinator event
feed with incarnation stamps ([2A]/[2B]) and one terminal completion frame for
both foreground and background runs, and :class:`AdmittingSubagentSpawn`
combines them around a spawn port. Retained idle agents count against the cap
until an explicit closure; there is no automatic eviction (design.md:448-456).

A child process is always fresh, never resumed (design.md:176-180): spawning
creates a new runtime, and the handle's ``close`` is the reap. Transcripts and
policy trees are later chunks ([5C]-[5E]) and never reach these ports.
"""

from __future__ import annotations

import asyncio
from collections.abc import (
    AsyncGenerator,
    Awaitable,
    Callable,
    Iterable,
    Iterator,
    Mapping,
)
from contextlib import aclosing, suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import count
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from chartreux.app_server._subagent_records import _ActiveWorkSlot
from chartreux.app_server._subagent_transcript import (
    AGENT_NAME_PAYLOAD_KEY,
    ROOT_SESSION_DIR_PAYLOAD_KEY,
    ChildResultRecord,
    ChildTranscriptError,
    ChildTranscriptStore,
    child_result_digest,
    resolve_child_root_session_dir,
)
from chartreux.app_server.agent_wire import (
    AGENT_TURN_ENDED_METHOD,
    AGENT_TURN_EVENT_METHOD,
    TURN_OUTCOME_COMPLETED,
    TURN_OUTCOME_FAILED,
    TURN_OUTCOME_INTERRUPTED,
    encode_event,
)
from chartreux.app_server.events import RELAY_INCARNATION_FIELD
from chartreux.app_server.transport import resolve_transport_selection
from chartreux.core.agent_loop._execution_port import (
    ExecutionTurnRequest,
    TurnExecutionPort,
)
from chartreux.core.events import BaseEvent
from chartreux.observability.logging import logger

if TYPE_CHECKING:
    from chartreux.core.agent_loop import AgentLoop


class SubagentSpawnError(RuntimeError):
    """Base class for typed subagent-spawn failures."""


class SubagentSpawnDeniedError(SubagentSpawnError):
    """The run's transport selection does not allow this adapter."""


@dataclass(frozen=True, slots=True)
class SubagentSpawnSpec:
    """JSON-safe spawn request for one subagent; crosses the process wire."""

    agent_id: str
    parent_session_id: str | None = None
    session_dir: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SubagentCompletion:
    """Terminal outcome of one subagent run, as the final stream frame."""

    outcome: str
    error: str | None = None


type SubagentFrame = BaseEvent | SubagentCompletion


class SubagentCompletionSource(Protocol):
    """One spawned subagent's run stream.

    Iterating ``stream`` yields the run's events in production order and ends
    with exactly one terminal :class:`SubagentCompletion` frame; a failed or
    interrupted run is carried by that frame, not raised.
    """

    @property
    def agent_id(self) -> str: ...

    def stream(
        self, request: ExecutionTurnRequest
    ) -> AsyncGenerator[SubagentFrame, None]: ...


class SpawnedSubagent(Protocol):
    """Handle for one physically spawned subagent runtime."""

    @property
    def agent_id(self) -> str: ...

    @property
    def completion_source(self) -> SubagentCompletionSource: ...

    async def close(self) -> None: ...


class SubagentSpawnPort(Protocol):
    """Physically spawns subagent runtimes; the coordinator owns this port."""

    async def spawn(self, spec: SubagentSpawnSpec) -> SpawnedSubagent: ...


def spawn_spec_to_wire(spec: SubagentSpawnSpec) -> dict[str, Any]:
    """Encode one spawn spec as the ``subagent/open`` params payload."""
    return {
        "agent_id": spec.agent_id,
        "parent_session_id": spec.parent_session_id,
        "session_dir": spec.session_dir,
        "payload": spec.payload,
    }


def spawn_spec_from_wire(params: Mapping[str, Any]) -> SubagentSpawnSpec:
    """Rebuild one spawn spec from its wire params, refusing a missing id."""
    agent_id = params.get("agent_id")
    if not isinstance(agent_id, str) or not agent_id:
        raise ValueError("a subagent open frame carries no agent id")
    payload = params.get("payload")
    return SubagentSpawnSpec(
        agent_id=agent_id,
        parent_session_id=_optional_str(params.get("parent_session_id")),
        session_dir=_optional_str(params.get("session_dir")),
        payload=dict(payload) if isinstance(payload, Mapping) else {},
    )


def child_spawn_spec(
    *,
    agent_id: str,
    parent_session_id: str | None,
    root_session_dir: Path,
    agent_name: str | None = None,
    payload: Mapping[str, Any] | None = None,
) -> SubagentSpawnSpec:
    """Build the spawn request for one FRESH child process (WP5 [5C]).

    The child creates its own transcript directory under
    ``<root>/agents/<agent>_<ts>_<id>`` and writes it itself, so the spec
    carries the ROOT session dir and the agent name instead of a child
    directory a previous spawn may have used. A later child process is
    therefore a new incarnation with a new transcript record, never a resume.
    """
    merged: dict[str, Any] = dict(payload or {})
    merged[ROOT_SESSION_DIR_PAYLOAD_KEY] = str(root_session_dir)
    merged[AGENT_NAME_PAYLOAD_KEY] = agent_name or agent_id
    return SubagentSpawnSpec(
        agent_id=agent_id, parent_session_id=parent_session_id, payload=merged
    )


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


class InProcessChildFactory(Protocol):
    """Builds one in-process child loop for a spawn spec."""

    async def open(self, spec: SubagentSpawnSpec) -> AgentLoop: ...


def child_result_store(loop: AgentLoop) -> ChildTranscriptStore | None:
    """The in-process child's durable marker store, when it owns one.

    An in-process child loop logs its own transcript under
    ``<root>/agents/<agent>_<ts>_<id>`` exactly like the child process does, so
    it owns the same durable records. A loop without a transcript directory
    (session logging disabled) has no store and no durable record to write.
    """
    session_logger = getattr(loop, "session_logger", None)
    session_dir = getattr(session_logger, "session_dir", None)
    if not isinstance(session_dir, (str, Path)):
        return None
    root_session_dir = resolve_child_root_session_dir(Path(session_dir))
    if root_session_dir is None:
        return None
    try:
        return ChildTranscriptStore(
            Path(session_dir), root_session_dir=root_session_dir
        )
    except ChildTranscriptError:
        return None


def child_terminal_record(
    *, agent_id: str, session_id: str, run_id: str, outcome: str, error: str | None
) -> ChildResultRecord:
    """The durable terminal result record for one child run (save-before-emit)."""
    return ChildResultRecord(
        agent_id=agent_id,
        session_id=session_id,
        run_id=run_id,
        result_id=f"{agent_id}:{run_id}",
        digest=child_result_digest(
            agent_id=agent_id,
            session_id=session_id,
            run_id=run_id,
            outcome=outcome,
            error=error,
        ),
        outcome=outcome,
        completed_at=datetime.now(UTC).isoformat(),
        error=error,
    )


async def record_child_terminal(
    store: ChildTranscriptStore | None, record: ChildResultRecord
) -> bool:
    """Write one child's terminal record durably; True when it was written.

    The write is atomic and fsynced ([5C]), and it happens BEFORE the terminal
    completion frame is relayed, so a coordinator reconciliation after a crash
    always adjudicates the run instead of reporting an unknown outcome. A
    record failure is reported (False) and logged by the caller, never
    concealed as a successful publication.
    """
    if store is None:
        return False
    try:
        await asyncio.to_thread(store.record_result, record)
    except (ChildTranscriptError, OSError):
        return False
    return True


class InProcessSubagentCompletionSource:
    """Stream a run of the child loop living in this process."""

    def __init__(
        self,
        agent_id: str,
        execution_port: TurnExecutionPort,
        *,
        store: ChildTranscriptStore | None = None,
        session_id: str | None = None,
    ) -> None:
        self._agent_id = agent_id
        self._execution_port = execution_port
        self._store = store
        self._session_id = session_id or agent_id

    @property
    def agent_id(self) -> str:
        return self._agent_id

    def stream(
        self, request: ExecutionTurnRequest
    ) -> AsyncGenerator[SubagentFrame, None]:
        return self._stream(request)

    async def _stream(
        self, request: ExecutionTurnRequest
    ) -> AsyncGenerator[SubagentFrame, None]:
        options = request.turn_options
        turn_id = options.turn_id if options is not None else None
        outcome = TURN_OUTCOME_COMPLETED
        error: str | None = None
        try:
            async with aclosing(self._execution_port.run_turn(request)) as events:
                async for event in events:
                    yield event
        except GeneratorExit:
            raise
        except asyncio.CancelledError:
            outcome = TURN_OUTCOME_INTERRUPTED
        except BaseException as exc:
            outcome = TURN_OUTCOME_FAILED
            error = f"{type(exc).__name__}: {exc}"
        await self._publish_terminal_record(turn_id, outcome=outcome, error=error)
        yield SubagentCompletion(outcome=outcome, error=error)

    async def _publish_terminal_record(
        self, turn_id: str | None, *, outcome: str, error: str | None
    ) -> None:
        """Save-before-emit: the durable record precedes the completion frame."""
        if self._store is None or turn_id is None:
            return
        record = child_terminal_record(
            agent_id=self._agent_id,
            session_id=self._session_id,
            run_id=turn_id,
            outcome=outcome,
            error=error,
        )
        try:
            written = await record_child_terminal(self._store, record)
        except BaseException:
            # A cancelled publication must not strand the run: the completion
            # frame still closes the stream, and the missing record surfaces
            # in reconciliation as an unknown outcome rather than a replay.
            written = False
        if not written:
            logger.warning(
                "Failed to record the in-process child terminal result for %s",
                self._agent_id,
            )


class InProcessSpawnedSubagent:
    """A spawned subagent whose runtime lives in this process."""

    def __init__(
        self,
        agent_id: str,
        loop: AgentLoop,
        completion_source: SubagentCompletionSource,
    ) -> None:
        self._agent_id = agent_id
        self._loop = loop
        self._completion_source = completion_source

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @property
    def completion_source(self) -> SubagentCompletionSource:
        return self._completion_source

    async def close(self) -> None:
        await self._loop.aclose()


class InProcessSubagentSpawn(SubagentSpawnPort):
    """The in-process spawn adapter: zero behavior change, no child process."""

    def __init__(self, child_factory: InProcessChildFactory) -> None:
        self._child_factory = child_factory

    async def spawn(self, spec: SubagentSpawnSpec) -> SpawnedSubagent:
        if not spec.agent_id:
            raise ValueError("a spawn spec carries an agent id")
        loop = await self._child_factory.open(spec)
        source = InProcessSubagentCompletionSource(
            spec.agent_id,
            loop.execution_port,
            store=child_result_store(loop),
            session_id=getattr(loop, "session_id", None),
        )
        return InProcessSpawnedSubagent(spec.agent_id, loop, source)


def subagent_spawn_port(
    environ: Mapping[str, str] | None = None, *, child_factory: InProcessChildFactory
) -> SubagentSpawnPort:
    """Select the spawn adapter for this run's transport selection.

    ``in_process`` — the default when nothing selects a transport — returns
    the in-process adapter, so an in-process run keeps every subagent in its
    own process. The multi-process selectors return the process-backed
    adapter that spawns a child via ``process_bootstrap.spawn_process``.
    """
    selection = resolve_transport_selection(environ=environ)
    if selection.kind == "in_process":
        return InProcessSubagentSpawn(child_factory)
    from chartreux.app_server.child_agent_process import ProcessSubagentSpawn

    return ProcessSubagentSpawn()


# --- WP5 [5B]: admission, event feed, admitting spawn wrapper ----------------

MAX_SUBAGENT_DEPTH = 1
"""The maximum subagent depth: a subagent cannot spawn another subagent."""


class SubagentAdmissionDeniedError(SubagentSpawnError):
    """Typed, wire-visible denial of one subagent admission request ([5B])."""


class SubagentAdmissionClosedError(SubagentAdmissionDeniedError):
    """Agent admission is closed: a drain is in progress or the session is shutting down."""

    def __init__(self) -> None:
        super().__init__(
            "Subagent launch rejected: agent admission is closed for shutdown; "
            "no child work starts while the session drains"
        )


class SubagentCapacityDeniedError(SubagentAdmissionDeniedError):
    """The concurrent-children cap is exhausted; retained idle agents count."""

    def __init__(self, *, running: int, cap: int) -> None:
        self.running = running
        self.cap = cap
        super().__init__(
            f"Subagent launch rejected: {running} agents already running "
            f"(cap {cap}); "
            "wait for completion or release agents before launching more"
        )


class SubagentDepthDeniedError(SubagentAdmissionDeniedError):
    """The request would exceed the maximum subagent depth."""

    def __init__(self, *, depth: int, max_depth: int) -> None:
        self.depth = depth
        self.max_depth = max_depth
        super().__init__(f"Agent depth limit of {max_depth} reached")


class SubagentAdmission:
    """Capacity + depth admission with an atomic slot reservation (WP5 [5B]).

    ``reserve`` is SYNCHRONOUS on purpose: between the occupancy check and
    the slot insertion there is no ``await``, so under the asyncio loop
    concurrent admission requests observe each other's reservations — the
    reservation is atomic and visible to concurrent requests. Over-cap and
    over-depth requests raise typed denials; nothing is ever silently queued.
    A reserved slot is released on spawn failure or on completion
    (:meth:`release`). A retained idle agent keeps counting against the cap
    (:meth:`mark_retained_idle`) until an explicit closure
    (:meth:`clear_retained`) — there is NO automatic eviction
    (design.md:448-456).
    """

    def __init__(
        self,
        slots: set[_ActiveWorkSlot] | None = None,
        *,
        max_depth: int = MAX_SUBAGENT_DEPTH,
    ) -> None:
        self.slots: set[_ActiveWorkSlot] = slots if slots is not None else set()
        self.max_depth = max_depth
        self.retained_idle: set[str] = set()

    @property
    def occupied(self) -> int:
        """Active slots plus retained idle agents."""
        return len(self.slots) + len(self.retained_idle)

    def reserve(
        self,
        *,
        cap: int,
        depth: int = 0,
        reuse_agent_id: str | None = None,
        retained_idle: Iterable[str] | None = None,
    ) -> _ActiveWorkSlot:
        """Reserve one slot atomically, or raise a typed denial.

        ``retained_idle`` overrides the internally tracked retained set (the
        registry derives it from its agent records). A reuse of a retained
        idle agent by ``reuse_agent_id`` converts that agent's retained
        occupancy into the new run's slot instead of double-counting it.
        """
        if depth >= self.max_depth:
            raise SubagentDepthDeniedError(depth=depth, max_depth=self.max_depth)
        retained = set(self.retained_idle if retained_idle is None else retained_idle)
        if reuse_agent_id is not None:
            retained.discard(reuse_agent_id)
        running = len(self.slots) + len(retained)
        if running >= cap:
            raise SubagentCapacityDeniedError(running=running, cap=cap)
        slot = _ActiveWorkSlot()
        self.slots.add(slot)
        if reuse_agent_id is not None:
            self.retained_idle.discard(reuse_agent_id)
        return slot

    def release(self, slot: _ActiveWorkSlot) -> None:
        """Release a reserved slot (spawn failure or completion)."""
        self.slots.discard(slot)

    def mark_retained_idle(self, agent_id: str) -> None:
        """A retained idle agent keeps counting against the cap."""
        self.retained_idle.add(agent_id)

    def clear_retained(self, agent_id: str) -> None:
        """Explicit closure only: free a retained idle agent's budget share."""
        self.retained_idle.discard(agent_id)


type SubagentFeedSink = Callable[[Mapping[str, Any]], Awaitable[None]]


def subagent_event_frame(
    agent_id: str, incarnation: int, event: BaseEvent
) -> dict[str, Any]:
    """One child event as a relay frame carrying the run's incarnation stamp."""
    return {
        "jsonrpc": "2.0",
        "method": AGENT_TURN_EVENT_METHOD,
        "params": {"agent_id": agent_id, "event": encode_event(event)},
        RELAY_INCARNATION_FIELD: incarnation,
    }


def subagent_completion_frame(
    agent_id: str, incarnation: int, completion: SubagentCompletion
) -> dict[str, Any]:
    """The terminal completion frame: always the run stream's last frame."""
    return {
        "jsonrpc": "2.0",
        "method": AGENT_TURN_ENDED_METHOD,
        "params": {
            "agent_id": agent_id,
            "outcome": completion.outcome,
            "error": completion.error,
        },
        RELAY_INCARNATION_FIELD: incarnation,
    }


class SubagentEventFeed:
    """Relay one subagent run into the coordinator event feed (WP5 [5B]).

    Every relayed frame carries the run's incarnation stamp
    (``relay_incarnation``, [2A]/[2B]); the stream ends with exactly one
    terminal completion frame. Foreground and background runs both pump
    through this feed, so both deliver the terminal frame.
    """

    def __init__(
        self, agent_id: str, sink: SubagentFeedSink, *, incarnation: int
    ) -> None:
        if type(incarnation) is not int or incarnation < 0:
            raise ValueError("incarnation must be a non-negative integer")
        self._agent_id = agent_id
        self._sink = sink
        self._incarnation = incarnation
        self._terminal_published = False

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @property
    def incarnation(self) -> int:
        return self._incarnation

    @property
    def terminal_published(self) -> bool:
        """Whether this run's single terminal completion frame was relayed."""
        return self._terminal_published

    async def relay_event(self, event: BaseEvent) -> None:
        """Relay one child event as an incarnation-stamped frame."""
        await self._sink(subagent_event_frame(self._agent_id, self._incarnation, event))

    async def relay_completion(self, completion: SubagentCompletion) -> None:
        """Relay the terminal completion frame."""
        await self._sink(
            subagent_completion_frame(self._agent_id, self._incarnation, completion)
        )

    async def publish_terminal(self, completion: SubagentCompletion) -> bool:
        """Publish the run's terminal completion frame exactly once.

        The feed is the durable publication point: the caller publishes only
        after the child's terminal result record exists (save-before-emit in
        [5C]), and the slot is released only after this returns. A repeated
        call is a no-op returning False, so a cancelled consumer and its
        run teardown can never publish two terminal frames for one run.
        """
        if self._terminal_published:
            return False
        self._terminal_published = True
        await self.relay_completion(completion)
        return True

    async def pump(
        self, source: SubagentCompletionSource, request: ExecutionTurnRequest
    ) -> SubagentCompletion:
        """Consume one run's stream, relaying every frame in production order.

        Ends with exactly one terminal completion frame: a frame delivered by
        the source, or — defensively, if a misbehaving source ends without
        one — a synthesized FAILED frame. Returns the terminal completion.
        """
        terminal: SubagentCompletion | None = None
        async with aclosing(source.stream(request)) as stream:
            async for frame in stream:
                if isinstance(frame, SubagentCompletion):
                    terminal = frame
                    await self.publish_terminal(frame)
                else:
                    await self.relay_event(frame)
        if terminal is None:
            terminal = SubagentCompletion(
                outcome=TURN_OUTCOME_FAILED,
                error="completion source ended without a terminal frame",
            )
            await self.publish_terminal(terminal)
        return terminal


class AdmittedSubagent:
    """A spawned subagent admitted under an atomically reserved slot."""

    def __init__(
        self,
        spawned: SpawnedSubagent,
        *,
        slot: _ActiveWorkSlot,
        admission: SubagentAdmission,
        feed: SubagentEventFeed,
        retain: bool,
    ) -> None:
        self._spawned = spawned
        self._slot = slot
        self._admission = admission
        self._feed = feed
        self._retain = retain

    @property
    def agent_id(self) -> str:
        return self._spawned.agent_id

    @property
    def completion_source(self) -> SubagentCompletionSource:
        return self._spawned.completion_source

    @property
    def incarnation(self) -> int:
        """This spawn's incarnation stamp for relay frames."""
        return self._feed.incarnation

    @property
    def feed(self) -> SubagentEventFeed:
        return self._feed

    async def run(self, request: ExecutionTurnRequest) -> SubagentCompletion:
        """Run the admitted subagent, relaying frames and the terminal frame.

        The active slot is released on completion; a retained run instead
        keeps counting against the cap as a retained idle agent until the
        explicit :meth:`close`. The terminal completion frame is published
        (durably, save-before-emit) BEFORE the slot is released: a cancelled
        consumer publishes its own exactly-once INTERRUPTED frame here, so no
        later admission can observe free capacity for a run whose terminal
        publication is still outstanding.
        """
        try:
            return await self._feed.pump(self._spawned.completion_source, request)
        finally:
            try:
                with suppress(Exception):
                    if not self._feed.terminal_published:
                        await self._feed.publish_terminal(
                            SubagentCompletion(outcome=TURN_OUTCOME_INTERRUPTED)
                        )
            finally:
                if self._retain:
                    self._admission.mark_retained_idle(self.agent_id)
                self._admission.release(self._slot)

    async def close(self) -> None:
        """Explicit closure — the only closure kind: reap and free the budget."""
        try:
            await self._spawned.close()
        finally:
            self._admission.release(self._slot)
            self._admission.clear_retained(self.agent_id)


class AdmittingSubagentSpawn:
    """A spawn port wrapped with capacity/depth admission (WP5 [5B]).

    ``spawn`` reserves a slot ATOMICALLY before the physical spawn (typed
    denial on over-cap/over-depth, never silent queueing), releases it when
    the spawn fails, and hands the admitted handle a fresh incarnation-stamped
    event feed. Each spawn is a fresh incarnation: the counter hands out a new
    generation per spawn so a respawned agent's old incarnation is stale.
    """

    def __init__(
        self,
        port: SubagentSpawnPort,
        admission: SubagentAdmission,
        *,
        sink: SubagentFeedSink,
        incarnations: Iterator[int] | None = None,
    ) -> None:
        self._port = port
        self._admission = admission
        self._sink = sink
        self._incarnations = incarnations if incarnations is not None else count(0)

    async def spawn(
        self, spec: SubagentSpawnSpec, *, cap: int, depth: int = 0, retain: bool = False
    ) -> AdmittedSubagent:
        """Admit then spawn; a spawn failure releases the reservation."""
        slot = self._admission.reserve(
            cap=cap, depth=depth, reuse_agent_id=spec.agent_id
        )
        try:
            spawned = await self._port.spawn(spec)
        except BaseException:
            self._admission.release(slot)
            raise
        feed = SubagentEventFeed(
            spec.agent_id, self._sink, incarnation=next(self._incarnations)
        )
        return AdmittedSubagent(
            spawned, slot=slot, admission=self._admission, feed=feed, retain=retain
        )


__all__ = [
    "MAX_SUBAGENT_DEPTH",
    "AdmittedSubagent",
    "AdmittingSubagentSpawn",
    "InProcessChildFactory",
    "InProcessSpawnedSubagent",
    "InProcessSubagentCompletionSource",
    "InProcessSubagentSpawn",
    "SpawnedSubagent",
    "SubagentAdmission",
    "SubagentAdmissionClosedError",
    "SubagentAdmissionDeniedError",
    "SubagentCapacityDeniedError",
    "SubagentCompletion",
    "SubagentCompletionSource",
    "SubagentDepthDeniedError",
    "SubagentEventFeed",
    "SubagentFeedSink",
    "SubagentFrame",
    "SubagentSpawnDeniedError",
    "SubagentSpawnError",
    "SubagentSpawnPort",
    "SubagentSpawnSpec",
    "child_result_store",
    "child_terminal_record",
    "record_child_terminal",
    "spawn_spec_from_wire",
    "spawn_spec_to_wire",
    "subagent_completion_frame",
    "subagent_event_frame",
    "subagent_spawn_port",
]
