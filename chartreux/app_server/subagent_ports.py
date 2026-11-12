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
    Sequence,
)
from contextlib import aclosing, suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
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
    CHILD_TEARDOWN_ATTEMPT_UNKNOWN,
    CHILD_TEARDOWN_INCONCLUSIVE,
    CHILD_TEARDOWN_OUTCOMES,
    CHILD_TEARDOWN_REAPED,
    CHILD_TEARDOWN_TOMBSTONED,
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
from chartreux.core.events import BaseEvent, ToolStreamEvent
from chartreux.core.llm.failures import FailureCategory, FailureInfo
from chartreux.core.llm_models import LLMMessage, Role
from chartreux.core.recovery_models import (
    CallEvidence,
    FailureEnvelope,
    build_failure_envelope,
)
from chartreux.core.recovery_observation import FailureObservation, observe_exception
from chartreux.core.subagents import TaskArgs, TaskResult
from chartreux.core.tools.base import InvokeContext
from chartreux.observability.logging import logger

if TYPE_CHECKING:
    from chartreux.core.agent_loop import AgentLoop


class SubagentSpawnError(RuntimeError):
    """Base class for typed subagent-spawn failures."""


class SubagentSpawnDeniedError(SubagentSpawnError):
    """The run's transport selection does not allow this adapter."""


# --- the child failure envelope and its derivation inputs (item 2 B.3) --------

CHILD_FAILURE_FRAME_KEY = "failure"
"""The run-completion frame's failure block: the derived envelope plus its inputs."""

_DEPLOYMENT_ARITY = 3
"""(base_model, provider, wire_name) — the attempted-deployment identity tuple."""


@dataclass(frozen=True, slots=True)
class ChildFailureEvidence:
    """One child run's derived failure envelope and the inputs it came from.

    The child derives the envelope where the evidence lives; the completion
    frame carries BOTH the derived result and the derivation inputs, so the
    coordinator can re-derive it eligibility-equivalently instead of trusting a
    field subset. The inputs matter: the in-process builder consumes
    ``stop_reason`` and ``exclusion_reasons`` to derive the class and
    eligibility, and a subset without ``stop_reason`` would wrongly admit a
    budget-stopped run as a recoverable transient.
    """

    envelope: FailureEnvelope
    derivation_inputs: dict[str, Any]

    def rebuild(self) -> FailureEnvelope:
        """Re-derive the envelope from its inputs, exactly as the child did."""
        inputs = self.derivation_inputs
        return build_failure_envelope(
            observation=_observation_from_inputs(_mapping(inputs.get("observation"))),
            agent_id=str(inputs.get("agent_id") or self.envelope.agent_id),
            run_id=str(inputs.get("run_id") or self.envelope.run_id),
            turn_id=_optional_str(inputs.get("turn_id")),
            turns_used=_optional_int(inputs.get("turns_used")),
            attempted_deployment=_deployment_identity(
                inputs.get("attempted_deployment")
            ),
            configured_model=_optional_str(inputs.get("configured_model")),
            stop_reason=_optional_str(inputs.get("stop_reason")),
            exclusion_reasons=_string_list(inputs.get("exclusion_reasons")),
            tool_calls_present=bool(inputs.get("tool_calls_present")),
            tool_calls_pending=bool(inputs.get("tool_calls_pending")),
            endpoint_correlated=bool(inputs.get("endpoint_correlated")),
        )

    @property
    def eligibility_equivalent(self) -> bool:
        """Whether the parent's rebuild reaches the child's own decision."""
        try:
            rebuilt = self.rebuild()
        except Exception as exc:
            logger.warning("Failed to rebuild the child failure envelope", exc_info=exc)
            return False
        return failure_decision(rebuilt) == failure_decision(self.envelope)


def failure_decision(envelope: FailureEnvelope) -> tuple[str, str, tuple[str, ...]]:
    """The decision triple a rebuild must reproduce: class, eligibility, reasons."""
    return (
        envelope.failure.failure_class.value,
        envelope.recovery.eligibility.value,
        tuple(envelope.recovery.reason_codes),
    )


def encode_child_failure(evidence: ChildFailureEvidence) -> dict[str, Any]:
    """Encode one run's failure evidence as the completion frame's block."""
    return {
        "envelope": evidence.envelope.model_dump(mode="json"),
        "derivation_inputs": dict(evidence.derivation_inputs),
    }


def decode_child_failure(payload: object) -> ChildFailureEvidence | None:
    """Rebuild one run's failure evidence; an absent or malformed block is None."""
    if not isinstance(payload, Mapping):
        return None
    raw_envelope = payload.get("envelope")
    if not isinstance(raw_envelope, Mapping):
        return None
    try:
        envelope = FailureEnvelope.model_validate(dict(raw_envelope))
    except Exception:
        return None
    inputs = payload.get("derivation_inputs")
    return ChildFailureEvidence(
        envelope=envelope, derivation_inputs=dict(_mapping(inputs))
    )


def child_failure_evidence(
    loop: object,
    exc: BaseException,
    *,
    agent_id: str,
    run_id: str,
    turn_id: str | None = None,
    turns_used: int | None = None,
    stop_reason: str | None = None,
    exclusion_reasons: list[str] | None = None,
) -> ChildFailureEvidence:
    """Derive one failed child run's envelope where its evidence still lives.

    The observation, the transcript-visible tool-effect evidence, the attempted
    deployment and the derived envelope are all captured agent-side, because a
    process-backed child's evidence dies with it. Everything the parent's
    re-derivation reads rides the returned inputs.
    """
    observation = observe_exception(
        exc, phase="turn", turn_id=turn_id, agent_id=agent_id, run_id=run_id
    )
    messages = _loop_messages(loop)
    tool_calls_present, tool_calls_pending = _transcript_tool_evidence(messages)
    committed = getattr(loop, "committed_model", None)
    attempted_deployment = observation.attempted_deployment or _committed_deployment(
        committed
    )
    configured_model = _optional_str(
        getattr(getattr(loop, "config", None), "active_model", None)
    )
    derivation_inputs: dict[str, Any] = {
        "agent_id": agent_id,
        "run_id": run_id,
        "turn_id": turn_id,
        "turns_used": turns_used,
        "configured_model": configured_model,
        "attempted_deployment": (
            list(attempted_deployment) if attempted_deployment is not None else None
        ),
        "stop_reason": stop_reason,
        "exclusion_reasons": list(exclusion_reasons)
        if exclusion_reasons is not None
        else None,
        "tool_calls_present": tool_calls_present,
        "tool_calls_pending": tool_calls_pending,
        "endpoint_correlated": False,
        "observation": _observation_inputs(observation),
    }
    envelope = build_failure_envelope(
        observation=observation,
        agent_id=agent_id,
        run_id=run_id,
        turn_id=turn_id,
        turns_used=turns_used,
        attempted_deployment=attempted_deployment,
        configured_model=configured_model,
        stop_reason=stop_reason,
        exclusion_reasons=exclusion_reasons,
        tool_calls_present=tool_calls_present,
        tool_calls_pending=tool_calls_pending,
        endpoint_correlated=False,
    )
    return ChildFailureEvidence(envelope=envelope, derivation_inputs=derivation_inputs)


def _loop_messages(loop: object) -> list[LLMMessage]:
    messages = getattr(loop, "messages", None)
    return list(messages) if isinstance(messages, (list, tuple)) else []


def _transcript_tool_evidence(messages: Sequence[LLMMessage]) -> tuple[bool, bool]:
    """Transcript-visible tool-effect inputs: (requested, pending).

    Mirrors the in-process monitor's derivation: assistant ``tool_calls``
    request effects and ``Role.tool`` messages settle them by id. Absence of
    tool calls is not proof of no effects — hooks, jobs and child launches are
    not transcript-visible — so "none requested" stays UNKNOWN, never SETTLED.
    """
    requested: set[str] = set()
    settled: set[str] = set()
    for message in messages:
        if message.role is Role.assistant:
            for call in message.tool_calls or ():
                if call.id:
                    requested.add(call.id)
        elif message.role is Role.tool and message.tool_call_id:
            settled.add(message.tool_call_id)
    return bool(requested), bool(requested - settled)


def _committed_deployment(committed: object) -> tuple[str, str, str] | None:
    if committed is None:
        return None
    base_model = getattr(committed, "base_model", None)
    provider = getattr(committed, "provider", None)
    wire_name = getattr(committed, "wire_name", None)
    if (
        isinstance(base_model, str)
        and isinstance(provider, str)
        and isinstance(wire_name, str)
    ):
        return (base_model, provider, wire_name)
    return None


def _observation_inputs(observation: FailureObservation) -> dict[str, Any]:
    """The serializable form of one observation: the parent's derivation input."""
    classification = observation.classification
    evidence = observation.call_evidence
    return {
        "exception_type": observation.exception_type,
        "message": observation.message,
        "phase": observation.phase,
        "observed_at": observation.observed_at,
        "classification": (
            {
                "category": classification.category.value,
                "status": classification.status,
                "error_code": classification.error_code,
                "error_type": classification.error_type,
                "retry_after": classification.retry_after,
            }
            if classification is not None
            else None
        ),
        "call_evidence": (
            {
                "logical_call_id": evidence.logical_call_id,
                "started_at": evidence.started_at,
                "budget_seconds": evidence.budget_seconds,
                "response_started": evidence.response_started,
                "semantic_output_published": evidence.semantic_output_published,
                "transport_attempts": evidence.transport_attempts,
                "elapsed_seconds": evidence.elapsed_seconds,
                "budget_exhausted": evidence.budget_exhausted,
            }
            if evidence is not None
            else None
        ),
        "attempted_deployment": (
            list(observation.attempted_deployment)
            if observation.attempted_deployment is not None
            else None
        ),
        "endpoint": observation.endpoint,
        "status": observation.status,
        "error_code": observation.error_code,
        "error_type": observation.error_type,
        "turn_id": observation.turn_id,
        "agent_id": observation.agent_id,
        "run_id": observation.run_id,
    }


def _observation_from_inputs(inputs: Mapping[str, Any]) -> FailureObservation:
    """Rebuild the observation a parent-side re-derivation reads."""
    raw_evidence = _mapping(inputs.get("call_evidence"))
    call_evidence = None
    if raw_evidence:
        logical_call_id = _optional_str(raw_evidence.get("logical_call_id"))
        started_at = raw_evidence.get("started_at")
        if logical_call_id is not None and isinstance(started_at, (int, float)):
            call_evidence = CallEvidence(
                logical_call_id=logical_call_id,
                started_at=float(started_at),
                budget_seconds=_optional_float(raw_evidence.get("budget_seconds")),
                response_started=_optional_bool(raw_evidence.get("response_started")),
                semantic_output_published=_optional_bool(
                    raw_evidence.get("semantic_output_published")
                ),
                transport_attempts=_optional_int(
                    raw_evidence.get("transport_attempts")
                ),
                elapsed_seconds=_optional_float(raw_evidence.get("elapsed_seconds")),
                budget_exhausted=_optional_bool(raw_evidence.get("budget_exhausted")),
            )
    classification = None
    raw_classification = _mapping(inputs.get("classification"))
    category = _optional_str(raw_classification.get("category"))
    if category is not None:
        try:
            classification = FailureInfo(
                FailureCategory(category),
                status=_optional_int(raw_classification.get("status")),
                error_code=_optional_str(raw_classification.get("error_code")),
                error_type=_optional_str(raw_classification.get("error_type")),
                retry_after=_optional_float(raw_classification.get("retry_after")),
            )
        except ValueError:
            classification = None
    status = inputs.get("status")
    observed_at = inputs.get("observed_at")
    return FailureObservation(
        exception_type=str(inputs.get("exception_type") or "unknown"),
        message=str(inputs.get("message") or ""),
        phase=str(inputs.get("phase") or "unknown"),
        cause_chain=(),
        classification=classification,
        observed_at=float(observed_at)
        if isinstance(observed_at, (int, float))
        else 0.0,
        turn_id=_optional_str(inputs.get("turn_id")),
        agent_id=_optional_str(inputs.get("agent_id")),
        run_id=_optional_str(inputs.get("run_id")),
        request_id=None,
        endpoint=_optional_str(inputs.get("endpoint")),
        status=status if isinstance(status, int) else None,
        error_code=_optional_str(inputs.get("error_code")),
        error_type=_optional_str(inputs.get("error_type")),
        call_evidence=call_evidence,
        attempted_deployment=_deployment_identity(inputs.get("attempted_deployment")),
    )


def _deployment_identity(value: object) -> tuple[str, str, str] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    parts = list(value)
    if len(parts) != _DEPLOYMENT_ARITY or not all(
        isinstance(part, str) for part in parts
    ):
        return None
    return (parts[0], parts[1], parts[2])


def _string_list(value: object) -> list[str] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    return [item for item in value if isinstance(item, str)]


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    return float(value) if isinstance(value, (int, float)) else None


def _optional_bool(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


@dataclass(frozen=True, slots=True)
class SubagentSpawnSpec:
    """JSON-safe spawn request for one subagent; crosses the process wire.

    ``attempt_id`` is the coordinator-minted identity of the publication
    attempt this spawn belongs to. It rides the spawn request so the child can
    persist it at its FIRST durable write: a torn persist-to-link window, or a
    lost aggregate reply, stays attributable to its attempt instead of leaking
    an adoptable transcript.
    """

    agent_id: str
    parent_session_id: str | None = None
    session_dir: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    attempt_id: str | None = None


@dataclass(frozen=True, slots=True)
class SubagentCompletion:
    """Terminal outcome of one subagent run, as the final stream frame.

    ``failure`` carries the run's typed failure evidence when the run failed
    inside its own process: the child derives the envelope where the evidence
    lives and the frame carries it (item 2 B.3), because the ``outcome``/
    ``error`` strings alone cannot reconstruct the measured call evidence and
    parent-side eligibility would fail closed for every transient family.
    """

    outcome: str
    error: str | None = None
    failure: ChildFailureEvidence | None = None


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


CHILD_HANDOFF_PAYLOAD_KEY = "file_state_handoff"
"""Spawn-payload key carrying the parent's rendered working-set handoff."""

CHILD_TEARDOWN_REASON = "publication-teardown"
"""Tombstone reason: an attempt whose aggregate outcome was never observed."""


class ChildTeardownOutcome(StrEnum):
    """What a conservative publication teardown ESTABLISHED, never claimed."""

    REAPED = CHILD_TEARDOWN_REAPED
    """The attempt held nothing durable to retract, and its process is gone."""

    TOMBSTONED = CHILD_TEARDOWN_TOMBSTONED
    """The attempt's persisted records were retracted before removal."""

    INCONCLUSIVE = CHILD_TEARDOWN_INCONCLUSIVE
    """The attempt could not be settled inside the bounded rollback wait."""

    ATTEMPT_UNKNOWN = CHILD_TEARDOWN_ATTEMPT_UNKNOWN
    """This end holds no attempt under that id (settled, or never admitted)."""


@dataclass(frozen=True, slots=True)
class ChildTeardownRequest:
    """One request to settle a publication attempt whose outcome was lost.

    The coordinator mints the attempt id, so it still holds the identity after
    a lost reply; the request asks the process that owns the children to settle
    the attempt conservatively. ``wait_seconds`` bounds how long the owner
    waits for the attempt's own atomic rollback before it must answer
    inconclusive.
    """

    attempt_id: str
    reason: str = CHILD_TEARDOWN_REASON
    agent_id: str | None = None
    wait_seconds: float | None = None

    def to_wire(self) -> dict[str, Any]:
        return {
            "attempt_id": self.attempt_id,
            "reason": self.reason,
            "agent_id": self.agent_id,
            "wait_seconds": self.wait_seconds,
        }

    @classmethod
    def from_wire(cls, params: Mapping[str, Any]) -> ChildTeardownRequest:
        attempt_id = params.get("attempt_id")
        if not isinstance(attempt_id, str) or not attempt_id:
            raise ValueError("a child teardown request carries no attempt id")
        wait = params.get("wait_seconds")
        return cls(
            attempt_id=attempt_id,
            reason=_optional_str(params.get("reason")) or CHILD_TEARDOWN_REASON,
            agent_id=_optional_str(params.get("agent_id")),
            wait_seconds=(
                float(wait)
                if isinstance(wait, (int, float)) and not isinstance(wait, bool)
                else None
            ),
        )


@dataclass(frozen=True, slots=True)
class ChildTeardownReport:
    """What a conservative teardown of one publication attempt established.

    ``retracted`` records whether a tombstone was written, so a caller never
    has to read a reason string to know whether the attempt's records can
    resurrect. ``settled`` is False for the inconclusive class: an attempt the
    owner could not reap inside the bounded wait is never reported as rolled
    back.
    """

    attempt_id: str
    outcome: ChildTeardownOutcome
    agent_id: str | None = None
    session_id: str | None = None
    child_dir: str | None = None
    retracted: bool = False
    detail: str | None = None

    @property
    def settled(self) -> bool:
        """Whether the attempt is provably settled (never a claimed rollback)."""
        return self.outcome in {
            ChildTeardownOutcome.REAPED,
            ChildTeardownOutcome.TOMBSTONED,
        }

    def to_wire(self) -> dict[str, Any]:
        return {
            "attempt_id": self.attempt_id,
            "outcome": self.outcome.value,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "child_dir": self.child_dir,
            "retracted": self.retracted,
            "detail": self.detail,
        }

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any]) -> ChildTeardownReport:
        raw_outcome = payload.get("outcome")
        if raw_outcome not in CHILD_TEARDOWN_OUTCOMES:
            raise ValueError("a child teardown report carries no known outcome")
        attempt_id = payload.get("attempt_id")
        if not isinstance(attempt_id, str) or not attempt_id:
            raise ValueError("a child teardown report carries no attempt id")
        return cls(
            attempt_id=attempt_id,
            outcome=ChildTeardownOutcome(str(raw_outcome)),
            agent_id=_optional_str(payload.get("agent_id")),
            session_id=_optional_str(payload.get("session_id")),
            child_dir=_optional_str(payload.get("child_dir")),
            retracted=bool(payload.get("retracted")),
            detail=_optional_str(payload.get("detail")),
        )


class ChildPublicationTeardownPort(Protocol):
    """Settles a publication attempt this end minted, after a lost outcome."""

    async def teardown_attempt(
        self, request: ChildTeardownRequest
    ) -> ChildTeardownReport: ...


class AgentOwnedSubagentRunner:
    """The turn offer a coordinator makes for a process-owned root.

    A stdio session's root agent process owns its children: the child launch
    resolves its candidate, renders the working-set handoff, and records the
    parent link against the root loop, and only that process holds the loop.
    So this port never runs a child itself -- it is what the offer means, and
    the agent builds the runner it owns when the coordinator offers it. A host
    run that still reaches this end (an agent holding no child host of its
    own) answers typed instead of streaming an empty result.
    """

    def run(
        self, args: TaskArgs, ctx: InvokeContext
    ) -> AsyncGenerator[ToolStreamEvent | TaskResult, None]:
        return self._refuse()

    async def _refuse(self) -> AsyncGenerator[ToolStreamEvent | TaskResult, None]:
        raise SubagentSpawnDeniedError(
            "This turn's children are owned by the root agent process, which "
            "serves no host subagent run; restart with "
            "CHARTREUX_TRANSPORT=in_process to run children in one process."
        )
        yield  # pragma: no cover - unreachable; makes this an async generator


def spawn_spec_to_wire(spec: SubagentSpawnSpec) -> dict[str, Any]:
    """Encode one spawn spec as the ``subagent/open`` params payload."""
    return {
        "agent_id": spec.agent_id,
        "parent_session_id": spec.parent_session_id,
        "session_dir": spec.session_dir,
        "payload": spec.payload,
        "attempt_id": spec.attempt_id,
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
        attempt_id=_optional_str(params.get("attempt_id")),
    )


def child_spawn_spec(
    *,
    agent_id: str,
    parent_session_id: str | None,
    root_session_dir: Path,
    agent_name: str | None = None,
    payload: Mapping[str, Any] | None = None,
    attempt_id: str | None = None,
) -> SubagentSpawnSpec:
    """Build the spawn request for one FRESH child process (WP5 [5C]).

    The child creates its own transcript directory under
    ``<root>/agents/<agent>_<ts>_<id>`` and writes it itself, so the spec
    carries the ROOT session dir and the agent name instead of a child
    directory a previous spawn may have used. A later child process is
    therefore a new incarnation with a new transcript record, never a resume.

    ``attempt_id`` is the publication attempt the spawn belongs to; the child
    persists it at its first durable write, which is what makes a torn
    persist-to-link window (and a lost aggregate reply) attributable.
    """
    merged: dict[str, Any] = dict(payload or {})
    merged[ROOT_SESSION_DIR_PAYLOAD_KEY] = str(root_session_dir)
    merged[AGENT_NAME_PAYLOAD_KEY] = agent_name or agent_id
    return SubagentSpawnSpec(
        agent_id=agent_id,
        parent_session_id=parent_session_id,
        payload=merged,
        attempt_id=attempt_id,
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
    """The terminal completion frame: always the run stream's last frame.

    A failed run's typed failure evidence (the derived envelope plus the inputs
    it came from) rides this frame, so a process-backed child's measured
    evidence reaches the coordinator instead of dying with the child.
    """
    params: dict[str, Any] = {
        "agent_id": agent_id,
        "outcome": completion.outcome,
        "error": completion.error,
    }
    if completion.failure is not None:
        params[CHILD_FAILURE_FRAME_KEY] = encode_child_failure(completion.failure)
    return {
        "jsonrpc": "2.0",
        "method": AGENT_TURN_ENDED_METHOD,
        "params": params,
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
    "CHILD_FAILURE_FRAME_KEY",
    "CHILD_HANDOFF_PAYLOAD_KEY",
    "CHILD_TEARDOWN_REASON",
    "MAX_SUBAGENT_DEPTH",
    "AdmittedSubagent",
    "AdmittingSubagentSpawn",
    "ChildFailureEvidence",
    "ChildPublicationTeardownPort",
    "ChildTeardownOutcome",
    "ChildTeardownReport",
    "ChildTeardownRequest",
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
    "child_failure_evidence",
    "child_result_store",
    "child_spawn_spec",
    "child_terminal_record",
    "decode_child_failure",
    "encode_child_failure",
    "failure_decision",
    "record_child_terminal",
    "spawn_spec_from_wire",
    "spawn_spec_to_wire",
    "subagent_completion_frame",
    "subagent_event_frame",
    "subagent_spawn_port",
]
