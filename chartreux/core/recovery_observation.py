"""Internal typed failure-observation carrier (WP0 foundation).

Private by design: the PUBLIC failure envelope is WP1's scope. This module
only captures and preserves typed failure evidence — exception identity, a
bounded cause chain, the low-level failure classification, the attempted
endpoint/request identity when the exception carries it, and the phase and
turn/run identity of the failure — at the points where that evidence is
still available, BEFORE the message-only ``PublicError`` projection discards
it (``chartreux/app_server/_utils.py:public_error``).

Nothing here is exposed through public models, protocol payloads, or tool
results. Raw exception objects are never retained: only bounded,
serializable summaries. A failing capture must never mask the primary
failure, so ``observe_exception`` is total (it cannot raise).

The process-local store exists so the recovery layer can classify observed
failures without log parsing and without keeping exception objects alive
after the turn context is gone. WP1 replaces this carrier's consumers with
the public envelope; the capture points stay. Fields that the current
evidence cannot supply stay ``None`` — null means unavailable, never zero.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
import time
from typing import cast

from chartreux.core.llm.failures import FailureInfo, classify_failure
from chartreux.core.recovery_models import CallEvidence

__all__ = [
    "FailureCause",
    "FailureObservation",
    "FailureObservationStore",
    "clear_observations",
    "observe_exception",
    "preserve_run_observation",
    "record_observation",
    "run_observation",
    "turn_observation",
]

_MAX_CAUSE_DEPTH = 8
_MAX_MESSAGE_CHARS = 500
_MAX_STORED_OBSERVATIONS = 256
_REQUEST_ID_HEADERS = ("x-request-id", "request-id")
#: Arity of an attempted-deployment identity tuple (base_model, provider, wire_name).
_ATTEMPTED_DEPLOYMENT_ARITY = 3


@dataclass(frozen=True, slots=True)
class FailureCause:
    """One bounded link of an exception cause chain."""

    exception_type: str
    message: str


@dataclass(frozen=True, slots=True)
class FailureObservation:
    """Typed failure evidence captured before public projection.

    ``phase`` names where the observation was captured (``"turn"`` for a
    failure handled inside the turn-phase runner — admission, model-call, and
    streaming failures; ``"turn_terminal"`` for a failure while terminalizing
    or finalizing a failed turn; ``"monitor"`` for a background-monitor
    failure). ``classification`` is the low-level classifier's current output
    for the observed exception; WP1 owns the subagent-level classification
    that consumes it.
    """

    exception_type: str
    message: str
    phase: str
    cause_chain: tuple[FailureCause, ...]
    classification: FailureInfo | None
    observed_at: float
    turn_id: str | None = None
    agent_id: str | None = None
    run_id: str | None = None
    request_id: str | None = None
    endpoint: str | None = None
    status: int | None = None
    error_code: str | None = None
    error_type: str | None = None
    # Measured per-logical-call evidence attached by the gateway (WP1):
    # logical-call identity, transport-attempt count, elapsed, the
    # semantic-output boundary, and the retry-budget state. None = the
    # measurement is unavailable, never zero.
    call_evidence: CallEvidence | None = None
    # The attempted deployment (base_model, provider, wire_name) attached by
    # the completion coordinator BEFORE the failed-attempt rollback restores
    # the committed identity (review R11): the deployment that actually
    # served the failed attempt, which may differ from the committed model.
    attempted_deployment: tuple[str, str, str] | None = None


def _bounded_message(exc: BaseException) -> str:
    try:
        text = str(exc).strip()
    except Exception:
        text = ""
    if not text:
        text = type(exc).__name__
    # Redact BEFORE bounding (review R4): truncating first would leave a
    # secret straddling the cut as an unredactable fragment prefix, which
    # no later redaction pass could match (the full value is gone).
    try:
        from chartreux.core.tools.secret_redaction import redact

        text = redact(text)
    except Exception:
        # Fail closed: this text is destined for public carriage, so a
        # redaction failure drops the message (fixed marker) rather than
        # risking an unredacted fragment.
        return "[REDACTED]"
    if len(text) > _MAX_MESSAGE_CHARS:
        text = f"{text[:_MAX_MESSAGE_CHARS]}…[truncated]"
    return text


def _walk_causes(exc: BaseException) -> list[BaseException]:
    """Bounded, cycle-safe cause walk (the exception itself first)."""
    chain: list[BaseException] = [exc]
    seen: set[int] = {id(exc)}
    cause = exc.__cause__
    while cause is not None and len(chain) < _MAX_CAUSE_DEPTH:
        if id(cause) in seen:
            break
        seen.add(id(cause))
        chain.append(cause)
        cause = cause.__cause__
    return chain


def _first_str_attribute(chain: list[BaseException], name: str) -> str | None:
    for error in chain:
        value = getattr(error, name, None)
        if isinstance(value, str) and value:
            return value
    return None


def _extract_request_id(chain: list[BaseException]) -> str | None:
    for error in chain:
        headers = getattr(error, "headers", None)
        if isinstance(headers, Mapping):
            for header in _REQUEST_ID_HEADERS:
                value = headers.get(header)
                if isinstance(value, str) and value:
                    return value
    return _first_str_attribute(chain, "request_id")


def _extract_status(chain: list[BaseException]) -> int | None:
    for error in chain:
        status = getattr(error, "status", None)
        if isinstance(status, int) and not isinstance(status, bool):
            return status
    return None


def _extract_call_evidence(chain: list[BaseException]) -> CallEvidence | None:
    """The first measured call evidence attached anywhere in the chain."""
    for error in chain:
        evidence = getattr(error, "call_evidence", None)
        if isinstance(evidence, CallEvidence):
            return evidence
    return None


def _extract_attempted_deployment(
    chain: list[BaseException],
) -> tuple[str, str, str] | None:
    """The first attempted-deployment identity attached anywhere in the chain."""
    for error in chain:
        identity = getattr(error, "attempted_deployment", None)
        if (
            isinstance(identity, tuple)
            and len(identity) == _ATTEMPTED_DEPLOYMENT_ARITY
            and all(isinstance(part, str) for part in identity)
        ):
            return cast("tuple[str, str, str]", identity)
    return None


def observe_exception(
    exc: BaseException,
    *,
    phase: str,
    turn_id: str | None = None,
    agent_id: str | None = None,
    run_id: str | None = None,
    clock: Callable[[], float] = time.time,
) -> FailureObservation:
    """Build a bounded typed observation of ``exc``.

    Total by contract: classification and attribute extraction failures are
    swallowed so a diagnostic capture can never mask the primary failure.
    """
    chain = _walk_causes(exc)
    try:
        classification = classify_failure(exc)
    except Exception:
        classification = None
    return FailureObservation(
        exception_type=type(exc).__name__,
        message=_bounded_message(exc),
        phase=phase,
        cause_chain=tuple(
            FailureCause(type(error).__name__, _bounded_message(error))
            for error in chain[1:]
        ),
        classification=classification,
        observed_at=clock(),
        turn_id=turn_id,
        agent_id=agent_id,
        run_id=run_id,
        request_id=_extract_request_id(chain),
        endpoint=_first_str_attribute(chain, "endpoint"),
        status=_extract_status(chain),
        error_code=_first_str_attribute(chain, "error_code"),
        error_type=_first_str_attribute(chain, "error_type"),
        call_evidence=_extract_call_evidence(chain),
        attempted_deployment=_extract_attempted_deployment(chain),
    )


class FailureObservationStore:
    """Bounded process-local store of failure observations.

    Indexed by turn id and, when known, by ``(agent_id, run_id)`` so the
    recovery layer can retrieve typed evidence for a failed run after the
    transient turn context is gone. The turn index keeps the latest
    observation per turn; the run index is FIRST-SEEN — the primary
    observation stays the run's authoritative evidence even when a later
    secondary (monitor/finalization) failure is recorded for the same run
    (review R12). Peek semantics: observations stay retrievable until
    evicted (FIFO) or cleared; nothing consumes them in WP0. The bound
    keeps the carrier from growing without limit when automation is off.
    """

    def __init__(self, *, max_observations: int = _MAX_STORED_OBSERVATIONS) -> None:
        if max_observations <= 0:
            raise ValueError("max_observations must be positive")
        self._max = max_observations
        self._observations: list[FailureObservation] = []
        self._by_turn: dict[str, FailureObservation] = {}
        self._by_run: dict[tuple[str, str], FailureObservation] = {}

    def record(self, observation: FailureObservation) -> None:
        """Store one observation, evicting the oldest when over capacity.

        The run index is first-seen: a secondary observation for a run that
        already has primary evidence never displaces it.
        """
        self._observations.append(observation)
        if observation.turn_id is not None:
            self._by_turn[observation.turn_id] = observation
        if observation.agent_id is not None and observation.run_id is not None:
            key = (observation.agent_id, observation.run_id)
            if key not in self._by_run:
                self._by_run[key] = observation
        while len(self._observations) > self._max:
            evicted = self._observations.pop(0)
            if evicted.turn_id is not None:
                if self._by_turn.get(evicted.turn_id) is evicted:
                    del self._by_turn[evicted.turn_id]
            if evicted.agent_id is not None and evicted.run_id is not None:
                key = (evicted.agent_id, evicted.run_id)
                if self._by_run.get(key) is evicted:
                    del self._by_run[key]

    def turn_observation(self, turn_id: str) -> FailureObservation | None:
        """Return the latest observation captured for one turn, if any."""
        return self._by_turn.get(turn_id)

    def run_observation(self, agent_id: str, run_id: str) -> FailureObservation | None:
        """Return the first (primary) observation preserved for one run."""
        return self._by_run.get((agent_id, run_id))

    def clear(self) -> None:
        """Drop every stored observation (diagnostics and test isolation)."""
        self._observations.clear()
        self._by_turn.clear()
        self._by_run.clear()


_default_store = FailureObservationStore()


def record_observation(observation: FailureObservation) -> None:
    """Record one observation in the process-local default store."""
    _default_store.record(observation)


def turn_observation(turn_id: str) -> FailureObservation | None:
    """Return the observation captured for one turn, if any."""
    return _default_store.turn_observation(turn_id)


def run_observation(agent_id: str, run_id: str) -> FailureObservation | None:
    """Return the primary (first) observation preserved for one run, if any."""
    return _default_store.run_observation(agent_id, run_id)


def preserve_run_observation(
    agent_id: str, run_id: str, observation: FailureObservation
) -> FailureObservation:
    """Re-key a turn-captured observation for run-level retrieval.

    The background monitor captures the turn-keyed observation and links it
    to the failed run so downstream recovery can retrieve typed evidence by
    ``(agent_id, run_id)`` after the turn context is gone. Returns the
    stored (run-keyed) observation.
    """
    stored = replace(observation, agent_id=agent_id, run_id=run_id)
    record_observation(stored)
    return stored


def clear_observations() -> None:
    """Clear the process-local default store (diagnostics and tests)."""
    _default_store.clear()
