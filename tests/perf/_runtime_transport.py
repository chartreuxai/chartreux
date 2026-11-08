"""Transport observation and the B3/B4 notification producer (v0.7.5 stage 2, X2).

The B3/B4 rows are the app-server's own egress cost: how many server-to-client
notifications a completed root turn serializes and how many UTF-8 bytes they
weigh. They are measured on the landed WP3b scenario harness, whose
:class:`~tests.perf._runtime_scenario.NotificationSink` serializes each
notification exactly once at send time and records the count and byte width on
the turn's :class:`~tests.perf._runtime_scenario.NotificationWindow`. This
module reads those records — it never serializes a payload for measurement —
and derives:

* **B3** — the in-window notification COUNT per completed root turn.
* **B4** — the in-window UTF-8 BYTE total per completed root turn.

The window-closure rule is Appendix B's and is asserted, not assumed: the
window opens at the root turn's user-message submission and closes at the
projector's completed-state event for that turn
(:data:`NOTIFICATION_WINDOW_RULE`); every serialized notification between open
and close is counted. Counts are deterministic under the fixed loaded-cell
schedule, so the expected D over the five run-p95s is 0 — recorded as a
property of the cell, with the ``max(6D, 0.20 x M)`` band of the threshold
formula as the protection against a degenerate threshold.

Transport primitives ride the EXISTING production seams (no production change,
no new seam):

* :data:`~chartreux.core.llm.backend.generic.request_start_observer` — the
  gateway arms it right before each backend advance and ``GenericBackend``
  fires it exactly at its transport start. :func:`observe_transport` chains a
  recorder in FRONT of the in-scope observer and delegates to it, so TTFB is
  measured transport-start -> first delivered chunk while the production
  accounting callback still runs.
* ``GenericBackend.on_retry`` — notified before each retry backoff; the
  retry/stall observation (:func:`retry_stall_observation`) drives it through a
  real ``GenericBackend`` over an ``httpx.MockTransport``, so no socket is
  created and the fail-closed egress guard stays closed.

Scope is exactly what a single backend instance observed: the claim is
WITHIN-BACKEND (:data:`CLAIM_SCOPE`) and deployment failover is out of claim
(:data:`FAILOVER_STANDING`) — a switched provider would be a different backend
and is never folded into one B3/B4 number. TTFB and inter-chunk gaps are
recorded as OBSERVATION-ONLY here (plan §4: A2/A3 are X1's gated rows; retry
and stall gaps beyond A3 are observation-only), which is what the records say.

Nothing here changes production behaviour: the transport wrapper is installed
on one test-side backend instance and removed on exit, and every probe is
offline.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
import functools
import os
from pathlib import Path
from time import perf_counter_ns
from typing import Any

import httpx

from chartreux.core.config import ModelConfig, ProviderConfig
from chartreux.core.llm.backend.generic import (
    GenericBackend,
    notify_request_started,
    request_start_observer,
)
from chartreux.core.llm_models import LLMMessage, Role
from chartreux.core.utils import RetryReason
from chartreux.utils.http import ChartreuxAsyncHTTPClient
from tests.mock.utils import mock_llm_chunk
from tests.perf._metrics import machine_context
from tests.perf._runtime_report import (
    COUNT_ROOT_TURNS,
    COVERAGE_ROOT_TURNS,
    METRIC_VERSION,
    build_run_record,
)
from tests.perf._runtime_scenario import (
    CELL_LOADED_HEADLESS,
    HeadlessScenario,
    RootTurnResult,
    ScenarioScale,
    WorkloadSpec,
    build_headless_scenario,
)

TRANSPORT_SCHEMA = "chartreux.runtime_transport.v1"

#: Appendix B's B3/B4 window-closure rule, in the scenario's own spelling.
NOTIFICATION_WINDOW_RULE = "turn_submission->projector_completed_state"
WINDOW_OPEN_EVENT = "root_turn_user_message_submission"
WINDOW_CLOSE_EVENT = "projector_completed_state"
ROOT_TURN_WINDOW_KIND = "root_turn"

#: The two campaign IDs this producer owns.
B3_CAMPAIGN = "B3"
B4_CAMPAIGN = "B4"

#: TTFB boundaries: the production seam when it fired, else the harness's own
#: backend-advance entry (a fake backend has no HTTP transport to announce).
TRANSPORT_START_BOUNDARY = "request_start_observer"
HARNESS_ENTRY_BOUNDARY = "harness_backend_entry"

#: The retry seam (``GenericBackend.on_retry``, generic.py:373/:409-413/:421-431).
RETRY_SEAM = "on_retry"

#: Claim scope: one backend instance, so a deployment failover is out of claim.
CLAIM_SCOPE = "within_backend"
FAILOVER_STANDING = "out_of_claim"

OBSERVATION_ONLY = "observation_only"
A2_A3_OWNERSHIP = "A2/A3 are X1's gated rows; X2 records the transport observation"
RETRY_STALL_REASON = "retry/stall gaps beyond A3 are observation-only (plan §4)"
B3_B4_DETERMINISM = (
    "counts are deterministic under the fixed loaded-cell seed/barrier/load"
    " schedule: the expected D over the five run-p95s is 0, recorded on every"
    " B3/B4 comparison; the max(6D, 0.20 x M) band protects the threshold"
)
MEASUREMENT_SERIALIZATION = "none: B3/B4 read the sink's send-time serialization record"
OPEN_CAMPAIGN_NOTE = "campaign open: this run is a cell attempt awaiting promotion"

#: The probe's own credential name: a value exists only for the probe's body.
PROBE_KEY_ENV = "CHARTREUX_PERF_TRANSPORT_PROBE_KEY"

_NANOSECONDS_PER_MILLISECOND = 1_000_000
_RECORD_PRECISION = 4


def _milliseconds(nanoseconds: float) -> float:
    return nanoseconds / _NANOSECONDS_PER_MILLISECOND


def _rounded(values: Sequence[float]) -> list[float]:
    return [round(value, _RECORD_PRECISION) for value in values]


def utc_now() -> str:
    """Return the run-start stamp in the report's UTC form."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------
# Transport boundary observation (TTFB, inter-chunk gaps, transport stalls)
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ChunkArrival:
    """One chunk's arrival at the transport boundary."""

    ordinal: int
    arrived_ns: int


@dataclass(slots=True)
class TransportRequest:
    """One streaming request as the transport boundary saw it.

    ``entered_ns`` is the gateway's advance into the backend; when the
    production transport-start seam fires, ``transport_starts_ns`` carries the
    real send times and TTFB is measured from the first of them. A second start
    inside one request is a retried transport attempt, so the gaps between
    starts are the request's stall samples.
    """

    ordinal: int
    entered_ns: int
    transport_starts_ns: list[int] = field(default_factory=list)
    arrivals: list[ChunkArrival] = field(default_factory=list)
    finished_ns: int | None = None
    outcome: str = "open"

    @property
    def boundary(self) -> str:
        return (
            TRANSPORT_START_BOUNDARY
            if self.transport_starts_ns
            else HARNESS_ENTRY_BOUNDARY
        )

    @property
    def start_ns(self) -> int:
        return (
            self.transport_starts_ns[0] if self.transport_starts_ns else self.entered_ns
        )

    @property
    def transport_starts(self) -> int:
        return len(self.transport_starts_ns)

    def transport_started(self) -> None:
        self.transport_starts_ns.append(perf_counter_ns())

    def arrival(self) -> None:
        self.arrivals.append(
            ChunkArrival(ordinal=len(self.arrivals) + 1, arrived_ns=perf_counter_ns())
        )

    def finish(self, outcome: str) -> None:
        if self.finished_ns is not None:
            return
        self.finished_ns = perf_counter_ns()
        self.outcome = outcome

    def ttfb_ms(self) -> float | None:
        if not self.arrivals:
            return None
        return _milliseconds(self.arrivals[0].arrived_ns - self.start_ns)

    def gap_ms(self) -> list[float]:
        return [
            _milliseconds(later.arrived_ns - earlier.arrived_ns)
            for earlier, later in zip(self.arrivals, self.arrivals[1:], strict=False)
        ]

    def stall_ms(self) -> list[float]:
        return [
            _milliseconds(later - earlier)
            for earlier, later in zip(
                self.transport_starts_ns, self.transport_starts_ns[1:], strict=False
            )
        ]

    def duration_ms(self) -> float | None:
        if self.finished_ns is None:
            return None
        return _milliseconds(self.finished_ns - self.entered_ns)

    def record(self) -> dict[str, Any]:
        ttfb = self.ttfb_ms()
        duration = self.duration_ms()
        return {
            "request": self.ordinal,
            "boundary": self.boundary,
            "transport_starts": self.transport_starts,
            "chunks": len(self.arrivals),
            "ttfb_ms": None if ttfb is None else round(ttfb, _RECORD_PRECISION),
            "gaps_ms": _rounded(self.gap_ms()),
            "stall_ms": _rounded(self.stall_ms()),
            "duration_ms": (
                None if duration is None else round(duration, _RECORD_PRECISION)
            ),
            "outcome": self.outcome,
        }


@dataclass(slots=True)
class TransportObserver:
    """The streaming requests one backend instance served."""

    attribute: str
    backend: str
    requests: list[TransportRequest] = field(default_factory=list)

    def begin(self) -> TransportRequest:
        request = TransportRequest(
            ordinal=len(self.requests) + 1, entered_ns=perf_counter_ns()
        )
        self.requests.append(request)
        return request

    def ttfb_ms(self) -> list[float]:
        values: list[float] = []
        for request in self.requests:
            ttfb = request.ttfb_ms()
            if ttfb is not None:
                values.append(ttfb)
        return values

    def gaps_ms(self) -> list[float]:
        return [gap for request in self.requests for gap in request.gap_ms()]

    def stall_ms(self) -> list[float]:
        return [stall for request in self.requests for stall in request.stall_ms()]

    def seam_requests(self) -> int:
        return sum(1 for request in self.requests if request.transport_starts)

    def retried_requests(self) -> int:
        return sum(1 for request in self.requests if request.transport_starts > 1)

    def record(self) -> dict[str, Any]:
        return {
            "claim_scope": CLAIM_SCOPE,
            "gated": False,
            "standing": OBSERVATION_ONLY,
            "reason": A2_A3_OWNERSHIP,
            "attribute": self.attribute,
            "backend": self.backend,
            "requests": len(self.requests),
            "seam_requests": self.seam_requests(),
            "harness_entry_requests": len(self.requests) - self.seam_requests(),
            "transport_starts": sum(
                request.transport_starts for request in self.requests
            ),
            "retried_requests": self.retried_requests(),
            "chunks": sum(len(request.arrivals) for request in self.requests),
            "ttfb_ms": _rounded(self.ttfb_ms()),
            "gaps_ms": _rounded(self.gaps_ms()),
            "stall_ms": _rounded(self.stall_ms()),
            "observations": [request.record() for request in self.requests],
        }


def _compose_transport_start(request: TransportRequest) -> Any:
    """Chain this request's recorder in front of the in-scope seam observer.

    The gateway arms ``request_start_observer`` with its own accounting callback
    before each advance; the recorder records the transport start and then
    delegates, so the production callback still runs exactly once per firing.
    """
    delegate = request_start_observer.get()

    def recorded() -> None:
        request.transport_started()
        if delegate is not None:
            delegate()

    return request_start_observer.set(recorded)


@contextmanager
def observe_transport(
    backend: object, *, attribute: str = "complete_streaming"
) -> Iterator[TransportObserver]:
    """Time one backend instance's streaming requests at the transport boundary.

    Installed on the instance (never the class) and removed on exit. The wrapper
    is itself an async generator, so the gateway's ``aclosing`` teardown keeps
    working exactly as it does on the real backend.
    """
    original = getattr(backend, attribute)
    observer = TransportObserver(attribute=attribute, backend=type(backend).__name__)
    instance_attributes = getattr(backend, "__dict__", {})
    had_own_attribute = (
        isinstance(instance_attributes, dict) and attribute in instance_attributes
    )

    @functools.wraps(original)
    async def observed(*args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
        request = observer.begin()
        stream = original(*args, **kwargs)
        outcome = "closed"
        try:
            while True:
                token = _compose_transport_start(request)
                try:
                    chunk = await anext(stream)
                except StopAsyncIteration:
                    break
                finally:
                    request_start_observer.reset(token)
                request.arrival()
                yield chunk
        except GeneratorExit:
            outcome = "aborted"
            raise
        except BaseException:
            outcome = "raised"
            raise
        finally:
            request.finish(outcome)

    setattr(backend, attribute, observed)
    try:
        yield observer
    finally:
        if had_own_attribute:
            setattr(backend, attribute, original)
        else:
            try:
                delattr(backend, attribute)
            except AttributeError:
                pass


class SeamProbeBackend:
    """A streaming backend that announces its transport start on the real seam.

    The deterministic scenario backend has no HTTP transport, so it never fires
    the seam; the probe is what makes the seam observation checkable. With
    ``fires_seam`` off it is the negative control: an unannounced backend must
    be measured from the backend-advance entry instead, and the in-scope
    observer must not be called.
    """

    def __init__(
        self, *, chunks: int = 3, first_byte_ms: float = 5.0, fires_seam: bool = True
    ) -> None:
        self.chunks = chunks
        self.first_byte_ms = first_byte_ms
        self.fires_seam = fires_seam
        self.seam_firings = 0

    async def complete_streaming(self, **_kwargs: Any) -> AsyncGenerator[Any, None]:
        if self.fires_seam:
            notify_request_started()
            self.seam_firings += 1
        await asyncio.sleep(self.first_byte_ms / 1000)
        for index in range(self.chunks):
            yield mock_llm_chunk(content=f"perf-transport-probe-{index}")
            if index + 1 < self.chunks:
                await asyncio.sleep(self.first_byte_ms / 1000)


# --------------------------------------------------------------------------
# Retry and stall observation on the on_retry seam
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RetryEvent:
    """One ``on_retry`` notification: the seam fires before each backoff."""

    ordinal: int
    category: str
    detail: str
    at_ns: int


@dataclass(slots=True)
class RetryStallObserver:
    """Collects the retry seam's notifications for one probe call."""

    events: list[RetryEvent] = field(default_factory=list)

    async def __call__(self, reason: RetryReason) -> None:
        self.events.append(
            RetryEvent(
                ordinal=len(self.events) + 1,
                category=getattr(reason.category, "name", str(reason.category)),
                detail=str(reason.detail),
                at_ns=perf_counter_ns(),
            )
        )

    def stall_ms(self) -> list[float]:
        stamps = [event.at_ns for event in self.events]
        return [
            _milliseconds(later - earlier)
            for earlier, later in zip(stamps, stamps[1:], strict=False)
        ]

    def record(self) -> dict[str, Any]:
        return {
            "seam": RETRY_SEAM,
            "gated": False,
            "standing": OBSERVATION_ONLY,
            "reason": RETRY_STALL_REASON,
            "retries": len(self.events),
            "events": [
                {
                    "ordinal": event.ordinal,
                    "category": event.category,
                    "detail": event.detail,
                }
                for event in self.events
            ],
            "stall_ms": _rounded(self.stall_ms()),
            "ordering": "notified before each retry backoff (generic.py:409-413)",
        }


_PROBE_COMPLETION: dict[str, Any] = {
    "id": "cmpl-perf-transport-probe",
    "object": "chat.completion",
    "model": "perf-transport-probe",
    "created": 1,
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "probe ok"},
        }
    ],
}


def _probe_responder(failures: Sequence[int]) -> Any:
    """Serve one status from ``failures`` per call, then succeed."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        index = calls["n"]
        calls["n"] += 1
        if index >= len(failures):
            return httpx.Response(200, json=_PROBE_COMPLETION)
        status = failures[index]
        body = (
            {"error": {"code": "rate_limit_exceeded", "message": "probe"}}
            if status == 429
            else {"message": "probe"}
        )
        return httpx.Response(status, json=body)

    handler.calls = calls  # type: ignore[attr-defined]
    return handler


def _probe_provider() -> ProviderConfig:
    return ProviderConfig.model_validate({
        "name": "perf-transport-probe",
        "api_base": "http://perf-transport-probe.invalid/v1",
        "api_key_env_var": PROBE_KEY_ENV,
        "backend": "generic",
    })


def _probe_model() -> ModelConfig:
    return ModelConfig.model_validate({
        "name": "perf-transport-probe",
        "provider": "perf-transport-probe",
        "alias": "perf-transport-probe",
    })


@contextmanager
def _fast_backoff() -> Iterator[None]:
    """Collapse the retry backoff inside the probe's own scope.

    The probe observes the seam, never a duration; the restore is immediate so
    no other awaiting code sees the shortened sleep.
    """
    from chartreux.core.utils import retry as retry_module

    original = retry_module.asyncio.sleep

    async def immediate(seconds: float) -> None:
        del seconds
        await original(0)

    retry_module.asyncio.sleep = immediate  # type: ignore[assignment]
    try:
        yield
    finally:
        retry_module.asyncio.sleep = original  # type: ignore[assignment]


async def retry_stall_observation(
    failures: Sequence[int] = (429, 503),
) -> RetryStallObserver:
    """Drive a real ``GenericBackend`` through the ``on_retry`` seam, offline.

    The client is an ``httpx.MockTransport``, so no socket is created and a
    fail-closed egress guard stays closed. The probe asserts that every injected
    failure was retried before it returns the observation.
    """
    handler = _probe_responder(failures)
    client = ChartreuxAsyncHTTPClient(transport=httpx.MockTransport(handler))
    observer = RetryStallObserver()
    backend = GenericBackend(
        client=client,
        provider=_probe_provider(),
        on_retry=observer,
        retry_max_elapsed_time=5.0,
    )
    previously_set = os.environ.get(PROBE_KEY_ENV)
    os.environ[PROBE_KEY_ENV] = "perf-transport-probe"
    try:
        async with backend:
            with _fast_backoff():
                await backend.complete(
                    model=_probe_model(),
                    messages=[
                        LLMMessage(role=Role.user, content="perf transport probe")
                    ],
                )
    finally:
        if previously_set is None:
            os.environ.pop(PROBE_KEY_ENV, None)
        else:
            os.environ[PROBE_KEY_ENV] = previously_set
    attempts = handler.calls["n"]
    if attempts != len(failures) + 1:
        raise AssertionError(
            f"the probe made {attempts} attempts for {len(failures)} failures"
        )
    if len(observer.events) != len(failures):
        raise AssertionError(
            f"the seam reported {len(observer.events)} retries for {len(failures)}"
        )
    return observer


# --------------------------------------------------------------------------
# B3/B4: the per-root-turn notification window
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RootTurnNotification:
    """One completed root turn's in-window serialized notification totals."""

    turn: int
    turn_id: str
    measured: bool
    notifications: int
    notification_bytes: int
    window_ms: float | None

    def record(self) -> dict[str, Any]:
        return {
            "turn": self.turn,
            "turn_id": self.turn_id,
            "measured": self.measured,
            "notifications": self.notifications,
            "notification_bytes": self.notification_bytes,
            "window_ms": (None if self.window_ms is None else round(self.window_ms, 3)),
        }


def root_turn_notifications(
    turns: Sequence[RootTurnResult], *, warmup_turns: int
) -> list[RootTurnNotification]:
    """Extract B3/B4's per-root-turn totals from the completed turn windows.

    A turn whose window never closed (no projector completed-state event) has no
    B3/B4 sample at all — the run is not silently measured on a partial window.
    """
    samples: list[RootTurnNotification] = []
    for turn in turns:
        window = turn.window
        if window.kind != ROOT_TURN_WINDOW_KIND:
            raise AssertionError(
                f"turn {turn.turn} carries a {window.kind!r} window, not a root turn's"
            )
        if window.open:
            raise AssertionError(
                f"the root turn {turn.turn} completed without its"
                " projector completed-state event: the window never closed"
            )
        samples.append(
            RootTurnNotification(
                turn=turn.turn,
                turn_id=turn.turn_id,
                measured=turn.turn >= warmup_turns,
                notifications=window.count,
                notification_bytes=window.wire_bytes,
                window_ms=window.duration_ms,
            )
        )
    return samples


def notification_coverage(samples: Sequence[RootTurnNotification]) -> dict[str, Any]:
    """Return the chunks-coverage block X2 produces for the B3/B4 rows.

    The field X2 owes is the run's counted root turns; a run that counted none
    reports it missing rather than claiming complete coverage of nothing.
    """
    measured = [sample for sample in samples if sample.measured]
    missing = [] if samples else [COUNT_ROOT_TURNS]
    return {
        "unit": COVERAGE_ROOT_TURNS,
        "root_turns": len(samples),
        "warmup_root_turns": len(samples) - len(measured),
        "measured_root_turns": len(measured),
        "notification_windows": len(samples),
        "notifications": sum(sample.notifications for sample in samples),
        "notification_bytes": sum(sample.notification_bytes for sample in samples),
        "notifications_per_root_turn": [sample.notifications for sample in samples],
        "notification_bytes_per_root_turn": [
            sample.notification_bytes for sample in samples
        ],
        "complete": not missing,
        "missing": missing,
    }


def notification_records(
    samples: Sequence[RootTurnNotification],
    *,
    run_id: str,
    run_start_utc: str,
    fixture_hash: str | None = None,
    closure_status: str | None = None,
    closure_note: str | None = None,
    machine: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build and validate the B3 and B4 run records of one measured run.

    The series is the measured turns' in-window counts (B3) or byte totals (B4);
    the warmup turns are declared and never counted, exactly as Appendix B's
    floors require.
    """
    measured = [sample for sample in samples if sample.measured]
    unit_counts = {COUNT_ROOT_TURNS: len(samples)}
    if closure_status is None and closure_note is None:
        closure_note = OPEN_CAMPAIGN_NOTE
    series = {
        B3_CAMPAIGN: [float(sample.notifications) for sample in measured],
        B4_CAMPAIGN: [float(sample.notification_bytes) for sample in measured],
    }
    return {
        campaign_id: build_run_record(
            campaign_id,
            run_id=f"{run_id}-{campaign_id.lower()}",
            run_start_utc=run_start_utc,
            measured_ops=len(measured),
            warmup_ops=len(samples) - len(measured),
            samples=values,
            unit_counts=unit_counts,
            closure_status=closure_status,
            closure_note=closure_note,
            fixture_hash=fixture_hash,
            machine=machine,
        )
        for campaign_id, values in series.items()
    }


# --------------------------------------------------------------------------
# The cell run
# --------------------------------------------------------------------------


@dataclass(slots=True)
class TransportCellRun:
    """One loaded-cell run: its scenario, transport requests and B3/B4 samples."""

    scenario: HeadlessScenario
    spec: WorkloadSpec
    observer: TransportObserver
    samples: list[RootTurnNotification]
    machine: dict[str, Any]
    run_start_utc: str

    def notification_coverage(self) -> dict[str, Any]:
        return notification_coverage(self.samples)

    def notification_records(self, *, run_id: str) -> dict[str, dict[str, Any]]:
        return notification_records(
            self.samples,
            run_id=run_id,
            run_start_utc=self.run_start_utc,
            fixture_hash=self.spec.fixture_hash,
            machine=self.machine,
        )

    def record(self) -> dict[str, Any]:
        return {
            "schema": TRANSPORT_SCHEMA,
            "metric_version": METRIC_VERSION,
            "cell": self.scenario.cell,
            "scale": self.scenario.scale.dimensions(),
            "workload_hash": self.spec.digest(),
            "run_start_utc": self.run_start_utc,
            "window": {
                "rule": NOTIFICATION_WINDOW_RULE,
                "workload_window": self.spec.notification_window,
                "opens_at": WINDOW_OPEN_EVENT,
                "closes_at": WINDOW_CLOSE_EVENT,
            },
            "claim": {
                "scope": CLAIM_SCOPE,
                "failover": FAILOVER_STANDING,
                "backend": type(self.scenario._backend).__name__,
            },
            "transport": self.observer.record(),
            "notification_coverage": self.notification_coverage(),
            "root_turns": [sample.record() for sample in self.samples],
            "properties": {
                "b3_b4_determinism": B3_B4_DETERMINISM,
                "measurement_serialization": MEASUREMENT_SERIALIZATION,
                "a2_a3_ownership": A2_A3_OWNERSHIP,
                "retry_stall": RETRY_STALL_REASON,
                "failover": FAILOVER_STANDING,
            },
        }


async def run_transport_cell(
    *, scale: ScenarioScale | None = None, session_root: Path | None = None
) -> TransportCellRun:
    """Run one loaded headless cell and collect its transport and B3/B4 data.

    The transport observer is installed on the cell's single root backend — the
    backend the gateway actually streams through — for the whole workload, and
    the B3/B4 samples are taken once every root turn's window has closed.
    """
    scenario = await build_headless_scenario(
        CELL_LOADED_HEADLESS, scale=scale, session_root=session_root
    )
    started = utc_now()
    try:
        with observe_transport(scenario._backend) as observer:
            await scenario.run_workload()
            await scenario.settle_transport()
            samples = root_turn_notifications(
                scenario.turns, warmup_turns=scenario.spec.warmup_turns
            )
        return TransportCellRun(
            scenario=scenario,
            spec=scenario.spec,
            observer=observer,
            samples=samples,
            machine=machine_context(),
            run_start_utc=started,
        )
    finally:
        await scenario.close()


__all__ = [
    "A2_A3_OWNERSHIP",
    "B3_B4_DETERMINISM",
    "B3_CAMPAIGN",
    "B4_CAMPAIGN",
    "CLAIM_SCOPE",
    "FAILOVER_STANDING",
    "HARNESS_ENTRY_BOUNDARY",
    "MEASUREMENT_SERIALIZATION",
    "NOTIFICATION_WINDOW_RULE",
    "OBSERVATION_ONLY",
    "OPEN_CAMPAIGN_NOTE",
    "RETRY_SEAM",
    "RETRY_STALL_REASON",
    "ROOT_TURN_WINDOW_KIND",
    "TRANSPORT_SCHEMA",
    "TRANSPORT_START_BOUNDARY",
    "WINDOW_CLOSE_EVENT",
    "WINDOW_OPEN_EVENT",
    "ChunkArrival",
    "RetryEvent",
    "RetryStallObserver",
    "RootTurnNotification",
    "SeamProbeBackend",
    "TransportCellRun",
    "TransportObserver",
    "TransportRequest",
    "notification_coverage",
    "notification_records",
    "observe_transport",
    "retry_stall_observation",
    "root_turn_notifications",
    "run_transport_cell",
    "utc_now",
]
