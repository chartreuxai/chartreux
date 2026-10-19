"""R2 — bounded automatic root-turn recovery (plan state.md R2, OD3).

The harness schedules the continuation turn itself after an authoritative
TERMINAL retryable root model-call failure — bounded and cooldown-aware, with
the manual ``/retry`` affordance (R1) as the fallback on exhaustion. Two
collaborators: the TUI controller (``cli/textual_ui/root_recovery.py`` — an
in-process timer owning local cancellation via an input-generation token) and
the app-server recovery gate (``app_server/_root_recovery.py`` —
authoritative eligibility, episode accounting, single-flight admission).

This file covers the gate's cooldown-aware scheduling primitives and the
acceptance items that are provable at the gate level: the NON-MUTATING
availability snapshot (never claims or releases another caller's probe),
the shared episode budget merging the legacy incomplete-stream continuation,
the strict eligibility matrix, the completed-actions warning on the
continuation, the advisory episode deadline, and the overnight notice
readability. The race matrix (dedup, single-flight, user-input cancellation,
barriers, invalidation, ambiguous delivery, meta-recovery) lives in
``test_root_recovery_races.py``; config validation in
``test_root_recovery_config.py``.

Harness: a real app-server session over the in-memory transport drives real
root turns against a scripted controlled transport (``tests/recovery/
conftest.py``); the shared availability registry runs on the opt-in
``fake_clock``; the gate's jitter draw is pinned to the exact midpoint so
every scheduled delay is exact. All transports are controlled fakes — an
unmocked provider call cannot occur (fail-closed egress).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import httpx
import pytest

from chartreux.app_server._projector import ProjectedUpdate
from chartreux.app_server._root_recovery import (
    RootRecoveryGate,
    RootRecoverySessionView,
)
from chartreux.app_server.models import PublicTurnStatus, RootRecoveryNoticeDetail
from chartreux.app_server.protocol import (
    RootRecoveryAdmitParams,
    RootRecoveryCancelParams,
    RootRecoveryScheduleParams,
)
from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
from chartreux.core.config.models import RecoveryEligibleClass, RootRecoveryConfig
from chartreux.core.llm.exceptions import BackendError, PayloadSummary
from chartreux.core.llm.failures import FailureCategory, FailureInfo
from chartreux.core.model_catalog.availability import (
    AvailabilityRegistry,
    AvailabilitySnapshot,
    ExclusionReason,
    availability_snapshot,
)
from chartreux.core.model_catalog.loader import CatalogSnapshot
from chartreux.core.model_catalog.schema import ModelCatalog
from chartreux.core.recovery_models import (
    CallEvidence,
    FailureClass,
    FailurePhase,
    RootTurnFailure,
    attach_call_evidence,
    build_root_turn_failure,
)
from chartreux.core.recovery_observation import observe_exception
from chartreux.core.session_types import CommittedModelIdentity
from tests.recovery.conftest import FakeClock

ENDPOINT_URL = "https://api.shared.test/v1"
ENDPOINT_IDENTITY = "api.shared.test"
RATE_LIMIT_SIGNATURE = "rate_limit:429:rate_limit:rate_limit"


# ---------------------------------------------------------------------------
# The non-mutating availability snapshot
# ---------------------------------------------------------------------------


def _catalog() -> CatalogSnapshot:
    return CatalogSnapshot(
        ModelCatalog.model_validate({
            "providers": {
                "test-first": {"api_base": ENDPOINT_URL},
                "test-second": {"api_base": "https://second.test"},
            },
            "models": {
                "base": {
                    "deployments": [
                        {
                            "provider": "test-first",
                            "name": "first",
                            "supported_thinking_levels": ["off"],
                        },
                        {
                            "provider": "test-second",
                            "name": "second",
                            "supported_thinking_levels": ["off"],
                        },
                    ]
                }
            },
            "roles": {},
        }),
        "test",
    )


def _config(snapshot: CatalogSnapshot) -> ChartreuxConfigSchema:
    return ChartreuxConfigSchema(active_model="base").attach_catalog_snapshot(snapshot)


def _committed() -> CommittedModelIdentity:
    return CommittedModelIdentity(
        base_model="base",
        provider="test-first",
        wire_name="first",
        catalog_revision="test",
    )


def _record(
    registry: AvailabilityRegistry,
    base_model: str,
    provider: str,
    *,
    endpoint: str = ENDPOINT_URL,
    signature: str = RATE_LIMIT_SIGNATURE,
) -> None:
    registry.record_failure(
        base_model,
        provider,
        FailureInfo(FailureCategory.RATE_LIMIT, status=429),
        endpoint=endpoint,
        signature=signature,
    )


def test_snapshot_available_when_a_deployment_survives(fake_clock: FakeClock) -> None:
    snapshot_catalog = _catalog()
    registry = AvailabilityRegistry(clock=fake_clock)

    snapshot = availability_snapshot(
        snapshot=snapshot_catalog,
        committed=_committed(),
        registry=registry,
        config=_config(snapshot_catalog),
        thinking="off",
    )

    assert snapshot.available is True
    assert snapshot.permanent is False
    assert snapshot.endpoint_blocked is False
    assert snapshot.probe_in_flight is False
    assert snapshot.next_check_delay == 0.0


def test_snapshot_never_claims_or_releases_a_probe(fake_clock: FakeClock) -> None:
    """Item 9: eligibility reads never claim or release another caller's probe.

    A snapshot taken while another caller's single-flight probe is claimed
    reports probe occupancy and changes nothing: the probe stays owned by its
    caller, and a snapshot taken after a real claim still sees it.
    """
    snapshot_catalog = _catalog()
    registry = AvailabilityRegistry(clock=fake_clock)
    committed = _committed()

    # Another caller claims the first deployment's probe through the real
    # admission path (a probe is the single-flight claim on a deployment
    # whose cooldown has expired).
    registry.record_failure(
        committed.base_model,
        "test-first",
        FailureInfo(FailureCategory.TIMEOUT),
        endpoint=ENDPOINT_URL,
    )
    fake_clock.advance(30.0)
    admitted, probe, token = registry.admission(
        committed.base_model, "test-first", endpoint=ENDPOINT_IDENTITY
    )
    assert admitted is True and probe is True and token is not None

    snap = availability_snapshot(
        snapshot=snapshot_catalog,
        committed=committed,
        registry=registry,
        config=_config(snapshot_catalog),
        thinking="off",
    )
    # The claimed deployment reads unavailable (probe occupancy), the other
    # deployment still serves, and the snapshot itself did not touch the
    # claim: the probe is still in flight afterwards.
    assert snap.available is True
    assert registry.probe_in_flight(committed.base_model, "test-first") is True

    # A cooldown-armed deployment reports the temporary gap and probe
    # occupancy separately (the first deployment's probe is still claimed
    # from the admission above — a second claim is refused, proving the
    # snapshot read never released it).
    registry.record_failure(
        committed.base_model,
        "test-second",
        FailureInfo(FailureCategory.TIMEOUT),
        endpoint="https://second.test",
    )
    refused = registry.admission(
        committed.base_model, "test-first", endpoint=ENDPOINT_IDENTITY
    )
    assert refused == (False, False, None)
    snap = availability_snapshot(
        snapshot=snapshot_catalog,
        committed=committed,
        registry=registry,
        config=_config(snapshot_catalog),
        thinking="off",
    )
    assert snap.available is False
    assert snap.permanent is False
    assert snap.probe_in_flight is True
    # The probe-occupied deployment's cooldown has expired, so its gap is 0
    # (the caller floors a probe-only wait with availability_recheck_seconds);
    # the cooling-down second deployment contributes its own 30s gap. The
    # earliest temporary exclusion wins.
    assert snap.next_check_delay == 0.0
    # The read-only check released nothing: both claims survive.
    assert registry.probe_in_flight(committed.base_model, "test-first") is True
    assert registry.cooldown_until(committed.base_model, "test-second") == 60.0


def test_snapshot_reports_cooldown_gap_for_waiting(fake_clock: FakeClock) -> None:
    snapshot_catalog = _catalog()
    registry = AvailabilityRegistry(clock=fake_clock)
    committed = _committed()
    registry.record_failure(
        committed.base_model,
        "test-first",
        FailureInfo(FailureCategory.TIMEOUT),
        endpoint=ENDPOINT_URL,
    )

    snap = availability_snapshot(
        snapshot=snapshot_catalog,
        committed=committed,
        registry=registry,
        config=_config(snapshot_catalog),
        thinking="off",
    )
    # The first deployment is cooling down; the second still serves, so the
    # snapshot is available (a healthy same-base deployment may serve via
    # the existing failover — R2 never chooses another base model).
    assert snap.available is True
    assert snap.permanent is False

    # With BOTH deployments cooling down, the snapshot waits: the gap is the
    # EARLIEST temporary exclusion, never a permanent stop.
    registry.record_failure(
        committed.base_model,
        "test-second",
        FailureInfo(FailureCategory.TIMEOUT),
        endpoint="https://second.test",
    )
    snap = availability_snapshot(
        snapshot=snapshot_catalog,
        committed=committed,
        registry=registry,
        config=_config(snapshot_catalog),
        thinking="off",
    )
    assert snap.available is False
    assert snap.permanent is False
    assert snap.endpoint_blocked is False
    assert snap.next_check_delay == pytest.approx(30.0)
    fake_clock.advance(30.0)
    snap = availability_snapshot(
        snapshot=snapshot_catalog,
        committed=committed,
        registry=registry,
        config=_config(snapshot_catalog),
        thinking="off",
    )
    assert snap.available is True


def test_snapshot_endpoint_correlation_blocks_and_gaps(fake_clock: FakeClock) -> None:
    """Endpoint-wide correlation excludes every affected deployment (item 8).

    Cross-model rate-limit evidence on ONE endpoint arms an endpoint-wide
    cooldown: the snapshot reports endpoint_blocked with the endpoint
    cooldown's gap, and the exclusion is checked FIRST (before the
    per-deployment chain).
    """
    snapshot_catalog = _catalog()
    registry = AvailabilityRegistry(
        clock=fake_clock, initial_cooldown=30.0, endpoint_cooldown=45.0
    )
    _record(registry, "model-a", "provider-1")
    _record(registry, "model-b", "provider-2")
    assert registry.endpoint_correlation_active(ENDPOINT_IDENTITY)

    snap = availability_snapshot(
        snapshot=snapshot_catalog,
        committed=_committed(),
        registry=registry,
        config=_config(snapshot_catalog),
        thinking="off",
    )
    # test-first is on the correlated endpoint; test-second is not, so the
    # snapshot is still available through the unaffected deployment.
    assert snap.available is True

    # A catalog where EVERY deployment shares the correlated endpoint.
    single = CatalogSnapshot(
        ModelCatalog.model_validate({
            "providers": {"test-first": {"api_base": ENDPOINT_URL}},
            "models": {
                "base": {
                    "deployments": [
                        {
                            "provider": "test-first",
                            "name": "first",
                            "supported_thinking_levels": ["off"],
                        }
                    ]
                }
            },
            "roles": {},
        }),
        "test",
    )
    snap = availability_snapshot(
        snapshot=single,
        committed=_committed(),
        registry=registry,
        config=_config(single),
        thinking="off",
    )
    assert snap.available is False
    assert snap.permanent is False
    assert snap.endpoint_blocked is True
    assert snap.next_check_delay == pytest.approx(45.0)
    assert all(
        exclusion.reason is ExclusionReason.ENDPOINT_COOLDOWN
        for exclusion in snap.exclusions
    )


def test_snapshot_rejects_permanent_incompatibility(fake_clock: FakeClock) -> None:
    """Item 10: permanent incompatibility is not auto-retryable.

    A disabled deployment (or allowlist/images/thinking exclusions) is a
    PERMANENT exclusion: the snapshot reports permanent=True so the gate
    declines instead of waiting.
    """
    snapshot_catalog = CatalogSnapshot(
        ModelCatalog.model_validate({
            "providers": {"test-first": {"api_base": ENDPOINT_URL}},
            "models": {
                "base": {
                    "deployments": [
                        {
                            "provider": "test-first",
                            "name": "first",
                            "supported_thinking_levels": ["off"],
                            "disabled": True,
                        }
                    ]
                }
            },
            "roles": {},
        }),
        "test",
    )
    registry = AvailabilityRegistry(clock=fake_clock)

    snap = availability_snapshot(
        snapshot=snapshot_catalog,
        committed=_committed(),
        registry=registry,
        config=_config(snapshot_catalog),
        thinking="off",
    )
    assert snap.available is False
    assert snap.permanent is True
    assert snap.endpoint_blocked is False
    assert snap.next_check_delay == 0.0
    assert snap.exclusions[0].reason is ExclusionReason.DISABLED


def test_snapshot_is_a_pure_read(fake_clock: FakeClock) -> None:
    """The snapshot mutates NOTHING: repeated reads are identical and leave
    the registry byte-for-byte where it was (no cooldown, no probe, no
    endpoint event).
    """
    snapshot_catalog = _catalog()
    registry = AvailabilityRegistry(clock=fake_clock)
    committed = _committed()
    registry.record_failure(
        committed.base_model,
        "test-first",
        FailureInfo(FailureCategory.TIMEOUT),
        endpoint=ENDPOINT_URL,
    )

    first = availability_snapshot(
        snapshot=snapshot_catalog,
        committed=committed,
        registry=registry,
        config=_config(snapshot_catalog),
        thinking="off",
    )
    second = availability_snapshot(
        snapshot=snapshot_catalog,
        committed=committed,
        registry=registry,
        config=_config(snapshot_catalog),
        thinking="off",
    )
    assert first == second
    assert first == AvailabilitySnapshot(
        available=True,
        permanent=False,
        endpoint_blocked=False,
        probe_in_flight=False,
        next_check_delay=0.0,
        exclusions=first.exclusions,
    )
    # The cooldown the failure armed is still armed exactly once.
    assert registry.cooldown_until(committed.base_model, "test-first") == 30.0
    assert registry.cooldown_until(committed.base_model, "test-second") is None


# ---------------------------------------------------------------------------
# The typed terminal projection (build_root_turn_failure) — eligibility evidence
# ---------------------------------------------------------------------------


def _payload_summary() -> PayloadSummary:
    return PayloadSummary(
        model="first",
        message_count=1,
        approx_chars=2,
        temperature=0.0,
        has_tools=False,
        tool_choice=None,
    )


def _timeout_error() -> BackendError:
    """A streaming timeout fault with its httpx cause (D2: classifies TIMEOUT)."""
    error = BackendError(
        provider="test-first",
        endpoint="https://api.shared.test/v1/chat",
        status=None,
        reason="read timed out",
        headers=None,
        body_text=None,
        parsed_error="Network error",
        model="first",
        payload_summary=_payload_summary(),
    )
    error.__cause__ = httpx.ReadTimeout("read timed out")
    return error


def _measured_evidence() -> CallEvidence:
    return CallEvidence(
        logical_call_id="call-1",
        started_at=0.0,
        budget_seconds=900.0,
        response_started=True,
        semantic_output_published=False,
        transport_attempts=2,
        elapsed_seconds=1.5,
        budget_exhausted=False,
    )


def _project(
    observation: Any, turn_id: str = "turn-1", **kwargs: Any
) -> RootTurnFailure:
    return build_root_turn_failure(
        observation=observation, turn_id=turn_id, committed=_committed(), **kwargs
    )


def test_projection_confirms_a_measured_transient_model_call() -> None:
    """A measured transient model-call failure carries the trusted-origin
    confirmation the gate's eligibility requires (items 1/5 evidence).
    """
    error = _timeout_error()
    attach_call_evidence(error, _measured_evidence())
    failure = _project(observe_exception(error, phase="turn", turn_id="turn-1"))

    assert failure.cause_category is FailureCategory.TIMEOUT
    assert failure.failure_class is FailureClass.PROVIDER_STALL
    assert failure.phase is FailurePhase.LLM_CALL
    assert failure.transient_cause_confirmed is True
    assert failure.attempted_base_model == "base"
    assert failure.attempted_provider == "test-first"
    assert failure.attempted_wire_name == "first"
    assert failure.endpoint == ENDPOINT_IDENTITY
    assert failure.logical_call_id == "call-1"
    assert failure.transport_attempts == 2
    assert failure.call_elapsed_seconds == 1.5


def test_projection_transport_exception_without_call_evidence_is_not_a_model_failure() -> (
    None
):
    """Item 6: a tool's network exception carries no measured call evidence
    and is never a root model-call failure — the trusted-origin confirmation
    is exactly what separates the two.
    """
    error = _timeout_error()
    failure = _project(observe_exception(error, phase="turn", turn_id="turn-1"))

    assert failure.cause_category is FailureCategory.TIMEOUT
    assert failure.failure_class is FailureClass.PROVIDER_STALL
    assert failure.transient_cause_confirmed is False
    assert failure.logical_call_id is None
    assert failure.transport_attempts is None
    assert failure.call_elapsed_seconds is None


def test_projection_finalization_and_monitor_phases_are_not_eligible() -> None:
    """Item 6: finalization/monitor failures never qualify, even with call
    evidence attached (a transport-like finalization exception is not a model
    failure).
    """
    error = _timeout_error()
    attach_call_evidence(error, _measured_evidence())

    finalize = _project(observe_exception(error, phase="turn_terminal", turn_id="t"))
    assert finalize.phase is FailurePhase.FINALIZE
    assert finalize.transient_cause_confirmed is False

    monitor = _project(observe_exception(error, phase="monitor", turn_id="t"))
    assert monitor.phase is FailurePhase.MONITOR
    assert monitor.failure_class is FailureClass.INTERNAL
    assert monitor.transient_cause_confirmed is False


@pytest.mark.parametrize("stop_reason", ["user_cancelled", "budget_exceeded"])
def test_projection_protected_stop_reasons_dominate(stop_reason: str) -> None:
    """Item 5 evidence: a protected stop reason dominates even a measured
    transient cause.
    """
    error = _timeout_error()
    attach_call_evidence(error, _measured_evidence())
    failure = _project(
        observe_exception(error, phase="turn", turn_id="turn-1"),
        stop_reason=stop_reason,
    )

    assert failure.failure_class is FailureClass.CANCELLATION
    assert failure.stop_reason == stop_reason


# ---------------------------------------------------------------------------
# The recovery gate — the authoritative decision matrix
# ---------------------------------------------------------------------------


def _failure(**overrides: Any) -> RootTurnFailure:
    """One eligible-by-default typed terminal projection (the gate consumes
    projections directly; the builder pins above cover their derivation).
    """
    base: dict[str, Any] = {
        "turn_id": "turn-1",
        "observed_at": 1000.0,
        "cause_category": FailureCategory.TIMEOUT,
        "phase": FailurePhase.LLM_CALL,
        "failure_class": FailureClass.PROVIDER_STALL,
        "transient_cause_confirmed": True,
        "attempted_base_model": "base",
        "attempted_provider": "test-first",
        "attempted_wire_name": "first",
        "endpoint": ENDPOINT_URL,
        "logical_call_id": "call-1",
        "transport_attempts": 2,
        "call_elapsed_seconds": 1.5,
    }
    base.update(overrides)
    return RootTurnFailure(**base)


class _GateHarness:
    """A directly-constructed gate over fake providers — fully deterministic.

    The jitter draw is pinned to the exact midpoint (``jittered_backoff`` with
    draw 0.5 returns the base unchanged), so every scheduled delay is exact.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        auto_retries_max: int = 2,
        deadline_seconds: float = 2400.0,
        cooldowns: tuple[float, ...] = (30.0, 120.0),
        recheck: float = 1.0,
        eligible_classes: list[RecoveryEligibleClass] | None = None,
        stage_raises: bool = False,
        availability_raises: bool = False,
        emit_raises: bool = False,
    ) -> None:
        self.clock = FakeClock(1000.0)
        self.view = RootRecoverySessionView(
            session_id="session-1",
            active_turn_id=None,
            queued_work=False,
            open_callbacks=False,
            stop_requested=False,
            handoff_pending=False,
            lifecycle_reserved=False,
            closing=False,
        )
        self.snapshot = AvailabilitySnapshot(
            available=True,
            permanent=False,
            endpoint_blocked=False,
            probe_in_flight=False,
            next_check_delay=0.0,
        )
        self.config = RootRecoveryConfig(
            enabled=enabled,
            auto_retries_max=auto_retries_max,
            deadline_seconds=deadline_seconds,
            cooldowns_seconds=list(cooldowns),
            availability_recheck_seconds=recheck,
            eligible_classes=(
                eligible_classes
                if eligible_classes is not None
                else list(RecoveryEligibleClass)
            ),
        )
        self.staged: list[tuple[RootRecoveryNoticeDetail, str]] = []
        self._stage_raises = stage_raises
        self._availability_raises = availability_raises
        self._emit_raises = emit_raises
        self.gate = RootRecoveryGate(
            config=lambda: self.config,
            session_view=lambda: self.view,
            availability=self._availability,
            stage_notice=self._stage_notice,
            emit_projected=self._emit_projected,
        )
        self.gate._clock = self.clock
        self.gate._jitter_draw = lambda: 0.5

    def _availability(self) -> AvailabilitySnapshot:
        if self._availability_raises:
            raise RuntimeError("availability read failed")
        return self.snapshot

    def _stage_notice(
        self, detail: RootRecoveryNoticeDetail, message: str
    ) -> ProjectedUpdate | None:
        if self._stage_raises:
            raise RuntimeError("notice staging failed")
        self.staged.append((detail, message))
        return cast(ProjectedUpdate, object())

    async def _emit_projected(self, update: ProjectedUpdate) -> None:
        if self._emit_raises:
            raise RuntimeError("notice delivery failed")

    def capture(self, failure: RootTurnFailure) -> None:
        self.gate.capture_terminal(failure)

    async def schedule(
        self, turn_id: str = "turn-1", *, input_generation: int = 0
    ) -> Any:
        return await self.gate.schedule(
            RootRecoveryScheduleParams(
                session_id="session-1",
                terminal_turn_id=turn_id,
                solicited=True,
                input_generation=input_generation,
            )
        )

    async def admit(
        self,
        episode_id: str,
        idempotency_key: str,
        turn_id: str = "turn-1",
        *,
        input_generation: int = 0,
    ) -> Any:
        return await self.gate.admit(
            RootRecoveryAdmitParams(
                session_id="session-1",
                episode_id=episode_id,
                idempotency_key=idempotency_key,
                terminal_turn_id=turn_id,
                input_generation=input_generation,
            )
        )

    async def cancel(self, episode_id: str, *, reason: str = "") -> Any:
        return await self.gate.cancel(
            RootRecoveryCancelParams(
                session_id="session-1", episode_id=episode_id, reason=reason
            )
        )

    def notices(self, milestone: str) -> list[tuple[RootRecoveryNoticeDetail, str]]:
        return [entry for entry in self.staged if entry[0].milestone == milestone]


@pytest.mark.asyncio
async def test_gate_schedules_admits_and_recovers_a_terminal_timeout() -> None:
    """Item 1 (gate half) + item 14: a terminal root timeout schedules, admits,
    and starts a continuation; the started notice carries the successor turn
    id and the preserved-conversation warning.
    """
    harness = _GateHarness()
    harness.capture(_failure())

    scheduled = await harness.schedule()
    assert scheduled.decision == "scheduled"
    assert scheduled.attempt == 1
    assert scheduled.limit == 2
    assert scheduled.episode_id == "root-auto-turn-1"
    assert scheduled.idempotency_key == "root-auto-recovery:root-auto-turn-1:1"
    # due = max(configured backoff 30, no Retry-After, available snapshot) +
    # bounded jitter — the pinned midpoint draw makes the jitter exactly zero.
    assert scheduled.delay_seconds == 30.0
    assert scheduled.eligible_class == "timeout"
    assert scheduled.affected_deployment == "test-first/first"

    ((detail, message),) = harness.notices("scheduled")
    assert detail.episode_id == "root-auto-turn-1"
    assert detail.attempt == 1
    assert detail.limit == 2
    assert detail.delay_seconds == 30.0
    assert detail.eligible_class == "timeout"
    assert detail.affected_deployment == "test-first/first"
    assert "Automatic retry 1 of 2 scheduled in 30.0s" in message
    assert "timeout" in message
    assert "test-first/first" in message
    assert "No user action is required" in message

    admitted = await harness.admit(scheduled.episode_id, scheduled.idempotency_key)
    assert admitted.decision == "admitted"
    assert admitted.attempt == 1
    assert admitted.limit == 2
    assert admitted.predecessor_turn_id == "turn-1"

    updates = harness.gate.observe_turn_started("turn-2", scheduled.idempotency_key)
    assert len(updates) == 1
    ((started, started_message),) = harness.notices("started")
    assert started.successor_turn_id == "turn-2"
    assert started.predecessor_turn_id == "turn-1"
    assert "started as turn turn-2" in started_message
    # Item 14: the completed-actions warning rides the continuation.
    assert "Completed tool results are kept; recent actions may repeat." in (
        started_message
    )

    await harness.gate.observe_turn_finished("turn-2", PublicTurnStatus.COMPLETED)
    ((recovered, recovered_message),) = harness.notices("recovered")
    assert recovered.successor_turn_id == "turn-2"
    assert "succeeded" in recovered_message

    # The episode ended: a late duplicate observation never re-schedules.
    again = await harness.schedule()
    assert again.decision == "declined"
    assert again.reason == "episode_ended"


@pytest.mark.asyncio
async def test_gate_one_schedule_per_failed_turn() -> None:
    """Item 2 (gate half): solicited and unsolicited observations of the same
    terminal turn share ONE schedule.
    """
    harness = _GateHarness()
    harness.capture(_failure())

    first = await harness.schedule()
    second = await harness.schedule()

    assert first.decision == "scheduled"
    assert second.decision == "already_scheduled"
    assert second.episode_id == first.episode_id
    assert second.idempotency_key == first.idempotency_key
    assert second.attempt == first.attempt
    assert second.delay_seconds == first.delay_seconds
    assert len(harness.notices("scheduled")) == 1


@pytest.mark.asyncio
async def test_gate_single_flight_admission_never_creates_two_successors() -> None:
    """Items 3/16 (gate half): duplicate admits and duplicate start acks
    reconcile to the original successor — never a second one, never a loop.
    """
    harness = _GateHarness()
    harness.capture(_failure())
    scheduled = await harness.schedule()
    key = scheduled.idempotency_key
    episode_id = scheduled.episode_id

    admitted = await harness.admit(episode_id, key)
    assert admitted.decision == "admitted"

    # An ambiguous redelivery before the successor start is observed.
    redelivered = await harness.admit(episode_id, key)
    assert redelivered.decision == "already_admitted"
    assert redelivered.attempt == 1
    assert redelivered.turn_id is None

    updates = harness.gate.observe_turn_started("turn-2", key)
    assert len(updates) == 1
    # A duplicate start ack: no second successor, no second notice.
    assert harness.gate.observe_turn_started("turn-2", key) == []
    assert len(harness.notices("started")) == 1

    # A redelivery after the linkage reconciles to the original successor.
    reconciled = await harness.admit(episode_id, key)
    assert reconciled.decision == "already_admitted"
    assert reconciled.turn_id == "turn-2"

    # A foreign turn start while the successor is in flight does NOT cancel
    # it (the admitted call settles via normal interruption rules).
    assert harness.gate.observe_turn_started("turn-9", None) == []
    await harness.gate.observe_turn_finished("turn-2", PublicTurnStatus.COMPLETED)
    assert len(harness.notices("recovered")) == 1


@pytest.mark.asyncio
async def test_gate_foreign_turn_start_cancels_a_pending_schedule() -> None:
    """A non-continuation turn while a schedule is pending is a takeover: the
    schedule is cancelled and the reason shown — never silently rearmed.
    """
    harness = _GateHarness()
    harness.capture(_failure())
    scheduled = await harness.schedule()

    updates = harness.gate.observe_turn_started("turn-9", None)
    assert len(updates) == 1
    ((cancelled, _message),) = harness.notices("cancelled")
    assert cancelled.reason == "superseded_by_turn"

    declined = await harness.admit(scheduled.episode_id, scheduled.idempotency_key)
    assert declined.decision == "declined"
    assert declined.reason == "no_episode"


@pytest.mark.asyncio
async def test_gate_alternating_failure_kinds_share_one_budget() -> None:
    """Item 4: alternating incomplete-stream/provider failures share ONE
    count (and one deadline) — the merged legacy budget.
    """
    harness = _GateHarness()
    harness.capture(
        _failure(
            cause_category=FailureCategory.INCOMPLETE_STREAM,
            failure_class=FailureClass.INCOMPLETE_STREAM,
        )
    )
    first = await harness.schedule()
    assert first.decision == "scheduled"
    assert first.attempt == 1
    assert first.eligible_class == "incomplete_stream"

    admitted = await harness.admit(first.episode_id, first.idempotency_key)
    assert admitted.decision == "admitted"
    harness.gate.observe_turn_started("turn-2", first.idempotency_key)

    # The successor fails with a DIFFERENT transient kind: same episode.
    harness.capture(_failure(turn_id="turn-2", observed_at=1010.0))
    second = await harness.schedule("turn-2")
    assert second.decision == "scheduled"
    assert second.episode_id == first.episode_id
    assert second.attempt == 2
    assert second.idempotency_key != first.idempotency_key

    admitted = await harness.admit(second.episode_id, second.idempotency_key, "turn-2")
    assert admitted.decision == "admitted"
    assert admitted.attempt == 2
    harness.gate.observe_turn_started("turn-3", second.idempotency_key)

    # A third failure exhausts the ONE shared budget.
    harness.capture(
        _failure(
            turn_id="turn-3",
            observed_at=1020.0,
            cause_category=FailureCategory.RATE_LIMIT,
            failure_class=FailureClass.PROVIDER_RATE_LIMIT,
        )
    )
    declined = await harness.schedule("turn-3")
    assert declined.decision == "declined"
    assert declined.reason == "budget_exhausted"
    assert declined.limit == 2

    ((exhausted, exhausted_message),) = harness.notices("exhausted")
    assert exhausted.attempt == 2
    assert exhausted.reason == "budget_exhausted"
    assert "manual Retry" in exhausted_message
    assert len(harness.notices("exhausted")) == 1


@pytest.mark.asyncio
async def test_gate_manual_retry_starts_a_new_episode_and_never_replenishes() -> None:
    """A manual user retry starts a NEW user-directed episode (keeping its
    intentional cooldown reset) and NEVER replenishes the automatic budget.
    """
    harness = _GateHarness()
    harness.capture(_failure())
    first = await harness.schedule()
    await harness.admit(first.episode_id, first.idempotency_key)
    harness.gate.observe_turn_started("turn-2", first.idempotency_key)
    harness.capture(_failure(turn_id="turn-2", observed_at=1010.0))
    second = await harness.schedule("turn-2")
    await harness.admit(second.episode_id, second.idempotency_key, "turn-2")
    harness.gate.observe_turn_started("turn-3", second.idempotency_key)
    harness.capture(_failure(turn_id="turn-3", observed_at=1020.0))
    exhausted = await harness.schedule("turn-3")
    assert exhausted.reason == "budget_exhausted"

    # A fresh USER turn (a manual retry's turn) is a fresh episode with a
    # fresh budget — the automatic one is not replenished.
    harness.capture(_failure(turn_id="turn-9", observed_at=1030.0))
    fresh = await harness.schedule("turn-9")
    assert fresh.decision == "scheduled"
    assert fresh.episode_id == "root-auto-turn-9"
    assert fresh.episode_id != first.episode_id
    assert fresh.attempt == 1


@pytest.mark.asyncio
async def test_gate_config_reload_never_extends_an_active_episode() -> None:
    """Episode policy is snapshotted at creation: a config reload never
    extends an active episode or replenishes its count.
    """
    harness = _GateHarness()
    harness.capture(_failure())
    first = await harness.schedule()
    await harness.admit(first.episode_id, first.idempotency_key)
    harness.gate.observe_turn_started("turn-2", first.idempotency_key)

    # A reload that would replenish the budget mid-episode.
    harness.config = harness.config.model_copy(update={"auto_retries_max": 10})

    harness.capture(_failure(turn_id="turn-2", observed_at=1010.0))
    second = await harness.schedule("turn-2")
    assert second.decision == "scheduled"
    assert second.attempt == 2
    assert second.limit == 2  # the snapshotted policy, not the reloaded one


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("overrides", "expected_hint"),
    [
        (
            {
                "failure_class": FailureClass.PREFLIGHT,
                "transient_cause_confirmed": False,
                "cause_category": None,
            },
            "preflight",
        ),
        (
            {
                "cause_category": FailureCategory.AUTH,
                "failure_class": FailureClass.AUTHORIZATION,
                "transient_cause_confirmed": False,
            },
            "authorization",
        ),
        # A budget stop is a protected cancellation even though its cause
        # category is transient — the protected CLASS is an absolute
        # exclusion (a stop-reason-driven cancellation is still one).
        (
            {
                "stop_reason": "budget_exceeded",
                "failure_class": FailureClass.CANCELLATION,
                "transient_cause_confirmed": True,
            },
            "cancellation",
        ),
        (
            {
                "cause_category": FailureCategory.CANCELLATION,
                "failure_class": FailureClass.CANCELLATION,
                "transient_cause_confirmed": False,
            },
            "cancellation",
        ),
        (
            {
                "failure_class": FailureClass.TASK,
                "transient_cause_confirmed": False,
                "cause_category": None,
            },
            "task",
        ),
        (
            {
                "cause_category": FailureCategory.BILLING,
                "failure_class": FailureClass.PERMANENT_CONFIG,
                "transient_cause_confirmed": False,
            },
            "permanent_config",
        ),
        (
            {
                "cause_category": FailureCategory.CONTEXT_SIZE,
                "failure_class": FailureClass.PERMANENT_CONFIG,
                "transient_cause_confirmed": False,
            },
            "permanent_config",
        ),
        (
            {
                "phase": FailurePhase.MONITOR,
                "failure_class": FailureClass.INTERNAL,
                "transient_cause_confirmed": False,
                "cause_category": None,
            },
            "internal",
        ),
        (
            {
                "failure_class": FailureClass.UNKNOWN,
                "cause_category": None,
                "transient_cause_confirmed": False,
            },
            "unknown",
        ),
        # Item 6: a measured transient category WITHOUT measured provenance
        # (a tool's network exception) is never eligible.
        ({"transient_cause_confirmed": False}, "provider_stall"),
    ],
)
async def test_gate_never_schedules_protected_or_unproven_failures(
    overrides: dict[str, Any], expected_hint: str
) -> None:
    """Item 5 + item 6: preflight, permission/authorization denial, budget
    stop, cancellation, task blocker, permanent config/context-size,
    unknown-internal, and unproven transients NEVER schedule.
    """
    harness = _GateHarness()
    harness.capture(_failure(**overrides))

    declined = await harness.schedule()

    assert declined.decision == "declined"
    assert declined.reason == "not_eligible"
    assert declined.eligible_class == expected_hint
    assert harness.staged == []


@pytest.mark.asyncio
async def test_gate_requires_a_typed_projection_and_enabled_class() -> None:
    """Never schedule without the typed projection; an eligible class outside
    the configured eligible_classes never schedules either.
    """
    harness = _GateHarness()
    declined = await harness.schedule()
    assert declined.decision == "declined"
    assert declined.reason == "no_typed_projection"

    harness = _GateHarness(eligible_classes=[RecoveryEligibleClass.INCOMPLETE_STREAM])
    harness.capture(_failure())
    declined = await harness.schedule()
    assert declined.decision == "declined"
    assert declined.reason == "class_not_enabled"
    assert declined.eligible_class == "timeout"
    assert harness.staged == []


@pytest.mark.asyncio
async def test_gate_declines_unsettled_effects_and_interaction_barriers() -> None:
    """Effect/interaction barriers captured BEFORE callback cancellation gate
    the schedule (their absence afterwards would prove nothing).
    """
    harness = _GateHarness()
    harness.capture(_failure(effects_settled=False))
    declined = await harness.schedule()
    assert declined.decision == "declined"
    assert declined.reason == "effects_unsettled"

    harness = _GateHarness()
    harness.capture(_failure(interaction_barrier=True, barrier_detail="question open"))
    declined = await harness.schedule()
    assert declined.decision == "declined"
    assert declined.reason == "interaction_barrier: question open"


@pytest.mark.asyncio
async def test_gate_delay_waits_for_cooldown_retry_after_and_endpoint() -> None:
    """Item 8: due = max(configured backoff, candidate cooldown,
    endpoint cooldown, applicable Retry-After) + bounded jitter.
    """
    # A temporary all-deployments-unavailable gap postpones the schedule.
    harness = _GateHarness()
    harness.capture(_failure())
    harness.snapshot = AvailabilitySnapshot(
        available=False,
        permanent=False,
        endpoint_blocked=False,
        probe_in_flight=False,
        next_check_delay=45.0,
    )
    scheduled = await harness.schedule()
    assert scheduled.decision == "scheduled"
    assert scheduled.delay_seconds == 45.0

    # Endpoint correlation postpones it the same way (checked FIRST).
    harness = _GateHarness()
    harness.capture(_failure())
    harness.snapshot = AvailabilitySnapshot(
        available=False,
        permanent=False,
        endpoint_blocked=True,
        probe_in_flight=False,
        next_check_delay=45.0,
    )
    scheduled = await harness.schedule()
    assert scheduled.decision == "scheduled"
    assert scheduled.delay_seconds == 45.0

    # An applicable Retry-After dominates the configured backoff.
    harness = _GateHarness()
    harness.capture(_failure(retry_after_seconds=90.0))
    scheduled = await harness.schedule()
    assert scheduled.decision == "scheduled"
    assert scheduled.delay_seconds == 90.0

    # A partially elapsed Retry-After contributes only its remainder.
    harness = _GateHarness()
    harness.capture(_failure(observed_at=990.0, retry_after_seconds=90.0))
    scheduled = await harness.schedule()
    assert scheduled.decision == "scheduled"
    assert scheduled.delay_seconds == 80.0


@pytest.mark.asyncio
async def test_gate_admission_waits_out_a_cooldown_without_consuming_a_retry() -> None:
    """Item 8 + item 9: at admission the LIVE availability is re-read; a
    probe in flight or a cooldown gap defers within the episode and a
    read-only check NEVER consumes a retry.
    """
    harness = _GateHarness()
    harness.capture(_failure())
    scheduled = await harness.schedule()

    # A probe in flight at admission: defer, floor the recheck wait.
    harness.snapshot = AvailabilitySnapshot(
        available=False,
        permanent=False,
        endpoint_blocked=False,
        probe_in_flight=True,
        next_check_delay=0.0,
    )
    deferred = await harness.admit(scheduled.episode_id, scheduled.idempotency_key)
    assert deferred.decision == "deferred"
    assert deferred.attempt == 1
    assert deferred.delay_seconds == 1.0  # availability_recheck_seconds floor

    # A cooldown gap: defer with the gap as the recheck delay.
    harness.snapshot = AvailabilitySnapshot(
        available=False,
        permanent=False,
        endpoint_blocked=False,
        probe_in_flight=False,
        next_check_delay=45.0,
    )
    deferred = await harness.admit(scheduled.episode_id, scheduled.idempotency_key)
    assert deferred.decision == "deferred"
    assert deferred.delay_seconds == 45.0

    # Availability returns: the SAME attempt is admitted — the deferred
    # read-only waits consumed no retry.
    harness.snapshot = AvailabilitySnapshot(
        available=True,
        permanent=False,
        endpoint_blocked=False,
        probe_in_flight=False,
        next_check_delay=0.0,
    )
    admitted = await harness.admit(scheduled.episode_id, scheduled.idempotency_key)
    assert admitted.decision == "admitted"
    assert admitted.attempt == 1
    assert admitted.limit == 2


@pytest.mark.asyncio
async def test_gate_all_unavailable_waits_for_temporary_rejects_permanent() -> None:
    """Item 10: all-deployments-unavailable waits when the exclusions are
    transient/cooldown-backed; permanent incompatibility is not
    auto-retryable.
    """
    harness = _GateHarness()
    harness.capture(_failure())
    harness.snapshot = AvailabilitySnapshot(
        available=False,
        permanent=False,
        endpoint_blocked=False,
        probe_in_flight=False,
        next_check_delay=30.0,
    )
    scheduled = await harness.schedule()
    assert scheduled.decision == "scheduled"
    assert scheduled.delay_seconds == 30.0

    harness = _GateHarness()
    harness.capture(_failure())
    harness.snapshot = AvailabilitySnapshot(
        available=False,
        permanent=True,
        endpoint_blocked=False,
        probe_in_flight=False,
        next_check_delay=0.0,
    )
    declined = await harness.schedule()
    assert declined.decision == "declined"
    assert declined.reason == "permanent_incompatibility"
    # An undriven episode emits nothing; the manual R1 fallback is the path.
    assert harness.staged == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "expected_reason"),
    [
        ("open_callbacks", "open_callbacks"),
        ("queued_work", "queued_work"),
        ("stop_requested", "stop_requested"),
        ("handoff_pending", "handoff_pending"),
        ("lifecycle_reserved", "lifecycle_reserved"),
        ("closing", "closing"),
    ],
)
async def test_gate_no_admission_with_a_live_interaction_barrier(
    field: str, expected_reason: str
) -> None:
    """Item 12 + item 13: no admission while any runtime approval,
    user-question callback, queued user work, unsettled predecessor cleanup,
    or reconciliation barrier exists. The barrier is re-read LIVE at admission
    — an expired callback or an old grant answer is never consent, and a
    clean schedule-time state does not carry over.
    """
    harness = _GateHarness()
    harness.capture(_failure())
    scheduled = await harness.schedule()
    assert scheduled.decision == "scheduled"

    harness.view = replace(harness.view, **{field: True})
    declined = await harness.admit(scheduled.episode_id, scheduled.idempotency_key)
    assert declined.decision == "declined"
    assert declined.reason == expected_reason

    ((cancelled, _message),) = harness.notices("cancelled")
    assert cancelled.reason == expected_reason


@pytest.mark.asyncio
async def test_gate_no_admission_with_an_active_turn_or_stale_input_generation() -> (
    None
):
    """Items 11/12: an active turn blocks admission, and user input that
    arrived after the schedule cancels it (never admits).
    """
    harness = _GateHarness()
    harness.capture(_failure())
    scheduled = await harness.schedule(input_generation=3)
    assert scheduled.decision == "scheduled"

    harness.view = replace(harness.view, active_turn_id="turn-9")
    declined = await harness.admit(
        scheduled.episode_id, scheduled.idempotency_key, input_generation=3
    )
    assert declined.decision == "declined"
    assert declined.reason == "turn_active"

    harness = _GateHarness()
    harness.capture(_failure())
    scheduled = await harness.schedule(input_generation=3)
    declined = await harness.admit(
        scheduled.episode_id, scheduled.idempotency_key, input_generation=4
    )
    assert declined.decision == "declined"
    assert declined.reason == "user_input"
    ((cancelled, _message),) = harness.notices("cancelled")
    assert cancelled.reason == "user_input"

    # A key mismatch is declined without ending the episode: the correct
    # admission still works afterwards.
    harness = _GateHarness()
    harness.capture(_failure())
    scheduled = await harness.schedule()
    declined = await harness.admit(scheduled.episode_id, "root-auto-recovery:wrong")
    assert declined.decision == "declined"
    assert declined.reason == "key_mismatch"
    admitted = await harness.admit(scheduled.episode_id, scheduled.idempotency_key)
    assert admitted.decision == "admitted"


@pytest.mark.asyncio
async def test_gate_session_mismatch_never_schedules_or_admits() -> None:
    harness = _GateHarness()
    harness.capture(_failure())
    harness.view = replace(harness.view, session_id="session-2")
    declined = await harness.schedule()
    assert declined.decision == "declined"
    assert declined.reason == "session_changed"


@pytest.mark.asyncio
async def test_gate_episode_deadline_blocks_new_admissions() -> None:
    """Item 19: no successor is admitted after the episode deadline. The
    deadline anchors at the initial failed logical call's START (capture time
    minus the measured call elapsed) — not at the observation.
    """
    harness = _GateHarness(deadline_seconds=100.0)
    # observed_at=1000, call_elapsed=40 → started_at=960 → deadline_at=1060.
    harness.capture(_failure(observed_at=1000.0, call_elapsed_seconds=40.0))
    scheduled = await harness.schedule()
    assert scheduled.decision == "scheduled"
    admitted = await harness.admit(scheduled.episode_id, scheduled.idempotency_key)
    assert admitted.decision == "admitted"
    harness.gate.observe_turn_started("turn-2", scheduled.idempotency_key)

    # Past the deadline (1061 > 1060): the linked successor's own failure
    # cannot schedule another continuation.
    harness.clock.advance(61.0)
    harness.capture(
        _failure(turn_id="turn-2", observed_at=1061.0, call_elapsed_seconds=5.0)
    )
    declined = await harness.schedule("turn-2")
    assert declined.decision == "declined"
    assert declined.reason == "deadline_exceeded"

    ((exhausted, exhausted_message),) = harness.notices("exhausted")
    assert exhausted.reason == "deadline_exceeded"
    assert "deadline" in exhausted_message
    assert "manual Retry" in exhausted_message


@pytest.mark.asyncio
async def test_gate_episode_deadline_blocks_admission_but_never_kills_in_flight() -> (
    None
):
    """Item 19: the deadline is an ADVISORY admission bound — distinguished
    from interrupting a running request: an admitted successor in flight
    still recovers normally after the deadline passes.
    """
    harness = _GateHarness(deadline_seconds=100.0)
    harness.capture(_failure(observed_at=1000.0, call_elapsed_seconds=40.0))
    scheduled = await harness.schedule()

    # The admission itself is refused once the deadline has passed (the
    # deadline anchors at the call START: 960 + 100 = 1060, not 1100).
    harness.clock.advance(61.0)
    declined = await harness.admit(scheduled.episode_id, scheduled.idempotency_key)
    assert declined.decision == "declined"
    assert declined.reason == "deadline_exceeded"

    # An admitted successor in flight is never killed by the deadline.
    harness = _GateHarness(deadline_seconds=100.0)
    harness.capture(_failure(observed_at=1000.0, call_elapsed_seconds=40.0))
    scheduled = await harness.schedule()
    admitted = await harness.admit(scheduled.episode_id, scheduled.idempotency_key)
    assert admitted.decision == "admitted"
    harness.gate.observe_turn_started("turn-2", scheduled.idempotency_key)

    harness.clock.advance(3600.0)  # an overnight interval, far past 1060
    await harness.gate.observe_turn_finished("turn-2", PublicTurnStatus.COMPLETED)
    ((recovered, _message),) = harness.notices("recovered")
    assert recovered.successor_turn_id == "turn-2"


@pytest.mark.asyncio
async def test_gate_cancel_and_invalidate_drop_references() -> None:
    """Item 15 (gate half): cancel drops the pending schedule with a shown
    reason; invalidation (session change/shutdown) drops every reference.
    """
    harness = _GateHarness()
    harness.capture(_failure())
    scheduled = await harness.schedule()

    cancelled = await harness.cancel(scheduled.episode_id, reason="user_input")
    assert cancelled.cancelled is True
    ((notice, _message),) = harness.notices("cancelled")
    assert notice.reason == "user_input"

    declined = await harness.admit(scheduled.episode_id, scheduled.idempotency_key)
    assert declined.decision == "declined"
    assert declined.reason == "no_episode"

    # Invalidation drops every episode and projection.
    harness = _GateHarness()
    harness.capture(_failure())
    scheduled = await harness.schedule()
    harness.gate.invalidate_all()
    declined = await harness.admit(scheduled.episode_id, scheduled.idempotency_key)
    assert declined.decision == "declined"
    assert declined.reason == "no_episode"
    # The projections are gone too: no typed projection, no schedule.
    declined = await harness.schedule()
    assert declined.decision == "declined"
    assert declined.reason == "no_typed_projection"


@pytest.mark.asyncio
async def test_gate_config_disable_cancels_pending_but_not_in_flight() -> None:
    """Item 15: a config disable cancels a pending schedule; an admitted
    successor in flight settles via normal interruption rules (never
    cancelled as rollback).
    """
    harness = _GateHarness()
    harness.capture(_failure())
    scheduled = await harness.schedule()
    harness.config = harness.config.model_copy(update={"enabled": False})

    declined = await harness.admit(scheduled.episode_id, scheduled.idempotency_key)
    assert declined.decision == "declined"
    assert declined.reason == "disabled"
    ((cancelled, _message),) = harness.notices("cancelled")
    assert cancelled.reason == "disabled"

    # An admitted successor in flight: the disable refuses further admission
    # but does NOT cancel the episode — the successor still recovers.
    harness = _GateHarness()
    harness.capture(_failure())
    scheduled = await harness.schedule()
    admitted = await harness.admit(scheduled.episode_id, scheduled.idempotency_key)
    assert admitted.decision == "admitted"
    harness.gate.observe_turn_started("turn-2", scheduled.idempotency_key)

    harness.config = harness.config.model_copy(update={"enabled": False})
    declined = await harness.admit(scheduled.episode_id, scheduled.idempotency_key)
    assert declined.decision == "declined"
    assert declined.reason == "disabled"
    await harness.gate.observe_turn_finished("turn-2", PublicTurnStatus.COMPLETED)
    assert len(harness.notices("recovered")) == 1


@pytest.mark.asyncio
async def test_gate_disabled_mode_runs_no_automatic_continuation() -> None:
    """Item 20 (gate half) + OD4a: with ``enabled`` false NO root automatic
    continuation runs at all — including the migrated incomplete-stream
    path. The test sets enabled EXPLICITLY (round-2 amendment).
    """
    harness = _GateHarness(enabled=False)
    harness.capture(
        _failure(
            cause_category=FailureCategory.INCOMPLETE_STREAM,
            failure_class=FailureClass.INCOMPLETE_STREAM,
        )
    )
    declined = await harness.schedule()
    assert declined.decision == "declined"
    assert declined.reason == "disabled"
    assert harness.staged == []

    # No episode exists: no admission is possible either.
    declined = await harness.admit("root-auto-turn-1", "root-auto-recovery:x:1")
    assert declined.decision == "declined"
    assert declined.reason == "no_episode"


@pytest.mark.asyncio
async def test_gate_notices_are_deterministic_exactly_once_and_clock_stable() -> None:
    """Item 18 (gate half): milestone notices carry deterministic identities
    and fire exactly once — a re-read after an overnight fake-clock interval
    neither duplicates nor loses them.
    """
    # A day-long episode deadline keeps the overnight advance inside it.
    harness = _GateHarness(deadline_seconds=86400.0)
    harness.capture(_failure())
    scheduled = await harness.schedule()
    assert len(harness.staged) == 1
    detail, message = harness.staged[0]

    # An overnight interval passes; the entry is a durable history record.
    harness.clock.advance(8 * 3600.0)
    again = await harness.schedule()
    assert again.decision == "already_scheduled"
    assert len(harness.staged) == 1
    assert harness.staged[0] == (detail, message)

    # The full milestone chain stays exactly-once per (milestone, attempt).
    await harness.admit(scheduled.episode_id, scheduled.idempotency_key)
    harness.gate.observe_turn_started("turn-2", scheduled.idempotency_key)
    await harness.gate.observe_turn_finished("turn-2", PublicTurnStatus.COMPLETED)
    milestones = [entry[0].milestone for entry in harness.staged]
    assert milestones == ["scheduled", "started", "recovered"]


@pytest.mark.asyncio
async def test_gate_meta_recovery_a_notice_or_read_failure_never_blocks() -> None:
    """Item 17 (gate half): a controller/notice/admission failure is not
    another provider failure — the decision stands, nothing propagates.
    """
    harness = _GateHarness(stage_raises=True)
    harness.capture(_failure())
    scheduled = await harness.schedule()
    assert scheduled.decision == "scheduled"
    assert scheduled.episode_id is not None

    # A failing availability read degrades to available (missing evidence
    # never blocks on its own — the admission itself is the authority).
    harness = _GateHarness(availability_raises=True)
    harness.capture(_failure())
    scheduled = await harness.schedule()
    assert scheduled.decision == "scheduled"

    # A failing notice DELIVERY does not propagate out of the observation.
    harness = _GateHarness(emit_raises=True)
    harness.capture(_failure())
    scheduled = await harness.schedule()
    admitted = await harness.admit(scheduled.episode_id, scheduled.idempotency_key)
    assert admitted.decision == "admitted"
    harness.gate.observe_turn_started("turn-2", scheduled.idempotency_key)
    await harness.gate.observe_turn_finished("turn-2", PublicTurnStatus.COMPLETED)
    assert len(harness.notices("recovered")) == 1
