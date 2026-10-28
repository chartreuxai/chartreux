"""C1 — root-scoped forced-attempt override ledger.

The un-stick override mints typed, endpoint-bound permits for forced starts
past a cooled-down endpoint. The ledger is root-owned and carried unchanged
through every ``ConfigOrchestrator.copy()`` (the same discipline as the
availability ),

registry), so every agent's forced attempts draw on the SAME
per-endpoint episode accounting: 2 starts per episode, starts at least 30
seconds apart, one attempt in flight, and the settled pressure bound
``min(2, 1 + floor(T/30s))``. The episode re-opens ONLY after a measured
recovery followed by a genuinely new measured failure; agent lifecycle,
replacement, release, timer expiry, and elapsed time NEVER replenish the
counters.

All clocks are the opt-in ``fake_clock`` fixture; no real sleeps.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
from chartreux.core.config.layers.default import DefaultConfigLayer
from chartreux.core.config.layers.overrides import OverridesLayer
from chartreux.core.config.orchestrator import ConfigOrchestrator
from chartreux.core.model_catalog.loader import CatalogSnapshot
from chartreux.core.model_catalog.schema import ModelCatalog
from chartreux.core.recovery_override import (
    CooldownOverrideLedger,
    OverrideDenial,
    OverridePhase,
    OverrideTransition,
    episode_pressure_bound,
)
from tests.recovery.conftest import FakeClock

ENDPOINT = "https://api.shared.test/v1"
OTHER_ENDPOINT = "https://api.other.test/v1"


def _opened(fake_clock: FakeClock, endpoint: str = ENDPOINT) -> CooldownOverrideLedger:
    ledger = CooldownOverrideLedger(clock=fake_clock)
    assert ledger.note_endpoint_failure(endpoint) is True
    return ledger


@pytest.mark.parametrize(
    ("elapsed", "expected"), [(0.0, 1), (29.9, 1), (30.0, 2), (59.9, 2), (300.0, 2)]
)
def test_pressure_bound_matches_the_settled_curve(
    elapsed: float, expected: int
) -> None:
    assert episode_pressure_bound(elapsed) == expected


def test_reserve_requires_an_opened_episode(fake_clock: FakeClock) -> None:
    ledger = CooldownOverrideLedger(clock=fake_clock)
    outcome = ledger.reserve(ENDPOINT)
    assert outcome.transition is OverrideTransition.RESERVE
    assert outcome.phase is None
    assert outcome.denial is OverrideDenial.NO_ACTIVE_EPISODE
    assert outcome.permit is None
    # Only a measured failure opens an episode.
    assert ledger.note_endpoint_failure(ENDPOINT) is True
    granted = ledger.reserve(ENDPOINT)
    assert granted.phase is OverridePhase.RESERVED
    assert granted.denial is None
    assert granted.permit is not None
    assert granted.permit.endpoint == "api.shared.test"


def test_two_starts_max_per_episode(fake_clock: FakeClock) -> None:
    ledger = _opened(fake_clock)
    first = ledger.reserve(ENDPOINT)
    assert first.permit is not None
    assert ledger.start(first.permit).phase is OverridePhase.STARTED
    fake_clock.advance(30.0)
    second = ledger.reserve(ENDPOINT)
    assert second.permit is not None
    assert ledger.start(second.permit).phase is OverridePhase.STARTED
    fake_clock.advance(30.0)
    third = ledger.reserve(ENDPOINT)
    assert third.transition is OverrideTransition.RESERVE
    assert third.denial is OverrideDenial.EPISODE_EXHAUSTED
    assert third.permit is None


def test_second_start_waits_for_the_thirty_second_spacing(
    fake_clock: FakeClock,
) -> None:
    ledger = _opened(fake_clock)
    first = ledger.reserve(ENDPOINT)
    assert first.permit is not None
    assert ledger.start(first.permit).phase is OverridePhase.STARTED
    fake_clock.advance(10.0)
    second = ledger.reserve(ENDPOINT)
    assert second.denial is OverrideDenial.SPACING_EXCEEDED
    fake_clock.advance(20.0)
    second = ledger.reserve(ENDPOINT)
    assert second.permit is not None
    assert ledger.start(second.permit).phase is OverridePhase.STARTED


def test_one_attempt_in_flight_per_endpoint(fake_clock: FakeClock) -> None:
    ledger = _opened(fake_clock)
    first = ledger.reserve(ENDPOINT)
    assert first.permit is not None
    # Another agent's reserve hits the same shared in-flight slot.
    assert ledger.reserve(ENDPOINT).denial is OverrideDenial.ATTEMPT_IN_FLIGHT
    assert ledger.release(first.permit).phase is OverridePhase.RELEASED_PRETRANSPORT
    assert ledger.reserve(ENDPOINT).phase is OverridePhase.RESERVED


def test_pretransport_release_consumes_no_start(fake_clock: FakeClock) -> None:
    ledger = _opened(fake_clock)
    first = ledger.reserve(ENDPOINT)
    assert first.permit is not None
    assert ledger.release(first.permit).phase is OverridePhase.RELEASED_PRETRANSPORT
    second = ledger.reserve(ENDPOINT)
    assert second.permit is not None
    assert ledger.start(second.permit).phase is OverridePhase.STARTED
    fake_clock.advance(30.0)
    third = ledger.reserve(ENDPOINT)
    assert third.permit is not None
    assert ledger.start(third.permit).phase is OverridePhase.STARTED
    # Only the two real starts were consumed.
    fake_clock.advance(30.0)
    assert ledger.reserve(ENDPOINT).denial is OverrideDenial.EPISODE_EXHAUSTED


def test_release_is_pretransport_only(fake_clock: FakeClock) -> None:
    ledger = _opened(fake_clock)
    first = ledger.reserve(ENDPOINT)
    assert first.permit is not None
    assert ledger.start(first.permit).phase is OverridePhase.STARTED
    # The permit was CONSUMED at its transport start (C3): a late release and
    # a replayed start are both unknown-permit, and the slot is already free.
    assert ledger.release(first.permit).denial is OverrideDenial.UNKNOWN_PERMIT
    fake_clock.advance(30.0)
    second = ledger.reserve(ENDPOINT)
    assert second.permit is not None
    assert ledger.start(second.permit).phase is OverridePhase.STARTED
    fake_clock.advance(30.0)
    assert ledger.reserve(ENDPOINT).denial is OverrideDenial.EPISODE_EXHAUSTED


def test_double_start_and_replayed_permits_are_rejected(fake_clock: FakeClock) -> None:
    ledger = _opened(fake_clock)
    permit = ledger.reserve(ENDPOINT).permit
    assert permit is not None
    assert ledger.start(permit).phase is OverridePhase.STARTED
    # The permit was CONSUMED at its transport start (C3): a replayed start, a
    # late release, and a second release are all unknown-permit.
    assert ledger.start(permit).denial is OverrideDenial.UNKNOWN_PERMIT
    assert ledger.release(permit).denial is OverrideDenial.UNKNOWN_PERMIT
    assert ledger.release(permit).denial is OverrideDenial.UNKNOWN_PERMIT


def test_reservation_token_check_ignores_the_episode(fake_clock: FakeClock) -> None:
    """The token check alone (the coordinator safety net's gate): a permit
    stranded by an episode close is no longer outstanding by the episode
    check, but it still holds the endpoint's single in-flight slot and
    stays releasable — so the slot can never strand.
    """
    ledger = _opened(fake_clock)
    permit = ledger.reserve(ENDPOINT).permit
    assert permit is not None
    # A live reservation matches by token.
    assert ledger.reservation_token_matches(permit) is True
    assert ledger.permit_outstanding(permit) is True
    # A measured recovery retires the episode: the permit is stranded —
    # outstanding by the episode check is False, the token check is still
    # True, and the release frees the slot.
    assert ledger.note_endpoint_recovery(ENDPOINT) is True
    assert ledger.permit_outstanding(permit) is False
    assert ledger.reservation_token_matches(permit) is True
    assert ledger.release(permit).phase is OverridePhase.RELEASED_PRETRANSPORT
    assert ledger.reservation_token_matches(permit) is False
    # A started (consumed) permit matches neither check.
    assert ledger.note_endpoint_failure(ENDPOINT) is True
    started = ledger.reserve(ENDPOINT).permit
    assert started is not None
    assert ledger.start(started).phase is OverridePhase.STARTED
    assert ledger.reservation_token_matches(started) is False
    assert ledger.permit_outstanding(started) is False
    # A foreign permit (another endpoint, or a forged token) never matches.
    assert ledger.reservation_token_matches(replace(started, token=999)) is False
    other = replace(started, endpoint="api.other.test")
    assert ledger.reservation_token_matches(other) is False


def test_recovery_then_new_failure_opens_a_fresh_episode(fake_clock: FakeClock) -> None:
    ledger = _opened(fake_clock)
    first = ledger.reserve(ENDPOINT)
    assert first.permit is not None
    assert ledger.start(first.permit).phase is OverridePhase.STARTED
    fake_clock.advance(30.0)
    second = ledger.reserve(ENDPOINT)
    assert second.permit is not None
    assert ledger.start(second.permit).phase is OverridePhase.STARTED
    fake_clock.advance(30.0)
    assert ledger.reserve(ENDPOINT).denial is OverrideDenial.EPISODE_EXHAUSTED
    # Only a measured recovery closes the episode.
    assert ledger.note_endpoint_recovery(ENDPOINT) is True
    assert ledger.note_endpoint_recovery(ENDPOINT) is False
    assert ledger.reserve(ENDPOINT).denial is OverrideDenial.NO_ACTIVE_EPISODE
    # A new measured failure opens a fresh episode with fresh counters.
    assert ledger.note_endpoint_failure(ENDPOINT) is True
    fresh = ledger.reserve(ENDPOINT)
    assert fresh.permit is not None
    assert fresh.permit.episode_id != second.permit.episode_id
    assert ledger.start(fresh.permit).phase is OverridePhase.STARTED


def test_nothing_replenishes_the_counters(fake_clock: FakeClock) -> None:
    ledger = _opened(fake_clock)
    first = ledger.reserve(ENDPOINT)
    assert first.permit is not None
    assert ledger.start(first.permit).phase is OverridePhase.STARTED
    fake_clock.advance(30.0)
    second = ledger.reserve(ENDPOINT)
    assert second.permit is not None
    assert ledger.start(second.permit).phase is OverridePhase.STARTED
    # The permit was consumed at its transport start: a late release is
    # unknown-permit and consumes nothing further.
    assert ledger.release(second.permit).denial is OverrideDenial.UNKNOWN_PERMIT
    # A repeated failure inside the active episode changes nothing.
    assert ledger.note_endpoint_failure(ENDPOINT) is False
    fake_clock.advance(10_000.0)
    assert ledger.reserve(ENDPOINT).denial is OverrideDenial.EPISODE_EXHAUSTED
    # Even after recovery, elapsed time alone never re-opens: only a new
    # measured failure does.
    assert ledger.note_endpoint_recovery(ENDPOINT) is True
    fake_clock.advance(100_000.0)
    assert ledger.reserve(ENDPOINT).denial is OverrideDenial.NO_ACTIVE_EPISODE


def test_agents_share_one_episode_budget(fake_clock: FakeClock) -> None:
    ledger = _opened(fake_clock)
    first = ledger.reserve(ENDPOINT)
    assert first.permit is not None
    assert ledger.start(first.permit).phase is OverridePhase.STARTED
    fake_clock.advance(30.0)
    # A second agent draws on the SAME episode budget.
    second = ledger.reserve(ENDPOINT)
    assert second.permit is not None
    assert second.permit.episode_id == first.permit.episode_id
    assert ledger.start(second.permit).phase is OverridePhase.STARTED
    fake_clock.advance(30.0)
    # A third agent's reserve hits the SAME exhausted episode.
    assert ledger.reserve(ENDPOINT).denial is OverrideDenial.EPISODE_EXHAUSTED


def test_endpoints_isolate_accounting(fake_clock: FakeClock) -> None:
    ledger = _opened(fake_clock)
    first = ledger.reserve(ENDPOINT)
    assert first.permit is not None
    assert ledger.start(first.permit).phase is OverridePhase.STARTED
    # No permit bypasses another endpoint's episode accounting: the other
    # endpoint has no episode at all until its own measured failure.
    assert ledger.reserve(OTHER_ENDPOINT).denial is OverrideDenial.NO_ACTIVE_EPISODE
    assert ledger.note_endpoint_failure(OTHER_ENDPOINT) is True
    other = ledger.reserve(OTHER_ENDPOINT)
    assert other.permit is not None
    assert other.permit.endpoint == "api.other.test"
    assert other.permit.episode_id != first.permit.episode_id
    assert ledger.start(other.permit).phase is OverridePhase.STARTED
    # Exhausting ENDPOINT leaves the other endpoint's budget untouched.
    fake_clock.advance(30.0)
    second_other = ledger.reserve(OTHER_ENDPOINT)
    assert second_other.permit is not None
    assert ledger.start(second_other.permit).phase is OverridePhase.STARTED
    fake_clock.advance(30.0)
    assert ledger.reserve(OTHER_ENDPOINT).denial is OverrideDenial.EPISODE_EXHAUSTED
    # A permit bound to ENDPOINT can never act on the other endpoint's state.
    forged = replace(first.permit, endpoint="api.other.test")
    assert ledger.start(forged).denial is OverrideDenial.UNKNOWN_PERMIT


def _minimal_snapshot() -> CatalogSnapshot:
    return CatalogSnapshot(
        ModelCatalog.model_validate({
            "providers": {"test-one": {"api_base": "https://one.invalid"}},
            "models": {
                "first": {"deployments": [{"provider": "test-one", "name": "first"}]}
            },
            "roles": {
                "ordered": {
                    "description": "test preset",
                    "model": "first",
                    "thinking": "high",
                }
            },
        }),
        "revision",
    )


@pytest.mark.asyncio
async def test_copied_orchestrators_share_the_override_ledger() -> None:
    layer = OverridesLayer(data={"active_model": "first"})
    root = await ConfigOrchestrator.create(
        schema=ChartreuxConfigSchema,
        layers=[DefaultConfigLayer(schema=ChartreuxConfigSchema), layer],
        default_layer_resolver=lambda: layer,
        catalog_snapshot=_minimal_snapshot(),
    )
    clock = FakeClock()
    root._cooldown_override_ledger = CooldownOverrideLedger(clock=clock)
    child = root.copy()
    assert child.cooldown_override_ledger is root.cooldown_override_ledger
    ledger = root.cooldown_override_ledger
    assert ledger.note_endpoint_failure(ENDPOINT) is True
    first = ledger.reserve(ENDPOINT)
    assert first.permit is not None
    assert ledger.start(first.permit).phase is OverridePhase.STARTED
    clock.advance(30.0)
    # The copy's handle draws on the SAME episode budget.
    second = child.cooldown_override_ledger.reserve(ENDPOINT)
    assert second.permit is not None
    assert second.permit.episode_id == first.permit.episode_id
    assert (
        child.cooldown_override_ledger.start(second.permit).phase
        is OverridePhase.STARTED
    )
    clock.advance(30.0)
    assert (
        root.cooldown_override_ledger.reserve(ENDPOINT).denial
        is OverrideDenial.EPISODE_EXHAUSTED
    )
