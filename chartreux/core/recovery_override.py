"""Root-scoped forced-attempt override ledger for cooldown episodes.

The orchestrator un-stick override lets an explicit request force a
completion attempt past an endpoint's cooldown. Every forced attempt is
metered by a typed ledger owned by the ROOT orchestrator and carried
unchanged through every ``ConfigOrchestrator.copy()`` — the same sharing
discipline as the availability registry — so every agent's forced attempts
draw on the SAME per-endpoint episode accounting no matter which agent asks.

Settled un-stick parameters: 2 forced starts per endpoint episode, starts at
least 30 seconds apart, at most one attempt in flight per endpoint, and the
pressure bound ``min(2, 1 + floor(T/30s))``. The endpoint-episode boundary is
PINNED: an episode opens only on a measured endpoint failure, closes only on
a measured endpoint recovery, and re-opens only after a measured recovery
followed by a genuinely new measured failure — agent lifecycle, replacement,
release, timer expiry, and elapsed time NEVER replenish the counters.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
import time

from chartreux.core.recovery_models import canonical_endpoint_identity

#: Settled un-stick parameters: 2 forced starts per endpoint episode, starts
#: at least 30 seconds apart.
MAX_FORCED_STARTS = 2
START_SPACING_SECONDS = 30.0


def episode_pressure_bound(
    elapsed_seconds: float,
    *,
    max_starts: int = MAX_FORCED_STARTS,
    start_spacing_seconds: float = START_SPACING_SECONDS,
) -> int:
    """The settled pressure bound ``min(max_starts, 1 + floor(T/spacing))``.

    ``T`` is the seconds elapsed since the endpoint episode opened. One
    forced start is available immediately; each further full spacing window
    admits one more, capped at the settled 2-start maximum.
    """
    if max_starts < 1:
        raise ValueError("max_starts must be >= 1")
    if start_spacing_seconds <= 0:
        raise ValueError("start_spacing_seconds must be positive")
    if elapsed_seconds <= 0:
        return 1
    return min(max_starts, 1 + int(elapsed_seconds // start_spacing_seconds))


class OverrideTransition(StrEnum):
    """The three typed ledger transitions."""

    RESERVE = "reserve"
    START = "start"
    RELEASE = "release"


class OverridePhase(StrEnum):
    """The settled phase a granted transition leaves the permit in."""

    RESERVED = "reserved"
    STARTED = "started"
    RELEASED_PRETRANSPORT = "released-pretransport"


class OverrideDenial(StrEnum):
    """Why the ledger refused a transition."""

    UNKNOWN_ENDPOINT = "unknown-endpoint"
    NO_ACTIVE_EPISODE = "no-active-episode"
    EPISODE_EXHAUSTED = "episode-exhausted"
    SPACING_EXCEEDED = "spacing-exceeded"
    ATTEMPT_IN_FLIGHT = "attempt-in-flight"
    EPISODE_CLOSED = "episode-closed"
    UNKNOWN_PERMIT = "unknown-permit"


@dataclass(frozen=True, slots=True)
class OverridePermit:
    """One endpoint- and episode-bound forced-attempt reservation."""

    token: int
    endpoint: str
    episode_id: int
    issued_at: float


@dataclass(frozen=True, slots=True)
class LedgerOutcome:
    """The typed result of one ledger transition."""

    transition: OverrideTransition
    phase: OverridePhase | None = None
    denial: OverrideDenial | None = None
    permit: OverridePermit | None = None


@dataclass(frozen=True, slots=True)
class OverrideEpisodeStatus:
    """Read-only snapshot of one endpoint's active override episode (C4).

    Bounded diagnostics source for the explicit task override: the caller
    derives the remaining start allowance and the next permitted start time
    from these fields. Carries no behavior over the ledger.
    """

    episode_id: int
    opened_at: float
    starts_used: int
    max_starts: int
    start_spacing_seconds: float
    last_started_at: float | None

    @property
    def starts_remaining(self) -> int:
        """Starts still admitted by this episode, INCLUDING one in flight."""
        return max(0, self.max_starts - self.starts_used)

    def next_start_delay_seconds(self, now: float) -> float | None:
        """Seconds until the episode's next start is admitted; None = never.

        The next start (number ``starts_used + 1``) is admitted by the
        pressure bound at ``opened_at + starts_used * spacing`` and by the
        spacing floor at ``last_started_at + spacing``; the delay is the
        larger of the two, clamped at zero.
        """
        if self.starts_used >= self.max_starts:
            return None
        next_time = self.opened_at + self.starts_used * self.start_spacing_seconds
        if self.last_started_at is not None:
            next_time = max(
                next_time, self.last_started_at + self.start_spacing_seconds
            )
        return max(0.0, next_time - now)


@dataclass(slots=True)
class _EndpointEpisode:
    episode_id: int
    opened_at: float
    starts_used: int = 0
    last_started_at: float | None = None


@dataclass(slots=True)
class _EndpointOverrideState:
    episode: _EndpointEpisode | None = None
    outstanding: OverridePermit | None = None


class CooldownOverrideLedger:
    """Root-owned forced-override accounting shared by every runtime copy.

    One instance is owned by the root orchestrator and carried unchanged
    through every ``ConfigOrchestrator.copy()`` (the same discipline as the
    availability registry), so every agent's forced attempts draw on the
    same per-endpoint episode accounting no matter which agent asks.

    Episodes are opened only by a measured endpoint failure and closed only
    by a measured endpoint recovery; a closed episode re-opens only through a
    NEW measured failure. Release, agent lifecycle, replacement, timer
    expiry, and elapsed time never open, re-open, or replenish anything.
    """

    def __init__(
        self,
        *,
        max_starts: int = MAX_FORCED_STARTS,
        start_spacing_seconds: float = START_SPACING_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_starts < 1:
            raise ValueError("max_starts must be >= 1")
        if start_spacing_seconds <= 0:
            raise ValueError("start_spacing_seconds must be positive")
        self.max_starts = max_starts
        self.start_spacing_seconds = start_spacing_seconds
        self._clock = clock
        self._states: dict[str, _EndpointOverrideState] = {}
        self._next_token = 0
        self._next_episode_id = 0

    def note_endpoint_failure(self, endpoint: str | None) -> bool:
        """Record one measured endpoint failure.

        Opens a NEW episode — fresh counters — only when no active episode
        exists: the first failure ever, or the first failure after a measured
        recovery. A failure inside an active episode changes nothing. Returns
        whether a new episode opened. Elapsed time, agent lifecycle,
        replacement, and release NEVER open or re-open an episode.
        """
        canonical = canonical_endpoint_identity(endpoint)
        if canonical is None:
            return False
        state = self._states.get(canonical)
        if state is not None and state.episode is not None:
            return False
        self._next_episode_id += 1
        state = self._states.setdefault(canonical, _EndpointOverrideState())
        state.episode = _EndpointEpisode(
            episode_id=self._next_episode_id, opened_at=self._clock()
        )
        return True

    def note_endpoint_recovery(self, endpoint: str | None) -> bool:
        """Record one measured endpoint recovery; closes the active episode.

        Counters are never reset by recovery: the whole episode object is
        retired, and only a NEW measured failure can open a fresh episode
        with fresh counters. An outstanding permit from the closed episode
        becomes unusable (its start is denied) but stays releasable so the
        in-flight slot can never strand. Returns whether an active episode
        was closed.
        """
        canonical = canonical_endpoint_identity(endpoint)
        if canonical is None:
            return False
        state = self._states.get(canonical)
        if state is None or state.episode is None:
            return False
        state.episode = None
        return True

    def reserve(self, endpoint: str | None) -> LedgerOutcome:
        """Mint one endpoint- and episode-bound forced-attempt reservation.

        Requires an active episode, no outstanding permit (ONE attempt in
        flight per endpoint), and the pressure bound: the episode admits
        ``min(2, 1 + floor(T/30s))`` starts by elapsed episode time. The
        minted permit is bound to BOTH the canonical endpoint and the episode
        id, so a permit can never act on another endpoint's accounting.
        """
        canonical = canonical_endpoint_identity(endpoint)
        if canonical is None:
            return LedgerOutcome(
                OverrideTransition.RESERVE, denial=OverrideDenial.UNKNOWN_ENDPOINT
            )
        state = self._states.get(canonical)
        if state is None or state.episode is None:
            return LedgerOutcome(
                OverrideTransition.RESERVE, denial=OverrideDenial.NO_ACTIVE_EPISODE
            )
        if state.outstanding is not None:
            return LedgerOutcome(
                OverrideTransition.RESERVE, denial=OverrideDenial.ATTEMPT_IN_FLIGHT
            )
        episode = state.episode
        if episode.starts_used >= self.max_starts:
            return LedgerOutcome(
                OverrideTransition.RESERVE, denial=OverrideDenial.EPISODE_EXHAUSTED
            )
        bound = episode_pressure_bound(
            self._clock() - episode.opened_at,
            max_starts=self.max_starts,
            start_spacing_seconds=self.start_spacing_seconds,
        )
        if episode.starts_used + 1 > bound:
            return LedgerOutcome(
                OverrideTransition.RESERVE, denial=OverrideDenial.SPACING_EXCEEDED
            )
        self._next_token += 1
        permit = OverridePermit(
            token=self._next_token,
            endpoint=canonical,
            episode_id=episode.episode_id,
            issued_at=self._clock(),
        )
        state.outstanding = permit
        return LedgerOutcome(
            OverrideTransition.RESERVE, phase=OverridePhase.RESERVED, permit=permit
        )

    def start(self, permit: OverridePermit) -> LedgerOutcome:
        """Consume the permit's start: one of at most 2 per episode.

        Requires the spacing floor since the episode's previous start and
        re-checks the pressure bound (fail-closed). A consumed start stays
        consumed — a later release never refunds it.
        """
        state = self._states.get(permit.endpoint)
        if (
            state is None
            or state.outstanding is None
            or state.outstanding.token != permit.token
        ):
            return LedgerOutcome(
                OverrideTransition.START, denial=OverrideDenial.UNKNOWN_PERMIT
            )
        episode = state.episode
        if episode is None or episode.episode_id != permit.episode_id:
            return LedgerOutcome(
                OverrideTransition.START, denial=OverrideDenial.EPISODE_CLOSED
            )
        now = self._clock()
        if (
            episode.last_started_at is not None
            and now - episode.last_started_at < self.start_spacing_seconds
        ):
            return LedgerOutcome(
                OverrideTransition.START, denial=OverrideDenial.SPACING_EXCEEDED
            )
        bound = episode_pressure_bound(
            now - episode.opened_at,
            max_starts=self.max_starts,
            start_spacing_seconds=self.start_spacing_seconds,
        )
        if episode.starts_used + 1 > bound:
            return LedgerOutcome(
                OverrideTransition.START,
                denial=(
                    OverrideDenial.EPISODE_EXHAUSTED
                    if episode.starts_used >= self.max_starts
                    else OverrideDenial.SPACING_EXCEEDED
                ),
            )
        episode.starts_used += 1
        episode.last_started_at = now
        # The reservation is CONSUMED at the actual transport start (C3): the
        # in-flight slot frees here, and the only remaining release path is
        # pretransport.
        state.outstanding = None
        return LedgerOutcome(
            OverrideTransition.START, phase=OverridePhase.STARTED, permit=permit
        )

    def release(self, permit: OverridePermit) -> LedgerOutcome:
        """Retire an unstarted permit and free the single in-flight slot.

        Release is the PRETRANSPORT path only: a started permit was already
        consumed at its transport start and is no longer outstanding, so a
        late release is unknown-permit. A pretransport release consumes
        nothing — the start budget is untouched; this frees the slot, it does
        not replenish a counter. A permit stranded by an episode close stays
        releasable so the in-flight slot can never strand.
        """
        state = self._states.get(permit.endpoint)
        if (
            state is None
            or state.outstanding is None
            or state.outstanding.token != permit.token
        ):
            return LedgerOutcome(
                OverrideTransition.RELEASE, denial=OverrideDenial.UNKNOWN_PERMIT
            )
        state.outstanding = None
        return LedgerOutcome(
            OverrideTransition.RELEASE, phase=OverridePhase.RELEASED_PRETRANSPORT
        )

    def permit_outstanding(self, permit: OverridePermit) -> bool:
        """Read-only: whether the permit is the endpoint's live reservation.

        True only when the permit is the endpoint's outstanding reservation
        AND its episode is still the active one. A used (started), released,
        foreign, or episode-closed permit reports False, so a stale permit
        never authorizes a cooldown bypass at candidate selection (C2).
        Never mutates: the counters, the in-flight slot, and the episode are
        all untouched.
        """
        state = self._states.get(permit.endpoint)
        if (
            state is None
            or state.outstanding is None
            or state.outstanding.token != permit.token
        ):
            return False
        episode = state.episode
        return episode is not None and episode.episode_id == permit.episode_id

    def reservation_token_matches(self, permit: OverridePermit) -> bool:
        """Read-only: whether the permit is the endpoint's live reservation.

        The TOKEN check alone — unlike :meth:`permit_outstanding`, the
        episode is not consulted. A permit stranded by an episode close is
        no longer outstanding by the episode check, but it still holds the
        endpoint's single in-flight slot and stays releasable, so the slot
        can never strand. A started (consumed), released, or foreign permit
        reports False. Never mutates: the counters, the in-flight slot, and
        the episode are all untouched.
        """
        state = self._states.get(permit.endpoint)
        return (
            state is not None
            and state.outstanding is not None
            and state.outstanding.token == permit.token
        )

    def episode_status(self, endpoint: str | None) -> OverrideEpisodeStatus | None:
        """Read-only snapshot of the endpoint's ACTIVE episode (C4).

        None when the endpoint is unknown or has no active episode. Never
        mutates: the counters, the in-flight slot, and the episode are all
        untouched — this feeds only the bounded un-stick diagnostics.
        """
        canonical = canonical_endpoint_identity(endpoint)
        if canonical is None:
            return None
        state = self._states.get(canonical)
        if state is None or state.episode is None:
            return None
        episode = state.episode
        return OverrideEpisodeStatus(
            episode_id=episode.episode_id,
            opened_at=episode.opened_at,
            starts_used=episode.starts_used,
            max_starts=self.max_starts,
            start_spacing_seconds=self.start_spacing_seconds,
            last_started_at=episode.last_started_at,
        )
