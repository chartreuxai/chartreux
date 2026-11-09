"""P5 cache keep-alive pilot — the ACCOUNTANT (5a), SNAPSHOT (5b), and
SCHEDULER + OBSERVE (5c) sections.

MODULE PIN: this is the ONE keep-alive module (``core/agent_loop/_keepalive.py``);
later chunks extend it in place under named sections (SNAPSHOT in 5b, SCHEDULER
+ OBSERVE in 5c). This file carries the ACCOUNTANT (5a), SNAPSHOT (5b), and
SCHEDULER + OBSERVE (5c) sections.

The accountant is the synchronous, conservative, PRE-SEND RESERVATION enforcer
for the absolute per-session count and spend caps of the keep-alive pilot
(design P5, "SPEND CAP"). Before each ping is sent, the prospective MAXIMUM cost
is reserved against the caps, priced in BILLABLE TOKENS, not bytes:

- input cost: a proven-conservative billable-input-token bound for the COMPLETE
  ping request (the snapshot prefix plus any request envelope) via a
  provider-specific conversion that provably never undercounts
  provider-billed input tokens, including framing additions, priced at the
  UNCACHED input rate;
- output cost: the ping's declared max completion tokens x the completion price.

FAIL-CLOSED: when the provider-specific conversion, the applicable prices, the
declared output limit, the attribution, or the ledger sink are unavailable, the
accountant halts for that deployment and no ping is sent. A ping is sent only
when the reservation fits the remaining caps. Reconciliation on the ping's
recorded usage releases reservation-minus-actual; pre-send supersession (a real
call starting mid-reserve) releases the reservation unspent. The count cap
counts ATTEMPTS (reserved pings, superseded ones included), not completions.
EVERY attempt writes a ledger row (purpose=keepalive, with a typed keepalive
entry carrying outcome=completed | unknown-usage | cancelled | superseded) so
the ledger reconstructs cap burn; unknown-usage and cancelled attempts RETAIN
their full reservation in the accumulator.

Pricing source: the ledger-recorded prices, frozen at accountant start with the
same ``capture_prices`` the gateway records on every row — the async ledger
write is the record, not the counter.

The SNAPSHOT section (chunk 5b) holds the IMMUTABLE serialized cache-relevant
prefix: ``KeepaliveSnapshotStore`` freezes the prefix together with the
deployment/provider it was taken on and the ping-only generation settings, and
INVALIDATES (drops, never mutates) the live snapshot on any of the six design
triggers — prompt swap, config swap, tool change, compaction, ANY new
conversation activity (partial wakes included), deployment failover. A ping
may only re-send a prefix that is still live for the SAME deployment/provider.

The SCHEDULER + OBSERVE sections (chunk 5c) drive the pings: ROOT-ONLY (never
children), only while owned dependency runs are active AND the idle gap since
the last conversation activity is >= the configured floor; at most one ping in
flight; an in-flight ping is superseded when a real call starts (in both the
reserved and the in-flight phase); returned tool calls are NEVER executed and
ping responses are NEVER appended; the ping's OWN reported usage is observed
and the idle cycle STOPs on a cache miss, missing/partial usage, error, or
insufficient cached-prefix coverage. The owned-dependency gate reads P2's
continuation machinery (the quiescence snapshot's active dependency runs) as
a READ-ONLY probe.

The pilot flag ships OFF by default (the ``[keepalive]`` config section landed
in chunk 5b defaults ``enabled`` to FALSE); nothing here changes any default
behavior of ordinary calls.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
import hashlib
from math import ceil
from threading import Lock
from typing import Any, Protocol
from uuid import uuid4

from chartreux.core.config import ModelConfig
from chartreux.core.config.models import KeepaliveSectionConfig
from chartreux.core.llm_models import LLMChunk, LLMUsage
from chartreux.core.usage import (
    KeepaliveLedgerEntry,
    KeepaliveOutcome,
    UsageAttribution,
    UsageOutcome,
    UsagePrices,
    UsagePurpose,
    UsageRecord,
    UsageState,
    UsageTokens,
    capture_prices,
    price_usage,
)


class KeepaliveAccountantError(Exception):
    """Base for the keep-alive accountant's local denials."""


class KeepaliveFailClosedError(KeepaliveAccountantError):
    """A required input is unavailable: the deployment halted, no ping sent."""


class KeepaliveCapExceededError(KeepaliveAccountantError):
    """The reservation does not fit the remaining absolute cap."""


class KeepaliveReservationStateError(KeepaliveAccountantError):
    """A reservation was settled twice or was never reserved."""


type BillableInputBound = Callable[[int], int]
"""Complete serialized request bytes -> billable input tokens, never undercounting."""


@dataclass(frozen=True)
class BillableInputRule:
    """A provider's conservative conversion constants.

    ``bound(bytes) = ceil(bytes / chars_per_token) + framing_tokens`` provably
    dominates the provider's actual billing
    ``ceil(bytes / actual_chars_per_token) + actual_framing_tokens`` whenever
    ``chars_per_token <= actual_chars_per_token`` and
    ``framing_tokens >= actual_framing_tokens``. The registered constants must
    satisfy that domination — that is what makes the bound proven-conservative
    rather than a heuristic estimate.
    """

    chars_per_token: float
    framing_tokens: int

    def bound(self, request_bytes: int) -> int:
        """The conservative billable-input-token bound for one request."""
        if request_bytes < 0:
            raise ValueError("request_bytes must be non-negative")
        return ceil(request_bytes / self.chars_per_token) + self.framing_tokens


_BILLABLE_INPUT_RULES: dict[str, BillableInputRule] = {}


def register_billable_input_rule(provider: str, rule: BillableInputRule) -> None:
    """Register the proven-conservative conversion for one provider."""
    if rule.chars_per_token <= 0:
        raise ValueError("chars_per_token must be positive")
    if rule.framing_tokens < 0:
        raise ValueError("framing_tokens must be non-negative")
    _BILLABLE_INPUT_RULES[provider] = rule


def billable_input_bound(provider: str) -> BillableInputBound | None:
    """The registered conversion for ``provider``; None when unavailable."""
    rule = _BILLABLE_INPUT_RULES.get(provider)
    return None if rule is None else rule.bound


class KeepaliveLedgerSink(Protocol):
    """Synchronous ledger settlement for keepalive attempt rows.

    The accountant itself is fully synchronous, so the sink is a plain
    callable (e.g. ``UsageWriter.append``); the async writer facade adapts at
    the scheduling layer. A missing sink is fail-closed: the accountant
    requires one.
    """

    def __call__(self, record: UsageRecord, /) -> None: ...


@dataclass(frozen=True)
class KeepaliveCaps:
    """Absolute per-session caps; the count cap counts ATTEMPTS, not completions."""

    max_attempts: int
    max_spend_usd: float

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if self.max_spend_usd < 0:
            raise ValueError("max_spend_usd must be non-negative")


@dataclass(frozen=True)
class KeepaliveReservation:
    """One reserved ping attempt; settle it exactly once."""

    attempt_id: str
    attempt_ordinal: int
    reserved_input_tokens: int
    reserved_output_tokens: int
    reserved_cost_usd: float


@dataclass(frozen=True)
class KeepaliveBurn:
    """Cap burn: reserved attempts and retained spend."""

    attempts: int
    retained_spend_usd: float


def reconstruct_keepalive_burn(records: Iterable[UsageRecord]) -> KeepaliveBurn:
    """Rebuild the cap burn from keepalive-purpose ledger rows alone.

    Counting the rows reconstructs the attempts-counted count-cap burn; summing
    each entry's retained_cost_usd reconstructs the spend-cap burn.
    """
    attempts = 0
    spend = 0.0
    for record in records:
        entry = record.keepalive
        if record.purpose is UsagePurpose.KEEPALIVE and entry is not None:
            attempts += 1
            spend += entry.retained_cost_usd
    return KeepaliveBurn(attempts=attempts, retained_spend_usd=spend)


#: The ordinary outcome axis a keepalive attempt row maps onto; the precise
#: keepalive outcome rides on the row's keepalive entry.
_USAGE_OUTCOME: dict[KeepaliveOutcome, UsageOutcome] = {
    KeepaliveOutcome.COMPLETED: UsageOutcome.COMPLETED,
    KeepaliveOutcome.UNKNOWN_USAGE: UsageOutcome.FAILED,
    KeepaliveOutcome.CANCELLED: UsageOutcome.INTERRUPTED,
    KeepaliveOutcome.SUPERSEDED: UsageOutcome.INTERRUPTED,
}


class KeepaliveAccountant:
    """Synchronous pre-send reservation accountant for keepalive pings (P5 5a).

    One instance per deployment (session). ``reserve`` must be called before
    each ping is sent and returns the reservation the caps were charged; the
    reservation is then settled exactly once — ``reconcile`` on recorded usage
    (releases reservation-minus-actual), ``mark_unknown_usage`` or ``cancel``
    (both retain the full reservation), or ``supersede`` (pre-send supersession
    by a real call: released unspent, still counted as an attempt). Every
    settlement writes one ledger row through the sink.
    """

    def __init__(
        self,
        *,
        caps: KeepaliveCaps,
        model: ModelConfig,
        provider: str,
        attribution: UsageAttribution | None,
        ledger: KeepaliveLedgerSink | None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._caps = caps
        self._model = model
        self._provider = provider
        # Prices are frozen at accountant start — the same capture the gateway
        # records on each ledger row (the ledger is the record, not the counter).
        self._prices: UsagePrices = capture_prices(model)
        self._attribution = attribution
        self._ledger = ledger
        self._clock = clock if clock is not None else lambda: datetime.now(UTC)
        self._lock = Lock()
        self._attempts = 0
        self._charged_spend_usd = 0.0
        self._halted = False
        self._open: dict[str, KeepaliveReservation] = {}

    @property
    def halted(self) -> bool:
        """Whether the deployment has fail-closed halted."""
        with self._lock:
            return self._halted

    @property
    def attempts(self) -> int:
        """Attempts reserved so far — the count cap counts these, superseded included."""
        with self._lock:
            return self._attempts

    @property
    def charged_spend_usd(self) -> float:
        """Spend charged against the spend cap: open reservations plus retained."""
        with self._lock:
            return self._charged_spend_usd

    @property
    def open_attempts(self) -> int:
        """Reservations not yet settled."""
        with self._lock:
            return len(self._open)

    def burn(self) -> KeepaliveBurn:
        """The accountant-side cap burn (attempts and charged spend)."""
        with self._lock:
            return KeepaliveBurn(
                attempts=self._attempts, retained_spend_usd=self._charged_spend_usd
            )

    def reserve(
        self, request_bytes: int, declared_max_tokens: int | None
    ) -> KeepaliveReservation:
        """Reserve the prospective maximum cost of one ping BEFORE it is sent.

        Raises ``KeepaliveFailClosedError`` (and halts the deployment) when the
        conversion, prices, declared output limit, attribution, or sink are
        unavailable, and ``KeepaliveCapExceededError`` when the reservation does
        not fit the remaining caps. Otherwise the attempt is counted and
        charged immediately.
        """
        with self._lock:
            reason = self._unavailable_reason(declared_max_tokens)
            if self._halted or reason is not None:
                self._halted = True
                raise KeepaliveFailClosedError(
                    reason or "the keepalive deployment is halted"
                )
            bound = billable_input_bound(self._provider)
            input_price = self._prices.input
            output_price = self._prices.output
            assert (
                bound is not None
                and input_price is not None
                and output_price is not None
                and declared_max_tokens is not None
            )
            reserved_input = bound(request_bytes)
            reserved_cost = (
                reserved_input * input_price + declared_max_tokens * output_price
            ) / 1_000_000
            if self._attempts + 1 > self._caps.max_attempts:
                raise KeepaliveCapExceededError(
                    f"count cap {self._caps.max_attempts} already reached by "
                    f"{self._attempts} reserved attempts"
                )
            if self._charged_spend_usd + reserved_cost > self._caps.max_spend_usd:
                raise KeepaliveCapExceededError(
                    f"spend cap {self._caps.max_spend_usd} USD would be exceeded: "
                    f"{self._charged_spend_usd} charged + {reserved_cost} reserved"
                )
            self._attempts += 1
            self._charged_spend_usd += reserved_cost
            reservation = KeepaliveReservation(
                attempt_id=str(uuid4()),
                attempt_ordinal=self._attempts,
                reserved_input_tokens=reserved_input,
                reserved_output_tokens=declared_max_tokens,
                reserved_cost_usd=reserved_cost,
            )
            self._open[reservation.attempt_id] = reservation
            return reservation

    def reconcile(
        self,
        reservation: KeepaliveReservation,
        *,
        input_tokens: int | None,
        output_tokens: int | None,
        cached_input_tokens: int | None,
    ) -> None:
        """Settle a sent ping on its recorded usage: outcome=completed.

        Releases reservation-minus-actual: the accumulator is charged only the
        recorded actual cost. Incomplete recorded usage cannot be reconciled
        and falls back to ``mark_unknown_usage`` (full reservation retained).
        """
        tokens = UsageTokens(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached_input_tokens,
        )
        if tokens.presence_state is not UsageState.COMPLETE:
            self.mark_unknown_usage(reservation)
            return
        pricing = price_usage(**tokens.model_dump(), prices=self._prices)
        retained = (
            pricing.known_cost_usd
            if not pricing.has_unknown_cost
            else reservation.reserved_cost_usd
        )
        self._settle(reservation, KeepaliveOutcome.COMPLETED, tokens, retained)

    def mark_unknown_usage(self, reservation: KeepaliveReservation) -> None:
        """Settle a ping whose usage was not recorded: outcome=unknown-usage.

        The full reservation is retained in the accumulator.
        """
        self._settle(
            reservation,
            KeepaliveOutcome.UNKNOWN_USAGE,
            None,
            reservation.reserved_cost_usd,
        )

    def cancel(self, reservation: KeepaliveReservation) -> None:
        """Settle a cancelled ping: outcome=cancelled, full reservation retained."""
        self._settle(
            reservation, KeepaliveOutcome.CANCELLED, None, reservation.reserved_cost_usd
        )

    def supersede(self, reservation: KeepaliveReservation) -> None:
        """Settle a pre-send supersession (a real call started mid-reserve).

        The reservation is released unspent; the attempt still counts against
        the count cap. Outcome=superseded.
        """
        self._settle(reservation, KeepaliveOutcome.SUPERSEDED, None, 0.0)

    def _unavailable_reason(self, declared_max_tokens: int | None) -> str | None:
        """Why this deployment cannot reserve right now, or None when it can."""
        checks: list[tuple[bool, str]] = [
            (self._ledger is None, "no ledger sink: the ping accountant requires one"),
            (
                self._attribution is None,
                "no usage attribution: the ping accountant requires one",
            ),
            (
                billable_input_bound(self._provider) is None,
                "no registered conservative billable-input conversion for "
                f"provider {self._provider!r}",
            ),
            (self._prices.input is None, "the uncached input price is unknown"),
            (self._prices.output is None, "the completion price is unknown"),
            (
                declared_max_tokens is None,
                "no declared max completion tokens: the output cost is unbounded",
            ),
        ]
        for failed, reason in checks:
            if failed:
                return reason
        return None

    def _settle(
        self,
        reservation: KeepaliveReservation,
        outcome: KeepaliveOutcome,
        tokens: UsageTokens | None,
        retained_cost_usd: float,
    ) -> None:
        with self._lock:
            if reservation.attempt_id not in self._open:
                raise KeepaliveReservationStateError(
                    f"attempt {reservation.attempt_ordinal} is not an open reservation"
                )
            del self._open[reservation.attempt_id]
            self._charged_spend_usd += retained_cost_usd - reservation.reserved_cost_usd
            record = self._attempt_record(
                reservation, outcome, tokens, retained_cost_usd
            )
        # The ledger write sits outside the accumulator lock: the ledger is the
        # record, the accumulator is the counter.
        assert self._ledger is not None
        self._ledger(record)

    def _attempt_record(
        self,
        reservation: KeepaliveReservation,
        outcome: KeepaliveOutcome,
        tokens: UsageTokens | None,
        retained_cost_usd: float,
    ) -> UsageRecord:
        assert self._attribution is not None
        attribution = self._attribution.model_copy(
            update={
                "model": self._model.alias,
                "provider": self._provider,
                "wire_name": self._model.name,
                "purpose": UsagePurpose.KEEPALIVE,
                "content_class": None,
                "run_id": None,
            }
        )
        usage_state = (
            tokens.presence_state if tokens is not None else UsageState.MISSING
        )
        return UsageRecord(
            **attribution.model_dump(),
            **(
                tokens.model_dump()
                if tokens is not None
                else UsageTokens().model_dump()
            ),
            outcome=_USAGE_OUTCOME[outcome],
            usage_state=usage_state,
            attempt_ordinal=reservation.attempt_ordinal,
            record_id=str(uuid4()),
            occurred_at=self._clock(),
            prices_usd_per_million=self._prices,
            known_cost_usd=retained_cost_usd,
            has_unknown_cost=(
                outcome is not KeepaliveOutcome.COMPLETED
                or usage_state is not UsageState.COMPLETE
            ),
            keepalive=KeepaliveLedgerEntry(
                outcome=outcome,
                reserved_input_tokens=reservation.reserved_input_tokens,
                reserved_output_tokens=reservation.reserved_output_tokens,
                reserved_cost_usd=reservation.reserved_cost_usd,
                retained_cost_usd=retained_cost_usd,
            ),
        )


# ---------------------------------------------------------------------------
# SNAPSHOT section (chunk 5b)
# ---------------------------------------------------------------------------

#: The pilot's declared ping output bound: the ping-only generation settings
#: default to this max completion tokens, which is what the accountant's
#: pre-send reservation prices as the ping's output cost.
DEFAULT_PING_MAX_TOKENS = 64

#: The ping-only temperature ceiling (the ping's own sampling bound, separate
#: from the ordinary call's model temperature).
PING_TEMPERATURE_MAX = 2.0


class KeepaliveInvalidation(StrEnum):
    """The six invalidation triggers of design P5's snapshot list.

    The design line "prompt/config swaps" counts as TWO triggers — a prompt
    swap and a config swap are distinct events — so together with tool
    changes, compaction, any new conversation activity (partial wakes
    included), and deployment failover the list is exactly six.
    """

    PROMPT_SWAP = "prompt_swap"
    CONFIG_SWAP = "config_swap"
    TOOL_CHANGE = "tool_change"
    COMPACTION = "compaction"
    CONVERSATION_ACTIVITY = "conversation_activity"
    DEPLOYMENT_FAILOVER = "deployment_failover"


@dataclass(frozen=True)
class PingGenerationSettings:
    """The ping-only generation settings, specified SEPARATELY from ordinary calls.

    A keepalive ping is built from THESE settings — never from the session's
    ordinary generation settings (the active model's temperature, the
    ordinary call's max tokens) — and the ordinary call path never reads
    them: the two surfaces are distinct frozen objects with no shared state,
    so a ping can neither inherit nor disturb an ordinary call's generation
    settings.
    """

    max_tokens: int | None = DEFAULT_PING_MAX_TOKENS
    temperature: float | None = None

    def __post_init__(self) -> None:
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValueError("the ping's declared max_tokens must be positive")
        if (
            self.temperature is not None
            and not 0.0 <= self.temperature <= PING_TEMPERATURE_MAX
        ):
            raise ValueError(
                f"the ping's temperature must lie in [0.0, {PING_TEMPERATURE_MAX}]"
            )


@dataclass(frozen=True)
class KeepaliveSnapshot:
    """The IMMUTABLE serialized cache-relevant prefix, pinned to one deployment.

    Frozen and bytes-backed: once taken, the prefix can never be mutated in
    place. Any invalidation DROPS the snapshot and a later re-snapshot mints
    a NEW object — the dropped one keeps its exact bytes and digest. The
    snapshot is only ever live for the deployment/provider pair it was taken
    on; the ping re-sends the prefix on that same deployment/provider only.
    """

    deployment: str
    provider: str
    serialized: bytes
    generation: int
    created_at: datetime
    ping_settings: PingGenerationSettings = field(
        default_factory=PingGenerationSettings
    )

    def __post_init__(self) -> None:
        if not isinstance(self.serialized, bytes) or not self.serialized:
            raise ValueError("the snapshotted prefix must be non-empty bytes")

    @property
    def digest(self) -> str:
        """The sha256 stability hash of the serialized prefix.

        ``bytes`` are immutable, so the digest is stable for the life of the
        snapshot — a "mutated" prefix would be a different object entirely.
        """
        return hashlib.sha256(self.serialized).hexdigest()

    @property
    def byte_length(self) -> int:
        return len(self.serialized)

    def is_for(self, deployment: str, provider: str) -> bool:
        """Whether this snapshot was taken on the same deployment/provider."""
        return self.deployment == deployment and self.provider == provider


class KeepaliveSnapshotError(Exception):
    """Base for the snapshot section's local denials."""


class KeepaliveSnapshotMissingError(KeepaliveSnapshotError):
    """No live snapshot for the requested deployment/provider: no ping."""


@dataclass(frozen=True)
class KeepaliveInvalidationRecord:
    """One recorded invalidation: the trigger, an optional detail, when."""

    trigger: KeepaliveInvalidation
    detail: str | None
    at: datetime


class KeepaliveSnapshotStore:
    """Holds at most ONE live cache-prefix snapshot (design P5, SNAPSHOT).

    ``snapshot`` freezes the serialized cache-relevant prefix together with
    the deployment/provider it was taken on and the ping-only generation
    settings. Every one of the six invalidation triggers — prompt swap,
    config swap, tool change, compaction, ANY new conversation activity
    (partial wakes included), deployment failover — drops the live snapshot
    via ``invalidate`` (never mutating it), so a ping can only ever re-send a
    prefix that is still the conversation's exact cache-relevant prefix on
    the same deployment/provider.
    """

    def __init__(self, *, clock: Callable[[], datetime] | None = None) -> None:
        self._clock = clock if clock is not None else lambda: datetime.now(UTC)
        self._lock = Lock()
        self._snapshot: KeepaliveSnapshot | None = None
        self._generation = 0
        self._invalidations: list[KeepaliveInvalidationRecord] = []

    def snapshot(
        self,
        *,
        deployment: str,
        provider: str,
        serialized: bytes,
        ping_settings: PingGenerationSettings | None = None,
    ) -> KeepaliveSnapshot:
        """Freeze the serialized cache-relevant prefix as the one live snapshot.

        A re-snapshot after an invalidation mints a NEW immutable snapshot
        (a fresh generation); holders of the dropped one keep it unmutated.
        """
        with self._lock:
            # Fail-closed: a rejected prefix (e.g. empty) raises before the
            # generation is consumed or any snapshot is stored.
            frozen = KeepaliveSnapshot(
                deployment=deployment,
                provider=provider,
                serialized=serialized,
                generation=self._generation + 1,
                created_at=self._clock(),
                ping_settings=(
                    ping_settings
                    if ping_settings is not None
                    else PingGenerationSettings()
                ),
            )
            self._generation += 1
            self._snapshot = frozen
            return frozen

    def current(self) -> KeepaliveSnapshot | None:
        """The live snapshot, or None when absent/invalidated."""
        with self._lock:
            return self._snapshot

    def is_live_for(self, deployment: str, provider: str) -> bool:
        """Whether a live snapshot exists AND pins this deployment/provider."""
        with self._lock:
            return self._snapshot is not None and self._snapshot.is_for(
                deployment, provider
            )

    def require_live(self, deployment: str, provider: str) -> KeepaliveSnapshot:
        """The live snapshot for this deployment/provider, or fail-closed raise.

        A snapshot taken on a different deployment/provider does not count:
        the prefix may only be re-sent on the same deployment/provider it was
        snapshotted on — a failover changes the deployment.
        """
        with self._lock:
            live = self._snapshot
        if live is None:
            raise KeepaliveSnapshotMissingError("no live keepalive snapshot")
        if not live.is_for(deployment, provider):
            raise KeepaliveSnapshotMissingError(
                f"the live snapshot pins {live.provider}/{live.deployment}, "
                f"not {provider}/{deployment}"
            )
        return live

    def invalidate(
        self, trigger: KeepaliveInvalidation, *, detail: str | None = None
    ) -> KeepaliveSnapshot | None:
        """INVALIDATE the live snapshot on one of the six triggers.

        The dropped snapshot is returned unmutated (None when none was
        live); the trigger is recorded in the invalidation log.
        """
        with self._lock:
            dropped = self._snapshot
            self._snapshot = None
            self._invalidations.append(
                KeepaliveInvalidationRecord(
                    trigger=trigger, detail=detail, at=self._clock()
                )
            )
            return dropped

    def note_conversation_activity(
        self, *, partial_wake: bool = False
    ) -> KeepaliveSnapshot | None:
        """Record ANY new conversation activity — partial wakes included.

        Any activity at all invalidates: even a partial wake (a notification
        that did not resume the conversation) still changes what the next
        request re-sends, so the cached prefix is no longer exact.
        """
        return self.invalidate(
            KeepaliveInvalidation.CONVERSATION_ACTIVITY,
            detail="partial_wake" if partial_wake else None,
        )

    @property
    def invalidations(self) -> tuple[KeepaliveInvalidationRecord, ...]:
        """The invalidation log, oldest first."""
        with self._lock:
            return tuple(self._invalidations)

    @property
    def generation(self) -> int:
        """How many snapshots have been minted (each a fresh generation)."""
        with self._lock:
            return self._generation


# ---------------------------------------------------------------------------
# SCHEDULER section (chunk 5c)
# ---------------------------------------------------------------------------

#: Default tick interval of the idle-cycle timer.
DEFAULT_SCHEDULER_TICK_SECONDS = 30.0

#: Default floor for the ping's cached-prefix coverage: the fraction of the
#: ping's reported input tokens that must have been served from the provider
#: cache before the OBSERVE section considers the prefix hit sufficient.
DEFAULT_MIN_CACHED_PREFIX_COVERAGE = 0.9


class KeepaliveStopReason(StrEnum):
    """Why the OBSERVE section stopped the current idle cycle (design P5)."""

    CACHE_MISS = "cache_miss"
    MISSING_USAGE = "missing_usage"
    PARTIAL_USAGE = "partial_usage"
    ERROR = "error"
    INSUFFICIENT_CACHED_COVERAGE = "insufficient_cached_coverage"
    CAP_EXCEEDED = "cap_exceeded"


class KeepalivePhase(StrEnum):
    """The scheduler's one-ping slot: IDLE -> RESERVED -> IN_FLIGHT -> IDLE."""

    IDLE = "idle"
    RESERVED = "reserved"
    IN_FLIGHT = "in_flight"


@dataclass(frozen=True)
class KeepalivePing:
    """One reserved ping handed to the transport."""

    snapshot: KeepaliveSnapshot
    reservation: KeepaliveReservation


class PingTransport(Protocol):
    """One keep-alive ping round-trip, driven by the scheduler.

    The implementation re-sends the snapshot's serialized prefix on the SAME
    deployment/provider (production: the gateway's non-streaming complete with
    ``purpose=keepalive``, ``account_conversation=False``, and an accounting
    sink that never appends to any transcript). The returned chunk is consumed
    by the OBSERVE section ONLY: its tool calls are never executed and its
    message is never appended anywhere.
    """

    async def __call__(self, ping: KeepalivePing, /) -> LLMChunk: ...


@dataclass(frozen=True)
class KeepalivePingRecord:
    """What the scheduler keeps of one ping — NEVER the response message.

    The ping response is dropped at the transport boundary: never appended to
    any transcript, its tool calls never executed. Only the accounting and the
    observation survive here; no reference to the response is stored.
    """

    attempt_ordinal: int
    phase_reached: KeepalivePhase
    observation: PingObservation | None
    stop: KeepaliveStopReason | None
    superseded: bool


@dataclass(frozen=True)
class KeepaliveGates:
    """The scheduler's eligibility gates (design P5, SCHEDULER).

    ``is_root``: ROOT-ONLY — children never ping (production wiring: the
    loop's ``not is_subagent()``). ``owned_dependency_runs_active``: whether
    owned dependency runs are active, read through the P2 continuation
    machinery (the quiescence snapshot's active dependency runs) as a
    READ-ONLY probe. ``min_idle_gap_seconds``: the idle-gap floor, from the
    ``[keepalive]`` config section.
    """

    is_root: bool | Callable[[], bool]
    owned_dependency_runs_active: Callable[[], bool]
    min_idle_gap_seconds: float


class KeepaliveScheduler:
    """ROOT-ONLY idle-cycle scheduler for the keep-alive pilot (P5, chunk 5c).

    A ping is considered only while ALL gates hold:

    - the pilot flag is enabled (FLAG OFF => zero behavior: no snapshot, no
      timer, no ledger row);
    - the loop is the ROOT session — never children;
    - owned dependency runs are active (the P2 continuation machinery's
      quiescence snapshot, read here through a READ-ONLY probe);
    - the idle gap since the last conversation activity is >= the configured
      floor (``min_idle_gap_seconds``);
    - no real call is active and no ping is open (at most ONE ping in flight);
    - a live snapshot exists for this deployment/provider.

    Supersession — a real call starting while a ping is open — applies in BOTH
    phases: a RESERVED ping (pre-send) is settled ``superseded`` (released
    unspent); an IN-FLIGHT ping is cancelled and settled ``cancelled`` (the
    wire request may have been billed, so its full reservation is retained).
    """

    def __init__(
        self,
        *,
        enabled: bool,
        gates: KeepaliveGates,
        snapshot_store: KeepaliveSnapshotStore,
        accountant: KeepaliveAccountant,
        deployment: str,
        provider: str,
        transport: PingTransport | None = None,
        clock: Callable[[], datetime] | None = None,
        min_cached_prefix_coverage: float = DEFAULT_MIN_CACHED_PREFIX_COVERAGE,
        tick_seconds: float = DEFAULT_SCHEDULER_TICK_SECONDS,
    ) -> None:
        if gates.min_idle_gap_seconds < 0:
            raise ValueError("min_idle_gap_seconds must be non-negative")
        if not 0.0 < min_cached_prefix_coverage <= 1.0:
            raise ValueError("min_cached_prefix_coverage must lie in (0.0, 1.0]")
        if tick_seconds <= 0:
            raise ValueError("tick_seconds must be positive")
        self._enabled = enabled
        self._gates = gates
        self._snapshot_store = snapshot_store
        self._accountant = accountant
        self._deployment = deployment
        self._provider = provider
        self._transport = transport
        self._clock = clock if clock is not None else lambda: datetime.now(UTC)
        self._min_cached_prefix_coverage = min_cached_prefix_coverage
        self._tick_seconds = tick_seconds
        self._lock = Lock()
        self._phase = KeepalivePhase.IDLE
        self._open_reservation: KeepaliveReservation | None = None
        self._in_flight: asyncio.Task[KeepalivePingRecord] | None = None
        self._real_call_active = False
        self._last_activity_at = self._clock()
        self._cycle_stopped: KeepaliveStopReason | None = None
        self._records: list[KeepalivePingRecord] = []

    @classmethod
    def from_config(
        cls, config: KeepaliveSectionConfig, *, gates: KeepaliveGates, **kwargs: Any
    ) -> KeepaliveScheduler:
        """Build the scheduler from the ``[keepalive]`` config section.

        The flag gates everything: with ``enabled`` FALSE the scheduler is
        inert — zero behavior — whatever the other inputs are. The idle-gap
        floor comes from the config section, overriding the gates' own value.
        """
        return cls(
            enabled=config.enabled,
            gates=replace(gates, min_idle_gap_seconds=config.min_idle_gap_seconds),
            **kwargs,
        )

    @property
    def enabled(self) -> bool:
        """Whether the pilot flag is on; off means zero behavior."""
        return self._enabled

    @property
    def phase(self) -> KeepalivePhase:
        """The one-ping slot's current phase."""
        with self._lock:
            return self._phase

    @property
    def in_flight(self) -> bool:
        """Whether a ping is open (reserved or in flight) — at most one."""
        with self._lock:
            return self._phase is not KeepalivePhase.IDLE

    @property
    def stopped(self) -> KeepaliveStopReason | None:
        """The current idle cycle's STOP reason, if it was stopped."""
        with self._lock:
            return self._cycle_stopped

    @property
    def records(self) -> tuple[KeepalivePingRecord, ...]:
        """Every settled ping, oldest first — never the responses."""
        with self._lock:
            return tuple(self._records)

    def _root(self) -> bool:
        is_root = self._gates.is_root
        return is_root() if callable(is_root) else is_root

    def idle_gap_seconds(self) -> float:
        """Seconds since the last recorded conversation activity."""
        with self._lock:
            last = self._last_activity_at
        return (self._clock() - last).total_seconds()

    def refresh_snapshot(self, *, serialized: bytes) -> KeepaliveSnapshot | None:
        """Freeze the current cache-relevant prefix as the live snapshot.

        FLAG OFF (or a child loop) => None and NO snapshot is taken.
        """
        if not self._enabled or not self._root():
            return None
        return self._snapshot_store.snapshot(
            deployment=self._deployment, provider=self._provider, serialized=serialized
        )

    def note_conversation_activity(self, *, partial_wake: bool = False) -> None:
        """ANY new conversation activity (partial wakes included).

        Invalidates the live snapshot (the 5b trigger) and opens a NEW idle
        cycle: the previous cycle's STOP does not carry over.
        """
        self._snapshot_store.note_conversation_activity(partial_wake=partial_wake)
        with self._lock:
            self._last_activity_at = self._clock()
            self._cycle_stopped = None

    def notify_real_call_started(self) -> None:
        """A real (conversation) call started.

        A real call is conversation activity: it invalidates the live snapshot
        (the 5b trigger) and restarts the idle gap. It also SUPERSEDES any
        open ping — in both phases: a RESERVED ping is settled ``superseded``
        here (released unspent); an IN-FLIGHT ping task is cancelled, and the
        task itself settles it ``cancelled`` (full reservation retained).
        """
        self._snapshot_store.note_conversation_activity()
        with self._lock:
            self._real_call_active = True
            self._last_activity_at = self._clock()
            self._cycle_stopped = None
            reservation = self._open_reservation
            task = self._in_flight
            phase = self._phase
            if phase is KeepalivePhase.RESERVED and reservation is not None:
                self._phase = KeepalivePhase.IDLE
                self._open_reservation = None
                self._in_flight = None
                supersede = reservation
            else:
                supersede = None
        if supersede is not None:
            # Pre-send supersession: released unspent, still a counted attempt.
            self._accountant.supersede(supersede)
        elif phase is KeepalivePhase.IN_FLIGHT and task is not None and not task.done():
            task.cancel()

    def notify_real_call_finished(self) -> None:
        """A real (conversation) call finished: the idle gap restarts here."""
        with self._lock:
            self._real_call_active = False
            self._last_activity_at = self._clock()

    def _gate(self) -> KeepaliveSnapshot | None:
        """The live snapshot when every scheduling gate passes, else None."""
        if not self._enabled or not self._root() or self._accountant.halted:
            return None
        if not self._gates.owned_dependency_runs_active():
            return None
        if self.idle_gap_seconds() < self._gates.min_idle_gap_seconds:
            return None
        with self._lock:
            blocked = (
                self._real_call_active
                or self._phase is not KeepalivePhase.IDLE
                or self._cycle_stopped is not None
            )
        if blocked:
            return None
        try:
            return self._snapshot_store.require_live(self._deployment, self._provider)
        except KeepaliveSnapshotMissingError:
            return None

    async def maybe_ping(self) -> asyncio.Task[KeepalivePingRecord] | None:
        """One scheduler tick: reserve + launch at most one ping.

        Returns the in-flight ping task (it resolves to the ping's record), or
        None when any gate refuses. Gate refusals never raise; the
        accountant's fail-closed halt is a silent no-op and a cap denial stops
        the current idle cycle.
        """
        snapshot = self._gate()
        if snapshot is None or self._transport is None:
            return None
        try:
            reservation = self._accountant.reserve(
                snapshot.byte_length, snapshot.ping_settings.max_tokens
            )
        except KeepaliveFailClosedError:
            return None
        except KeepaliveCapExceededError:
            with self._lock:
                self._cycle_stopped = KeepaliveStopReason.CAP_EXCEEDED
            return None
        with self._lock:
            self._phase = KeepalivePhase.RESERVED
            self._open_reservation = reservation
        task = asyncio.create_task(self._send(snapshot, reservation))
        with self._lock:
            self._in_flight = task
        return task

    async def run(
        self, sleep: Callable[[float], Awaitable[None]] = asyncio.sleep
    ) -> None:
        """The idle-cycle timer.

        FLAG OFF => returns immediately: no timer, no tick, no ping. While
        enabled, ticks ``maybe_ping`` every ``tick_seconds`` until cancelled.
        """
        if not self._enabled:
            return
        while True:
            await self.maybe_ping()
            await sleep(self._tick_seconds)

    async def _send(
        self, snapshot: KeepaliveSnapshot, reservation: KeepaliveReservation
    ) -> KeepalivePingRecord:
        with self._lock:
            superseded_pre_send = (
                self._open_reservation is not reservation
                or self._phase is not KeepalivePhase.RESERVED
            )
            if superseded_pre_send:
                phase_reached = KeepalivePhase.RESERVED
            else:
                self._phase = KeepalivePhase.IN_FLIGHT
                phase_reached = KeepalivePhase.IN_FLIGHT
        if superseded_pre_send:
            # Superseded pre-send before the transport started (settled by
            # notify_real_call_started): no request is sent.
            return self._finish(
                reservation, phase_reached, observation=None, stop=None, superseded=True
            )
        observation: PingObservation | None = None
        stop: KeepaliveStopReason | None = None
        superseded = False
        try:
            assert self._transport is not None
            chunk = await self._transport(
                KeepalivePing(snapshot=snapshot, reservation=reservation)
            )
            # OBSERVE the ping's OWN reported usage. The response message is
            # dropped here: never appended, its tool calls never executed.
            observation = observe_ping_usage(
                chunk.usage, min_cached_prefix_coverage=self._min_cached_prefix_coverage
            )
            stop = observation.stop
            tokens = observation.usage
            if tokens is not None and tokens.presence_state is UsageState.COMPLETE:
                self._accountant.reconcile(
                    reservation,
                    input_tokens=tokens.input_tokens,
                    output_tokens=tokens.output_tokens,
                    cached_input_tokens=tokens.cached_input_tokens,
                )
            else:
                self._accountant.mark_unknown_usage(reservation)
        except asyncio.CancelledError:
            # In-flight supersession by a real call: the wire request may have
            # been billed, so the FULL reservation is retained.
            superseded = True
            self._accountant.cancel(reservation)
        except Exception:
            stop = KeepaliveStopReason.ERROR
            self._accountant.mark_unknown_usage(reservation)
        return self._finish(
            reservation,
            phase_reached,
            observation=observation,
            stop=stop,
            superseded=superseded,
        )

    def _finish(
        self,
        reservation: KeepaliveReservation,
        phase_reached: KeepalivePhase,
        *,
        observation: PingObservation | None,
        stop: KeepaliveStopReason | None,
        superseded: bool,
    ) -> KeepalivePingRecord:
        with self._lock:
            if self._open_reservation is reservation:
                self._open_reservation = None
                self._phase = KeepalivePhase.IDLE
                self._in_flight = None
            if stop is not None and self._cycle_stopped is None:
                self._cycle_stopped = stop
            record = KeepalivePingRecord(
                attempt_ordinal=reservation.attempt_ordinal,
                phase_reached=phase_reached,
                observation=observation,
                stop=stop,
                superseded=superseded,
            )
            self._records.append(record)
        return record


# ---------------------------------------------------------------------------
# OBSERVE section (chunk 5c)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PingObservation:
    """The OBSERVE section's verdict on one ping's OWN reported usage.

    ``stop`` is set when the idle cycle must stop: a cache MISS, missing or
    partial usage, or insufficient cached-prefix coverage. ``usage`` carries
    the reported component counts (unreported components stay unknown, never
    fabricated as zero counts).
    """

    usage: UsageTokens | None
    cached_coverage: float | None
    stop: KeepaliveStopReason | None

    @property
    def stopped(self) -> bool:
        """Whether this observation stops the idle cycle."""
        return self.stop is not None


def observe_ping_usage(
    usage: LLMUsage | None,
    *,
    min_cached_prefix_coverage: float = DEFAULT_MIN_CACHED_PREFIX_COVERAGE,
) -> PingObservation:
    """Observe one ping's own reported usage (design P5, OBSERVE).

    STOP the idle cycle on a cache MISS (nothing served from the cache),
    missing or partial usage, or insufficient cached-prefix coverage (the
    cached share of the reported input below the configured floor). A
    full-coverage hit observes cleanly. A transport error is observed by the
    scheduler as ``KeepaliveStopReason.ERROR``.
    """
    if usage is None:
        return PingObservation(
            usage=None, cached_coverage=None, stop=KeepaliveStopReason.MISSING_USAGE
        )
    tokens = UsageTokens(
        input_tokens=usage.prompt_tokens if usage.prompt_tokens_reported else None,
        output_tokens=usage.completion_tokens
        if usage.completion_tokens_reported
        else None,
        cached_input_tokens=usage.cached_tokens
        if usage.cached_tokens_reported
        else None,
    )
    state = tokens.presence_state
    if state is UsageState.MISSING:
        return PingObservation(
            usage=tokens, cached_coverage=None, stop=KeepaliveStopReason.MISSING_USAGE
        )
    if state is UsageState.PARTIAL:
        return PingObservation(
            usage=tokens, cached_coverage=None, stop=KeepaliveStopReason.PARTIAL_USAGE
        )
    input_tokens = tokens.input_tokens or 0
    cached_tokens = tokens.cached_input_tokens or 0
    coverage = cached_tokens / input_tokens if input_tokens > 0 else 0.0
    if cached_tokens == 0:
        return PingObservation(
            usage=tokens, cached_coverage=coverage, stop=KeepaliveStopReason.CACHE_MISS
        )
    if coverage < min_cached_prefix_coverage:
        return PingObservation(
            usage=tokens,
            cached_coverage=coverage,
            stop=KeepaliveStopReason.INSUFFICIENT_CACHED_COVERAGE,
        )
    return PingObservation(usage=tokens, cached_coverage=coverage, stop=None)
