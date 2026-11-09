"""P5 chunk 5a: the strict ping accountant (core/agent_loop/_keepalive.py).

The settled contract under test: the accountant is synchronous, conservative,
and fail-closed — before each ping it reserves the prospective MAXIMUM cost
against the absolute count + spend caps, priced in BILLABLE tokens (a
proven-conservative provider-specific input bound incl. framing at the
uncached input rate, plus declared max completion tokens at the completion
price); without the conversion, prices, declared limit, attribution, or ledger
sink it halts the deployment and no ping is sent. Supersession releases the
reservation unspent; unknown-usage and cancelled attempts retain their full
reservation; reconciliation on recorded usage releases reservation-minus-
actual; the count cap counts ATTEMPTS (superseded included); every attempt
writes a purpose=keepalive ledger row so the ledger reconstructs cap burn; and
the gateway's ping path bypasses the finalize missing-sink tolerance — pings
REQUIRE a sink, while ordinary calls keep the tolerance.

Chunk 5b (TestSnapshot): the SNAPSHOT section — the IMMUTABLE serialized
cache-relevant prefix is snapshotted pinned to one deployment/provider and
INVALIDATED (dropped, never mutated) on any of the six design triggers
(prompt swap, config swap, tool change, compaction, ANY new conversation
activity incl. partial wakes, deployment failover); the ping-only generation
settings are specified separately from the session's ordinary generation
settings; and the [keepalive] config section ships enabled=false with
validated caps (absolute count cap, spend cap, idle-gap floor).

Chunk 5c (TestScheduling, TestObserve, TestBudgetIsolation, TestFlagOff): the
SCHEDULER + OBSERVE sections — ROOT-ONLY (never children), only while owned
dependency runs are active AND the idle gap >= the configured floor, at most
one ping in flight, supersession in BOTH phases (reserved pre-send =>
``superseded`` released unspent; in-flight => cancelled, full reservation
retained), ping responses never executed and never appended; the ping's OWN
reported usage is observed and the idle cycle STOPs on cache miss,
missing/partial usage, error, or insufficient cached-prefix coverage; the
purpose split at the gateway's stats accumulation excludes pings from the
session totals AND known_cost_total so the pilot cannot flip a token/price
STOP (core/middleware.py:64-106) — the divergence is accepted and the ledger
(purpose=keepalive) is the single source of truth for ping spend; FLAG OFF =>
zero behavior (no snapshot, no timer, no ledger row — asserted).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import FrozenInstanceError, fields, replace
from datetime import UTC, datetime, timedelta
import hashlib
import json
from math import ceil
from pathlib import Path
from typing import Any

from pydantic import ValidationError
import pytest

from chartreux.core.agent_loop._keepalive import (
    DEFAULT_MIN_CACHED_PREFIX_COVERAGE,
    DEFAULT_PING_MAX_TOKENS,
    BillableInputRule,
    KeepaliveAccountant,
    KeepaliveCapExceededError,
    KeepaliveCaps,
    KeepaliveFailClosedError,
    KeepaliveGates,
    KeepaliveInvalidation,
    KeepaliveOutcome,
    KeepalivePhase,
    KeepalivePing,
    KeepalivePingRecord,
    KeepaliveReservationStateError,
    KeepaliveScheduler,
    KeepaliveSnapshot,
    KeepaliveSnapshotMissingError,
    KeepaliveSnapshotStore,
    KeepaliveStopReason,
    PingGenerationSettings,
    PingTransport,
    observe_ping_usage,
    reconstruct_keepalive_burn,
    register_billable_input_rule,
)
from chartreux.core.agent_loop.errors import KeepaliveSinkRequiredError
from chartreux.core.agent_loop.llm_gateway import (
    CallResources,
    CompletionInputs,
    LLMGateway,
    _apply_interrupted_usage,
    apply_usage,
)
from chartreux.core.config import ModelConfig
from chartreux.core.config.chartreux_schema import (
    ChartreuxConfigSchema,
    create_default_config,
)
from chartreux.core.config.models import (
    DEFAULT_KEEPALIVE_MAX_ATTEMPTS,
    DEFAULT_KEEPALIVE_MAX_SPEND_USD,
    DEFAULT_KEEPALIVE_MIN_IDLE_GAP_SECONDS,
    KeepaliveSectionConfig,
)
from chartreux.core.llm.backend.generic import notify_request_started
from chartreux.core.llm_models import (
    FunctionCall,
    LLMChunk,
    LLMMessage,
    LLMUsage,
    Role,
    StopInfo,
    ToolCall,
)
from chartreux.core.message_list import MessageList
from chartreux.core.middleware import (
    ConversationContext,
    MiddlewareAction,
    PriceLimitMiddleware,
    TokenLimitMiddleware,
)
from chartreux.core.session_types import AgentStats
from chartreux.core.usage import (
    KeepaliveLedgerEntry,
    SnapshotState,
    UsageAttribution,
    UsagePurpose,
    UsageReader,
    UsageRecord,
    UsageState,
    UsageWriteDisposition,
    UsageWriter,
    capture_prices,
    price_usage,
)
from tests.stubs.fake_backend import FakeBackend

# The synthetic provider bills ceil(bytes / 4) + 16 framing tokens; the
# registered rule (ceil(bytes / 3) + 32) provably dominates it for every size.
SYNTHETIC_CHARS_PER_TOKEN = 4
SYNTHETIC_FRAMING_TOKENS = 16
SYNTHETIC_RULE = BillableInputRule(chars_per_token=3.0, framing_tokens=32)
register_billable_input_rule("synthetic", SYNTHETIC_RULE)

INPUT_PRICE = 3.0
OUTPUT_PRICE = 6.0
CACHED_INPUT_PRICE = 0.5
PING_MAX_TOKENS = 64
ATTRIBUTION = UsageAttribution(
    root_session_id="root",
    session_id="session",
    agent_role="root",
    model="stale-model",
    provider="stale-provider",
    wire_name="stale-wire",
    project_key="project",
)


def synthetic_billed_input_tokens(request_bytes: int) -> int:
    """What the synthetic provider bills for a complete request, framing included."""
    return ceil(request_bytes / SYNTHETIC_CHARS_PER_TOKEN) + SYNTHETIC_FRAMING_TOKENS


def _model(**overrides: Any) -> ModelConfig:
    return ModelConfig(
        name="wire",
        provider="synthetic",
        alias="alias-x",
        supports_images=True,
        input_price=INPUT_PRICE,
        output_price=OUTPUT_PRICE,
        cached_input_price=CACHED_INPUT_PRICE,
        **overrides,
    )


class _ListSink:
    def __init__(self) -> None:
        self.records: list[UsageRecord] = []

    def __call__(self, record: UsageRecord) -> None:
        self.records.append(record)


def _accountant(
    *,
    caps: KeepaliveCaps | None = None,
    model: ModelConfig | None = None,
    provider: str = "synthetic",
    ledger: _ListSink | None = None,
    attribution: UsageAttribution | None = ATTRIBUTION,
) -> KeepaliveAccountant:
    return KeepaliveAccountant(
        caps=caps or KeepaliveCaps(max_attempts=100, max_spend_usd=1.0),
        model=model or _model(),
        provider=provider,
        attribution=attribution,
        ledger=ledger if ledger is not None else _ListSink(),
    )


def _entry(record: UsageRecord) -> KeepaliveLedgerEntry:
    assert record.keepalive is not None
    return record.keepalive


def _chunk() -> LLMChunk:
    return LLMChunk(
        message=LLMMessage(role=Role.assistant, content="pong"),
        usage=LLMUsage(prompt_tokens=10, completion_tokens=2, cached_tokens=0),
        stop=StopInfo(reason="stop"),
    )


def _inputs(purpose: UsagePurpose) -> CompletionInputs:
    return CompletionInputs(
        model=_model(),
        provider_name="synthetic",
        emits_finish_reason=True,
        messages=(
            LLMMessage(role=Role.system, content="You are a focused helper."),
            LLMMessage(role=Role.user, content="ping"),
        ),
        tools=None,
        tool_choice=None,
        extra_headers={},
        metadata={},
        max_tokens=PING_MAX_TOKENS,
        purpose=purpose,
    )


class _AttemptBackend(FakeBackend):
    """A backend that counts its calls and reports its transport start."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    async def complete(self, **kwargs: Any) -> LLMChunk:
        self.calls += 1
        notify_request_started()
        return _chunk()


class TestAccountant:
    def test_fail_closed_without_a_registered_conversion_halts_the_deployment(self):
        sink = _ListSink()
        accountant = _accountant(provider="unregistered-provider", ledger=sink)

        with pytest.raises(KeepaliveFailClosedError):
            accountant.reserve(1000, PING_MAX_TOKENS)
        assert accountant.halted
        assert accountant.attempts == 0
        # The halt sticks for the deployment: later reserves stay fail-closed.
        with pytest.raises(KeepaliveFailClosedError):
            accountant.reserve(1000, PING_MAX_TOKENS)
        assert sink.records == []

    def test_fail_closed_without_prices_halts_the_deployment(self):
        for overrides in ({"input_price_known": False}, {"output_price_known": False}):
            accountant = _accountant(model=_model(**overrides))
            with pytest.raises(KeepaliveFailClosedError):
                accountant.reserve(1000, PING_MAX_TOKENS)
            assert accountant.halted
            assert accountant.attempts == 0

    def test_fail_closed_without_a_ledger_sink_or_attribution(self):
        no_sink = KeepaliveAccountant(
            caps=KeepaliveCaps(max_attempts=10, max_spend_usd=1.0),
            model=_model(),
            provider="synthetic",
            attribution=ATTRIBUTION,
            ledger=None,
        )
        with pytest.raises(KeepaliveFailClosedError):
            no_sink.reserve(1000, PING_MAX_TOKENS)
        assert no_sink.halted

        no_attribution = _accountant(attribution=None)
        with pytest.raises(KeepaliveFailClosedError):
            no_attribution.reserve(1000, PING_MAX_TOKENS)
        assert no_attribution.halted

    def test_fail_closed_without_a_declared_output_limit(self):
        accountant = _accountant()
        with pytest.raises(KeepaliveFailClosedError):
            accountant.reserve(1000, None)
        assert accountant.halted
        assert accountant.attempts == 0

    @pytest.mark.parametrize(
        "request_bytes",
        [0, 1, 3, 4, 5, 12, 16, 17, 63, 64, 100, 255, 1024, 4097, 65537],
    )
    def test_the_reservation_never_undercounts_billed_tokens_incl_framing(
        self, request_bytes: int
    ) -> None:
        accountant = _accountant()
        reservation = accountant.reserve(request_bytes, PING_MAX_TOKENS)

        billed = synthetic_billed_input_tokens(request_bytes)
        assert reservation.reserved_input_tokens >= billed
        assert reservation.reserved_input_tokens == SYNTHETIC_RULE.bound(request_bytes)

    def test_the_reservation_is_priced_in_billable_tokens_at_the_uncached_rate(self):
        accountant = _accountant()
        reservation = accountant.reserve(1000, PING_MAX_TOKENS)

        # Billable tokens, not bytes: pricing the byte count would differ.
        expected = (
            reservation.reserved_input_tokens * INPUT_PRICE
            + PING_MAX_TOKENS * OUTPUT_PRICE
        ) / 1_000_000
        assert reservation.reserved_cost_usd == pytest.approx(expected)
        assert reservation.reserved_cost_usd != pytest.approx(
            (1000 * INPUT_PRICE + PING_MAX_TOKENS * OUTPUT_PRICE) / 1_000_000
        )
        # The whole input bound is priced at the UNCACHED rate, never the
        # cheaper cached rate.
        assert (
            reservation.reserved_cost_usd
            > (
                reservation.reserved_input_tokens * CACHED_INPUT_PRICE
                + PING_MAX_TOKENS * OUTPUT_PRICE
            )
            / 1_000_000
        )

    def test_supersession_releases_the_reservation_unspent(self):
        sink = _ListSink()
        accountant = _accountant(ledger=sink)
        reservation = accountant.reserve(1000, PING_MAX_TOKENS)
        assert accountant.charged_spend_usd == pytest.approx(
            reservation.reserved_cost_usd
        )

        accountant.supersede(reservation)

        assert accountant.charged_spend_usd == 0.0
        # The superseded attempt still counts against the count cap.
        assert accountant.attempts == 1
        assert accountant.open_attempts == 0
        (row,) = sink.records
        assert row.purpose is UsagePurpose.KEEPALIVE
        assert row.keepalive is not None
        assert row.keepalive.outcome is KeepaliveOutcome.SUPERSEDED
        assert row.keepalive.retained_cost_usd == 0.0
        assert row.keepalive.reserved_cost_usd == pytest.approx(
            reservation.reserved_cost_usd
        )

    def test_unknown_usage_and_cancelled_attempts_retain_the_full_reservation(self):
        sink = _ListSink()
        accountant = _accountant(ledger=sink)
        first = accountant.reserve(1000, PING_MAX_TOKENS)
        second = accountant.reserve(2000, PING_MAX_TOKENS)

        accountant.mark_unknown_usage(first)
        accountant.cancel(second)

        assert accountant.charged_spend_usd == pytest.approx(
            first.reserved_cost_usd + second.reserved_cost_usd
        )
        assert accountant.attempts == 2
        assert [_entry(row).outcome for row in sink.records] == [
            KeepaliveOutcome.UNKNOWN_USAGE,
            KeepaliveOutcome.CANCELLED,
        ]
        for row, reservation in zip(sink.records, (first, second), strict=True):
            assert row.keepalive is not None
            assert row.keepalive.retained_cost_usd == pytest.approx(
                reservation.reserved_cost_usd
            )

    def test_reconciliation_releases_reservation_minus_actual(self):
        sink = _ListSink()
        accountant = _accountant(ledger=sink)
        request_bytes = 1000
        reservation = accountant.reserve(request_bytes, PING_MAX_TOKENS)
        billed_input = synthetic_billed_input_tokens(request_bytes)

        accountant.reconcile(
            reservation,
            input_tokens=billed_input,
            output_tokens=20,
            cached_input_tokens=100,
        )

        actual = price_usage(
            input_tokens=billed_input,
            output_tokens=20,
            cached_input_tokens=100,
            prices=capture_prices(_model()),
        ).known_cost_usd
        # The reservation is conservative: it never undercounts the billed input.
        assert reservation.reserved_input_tokens >= billed_input
        assert actual < reservation.reserved_cost_usd
        assert accountant.charged_spend_usd == pytest.approx(actual)
        (row,) = sink.records
        assert row.keepalive is not None
        assert row.keepalive.outcome is KeepaliveOutcome.COMPLETED
        assert row.keepalive.retained_cost_usd == pytest.approx(actual)
        assert row.keepalive.reserved_cost_usd == pytest.approx(
            reservation.reserved_cost_usd
        )
        assert row.purpose is UsagePurpose.KEEPALIVE
        assert row.input_tokens == billed_input
        assert row.usage_state is UsageState.COMPLETE

    def test_partial_recorded_usage_falls_back_to_unknown_usage(self):
        sink = _ListSink()
        accountant = _accountant(ledger=sink)
        reservation = accountant.reserve(1000, PING_MAX_TOKENS)

        accountant.reconcile(
            reservation, input_tokens=266, output_tokens=None, cached_input_tokens=None
        )

        assert accountant.charged_spend_usd == pytest.approx(
            reservation.reserved_cost_usd
        )
        assert _entry(sink.records[0]).outcome is KeepaliveOutcome.UNKNOWN_USAGE

    def test_the_count_cap_counts_attempts_not_completions(self):
        accountant = _accountant(caps=KeepaliveCaps(max_attempts=2, max_spend_usd=1.0))
        first = accountant.reserve(100, PING_MAX_TOKENS)
        # Superseded — never completed — yet still a counted attempt.
        accountant.supersede(first)
        accountant.reserve(100, PING_MAX_TOKENS)

        with pytest.raises(KeepaliveCapExceededError):
            accountant.reserve(100, PING_MAX_TOKENS)

        assert accountant.attempts == 2
        assert not accountant.halted

    def test_the_spend_cap_denies_a_reservation_that_does_not_fit(self):
        probe = _accountant()
        first = probe.reserve(1000, PING_MAX_TOKENS)
        sink = _ListSink()
        accountant = _accountant(
            caps=KeepaliveCaps(max_attempts=10, max_spend_usd=first.reserved_cost_usd),
            ledger=sink,
        )

        reserved = accountant.reserve(1000, PING_MAX_TOKENS)
        with pytest.raises(KeepaliveCapExceededError):
            accountant.reserve(1000, PING_MAX_TOKENS)

        assert accountant.attempts == 1
        assert accountant.charged_spend_usd == pytest.approx(reserved.reserved_cost_usd)
        assert sink.records == []

    def test_a_settled_reservation_cannot_be_settled_twice(self):
        sink = _ListSink()
        accountant = _accountant(ledger=sink)
        reservation = accountant.reserve(1000, PING_MAX_TOKENS)
        accountant.supersede(reservation)

        with pytest.raises(KeepaliveReservationStateError):
            accountant.cancel(reservation)

        assert len(sink.records) == 1

    def test_the_ledger_reconstructs_cap_burn_from_rows(self, tmp_path: Path) -> None:
        writer = UsageWriter(tmp_path / "usage")
        dispositions = []

        def sink(record: UsageRecord) -> None:
            dispositions.append(writer.append(record))

        accountant = KeepaliveAccountant(
            caps=KeepaliveCaps(max_attempts=10, max_spend_usd=1.0),
            model=_model(),
            provider="synthetic",
            attribution=ATTRIBUTION,
            ledger=sink,
        )
        superseded = accountant.reserve(1000, PING_MAX_TOKENS)
        accountant.supersede(superseded)
        unknown = accountant.reserve(2000, PING_MAX_TOKENS)
        accountant.mark_unknown_usage(unknown)
        cancelled = accountant.reserve(3000, PING_MAX_TOKENS)
        accountant.cancel(cancelled)
        completed = accountant.reserve(4000, PING_MAX_TOKENS)
        accountant.reconcile(
            completed, input_tokens=1010, output_tokens=8, cached_input_tokens=10
        )

        assert all(
            result.disposition is UsageWriteDisposition.APPENDED
            for result in dispositions
        )
        snapshot = UsageReader(tmp_path / "usage").reconcile()
        assert snapshot.state is SnapshotState.READY
        records = snapshot.records
        assert len(records) == 4
        assert all(record.purpose is UsagePurpose.KEEPALIVE for record in records)
        assert [_entry(record).outcome for record in records] == [
            KeepaliveOutcome.SUPERSEDED,
            KeepaliveOutcome.UNKNOWN_USAGE,
            KeepaliveOutcome.CANCELLED,
            KeepaliveOutcome.COMPLETED,
        ]

        burn = reconstruct_keepalive_burn(records)
        assert burn == accountant.burn()
        assert burn.attempts == 4
        expected_spend = (
            unknown.reserved_cost_usd
            + cancelled.reserved_cost_usd
            + (
                price_usage(
                    input_tokens=1010,
                    output_tokens=8,
                    cached_input_tokens=10,
                    prices=capture_prices(_model()),
                ).known_cost_usd
            )
        )
        assert burn.retained_spend_usd == pytest.approx(expected_spend)

        # The serialized row carries the keepalive entry; a row without one
        # keeps its exact pre-keepalive shape (the field is omitted when unset).
        first_line = (tmp_path / "usage" / "root" / "usage.jsonl").read_bytes()
        payload = json.loads(first_line.splitlines()[0])
        assert payload["purpose"] == "keepalive"
        assert payload["keepalive"]["outcome"] == "superseded"
        ordinary = records[0].model_copy(
            update={"purpose": UsagePurpose.CONVERSATION, "keepalive": None}
        )
        assert "keepalive" not in json.loads(ordinary.model_dump_json())

    @pytest.mark.asyncio
    async def test_the_gateway_rejects_a_keepalive_ping_without_a_sink(self) -> None:
        backend = _AttemptBackend()
        resources = CallResources(
            backend=backend,
            stats=AgentStats(),
            process_message=lambda message: message,
            accounting_sink=None,
            usage_attribution=None,
        )

        with pytest.raises(KeepaliveSinkRequiredError):
            await LLMGateway().complete(_inputs(UsagePurpose.KEEPALIVE), resources)

        # The ping is rejected before any transport: no request was sent.
        assert backend.calls == 0

    @pytest.mark.asyncio
    async def test_the_gateway_rejects_a_keepalive_ping_without_attribution(
        self,
    ) -> None:
        backend = _AttemptBackend()
        records: list[UsageRecord] = []

        async def sink(record: UsageRecord) -> None:
            records.append(record)

        resources = CallResources(
            backend=backend,
            stats=AgentStats(),
            process_message=lambda message: message,
            accounting_sink=sink,
            usage_attribution=None,
        )

        with pytest.raises(KeepaliveSinkRequiredError):
            await LLMGateway().complete(_inputs(UsagePurpose.KEEPALIVE), resources)

        assert backend.calls == 0
        assert records == []

    @pytest.mark.asyncio
    async def test_the_gateway_keeps_the_missing_sink_tolerance_for_ordinary_calls(
        self,
    ) -> None:
        backend = _AttemptBackend()
        resources = CallResources(
            backend=backend, stats=AgentStats(), process_message=lambda message: message
        )

        result = await LLMGateway().complete(
            _inputs(UsagePurpose.CONVERSATION), resources
        )

        assert result.message.content == "pong"
        assert backend.calls == 1


DEPLOYMENT = "alias-x"
PROVIDER = "synthetic"
PREFIX = b"serialized cache-relevant prefix: system prompt + tool definitions"


def _store() -> KeepaliveSnapshotStore:
    return KeepaliveSnapshotStore()


def _snapshot(store: KeepaliveSnapshotStore, **kwargs: Any) -> KeepaliveSnapshot:
    return store.snapshot(
        deployment=DEPLOYMENT, provider=PROVIDER, serialized=PREFIX, **kwargs
    )


class TestSnapshot:
    """P5 chunk 5b: the SNAPSHOT section and the [keepalive] config surface."""

    def test_the_snapshot_is_immutable_and_a_prompt_change_invalidates_never_mutates(
        self,
    ) -> None:
        store = _store()
        snap = _snapshot(
            store, ping_settings=PingGenerationSettings(max_tokens=PING_MAX_TOKENS)
        )

        assert store.current() is snap
        assert snap.serialized == PREFIX
        assert isinstance(snap.serialized, bytes)
        assert snap.digest == hashlib.sha256(PREFIX).hexdigest()
        assert snap.byte_length == len(PREFIX)

        dropped = store.invalidate(KeepaliveInvalidation.PROMPT_SWAP)

        # Invalidation DROPS the snapshot; the dropped object is never mutated.
        assert dropped is snap
        assert store.current() is None
        assert snap.serialized == PREFIX
        assert snap.digest == hashlib.sha256(PREFIX).hexdigest()
        # The snapshot is frozen outright: attribute mutation is refused.
        with pytest.raises(FrozenInstanceError):
            snap.serialized = b"mutated"  # type: ignore[misc]
        assert snap.serialized == PREFIX

        # A re-snapshot mints a NEW object (a fresh generation), not a mutation.
        again = _snapshot(store)
        assert again is not snap
        assert again.generation == snap.generation + 1
        assert again.serialized == PREFIX
        assert store.current() is again

    def test_the_design_lists_exactly_six_invalidation_triggers(self) -> None:
        assert len(KeepaliveInvalidation) == 6
        assert {trigger.value for trigger in KeepaliveInvalidation} == {
            "prompt_swap",
            "config_swap",
            "tool_change",
            "compaction",
            "conversation_activity",
            "deployment_failover",
        }

    @pytest.mark.parametrize("trigger", list(KeepaliveInvalidation))
    def test_every_one_of_the_six_triggers_invalidates_the_live_snapshot(
        self, trigger: KeepaliveInvalidation
    ) -> None:
        store = _store()
        snap = _snapshot(store)
        assert store.is_live_for(DEPLOYMENT, PROVIDER)

        dropped = store.invalidate(trigger)

        assert dropped is snap
        assert store.current() is None
        assert not store.is_live_for(DEPLOYMENT, PROVIDER)
        with pytest.raises(KeepaliveSnapshotMissingError):
            store.require_live(DEPLOYMENT, PROVIDER)
        (record,) = store.invalidations
        assert record.trigger is trigger
        # The dropped snapshot was not mutated by the invalidation.
        assert snap.serialized == PREFIX
        assert snap.digest == hashlib.sha256(PREFIX).hexdigest()

    def test_any_new_conversation_activity_including_partial_wakes_invalidates(
        self,
    ) -> None:
        store = _store()
        _snapshot(store)
        store.note_conversation_activity(partial_wake=True)
        assert store.current() is None
        (record,) = store.invalidations
        assert record.trigger is KeepaliveInvalidation.CONVERSATION_ACTIVITY
        assert record.detail == "partial_wake"

        # A full (non-partial) activity invalidates the same way, no detail.
        _snapshot(store)
        store.note_conversation_activity()
        assert store.current() is None
        assert store.invalidations[-1].detail is None

    def test_the_snapshot_is_pinned_to_the_same_deployment_and_provider(self) -> None:
        store = _store()
        snap = _snapshot(store)
        assert store.require_live(DEPLOYMENT, PROVIDER) is snap

        # A different deployment/provider (a failover) is NOT live even before
        # any explicit trigger fires: the ping may only re-send the prefix on
        # the same deployment/provider it was snapshotted on.
        assert not store.is_live_for("other-deployment", PROVIDER)
        assert not store.is_live_for(DEPLOYMENT, "other-provider")
        with pytest.raises(KeepaliveSnapshotMissingError):
            store.require_live("other-deployment", PROVIDER)
        with pytest.raises(KeepaliveSnapshotMissingError):
            store.require_live(DEPLOYMENT, "other-provider")

        # The explicit failover trigger drops it and logs the trigger.
        dropped = store.invalidate(KeepaliveInvalidation.DEPLOYMENT_FAILOVER)
        assert dropped is snap
        assert store.current() is None
        assert (
            store.invalidations[-1].trigger is KeepaliveInvalidation.DEPLOYMENT_FAILOVER
        )

        # After the failover the new deployment can snapshot its own prefix.
        moved = store.snapshot(
            deployment="other-deployment", provider=PROVIDER, serialized=PREFIX
        )
        assert store.is_live_for("other-deployment", PROVIDER)
        assert not store.is_live_for(DEPLOYMENT, PROVIDER)
        assert store.require_live("other-deployment", PROVIDER) is moved

    def test_a_snapshot_requires_a_non_empty_byte_prefix(self) -> None:
        store = _store()
        with pytest.raises(ValueError):
            store.snapshot(deployment=DEPLOYMENT, provider=PROVIDER, serialized=b"")
        assert store.current() is None
        assert store.generation == 0

    def test_ping_only_generation_settings_are_isolated_from_the_ordinary_settings(
        self,
    ) -> None:
        # The session's ordinary generation settings: an ordinary conversation
        # call with its own max_tokens and purpose.
        ordinary = replace(_inputs(UsagePurpose.CONVERSATION), max_tokens=4096)
        assert ordinary.max_tokens == 4096

        ping_settings = PingGenerationSettings(max_tokens=32, temperature=0.0)
        store = _store()
        snap = _snapshot(store, ping_settings=ping_settings)

        # The snapshot carries the PING-ONLY settings...
        assert snap.ping_settings is ping_settings
        assert snap.ping_settings.max_tokens == 32
        assert snap.ping_settings.temperature == 0.0
        # ...while the ordinary call's generation settings are untouched and
        # unread by the snapshot: distinct objects, no shared state.
        assert ordinary.max_tokens == 4096
        assert ordinary.purpose is UsagePurpose.CONVERSATION
        assert snap.ping_settings.max_tokens != ordinary.max_tokens

        # A snapshot without explicit ping settings gets the pilot's declared
        # ping bound — never the ordinary call's max_tokens.
        default_snap = _snapshot(_store())
        assert default_snap.ping_settings.max_tokens == DEFAULT_PING_MAX_TOKENS
        assert default_snap.ping_settings.max_tokens != ordinary.max_tokens

        # The ping settings are validated...
        with pytest.raises(ValueError):
            PingGenerationSettings(max_tokens=0)
        with pytest.raises(ValueError):
            PingGenerationSettings(temperature=2.5)
        # ...and frozen: neither surface can be mutated into the other.
        with pytest.raises(FrozenInstanceError):
            ping_settings.max_tokens = 4096  # type: ignore[misc]
        assert ping_settings.max_tokens == 32
        assert ordinary.max_tokens == 4096

    def test_the_keepalive_section_ships_disabled_by_default(self) -> None:
        section = KeepaliveSectionConfig()
        assert section.enabled is False

        config = ChartreuxConfigSchema()
        assert config.keepalive == section
        assert config.keepalive.enabled is False
        assert create_default_config()["keepalive"]["enabled"] is False

    def test_the_flag_off_default_writes_only_the_inert_surface(self) -> None:
        # With enabled=false the pilot wires nothing — no snapshot, no timer,
        # no ledger row (the runtime TestFlagOff nodes are chunk 5c's); at the
        # config level the shipped default is exactly the inert section.
        assert create_default_config()["keepalive"] == {
            "enabled": False,
            "max_attempts": DEFAULT_KEEPALIVE_MAX_ATTEMPTS,
            "max_spend_usd": DEFAULT_KEEPALIVE_MAX_SPEND_USD,
            "min_idle_gap_seconds": DEFAULT_KEEPALIVE_MIN_IDLE_GAP_SECONDS,
        }

    def test_the_keepalive_schema_validates_the_caps_like_every_other_section(
        self,
    ) -> None:
        with pytest.raises(ValidationError):
            KeepaliveSectionConfig(max_attempts=0)
        with pytest.raises(ValidationError):
            KeepaliveSectionConfig(max_attempts=-1)
        with pytest.raises(ValidationError):
            KeepaliveSectionConfig(max_attempts=2.5)  # type: ignore[arg-type] # strict int
        with pytest.raises(ValidationError):
            KeepaliveSectionConfig(max_spend_usd=-0.01)
        with pytest.raises(ValidationError):
            KeepaliveSectionConfig(max_spend_usd=float("nan"))
        with pytest.raises(ValidationError):
            KeepaliveSectionConfig(max_spend_usd=True)  # a bool is not a cap
        with pytest.raises(ValidationError):
            KeepaliveSectionConfig(min_idle_gap_seconds=-1.0)
        with pytest.raises(ValidationError):
            KeepaliveSectionConfig(min_idle_gap_seconds=True)
        with pytest.raises(ValidationError):
            KeepaliveSectionConfig(enabled=True, max_spend_usd=0.0)
        with pytest.raises(ValidationError):
            KeepaliveSectionConfig(enabled="yes")  # type: ignore[arg-type] # strict bool
        with pytest.raises(ValidationError):
            KeepaliveSectionConfig(unknown_key=1)  # type: ignore[call-issue] # extra="forbid"

        config = ChartreuxConfigSchema.model_validate({
            "keepalive": {
                "enabled": True,
                "max_attempts": 7,
                "max_spend_usd": 0.25,
                "min_idle_gap_seconds": 300.0,
            }
        })
        assert config.keepalive.enabled is True
        assert config.keepalive.max_attempts == 7
        assert config.keepalive.max_spend_usd == 0.25
        assert config.keepalive.min_idle_gap_seconds == 300.0

        # Deep-merge: a sparse layer keeps every other typed default.
        sparse = ChartreuxConfigSchema.model_validate({"keepalive": {"enabled": True}})
        assert sparse.keepalive.enabled is True
        assert sparse.keepalive.max_attempts == DEFAULT_KEEPALIVE_MAX_ATTEMPTS
        assert sparse.keepalive.max_spend_usd == DEFAULT_KEEPALIVE_MAX_SPEND_USD
        assert (
            sparse.keepalive.min_idle_gap_seconds
            == DEFAULT_KEEPALIVE_MIN_IDLE_GAP_SECONDS
        )


# ---------------------------------------------------------------------------
# Chunk 5c: TestScheduling, TestObserve, TestBudgetIsolation, TestFlagOff
# ---------------------------------------------------------------------------

HIT_USAGE = LLMUsage(prompt_tokens=100, completion_tokens=5, cached_tokens=100)
MISS_USAGE = LLMUsage(prompt_tokens=100, completion_tokens=5, cached_tokens=0)
LOW_COVERAGE_USAGE = LLMUsage(prompt_tokens=100, completion_tokens=5, cached_tokens=30)
PARTIAL_USAGE = LLMUsage(prompt_tokens=100)


class _Clock:
    """A manually advanced clock for the scheduler tests."""

    def __init__(self) -> None:
        self.now = datetime(2026, 11, 9, 12, 0, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


class _PingBackend(FakeBackend):
    """A backend whose ping responses carry a scripted usage/error/block."""

    def __init__(
        self,
        *,
        usage: LLMUsage | None = None,
        error: BaseException | None = None,
        block: asyncio.Event | None = None,
        tool_calls: list[ToolCall] | None = None,
    ) -> None:
        super().__init__()
        self.calls = 0
        self.requests: list[dict[str, Any]] = []
        self._usage = usage
        self._error = error
        self._block = block
        self._tool_calls = tool_calls

    async def complete(self, **kwargs: Any) -> LLMChunk:
        self.calls += 1
        self.requests.append(kwargs)
        notify_request_started()
        if self._block is not None:
            await self._block.wait()
        if self._error is not None:
            raise self._error
        return LLMChunk(
            message=LLMMessage(
                role=Role.assistant, content="pong", tool_calls=self._tool_calls
            ),
            usage=self._usage if self._usage is not None else HIT_USAGE,
            stop=StopInfo(reason="stop"),
        )


def _gateway_transport(
    backend: _PingBackend,
    stats: AgentStats,
    gateway_rows: list[UsageRecord],
    processed: list[LLMMessage] | None = None,
) -> PingTransport:
    """The production-shaped ping transport: the gateway's non-streaming
    complete with purpose=keepalive and account_conversation=False (the
    non-streaming path has no transcript sink, so a ping response can never
    be appended).
    """

    async def sink(record: UsageRecord) -> None:
        gateway_rows.append(record)

    def process_message(message: LLMMessage) -> LLMMessage:
        if processed is not None:
            processed.append(message)
        return message

    async def transport(ping: KeepalivePing) -> LLMChunk:
        inputs = CompletionInputs(
            model=_model(),
            provider_name="synthetic",
            emits_finish_reason=True,
            messages=(
                LLMMessage(role=Role.system, content="cache-relevant prefix"),
                LLMMessage(role=Role.user, content="ping"),
            ),
            tools=None,
            tool_choice=None,
            extra_headers={},
            metadata={},
            max_tokens=ping.snapshot.ping_settings.max_tokens,
            account_conversation=False,
            purpose=UsagePurpose.KEEPALIVE,
        )
        resources = CallResources(
            backend=backend,
            stats=stats,
            process_message=process_message,
            accounting_sink=sink,
            usage_attribution=ATTRIBUTION,
        )
        return await LLMGateway().complete(inputs, resources)

    return transport


class _Rig:
    """One scheduler test rig: scheduler plus its store, accountant, sink,
    clock, backend, and stats.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        is_root: bool | Callable[[], bool] = True,
        owned: bool = True,
        min_idle_gap_seconds: float = 240.0,
        caps: KeepaliveCaps | None = None,
        min_coverage: float = DEFAULT_MIN_CACHED_PREFIX_COVERAGE,
        usage: LLMUsage | None = None,
        error: BaseException | None = None,
        block: asyncio.Event | None = None,
        tool_calls: list[ToolCall] | None = None,
    ) -> None:
        self.clock = _Clock()
        self.store = KeepaliveSnapshotStore(clock=self.clock)
        self.sink = _ListSink()
        self.accountant = KeepaliveAccountant(
            caps=caps or KeepaliveCaps(max_attempts=100, max_spend_usd=1.0),
            model=_model(),
            provider="synthetic",
            attribution=ATTRIBUTION,
            ledger=self.sink,
            clock=self.clock,
        )
        self.backend = _PingBackend(
            usage=usage, error=error, block=block, tool_calls=tool_calls
        )
        self.stats = AgentStats()
        self.gateway_rows: list[UsageRecord] = []
        self.processed: list[LLMMessage] = []
        self._owned = owned
        self.scheduler = KeepaliveScheduler(
            enabled=enabled,
            gates=KeepaliveGates(
                is_root=is_root,
                owned_dependency_runs_active=lambda: self._owned,
                min_idle_gap_seconds=min_idle_gap_seconds,
            ),
            snapshot_store=self.store,
            accountant=self.accountant,
            deployment=DEPLOYMENT,
            provider=PROVIDER,
            transport=_gateway_transport(
                self.backend, self.stats, self.gateway_rows, self.processed
            ),
            clock=self.clock,
            min_cached_prefix_coverage=min_coverage,
        )

    def set_owned(self, owned: bool) -> None:
        self._owned = owned

    def snapshot(self) -> KeepaliveSnapshot:
        return self.store.snapshot(
            deployment=DEPLOYMENT, provider=PROVIDER, serialized=PREFIX
        )

    def refresh(self) -> KeepaliveSnapshot | None:
        return self.scheduler.refresh_snapshot(serialized=PREFIX)


class TestScheduling:
    """P5 chunk 5c: the SCHEDULER section's gates and supersession."""

    @pytest.mark.asyncio
    async def test_root_only_a_child_never_pings_or_snapshots(self) -> None:
        rig = _Rig(is_root=False)
        rig.snapshot()
        rig.clock.advance(300)

        assert await rig.scheduler.maybe_ping() is None
        assert rig.backend.calls == 0
        assert rig.accountant.attempts == 0
        assert rig.sink.records == []
        assert not rig.scheduler.in_flight

        # The child never snapshots either.
        child = _Rig(is_root=False)
        assert child.scheduler.refresh_snapshot(serialized=PREFIX) is None
        assert child.store.generation == 0
        assert child.store.current() is None

        # is_root as a callable getter (production wiring: not is_subagent()).
        getter = _Rig(is_root=lambda: False)
        getter.snapshot()
        getter.clock.advance(300)
        assert await getter.scheduler.maybe_ping() is None
        assert getter.backend.calls == 0

    @pytest.mark.asyncio
    async def test_the_owned_dependency_gate(self) -> None:
        rig = _Rig(owned=False)
        rig.snapshot()
        rig.clock.advance(300)

        assert await rig.scheduler.maybe_ping() is None
        assert rig.backend.calls == 0

        rig.set_owned(True)
        task = await rig.scheduler.maybe_ping()
        assert task is not None
        record = await task
        assert not record.superseded
        assert rig.backend.calls == 1

    @pytest.mark.asyncio
    async def test_the_idle_gap_floor(self) -> None:
        rig = _Rig(min_idle_gap_seconds=240.0)
        rig.snapshot()
        rig.clock.advance(239.9)

        assert await rig.scheduler.maybe_ping() is None
        assert rig.backend.calls == 0

        rig.clock.advance(0.2)
        assert rig.scheduler.idle_gap_seconds() >= 240.0
        task = await rig.scheduler.maybe_ping()
        assert task is not None
        await task
        assert rig.backend.calls == 1

        # Conversation activity restarts the gap (and invalidates the snapshot).
        rig.scheduler.note_conversation_activity()
        rig.clock.advance(100)
        assert await rig.scheduler.maybe_ping() is None
        assert rig.backend.calls == 1

    @pytest.mark.asyncio
    async def test_no_live_snapshot_no_ping(self) -> None:
        rig = _Rig()
        rig.clock.advance(300)
        assert await rig.scheduler.maybe_ping() is None
        assert rig.backend.calls == 0
        assert rig.accountant.attempts == 0

    @pytest.mark.asyncio
    async def test_at_most_one_ping_in_flight(self) -> None:
        block = asyncio.Event()
        rig = _Rig(block=block)
        rig.snapshot()
        rig.clock.advance(300)

        first = await rig.scheduler.maybe_ping()
        assert first is not None
        await asyncio.sleep(0)
        assert rig.scheduler.phase is KeepalivePhase.IN_FLIGHT
        assert rig.scheduler.in_flight

        assert await rig.scheduler.maybe_ping() is None
        assert rig.backend.calls == 1

        block.set()
        record = await first
        assert record.phase_reached is KeepalivePhase.IN_FLIGHT
        assert rig.scheduler.phase is KeepalivePhase.IDLE
        assert not rig.scheduler.in_flight

    @pytest.mark.asyncio
    async def test_supersession_in_the_reserved_phase_releases_unspent(self) -> None:
        rig = _Rig(block=asyncio.Event())
        rig.snapshot()
        rig.clock.advance(300)

        task = await rig.scheduler.maybe_ping()
        assert task is not None
        # The task has not run yet: the ping is still RESERVED, no request sent.
        assert rig.scheduler.phase is KeepalivePhase.RESERVED
        assert rig.backend.calls == 0

        rig.scheduler.notify_real_call_started()
        record = await task

        assert record.superseded
        assert record.phase_reached is KeepalivePhase.RESERVED
        assert rig.backend.calls == 0
        assert rig.accountant.attempts == 1
        assert rig.accountant.open_attempts == 0
        assert rig.accountant.charged_spend_usd == 0.0
        (row,) = rig.sink.records
        assert _entry(row).outcome is KeepaliveOutcome.SUPERSEDED
        assert _entry(row).retained_cost_usd == 0.0
        assert not rig.scheduler.in_flight

    @pytest.mark.asyncio
    async def test_supersession_in_the_in_flight_phase_cancels_and_retains(
        self,
    ) -> None:
        block = asyncio.Event()
        rig = _Rig(block=block)
        rig.snapshot()
        rig.clock.advance(300)

        task = await rig.scheduler.maybe_ping()
        assert task is not None
        await asyncio.sleep(0)
        assert rig.scheduler.phase is KeepalivePhase.IN_FLIGHT
        assert rig.backend.calls == 1

        rig.scheduler.notify_real_call_started()
        record = await task

        assert record.superseded
        assert record.phase_reached is KeepalivePhase.IN_FLIGHT
        assert rig.accountant.open_attempts == 0
        # The wire request may have been billed: the FULL reservation is retained.
        assert rig.accountant.charged_spend_usd > 0.0
        (row,) = rig.sink.records
        assert _entry(row).outcome is KeepaliveOutcome.CANCELLED
        assert _entry(row).retained_cost_usd > 0.0
        assert not rig.scheduler.in_flight

        # While the real call is active, no new ping — and its start
        # invalidated the snapshot, so a re-snapshot is needed afterwards.
        rig.clock.advance(300)
        assert await rig.scheduler.maybe_ping() is None
        rig.scheduler.notify_real_call_finished()
        rig.refresh()
        rig.clock.advance(300)
        block.set()
        task2 = await rig.scheduler.maybe_ping()
        assert task2 is not None
        await task2
        assert rig.backend.calls == 2

    @pytest.mark.asyncio
    async def test_the_ping_response_is_never_executed_and_never_appended(self) -> None:
        tool_calls = [
            ToolCall(
                id="call-1", index=0, function=FunctionCall(name="bash", arguments="{}")
            )
        ]
        rig = _Rig(tool_calls=tool_calls)
        rig.snapshot()
        rig.clock.advance(300)

        task = await rig.scheduler.maybe_ping()
        assert task is not None
        record = await task

        # The response DID carry a tool call and flowed through the gateway...
        assert rig.processed and rig.processed[-1].tool_calls == tool_calls
        assert rig.backend.calls == 1
        # ...the request offered no tools...
        assert rig.backend.requests[0]["tools"] is None
        # ...and the scheduler keeps NO reference to the response: the record
        # carries only the observation, and no chunk/message is stored on it.
        assert {f.name for f in fields(KeepalivePingRecord)} == {
            "attempt_ordinal",
            "phase_reached",
            "observation",
            "stop",
            "superseded",
        }
        assert not any(
            isinstance(value, (LLMChunk, LLMMessage))
            for value in vars(rig.scheduler).values()
        )
        assert record.observation is not None
        assert record.observation.stop is None

    @pytest.mark.asyncio
    async def test_the_timer_ticks_while_enabled_and_stops_on_the_cap(self) -> None:
        rig = _Rig(caps=KeepaliveCaps(max_attempts=2, max_spend_usd=1.0))
        rig.refresh()
        rig.clock.advance(300)
        ticks = 0

        async def fake_sleep(seconds: float) -> None:
            nonlocal ticks
            ticks += 1
            rig.clock.advance(seconds)
            while rig.scheduler.in_flight:
                await asyncio.sleep(0)
            if ticks >= 4:
                raise asyncio.CancelledError

        run_task = asyncio.create_task(rig.scheduler.run(fake_sleep))
        with pytest.raises(asyncio.CancelledError):
            await run_task
        await asyncio.sleep(0)

        assert rig.backend.calls == 2
        assert rig.accountant.attempts == 2
        assert rig.scheduler.stopped is KeepaliveStopReason.CAP_EXCEEDED
        assert len(rig.scheduler.records) == 2

    @pytest.mark.asyncio
    async def test_from_config_wires_the_flag_and_the_idle_floor(self) -> None:
        clock = _Clock()
        store = KeepaliveSnapshotStore(clock=clock)
        sink = _ListSink()
        accountant = KeepaliveAccountant(
            caps=KeepaliveCaps(max_attempts=100, max_spend_usd=1.0),
            model=_model(),
            provider="synthetic",
            attribution=ATTRIBUTION,
            ledger=sink,
            clock=clock,
        )
        backend = _PingBackend()
        stats = AgentStats()
        rows: list[UsageRecord] = []
        config = KeepaliveSectionConfig(
            enabled=True, max_spend_usd=1.0, min_idle_gap_seconds=600.0
        )
        scheduler = KeepaliveScheduler.from_config(
            config,
            gates=KeepaliveGates(
                is_root=True,
                owned_dependency_runs_active=lambda: True,
                min_idle_gap_seconds=240.0,
            ),
            snapshot_store=store,
            accountant=accountant,
            deployment=DEPLOYMENT,
            provider=PROVIDER,
            transport=_gateway_transport(backend, stats, rows),
            clock=clock,
        )
        assert scheduler.enabled is True
        store.snapshot(deployment=DEPLOYMENT, provider=PROVIDER, serialized=PREFIX)
        clock.advance(300)
        assert await scheduler.maybe_ping() is None
        assert backend.calls == 0
        clock.advance(300)
        task = await scheduler.maybe_ping()
        assert task is not None
        await task
        assert backend.calls == 1


class TestObserve:
    """P5 chunk 5c: the OBSERVE section — the ping's OWN reported usage."""

    def test_a_clean_full_coverage_hit_observes_without_stop(self) -> None:
        observation = observe_ping_usage(HIT_USAGE)
        assert observation.stop is None
        assert not observation.stopped
        assert observation.cached_coverage == pytest.approx(1.0)
        assert observation.usage is not None
        assert observation.usage.presence_state is UsageState.COMPLETE

    def test_a_cache_miss_stops(self) -> None:
        observation = observe_ping_usage(MISS_USAGE)
        assert observation.stop is KeepaliveStopReason.CACHE_MISS
        assert observation.cached_coverage == 0.0

    def test_missing_and_partial_usage_stop(self) -> None:
        assert observe_ping_usage(None).stop is KeepaliveStopReason.MISSING_USAGE
        assert observe_ping_usage(LLMUsage()).stop is KeepaliveStopReason.MISSING_USAGE
        assert (
            observe_ping_usage(PARTIAL_USAGE).stop is KeepaliveStopReason.PARTIAL_USAGE
        )

    def test_insufficient_cached_prefix_coverage_stops(self) -> None:
        observation = observe_ping_usage(LOW_COVERAGE_USAGE)
        assert observation.stop is KeepaliveStopReason.INSUFFICIENT_CACHED_COVERAGE
        assert observation.cached_coverage == pytest.approx(0.3)
        # A lower configured floor observes the same usage cleanly.
        assert (
            observe_ping_usage(LOW_COVERAGE_USAGE, min_cached_prefix_coverage=0.2).stop
            is None
        )

    @pytest.mark.asyncio
    async def test_a_cache_miss_stops_the_idle_cycle_once_per_cycle(self) -> None:
        rig = _Rig(usage=MISS_USAGE)
        rig.refresh()
        rig.clock.advance(300)

        task = await rig.scheduler.maybe_ping()
        assert task is not None
        record = await task

        assert record.stop is KeepaliveStopReason.CACHE_MISS
        assert record.observation is not None
        assert record.observation.stop is KeepaliveStopReason.CACHE_MISS
        assert rig.scheduler.stopped is KeepaliveStopReason.CACHE_MISS
        # The cycle is stopped: no further pings...
        rig.clock.advance(300)
        assert await rig.scheduler.maybe_ping() is None
        assert rig.backend.calls == 1
        # ...until new conversation activity opens a NEW cycle.
        rig.scheduler.note_conversation_activity()
        rig.refresh()
        rig.clock.advance(300)
        task2 = await rig.scheduler.maybe_ping()
        assert task2 is not None
        record2 = await task2
        assert record2.stop is KeepaliveStopReason.CACHE_MISS
        assert rig.backend.calls == 2
        # The miss was reconciled on the ping's own reported usage.
        assert [_entry(row).outcome for row in rig.sink.records] == [
            KeepaliveOutcome.COMPLETED,
            KeepaliveOutcome.COMPLETED,
        ]

    @pytest.mark.asyncio
    async def test_an_error_stops_the_cycle_and_retains_the_reservation(self) -> None:
        rig = _Rig(error=RuntimeError("boom"))
        rig.refresh()
        rig.clock.advance(300)

        task = await rig.scheduler.maybe_ping()
        assert task is not None
        record = await task

        assert record.stop is KeepaliveStopReason.ERROR
        assert rig.scheduler.stopped is KeepaliveStopReason.ERROR
        assert rig.accountant.open_attempts == 0
        assert rig.accountant.charged_spend_usd > 0.0
        (row,) = rig.sink.records
        assert _entry(row).outcome is KeepaliveOutcome.UNKNOWN_USAGE

    @pytest.mark.asyncio
    async def test_partial_usage_stops_and_settles_unknown_usage(self) -> None:
        rig = _Rig(usage=PARTIAL_USAGE)
        rig.refresh()
        rig.clock.advance(300)

        task = await rig.scheduler.maybe_ping()
        assert task is not None
        record = await task

        assert record.stop is KeepaliveStopReason.PARTIAL_USAGE
        (row,) = rig.sink.records
        assert _entry(row).outcome is KeepaliveOutcome.UNKNOWN_USAGE
        assert rig.accountant.charged_spend_usd > 0.0

    @pytest.mark.asyncio
    async def test_insufficient_coverage_stops_but_reconciles_completed(self) -> None:
        rig = _Rig(usage=LOW_COVERAGE_USAGE)
        rig.refresh()
        rig.clock.advance(300)

        task = await rig.scheduler.maybe_ping()
        assert task is not None
        record = await task

        assert record.stop is KeepaliveStopReason.INSUFFICIENT_CACHED_COVERAGE
        (row,) = rig.sink.records
        assert _entry(row).outcome is KeepaliveOutcome.COMPLETED
        # Reservation-minus-actual was released on the recorded usage.
        assert rig.accountant.charged_spend_usd < _entry(row).reserved_cost_usd

    @pytest.mark.asyncio
    async def test_a_clean_hit_does_not_stop_and_the_cycle_can_ping_again(self) -> None:
        rig = _Rig(usage=HIT_USAGE)
        rig.refresh()
        rig.clock.advance(300)

        first_task = await rig.scheduler.maybe_ping()
        assert first_task is not None
        first = await first_task
        assert first.stop is None
        assert rig.scheduler.stopped is None

        rig.clock.advance(300)
        second_task = await rig.scheduler.maybe_ping()
        assert second_task is not None
        second = await second_task
        assert second.stop is None
        assert rig.accountant.attempts == 2
        assert rig.backend.calls == 2
        assert [_entry(row).outcome for row in rig.sink.records] == [
            KeepaliveOutcome.COMPLETED,
            KeepaliveOutcome.COMPLETED,
        ]


class TestBudgetIsolation:
    """P5 chunk 5c: the purpose split — pings never enter the session totals
    or known_cost_total, so the STOPs at core/middleware.py:64-106 cannot
    flip; the ledger (purpose=keepalive) is the single source of truth.
    """

    def test_apply_usage_splits_keepalive_out_of_the_session_totals(self) -> None:
        stats = AgentStats()
        model = _model()
        apply_usage(
            stats,
            LLMUsage(prompt_tokens=100, completion_tokens=50, cached_tokens=40),
            time_seconds=1.0,
            model=model,
        )
        assert stats.session_total_llm_tokens == 150
        cost_after_conversation = stats.known_cost_total
        assert cost_after_conversation > 0.0

        apply_usage(
            stats,
            LLMUsage(prompt_tokens=1000, completion_tokens=10, cached_tokens=900),
            time_seconds=1.0,
            model=model,
            account_conversation=False,
            purpose=UsagePurpose.KEEPALIVE,
        )

        # Session/UI totals exclude the ping...
        assert stats.session_prompt_tokens == 100
        assert stats.session_completion_tokens == 50
        assert stats.session_cached_tokens == 40
        assert stats.session_total_llm_tokens == 150
        assert stats.known_cost_total == pytest.approx(cost_after_conversation)
        assert stats.context_tokens == 150
        assert not stats.has_unknown_cost
        # ...while the keepalive accumulators saw it.
        assert stats.keepalive_prompt_tokens == 1000
        assert stats.keepalive_completion_tokens == 10
        assert stats.keepalive_cached_tokens == 900
        assert stats.keepalive_total_llm_tokens == 1010
        assert stats.keepalive_known_cost_usd > 0.0

    def test_a_keepalive_ping_cannot_flip_has_unknown_cost(self) -> None:
        stats = AgentStats()
        unpriced = _model(input_price_known=False, output_price_known=False)
        ping = LLMUsage(prompt_tokens=100, completion_tokens=5, cached_tokens=0)
        apply_usage(
            stats,
            ping,
            time_seconds=1.0,
            model=unpriced,
            account_conversation=False,
            purpose=UsagePurpose.KEEPALIVE,
        )
        assert not stats.has_unknown_cost
        assert stats.session_total_llm_tokens == 0
        assert stats.known_cost_total == 0.0
        # Control: the same unpriced usage as an ordinary call DOES mark it.
        apply_usage(
            stats,
            ping,
            time_seconds=1.0,
            model=unpriced,
            account_conversation=False,
            purpose=UsagePurpose.CONVERSATION,
        )
        assert stats.has_unknown_cost

    def test_an_interrupted_keepalive_with_no_usage_never_marks_unknown_cost(
        self,
    ) -> None:
        stats = AgentStats()
        _apply_interrupted_usage(
            stats,
            None,
            time_seconds=1.0,
            model=_model(),
            account_conversation=False,
            purpose=UsagePurpose.KEEPALIVE,
        )
        assert not stats.has_unknown_cost
        _apply_interrupted_usage(
            stats,
            None,
            time_seconds=1.0,
            model=_model(),
            account_conversation=False,
            purpose=UsagePurpose.CONVERSATION,
        )
        assert stats.has_unknown_cost

    @pytest.mark.asyncio
    async def test_the_gateway_ping_excludes_session_totals_and_stops_cannot_flip(
        self,
    ) -> None:
        stats = AgentStats()
        backend = _AttemptBackend()
        await LLMGateway().complete(
            _inputs(UsagePurpose.CONVERSATION),
            CallResources(
                backend=backend, stats=stats, process_message=lambda message: message
            ),
        )
        assert stats.session_total_llm_tokens == 12
        cost_after_conversation = stats.known_cost_total
        assert cost_after_conversation > 0.0

        # Pin the STOP thresholds exactly at the pre-ping session totals.
        context = ConversationContext(
            messages=MessageList(), stats=stats, config=ChartreuxConfigSchema()
        )
        token_stop = TokenLimitMiddleware(max_tokens=stats.session_total_llm_tokens)
        price_stop = PriceLimitMiddleware(max_price=stats.known_cost_total)
        assert (
            await token_stop.before_turn(context)
        ).action is MiddlewareAction.CONTINUE
        assert (
            await price_stop.before_turn(context)
        ).action is MiddlewareAction.CONTINUE

        # A ping through the gateway with purpose=keepalive.
        ping_backend = _PingBackend(usage=HIT_USAGE)
        rows: list[UsageRecord] = []

        async def sink(record: UsageRecord) -> None:
            rows.append(record)

        ping_inputs = replace(
            _inputs(UsagePurpose.KEEPALIVE), account_conversation=False
        )
        result = await LLMGateway().complete(
            ping_inputs,
            CallResources(
                backend=ping_backend,
                stats=stats,
                process_message=lambda message: message,
                accounting_sink=sink,
                usage_attribution=ATTRIBUTION,
            ),
        )
        assert result.usage is not None and result.usage.cached_tokens == 100

        # Session/UI totals exclude the ping...
        assert stats.session_total_llm_tokens == 12
        assert stats.known_cost_total == pytest.approx(cost_after_conversation)
        assert stats.context_tokens == 12
        assert not stats.has_unknown_cost
        assert stats.keepalive_total_llm_tokens == 105
        # ...the STOPs cannot flip on the ping...
        assert (
            await token_stop.before_turn(context)
        ).action is MiddlewareAction.CONTINUE
        assert (
            await price_stop.before_turn(context)
        ).action is MiddlewareAction.CONTINUE
        # ...and the LEDGER carries the ping spend (ledger-authoritative): the
        # divergence between the ledger and the session totals is accepted.
        assert rows
        assert all(row.purpose is UsagePurpose.KEEPALIVE for row in rows)
        assert sum(row.known_cost_usd for row in rows) > 0.0
        assert stats.known_cost_total == pytest.approx(cost_after_conversation)

        # Control: the same usage as an ordinary conversation call DOES flip
        # both STOPs — proving the middlewares would have flipped without the
        # purpose split.
        await LLMGateway().complete(
            _inputs(UsagePurpose.CONVERSATION),
            CallResources(
                backend=_PingBackend(usage=HIT_USAGE),
                stats=stats,
                process_message=lambda message: message,
            ),
        )
        assert stats.session_total_llm_tokens == 12 + 105
        assert (await token_stop.before_turn(context)).action is MiddlewareAction.STOP
        assert (await price_stop.before_turn(context)).action is MiddlewareAction.STOP


class TestFlagOff:
    """P5 chunk 5c: FLAG OFF => zero behavior (no snapshot, no timer, no
    ledger row — asserted).
    """

    @pytest.mark.asyncio
    async def test_flag_off_means_zero_behavior(self) -> None:
        config = KeepaliveSectionConfig()
        assert config.enabled is False

        clock = _Clock()
        store = KeepaliveSnapshotStore(clock=clock)
        sink = _ListSink()
        accountant = KeepaliveAccountant(
            caps=KeepaliveCaps(max_attempts=100, max_spend_usd=1.0),
            model=_model(),
            provider="synthetic",
            attribution=ATTRIBUTION,
            ledger=sink,
            clock=clock,
        )
        backend = _PingBackend()
        stats = AgentStats()
        rows: list[UsageRecord] = []
        scheduler = KeepaliveScheduler.from_config(
            config,
            gates=KeepaliveGates(
                is_root=True,
                owned_dependency_runs_active=lambda: True,
                min_idle_gap_seconds=240.0,
            ),
            snapshot_store=store,
            accountant=accountant,
            deployment=DEPLOYMENT,
            provider=PROVIDER,
            transport=_gateway_transport(backend, stats, rows),
            clock=clock,
        )
        assert scheduler.enabled is False

        # No snapshot is taken.
        assert scheduler.refresh_snapshot(serialized=PREFIX) is None
        assert store.generation == 0
        assert store.current() is None

        # No ping, even with every other gate green.
        clock.advance(10_000)
        assert await scheduler.maybe_ping() is None
        assert backend.calls == 0
        assert accountant.attempts == 0
        assert sink.records == []
        assert rows == []
        assert scheduler.records == ()

        # No timer: run() returns immediately without ever sleeping.
        slept: list[float] = []

        async def spy_sleep(seconds: float) -> None:
            slept.append(seconds)

        await scheduler.run(spy_sleep)
        assert slept == []

        # The hooks are harmless no-ops: still no ledger row, no records.
        scheduler.notify_real_call_started()
        scheduler.notify_real_call_finished()
        scheduler.note_conversation_activity()
        assert sink.records == []
        assert rows == []
        assert scheduler.records == ()
