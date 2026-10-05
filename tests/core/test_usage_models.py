from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
import json

from pydantic import ValidationError
import pytest

from chartreux.core.usage import (
    AccountingSink,
    CallAccountingOutcome,
    CoverageWarning,
    CoverageWarningCode,
    SnapshotState,
    UsageAggregateSnapshot,
    UsageAttribution,
    UsageComponentSnapshot,
    UsageModelSnapshot,
    UsageOutcome,
    UsagePrices,
    UsagePurpose,
    UsageRecord,
    UsageState,
    UsageTokens,
)


def make_record() -> UsageRecord:
    return UsageRecord(
        record_id="record",
        occurred_at=datetime(2026, 3, 1, tzinfo=UTC),
        root_session_id="root",
        session_id="child",
        parent_session_id="root",
        agent_role="subagent",
        agent_profile="implementor",
        purpose=UsagePurpose.CONVERSATION,
        model="base",
        provider="provider",
        wire_name="wire",
        project_key="project",
        outcome=UsageOutcome.COMPLETED,
        usage_state=UsageState.COMPLETE,
        input_tokens=100,
        output_tokens=50,
        cached_input_tokens=20,
        prices_usd_per_million=UsagePrices(input=2, output=4, cached_input=1),
        known_cost_usd=0.00038,
        has_unknown_cost=False,
    )


def test_record_round_trip_has_only_approved_fields() -> None:
    record = make_record()
    serialized = record.model_dump_json()
    payload = json.loads(serialized)
    assert set(payload) == {
        "schema_version",
        "record_id",
        "occurred_at",
        "root_session_id",
        "session_id",
        "parent_session_id",
        "agent_role",
        "agent_profile",
        "purpose",
        "model",
        "provider",
        "wire_name",
        "project_key",
        "outcome",
        "usage_state",
        "input_tokens",
        "output_tokens",
        "cached_input_tokens",
        "prices_usd_per_million",
        "known_cost_usd",
        "has_unknown_cost",
        "currency",
    }
    assert set(payload["prices_usd_per_million"]) == {"input", "output", "cached_input"}
    assert payload["occurred_at"].endswith("Z")
    assert payload["schema_version"] == 1
    assert payload["currency"] == "USD"
    assert UsageRecord.model_validate_json(serialized) == record


@pytest.mark.parametrize(
    "field",
    [
        "prompt",
        "prompts",
        "messages",
        "headers",
        "metadata",
        "exception",
        "exception_text",
    ],
)
def test_content_fields_are_rejected_not_serialized(field: str) -> None:
    payload = make_record().model_dump()
    payload[field] = "private content"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        UsageRecord.model_validate(payload)
    assert "private content" not in make_record().model_dump_json()


@pytest.mark.parametrize(
    "timestamp",
    [datetime(2026, 3, 1), datetime(2026, 3, 1, tzinfo=timezone(timedelta(hours=1)))],
)
def test_non_utc_timestamps_are_rejected(timestamp: datetime) -> None:
    payload = make_record().model_dump()
    payload["occurred_at"] = timestamp
    with pytest.raises(ValidationError, match="timezone-aware UTC"):
        UsageRecord.model_validate(payload)


@pytest.mark.parametrize("field,value", [("schema_version", 2), ("currency", "EUR")])
def test_record_version_and_currency_are_fixed(field: str, value: object) -> None:
    payload = make_record().model_dump()
    payload[field] = value
    with pytest.raises(ValidationError):
        UsageRecord.model_validate(payload)


@pytest.mark.parametrize("bad", [-1, float("inf"), float("-inf"), float("nan")])
def test_prices_and_costs_are_finite_nonnegative(bad: float) -> None:
    for field in ("input", "output", "cached_input"):
        with pytest.raises(ValidationError):
            UsagePrices.model_validate({field: bad})
    payload = make_record().model_dump()
    payload["known_cost_usd"] = bad
    with pytest.raises(ValidationError):
        UsageRecord.model_validate(payload)
    with pytest.raises(ValidationError):
        UsageComponentSnapshot(known_cost_usd=bad)
    with pytest.raises(ValidationError):
        UsageAggregateSnapshot(known_cost_usd=bad)


@pytest.mark.parametrize(
    "field", ["input_tokens", "output_tokens", "cached_input_tokens"]
)
@pytest.mark.parametrize("bad", [-1, 1.5, True, "3"])
def test_counts_are_nullable_nonnegative_integers(field: str, bad: object) -> None:
    with pytest.raises(ValidationError):
        UsageTokens.model_validate({field: bad})


def test_presence_distinguishes_missing_partial_and_reported_zero() -> None:
    assert UsageTokens().presence_state == UsageState.MISSING
    assert UsageTokens(output_tokens=0).presence_state == UsageState.PARTIAL
    assert (
        UsageTokens(
            input_tokens=0, output_tokens=0, cached_input_tokens=0
        ).presence_state
        == UsageState.COMPLETE
    )
    for state, counts in (
        (UsageState.MISSING, {}),
        (UsageState.PARTIAL, {"output_tokens": 0}),
        (
            UsageState.COMPLETE,
            {"input_tokens": 0, "output_tokens": 0, "cached_input_tokens": 0},
        ),
    ):
        outcome = CallAccountingOutcome.model_validate({
            "outcome": "interrupted",
            "usage_state": state,
            **counts,
        })
        assert outcome.usage_state == state


@pytest.mark.parametrize(
    "payload",
    [
        {"usage_state": "missing", "output_tokens": 0},
        {"usage_state": "partial"},
        {"usage_state": "complete", "output_tokens": 0},
    ],
)
def test_usage_state_must_agree_with_reported_presence(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        CallAccountingOutcome.model_validate({"outcome": "failed", **payload})


def test_incomplete_stream_can_have_all_counts_reported() -> None:
    outcome = CallAccountingOutcome(
        outcome=UsageOutcome.INTERRUPTED,
        usage_state=UsageState.PARTIAL,
        input_tokens=100,
        output_tokens=20,
        cached_input_tokens=0,
    )
    assert outcome.presence_state == UsageState.COMPLETE
    assert outcome.usage_state == UsageState.PARTIAL


def test_records_preserve_missing_and_partial_counts_as_null() -> None:
    payload = make_record().model_dump()
    payload.update(
        outcome="failed",
        usage_state="missing",
        input_tokens=None,
        output_tokens=None,
        cached_input_tokens=None,
        known_cost_usd=0,
        has_unknown_cost=True,
    )
    missing = UsageRecord.model_validate(payload)
    assert json.loads(missing.model_dump_json())["input_tokens"] is None
    payload.update(usage_state="partial", output_tokens=0)
    partial = UsageRecord.model_validate(payload)
    assert partial.output_tokens == 0
    assert partial.input_tokens is None


def test_nested_contracts_are_immutable() -> None:
    record = make_record()
    attribution = UsageAttribution.model_validate({
        field: getattr(record, field) for field in UsageAttribution.model_fields
    })
    outcome = CallAccountingOutcome(
        outcome=UsageOutcome.FAILED, usage_state=UsageState.MISSING
    )
    warning = CoverageWarning(code=CoverageWarningCode.WRITE_FAILED)
    component = UsageComponentSnapshot()
    aggregate = UsageAggregateSnapshot(warnings=(warning,))
    row = UsageModelSnapshot(model="base", provider="provider", wire_name="wire")
    for instance, field, value in (
        (record, "known_cost_usd", 10),
        (record.prices_usd_per_million, "input", 10),
        (attribution, "session_id", "different"),
        (outcome, "output_tokens", 10),
        (warning, "record_id", "different"),
        (component, "tokens", 10),
        (aggregate, "warnings", ()),
        (row, "model", "different"),
    ):
        with pytest.raises(ValidationError, match="frozen"):
            setattr(instance, field, value)


def test_snapshot_states_remain_distinct_through_serialization() -> None:
    snapshots = [
        UsageAggregateSnapshot(state=SnapshotState.LOADING),
        UsageAggregateSnapshot(state=SnapshotState.UNAVAILABLE),
        UsageAggregateSnapshot(),
        UsageAggregateSnapshot(request_count=1, has_unknown_cost=True),
        UsageAggregateSnapshot(
            request_count=2,
            known_cost_usd=1.5,
            has_known_cost=True,
            has_unknown_cost=True,
        ),
        UsageAggregateSnapshot(request_count=1, has_known_cost=True),
    ]
    serialized = [snapshot.model_dump_json() for snapshot in snapshots]
    assert len(set(serialized)) == 6
    for snapshot, payload in zip(snapshots, serialized, strict=True):
        assert UsageAggregateSnapshot.model_validate_json(payload) == snapshot
    row = UsageModelSnapshot(
        model="base",
        provider="provider",
        wire_name="wire",
        request_count=1,
        has_unknown_cost=True,
        input=UsageComponentSnapshot(tokens=100, has_unknown_cost=True),
    )
    assert row.has_unknown_cost
    assert not row.has_known_cost
    assert row.input.has_unknown_cost


def test_coverage_warning_is_structured_and_independent_of_record_cost() -> None:
    snapshot = UsageAggregateSnapshot(
        warnings=(
            CoverageWarning(
                code=CoverageWarningCode.UNREADABLE, root_session_id="root"
            ),
        )
    )
    assert snapshot.request_count == 0
    assert not snapshot.has_unknown_cost
    assert snapshot.warnings
    with pytest.raises(ValidationError):
        CoverageWarning.model_validate({"code": "write-failed", "exception": "private"})


@pytest.mark.asyncio
async def test_async_accounting_sink_contract() -> None:
    records: list[UsageRecord] = []

    async def append(record: UsageRecord, /) -> None:
        records.append(record)

    sink: AccountingSink = append
    record = make_record()
    await sink(record)
    assert records == [record]
