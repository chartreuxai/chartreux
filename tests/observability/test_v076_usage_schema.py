"""P3 additive ledger schema: KEEPALIVE purpose and request-shape telemetry."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
import json
from pathlib import Path

from pydantic import ValidationError
import pytest

from chartreux.core.usage import (
    UsageAttribution,
    UsageOutcome,
    UsagePrices,
    UsagePurpose,
    UsageReader,
    UsageRecord,
    UsageState,
)

TELEMETRY_FIELDS = (
    "system_prompt_bytes",
    "tool_def_bytes",
    "prefix_stability_hash",
    "gap_since_previous_call_ms",
    "concurrency_snapshot",
    "toolset_prompt_fingerprint",
    "declared_max_tokens",
)

TELEMETRY_VALUES: dict[str, object] = {
    "system_prompt_bytes": 12_034,
    "tool_def_bytes": 24_096,
    "prefix_stability_hash": "sha256:5f0c4e",
    "gap_since_previous_call_ms": 4_200,
    "concurrency_snapshot": 3,
    "toolset_prompt_fingerprint": "sha256:ab41d2",
    "declared_max_tokens": 8_192,
}


def make_record(
    record_id: str = "record", overrides: Mapping[str, object] | None = None
) -> UsageRecord:
    payload: dict[str, object] = {
        "record_id": record_id,
        "occurred_at": datetime(2026, 3, 1, tzinfo=UTC),
        "root_session_id": "root",
        "session_id": "child",
        "parent_session_id": "root",
        "agent_role": "subagent",
        "agent_profile": "implementor",
        "purpose": UsagePurpose.CONVERSATION,
        "model": "base",
        "provider": "provider",
        "wire_name": "wire",
        "project_key": "project",
        "outcome": UsageOutcome.COMPLETED,
        "usage_state": UsageState.COMPLETE,
        "input_tokens": 100,
        "output_tokens": 50,
        "cached_input_tokens": 20,
        "prices_usd_per_million": UsagePrices(input=2, output=4, cached_input=1),
        "known_cost_usd": 0.00038,
        "has_unknown_cost": False,
        **(overrides or {}),
    }
    return UsageRecord.model_validate(payload)


def test_keepalive_purpose_is_an_extension_not_a_renumbering() -> None:
    assert UsagePurpose.KEEPALIVE == "keepalive"
    assert UsagePurpose("keepalive") is UsagePurpose.KEEPALIVE
    assert {purpose.value for purpose in UsagePurpose} == {
        "conversation",
        "compaction",
        "title",
        "worktree-naming",
        "keepalive",
    }


def test_keepalive_purpose_serializes_on_the_ledger_row() -> None:
    record = make_record(overrides={"purpose": UsagePurpose.KEEPALIVE})
    payload = json.loads(record.model_dump_json())
    assert payload["purpose"] == "keepalive"
    assert UsageRecord.model_validate_json(record.model_dump_json()) == record


def test_every_telemetry_field_round_trips_through_the_ledger_row() -> None:
    record = make_record(overrides=TELEMETRY_VALUES)
    payload = json.loads(record.model_dump_json())
    assert set(TELEMETRY_FIELDS) <= set(payload)
    for field, value in TELEMETRY_VALUES.items():
        assert payload[field] == value
    assert UsageRecord.model_validate_json(record.model_dump_json()) == record


def test_telemetry_round_trips_through_the_usage_reader(tmp_path: Path) -> None:
    record = make_record(overrides=TELEMETRY_VALUES)
    ledger = tmp_path / "root" / "usage.jsonl"
    ledger.parent.mkdir(parents=True)
    ledger.write_bytes((record.model_dump_json() + "\n").encode())

    snapshot = UsageReader(tmp_path).reconcile()

    assert snapshot.warnings == ()
    assert snapshot.records == (record,)
    restored = snapshot.records[0]
    for field, value in TELEMETRY_VALUES.items():
        assert getattr(restored, field) == value


@pytest.mark.parametrize("schema_version", [1, 2])
def test_pre_existing_rows_without_telemetry_still_parse(
    tmp_path: Path, schema_version: int
) -> None:
    payload = json.loads(make_record().model_dump_json())
    for field in TELEMETRY_FIELDS:
        payload.pop(field, None)
    assert not set(TELEMETRY_FIELDS) & set(payload)
    payload["schema_version"] = schema_version
    ledger = tmp_path / "root" / "usage.jsonl"
    ledger.parent.mkdir(parents=True)
    ledger.write_bytes((json.dumps(payload) + "\n").encode())

    snapshot = UsageReader(tmp_path).reconcile()

    assert snapshot.warnings == ()
    assert len(snapshot.records) == 1
    restored = snapshot.records[0]
    assert all(getattr(restored, field) is None for field in TELEMETRY_FIELDS)


def test_unset_telemetry_defaults_to_none_and_is_omitted() -> None:
    record = make_record()
    assert all(getattr(record, field) is None for field in TELEMETRY_FIELDS)
    payload = json.loads(record.model_dump_json())
    assert not set(TELEMETRY_FIELDS) & set(payload)


def test_explicit_none_telemetry_is_tolerated() -> None:
    explicit = make_record(overrides=dict.fromkeys(TELEMETRY_FIELDS))
    assert explicit == make_record()
    payload = json.loads(explicit.model_dump_json())
    assert not set(TELEMETRY_FIELDS) & set(payload)


def test_telemetry_names_are_attribution_fields() -> None:
    assert set(TELEMETRY_FIELDS) <= set(UsageAttribution.model_fields)


@pytest.mark.parametrize(
    "field",
    [
        "system_prompt_bytes",
        "tool_def_bytes",
        "gap_since_previous_call_ms",
        "concurrency_snapshot",
        "declared_max_tokens",
    ],
)
@pytest.mark.parametrize("bad", [-1, 1.5, True, "3"])
def test_count_telemetry_is_a_strict_nonnegative_integer(
    field: str, bad: object
) -> None:
    with pytest.raises(ValidationError):
        make_record(overrides={field: bad})
