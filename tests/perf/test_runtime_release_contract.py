"""WP8 contract node: the runtime measurement release contract.

This node holds ONE hard assertion about the NEW runtime-metric threshold table
(``tests/perf/runtime_thresholds.json``): the activation commit populated it, so
every registry metric must now carry a disposition there — ACTIVE when its
committed baseline set is decision-ready with a finite positive T, otherwise
DORMANT with the reason the pinned reason order recorded (plan §7). That row
enumeration is the ATOMIC FLIP TARGET: the commit that populated the table
amended the pre-activation "the table is empty" assertion into it, so a registry
metric carrying no table row fails. Frozen ``tests/perf`` ceiling assertions are
untouched by this module.

The node is the pinned WP9-safe home: it reads COMMITTED artifacts only (the
pinned registry, the pinned determinants document, the threshold table),
enumerates the 17 gated rows, and executes no measurement, so it stays inside
the in-suite contract-node envelope. It makes no network call and spawns no
process, so it carries no egress path to close. WP9 runs it with
``CHARTREUX_PERF_REPORT`` unset; when a report path is configured the blob goes
through the WP1 path guard.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

import pytest

from tests.perf._runtime_metrics import (
    REPORT_ENV,
    RUN_DIR_ENV,
    PerfPathError,
    repository_root,
)
from tests.perf._runtime_report import (
    CLOSURE_ELIGIBLE_COMPLETE,
    GATED_CAMPAIGN_IDS,
    METRIC_VERSION,
    SCHEMA_ID,
    validate_report,
)
from tests.perf._runtime_thresholds import (
    DORMANCY_REASONS,
    STATE_ACTIVE,
    STATE_DORMANT,
    THRESHOLDS_SCHEMA_ID,
)
from tests.perf._runtime_usage import (
    ACTIVATION_ACTIVATED,
    BLOB_LINE_PREFIX,
    BLOB_SECTION,
    DETERMINANTS_FILENAME,
    REGISTRY_FILENAME,
    TABLE_ABSENT,
    TABLE_EMPTY,
    TABLE_POPULATED,
    ReleaseArtifacts,
    ReleaseRow,
    ReleaseUsageError,
    ThresholdTableState,
    build_release_blob,
    determinant_rows,
    emit_release_blob,
    load_release_artifacts,
    read_threshold_table,
    release_rows,
)

pytestmark = [
    pytest.mark.perf,
    pytest.mark.skipif(
        sys.platform != "linux",
        reason="the runtime perf release contract covers Linux-only machinery",
    ),
]

# Plan §1 in-suite envelope: a contract node plans for a 1s budget (measured
# target <= 0.5s) inside a 2s hard window, and its timeout marker is 2x budget.
_CONTRACT_BUDGET_S = 1.0
_CONTRACT_WINDOW_S = 2.0
_CONTRACT_TIMEOUT_S = 2 * _CONTRACT_BUDGET_S

#: The closed registry's size (plan §4): 17 gated metric-cell campaign IDs.
_RELEASE_ROW_COUNT = 17

#: Plan §7 pins of the P-Artifacts commit. A mismatch is STOP + ESCALATE:
#: re-emitting, re-formatting or editing the JSON to force a match is never
#: authorized without a user-accepted plan amendment.
_REGISTRY_SHA256 = "7b336621b946b5760fb13ad673c70d4b4c14bf90d6325543ca9e530a35086de5"
_DETERMINANTS_SHA256 = (
    "a1847c336f05c83cd3f50cfe3cc756ce2a4b453f79eb9a4a0ddc02a60a7cf21f"
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_pinned_artifacts(artifacts: ReleaseArtifacts) -> None:
    """The two pinned artifacts are byte-exact against the plan's §7 pins."""
    registry_digest = _digest(artifacts.perf_dir / REGISTRY_FILENAME)
    assert registry_digest == _REGISTRY_SHA256, (
        f"the pinned registry drifted from its plan §7 pin ({registry_digest}); "
        "STOP + ESCALATE — never re-emit or re-format it to force a match"
    )
    determinants_digest = _digest(artifacts.perf_dir / DETERMINANTS_FILENAME)
    assert determinants_digest == _DETERMINANTS_SHA256, (
        "the pinned determinants document drifted from its plan §7 pin "
        f"({determinants_digest}); STOP + ESCALATE — never re-emit or "
        "re-format it to force a match"
    )


def _assert_row_enumeration(artifacts: ReleaseArtifacts) -> tuple[ReleaseRow, ...]:
    """Enumerate the 17 registry rows and bind them to the determinants rows."""
    rows = release_rows(artifacts.registry)
    assert len(rows) == _RELEASE_ROW_COUNT, (
        f"the registry carries {len(rows)} rows; the closed registry pins "
        f"{_RELEASE_ROW_COUNT} (a changed registry is a replan, never a "
        "silent edit of this bound)"
    )
    assert tuple(row.campaign_id for row in rows) == GATED_CAMPAIGN_IDS

    determinants = determinant_rows(artifacts.determinants)
    assert set(determinants) == {row.campaign_id for row in rows}, (
        "every gated registry row needs its determinants row before any "
        "activation disposition can be constructed"
    )
    for row in rows:
        entry = determinants[row.campaign_id]
        assert entry["campaign_id"] == row.campaign_id
        assert entry["metric"] == row.metric
        assert entry["cell"] == row.cell
        assert entry["producer_wp"] == row.producer_wp
        assert entry.get("unit", row.unit) == row.unit
    return rows


def _assert_threshold_dispositions(
    artifacts: ReleaseArtifacts, rows: tuple[ReleaseRow, ...]
) -> ThresholdTableState:
    """THE ACTIVATION FLIP TARGET (plan §7): every registry metric is disposed.

    The activation commit populated ``tests/perf/runtime_thresholds.json``, so the
    table is the authority on each row: ACTIVE when the committed baseline set is
    decision-ready with a finite positive T, else DORMANT with the reason the
    pinned reason order recorded. Coverage is total — a registry metric with no
    table row fails, and so does a table row the registry does not carry.
    """
    table = read_threshold_table(artifacts.threshold_path)
    assert table.state == TABLE_POPULATED, (
        "the runtime-metric threshold table must carry one row per registry "
        f"metric after activation (plan §7): {table.path} is {table.state!r} "
        f"({table.detail})"
    )
    document = json.loads(table.path.read_text(encoding="utf-8"))
    assert document.get("schema") == THRESHOLDS_SCHEMA_ID, document.get("schema")
    assert document.get("dormancy_reasons") == list(DORMANCY_REASONS)
    inputs = document.get("inputs")
    assert isinstance(inputs, Mapping), inputs
    assert inputs.get("registry_sha256") == _REGISTRY_SHA256
    assert inputs.get("determinants_sha256") == _DETERMINANTS_SHA256

    records = document.get("rows")
    assert isinstance(records, list) and records, "the table carries no rows"
    by_id: dict[str, Mapping[str, Any]] = {}
    for record in records:
        assert isinstance(record, Mapping), record
        campaign_id = record.get("campaign_id")
        assert campaign_id not in by_id, f"duplicate threshold row: {campaign_id}"
        by_id[campaign_id] = record

    registry_ids = {row.campaign_id for row in rows}
    missing = sorted(registry_ids - set(by_id))
    assert not missing, (
        f"{missing} carry no threshold row: after activation every registry "
        "metric is ACTIVE-eligible or DORMANT-with-reason, never absent (plan §7)"
    )
    assert set(by_id) == registry_ids, sorted(set(by_id) - registry_ids)
    assert document.get("row_count") == len(records) == _RELEASE_ROW_COUNT

    for row in rows:
        record = by_id[row.campaign_id]
        assert record.get("metric") == row.metric, row.campaign_id
        assert record.get("cell") == row.cell, row.campaign_id
        assert record.get("unit") == row.unit, row.campaign_id
        # Activation never evaluates a measured value (Appendix B active_test).
        assert record.get("measured_value") is None, row.campaign_id
        state = record.get("state")
        reason = record.get("reason")
        if state == STATE_ACTIVE:
            threshold = record.get("threshold")
            assert isinstance(threshold, (int, float)) and math.isfinite(threshold), (
                row.campaign_id,
                threshold,
            )
            assert threshold > 0, (row.campaign_id, threshold)
            assert reason is None, (row.campaign_id, reason)
            assert record.get("decision_ready") is True, row.campaign_id
            assert record.get("closure_status") == CLOSURE_ELIGIBLE_COMPLETE
        else:
            assert state == STATE_DORMANT, (row.campaign_id, state)
            assert reason in DORMANCY_REASONS, (row.campaign_id, reason)

    active = sum(1 for record in by_id.values() if record.get("state") == STATE_ACTIVE)
    assert document.get("active_count") == active
    assert document.get("dormant_count") == len(by_id) - active
    return table


def _assert_table_state_semantics(tmp_path: Path) -> None:
    """Pin the state vocabulary the row assertion distinguishes.

    The contract asserts coverage over a POPULATED table, while "absent", "blank"
    and "a JSON document carrying no entries" must stay distinct from it, so a
    table that lost its rows can never read as populated. Unparseable content is
    an error, never a silent empty.
    """
    absent = read_threshold_table(tmp_path / "not-committed-yet.json")
    assert (absent.state, absent.is_empty) == (TABLE_ABSENT, True)

    for name, content in (
        ("blank.json", "\n \n"),
        ("empty-object.json", "{}"),
        ("empty-array.json", "[]"),
    ):
        path = tmp_path / name
        path.write_text(content, encoding="utf-8")
        state = read_threshold_table(path)
        assert (state.state, state.is_empty) == (TABLE_EMPTY, True), path

    populated_path = tmp_path / "populated.json"
    populated_path.write_text(
        json.dumps({"entries": [{"campaign_id": "A1"}]}), encoding="utf-8"
    )
    populated = read_threshold_table(populated_path)
    assert (populated.state, populated.is_empty) == (TABLE_POPULATED, False)

    malformed_path = tmp_path / "malformed.json"
    malformed_path.write_text("{", encoding="utf-8")
    with pytest.raises(ReleaseUsageError) as malformed:
        read_threshold_table(malformed_path)
    assert malformed.value.reason == "threshold_table_malformed"

    directory = tmp_path / "table-is-a-directory"
    directory.mkdir()
    with pytest.raises(ReleaseUsageError) as as_directory:
        read_threshold_table(directory)
    assert as_directory.value.reason == "threshold_table_not_file"


def _assert_blob(
    artifacts: ReleaseArtifacts,
    table: ThresholdTableState,
    rows: tuple[ReleaseRow, ...],
) -> dict[str, object]:
    """The contract's blob is a WP6 report carrying the release section."""
    blob = build_release_blob(
        artifacts=artifacts,
        threshold_table=table,
        activation_state=ACTIVATION_ACTIVATED,
    )
    validate_report(blob)
    assert blob["schema"] == SCHEMA_ID
    assert blob["metric_version"] == METRIC_VERSION
    assert blob["runs"] == []
    section = blob[BLOB_SECTION]
    campaign_ids = [row.campaign_id for row in rows]
    assert section["activation_state"] == ACTIVATION_ACTIVATED
    assert section["registry"] == {
        "schema": "chartreux.runtime_metric_registry.v1",
        "entry_count": _RELEASE_ROW_COUNT,
        "campaign_ids": campaign_ids,
    }
    assert section["determinants"]["row_count"] == _RELEASE_ROW_COUNT
    assert [row["campaign_id"] for row in section["rows"]] == campaign_ids
    assert section["threshold_table"]["state"] == table.state
    assert section["threshold_table"]["empty"] is False

    line = emit_release_blob(blob)
    assert line.startswith(BLOB_LINE_PREFIX)
    assert json.loads(line[len(BLOB_LINE_PREFIX) :]) == blob

    # A populated table under the pre-activation state is a drift, not a blob.
    synthetic = ThresholdTableState(
        path=artifacts.threshold_path, state=TABLE_POPULATED, detail="synthetic"
    )
    with pytest.raises(ReleaseUsageError) as populated:
        build_release_blob(artifacts=artifacts, threshold_table=synthetic)
    assert populated.value.reason == "threshold_table_populated_before_activation"
    # ... which the activation state is what admits.
    activated = build_release_blob(
        artifacts=artifacts,
        threshold_table=synthetic,
        activation_state=ACTIVATION_ACTIVATED,
    )
    assert activated[BLOB_SECTION]["activation_state"] == ACTIVATION_ACTIVATED
    return blob


def _assert_guarded_emission(blob: dict[str, object], tmp_path: Path) -> None:
    """The report file is written only through the WP1 path guard."""
    scratchpad = tmp_path / "chartreux-scratchpad-wp8"
    run_dir = scratchpad / "run"
    run_dir.mkdir(parents=True)
    report = run_dir / "report.jsonl"
    line = emit_release_blob(
        blob, environ={RUN_DIR_ENV: str(run_dir), REPORT_ENV: str(report)}
    )
    assert report.read_text(encoding="utf-8") == f"{line}\n"

    escaping_report = repository_root() / "perf-release-contract-never-written.jsonl"
    with pytest.raises(PerfPathError) as escaping:
        emit_release_blob(
            blob, environ={RUN_DIR_ENV: str(run_dir), REPORT_ENV: str(escaping_report)}
        )
    assert escaping.value.reason == "report_under_repo_root"
    assert not escaping_report.exists()

    with pytest.raises(PerfPathError) as orphaned:
        emit_release_blob(blob, environ={REPORT_ENV: str(report)})
    assert orphaned.value.reason == "report_without_run_dir"


@pytest.mark.timeout(_CONTRACT_TIMEOUT_S)
def test_measurement_release_contract(tmp_path: Path) -> None:
    """The release contract holds: pinned inputs, 17 rows, every row disposed."""
    started = perf_counter()
    artifacts = load_release_artifacts()
    _assert_pinned_artifacts(artifacts)
    rows = _assert_row_enumeration(artifacts)
    table = _assert_threshold_dispositions(artifacts, rows)
    _assert_table_state_semantics(tmp_path)
    blob = _assert_blob(artifacts, table, rows)
    _assert_guarded_emission(blob, tmp_path)
    assert perf_counter() - started <= _CONTRACT_WINDOW_S
