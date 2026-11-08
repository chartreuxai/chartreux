"""Usage and reporting helpers for the WP8 runtime release contract.

The release contract asserts against committed artifacts only, so nothing here
measures: these helpers load the pinned registry and determinants, enumerate the
gated rows, read the state of the runtime-metric threshold table, and emit one
PERF-BLOB documenting the contract.

Blob emission is WP6-shaped — the document is a ``chartreux.runtime_perf.v1``
report carrying no runs (a contract evaluation produces none) whose release
section arrives through the schema's documented additive-section mechanism.
Report-file emission goes through the WP1 path guard, so a
``CHARTREUX_PERF_REPORT`` that escapes the worker boundary fails closed instead
of writing beside the checkout.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from tests.perf._runtime_metrics import resolve_perf_paths
from tests.perf._runtime_report import (
    DETERMINANTS_SCHEMA_ID,
    build_report,
    validate_report,
)

#: The registry document's own schema id; its entries are the gated campaign IDs.
REGISTRY_SCHEMA_ID = "chartreux.runtime_metric_registry.v1"

RELEASE_SCENARIO = "runtime_release_contract"
BLOB_SECTION = "release_contract"
BLOB_LINE_PREFIX = f"PERF-BLOB:{RELEASE_SCENARIO}:"

REGISTRY_FILENAME = "runtime_metric_registry.json"
DETERMINANTS_FILENAME = "runtime_metric_determinants.json"
THRESHOLDS_FILENAME = "runtime_thresholds.json"
PERF_DIR_NAME = Path("tests") / "perf"

#: Threshold-table states. The plan's own wording makes an absent table empty
#: too: the file is created by the activation event, so "not committed yet" and
#: "committed empty" are the same pre-activation state.
TABLE_ABSENT = "absent"
TABLE_EMPTY = "empty"
TABLE_POPULATED = "populated"

ACTIVATION_PRE = "pre_activation"
ACTIVATION_ACTIVATED = "activated"
ACTIVATION_STATES: tuple[str, ...] = (ACTIVATION_PRE, ACTIVATION_ACTIVATED)

_REGISTRY_FIELDS: tuple[str, ...] = ("metric", "unit", "cell", "producer_wp")
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class ReleaseUsageError(ValueError):
    """A release-contract input is unreadable or violates its committed shape."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"release contract rejected ({reason}): {detail}")
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True, slots=True)
class ThresholdTableState:
    """The runtime-metric threshold table as the contract finds it on disk."""

    path: Path
    state: str
    detail: str

    @property
    def is_empty(self) -> bool:
        """True while no threshold row is committed (absent counts as empty)."""
        return self.state != TABLE_POPULATED


@dataclass(frozen=True, slots=True)
class ReleaseRow:
    """One registry entry: the identity every release assertion enumerates."""

    campaign_id: str
    metric: str
    unit: str
    cell: str
    producer_wp: str

    def as_record(self) -> dict[str, str]:
        """Return the row in blob form."""
        return {
            "campaign_id": self.campaign_id,
            "metric": self.metric,
            "unit": self.unit,
            "cell": self.cell,
            "producer_wp": self.producer_wp,
        }


@dataclass(frozen=True, slots=True)
class ReleaseArtifacts:
    """The three committed artifacts a release-contract evaluation reads."""

    repo_root: Path
    registry: Mapping[str, Any]
    determinants: Mapping[str, Any]

    @property
    def perf_dir(self) -> Path:
        """Return the directory the artifacts are committed in."""
        return self.repo_root / PERF_DIR_NAME

    @property
    def threshold_path(self) -> Path:
        """Return the pinned location of the runtime-metric threshold table."""
        return self.perf_dir / THRESHOLDS_FILENAME


def repository_root() -> Path:
    """Return the repository root that contains this module."""
    return _REPOSITORY_ROOT


def threshold_table_path(repo_root: Path | None = None) -> Path:
    """Return the pinned threshold-table path for one repository root."""
    root = _REPOSITORY_ROOT if repo_root is None else Path(repo_root)
    return root / PERF_DIR_NAME / THRESHOLDS_FILENAME


def _is_empty_document(document: Any) -> bool:
    """Return whether a parsed table carries no entries of any shape.

    Emptiness is the absence of entries, not a particular schema: an empty
    mapping, an empty sequence, or blank text is empty; every other parsed
    document (including any scalar) carries content.
    """
    if isinstance(document, Mapping):
        return not document
    if isinstance(document, (list, tuple)):
        return not document
    return False


def read_threshold_table(path: Path) -> ThresholdTableState:
    """Return the threshold table's state, failing closed on unusable input.

    An absent table and an empty file are the pre-activation state the plan
    pins; a table carrying any content is ``populated``. Unreadable or
    malformed content is an error rather than a silent "empty" — the contract
    cannot assert emptiness about input it could not parse.
    """
    table_path = Path(path)
    if not table_path.exists():
        return ThresholdTableState(
            table_path, TABLE_ABSENT, "the table is created by the activation event"
        )
    if table_path.is_dir():
        raise ReleaseUsageError(
            "threshold_table_not_file", f"{table_path} is a directory"
        )
    try:
        text = table_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ReleaseUsageError(
            "threshold_table_unreadable", f"{table_path}: {error}"
        ) from error
    if not text.strip():
        return ThresholdTableState(table_path, TABLE_EMPTY, "the file is empty")
    try:
        document = json.loads(text)
    except json.JSONDecodeError as error:
        raise ReleaseUsageError(
            "threshold_table_malformed", f"{table_path}: {error}"
        ) from error
    if _is_empty_document(document):
        return ThresholdTableState(table_path, TABLE_EMPTY, "no entries are carried")
    return ThresholdTableState(table_path, TABLE_POPULATED, "threshold rows present")


def _read_object(path: Path) -> Mapping[str, Any]:
    """Read one committed JSON artifact as an object, failing closed."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ReleaseUsageError("artifact_unreadable", f"{path}: {error}") from error
    try:
        document: Any = json.loads(text)
    except json.JSONDecodeError as error:
        raise ReleaseUsageError("artifact_malformed", f"{path}: {error}") from error
    if not isinstance(document, Mapping):
        raise ReleaseUsageError("artifact_not_object", f"{path} is not an object")
    return document


def load_release_artifacts(repo_root: Path | None = None) -> ReleaseArtifacts:
    """Load the pinned registry and determinants documents from the checkout."""
    root = _REPOSITORY_ROOT if repo_root is None else Path(repo_root)
    perf_dir = root / PERF_DIR_NAME
    return ReleaseArtifacts(
        repo_root=root,
        registry=_read_object(perf_dir / REGISTRY_FILENAME),
        determinants=_read_object(perf_dir / DETERMINANTS_FILENAME),
    )


def _row_field(entry: Mapping[str, Any], field_name: str, campaign_id: str) -> str:
    value = entry.get(field_name)
    if not isinstance(value, str) or not value:
        raise ReleaseUsageError(
            "registry_field_invalid", f"{campaign_id}: {field_name}={value!r}"
        )
    return value


def release_rows(registry: Mapping[str, Any]) -> tuple[ReleaseRow, ...]:
    """Return the registry's gated rows in registry order."""
    if registry.get("schema") != REGISTRY_SCHEMA_ID:
        raise ReleaseUsageError(
            "registry_schema_mismatch", repr(registry.get("schema"))
        )
    entries = registry.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ReleaseUsageError("registry_entries_invalid", repr(entries))
    rows: list[ReleaseRow] = []
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ReleaseUsageError("registry_entry_invalid", repr(entry))
        campaign_id = _row_field(entry, "campaign_id", repr(entry))
        if campaign_id in seen:
            raise ReleaseUsageError("registry_duplicate_entry", campaign_id)
        seen.add(campaign_id)
        fields = {
            name: _row_field(entry, name, campaign_id) for name in _REGISTRY_FIELDS
        }
        rows.append(ReleaseRow(campaign_id=campaign_id, **fields))
    return tuple(rows)


def determinant_rows(determinants: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the determinants document's per-row methodology section."""
    if determinants.get("schema") != DETERMINANTS_SCHEMA_ID:
        raise ReleaseUsageError(
            "determinants_schema_mismatch", repr(determinants.get("schema"))
        )
    rows = determinants.get("rows")
    if not isinstance(rows, Mapping) or not rows:
        raise ReleaseUsageError("determinants_rows_invalid", repr(rows))
    return rows


def build_release_blob(
    *,
    artifacts: ReleaseArtifacts,
    threshold_table: ThresholdTableState,
    activation_state: str = ACTIVATION_PRE,
) -> dict[str, Any]:
    """Build and validate the WP6-shaped blob of one contract evaluation.

    A pre-activation blob whose threshold table already carries rows is
    refused: the table is populated by the activation commit, so a populated
    table under the pre-activation state means the artifact and the contract
    have drifted apart.
    """
    if activation_state not in ACTIVATION_STATES:
        raise ReleaseUsageError("activation_state_invalid", activation_state)
    rows = release_rows(artifacts.registry)
    rows_by_id = determinant_rows(artifacts.determinants)
    if activation_state == ACTIVATION_PRE and not threshold_table.is_empty:
        raise ReleaseUsageError(
            "threshold_table_populated_before_activation", str(threshold_table.path)
        )
    document = build_report(())
    document[BLOB_SECTION] = {
        "activation_state": activation_state,
        "threshold_table": {
            "path": str(threshold_table.path),
            "state": threshold_table.state,
            "empty": threshold_table.is_empty,
            "detail": threshold_table.detail,
        },
        "registry": {
            "schema": REGISTRY_SCHEMA_ID,
            "entry_count": len(rows),
            "campaign_ids": [row.campaign_id for row in rows],
        },
        "determinants": {
            "schema": DETERMINANTS_SCHEMA_ID,
            "row_count": len(rows_by_id),
        },
        "rows": [row.as_record() for row in rows],
    }
    validate_report(document)
    return document


def render_blob_line(blob: Mapping[str, Any]) -> str:
    """Render a blob as the single PERF-BLOB line the suite emits."""
    blob_text = json.dumps(blob, sort_keys=True, separators=(",", ":"))
    return f"{BLOB_LINE_PREFIX}{blob_text}"


def emit_release_blob(
    blob: Mapping[str, Any], *, environ: Mapping[str, str] | None = None
) -> str:
    """Print the blob and append it to the guarded report file when configured.

    The report path is resolved by the WP1 guard, so an escaping
    ``CHARTREUX_PERF_REPORT`` raises ``PerfPathError`` before anything is
    written.
    """
    paths = resolve_perf_paths(environ)
    line = render_blob_line(blob)
    print(line)
    if paths.report is not None:
        with paths.report.open("a", encoding="utf-8") as report_file:
            report_file.write(f"{line}\n")
    return line
