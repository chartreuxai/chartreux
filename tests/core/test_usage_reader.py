from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
import json
import os
from pathlib import Path
from threading import Event

import pytest

from chartreux.core import _usage_io
from chartreux.core.paths import USAGE_DIR
from chartreux.core.usage import (
    AsyncUsageWriter,
    CoverageWarningCode,
    SnapshotState,
    UsageOutcome,
    UsagePrices,
    UsageReader,
    UsageRecord,
    UsageState,
    UsageWriteDisposition,
    UsageWriter,
    UsageWriteResult,
)


def record(record_id: str = "one", root: str = "root") -> UsageRecord:
    return UsageRecord(
        record_id=record_id,
        occurred_at=datetime(2026, 2, 20, tzinfo=UTC),
        root_session_id=root,
        session_id="child",
        parent_session_id=root,
        agent_role="agent",
        agent_profile="researcher",
        model="model",
        provider="provider",
        wire_name="deployment",
        project_key="project",
        outcome=UsageOutcome.COMPLETED,
        usage_state=UsageState.COMPLETE,
        input_tokens=10,
        output_tokens=2,
        cached_input_tokens=0,
        prices_usd_per_million=UsagePrices(input=1, output=2),
        known_cost_usd=0.000014,
        has_unknown_cost=False,
    )


def put(usage_dir: Path, root: str, content: bytes) -> Path:
    path = usage_dir / root / "usage.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def line(item: UsageRecord) -> bytes:
    return (item.model_dump_json() + "\n").encode()


def test_empty_home_and_lazy_default_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reader = UsageReader()
    assert reader.snapshot.state == SnapshotState.LOADING
    monkeypatch.setenv("CHARTREUX_HOME", str(tmp_path / "missing"))
    assert reader.usage_dir == USAGE_DIR.path == tmp_path / "missing" / "usage"
    snapshot = reader.reconcile()
    assert snapshot.state == SnapshotState.READY
    assert snapshot.records == ()
    assert snapshot.warnings == ()
    assert not (tmp_path / "missing").exists()


def test_unavailable_store(tmp_path: Path) -> None:
    path = tmp_path / "not-a-directory"
    path.write_text("file")
    snapshot = UsageReader(path).reconcile()
    assert snapshot.state == SnapshotState.UNAVAILABLE
    assert snapshot.records == ()
    assert [warning.code for warning in snapshot.warnings] == [
        CoverageWarningCode.UNREADABLE
    ]


def test_preserves_priced_unknown_free_missing_and_attribution(tmp_path: Path) -> None:
    priced = record(root="2000-01-01-old-session")
    unknown = priced.model_copy(
        update={
            "record_id": "unknown",
            "prices_usd_per_million": UsagePrices(),
            "known_cost_usd": 0.0,
            "has_unknown_cost": True,
        }
    )
    free = priced.model_copy(
        update={
            "record_id": "free",
            "prices_usd_per_million": UsagePrices(input=0, output=0, cached_input=0),
            "known_cost_usd": 0.0,
        }
    )
    missing = unknown.model_copy(
        update={
            "record_id": "missing",
            "usage_state": UsageState.MISSING,
            "input_tokens": None,
            "output_tokens": None,
            "cached_input_tokens": None,
        }
    )
    items = (priced, unknown, free, missing)
    put(tmp_path, priced.root_session_id, b"".join(line(item) for item in items))
    snapshot = UsageReader(tmp_path).reconcile()
    assert snapshot.records == items
    assert not snapshot.warnings
    assert snapshot.records[-1].input_tokens is None


def test_duplicate_ids_across_files_have_stable_first_winner(tmp_path: Path) -> None:
    first = record(root="a")
    second = record(root="z")
    put(tmp_path, "z", line(second) + line(record("other", "z")))
    put(tmp_path, "a", line(first) + line(first))
    snapshot = UsageReader(tmp_path).reconcile()
    assert snapshot.records == (first, record("other", "z"))


def test_torn_tail_then_completion(tmp_path: Path) -> None:
    tail = line(record("two"))
    path = put(tmp_path, "root", line(record()) + tail[:30])
    reader = UsageReader(tmp_path)
    snapshot = reader.reconcile()
    assert snapshot.records == (record(),)
    assert [warning.code for warning in snapshot.warnings] == [
        CoverageWarningCode.TORN_TAIL
    ]
    with path.open("ab") as stream:
        stream.write(tail[30:])
    snapshot = reader.reconcile()
    assert snapshot.records == (record(), record("two"))
    assert not snapshot.warnings


def test_malformed_and_unsupported_are_coverage_warnings(tmp_path: Path) -> None:
    future = record().model_dump(mode="json")
    future["schema_version"] = 2
    put(
        tmp_path,
        "root",
        b'not json\n[]\n{"record_id":"invalid"}\n'
        + json.dumps(future).encode()
        + b"\n"
        + line(record()),
    )
    snapshot = UsageReader(tmp_path).reconcile()
    assert snapshot.records == (record(),)
    assert {warning.code for warning in snapshot.warnings} == {
        CoverageWarningCode.MALFORMED_RECORD,
        CoverageWarningCode.UNSUPPORTED_SCHEMA,
    }


@pytest.mark.parametrize("terminated", [False, True])
def test_deeply_nested_line_recovers_valid_records(
    tmp_path: Path, terminated: bool
) -> None:
    damaged = b"{" + b'"nested":[' + b"[" * 20_000 + b"]" * 20_000 + b"]}"
    content = line(record()) + damaged
    if terminated:
        content += b"\n" + line(record("two"))
    put(tmp_path, "root", content)
    snapshot = UsageReader(tmp_path).reconcile()
    assert snapshot.records == (
        (record(), record("two")) if terminated else (record(),)
    )
    assert [warning.code for warning in snapshot.warnings] == [
        CoverageWarningCode.MALFORMED_RECORD
        if terminated
        else CoverageWarningCode.TORN_TAIL
    ]


def test_unreadable_file_retries_without_stat_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = put(tmp_path, "root", line(record()))
    original = _usage_io.read_usage_file

    def unreadable(target: Path) -> bytes:
        assert target == path
        raise PermissionError("private detail")

    monkeypatch.setattr(_usage_io, "read_usage_file", unreadable)
    reader = UsageReader(tmp_path)
    snapshot = reader.reconcile()
    assert snapshot.state == SnapshotState.READY
    assert not snapshot.records
    assert [warning.code for warning in snapshot.warnings] == [
        CoverageWarningCode.UNREADABLE
    ]
    monkeypatch.setattr(_usage_io, "read_usage_file", original)
    assert reader.reconcile().records == (record(),)


def test_external_append_and_unchanged_files_not_reread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    put(tmp_path, "root", line(record()))
    put(tmp_path, "other", line(record("other", "other")))
    reads: list[Path] = []
    original = _usage_io.read_usage_file

    def tracked(path: Path) -> bytes:
        reads.append(path)
        return original(path)

    monkeypatch.setattr(_usage_io, "read_usage_file", tracked)
    reader = UsageReader(tmp_path)
    initial = reader.reconcile()
    assert len(reads) == 2
    assert reader.reconcile() == initial
    assert len(reads) == 2
    UsageWriter(tmp_path).append(record("two"))
    assert len(reader.reconcile().records) == 3
    assert reads[2:] == [tmp_path / "root" / "usage.jsonl"]


def test_replacement_same_size_mtime_then_truncation(tmp_path: Path) -> None:
    path = put(tmp_path, "root", line(record("one")))
    reader = UsageReader(tmp_path)
    reader.reconcile()
    info = path.stat()
    replacement = path.with_name("replacement")
    replacement.write_bytes(line(record("two")))
    os.utime(replacement, ns=(info.st_atime_ns, info.st_mtime_ns))
    replacement.replace(path)
    assert reader.reconcile().records == (record("two"),)
    path.write_bytes(b"")
    assert reader.reconcile().records == ()


@pytest.mark.asyncio
async def test_local_settlement_refresh_never_scans_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    put(tmp_path, "other", line(record("other", "other")))
    reader = UsageReader(tmp_path)
    await asyncio.to_thread(reader.reconcile)
    writer = AsyncUsageWriter(tmp_path)
    unsubscribe = writer.subscribe(reader.on_settlement)
    await writer.append(record())
    reads: list[Path] = []
    original = _usage_io.read_usage_file

    def tracked(path: Path) -> bytes:
        reads.append(path)
        return original(path)

    def no_scan(path: Path):
        raise AssertionError("local refresh must not enumerate directories")

    monkeypatch.setattr(_usage_io, "read_usage_file", tracked)
    monkeypatch.setattr(Path, "iterdir", no_scan)
    snapshot = await asyncio.to_thread(reader.refresh_invalidated)
    assert len(snapshot.records) == 2
    assert reads == [tmp_path / "root" / "usage.jsonl"]
    assert await asyncio.to_thread(reader.refresh_invalidated) == snapshot
    assert len(reads) == 1
    unsubscribe()
    await writer.aclose()


@pytest.mark.asyncio
async def test_scan_racing_local_append_retains_epoch_invalidation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = put(tmp_path, "root", line(record()))
    reader = UsageReader(tmp_path)
    initial = reader.reconcile()
    reader.invalidate(path)
    started, release = Event(), Event()
    original = _usage_io.read_usage_file

    def delayed(target: Path) -> bytes:
        content = original(target)
        started.set()
        assert release.wait(10)
        return content

    monkeypatch.setattr(_usage_io, "read_usage_file", delayed)
    writer = AsyncUsageWriter(tmp_path)
    writer.subscribe(reader.on_settlement)
    with ThreadPoolExecutor(max_workers=1) as executor:
        scan = executor.submit(reader.reconcile)
        try:
            assert await asyncio.to_thread(started.wait, 10)
            await writer.append(record("two"))
        finally:
            release.set()
        assert scan.result(timeout=10) == initial
    monkeypatch.setattr(_usage_io, "read_usage_file", original)
    assert reader.refresh_invalidated().records == (record(), record("two"))
    await writer.aclose()


def test_uncertain_failed_settlement_warns_and_invalidates(tmp_path: Path) -> None:
    reader = UsageReader(tmp_path)
    reader.reconcile()
    item = record()
    # Simulate visible bytes with uncertain durability.
    put(tmp_path, "root", line(item))
    from chartreux.core.usage import CoverageWarning

    warning = CoverageWarning(
        code=CoverageWarningCode.WRITE_FAILED, record_id=item.record_id
    )
    reader.on_settlement(item, UsageWriteResult(UsageWriteDisposition.FAILED, warning))
    snapshot = reader.refresh_invalidated()
    assert snapshot.records == (item,)
    assert snapshot.warnings == (warning,)
    reader.on_settlement(item, UsageWriteResult(UsageWriteDisposition.ALREADY_PRESENT))
    assert not reader.refresh_invalidated().warnings
