from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
import fcntl
import os
from pathlib import Path
import stat
from threading import Event, get_ident

import pytest

from chartreux.core import _usage_io
from chartreux.core.paths import USAGE_DIR
from chartreux.core.usage import (
    AsyncUsageWriter,
    CoverageWarningCode,
    UsageOutcome,
    UsagePrices,
    UsageRecord,
    UsageState,
    UsageWriteDisposition,
    UsageWriter,
    UsageWriteResult,
)
from chartreux.utils import durable_io


def make_record(record_id: str = "call-1", root: str = "root") -> UsageRecord:
    return UsageRecord(
        record_id=record_id,
        occurred_at=datetime(2026, 1, 1, tzinfo=UTC),
        root_session_id=root,
        session_id="child",
        parent_session_id="root",
        agent_role="agent",
        model="model",
        provider="provider",
        wire_name="wire",
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


def ledger(usage_dir: Path) -> Path:
    return usage_dir / "root" / "usage.jsonl"


def ids(usage_dir: Path) -> list[str]:
    return [
        item["record_id"]
        for item in _usage_io.parse_usage_jsonl(ledger(usage_dir).read_bytes()).records
    ]


@pytest.mark.parametrize("terminated", [False, True])
def test_append_after_deeply_nested_corrupt_line(
    tmp_path: Path, terminated: bool
) -> None:
    writer = UsageWriter(tmp_path)
    assert writer.append(make_record()).disposition == UsageWriteDisposition.APPENDED
    damaged = b"[" * 20_000 + b"]" * 20_000
    with ledger(tmp_path).open("ab") as stream:
        stream.write(damaged + (b"\n" if terminated else b""))
    before = ledger(tmp_path).read_bytes()
    assert (
        writer.append(make_record("call-2")).disposition
        == UsageWriteDisposition.APPENDED
    )
    assert ledger(tmp_path).read_bytes().startswith(before)
    parsed = _usage_io.parse_usage_jsonl(ledger(tmp_path).read_bytes())
    assert ids(tmp_path) == ["call-1", "call-2"]
    assert parsed.malformed_lines == 1
    assert not parsed.torn_tail
    assert (
        writer.append(make_record("call-2")).disposition
        == UsageWriteDisposition.ALREADY_PRESENT
    )


@pytest.mark.parametrize("separate_writers", [False, True])
def test_concurrent_appenders(tmp_path: Path, separate_writers: bool) -> None:
    shared = UsageWriter(tmp_path)

    def append(index: int) -> UsageWriteDisposition:
        writer = UsageWriter(tmp_path) if separate_writers else shared
        return writer.append(make_record(f"call-{index % 30}")).disposition

    with ThreadPoolExecutor(max_workers=12) as executor:
        results = list(executor.map(append, range(90)))
    assert results.count(UsageWriteDisposition.APPENDED) == 30
    assert results.count(UsageWriteDisposition.ALREADY_PRESENT) == 60
    assert sorted(ids(tmp_path)) == sorted(f"call-{index}" for index in range(30))


@pytest.mark.parametrize("after_write", [False, True])
def test_transient_append_failure_reconciles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, after_write: bool
) -> None:
    original = _usage_io.durable_append
    calls = 0

    def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            if after_write:
                original(*args, **kwargs)
            raise OSError("private exception detail")
        return original(*args, **kwargs)

    monkeypatch.setattr(_usage_io, "durable_append", fail_once)
    result = UsageWriter(tmp_path).append(make_record())
    assert result.disposition == (
        UsageWriteDisposition.ALREADY_PRESENT
        if after_write
        else UsageWriteDisposition.APPENDED
    )
    assert result.warning is None
    assert calls == 2
    assert ids(tmp_path) == ["call-1"]


def test_record_present_requires_file_sync_before_settlement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    original = os.fsync
    file_barriers = 0

    def fail_file_sync(fd: int) -> None:
        nonlocal file_barriers
        if stat.S_ISREG(os.fstat(fd).st_mode):
            file_barriers += 1
            raise OSError("sensitive sync detail")
        original(fd)

    monkeypatch.setattr(os, "fsync", fail_file_sync)
    writer = UsageWriter(tmp_path)
    result = writer.append(make_record())
    assert result.disposition == UsageWriteDisposition.FAILED
    assert result.warning is not None
    assert result.warning.code == CoverageWarningCode.WRITE_FAILED
    assert file_barriers == 2
    assert ids(tmp_path) == ["call-1"]
    assert "degraded" in caplog.text
    assert "sensitive sync detail" not in caplog.text
    monkeypatch.setattr(os, "fsync", original)
    assert (
        writer.append(make_record()).disposition
        == UsageWriteDisposition.ALREADY_PRESENT
    )
    assert ids(tmp_path) == ["call-1"]


def test_directory_publication_barrier_is_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = durable_io.fsync_directory
    publication_barriers = 0

    def fail_publication(path: Path) -> None:
        nonlocal publication_barriers
        if path == ledger(tmp_path).parent and ledger(tmp_path).exists():
            publication_barriers += 1
            raise OSError("directory publication failed")
        original(path)

    monkeypatch.setattr(durable_io, "fsync_directory", fail_publication)
    assert (
        UsageWriter(tmp_path).append(make_record()).disposition
        == UsageWriteDisposition.FAILED
    )
    assert publication_barriers == 2
    assert ids(tmp_path) == ["call-1"]
    monkeypatch.setattr(durable_io, "fsync_directory", original)
    assert (
        UsageWriter(tmp_path).append(make_record()).disposition
        == UsageWriteDisposition.ALREADY_PRESENT
    )
    assert ids(tmp_path) == ["call-1"]


@pytest.mark.parametrize("persistent", [False, True])
def test_directory_creation_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, persistent: bool
) -> None:
    original = durable_io.fsync_directory
    barriers = 0

    def fail_creation(path: Path) -> None:
        nonlocal barriers
        barriers += 1
        if persistent or barriers == 1:
            raise OSError("directory creation barrier failed")
        original(path)

    monkeypatch.setattr(durable_io, "fsync_directory", fail_creation)
    result = UsageWriter(tmp_path).append(make_record())
    assert result.disposition == (
        UsageWriteDisposition.FAILED if persistent else UsageWriteDisposition.APPENDED
    )
    if persistent:
        assert result.warning is not None
        assert not ledger(tmp_path).exists()
    else:
        assert ids(tmp_path) == ["call-1"]


def test_torn_tail_is_terminated_without_truncation(tmp_path: Path) -> None:
    writer = UsageWriter(tmp_path)
    writer.append(make_record("first"))
    with ledger(tmp_path).open("ab") as stream:
        stream.write(b'{"record_id":"broken", "model":')
    before = ledger(tmp_path).read_bytes()
    assert _usage_io.parse_usage_jsonl(before).torn_tail
    assert (
        writer.append(make_record("second")).disposition
        == UsageWriteDisposition.APPENDED
    )
    after = ledger(tmp_path).read_bytes()
    assert after.startswith(before + b"\n")
    parsed = _usage_io.parse_usage_jsonl(after)
    assert ids(tmp_path) == ["first", "second"]
    assert parsed.malformed_lines == 1
    assert not parsed.torn_tail


@pytest.mark.parametrize("complete_object", [False, True])
def test_short_write_is_reconciled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, complete_object: bool
) -> None:
    original = _usage_io.durable_append
    calls = 0

    def short_write(path: Path, content: bytes, **kwargs) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            path.write_bytes(
                content[:-1] if complete_object else content[: len(content) // 2]
            )
            raise OSError("short write")
        original(path, content, **kwargs)

    monkeypatch.setattr(_usage_io, "durable_append", short_write)
    assert UsageWriter(tmp_path).append(make_record()).disposition == (
        UsageWriteDisposition.ALREADY_PRESENT
        if complete_object
        else UsageWriteDisposition.APPENDED
    )
    assert ids(tmp_path) == ["call-1"]
    assert ledger(tmp_path).read_bytes().endswith(b"\n")


def test_parser_tolerates_invalid_utf8_and_torn_tail() -> None:
    result = _usage_io.parse_usage_jsonl(b'{"record_id":"valid"}\n\xff\n{"record_id":')
    assert result.records == ({"record_id": "valid"},)
    assert result.malformed_lines == 1
    assert result.torn_tail


def test_filesystem_failure_returns_warning_without_raising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unavailable(*args, **kwargs):
        raise PermissionError("do not expose this exception")

    monkeypatch.setattr(_usage_io, "durable_mkdir", unavailable)
    result = UsageWriter(tmp_path).append(make_record())
    assert result.disposition == UsageWriteDisposition.FAILED
    assert result.warning is not None
    assert result.warning.root_session_id == "root"
    assert result.warning.record_id == "call-1"


def test_private_permissions(tmp_path: Path) -> None:
    usage_dir = tmp_path / "home" / "usage"
    assert (
        UsageWriter(usage_dir).append(make_record()).disposition
        == UsageWriteDisposition.APPENDED
    )
    for directory in (usage_dir.parent, usage_dir, ledger(usage_dir).parent):
        assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    for path in (ledger(usage_dir), ledger(usage_dir).parent / ".usage.lock"):
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_location_is_lazy_and_independent_of_session_logging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = UsageWriter()
    for home in (tmp_path / "one", tmp_path / "two"):
        monkeypatch.setenv("CHARTREUX_HOME", str(home))
        assert USAGE_DIR.path == home / "usage"
        assert (
            writer.append(make_record()).disposition == UsageWriteDisposition.APPENDED
        )
        assert ids(home / "usage") == ["call-1"]
        assert not (home / "logs").exists()


@pytest.mark.parametrize("root", ["..", "../outside", "/outside", "", "a/b"])
def test_invalid_root_cannot_escape_ledger_directory(tmp_path: Path, root: str) -> None:
    assert (
        UsageWriter(tmp_path).append(make_record(root=root)).disposition
        == UsageWriteDisposition.FAILED
    )
    assert list(tmp_path.iterdir()) == []


def block_write(
    monkeypatch: pytest.MonkeyPatch, stage: str
) -> tuple[asyncio.Event, Event]:
    """Hold a real filesystem operation without blocking the event loop."""
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    release = Event()
    loop_thread = get_ident()

    def pause() -> None:
        assert get_ident() != loop_thread
        loop.call_soon_threadsafe(started.set)
        assert release.wait(5), "test did not release the filesystem barrier"

    if stage == "append":
        original_append = _usage_io.durable_append

        def slow_append(path: Path, content: bytes, **kwargs) -> None:
            pause()
            original_append(path, content, **kwargs)

        monkeypatch.setattr(_usage_io, "durable_append", slow_append)
    else:
        original_sync = os.fsync

        def slow_sync(fd: int) -> None:
            if stat.S_ISREG(os.fstat(fd).st_mode):
                pause()
            original_sync(fd)

        monkeypatch.setattr(os, "fsync", slow_sync)
    return started, release


@pytest.mark.parametrize("stage", ["append", "sync"])
@pytest.mark.parametrize("repeated", [False, True])
@pytest.mark.asyncio
async def test_async_cancellation_waits_for_settlement_and_loop_is_responsive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str, repeated: bool
) -> None:
    started, release = block_write(monkeypatch, stage)
    writer = AsyncUsageWriter(tmp_path)
    notifications: list[UsageWriteResult] = []
    writer.subscribe(lambda record, result: notifications.append(result))
    append = asyncio.create_task(writer.append(make_record()))
    try:
        await asyncio.wait_for(started.wait(), 5)
        append.cancel()
        await asyncio.sleep(0)
        if repeated:
            append.cancel()
            await asyncio.sleep(0)
            append.cancel()
        # A ready coroutine can still execute while real append/fsync is held.
        heartbeat = asyncio.Event()
        asyncio.get_running_loop().call_soon(heartbeat.set)
        await asyncio.wait_for(heartbeat.wait(), 1)
        assert not append.done()
        assert not notifications
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await append
    assert ids(tmp_path) == ["call-1"]
    assert len(notifications) == 1
    assert notifications[0].disposition == UsageWriteDisposition.APPENDED
    await writer.aclose()


@pytest.mark.parametrize("close", [False, True])
@pytest.mark.asyncio
async def test_shutdown_drains_outstanding_writes_despite_repeated_cancellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, close: bool
) -> None:
    started, release = block_write(monkeypatch, "sync")
    writer = AsyncUsageWriter(tmp_path)
    appends = [
        asyncio.create_task(writer.append(make_record(f"call-{index}")))
        for index in range(3)
    ]
    await asyncio.wait_for(started.wait(), 5)
    shutdown = asyncio.create_task(writer.aclose() if close else writer.drain())
    try:
        await asyncio.sleep(0)
        shutdown.cancel()
        await asyncio.sleep(0)
        shutdown.cancel()
        await asyncio.sleep(0)
        assert not shutdown.done()
        if close:
            assert (
                await writer.append(make_record("too-late"))
            ).disposition == UsageWriteDisposition.FAILED
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await shutdown
    results = await asyncio.gather(*appends)
    assert all(
        result.disposition == UsageWriteDisposition.APPENDED for result in results
    )
    assert sorted(ids(tmp_path)) == ["call-0", "call-1", "call-2"]
    await writer.aclose()
    await writer.aclose()


@pytest.mark.asyncio
async def test_callbacks_run_after_sync_outside_locks_on_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started, release = block_write(monkeypatch, "sync")
    writer = AsyncUsageWriter(tmp_path)
    loop_thread = get_ident()
    observed: list[str] = []

    def callback(record: UsageRecord, result: UsageWriteResult) -> None:
        assert get_ident() == loop_thread
        assert release.is_set()
        assert result.disposition == UsageWriteDisposition.APPENDED
        assert writer._writer._append_lock.acquire(blocking=False)
        writer._writer._append_lock.release()
        fd = os.open(ledger(tmp_path).parent / ".usage.lock", os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(fd)
        assert ids(tmp_path) == [record.record_id]
        observed.append(record.record_id)

    unsubscribe = writer.subscribe(callback)
    task = asyncio.create_task(writer.append(make_record()))
    try:
        await asyncio.wait_for(started.wait(), 5)
        assert not observed
    finally:
        release.set()
    await task
    assert observed == ["call-1"]
    unsubscribe()
    unsubscribe()
    await writer.append(make_record("call-2"))
    assert observed == ["call-1"]
    await writer.aclose()


@pytest.mark.asyncio
async def test_async_failure_and_callback_errors_do_not_raise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def unavailable(*args, **kwargs):
        raise PermissionError("sensitive detail")

    def broken_callback(record: UsageRecord, result: UsageWriteResult) -> None:
        raise RuntimeError("sensitive callback detail")

    monkeypatch.setattr(_usage_io, "durable_mkdir", unavailable)
    writer = AsyncUsageWriter(tmp_path)
    notifications: list[UsageWriteResult] = []
    writer.subscribe(broken_callback)
    writer.subscribe(lambda record, result: notifications.append(result))
    result = await writer.append(make_record())
    assert result.disposition == UsageWriteDisposition.FAILED
    assert result.warning is not None
    assert result.warning.code == CoverageWarningCode.WRITE_FAILED
    assert notifications == [result]
    # The AccountingSink adapter also contains failures, with no retry/inference
    # operation beyond the sync core's bounded filesystem reconciliation.
    await writer(make_record("call-2"))
    assert len(notifications) == 2
    assert "sensitive" not in caplog.text
    assert not ledger(tmp_path).exists()
    await writer.aclose()


@pytest.mark.asyncio
async def test_async_uncertain_append_reconciles_without_duplicate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = os.fsync
    file_barriers = 0

    def fail_once(fd: int) -> None:
        nonlocal file_barriers
        if stat.S_ISREG(os.fstat(fd).st_mode):
            file_barriers += 1
            if file_barriers == 1:
                raise OSError("uncertain sync")
        original(fd)

    monkeypatch.setattr(os, "fsync", fail_once)
    writer = AsyncUsageWriter(tmp_path)
    notifications: list[UsageWriteResult] = []
    writer.subscribe(lambda record, result: notifications.append(result))
    result = await writer.append(make_record())
    assert result.disposition == UsageWriteDisposition.ALREADY_PRESENT
    assert result.warning is None
    assert notifications == [result]
    assert file_barriers == 2
    assert ids(tmp_path) == ["call-1"]
    await writer.aclose()
