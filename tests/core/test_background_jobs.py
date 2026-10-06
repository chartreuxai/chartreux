"""Pure state/capture tests; fake lifecycle adapters do not spawn processes."""

from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

from pydantic import ValidationError
import pytest

from chartreux.core.background_jobs import (
    MAX_OUTPUT_BYTES,
    BackgroundJobRegistry,
    BashListArgs,
    BashReadArgs,
    BashStartArgs,
    BashStopArgs,
    OutputRecord,
)
from chartreux.core.tools import secret_redaction as sr


@pytest.mark.asyncio
async def test_complete_metadata_and_normal_eof_preserve_benign_tokens(
    synthetic_credentials,
):
    synthetic_credentials.append(("TEST_KEY", SECRET))
    registry = BackgroundJobRegistry()
    port = registry.root_port()
    for label, command, expected_label, expected_command in (
        ("worker", "printf done", "worker", "printf done"),
        (
            SECRET,
            f"printf {SECRET}",
            sr.REDACTED_PLACEHOLDER,
            f"printf '{sr.REDACTED_PLACEHOLDER}'",
        ),
    ):
        job_id = port.reserve(BashStartArgs(command=command, label=label))
        started = registry.commit(job_id)
        assert started.job.label == expected_label
        assert started.job.command == expected_command
        registry.capture(job_id, b"done", final=True)
        page = await port.read(BashReadArgs(job_id=job_id))
        assert "".join(record.text for record in page.records) == "done"
        registry.observe_leader(job_id, 0)
        registry.settle(job_id)
        listed = port.list(BashListArgs(include_finished=True)).jobs[-1]
        assert listed.label == expected_label and listed.command == expected_command
    await registry.aclose()


@pytest.mark.asyncio
async def test_stop_pending_reservation_releases_slot():
    registry = BackgroundJobRegistry()
    port = registry.root_port()
    job_id = port.reserve(BashStartArgs(command="sleep 60"))
    result = await port.stop(BashStopArgs(job_id=job_id))
    assert result.job.state == "stopped" and not result.already_finished
    assert registry.active_count == 0
    assert (await port.stop(BashStopArgs(job_id=job_id))).already_finished
    await registry.aclose()


@pytest.mark.asyncio
async def test_outward_mask_cache_reuses_window_but_refreshes_credentials(
    monkeypatch, synthetic_credentials
):
    from chartreux.core import background_jobs as bg

    registry = BackgroundJobRegistry()
    port = registry.root_port()
    job_id = port.reserve(BashStartArgs(command="true"))
    registry.commit(job_id)
    registry.capture(job_id, (SECRET + "!").encode())
    original = bg.IncrementalRedactor
    scans = []

    def redactor(policy):
        scans.append(policy)
        return original(policy)

    monkeypatch.setattr(bg, "IncrementalRedactor", redactor)
    args = BashReadArgs(job_id=job_id)
    first = await port.read(args)
    assert SECRET in first.records[0].text
    assert await port.read(args) == first
    assert len(scans) == 1
    synthetic_credentials.append(("TEST_KEY", SECRET))
    refreshed = await port.read(args)
    assert SECRET not in refreshed.records[0].text
    assert len(scans) == 2
    registry.observe_leader(job_id, 0)
    registry.capture(job_id, b"", final=True)
    registry.settle(job_id)
    await registry.aclose()


SECRET = "synthetic-credential-12345678"


@pytest.fixture(autouse=True)
def synthetic_credentials(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    # Isolate all credential discovery; no real environment/keyring/file values.
    values: list[tuple[str, str]] = []
    monkeypatch.setattr(sr, "_loaded_credentials", lambda: list(values))
    monkeypatch.setattr(sr, "_oauth_secret_values", lambda: frozenset())
    monkeypatch.setattr(
        sr, "credential_env_var_names", lambda *args: frozenset({"TEST_KEY"})
    )
    return values


@dataclass
class FakeProcess:
    registry: BackgroundJobRegistry
    job_id: str

    @classmethod
    def launch(cls, registry: BackgroundJobRegistry) -> FakeProcess:
        job_id = registry.root_port().reserve(BashStartArgs(command="printf ready"))
        registry.commit(job_id)
        return cls(registry, job_id)

    def finish(self, code: int = 0, *, abnormal: bool = False) -> None:
        self.registry.observe_leader(self.job_id, code)
        self.registry.capture(self.job_id, b"", final=True, abnormal=abnormal)
        self.registry.settle(self.job_id)


@pytest.mark.parametrize(
    ("model", "args"),
    [
        (BashStartArgs, {"command": ""}),
        (BashStartArgs, {"command": "   "}),
        (BashStartArgs, {"command": "é" * 32769}),
        (BashStartArgs, {"command": "true", "label": "x" * 65}),
        (BashStartArgs, {"command": "true", "label": "x\ny"}),
        (BashStartArgs, {"command": "true", "cwd": "/tmp"}),
        (BashStartArgs, {"command": "true", "env": {}}),
        (BashStartArgs, {"command": "true", "launch_cap": 9}),
        (BashReadArgs, {"job_id": "id", "cursor": -1}),
        (BashReadArgs, {"job_id": "id", "cursor": True}),
        (BashReadArgs, {"job_id": "id", "cursor": 1.0}),
        (BashReadArgs, {"job_id": "id", "cursor": "1"}),
        (BashReadArgs, {"job_id": "id", "max_bytes": 4095}),
        (BashReadArgs, {"job_id": "id", "max_bytes": 64001}),
        (BashReadArgs, {"job_id": "id", "wait_seconds": -1}),
        (BashReadArgs, {"job_id": "id", "wait_seconds": 31}),
        (BashReadArgs, {"job_id": "id", "wait_seconds": float("nan")}),
        (BashStopArgs, {"job_id": "id", "signal": 9}),
        (BashListArgs, {"include_finished": "true"}),
    ],
)
def test_schemas_reject(model: Any, args: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        model(**args)


def test_schema_edges_and_immutable_records() -> None:
    assert BashStartArgs(command="x" * 65536, label="x" * 64).label
    assert BashReadArgs(job_id="id", cursor=0, max_bytes=64000, wait_seconds=30)
    record = OutputRecord(sequence=0, text="é" * 2048)
    with pytest.raises(ValidationError):
        record.text = "changed"
    with pytest.raises(ValidationError):
        OutputRecord(sequence=0, text="é" * 2049)


def test_capacity_reservations_rollback_and_lifecycle() -> None:
    registry = BackgroundJobRegistry()
    port = registry.root_port()
    ids = [port.reserve(BashStartArgs(command="true")) for _ in range(8)]
    assert registry.active_count == 8
    assert not port.list().jobs  # Pending spawn is capacity, not a phantom job.
    with pytest.raises(ValueError, match="capacity"):
        port.reserve(BashStartArgs(command="true"))
    registry.rollback(ids[0])
    replacement = port.reserve(BashStartArgs(command="true"))
    assert replacement not in ids
    start = registry.commit(replacement)
    assert start.next_cursor == 0
    with pytest.raises(ValueError):
        registry.rollback(replacement)
    registry.observe_leader(replacement, 23)
    assert registry.active_count == 8  # Leader exit does NOT release ownership.
    with pytest.raises(ValueError, match="cleanup"):
        registry.settle(replacement)
    registry.capture(replacement, b"done!", final=True)
    assert registry.active_count == 8
    registry.settle(replacement)
    revision = registry.revision
    registry.settle(replacement)
    assert registry.revision == revision
    assert registry.active_count == 7
    summary = port.list(BashListArgs(include_finished=True)).jobs[0]
    assert summary.state == "exited" and summary.exit_code == 23
    assert not port.list().jobs


@pytest.mark.parametrize("abnormal", [False, True])
def test_final_status(abnormal: bool) -> None:
    registry = BackgroundJobRegistry()
    process = FakeProcess.launch(registry)
    process.finish(abnormal=abnormal)
    summary = registry.root_port().list(BashListArgs(include_finished=True)).jobs[0]
    assert summary.state == ("failed" if abnormal else "exited")
    assert summary.output_complete
    assert summary.output_incomplete is abnormal


@pytest.mark.asyncio
async def test_creator_incarnation_visibility_and_concurrent_readers() -> None:
    registry = BackgroundJobRegistry()
    creator = object()
    child = registry.creator_port(creator)
    sibling = registry.creator_port(object())
    resumed = registry.creator_port(object())  # Same stored session is irrelevant.
    root = registry.root_port()
    job_id = child.reserve(BashStartArgs(command="true"))
    registry.commit(job_id)
    registry.capture(job_id, b"ready!")
    args = BashReadArgs(job_id=job_id)
    pages = await asyncio.gather(root.read(args), child.read(args), root.read(args))
    assert pages[0] == pages[1] == pages[2]
    for denied in (sibling, resumed, BackgroundJobRegistry().root_port()):
        assert not denied.list().jobs
        with pytest.raises(ValueError, match="unavailable"):
            await denied.read(args)
        with pytest.raises(ValueError, match="unavailable"):
            denied.authorize(job_id)
    root.authorize(job_id)
    child.authorize(job_id)
    assert root.list().jobs == child.list().jobs


@pytest.mark.asyncio
@pytest.mark.parametrize("max_bytes", [4096, 64000])
@pytest.mark.parametrize("text", ["", "x!" * 6000, "é!" * 4000, "x" * 8000 + "!"])
async def test_cursor_replay_page_limits(max_bytes: int, text: str) -> None:
    registry = BackgroundJobRegistry()
    process = FakeProcess.launch(registry)
    registry.capture(process.job_id, text.encode())
    root = registry.root_port()
    cursor = 0
    output = ""
    while True:
        args = BashReadArgs(job_id=process.job_id, cursor=cursor, max_bytes=max_bytes)
        page = await root.read(args)
        assert page == await root.read(args)
        assert sum(len(r.text.encode()) for r in page.records) <= max_bytes
        if not page.records:
            assert page.next_cursor == cursor == page.end_cursor
            break
        assert page.next_cursor > cursor
        output += "".join(r.text for r in page.records)
        cursor = page.next_cursor
    assert output == text
    with pytest.raises(ValueError, match="end"):
        await root.read(BashReadArgs(job_id=process.job_id, cursor=cursor + 1))
    assert registry._jobs[process.job_id].end_cursor == cursor


@pytest.mark.asyncio
@pytest.mark.parametrize("record", ["x!", "x" * 4095 + "!"])
async def test_count_byte_eviction_multiple_gaps(record: str) -> None:
    registry = BackgroundJobRegistry()
    process = FakeProcess.launch(registry)
    for _ in range(65):
        registry.capture(process.job_id, record.encode())
    job = registry._jobs[process.job_id]
    assert len(job.records) == 64
    assert job.retained_bytes == 64 * len(record.encode()) <= MAX_OUTPUT_BYTES
    if len(record) == 4096:
        assert job.retained_bytes == MAX_OUTPUT_BYTES
    page = await registry.root_port().read(BashReadArgs(job_id=process.job_id))
    assert page.first_cursor == page.lost_records == 1
    assert page.records[0].sequence == 1
    for _ in range(70):
        registry.capture(process.job_id, record.encode())
    page = await registry.root_port().read(
        BashReadArgs(job_id=process.job_id, cursor=1)
    )
    assert page.first_cursor == 71
    assert page.lost_records == 70
    assert page.next_cursor > page.first_cursor
    # Evicted context is never assumed safe, even after a current-policy rotation.
    assert all(r.text == sr.REDACTED_PLACEHOLDER for r in page.records)


@pytest.mark.asyncio
async def test_all_records_evicted_tail_cursor(monkeypatch: pytest.MonkeyPatch) -> None:
    # Exercise the empty retained-tail selector without adding a production
    # purge API (buffer release belongs to WP2). The default remains 64.
    monkeypatch.setattr("chartreux.core.background_jobs.MAX_RECORDS", 0)
    registry = BackgroundJobRegistry()
    process = FakeProcess.launch(registry)
    registry.capture(process.job_id, b"one!")
    registry.capture(process.job_id, b"two!")
    port = registry.root_port()
    gap = await port.read(BashReadArgs(job_id=process.job_id))
    assert not gap.records
    assert (
        gap.first_cursor == gap.end_cursor == gap.next_cursor == gap.lost_records == 2
    )
    end = await port.read(BashReadArgs(job_id=process.job_id, cursor=2))
    assert not end.records and end.lost_records == 0 and end.next_cursor == 2
    registry.capture(process.job_id, b"three!")
    assert registry._jobs[process.job_id].end_cursor == 3


def test_finished_job_retention_never_evicts_live_job() -> None:
    registry = BackgroundJobRegistry()
    live = FakeProcess.launch(registry)
    finished_ids = []
    for _ in range(40):
        process = FakeProcess.launch(registry)
        finished_ids.append(process.job_id)
        process.finish()
    assert len(registry._jobs) == 32
    assert live.job_id in registry._jobs
    assert finished_ids[0] not in registry._jobs
    assert len(set([live.job_id, *finished_ids])) == 41
    assert registry.active_count == 1


@pytest.mark.asyncio
async def test_wait_timeout_wakeup_cancellation_and_no_raw_flush(
    synthetic_credentials: list[tuple[str, str]],
) -> None:
    synthetic_credentials.append(("TEST_KEY", SECRET))
    registry = BackgroundJobRegistry()
    process = FakeProcess.launch(registry)
    port = registry.root_port()
    registry.capture(process.job_id, SECRET[:10].encode())
    page = await port.read(BashReadArgs(job_id=process.job_id, wait_seconds=0.001))
    assert not page.records and page.next_cursor == 0
    waiting = asyncio.create_task(
        port.read(BashReadArgs(job_id=process.job_id, wait_seconds=30))
    )
    await asyncio.sleep(0)
    registry.capture(process.job_id, (SECRET[10:] + "!").encode())
    page = await asyncio.wait_for(waiting, 1)
    assert sr.REDACTED_PLACEHOLDER in page.records[0].text
    waiting = asyncio.create_task(
        port.read(
            BashReadArgs(job_id=process.job_id, cursor=page.end_cursor, wait_seconds=30)
        )
    )
    await asyncio.sleep(0)
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    assert registry.active_count == 1
    waiting = asyncio.create_task(
        port.read(
            BashReadArgs(job_id=process.job_id, cursor=page.end_cursor, wait_seconds=30)
        )
    )
    await asyncio.sleep(0)
    process.finish()
    assert (await asyncio.wait_for(waiting, 1)).job.output_complete


def carriers() -> list[str]:
    payload = ("prefix-data\nTEST_KEY=" + SECRET + "\ntrailing-data").encode()
    encoded = base64.b64encode(payload).decode()
    return [
        SECRET,
        SECRET[::-1],
        quote(SECRET, safe=""),
        "".join(f"%{b:02x}" for b in SECRET.encode()),
        encoded,
        base64.urlsafe_b64encode(payload).decode(),
        payload.hex(),
        " ".join(f"{b:02x}" for b in SECRET.encode()),
        base64.b32encode(payload).decode(),
        "\n".join(encoded[i : i + 12] for i in range(0, len(encoded), 12)),
        " \n".join(encoded[i : i + 12] for i in range(0, len(encoded), 12)),
        "\n".join(
            " ".join(f"{b:02x}" for b in payload[i : i + 16])
            for i in range(0, len(payload), 16)
        ),
        "\n".join(
            f"{i:07o} " + " ".join(f"{b:02x}" for b in payload[i : i + 16])
            for i in range(0, len(payload), 16)
        ),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("carrier", carriers())
async def test_every_carrier_write_split_redacted_before_retention(
    carrier: str, synthetic_credentials: list[tuple[str, str]]
) -> None:
    synthetic_credentials.append(("TEST_KEY", SECRET))
    raw = carrier.encode()
    for split in range(len(raw) + 1):
        registry = BackgroundJobRegistry()
        process = FakeProcess.launch(registry)
        registry.capture(process.job_id, raw[:split])
        # Unresolved suffix cannot be forced into retention by a read.
        assert not (
            await registry.root_port().read(BashReadArgs(job_id=process.job_id))
        ).records
        registry.capture(process.job_id, raw[split:] + b"!")
        job = registry._jobs[process.job_id]
        stored = "".join(r.text for r in job.records)
        assert sr.REDACTED_PLACEHOLDER in stored, (carrier, split, stored)
        assert carrier not in stored
        assert (
            await registry.root_port().read(BashReadArgs(job_id=process.job_id))
        ).records == tuple(job.records)


@pytest.mark.asyncio
async def test_every_utf8_split_and_invalid_eof() -> None:
    raw = "雪☃é!".encode()
    for split in range(len(raw) + 1):
        registry = BackgroundJobRegistry()
        process = FakeProcess.launch(registry)
        registry.capture(process.job_id, raw[:split])
        registry.capture(process.job_id, raw[split:], final=True)
        page = await registry.root_port().read(BashReadArgs(job_id=process.job_id))
        assert "".join(r.text for r in page.records) == raw.decode()
    registry = BackgroundJobRegistry()
    process = FakeProcess.launch(registry)
    registry.capture(process.job_id, b"\xe9", final=True)
    page = await registry.root_port().read(BashReadArgs(job_id=process.job_id))
    assert page.records[0].text == "�"


@pytest.mark.parametrize("abnormal", [False, True])
@pytest.mark.parametrize("text", ["ready\n", "\n ready\n", f"ready\n{SECRET}\nready\n"])
def test_benign_short_words_adjacent_to_secret_every_split(
    text: str, abnormal: bool, synthetic_credentials: list[tuple[str, str]]
) -> None:
    synthetic_credentials.append(("TEST_KEY", SECRET))
    policy = sr.current_policy().capture_for_redaction()
    expected = text.replace(SECRET, sr.REDACTED_PLACEHOLDER)
    for split in range(len(text) + 1):
        redactor = sr.IncrementalRedactor(policy)
        output = redactor.feed(text[:split]) + redactor.feed(text[split:])
        # Completed benign words must be released before EOF, including when
        # capture is later cancelled under load. Chunking cannot change this.
        assert output == expected, (split, output)
        assert redactor.feed("", final=True, abnormal=abnormal) == ""


@pytest.mark.parametrize("secret", ["ready\ncredential123", "a ready\ncredential123"])
def test_short_word_boundary_retains_direct_secret_prefix_every_split(
    secret: str, synthetic_credentials: list[tuple[str, str]]
) -> None:
    synthetic_credentials.append(("TEST_KEY", secret))
    policy = sr.current_policy().capture_for_redaction()
    for split in range(len(secret) + 1):
        redactor = sr.IncrementalRedactor(policy)
        output = redactor.feed(secret[:split]) + redactor.feed(secret[split:] + "☃")
        assert output == sr.REDACTED_PLACEHOLDER + "☃", (split, output)


@pytest.mark.parametrize("abnormal", [False, True])
def test_unresolved_eof_and_oversized_masking(
    abnormal: bool, synthetic_credentials: list[tuple[str, str]]
) -> None:
    synthetic_credentials.append(("TEST_KEY", SECRET))
    policy = sr.current_policy().capture_for_redaction()
    redactor = sr.IncrementalRedactor(policy)
    assert redactor.feed(SECRET[:10]) == ""
    assert redactor.feed("", final=True, abnormal=abnormal) == sr.REDACTED_PLACEHOLDER
    redactor = sr.IncrementalRedactor(policy)
    assert redactor.feed("a" * 8192) == sr.REDACTED_PLACEHOLDER
    assert redactor.pending_chars == 0 and redactor.masking
    assert redactor.feed(SECRET + "a" * 100000) == ""
    assert redactor.masking
    assert redactor.feed("!safe!") == "!safe!"
    assert not redactor.masking


def test_candidate_exhaustion_masks_whole_region(
    synthetic_credentials: list[tuple[str, str]],
) -> None:
    synthetic_credentials.append(("TEST_KEY", SECRET))
    region = "YWJjZGVmZ2g= " * 129 + base64.b64encode(SECRET.encode()).decode()
    redactor = sr.IncrementalRedactor(sr.current_policy().capture_for_redaction())
    assert redactor.feed(region[:500]) == ""
    assert redactor.feed(region[500:] + "!") == sr.REDACTED_PLACEHOLDER + "!"
    assert redactor.pending_chars == 0


@pytest.mark.asyncio
async def test_rotation_keeps_admitted_values_and_shell_metadata(
    synthetic_credentials: list[tuple[str, str]],
) -> None:
    synthetic_credentials.append(("TEST_KEY", SECRET))
    registry = BackgroundJobRegistry()
    quoted = "".join("\\" + char for char in SECRET)
    job_id = registry.root_port().reserve(
        BashStartArgs(command="echo " + quoted, label=SECRET)
    )
    start = registry.commit(job_id)
    synthetic_credentials[:] = [("TEST_KEY", "rotated-synthetic-credential")]
    registry.capture(job_id, (SECRET + "!").encode())
    registry.observe_leader(job_id, 1)
    registry.capture(job_id, b"", final=True)
    registry.settle(job_id, error="error " + SECRET)
    summaries = [
        start.job,
        (await registry.root_port().read(BashReadArgs(job_id=job_id))).job,
        *registry.root_port().list(BashListArgs(include_finished=True)).jobs,
    ]
    for summary in summaries:
        assert SECRET not in summary.model_dump_json()
        assert quoted not in summary.command
        assert len(summary.command) <= 1024
        assert len(summary.error or "") <= 256
    assert SECRET not in "".join(r.text for r in registry._jobs[job_id].records)


@pytest.mark.asyncio
@pytest.mark.parametrize("carrier", carriers())
async def test_post_admission_defense_every_record_page_split(
    carrier: str, synthetic_credentials: list[tuple[str, str]]
) -> None:
    for split in range(1, len(carrier)):
        registry = BackgroundJobRegistry()
        process = FakeProcess.launch(registry)
        # Place the carrier across the 4096-byte record/minimum-page boundary.
        text = "!" * (4096 - len(carrier[:split].encode())) + carrier + "!"
        registry.capture(process.job_id, text.encode())
        synthetic_credentials[:] = [("TEST_KEY", SECRET)]
        # Only the retained context reveals the new match. Registry data is not
        # rewritten, and independent readers replay the same safe record pages.
        port = registry.root_port()
        page = await port.read(BashReadArgs(job_id=process.job_id))
        assert page == await port.read(BashReadArgs(job_id=process.job_id))
        assert all(r.text == sr.REDACTED_PLACEHOLDER for r in page.records)
        assert sum(len(r.text.encode()) for r in page.records) <= 4096
        synthetic_credentials.clear()


@pytest.mark.asyncio
async def test_placeholder_expansion_is_budgeted_before_record_construction(
    synthetic_credentials: list[tuple[str, str]],
) -> None:
    synthetic_credentials.append(("TEST_KEY", "abcdefgh"))
    registry = BackgroundJobRegistry()
    process = FakeProcess.launch(registry)
    raw = "!" * 4088 + "abcdefgh!"
    registry.capture(process.job_id, raw.encode())
    job = registry._jobs[process.job_id]
    assert len(job.records) == 2
    assert len(job.records[0].text.encode()) == 4096
    assert all(len(record.text.encode()) <= 4096 for record in job.records)
    port = registry.root_port()
    first = await port.read(BashReadArgs(job_id=process.job_id))
    second = await port.read(
        BashReadArgs(job_id=process.job_id, cursor=first.next_cursor)
    )
    assert first.next_cursor == 1 and second.next_cursor == 2
    assert (
        "".join(r.text for r in (*first.records, *second.records))
        == "!" * 4088 + sr.REDACTED_PLACEHOLDER + "!"
    )


@pytest.mark.parametrize(
    "secret",
    ["synthetic!credential:123", "synthetic\ncredential123", "秘密synthetic123"],
)
def test_direct_secret_boundary_alphabet_and_utf8_every_split(
    secret: str, synthetic_credentials: list[tuple[str, str]]
) -> None:
    synthetic_credentials.append(("TEST_KEY", secret))
    for split in range(len(secret) + 1):
        redactor = sr.IncrementalRedactor(sr.current_policy().capture_for_redaction())
        first = redactor.feed(secret[:split])
        last = redactor.feed(secret[split:] + "☃")
        assert first + last == sr.REDACTED_PLACEHOLDER + "☃"


@pytest.mark.asyncio
async def test_competing_eighth_ninth_admissions_and_spawn_rollback() -> None:
    registry = BackgroundJobRegistry()
    port = registry.root_port()
    for _ in range(7):
        port.reserve(BashStartArgs(command="true"))
    gate = asyncio.Event()

    async def admit() -> str | None:
        await gate.wait()
        try:
            return port.reserve(BashStartArgs(command="true"))
        except ValueError:
            return None

    tasks = [asyncio.create_task(admit()) for _ in range(2)]
    gate.set()
    results = await asyncio.gather(*tasks)
    admitted = [job_id for job_id in results if job_id is not None]
    assert len(admitted) == 1 and registry.active_count == 8
    registry.rollback(admitted[0])  # Fake spawn adapter rejects before commit.
    assert registry.active_count == 7
    assert port.reserve(BashStartArgs(command="true")) != admitted[0]


def test_giant_capture_bounds_and_output_does_not_increment_revision() -> None:
    registry = BackgroundJobRegistry()
    process = FakeProcess.launch(registry)
    revision = registry.revision
    registry.capture(process.job_id, b"x!" * 500000)
    job = registry._jobs[process.job_id]
    assert len(job.records) == 64
    assert job.retained_bytes <= MAX_OUTPUT_BYTES
    assert job.first_cursor > 0 and job.end_cursor > 64
    assert job.redactor.pending_chars <= 8192
    assert registry.revision == revision


class AdapterProcess:
    """Explicit spawn/stop/drain barriers; no OS ownership in these tests."""

    def __init__(self) -> None:
        self.pid = 123
        self.returncode: int | None = None
        self.stdout = asyncio.StreamReader()
        self._transport = self
        self.closes = 0
        self.exited = asyncio.Event()

    def finish(self, code: int = 0, *, eof: bool = True) -> None:
        self.returncode = code
        self.exited.set()
        if eof:
            self.stdout.feed_eof()

    def members(self) -> bool:
        return self.returncode is None

    def close(self) -> None:
        self.closes += 1

    async def wait(self) -> int:
        await self.exited.wait()
        assert self.returncode is not None
        return self.returncode


@pytest.fixture
def process_adapter(monkeypatch: pytest.MonkeyPatch) -> Any:
    from chartreux.core import background_jobs as bg

    class Adapter:
        def __init__(self) -> None:
            self.spawn_entered = asyncio.Event()
            self.spawn_gate = asyncio.Event()
            self.stop_entered = asyncio.Event()
            self.stop_gate = asyncio.Event()
            self.processes: list[AdapterProcess] = []
            self.stops = 0
            self.releases = 0
            self.fail_spawn = False
            self.fail_stop = False
            self.on_spawn: Any = None

        async def spawn(self, command: str, *, cwd=None, merge_stderr: bool) -> Any:
            assert merge_stderr
            self.spawn_entered.set()
            await self.spawn_gate.wait()
            if self.fail_spawn:
                raise OSError("synthetic spawn failure")
            proc = AdapterProcess()
            self.processes.append(proc)
            if self.on_spawn is not None:
                self.on_spawn()
            return proc

        async def terminate(self, proc: AdapterProcess) -> None:
            self.stops += 1
            self.stop_entered.set()
            await self.stop_gate.wait()
            if self.fail_stop:
                raise PermissionError("synthetic signaling failure")
            if proc.returncode is None:
                proc.finish(-15)

        def release(self, proc: AdapterProcess) -> None:
            self.releases += 1

    adapter = Adapter()
    monkeypatch.setattr(bg, "spawn_shell_command", adapter.spawn)
    monkeypatch.setattr(bg, "registered_process_group", lambda proc: proc)
    monkeypatch.setattr(bg, "terminate_process_group", adapter.terminate)
    monkeypatch.setattr(bg, "release_process_group", adapter.release)
    return adapter


@pytest.mark.asyncio
@pytest.mark.parametrize("close", [False, True])
async def test_pending_launch_repeated_cancellation_and_close(
    process_adapter: Any, close: bool
) -> None:
    adapter = process_adapter
    registry = BackgroundJobRegistry()
    launch = asyncio.create_task(
        registry.root_port().start(BashStartArgs(command="true"))
    )
    try:
        await asyncio.wait_for(adapter.spawn_entered.wait(), 2)
        assert registry.active_count == 1
        closer = asyncio.create_task(registry.aclose()) if close else None
        launch.cancel()
        # Explicit stop barrier proves handle recovery survived all cancellations.
        adapter.spawn_gate.set()
        await asyncio.wait_for(adapter.stop_entered.wait(), 2)
        for _ in range(3):
            launch.cancel()
            await asyncio.sleep(0)
        assert registry.active_count == 1 and not launch.done()
        adapter.stop_gate.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(launch, 2)
        if closer is not None:
            await asyncio.wait_for(closer, 2)
        assert registry.active_count == 0
        assert adapter.stops == adapter.releases == 1
        assert adapter.processes[0].closes == 1
    finally:
        adapter.spawn_gate.set()
        adapter.stop_gate.set()
        await registry.aclose()


@pytest.mark.asyncio
async def test_pending_eighth_ninth_launch_and_spawn_error_frees_slot(
    process_adapter: Any,
) -> None:
    adapter = process_adapter
    registry = BackgroundJobRegistry()
    port = registry.root_port()
    for _ in range(7):
        port.reserve(BashStartArgs(command="true"))
    launches = [
        asyncio.create_task(port.start(BashStartArgs(command="true"))) for _ in range(2)
    ]
    try:
        await asyncio.wait_for(adapter.spawn_entered.wait(), 2)
        assert registry.active_count == 8
        adapter.fail_spawn = True
        adapter.spawn_gate.set()
        results = await asyncio.gather(*launches, return_exceptions=True)
        assert sum(isinstance(result, OSError) for result in results) == 1
        assert sum(isinstance(result, ValueError) for result in results) == 1
        assert registry.active_count == 7
        port.reserve(BashStartArgs(command="true"))
        assert registry.active_count == 8
    finally:
        adapter.spawn_gate.set()
        adapter.stop_gate.set()
        await registry.aclose()


@pytest.mark.asyncio
async def test_committed_interrupted_launch_survives(
    process_adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = process_adapter
    registry = BackgroundJobRegistry()
    adapter.spawn_gate.set()
    launch = asyncio.create_task(
        registry.root_port().start(BashStartArgs(command="true"))
    )
    original_commit = registry.commit

    def commit(job_id: str) -> Any:
        result = original_commit(job_id)
        launch.cancel()
        return result

    monkeypatch.setattr(registry, "commit", commit)
    try:
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(launch, 2)
        assert registry.active_count == 1
        job = next(iter(registry._jobs.values()))
        assert job.committed and job.supervisor is not None
        assert not job.supervisor.cancelled()
    finally:
        adapter.stop_gate.set()
        await registry.aclose()
    assert adapter.releases == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("natural", [False, True])
async def test_shared_stop_finalization_cancellation_and_capacity(
    process_adapter: Any, natural: bool
) -> None:
    adapter = process_adapter
    adapter.spawn_gate.set()
    registry = BackgroundJobRegistry()
    port = registry.root_port()
    start = await port.start(BashStartArgs(command="true"))
    job = registry._jobs[start.job.job_id]
    for _ in range(7):
        port.reserve(BashStartArgs(command="true"))
    if natural:
        adapter.processes[0].finish(7)
    first = asyncio.create_task(port.stop(BashStopArgs(job_id=job.job_id)))
    second = asyncio.create_task(port.stop(BashStopArgs(job_id=job.job_id)))
    reader = asyncio.create_task(
        port.read(BashReadArgs(job_id=job.job_id, wait_seconds=30))
    )
    try:
        await asyncio.wait_for(adapter.stop_entered.wait(), 2)
        for _ in range(3):
            first.cancel()
            reader.cancel()
            await asyncio.sleep(0)
        with pytest.raises(ValueError, match="capacity"):
            port.reserve(BashStartArgs(command="true"))
        assert job.allocated and adapter.stops == 1
        revision = registry.revision
        adapter.stop_gate.set()
        result = await asyncio.wait_for(second, 2)
        await asyncio.gather(first, reader, return_exceptions=True)
        assert result.job.state == ("exited" if natural else "stopped")
        assert result.already_finished == natural
        assert registry.revision == revision + 1
        assert adapter.releases == adapter.processes[0].closes == 1
        late = await port.stop(BashStopArgs(job_id=job.job_id))
        assert late.already_finished and adapter.stops == 1
        port.reserve(BashStartArgs(command="true"))
    finally:
        adapter.stop_gate.set()
        await registry.aclose()


@pytest.mark.asyncio
async def test_close_waiter_cancellation_and_failed_cleanup_retry(
    process_adapter: Any,
) -> None:
    adapter = process_adapter
    adapter.spawn_gate.set()
    registry = BackgroundJobRegistry()
    start = await registry.root_port().start(BashStartArgs(command="true"))
    job = registry._jobs[start.job.job_id]
    adapter.fail_stop = True
    registry.capture(job.job_id, b"retained!\n")
    retained = tuple(job.records)
    first = asyncio.create_task(registry.aclose())
    await asyncio.wait_for(adapter.stop_entered.wait(), 2)
    for _ in range(3):
        first.cancel()
        await asyncio.sleep(0)
    adapter.stop_gate.set()
    await asyncio.gather(first, return_exceptions=True)
    with pytest.raises(RuntimeError, match="retry close"):
        await asyncio.wait_for(registry.aclose(), 2)
    assert job.allocated and registry.active_count == 1
    assert adapter.releases == 0 and job.process is not None
    assert retained and tuple(job.records) == retained
    adapter.fail_stop = False
    await asyncio.wait_for(registry.aclose(), 2)
    assert adapter.releases == 1 and not registry._jobs
    assert not job.records and job.retained_bytes == 0


@pytest.mark.parametrize(
    "text", ["Compilation finished", "listening on localhost 8080", "OK 12345678"]
)
def test_normal_eof_releases_benign_prose_every_split(text, synthetic_credentials):
    synthetic_credentials.append(("TEST_KEY", SECRET))
    policy = sr.current_policy().capture_for_redaction()
    for split in range(len(text) + 1):
        redactor = sr.IncrementalRedactor(policy)
        output = redactor.feed(text[:split]) + redactor.feed(text[split:], final=True)
        assert output == text, (split, output)


def encoded_prefixes():
    prefix = SECRET[:7].encode()
    b64 = base64.b64encode(SECRET.encode()).decode()[:14]
    b32 = base64.b32encode(SECRET.encode()).decode()[:14]
    return [
        prefix.hex(),
        " ".join(f"{b:02x}" for b in prefix),
        "0000000 " + " ".join(f"{b:02x}" for b in prefix),
        base64.b64encode(prefix).decode(),
        b64,
        b64[:8] + "\n" + b64[8:],
        base64.b32encode(prefix).decode(),
        b32,
        b32[:8] + " \n" + b32[8:],
        "".join(f"%{b:02x}" for b in prefix),
        "".join(f"%{b:02X}" for b in prefix),
        base64.b64encode(b"payload\n" + prefix).decode(),
        base64.b64encode(b"payload\n" + prefix + b"\n").decode(),
        base64.b64encode(b"payload\n" + prefix + b" \t\n").decode(),
        *(b64[:8] + "\n" + b64[8:] + whitespace for whitespace in (" ", "\n", " \t\n")),
        *(b32[:8] + "\n" + b32[8:] + whitespace for whitespace in (" ", "\n")),
    ]


@pytest.mark.parametrize("fragment", encoded_prefixes())
def test_normal_eof_masks_encoded_secret_prefix_every_split(
    fragment, synthetic_credentials
):
    synthetic_credentials.append(("TEST_KEY", SECRET))
    policy = sr.current_policy().capture_for_redaction()
    for split in range(len(fragment) + 1):
        redactor = sr.IncrementalRedactor(policy)
        output = redactor.feed(fragment[:split]) + redactor.feed(
            fragment[split:], final=True
        )
        assert output == sr.REDACTED_PLACEHOLDER, (split, output)


@pytest.mark.timeout(60)
@pytest.mark.parametrize("word_count", [1, 129])
def test_normal_eof_candidate_budget_every_split(word_count, synthetic_credentials):
    synthetic_credentials.append(("TEST_KEY", SECRET))
    policy = sr.current_policy().capture_for_redaction()
    text = "YWJjZGVmZ2hpamts " * word_count
    if word_count == 129:
        text += base64.b64encode(b"payload\n" + SECRET.encode()).decode()
    for split in range(len(text) + 1):
        redactor = sr.IncrementalRedactor(policy)
        output = redactor.feed(text[:split]) + redactor.feed(text[split:], final=True)
        assert output == (sr.REDACTED_PLACEHOLDER if word_count == 129 else text), (
            split,
            output,
        )


@pytest.mark.asyncio
async def test_name_only_change_invalidates_finished_output_mask_cache(
    monkeypatch, synthetic_credentials
):
    synthetic_credentials.append(("TEST_KEY", SECRET))
    names = {"TEST_KEY"}
    monkeypatch.setattr(sr, "credential_env_var_names", lambda *args: frozenset(names))
    registry = BackgroundJobRegistry()
    process = FakeProcess.launch(registry)
    carrier = base64.b64encode(b"NEW_KEY=unloaded-value").decode()
    registry.capture(process.job_id, carrier.encode(), final=True)
    registry.observe_leader(process.job_id, 0)
    registry.settle(process.job_id)
    args = BashReadArgs(job_id=process.job_id)
    port = registry.root_port()
    first = await port.read(args)
    assert first.records[0].text == carrier
    values = sr.known_secret_values()
    names.add("NEW_KEY")
    assert sr.known_secret_values() == values
    refreshed = await port.read(args)
    assert refreshed.records[0].text == sr.REDACTED_PLACEHOLDER
    assert await port.read(args) == refreshed
    await registry.aclose()


@pytest.mark.asyncio
async def test_close_at_handle_delivery_prevents_commit(process_adapter: Any) -> None:
    adapter = process_adapter
    registry = BackgroundJobRegistry()
    closers: list[asyncio.Task[None]] = []
    adapter.on_spawn = lambda: closers.append(asyncio.create_task(registry.aclose()))
    adapter.spawn_gate.set()
    adapter.stop_gate.set()
    try:
        with pytest.raises((asyncio.CancelledError, ValueError)):
            await asyncio.wait_for(
                registry.root_port().start(BashStartArgs(command="true")), 2
            )
        await asyncio.wait_for(asyncio.gather(*closers), 2)
        assert adapter.stops == adapter.releases == 1
        assert not registry._jobs and registry.active_count == 0
    finally:
        await registry.aclose()


@pytest.mark.asyncio
async def test_failed_precommit_recovery_keeps_handle_for_close_retry(
    process_adapter: Any,
) -> None:
    adapter = process_adapter
    registry = BackgroundJobRegistry()
    adapter.fail_stop = True
    launch = asyncio.create_task(
        registry.root_port().start(BashStartArgs(command="true"))
    )
    try:
        await asyncio.wait_for(adapter.spawn_entered.wait(), 2)
        launch.cancel()
        adapter.spawn_gate.set()
        await asyncio.wait_for(adapter.stop_entered.wait(), 2)
        adapter.stop_gate.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(launch, 2)
        assert registry.active_count == 1 and adapter.releases == 0
        job = next(iter(registry._jobs.values()))
        assert job.process is not None and not job.committed
        adapter.fail_stop = False
        await asyncio.wait_for(registry.aclose(), 2)
        assert adapter.releases == 1 and registry.active_count == 0
    finally:
        adapter.spawn_gate.set()
        adapter.stop_gate.set()
        adapter.fail_stop = False
        await registry.aclose()
