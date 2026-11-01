"""Persisted-transcript verification yield cadence tests (P6).

``_verify_persisted_boundary`` runs on the save worker thread while the
event loop keeps rendering the stream, so both of its phases — the
comparison and the drain that follows it — release the GIL
(``time.sleep(0)``) every ``PERSISTED_VERIFY_YIELD_INTERVAL`` persisted
records. These tests pin the cadence per phase, the chunk boundaries, and
that chunking preserves the streaming iterator, the returned tuple, and the
exception behavior of the unmodified comparison.

The sleep is replaced by a call counter: the tests assert the cadence, not
that a real handoff happened. Nothing is materialized on either side, so the
comparison pulls one persisted record per live record (asserted by the
interleaving spy).
"""

from __future__ import annotations

from collections.abc import Iterator
import json
from pathlib import Path
from typing import Any

import pytest

from chartreux.core.config import SessionLoggingConfig
from chartreux.core.llm_models import LLMMessage, Role
from chartreux.core.session.session_loader import MESSAGES_FILENAME
import chartreux.core.session.session_logger as session_logger_module
from chartreux.core.session.session_logger import (
    PERSISTED_VERIFY_YIELD_INTERVAL,
    SessionLogger,
)


def _messages(count: int) -> list[LLMMessage]:
    """Live messages with pinned ids.

    An auto-generated id would make two separately built messages unequal and
    defeat the comparison.
    """
    return [
        LLMMessage(
            role=Role.user,
            content=f"message {index}",
            message_id=f"yield-message-{index:05d}",
        )
        for index in range(count)
    ]


def _transcript(record_count: int, extra: int = 0) -> str:
    """The transcript payload for ``record_count`` records plus ``extra`` more.

    Record N is the record of live message N, so a transcript built with no
    extras matches ``_messages(record_count)`` exactly.
    """
    records = _messages(record_count + extra)
    return "".join(
        json.dumps(SessionLogger._message_record(message), ensure_ascii=False) + "\n"
        for message in records
    )


class _SleepSpy:
    """Count the verify's cooperative GIL releases."""

    def __init__(self) -> None:
        self.calls = 0

    def sleep(self, seconds: float) -> None:
        assert seconds == 0
        self.calls += 1


@pytest.fixture
def logger(tmp_path: Path) -> SessionLogger:
    session_dir = tmp_path / "sessions"
    session_dir.mkdir()
    config = SessionLoggingConfig(
        save_dir=str(session_dir), session_prefix="test", enabled=True
    )
    return SessionLogger(config, "verify-yield")


@pytest.fixture
def sleep_spy(monkeypatch: pytest.MonkeyPatch) -> _SleepSpy:
    spy = _SleepSpy()
    monkeypatch.setattr(session_logger_module, "time", spy)
    return spy


def test_yield_interval_is_pinned() -> None:
    assert PERSISTED_VERIFY_YIELD_INTERVAL == 128


def test_comparison_phase_yields_every_interval(
    tmp_path: Path, logger: SessionLogger, sleep_spy: _SleepSpy
) -> None:
    """400 matched records: releases at 128, 256 and 384, and the drain is empty."""
    messages = _messages(400)
    path = tmp_path / MESSAGES_FILENAME
    path.write_text(_transcript(400), encoding="utf-8")

    result = logger._verify_persisted_boundary(messages, path)

    assert sleep_spy.calls == 3
    assert result == (400, True, True, messages[-1])


def test_drain_phase_yields_every_interval(
    tmp_path: Path, logger: SessionLogger, sleep_spy: _SleepSpy
) -> None:
    """A comparison that stops at record 1 leaves 397 records to the drain.

    The drain releases at 128, 256 and 384 of its own count, so the yields
    come from the drain alone.
    """
    messages = _messages(1)
    path = tmp_path / MESSAGES_FILENAME
    mismatched = SessionLogger._message_record(_messages(1)[0])
    mismatched["content"] = "a different first record"
    lines = [json.dumps(mismatched, ensure_ascii=False)]
    lines.extend(
        json.dumps(SessionLogger._message_record(message), ensure_ascii=False)
        for message in _messages(397)
    )
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")

    result = logger._verify_persisted_boundary(messages, path)

    assert result == (0, True, False, None)
    assert sleep_spy.calls == 3


def test_both_phases_yield(
    tmp_path: Path, logger: SessionLogger, sleep_spy: _SleepSpy
) -> None:
    """384 matched records then 256 drained records: 3 + 2 releases.

    The comparison completes, the else-branch probe consumes record 385 and
    finds a longer transcript (the boundary is not a prefix), and the drain
    finishes the remaining 256 records with two more releases.
    """
    messages = _messages(384)
    path = tmp_path / MESSAGES_FILENAME
    path.write_text(_transcript(384, extra=257), encoding="utf-8")

    result = logger._verify_persisted_boundary(messages, path)

    assert result == (384, True, False, messages[-1])
    assert sleep_spy.calls == 5


@pytest.mark.parametrize(
    ("record_count", "expected_yields"),
    [(127, 0), (128, 1), (129, 1), (255, 1), (256, 2), (300, 2)],
)
def test_chunk_boundaries_preserve_the_result(
    tmp_path: Path,
    logger: SessionLogger,
    sleep_spy: _SleepSpy,
    record_count: int,
    expected_yields: int,
) -> None:
    messages = _messages(record_count)
    path = tmp_path / MESSAGES_FILENAME
    path.write_text(_transcript(record_count), encoding="utf-8")

    result = logger._verify_persisted_boundary(messages, path)

    assert sleep_spy.calls == expected_yields
    assert result == (record_count, True, True, messages[-1])


def test_mismatch_on_a_yield_boundary(
    tmp_path: Path, logger: SessionLogger, sleep_spy: _SleepSpy
) -> None:
    """The 128th record releases the GIL before it is compared.

    Its mismatch then stops the comparison at 127 verified records; the
    remaining 73 records drain without another release.
    """
    messages = _messages(200)
    lines = [
        json.dumps(SessionLogger._message_record(message), ensure_ascii=False)
        for message in messages
    ]
    changed = json.loads(lines[127])
    changed["content"] = "record 128 differs"
    lines[127] = json.dumps(changed)
    path = tmp_path / MESSAGES_FILENAME
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")

    result = logger._verify_persisted_boundary(messages, path)

    assert sleep_spy.calls == 1
    assert result == (127, True, False, messages[126])


def test_persisted_transcript_longer_than_live(
    tmp_path: Path, logger: SessionLogger, sleep_spy: _SleepSpy
) -> None:
    """100 live records against 300 persisted: no comparison release, one drain release."""
    messages = _messages(100)
    path = tmp_path / MESSAGES_FILENAME
    path.write_text(_transcript(100, extra=200), encoding="utf-8")

    result = logger._verify_persisted_boundary(messages, path)

    assert sleep_spy.calls == 1
    assert result == (100, True, False, messages[-1])


class _CountingStream:
    """Wrap the streamed records and count how many were pulled."""

    def __init__(self, records: Iterator[dict[str, Any]]) -> None:
        self._records = records
        self.pulled = 0

    def __iter__(self) -> _CountingStream:
        return self

    def __next__(self) -> dict[str, Any]:
        record = next(self._records)
        self.pulled += 1
        return record


def test_comparison_never_materializes_the_persisted_records(
    tmp_path: Path, logger: SessionLogger, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The comparison pulls record N only when it compares live record N.

    A materializing implementation would have pulled the whole transcript
    before the first comparison.
    """
    messages = _messages(300)
    path = tmp_path / MESSAGES_FILENAME
    path.write_text(_transcript(300), encoding="utf-8")

    real_stream = SessionLogger._stream_persisted_records
    stream = _CountingStream(real_stream(path))
    monkeypatch.setattr(
        SessionLogger, "_stream_persisted_records", staticmethod(lambda _path: stream)
    )

    pulls_at_compare: list[int] = []
    real_message_record = SessionLogger._message_record

    def spy(message: LLMMessage, **kwargs: Any) -> dict[str, Any]:
        pulls_at_compare.append(stream.pulled)
        return real_message_record(message, **kwargs)

    monkeypatch.setattr(SessionLogger, "_message_record", staticmethod(spy))

    result = logger._verify_persisted_boundary(messages, path)

    assert result == (300, True, True, messages[-1])
    assert pulls_at_compare == list(range(1, 301))
    assert stream.pulled == 300


def test_corrupt_record_during_comparison(
    tmp_path: Path, logger: SessionLogger, sleep_spy: _SleepSpy
) -> None:
    """A malformed line reached by the comparison still quarantines the transcript."""
    messages = _messages(200)
    lines = [
        json.dumps(SessionLogger._message_record(message), ensure_ascii=False)
        for message in messages[:130]
    ]
    lines.append("not-json")
    lines.extend(
        json.dumps(SessionLogger._message_record(message), ensure_ascii=False)
        for message in messages[130:]
    )
    path = tmp_path / MESSAGES_FILENAME
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")

    result = logger._verify_persisted_boundary(messages, path)

    assert sleep_spy.calls == 1
    assert result == (130, False, False, messages[129])


def test_corrupt_record_during_drain(
    tmp_path: Path, logger: SessionLogger, sleep_spy: _SleepSpy
) -> None:
    """The drain releases the GIL on its own cadence before it hits the bad line."""
    messages = _messages(10)
    lines = [
        json.dumps(SessionLogger._message_record(message), ensure_ascii=False)
        for message in _messages(300)
    ]
    lines.append("not-json")
    path = tmp_path / MESSAGES_FILENAME
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")

    result = logger._verify_persisted_boundary(messages, path)

    assert result == (10, False, False, messages[-1])
    assert sleep_spy.calls == 2


def test_missing_transcript_yields_nothing(
    tmp_path: Path, logger: SessionLogger, sleep_spy: _SleepSpy
) -> None:
    messages = _messages(200)

    result = logger._verify_persisted_boundary(messages, tmp_path / MESSAGES_FILENAME)

    assert sleep_spy.calls == 0
    assert result == (0, True, True, None)
