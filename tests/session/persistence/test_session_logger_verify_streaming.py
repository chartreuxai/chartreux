"""Session logger full-verify streaming tests (P4a).

The full-verify comparison streams both sides and compares parsed records
as dicts, one pair at a time. These tests pin the bounded-memory property
(tracemalloc — a full-verify without a rewrite never materializes a
full-transcript record list, bound derived from the P0b breakdown) and the
preserved parsed-dict equality basis: numeric-type drift (1 vs 1.0, True vs
1) and key-order-only differences on disk verify as equal (no rewrite),
while a NaN record fails its own equality (nan != nan under dict
comparison, today's behavior preserved exactly), so the repair rewrites it.
"""

from __future__ import annotations

from collections.abc import Iterable
import json
from pathlib import Path
import tracemalloc

import pytest

from chartreux.core.config import ChartreuxConfigSchema, SessionLoggingConfig
from chartreux.core.llm_models import FunctionCall, LLMMessage, Role, ToolCall
from chartreux.core.session.session_loader import MESSAGES_FILENAME, METADATA_FILENAME
from chartreux.core.session.session_logger import (
    SessionLogger,
    _PersistedTranscriptError,
)
from chartreux.core.session_types import AgentStats
from chartreux.core.tools.manager import ToolManager
from tests.session.persistence._session_persistence_support import (
    mock_tool_manager as mock_tool_manager,
    mock_vibe_config as mock_vibe_config,
    session_config as session_config,
    temp_session_dir as temp_session_dir,
)

# Structural bound (P4a), derived from the P0b breakdown
# (perf-baselines/p0b-persistence-baseline.jsonl): the pre-streaming
# full-verify materialized 14.30 MiB of parsed persisted records plus
# 5.32 MiB of live records (19.63 MiB co-resident) and a 6.13 MiB raw-text
# buffer at 20k records, while the streamed comparison holds one record
# pair at a time (~2 KiB). 2 MiB is ~10% of the smallest single-side
# materialization, so any full-transcript record list exceeds it by an
# order of magnitude, while leaving room for the snapshot/refs transients
# the save boundary legitimately holds.
_STREAMING_VERIFY_PEAK_CEILING_MIB = 2.0

_VERIFY_MESSAGE_COUNT = 20_000


def _verify_messages(count: int = _VERIFY_MESSAGE_COUNT) -> list[LLMMessage]:
    """A 20k-record transcript-scale fixture shaped like the P0b baseline.

    Complete five-message turns — user, tool-calling assistant, user, tool
    response, assistant — with each tool call carrying the int ``index``
    field the equality tests below mutate on disk.
    """
    messages: list[LLMMessage] = []
    for index in range(count):
        call_id = f"verify-call-{index // 5:06d}"
        position = index % 5
        if position == 0:
            message = LLMMessage(role=Role.user, content=f"turn {index // 5}")
        elif position == 1:
            message = LLMMessage(
                role=Role.assistant,
                content="calling tool",
                tool_calls=[
                    ToolCall(
                        id=call_id,
                        index=1,
                        function=FunctionCall(name="verify_tool", arguments="{}"),
                    )
                ],
            )
        elif position == 2:
            message = LLMMessage(role=Role.user, content=f"follow-up {index // 5}")
        elif position == 3:
            message = LLMMessage(role=Role.tool, content="result", tool_call_id=call_id)
        else:
            message = LLMMessage(role=Role.assistant, content=f"answer {index // 5}")
        messages.append(message)
    return messages


class TestStreamPersistedRecords:
    def test_yields_parsed_records(self, tmp_path: Path) -> None:
        path = tmp_path / MESSAGES_FILENAME
        path.write_text(
            '{"role": "user", "content": "a"}\n{"role": "user", "content": "b"}\n',
            encoding="utf-8",
        )
        assert list(SessionLogger._stream_persisted_records(path)) == [
            {"role": "user", "content": "a"},
            {"role": "user", "content": "b"},
        ]

    def test_missing_file_yields_nothing(self, tmp_path: Path) -> None:
        path = tmp_path / MESSAGES_FILENAME
        assert list(SessionLogger._stream_persisted_records(path)) == []

    def test_malformed_line_raises(self, tmp_path: Path) -> None:
        path = tmp_path / MESSAGES_FILENAME
        path.write_text('{"role": "user"}\nnot-json\n', encoding="utf-8")
        records = SessionLogger._stream_persisted_records(path)
        assert next(records) == {"role": "user"}
        with pytest.raises(_PersistedTranscriptError):
            next(records)

    def test_non_object_line_raises(self, tmp_path: Path) -> None:
        path = tmp_path / MESSAGES_FILENAME
        path.write_text('{"role": "user"}\n[1, 2]\n', encoding="utf-8")
        records = SessionLogger._stream_persisted_records(path)
        assert next(records) == {"role": "user"}
        with pytest.raises(_PersistedTranscriptError):
            next(records)

    def test_truncated_trailing_line_raises(self, tmp_path: Path) -> None:
        path = tmp_path / MESSAGES_FILENAME
        path.write_text('{"role": "user"}\n{"role": "use', encoding="utf-8")
        records = SessionLogger._stream_persisted_records(path)
        assert next(records) == {"role": "user"}
        with pytest.raises(_PersistedTranscriptError):
            next(records)

    @pytest.mark.parametrize("codec", ["utf-8-sig", "utf-16", "utf-32"])
    def test_bom_transcript_yields_parsed_records(
        self, tmp_path: Path, codec: str
    ) -> None:
        """A BOM-marked transcript decodes like the pre-P4a bulk reader (E2).

        The P4a streamed reader forced UTF-8, so a UTF-16/UTF-32 transcript
        that ``read_safe`` decoded cleanly parsed as nothing.
        """
        path = tmp_path / MESSAGES_FILENAME
        path.write_bytes(
            '{"role": "user", "content": "a"}\n'
            '{"role": "assistant", "content": "b"}\n'.encode(codec)
        )
        assert list(SessionLogger._stream_persisted_records(path)) == [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
        ]

    def test_line_separator_inside_a_string_does_not_split(
        self, tmp_path: Path
    ) -> None:
        """Universal-newline iteration ends lines on CR/LF only.

        ``str.splitlines`` would also split on U+2028, U+2029, and U+0085 and
        misread this record as corrupt; the decode path must keep the
        documented line-splitting behavior.
        """
        content = "a\u2028b\u2029c\u0085d"
        path = tmp_path / MESSAGES_FILENAME
        path.write_text(
            json.dumps({"role": "user", "content": content}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        assert list(SessionLogger._stream_persisted_records(path)) == [
            {"role": "user", "content": content}
        ]

    def test_bulk_reader_matches_streaming_contract(self, tmp_path: Path) -> None:
        path = tmp_path / MESSAGES_FILENAME
        path.write_text('{"role": "user"}\nnot-json\n', encoding="utf-8")
        assert SessionLogger._read_persisted_messages(path) == ([], False)
        path.write_text("", encoding="utf-8")
        assert SessionLogger._read_persisted_messages(path) == ([], True)
        path.write_text('{"role": "user"}\n', encoding="utf-8")
        records, valid = SessionLogger._read_persisted_messages(path)
        assert valid
        assert records == [{"role": "user"}]


class TestFullVerifyEncodingDetection:
    async def _run_save(
        self,
        logger: SessionLogger,
        messages: list[LLMMessage],
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> list[str]:
        """Save through the full verify, recording any transcript write."""
        writes: list[str] = []
        original_append = SessionLogger._persist_messages_sync
        original_overwrite = SessionLogger._overwrite_messages_sync

        def append(records: Iterable[dict], session_dir: Path) -> int:
            writes.append("append")
            return original_append(records, session_dir)

        def overwrite(records: list[dict], session_dir: Path) -> None:
            writes.append("overwrite")
            original_overwrite(records, session_dir)

        monkeypatch.setattr(
            SessionLogger, "_persist_messages_sync", staticmethod(append)
        )
        monkeypatch.setattr(
            SessionLogger, "_overwrite_messages_sync", staticmethod(overwrite)
        )
        await logger.save_interaction(
            messages, AgentStats(steps=1), mock_vibe_config, mock_tool_manager
        )
        return writes

    @pytest.mark.asyncio
    async def test_utf16_bom_transcript_verifies_clean(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A UTF-16/BOM transcript verifies clean instead of being quarantined.

        The pre-P4a bulk reader decoded it through ``read_safe``; forcing the
        streamed reader to UTF-8 classified the same transcript as corrupt,
        quarantined it, and rewrote it. The reader now decodes by BOM, so the
        persisted boundary survives untouched.
        """
        message = LLMMessage(role=Role.user, content="kept")
        line = (
            json.dumps(SessionLogger._message_record(message), ensure_ascii=False)
            + "\n"
        )
        logger = SessionLogger(session_config, "verify-utf16-transcript")
        assert logger.session_dir is not None
        logger.session_dir.mkdir(parents=True, exist_ok=True)
        messages_path = logger.session_dir / MESSAGES_FILENAME
        messages_path.write_bytes(line.encode("utf-16"))
        persisted_before = messages_path.read_bytes()

        writes = await self._run_save(
            logger, [message], mock_vibe_config, mock_tool_manager, monkeypatch
        )

        assert writes == []
        assert messages_path.read_bytes() == persisted_before
        assert not list(logger.session_dir.glob("messages.jsonl.corrupt-*"))
        assert logger._transcript_cursor is not None


class TestFullVerifyStreamsTranscript:
    def _verify(
        self,
        logger: SessionLogger,
        messages: list[LLMMessage],
        mock_tool_manager: ToolManager,
    ) -> None:
        """Run the full verify directly, excluding the save-boundary snapshots."""
        session_info = logger._get_session_info()
        assert session_info is not None
        session_dir, session_metadata = session_info
        logger._save_full_verify(
            messages,
            AgentStats(steps=1),
            mock_tool_manager,
            session_dir,
            session_metadata,
            launch_config_dirty=False,
        )

    def _write_transcript(self, logger: SessionLogger, lines: list[str]) -> Path:
        assert logger.session_dir is not None
        logger.session_dir.mkdir(parents=True, exist_ok=True)
        messages_path = logger.session_dir / MESSAGES_FILENAME
        messages_path.write_text("".join(lines), encoding="utf-8")
        return messages_path

    @pytest.mark.asyncio
    async def test_full_verify_of_matching_transcript_does_not_materialize_records(
        self,
        session_config: SessionLoggingConfig,
        mock_tool_manager: ToolManager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """tracemalloc: a matching full-verify stays inside the pair budget.

        The transcript matches the live messages and the metadata cache
        confirms the boundary, so the verify is a pure stream comparison:
        no transcript write happens, and the peak stays far below the
        pre-streaming materialization cost (see the bound's derivation).
        """
        message_count = _VERIFY_MESSAGE_COUNT
        messages = _verify_messages(message_count)
        logger = SessionLogger(session_config, "verify-streaming-match")
        assert logger.session_dir is not None
        messages_path = self._write_transcript(
            logger,
            [
                json.dumps(SessionLogger._message_record(message), ensure_ascii=False)
                + "\n"
                for message in messages
            ],
        )
        (logger.session_dir / METADATA_FILENAME).write_text(
            json.dumps({
                "total_messages": message_count,
                "last_message_fingerprint": SessionLogger._message_fingerprint(
                    messages[-1]
                ),
            }),
            encoding="utf-8",
        )
        persisted_before = messages_path.read_bytes()

        writes: list[str] = []
        original_append = SessionLogger._persist_messages_sync
        original_overwrite = SessionLogger._overwrite_messages_sync

        def append(records: Iterable[dict], session_dir: Path) -> int:
            writes.append("append")
            return original_append(records, session_dir)

        def overwrite(records: list[dict], session_dir: Path) -> None:
            writes.append("overwrite")
            original_overwrite(records, session_dir)

        monkeypatch.setattr(
            SessionLogger, "_persist_messages_sync", staticmethod(append)
        )
        monkeypatch.setattr(
            SessionLogger, "_overwrite_messages_sync", staticmethod(overwrite)
        )

        tracemalloc.start()
        try:
            tracemalloc.reset_peak()
            self._verify(logger, messages, mock_tool_manager)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        assert writes == []
        assert messages_path.read_bytes() == persisted_before
        assert peak / (1024 * 1024) < _STREAMING_VERIFY_PEAK_CEILING_MIB

    @pytest.mark.asyncio
    async def test_full_verify_append_streams_the_tail(
        self,
        session_config: SessionLoggingConfig,
        mock_tool_manager: ToolManager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """tracemalloc: a verified-prefix append streams its tail records.

        The persisted transcript matches the live prefix and ten messages
        are new; with no metadata file the cache misses, so the verify
        appends the tail. The tail records are never materialized as a
        list — the peak stays inside the pair budget.
        """
        message_count = _VERIFY_MESSAGE_COUNT
        messages = _verify_messages(message_count)
        logger = SessionLogger(session_config, "verify-streaming-append")
        assert logger.session_dir is not None
        messages_path = self._write_transcript(
            logger,
            [
                json.dumps(SessionLogger._message_record(message), ensure_ascii=False)
                + "\n"
                for message in messages[:-10]
            ],
        )
        persisted_before = messages_path.read_bytes()

        append_sizes: list[int] = []
        original_append = SessionLogger._persist_messages_sync

        def append(records: Iterable[dict], session_dir: Path) -> int:
            record_list = list(records)
            append_sizes.append(len(record_list))
            return original_append(record_list, session_dir)

        monkeypatch.setattr(
            SessionLogger, "_persist_messages_sync", staticmethod(append)
        )

        tracemalloc.start()
        try:
            tracemalloc.reset_peak()
            self._verify(logger, messages, mock_tool_manager)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        assert append_sizes == [10]
        assert messages_path.stat().st_size > len(persisted_before)
        assert peak / (1024 * 1024) < _STREAMING_VERIFY_PEAK_CEILING_MIB


class TestFullVerifyEqualityBasis:
    """The preserved parsed-dict equality basis of the full-verify (P4a).

    The comparison compares parsed records as dicts: numeric-type drift
    (1 vs 1.0, True vs 1) and key-order-only differences on disk are equal
    (no rewrite), while a NaN record fails its own equality (nan != nan
    under dict comparison — today's behavior, preserved exactly), so the
    repair rewrites it. A NaN record cannot verify as equal without
    replacing the dict-equality basis with a serialization-based
    comparison, which P4a forbids.
    """

    async def _save(
        self,
        logger: SessionLogger,
        messages: list[LLMMessage],
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
    ) -> None:
        await logger.save_interaction(
            messages, AgentStats(steps=1), mock_vibe_config, mock_tool_manager
        )

    @staticmethod
    def _tool_call_message() -> LLMMessage:
        return LLMMessage(
            role=Role.assistant,
            content="calling tool",
            tool_calls=[
                ToolCall(
                    id="call-1",
                    index=1,
                    function=FunctionCall(name="verify_tool", arguments="{}"),
                )
            ],
        )

    async def _assert_rewrites_nothing(
        self,
        logger: SessionLogger,
        messages: list[LLMMessage],
        lines: list[str],
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        assert logger.session_dir is not None
        logger.session_dir.mkdir(parents=True, exist_ok=True)
        messages_path = logger.session_dir / MESSAGES_FILENAME
        messages_path.write_text("".join(lines), encoding="utf-8")
        persisted_before = messages_path.read_bytes()

        writes: list[str] = []
        original_append = SessionLogger._persist_messages_sync
        original_overwrite = SessionLogger._overwrite_messages_sync

        def append(records: Iterable[dict], session_dir: Path) -> int:
            writes.append("append")
            return original_append(records, session_dir)

        def overwrite(records: list[dict], session_dir: Path) -> None:
            writes.append("overwrite")
            original_overwrite(records, session_dir)

        monkeypatch.setattr(
            SessionLogger, "_persist_messages_sync", staticmethod(append)
        )
        monkeypatch.setattr(
            SessionLogger, "_overwrite_messages_sync", staticmethod(overwrite)
        )

        await self._save(logger, messages, mock_vibe_config, mock_tool_manager)

        assert writes == []
        assert messages_path.read_bytes() == persisted_before
        assert not list(logger.session_dir.glob("messages.jsonl.corrupt-*"))
        assert logger._transcript_cursor is not None

    @pytest.mark.asyncio
    async def test_float_index_drift_on_disk_verifies_equal(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The disk record carries 1.0 where the live record carries 1.

        Dict equality accepts it; a serialization-based comparison would
        not. (P4a preserved basis.)
        """
        message = self._tool_call_message()
        drift = SessionLogger._message_record(message)
        drift["tool_calls"][0]["index"] = 1.0
        line = json.dumps(drift, ensure_ascii=False) + "\n"
        assert json.loads(line) == SessionLogger._message_record(message)

        logger = SessionLogger(session_config, "verify-drift-float")
        await self._assert_rewrites_nothing(
            logger, [message], [line], mock_vibe_config, mock_tool_manager, monkeypatch
        )

    @pytest.mark.asyncio
    async def test_bool_index_drift_on_disk_verifies_equal(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The disk record carries true where the live record carries 1.

        Dict equality accepts it; a serialization-based comparison would
        not. (P4a preserved basis.)
        """
        message = self._tool_call_message()
        drift = SessionLogger._message_record(message)
        drift["tool_calls"][0]["index"] = True
        line = json.dumps(drift, ensure_ascii=False) + "\n"
        assert json.loads(line) == SessionLogger._message_record(message)

        logger = SessionLogger(session_config, "verify-drift-bool")
        await self._assert_rewrites_nothing(
            logger, [message], [line], mock_vibe_config, mock_tool_manager, monkeypatch
        )

    @pytest.mark.asyncio
    async def test_key_order_drift_on_disk_verifies_equal(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The disk record's keys are sorted; the live record's are not.

        Dict equality is insertion-order-insensitive, so the record
        verifies as equal (no rewrite) — the canonical-serialization
        comparison the P4a plan rejected as spurious would not.
        """
        message = self._tool_call_message()
        record = SessionLogger._message_record(message)
        line = json.dumps(json.loads(json.dumps(record, sort_keys=True))) + "\n"
        assert json.loads(line) == SessionLogger._message_record(message)
        assert line != json.dumps(record, ensure_ascii=False) + "\n"

        logger = SessionLogger(session_config, "verify-drift-key-order")
        await self._assert_rewrites_nothing(
            logger, [message], [line], mock_vibe_config, mock_tool_manager, monkeypatch
        )

    @pytest.mark.asyncio
    async def test_nan_record_round_trips_as_equal(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A NaN record round-trips as equal (P4a preserved basis).

        The JSON parser shares one NaN object per literal and that object
        survives ``_message_record`` identity-wise, so the persisted NaN
        line parses to the same object the live record holds: dict
        equality (its identity shortcut) verifies it as equal — no
        rewrite. A message whose NaN arrived through JSON (a provider
        payload or a previously persisted transcript) therefore re-verifies
        cleanly; a serialization-based comparison would also accept it,
        but only via its "NaN" text.
        """
        message = LLMMessage(
            role=Role.user, content="kept", turn_duration=json.loads("NaN")
        )
        line = (
            json.dumps(SessionLogger._message_record(message), ensure_ascii=False)
            + "\n"
        )
        assert json.loads(line) == SessionLogger._message_record(message)

        logger = SessionLogger(session_config, "verify-nan-roundtrip")
        await self._assert_rewrites_nothing(
            logger, [message], [line], mock_vibe_config, mock_tool_manager, monkeypatch
        )

    @pytest.mark.asyncio
    async def test_distinct_nan_record_is_repaired_not_accepted(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A NaN object distinct from the parser's singleton mismatches.

        nan != nan under dict comparison when the objects differ: a
        computed NaN never equals the NaN singleton a re-parsed line
        carries, so the full-verify detects the mismatch and the repair
        rewrites the transcript — today's behavior, preserved exactly.
        The line parses as a valid record, so no quarantine fires.
        """
        message = LLMMessage(role=Role.user, content="kept", turn_duration=float("nan"))
        disk_line = (
            json.dumps(SessionLogger._message_record(message), ensure_ascii=False)
            + "\n"
        )
        # The parsed line's NaN is the parser's shared singleton; the live
        # record's NaN is a distinct object, so the pair is dict-unequal.
        assert json.loads(disk_line) != SessionLogger._message_record(message)
        assert json.loads(disk_line) == json.loads(disk_line)

        logger = SessionLogger(session_config, "verify-nan-distinct")
        assert logger.session_dir is not None
        logger.session_dir.mkdir(parents=True, exist_ok=True)
        messages_path = logger.session_dir / MESSAGES_FILENAME
        messages_path.write_text(disk_line, encoding="utf-8")

        rewrites: list[list[dict]] = []
        original_overwrite = SessionLogger._overwrite_messages_sync

        def overwrite(records: list[dict], session_dir: Path) -> None:
            rewrites.append(records)
            original_overwrite(records, session_dir)

        monkeypatch.setattr(
            SessionLogger, "_overwrite_messages_sync", staticmethod(overwrite)
        )

        await self._save(logger, [message], mock_vibe_config, mock_tool_manager)

        assert len(rewrites) == 1
        assert not list(logger.session_dir.glob("messages.jsonl.corrupt-*"))
        assert [
            json.loads(line)["content"]
            for line in messages_path.read_text().splitlines()
        ] == ["kept"]
