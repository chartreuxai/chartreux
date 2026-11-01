"""P4a3: the M1 save-behavior matrix over an authoritative archive descriptor.

Six rows, each pinned by named tests in this file:

1. Normal append preserves the archived prefix (serializes only the live
   tail; count/boundary checks stay applicable because the newest boundary
   stays live).
2. Cursor invalidation discards append acceleration, NOT archive authority.
3. Archive-aware full verify: stream-hashes the archived bytes, compares the
   retained prefix fields against the declared trim transformation, compares
   the remaining disk records exactly against live messages; a live-tail
   mismatch publishes a temporary replacement carrying the ORIGINAL archived
   bytes + the authoritative live suffix, atomically, and the published
   replacement is byte-checked to contain the original archived prefix.
4. Missing/truncated/changed archived bytes fail closed with the typed
   ``ArchiveUnavailableError`` surfaced through the non-raising save facade as
   a FAILED save with the diagnostic recorded; the session never re-enters
   the legacy rewrite path while a descriptor is retained; authority-state
   failures recover by session restart; missing/truncated/changed bytes are
   unrecoverable in place (never rebuilt from trimmed messages).
5. Clear/rebind: archive state is cleared only alongside replacement with
   complete messages or an explicitly empty/new conversation — a
   mid-conversation clear never silently drops authority.
6. Ordinary saves never hydrate (M1 §3) and never enter the corrupt-transcript
   quarantine; no watermark is persisted (no sidecar, no JSONL marker, no
   metadata field); cold resume loads complete records via the unchanged
   legacy path.

Legacy sessions (no descriptor) keep the exact-equality and
intentional-shrink behavior covered by test_session_logger_save.py; the
streaming-verify suite (test_session_logger_verify_streaming.py) and the
P6 yield suite (test_session_logger_verify_yield.py) are unchanged and run at
this package's gate.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from chartreux.core.config import ChartreuxConfigSchema, SessionLoggingConfig
from chartreux.core.llm_models import LLMMessage, Role
from chartreux.core.message_list import MessageList
from chartreux.core.session.session_loader import SessionLoader
import chartreux.core.session.session_logger as session_logger_module
from chartreux.core.session.session_logger import ArchiveUnavailableError, SessionLogger
import chartreux.core.session.transcript_archive as archive_module
from chartreux.core.session.transcript_archive import ArchivedPrefixV1
from chartreux.core.session_types import AgentStats
from chartreux.core.tools.manager import ToolManager
from tests.session.persistence._session_persistence_support import (
    mock_tool_manager as mock_tool_manager,
    mock_vibe_config as mock_vibe_config,
    session_config as session_config,
    temp_session_dir as temp_session_dir,
)


def _user(content: str, message_id: str) -> LLMMessage:
    return LLMMessage(role=Role.user, content=content, message_id=message_id)


def _assistant_text(content: str, message_id: str) -> LLMMessage:
    return LLMMessage(role=Role.assistant, content=content, message_id=message_id)


def _boundary() -> LLMMessage:
    return LLMMessage(
        role=Role.user,
        content="compaction boundary",
        injected=True,
        context_boundary="compaction",
    )


def _record_line(message: LLMMessage) -> bytes:
    record = SessionLogger._message_record(message)
    return (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")


async def _sealed_logger(
    session_config: SessionLoggingConfig,
    session_id: str,
    mock_vibe_config: ChartreuxConfigSchema,
    mock_tool_manager: ToolManager,
    *,
    trim_mode: str = "payloads",
) -> tuple[SessionLogger, MessageList, ArchivedPrefixV1]:
    """Build a logger whose eligible closed prefix is sealed (P4a2 receipt).

    The live list is [system, u1, a1, boundary, u2, a2]; the sealed prefix is
    the settled round (u1, a1) strictly before the newest boundary, and the
    boundary envelope plus (u2, a2) stay live. ``a1`` carries an opaque
    reasoning payload so the ``payloads`` trim mode has a real field to drop.
    """
    logger = SessionLogger(session_config, session_id)
    messages = MessageList([
        LLMMessage(role=Role.system, content="prompt"),
        _user("u1", "u1"),
        LLMMessage(
            role=Role.assistant,
            content="a1",
            message_id="a1",
            reasoning_payloads=[{"opaque": "payload-1"}],
        ),
        _boundary(),
        _user("u2", "u2"),
        _assistant_text("a2", "a2"),
    ])
    receipt = await logger.seal_archive(
        messages,
        AgentStats(steps=1),
        mock_vibe_config,
        mock_tool_manager,
        trim_mode=trim_mode,  # type: ignore[arg-type]
    )
    return logger, messages, receipt.descriptor


async def _save(
    logger: SessionLogger,
    messages: Any,
    mock_vibe_config: ChartreuxConfigSchema,
    mock_tool_manager: ToolManager,
    *,
    allow_empty: bool = False,
    steps: int = 1,
) -> None:
    await logger.save_interaction(
        messages,
        AgentStats(steps=steps),
        mock_vibe_config,
        mock_tool_manager,
        allow_empty=allow_empty,
    )


def _messages_path(logger: SessionLogger) -> Path:
    assert logger.session_dir is not None
    return logger.session_dir / "messages.jsonl"


def _prefix_bytes(logger: SessionLogger, descriptor: ArchivedPrefixV1) -> bytes:
    return _messages_path(logger).read_bytes()[: descriptor.prefix_end_byte]


def _contents(path: Path) -> list[Any]:
    return [
        json.loads(line).get("content")
        for line in path.read_bytes().decode("utf-8").splitlines()
    ]


class TestRow1NormalAppendPreservesArchivedPrefix:
    @pytest.mark.asyncio
    async def test_normal_append_serializes_only_the_live_tail(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
    ) -> None:
        logger, messages, descriptor = await _sealed_logger(
            session_config, "p4a3-append", mock_vibe_config, mock_tool_manager
        )
        assert logger.archive_descriptor is descriptor
        original_prefix = _prefix_bytes(logger, descriptor)
        original_all = _messages_path(logger).read_bytes()

        captured: list[dict[str, Any]] = []
        real_persist = SessionLogger._persist_messages_sync

        def spy(records: Any, session_dir: Path) -> int:
            materialized = list(records)
            captured.extend(materialized)
            return real_persist(materialized, session_dir)

        messages.append(_user("u3", "u3"))
        messages.append(_assistant_text("a3", "a3"))
        with mock.patch.object(
            SessionLogger, "_persist_messages_sync", staticmethod(spy)
        ):
            await _save(logger, messages, mock_vibe_config, mock_tool_manager, steps=2)

        # Only the live tail was serialized — the archived prefix was not.
        assert captured == [
            SessionLogger._message_record(messages[6]),
            SessionLogger._message_record(messages[7]),
        ]
        raw = _messages_path(logger).read_bytes()
        assert raw[: descriptor.prefix_end_byte] == original_prefix
        assert hashlib.sha256(raw[: descriptor.prefix_end_byte]).hexdigest() == (
            descriptor.prefix_sha256
        )
        # The pre-existing file content is an exact byte prefix of the new one.
        assert raw[: len(original_all)] == original_all
        assert len(raw.splitlines()) == 7
        assert logger.last_save_diagnostic is None
        assert logger._transcript_cursor is not None
        assert logger._transcript_cursor.count == 7

    @pytest.mark.asyncio
    async def test_count_and_boundary_checks_stay_applicable(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
    ) -> None:
        """An external append breaks the cursor's file_size check and forces the
        archive-aware full verify, which repairs without touching the prefix.
        """
        logger, messages, descriptor = await _sealed_logger(
            session_config, "p4a3-external-append", mock_vibe_config, mock_tool_manager
        )
        original_prefix = _prefix_bytes(logger, descriptor)
        overwrite_calls = 0
        real_overwrite = SessionLogger._overwrite_messages_sync

        def spy_overwrite(*args: Any, **kwargs: Any) -> Any:
            nonlocal overwrite_calls
            overwrite_calls += 1
            return real_overwrite(*args, **kwargs)

        # External writer appends a foreign record past the live boundary.
        with _messages_path(logger).open("ab") as stream:
            stream.write(_record_line(_user("foreign", "fx")))

        messages.append(_user("u3", "u3"))
        with mock.patch.object(
            SessionLogger, "_overwrite_messages_sync", staticmethod(spy_overwrite)
        ):
            await _save(logger, messages, mock_vibe_config, mock_tool_manager, steps=2)

        raw = _messages_path(logger).read_bytes()
        # The archived prefix is byte-identical; the foreign line is gone; the
        # live tail (boundary, u2, a2, u3) is intact. The repair went through
        # the archive-preserving replacement, never the RAM-only overwrite.
        assert raw[: descriptor.prefix_end_byte] == original_prefix
        assert overwrite_calls == 0
        assert _contents(_messages_path(logger)) == [
            "u1",
            "a1",
            "compaction boundary",
            "u2",
            "a2",
            "u3",
        ]
        assert logger.last_save_diagnostic is None


class TestRow2CursorInvalidationKeepsAuthority:
    @pytest.mark.asyncio
    async def test_cursor_invalidation_discards_acceleration_not_authority(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
    ) -> None:
        logger, messages, descriptor = await _sealed_logger(
            session_config,
            "p4a3-cursor-invalidation",
            mock_vibe_config,
            mock_tool_manager,
        )
        await _save(logger, messages, mock_vibe_config, mock_tool_manager, steps=2)
        assert logger._transcript_cursor is not None
        original_prefix = _prefix_bytes(logger, descriptor)

        logger.invalidate_transcript_cursor()

        assert logger._transcript_cursor is None
        # Acceleration discarded, authority retained.
        assert logger.archive_descriptor is descriptor

        messages.append(_assistant_text("a3", "a3"))
        await _save(logger, messages, mock_vibe_config, mock_tool_manager, steps=3)

        # The forced full verify was archive-aware: the prefix survived and
        # the new live record was appended.
        raw = _messages_path(logger).read_bytes()
        assert raw[: descriptor.prefix_end_byte] == original_prefix
        assert _contents(_messages_path(logger))[-1] == "a3"
        assert logger.last_save_diagnostic is None
        assert logger._transcript_cursor is not None
        assert logger._transcript_cursor.count == 6


class TestRow3ArchiveAwareFullVerify:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("trim_mode", ["payloads", "tombstones"])
    async def test_full_verify_matches_the_declared_trim_transformation(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
        trim_mode: str,
    ) -> None:
        """A retained (trimmed) prefix verifies against its DECLARED trim, and
        the transcript on disk is left byte-identical.
        """
        logger, messages, descriptor = await _sealed_logger(
            session_config,
            f"p4a3-trim-{trim_mode}",
            mock_vibe_config,
            mock_tool_manager,
            trim_mode=trim_mode,
        )
        before = _messages_path(logger).read_bytes()
        # Trim the archived slots in place (length-preserving): tombstones
        # keep only role/message_id; payloads mode drops only
        # reasoning_payloads from a1.
        if trim_mode == "tombstones":
            messages.replace(
                1, LLMMessage(role=Role.user, content=None, message_id="u1")
            )
            messages.replace(
                2, LLMMessage(role=Role.assistant, content=None, message_id="a1")
            )
        else:
            messages.replace(2, _assistant_text("a1", "a1"))
        logger.invalidate_transcript_cursor()
        await _save(logger, messages, mock_vibe_config, mock_tool_manager, steps=2)

        assert _messages_path(logger).read_bytes() == before
        assert logger.last_save_diagnostic is None
        assert logger.persisted

    @pytest.mark.asyncio
    async def test_retained_prefix_field_mismatch_fails_closed(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
    ) -> None:
        """A retained prefix record that is NOT the declared trim (altered
        message_id) is an authority-state failure: typed exception on the
        verify path, failed save on the facade, no rewrite.
        """
        logger, messages, descriptor = await _sealed_logger(
            session_config,
            "p4a3-trim-mismatch",
            mock_vibe_config,
            mock_tool_manager,
            trim_mode="tombstones",
        )
        before = _messages_path(logger).read_bytes()
        # Alter a RETAINED field (message_id) — not a declared drop.
        messages.replace(
            1, LLMMessage(role=Role.user, content=None, message_id="hacked")
        )
        logger.invalidate_transcript_cursor()

        await _save(logger, messages, mock_vibe_config, mock_tool_manager, steps=2)

        diagnostic = logger.last_save_diagnostic
        assert isinstance(diagnostic, ArchiveUnavailableError)
        assert diagnostic.recoverable_by_restart
        assert "declared" in diagnostic.reason
        assert _messages_path(logger).read_bytes() == before
        assert logger.archive_descriptor is descriptor

        # The typed exception is raised on the save/verify path itself.
        session_info = logger._get_session_info()
        assert session_info is not None
        session_dir, session_metadata = session_info
        with pytest.raises(ArchiveUnavailableError):
            logger._save_full_verify(
                list(messages),
                AgentStats(steps=3),
                mock_tool_manager,
                session_dir,
                session_metadata,
                launch_config_dirty=False,
            )

    @pytest.mark.asyncio
    async def test_live_tail_mismatch_rewrites_original_archived_bytes_plus_live_suffix(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
    ) -> None:
        """A live-tail mismatch publishes the temporary replacement carrying the
        ORIGINAL archived bytes + the authoritative live suffix, atomically,
        and the published file is byte-checked to contain the original prefix.
        """
        logger, messages, descriptor = await _sealed_logger(
            session_config,
            "p4a3-tail-rewrite",
            mock_vibe_config,
            mock_tool_manager,
            trim_mode="tombstones",
        )
        original_prefix = _prefix_bytes(logger, descriptor)
        # Trim the archived slots so a RAM-only rewrite would emit tombstone
        # records and fail the byte check.
        messages.replace(1, LLMMessage(role=Role.user, content=None, message_id="u1"))
        messages.replace(
            2, LLMMessage(role=Role.assistant, content=None, message_id="a1")
        )
        # Same-count interior edit in the live tail.
        messages.replace(5, _assistant_text("a2-edited", "a2"))

        ram_only_calls = 0

        def ram_only(*args: Any, **kwargs: Any) -> Any:
            nonlocal ram_only_calls
            ram_only_calls += 1
            raise AssertionError("RAM-only _overwrite_messages_sync publish used")

        logger.invalidate_transcript_cursor()
        with mock.patch.object(
            SessionLogger, "_overwrite_messages_sync", staticmethod(ram_only)
        ):
            await _save(logger, messages, mock_vibe_config, mock_tool_manager, steps=2)

        assert ram_only_calls == 0
        raw = _messages_path(logger).read_bytes()
        # BYTE-CHECK: the published replacement contains the original archived
        # prefix, not the RAM tombstone records.
        assert raw[: descriptor.prefix_end_byte] == original_prefix
        assert hashlib.sha256(raw[: descriptor.prefix_end_byte]).hexdigest() == (
            descriptor.prefix_sha256
        )
        assert _contents(_messages_path(logger)) == [
            "u1",
            "a1",
            "compaction boundary",
            "u2",
            "a2-edited",
        ]
        assert logger.last_save_diagnostic is None
        session_dir = logger.session_dir
        assert session_dir is not None
        loaded, _metadata = SessionLoader.load_session(session_dir)
        assert [m.content for m in loaded] == [
            "u1",
            "a1",
            "compaction boundary",
            "u2",
            "a2-edited",
        ]

    @pytest.mark.asyncio
    async def test_reseal_extends_the_prefix_and_preserves_the_old_archived_bytes(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
    ) -> None:
        logger, messages, first = await _sealed_logger(
            session_config, "p4a3-reseal", mock_vibe_config, mock_tool_manager
        )
        old_prefix = _prefix_bytes(logger, first)

        messages.append(_user("u3", "u3"))
        messages.append(_assistant_text("a3", "a3"))
        messages.append(_boundary())
        messages.append(_user("u4", "u4"))
        messages.append(_assistant_text("a4", "a4"))
        await _save(logger, messages, mock_vibe_config, mock_tool_manager, steps=2)

        receipt = await logger.seal_archive(
            messages, AgentStats(steps=2), mock_vibe_config, mock_tool_manager
        )
        second = receipt.descriptor
        assert second.archived_count > first.archived_count
        raw = _messages_path(logger).read_bytes()
        # The old archived prefix is still the exact byte prefix of the file.
        assert raw[: first.prefix_end_byte] == old_prefix
        assert raw[: second.prefix_end_byte].startswith(old_prefix)
        assert logger.archive_descriptor is second
        assert logger.last_save_archive_snapshot is not None
        assert logger.last_save_archive_snapshot.is_authoritative


class TestRow4ArchiveUnavailableDisposition:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("damage", ["delete", "truncate", "flip-byte"])
    async def test_missing_truncated_or_changed_bytes_fail_closed_unrecoverable(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
        damage: str,
    ) -> None:
        logger, messages, descriptor = await _sealed_logger(
            session_config,
            f"p4a3-unavailable-{damage}",
            mock_vibe_config,
            mock_tool_manager,
        )
        path = _messages_path(logger)
        original = path.read_bytes()
        flipped = bytearray(original)
        if damage == "delete":
            path.unlink()
        elif damage == "truncate":
            path.write_bytes(original[: descriptor.prefix_end_byte - 1])
        else:
            flipped[0] ^= 0xFF
            path.write_bytes(bytes(flipped))
        logger.invalidate_transcript_cursor()

        # The non-raising facade reports a FAILED save with the diagnostic
        # recorded; it never raises and never rewrites.
        await _save(logger, messages, mock_vibe_config, mock_tool_manager, steps=2)

        diagnostic = logger.last_save_diagnostic
        assert isinstance(diagnostic, ArchiveUnavailableError)
        assert diagnostic.code == "archive_unavailable"
        assert not diagnostic.recoverable_by_restart
        assert "unrecoverable" in str(diagnostic)
        assert "new conversation is required" in str(diagnostic)
        # The descriptor is retained (authority never silently dropped) and
        # the damaged file is left exactly as it was — no legacy rewrite.
        assert logger.archive_descriptor is descriptor
        if damage == "delete":
            assert not path.exists()
        elif damage == "truncate":
            assert path.read_bytes() == original[: descriptor.prefix_end_byte - 1]
        else:
            assert path.read_bytes() == bytes(flipped)

        # The session never re-enters the legacy rewrite path while a
        # descriptor is retained: the next save fails closed again.
        await _save(logger, messages, mock_vibe_config, mock_tool_manager, steps=3)
        assert isinstance(logger.last_save_diagnostic, ArchiveUnavailableError)
        assert not logger.last_save_diagnostic.recoverable_by_restart

        # The typed exception is raised on the save/verify path itself.
        session_info = logger._get_session_info()
        assert session_info is not None
        session_dir, session_metadata = session_info
        with pytest.raises(ArchiveUnavailableError) as exc_info:
            logger._save_full_verify(
                list(messages),
                AgentStats(steps=4),
                mock_tool_manager,
                session_dir,
                session_metadata,
                launch_config_dirty=False,
            )
        assert not exc_info.value.recoverable_by_restart

    @pytest.mark.asyncio
    async def test_changed_archived_bytes_are_never_rebuilt_from_ram(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
    ) -> None:
        """Even with a COMPLETE live list in RAM, changed archived bytes are not
        rebuilt from RAM — the diagnostic states the history is unrecoverable.
        """
        logger, messages, descriptor = await _sealed_logger(
            session_config, "p4a3-never-rebuild", mock_vibe_config, mock_tool_manager
        )
        path = _messages_path(logger)
        original = path.read_bytes()
        flipped = bytearray(original)
        flipped[10] ^= 0x01
        path.write_bytes(bytes(flipped))

        quarantine_calls = 0
        real_quarantine = SessionLogger._quarantine_corrupt_transcript

        def spy_quarantine(*args: Any, **kwargs: Any) -> Any:
            nonlocal quarantine_calls
            quarantine_calls += 1
            return real_quarantine(*args, **kwargs)

        logger.invalidate_transcript_cursor()
        with mock.patch.object(
            SessionLogger,
            "_quarantine_corrupt_transcript",
            staticmethod(spy_quarantine),
        ):
            await _save(logger, messages, mock_vibe_config, mock_tool_manager, steps=2)

        assert quarantine_calls == 0
        assert path.read_bytes() == bytes(flipped)
        diagnostic = logger.last_save_diagnostic
        assert isinstance(diagnostic, ArchiveUnavailableError)
        assert not diagnostic.recoverable_by_restart

    @pytest.mark.asyncio
    async def test_authority_state_failure_recovers_by_session_restart(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
    ) -> None:
        """A corrupt live tail with a retained descriptor fails closed
        (recover-by-restart class); a fresh logger on the same session — the
        session restart — saves via the unchanged legacy path.
        """
        logger, messages, descriptor = await _sealed_logger(
            session_config, "p4a3-restart-recovery", mock_vibe_config, mock_tool_manager
        )
        path = _messages_path(logger)
        original = path.read_bytes()
        # Corrupt a record in the LIVE TAIL (after the archived prefix).
        tail = original[descriptor.prefix_end_byte :]
        tail_lines = tail.splitlines(keepends=True)
        damaged = (
            original[: descriptor.prefix_end_byte]
            + tail_lines[0]
            + b"not-json\n"
            + b"".join(tail_lines[1:])
        )
        path.write_bytes(damaged)
        logger.invalidate_transcript_cursor()

        await _save(logger, messages, mock_vibe_config, mock_tool_manager, steps=2)
        diagnostic = logger.last_save_diagnostic
        assert isinstance(diagnostic, ArchiveUnavailableError)
        assert diagnostic.recoverable_by_restart
        assert "session restart" in str(diagnostic)
        assert logger.archive_descriptor is descriptor
        assert path.read_bytes() == damaged

        # Session restart: a fresh logger bound to the same session dir retains
        # no descriptor and takes the unchanged legacy path (which may
        # quarantine + rebuild from the complete RAM list, as before).
        session_dir = logger.session_dir
        assert session_dir is not None
        restarted = SessionLogger(
            session_config, logger.session_id, session_dir=session_dir
        )
        assert restarted.archive_descriptor is None
        with pytest.warns(RuntimeWarning, match="quarantined"):
            await _save(
                restarted, list(messages), mock_vibe_config, mock_tool_manager, steps=2
            )
        assert restarted.last_save_diagnostic is None
        assert restarted.persisted
        assert _contents(_messages_path(restarted)) == [
            "u1",
            "a1",
            "compaction boundary",
            "u2",
            "a2",
        ]


class TestRow5ClearRebind:
    @pytest.mark.asyncio
    async def test_mid_conversation_in_place_clear_never_drops_authority(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
    ) -> None:
        """An in-place clear (allow_empty rewind shape, same session) keeps the
        retained descriptor and the original archived bytes on disk.
        """
        logger, messages, descriptor = await _sealed_logger(
            session_config, "p4a3-inplace-clear", mock_vibe_config, mock_tool_manager
        )
        original_prefix = _prefix_bytes(logger, descriptor)

        # An empty MessageList keeps the sealed epoch, so the save stays
        # archive-aware (authoritative) — this is the mid-conversation clear.
        await _save(
            logger, MessageList(), mock_vibe_config, mock_tool_manager, allow_empty=True
        )

        # Authority never silently dropped.
        assert logger.archive_descriptor is descriptor
        raw = _messages_path(logger).read_bytes()
        # The replacement carries the ORIGINAL archived bytes; the live suffix
        # is empty.
        assert raw == original_prefix
        assert hashlib.sha256(raw).hexdigest() == descriptor.prefix_sha256

    @pytest.mark.asyncio
    async def test_reset_session_to_a_new_conversation_clears_archive_state(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
    ) -> None:
        logger, messages, descriptor = await _sealed_logger(
            session_config, "p4a3-reset-clear", mock_vibe_config, mock_tool_manager
        )
        assert logger.archive_descriptor is descriptor

        logger.reset_session("p4a3-reset-clear-new")

        assert logger.archive_descriptor is None
        fresh = MessageList([_user("fresh", "fresh")])
        await _save(logger, fresh, mock_vibe_config, mock_tool_manager)
        assert logger.last_save_diagnostic is None
        assert logger.persisted
        assert _contents(_messages_path(logger)) == ["fresh"]

    @pytest.mark.asyncio
    async def test_rebind_to_a_foreign_session_clears_archive_state(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
        tmp_path: Path,
    ) -> None:
        logger, messages, descriptor = await _sealed_logger(
            session_config, "p4a3-rebind-clear", mock_vibe_config, mock_tool_manager
        )
        assert logger.archive_descriptor is descriptor
        assert logger.session_metadata is not None
        other_dir = tmp_path / "other-session"
        other_dir.mkdir()

        logger.apply_resumed_session(
            "another-session", other_dir, logger.session_metadata
        )

        assert logger.archive_descriptor is None


class TestRow6NoHydrationNoQuarantineNoWatermark:
    @pytest.mark.asyncio
    async def test_ordinary_saves_never_hydrate_the_archive(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """M1 §3: a spy/counter proves the archive read/hydration entry points
        are never entered during a normal append or a full-verify save —
        hydration is reserved for rewind/fork (P4c1). The streaming shared
        read interface IS used (positive control).
        """
        calls = {"hydrate": 0, "read": 0, "iter": 0}
        real_hydrate = archive_module.hydrate_archived_messages
        real_read = archive_module.read_archived_records
        real_iter = session_logger_module.iter_archived_records

        def spy_hydrate(*args: Any, **kwargs: Any) -> Any:
            calls["hydrate"] += 1
            return real_hydrate(*args, **kwargs)

        def spy_read(*args: Any, **kwargs: Any) -> Any:
            calls["read"] += 1
            return real_read(*args, **kwargs)

        def spy_iter(*args: Any, **kwargs: Any) -> Any:
            calls["iter"] += 1
            return real_iter(*args, **kwargs)

        monkeypatch.setattr(archive_module, "hydrate_archived_messages", spy_hydrate)
        monkeypatch.setattr(archive_module, "read_archived_records", spy_read)
        monkeypatch.setattr(session_logger_module, "iter_archived_records", spy_iter)

        logger, messages, descriptor = await _sealed_logger(
            session_config, "p4a3-no-hydrate", mock_vibe_config, mock_tool_manager
        )
        # Normal append (cursor fast path).
        messages.append(_user("u3", "u3"))
        await _save(logger, messages, mock_vibe_config, mock_tool_manager, steps=2)
        # Full-verify save (cursor invalidated).
        logger.invalidate_transcript_cursor()
        messages.append(_assistant_text("a3", "a3"))
        await _save(logger, messages, mock_vibe_config, mock_tool_manager, steps=3)

        assert calls["hydrate"] == 0
        assert calls["read"] == 0
        # Positive control: the full verify streamed the archive through the
        # P4a1-owned shared read interface.
        assert calls["iter"] >= 1
        assert logger.last_save_diagnostic is None

    @pytest.mark.asyncio
    async def test_archive_sessions_never_enter_the_quarantine(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A corrupt tail line with a retained descriptor fails closed — the
        quarantine entry point is never entered (spy/counter).
        """
        logger, messages, descriptor = await _sealed_logger(
            session_config, "p4a3-no-quarantine", mock_vibe_config, mock_tool_manager
        )
        path = _messages_path(logger)
        original = path.read_bytes()
        tail = original[descriptor.prefix_end_byte :]
        damaged = original[: descriptor.prefix_end_byte] + b"not-json\n" + tail
        path.write_bytes(damaged)
        counter = {"quarantine": 0}
        real_quarantine = SessionLogger._quarantine_corrupt_transcript

        def spy_quarantine(*args: Any, **kwargs: Any) -> Any:
            counter["quarantine"] += 1
            return real_quarantine(*args, **kwargs)

        monkeypatch.setattr(
            SessionLogger,
            "_quarantine_corrupt_transcript",
            staticmethod(spy_quarantine),
        )
        logger.invalidate_transcript_cursor()
        await _save(logger, messages, mock_vibe_config, mock_tool_manager, steps=2)

        assert counter["quarantine"] == 0
        assert isinstance(logger.last_save_diagnostic, ArchiveUnavailableError)
        assert logger.last_save_diagnostic.recoverable_by_restart
        assert path.read_bytes() == damaged
        assert logger.session_dir is not None
        assert not list(logger.session_dir.glob("messages.jsonl.corrupt-*"))

    @pytest.mark.asyncio
    async def test_no_watermark_is_persisted_anywhere_on_disk(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
    ) -> None:
        """Negative test: after sealing and saving, no sidecar file, no JSONL
        marker, and no metadata field records the archive state.
        """
        logger, messages, descriptor = await _sealed_logger(
            session_config, "p4a3-no-watermark", mock_vibe_config, mock_tool_manager
        )
        messages.append(_user("u3", "u3"))
        await _save(logger, messages, mock_vibe_config, mock_tool_manager, steps=2)
        logger.invalidate_transcript_cursor()
        await _save(logger, messages, mock_vibe_config, mock_tool_manager, steps=3)

        assert logger.session_dir is not None
        names = sorted(p.name for p in logger.session_dir.iterdir())
        assert names == ["messages.jsonl", "meta.json"]

        marker_keys = {"archived", "archive", "watermark", "trim", "prefix_sha256"}
        for line in _messages_path(logger).read_bytes().decode("utf-8").splitlines():
            record = json.loads(line)
            assert not (set(record) & marker_keys)

        metadata = json.loads((logger.session_dir / "meta.json").read_text())
        for key in metadata:
            assert not any(
                token in key.lower() for token in ("archive", "watermark", "trim")
            )

    @pytest.mark.asyncio
    async def test_cold_resume_of_a_trimmed_session_loads_complete_records_via_legacy_path(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
    ) -> None:
        """Cold resume: a new logger on the previously-trimmed session retains
        no descriptor and loads the COMPLETE records via the unchanged legacy
        path.
        """
        logger, messages, descriptor = await _sealed_logger(
            session_config, "p4a3-cold-resume", mock_vibe_config, mock_tool_manager
        )
        messages.append(_user("u3", "u3"))
        await _save(logger, messages, mock_vibe_config, mock_tool_manager, steps=2)
        session_dir = logger.session_dir
        assert session_dir is not None
        expected_contents = ["u1", "a1", "compaction boundary", "u2", "a2", "u3"]

        cold = SessionLogger(session_config, logger.session_id, session_dir=session_dir)
        assert cold.archive_descriptor is None
        loaded, _metadata = SessionLoader.load_session(session_dir)
        # Complete records, archived prefix included, full payloads intact.
        assert [m.content for m in loaded] == expected_contents

        await _save(cold, loaded, mock_vibe_config, mock_tool_manager, steps=2)
        assert cold.last_save_diagnostic is None
        assert cold.persisted
        reloaded, metadata = SessionLoader.load_session(session_dir)
        assert [m.content for m in reloaded] == expected_contents
        assert metadata["total_messages"] == len(expected_contents)


def test_message_record_rejects_archived_marker_rather_than_persisting_it() -> None:
    # M1 (P4b): a message carrying the private archive reference is an
    # archived replacement; the persist-path record must REJECT it fail-closed
    # rather than ever write a marker to messages.jsonl. Only the streaming
    # verify/comparison paths (allow_archived=True) may read the retained
    # trimmed record — and the private marker itself never serializes.
    from chartreux.core.llm_models import archived_replacement

    message = LLMMessage(
        role=Role.assistant, content="answer", reasoning_payloads=[{"opaque": "p"}]
    )
    replacement = archived_replacement(message, "descriptor-ref")

    with pytest.raises(ValueError, match="must never be persisted"):
        SessionLogger._message_record(replacement)

    record = SessionLogger._message_record(replacement, allow_archived=True)
    assert "_archive_reference" not in record
    assert record["content"] == "answer"
    assert "reasoning_payloads" not in record  # the payloads trim is visible
    # An unmarked message serializes normally on the persist path.
    assert SessionLogger._message_record(message)["content"] == "answer"


def test_message_record_rejects_tombstone_marker() -> None:
    from chartreux.core.llm_models import archived_replacement

    message = LLMMessage(role=Role.user, content="ask", injected=True)
    tombstone = archived_replacement(message, "descriptor-ref", trim_mode="tombstones")

    with pytest.raises(ValueError, match="must never be persisted"):
        SessionLogger._message_record(tombstone)

    record = SessionLogger._message_record(tombstone, allow_archived=True)
    assert set(record) <= {"role", "message_id", "injected"}
