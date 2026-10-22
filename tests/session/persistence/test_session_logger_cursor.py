"""Session logger transcript cursor tests (invalidation during save, warm-growth appends, interior-insertion invalidation, external-append repair, periodic verify, shrink/boundary rewrite, metadata-failure retry, reset/resume clearing, and resumed-logger verify-once)."""

from __future__ import annotations

import asyncio
import builtins
import json
from pathlib import Path
from threading import Event
from typing import IO, Any

import pytest

from chartreux.core.config import ChartreuxConfigSchema, SessionLoggingConfig
from chartreux.core.llm_models import LLMMessage, Role
from chartreux.core.session.session_loader import SessionLoader
from chartreux.core.session.session_logger import SessionLogger
from chartreux.core.session_types import AgentStats
from chartreux.core.tools.manager import ToolManager
from tests.session.persistence._session_persistence_support import (
    mock_tool_manager as mock_tool_manager,
    mock_vibe_config as mock_vibe_config,
    session_config as session_config,
    temp_session_dir as temp_session_dir,
)


class TestTranscriptCursor:
    async def _save(
        self,
        logger: SessionLogger,
        messages: list[LLMMessage],
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
        *,
        allow_empty: bool = False,
    ) -> None:
        await logger.save_interaction(
            messages,
            AgentStats(steps=1),
            mock_vibe_config,
            mock_tool_manager,
            allow_empty=allow_empty,
        )

    @pytest.mark.asyncio
    async def test_invalidation_during_save_keeps_cursor_cleared(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        logger = SessionLogger(session_config, "cursor-invalidation-race")
        messages = [LLMMessage(role=Role.user, content="first")]
        original_save = logger._save_interaction_sync
        started = Event()
        release = Event()

        def block_save(*args: Any, **kwargs: Any) -> Any:
            started.set()
            if not release.wait(timeout=5):
                raise TimeoutError("save worker was not released")
            return original_save(*args, **kwargs)

        original_read = SessionLogger._read_persisted_messages
        reads: list[Path] = []

        def track_read(messages_path: Path) -> tuple[list[dict], bool]:
            reads.append(messages_path)
            return original_read(messages_path)

        monkeypatch.setattr(logger, "_save_interaction_sync", block_save)
        monkeypatch.setattr(
            SessionLogger, "_read_persisted_messages", staticmethod(track_read)
        )

        save_task = asyncio.create_task(
            self._save(logger, messages, mock_vibe_config, mock_tool_manager)
        )
        if not await asyncio.to_thread(started.wait, 5):
            release.set()
            await save_task
            pytest.fail("save worker did not start")

        logger.invalidate_transcript_cursor()
        release.set()
        await save_task
        assert logger._transcript_cursor is None

        await self._save(
            logger,
            [*messages, LLMMessage(role=Role.assistant, content="next")],
            mock_vibe_config,
            mock_tool_manager,
        )

        assert len(reads) == 2
        assert logger._transcript_cursor is not None

    @pytest.mark.asyncio
    async def test_warm_growth_appends_only_new_tail(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        logger = SessionLogger(session_config, "cursor-tail")
        assert logger.session_dir is not None
        messages_path = logger.session_dir / "messages.jsonl"
        original_append = SessionLogger._persist_messages_sync
        original_overwrite = SessionLogger._overwrite_messages_sync
        original_read = SessionLogger._read_persisted_messages
        original_path_open = Path.open
        original_builtin_open = builtins.open
        append_sizes: list[int] = []
        overwrite_calls: list[list[dict]] = []
        reads: list[Path] = []
        message_file_reads: list[Path] = []

        def is_messages_read(file: Any, mode: str) -> bool:
            if "r" not in mode and "+" not in mode:
                return False
            try:
                return Path(file).resolve() == messages_path.resolve()
            except TypeError:
                return False

        def track_path_open(
            path: Path, mode: str = "r", *args: Any, **kwargs: Any
        ) -> IO[Any]:
            if is_messages_read(path, mode):
                message_file_reads.append(path)
            return original_path_open(path, mode, *args, **kwargs)

        def track_builtin_open(
            file: Any, mode: str = "r", *args: Any, **kwargs: Any
        ) -> IO[Any]:
            if is_messages_read(file, mode):
                message_file_reads.append(Path(file))
            return original_builtin_open(file, mode, *args, **kwargs)

        def append(messages: list[dict], session_dir: Path) -> int:
            append_sizes.append(len(messages))
            return original_append(messages, session_dir)

        def overwrite(messages: list[dict], session_dir: Path) -> None:
            overwrite_calls.append(messages)
            original_overwrite(messages, session_dir)

        def read(messages_path: Path) -> tuple[list[dict], bool]:
            reads.append(messages_path)
            return original_read(messages_path)

        monkeypatch.setattr(Path, "open", track_path_open)
        monkeypatch.setattr(builtins, "open", track_builtin_open)
        monkeypatch.setattr(
            SessionLogger, "_persist_messages_sync", staticmethod(append)
        )
        monkeypatch.setattr(
            SessionLogger, "_overwrite_messages_sync", staticmethod(overwrite)
        )
        monkeypatch.setattr(
            SessionLogger, "_read_persisted_messages", staticmethod(read)
        )

        messages = [LLMMessage(role=Role.user, content="one")]
        await self._save(logger, messages, mock_vibe_config, mock_tool_manager)
        messages.append(LLMMessage(role=Role.assistant, content="two"))
        await self._save(logger, messages, mock_vibe_config, mock_tool_manager)
        messages.append(LLMMessage(role=Role.user, content="three"))
        await self._save(logger, messages, mock_vibe_config, mock_tool_manager)

        assert append_sizes == [1, 1]
        assert len(overwrite_calls) == 1
        assert len(reads) == 1
        assert message_file_reads == []

    @pytest.mark.asyncio
    async def test_interior_insertion_invalidates_repeated_boundary_cursor(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        logger = SessionLogger(session_config, "cursor-interior-insert")
        repeated = LLMMessage(role=Role.user, content="C")
        messages = [LLMMessage(role=Role.user, content="A"), repeated, repeated]
        await self._save(logger, messages, mock_vibe_config, mock_tool_manager)
        cursor = logger._transcript_cursor
        assert cursor is not None
        assert cursor.count == 3

        original_append = SessionLogger._persist_messages_sync
        original_overwrite = SessionLogger._overwrite_messages_sync
        append_calls: list[list[dict]] = []
        overwrite_calls: list[list[dict]] = []

        def append(messages: list[dict], session_dir: Path) -> int:
            append_calls.append(messages)
            return original_append(messages, session_dir)

        def overwrite(messages: list[dict], session_dir: Path) -> None:
            overwrite_calls.append(messages)
            original_overwrite(messages, session_dir)

        monkeypatch.setattr(
            SessionLogger, "_persist_messages_sync", staticmethod(append)
        )
        monkeypatch.setattr(
            SessionLogger, "_overwrite_messages_sync", staticmethod(overwrite)
        )

        messages.insert(1, LLMMessage(role=Role.user, content="X"))
        shifted_boundary = messages[cursor.count - 1].model_dump(
            exclude_none=True, mode="json"
        )
        # Without explicit invalidation, the repeated C would pass the old
        # boundary check and the warm path would append just the final C.
        assert (
            SessionLogger._transcript_line_digest(shifted_boundary)
            == cursor.boundary_digest
        )
        logger.invalidate_transcript_cursor()

        await self._save(logger, messages, mock_vibe_config, mock_tool_manager)

        expected = [
            message.model_dump(exclude_none=True, mode="json") for message in messages
        ]
        assert overwrite_calls == [expected]
        assert append_calls == []
        assert logger.session_dir is not None
        records = [
            json.loads(line)
            for line in (logger.session_dir / "messages.jsonl").read_text().splitlines()
        ]
        assert records == expected

    @pytest.mark.asyncio
    async def test_external_append_forces_full_repair(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
    ) -> None:
        logger = SessionLogger(session_config, "cursor-external-append")
        messages = [LLMMessage(role=Role.user, content="kept")]
        await self._save(logger, messages, mock_vibe_config, mock_tool_manager)
        assert logger.session_dir is not None
        SessionLogger._persist_messages_sync(
            [{"role": "user", "content": "external"}], logger.session_dir
        )

        await self._save(logger, messages, mock_vibe_config, mock_tool_manager)

        records = [
            json.loads(line)
            for line in (logger.session_dir / "messages.jsonl").read_text().splitlines()
        ]
        assert [record["content"] for record in records] == ["kept"]

    @pytest.mark.asyncio
    async def test_periodic_verify_repairs_same_size_interior_edit(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "chartreux.core.session.session_logger.TRANSCRIPT_VERIFY_INTERVAL", 1
        )
        logger = SessionLogger(session_config, "cursor-periodic-verify")
        messages = [
            LLMMessage(role=Role.user, content="first"),
            LLMMessage(role=Role.assistant, content="middle"),
            LLMMessage(role=Role.user, content="last"),
        ]
        await self._save(logger, messages, mock_vibe_config, mock_tool_manager)
        assert logger.session_dir is not None
        messages_path = logger.session_dir / "messages.jsonl"
        original_size = messages_path.stat().st_size
        changed = messages_path.read_text().replace('"first"', '"other"', 1)
        messages_path.write_text(changed)
        assert messages_path.stat().st_size == original_size

        await self._save(logger, messages, mock_vibe_config, mock_tool_manager)

        records = [json.loads(line) for line in messages_path.read_text().splitlines()]
        assert [record["content"] for record in records] == ["first", "middle", "last"]

    @pytest.mark.asyncio
    async def test_warm_cursor_shrink_and_boundary_edit_rewrite(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        logger = SessionLogger(session_config, "cursor-boundary-edit")
        original_overwrite = SessionLogger._overwrite_messages_sync
        overwrite_calls: list[list[dict]] = []

        def overwrite(messages: list[dict], session_dir: Path) -> None:
            overwrite_calls.append(messages)
            original_overwrite(messages, session_dir)

        monkeypatch.setattr(
            SessionLogger, "_overwrite_messages_sync", staticmethod(overwrite)
        )
        messages = [
            LLMMessage(role=Role.user, content="first"),
            LLMMessage(role=Role.assistant, content="middle"),
            LLMMessage(role=Role.user, content="last"),
        ]
        await self._save(logger, messages, mock_vibe_config, mock_tool_manager)
        await self._save(logger, messages[:2], mock_vibe_config, mock_tool_manager)
        edited = [messages[0], LLMMessage(role=Role.assistant, content="edited")]
        await self._save(logger, edited, mock_vibe_config, mock_tool_manager)

        assert len(overwrite_calls) == 3
        assert logger.session_dir is not None
        records = [
            json.loads(line)
            for line in (logger.session_dir / "messages.jsonl").read_text().splitlines()
        ]
        assert [record["content"] for record in records] == ["first", "edited"]

    @pytest.mark.asyncio
    async def test_metadata_failure_after_append_retries_without_duplicate_records(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        logger = SessionLogger(session_config, "cursor-metadata-failure")
        messages = [LLMMessage(role=Role.user, content="first")]
        await self._save(logger, messages, mock_vibe_config, mock_tool_manager)
        prior_cursor = logger._transcript_cursor
        messages.append(LLMMessage(role=Role.assistant, content="second"))
        original_persist_metadata = SessionLogger._persist_metadata_sync
        attempts = 0

        def fail_once(metadata: dict, session_dir: Path) -> None:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise OSError("injected metadata failure")
            original_persist_metadata(metadata, session_dir)

        monkeypatch.setattr(
            SessionLogger, "_persist_metadata_sync", staticmethod(fail_once)
        )
        with pytest.raises(RuntimeError, match="injected metadata failure"):
            await self._save(logger, messages, mock_vibe_config, mock_tool_manager)
        assert logger._transcript_cursor == prior_cursor

        await self._save(logger, messages, mock_vibe_config, mock_tool_manager)

        assert logger.session_dir is not None
        records = [
            json.loads(line)
            for line in (logger.session_dir / "messages.jsonl").read_text().splitlines()
        ]
        assert [record["content"] for record in records] == ["first", "second"]
        metadata = json.loads((logger.session_dir / "meta.json").read_text())
        assert metadata["total_messages"] == 2

    @pytest.mark.asyncio
    async def test_reset_and_resume_clear_cursor_without_appending_old_transcript(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
    ) -> None:
        logger = SessionLogger(session_config, "cursor-reset")
        old_messages = [LLMMessage(role=Role.user, content="old session")]
        await self._save(logger, old_messages, mock_vibe_config, mock_tool_manager)
        old_session_dir = logger.session_dir
        assert old_session_dir is not None
        old_path = old_session_dir / "messages.jsonl"
        old_content = old_path.read_text()

        logger.reset_session("new-session-id")
        assert logger._transcript_cursor is None
        new_messages = [LLMMessage(role=Role.user, content="new session")]
        await self._save(logger, new_messages, mock_vibe_config, mock_tool_manager)
        assert logger.session_dir != old_session_dir
        assert old_path.read_text() == old_content

        resumed_metadata = SessionLoader.load_metadata(old_session_dir)
        logger.apply_resumed_session("cursor-reset", old_session_dir, resumed_metadata)
        assert logger._transcript_cursor is None
        await self._save(logger, old_messages, mock_vibe_config, mock_tool_manager)
        assert old_path.read_text() == old_content

    @pytest.mark.asyncio
    async def test_resumed_logger_full_verifies_once_then_saves_incrementally(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        original_logger = SessionLogger(session_config, "cursor-resume")
        messages = [LLMMessage(role=Role.user, content="persisted")]
        await self._save(original_logger, messages, mock_vibe_config, mock_tool_manager)
        assert original_logger.session_dir is not None
        resumed_logger = SessionLogger(
            session_config, "cursor-resume", session_dir=original_logger.session_dir
        )
        original_read = SessionLogger._read_persisted_messages
        reads: list[Path] = []

        def read(messages_path: Path) -> tuple[list[dict], bool]:
            reads.append(messages_path)
            return original_read(messages_path)

        monkeypatch.setattr(
            SessionLogger, "_read_persisted_messages", staticmethod(read)
        )
        await self._save(resumed_logger, messages, mock_vibe_config, mock_tool_manager)
        assert len(reads) == 1
        messages.append(LLMMessage(role=Role.assistant, content="incremental"))
        await self._save(resumed_logger, messages, mock_vibe_config, mock_tool_manager)

        assert len(reads) == 1
        assert resumed_logger.session_dir is not None
        records = [
            json.loads(line)
            for line in (resumed_logger.session_dir / "messages.jsonl")
            .read_text()
            .splitlines()
        ]
        assert [record["content"] for record in records] == ["persisted", "incremental"]
