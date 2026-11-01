from __future__ import annotations

import errno
import json
import logging
import os
from pathlib import Path
import stat
from unittest.mock import MagicMock

import pytest

from chartreux.core.agent_loop._transcript_integrity import TranscriptIntegrityError
from chartreux.core.config import SessionLoggingConfig
from chartreux.core.llm_models import FunctionCall, LLMMessage, Role, ToolCall
from chartreux.core.message_list import MessageList
from chartreux.core.session.session_logger import (
    SessionDiskFullError,
    SessionLogger,
    _is_enospc,
)
from chartreux.core.session_types import AgentStats, LaunchMetadataV1, LaunchPersonaV1
from chartreux.core.subagents import LaunchConfig
from chartreux.core.tools.manager import ToolManager
from tests.conftest import build_test_vibe_config

LOGGER_NAME = "chartreux.core.session.session_logger"

# Captured before any monkeypatching so tests can restore real behavior while
# os.replace is patched at the module level.
_REAL_OS_REPLACE = os.replace


@pytest.fixture
def session_config(tmp_path: Path) -> SessionLoggingConfig:
    return SessionLoggingConfig(
        save_dir=str(tmp_path / "sessions"), session_prefix="test", enabled=True
    )


def _enospc(*args, **kwargs) -> None:
    raise OSError(errno.ENOSPC, "No space left on device")


def _other_io_error(*args, **kwargs) -> None:
    raise OSError(errno.EIO, "I/O error")


def _real_replace(*args, **kwargs) -> None:
    return _REAL_OS_REPLACE(*args, **kwargs)


def _patch_replace(monkeypatch, raiser) -> None:
    monkeypatch.setattr("chartreux.core.session.session_logger.os.replace", raiser)


async def _save(
    logger: SessionLogger, messages: list[LLMMessage] | None = None
) -> None:
    await logger.save_interaction(
        messages=messages
        if messages is not None
        else [
            LLMMessage(role=Role.system, content="System prompt"),
            LLMMessage(role=Role.user, content="Hello"),
        ],
        stats=AgentStats(steps=1),
        config=build_test_vibe_config(),
        tool_manager=MagicMock(spec=ToolManager),
    )


def _disk_full_warnings(caplog) -> list:
    return [
        record for record in caplog.records if "disk is full" in record.getMessage()
    ]


class TestEnospcClassification:
    def test_detects_enospc_including_inside_exception_groups(self):
        assert _is_enospc(OSError(errno.ENOSPC, "No space left on device"))
        assert _is_enospc(ExceptionGroup("g", [OSError(errno.ENOSPC, "full")]))
        assert not _is_enospc(OSError(errno.EIO, "I/O error"))
        assert not _is_enospc(RuntimeError("not an os error"))

    def test_disk_full_error_names_path_and_condition(self):
        exc = SessionDiskFullError(
            Path("/sessions/x/meta.json"), "persist session metadata"
        )
        assert exc.code == "session_disk_full"
        assert exc.path == Path("/sessions/x/meta.json")
        assert "Disk is full" in str(exc)
        assert "/sessions/x/meta.json" in str(exc)


class TestDiskFullSaveInteraction:
    @pytest.mark.asyncio
    async def test_mkdir_enospc_save_fails_soft(
        self, session_config: SessionLoggingConfig, monkeypatch, caplog
    ) -> None:
        logger = SessionLogger(session_config, "disk-full-session")
        assert logger.session_dir is not None
        assert not logger.session_dir.exists()
        monkeypatch.setattr(Path, "mkdir", _enospc)
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            await _save(logger)

        warnings = _disk_full_warnings(caplog)
        assert len(warnings) == 1
        assert str(logger.session_dir) in warnings[0].getMessage()
        assert not logger.persisted
        assert not logger.session_dir.exists()

    @pytest.mark.asyncio
    async def test_repeated_mkdir_enospc_saves_do_not_spam(
        self, session_config: SessionLoggingConfig, monkeypatch, caplog
    ) -> None:
        logger = SessionLogger(session_config, "disk-full-session")
        monkeypatch.setattr(Path, "mkdir", _enospc)
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            for _ in range(3):
                await _save(logger)

        assert len(_disk_full_warnings(caplog)) == 1
        assert not logger.persisted

    @pytest.mark.asyncio
    async def test_mkdir_enospc_retry_does_not_duplicate_records(
        self, session_config: SessionLoggingConfig, monkeypatch, caplog
    ) -> None:
        logger = SessionLogger(session_config, "disk-full-session")
        messages = [LLMMessage(role=Role.user, content="Hello")]
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            with monkeypatch.context() as patch:
                patch.setattr(Path, "mkdir", _enospc)
                await _save(logger, messages)
                await _save(logger, messages)
            assert not logger.persisted

            await _save(logger, messages)
            await _save(logger, messages)
            assert logger.persisted
            assert logger.session_dir is not None
            records = [
                json.loads(line)
                for line in (logger.session_dir / "messages.jsonl")
                .read_text()
                .splitlines()
            ]
            assert records == [
                message.model_dump(exclude_none=True, mode="json")
                for message in messages
            ]

            # A successful retry also re-arms the warning for the next failure.
            logger.invalidate_transcript_cursor()
            with monkeypatch.context() as patch:
                patch.setattr(Path, "mkdir", _enospc)
                await _save(logger)

        assert len(_disk_full_warnings(caplog)) == 2

    @pytest.mark.asyncio
    async def test_mkdir_eacces_still_raises(
        self, session_config: SessionLoggingConfig, monkeypatch
    ) -> None:
        logger = SessionLogger(session_config, "permission-error-session")

        def permission_denied(*args, **kwargs) -> None:
            raise OSError(errno.EACCES, "Permission denied")

        monkeypatch.setattr(Path, "mkdir", permission_denied)
        with pytest.raises(
            RuntimeError, match="Failed to create session directory"
        ) as exc:
            await _save(logger)

        assert not isinstance(exc.value, SessionDiskFullError)
        assert isinstance(exc.value.__cause__, OSError)
        assert exc.value.__cause__.errno == errno.EACCES

    @pytest.mark.asyncio
    async def test_enospc_save_fails_soft_and_session_continues(
        self, session_config: SessionLoggingConfig, monkeypatch, caplog
    ) -> None:
        logger = SessionLogger(session_config, "disk-full-session")
        _patch_replace(monkeypatch, _enospc)
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            # Fail soft: a full disk must not crash the agent loop.
            await _save(logger)

        assert logger.session_dir is not None
        warnings = _disk_full_warnings(caplog)
        assert len(warnings) == 1
        assert "messages.jsonl" in warnings[0].getMessage()
        assert not logger.persisted

        # The session continues in memory: a later save (disk freed) persists.
        _patch_replace(monkeypatch, _real_replace)
        await _save(logger)
        assert logger.persisted
        assert (logger.session_dir / "messages.jsonl").exists()

    @pytest.mark.asyncio
    async def test_repeated_disk_full_saves_do_not_spam(
        self, session_config: SessionLoggingConfig, monkeypatch, caplog
    ) -> None:
        logger = SessionLogger(session_config, "disk-full-session")
        _patch_replace(monkeypatch, _enospc)
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            for _ in range(3):
                await _save(logger)

        # Emission discipline: logged once, then rate-limited.
        assert len(_disk_full_warnings(caplog)) == 1

    @pytest.mark.asyncio
    async def test_successful_save_rearms_the_disk_full_warning(
        self, session_config: SessionLoggingConfig, monkeypatch, caplog
    ) -> None:
        logger = SessionLogger(session_config, "disk-full-session")
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            _patch_replace(monkeypatch, _enospc)
            await _save(logger)
            _patch_replace(monkeypatch, _real_replace)
            await _save(logger)
            _patch_replace(monkeypatch, _enospc)
            await _save(logger)

        # State change (successful save) re-arms the rate limit: the next
        # occurrence is reported immediately instead of staying suppressed.
        assert len(_disk_full_warnings(caplog)) == 2

    @pytest.mark.asyncio
    async def test_non_enospc_persist_failure_still_raises(
        self, session_config: SessionLoggingConfig, monkeypatch
    ) -> None:
        logger = SessionLogger(session_config, "io-error-session")
        _patch_replace(monkeypatch, _other_io_error)
        # Only ENOSPC fails soft; other persist errors keep raising.
        with pytest.raises(RuntimeError, match="Failed to save session"):
            await _save(logger)


class TestLaunchConfigDirectoryFailure:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("error_number", [errno.ENOSPC, errno.EACCES])
    async def test_launch_config_mkdir_errors_remain_observable(
        self, session_config: SessionLoggingConfig, monkeypatch, error_number: int
    ) -> None:
        logger = SessionLogger(session_config, "launch-config-session")
        launch_config = LaunchMetadataV1(
            version=1,
            profile="test-agent",
            overrides=LaunchConfig(),
            persona=LaunchPersonaV1(system_prompt_id="tests", instructions=None),
        )

        def mkdir_failure(*args, **kwargs) -> None:
            raise OSError(error_number, "mkdir failed")

        monkeypatch.setattr(Path, "mkdir", mkdir_failure)
        expected_error = (
            SessionDiskFullError if error_number == errno.ENOSPC else OSError
        )
        with pytest.raises(expected_error) as exc:
            await logger.persist_launch_config(launch_config)

        if error_number == errno.ENOSPC:
            assert isinstance(exc.value, SessionDiskFullError)
            assert exc.value.path == logger.session_dir
            assert exc.value.operation == "create session directory"
            assert isinstance(exc.value.__cause__, OSError)
            assert exc.value.__cause__.errno == errno.ENOSPC
        else:
            assert isinstance(exc.value, OSError)
            assert exc.value.errno == errno.EACCES
        assert not logger.persisted
        assert logger.session_metadata is not None
        assert logger.session_metadata.launch_config == launch_config
        assert logger._launch_config_dirty


class TestQuarantineDiskFull:
    def test_quarantine_enospc_raises_distinct_error_and_keeps_original(
        self, tmp_path: Path, monkeypatch, caplog
    ) -> None:
        messages_path = tmp_path / "messages.jsonl"
        messages_path.write_text("not valid json\n")
        _patch_replace(monkeypatch, _enospc)
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            with pytest.raises(SessionDiskFullError) as exc_info:
                SessionLogger._quarantine_corrupt_transcript(messages_path)

        assert exc_info.value.path == messages_path
        # The original OSError is chained, not swallowed.
        assert isinstance(exc_info.value.__cause__, OSError)
        assert exc_info.value.__cause__.errno == errno.ENOSPC
        # The corruption (the reason quarantine was attempted) is still reported.
        assert any("corrupted" in record.getMessage() for record in caplog.records)

    @pytest.mark.asyncio
    async def test_corrupt_transcript_quarantine_enospc_does_not_mask_corruption(
        self, session_config: SessionLoggingConfig, monkeypatch, caplog
    ) -> None:
        logger = SessionLogger(session_config, "corrupt-session")
        assert logger.session_dir is not None
        messages_path = logger.session_dir / "messages.jsonl"
        messages_path.parent.mkdir(parents=True, exist_ok=True)
        messages_path.write_text("not valid json\n")

        _patch_replace(monkeypatch, _enospc)
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            # Fail soft overall, but both the corruption and the disk-full
            # condition are surfaced.
            await _save(logger)

        assert any("corrupted" in record.getMessage() for record in caplog.records)
        assert _disk_full_warnings(caplog)


class TestRepairRewriteDiskFull:
    def test_recovery_artifact_enospc_fails_closed_and_keeps_the_original(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        messages_path = tmp_path / "messages.jsonl"
        original_bytes = b'{"role": "user", "content": "kept"}\n'
        messages_path.write_bytes(original_bytes)
        _patch_replace(monkeypatch, _enospc)

        with pytest.raises(TranscriptIntegrityError) as excinfo:
            SessionLogger.migrate_repaired_transcript(
                messages_path, [LLMMessage(role=Role.user, content="repaired")]
            )

        assert excinfo.value.violations[0].kind == "transcript_repair_failed"
        assert isinstance(excinfo.value.__cause__, OSError)
        assert excinfo.value.__cause__.errno == errno.ENOSPC
        # Fail closed: the live transcript is intact and no partial is visible.
        assert messages_path.read_bytes() == original_bytes
        assert list(tmp_path.iterdir()) == [messages_path]

    @pytest.mark.asyncio
    async def test_streamed_repair_rewrite_enospc_leaves_no_partial_transcript(
        self, session_config: SessionLoggingConfig, monkeypatch, caplog
    ) -> None:
        logger = SessionLogger(session_config, "repair-disk-full")
        assert logger.session_dir is not None
        session_dir = logger.session_dir
        messages_path = session_dir / "messages.jsonl"
        session_dir.mkdir(parents=True, exist_ok=True)
        corrupt_bytes = b'{"role": "user", "content": "kept"}\n{"corrupt": true\n'
        messages_path.write_bytes(corrupt_bytes)

        original_fsync = os.fsync

        def fsync(descriptor: int) -> None:
            # Deferred allocation reports ENOSPC at the sync of the streamed
            # rewrite's temp file, before it can be renamed into place.
            if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
                _enospc()
            original_fsync(descriptor)

        monkeypatch.setattr(os, "fsync", fsync)
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            await _save(logger, [LLMMessage(role=Role.user, content="kept")])

        assert not messages_path.exists()
        quarantined = list(session_dir.glob("messages.jsonl.corrupt-*"))
        assert len(quarantined) == 1
        assert quarantined[0].read_bytes() == corrupt_bytes
        # The temp file is gone: nothing partial is published or left behind.
        assert [path.name for path in session_dir.iterdir()] == [quarantined[0].name]
        assert _disk_full_warnings(caplog)
        assert not logger.persisted


def _seal_messages() -> MessageList:
    """A settled round, a compaction boundary, and a live tail."""
    return MessageList([
        LLMMessage(role=Role.user, content="u1"),
        LLMMessage(
            role=Role.assistant,
            content="working",
            tool_calls=[
                ToolCall(
                    id="c1",
                    index=0,
                    function=FunctionCall(name="read_file", arguments="{}"),
                )
            ],
        ),
        LLMMessage(
            role=Role.tool, tool_call_id="c1", name="read_file", content="result for c1"
        ),
        LLMMessage(
            role=Role.user,
            content="compaction boundary",
            injected=True,
            context_boundary="compaction",
        ),
        LLMMessage(role=Role.user, content="u2"),
        LLMMessage(role=Role.assistant, content="done"),
    ])


async def _seal(logger: SessionLogger, messages: MessageList) -> None:
    await logger.seal_archive(
        messages,
        AgentStats(steps=1),
        build_test_vibe_config(),
        MagicMock(spec=ToolManager),
    )


class TestSealDiskFullNoReceipt:
    """P4a2: the sealing operation closes the disk-full authorization gap.

    A returned (fail-soft) save call is NOT authorization: sealing runs its
    own durable persist, SessionDiskFullError propagates (never
    caught-and-continued on this path), and NO receipt is issued or retained.
    """

    @pytest.mark.asyncio
    async def test_seal_enospc_propagates_and_issues_no_receipt(
        self, session_config: SessionLoggingConfig, monkeypatch, caplog
    ) -> None:
        logger = SessionLogger(session_config, "seal-disk-full")
        _patch_replace(monkeypatch, _enospc)
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            with pytest.raises(SessionDiskFullError):
                await _seal(logger, _seal_messages())

        assert logger.archive_descriptor is None
        assert logger.last_save_archive_snapshot is None
        assert not logger.persisted

    @pytest.mark.asyncio
    async def test_fail_soft_save_return_is_not_sealing_authorization(
        self, session_config: SessionLoggingConfig, monkeypatch, caplog
    ) -> None:
        logger = SessionLogger(session_config, "seal-after-fail-soft")
        _patch_replace(monkeypatch, _enospc)
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            # The ordinary save fails soft and returns normally...
            await _save(logger)
            assert not logger.persisted
            # ...but that returned save call is NOT authorization: sealing
            # still runs its own persist, which raises, and no receipt exists.
            with pytest.raises(SessionDiskFullError):
                await _seal(logger, _seal_messages())

        assert logger.archive_descriptor is None
        assert not logger.persisted

    @pytest.mark.asyncio
    async def test_seal_mkdir_enospc_issues_no_receipt(
        self, session_config: SessionLoggingConfig, monkeypatch
    ) -> None:
        logger = SessionLogger(session_config, "seal-mkdir-full")
        monkeypatch.setattr(Path, "mkdir", _enospc)
        with pytest.raises(SessionDiskFullError):
            await _seal(logger, _seal_messages())

        assert logger.archive_descriptor is None
        assert not logger.persisted
