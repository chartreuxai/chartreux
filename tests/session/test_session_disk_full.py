from __future__ import annotations

import errno
import logging
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from chartreux.core.config import SessionLoggingConfig
from chartreux.core.llm_models import LLMMessage, Role
from chartreux.core.session.session_logger import (
    SessionDiskFullError,
    SessionLogger,
    _is_enospc,
)
from chartreux.core.session_types import AgentStats
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


async def _save(logger: SessionLogger) -> None:
    await logger.save_interaction(
        messages=[
            LLMMessage(role=Role.system, content="System prompt"),
            LLMMessage(role=Role.user, content="Hello"),
        ],
        stats=AgentStats(steps=1),
        config=build_test_vibe_config(),
        tool_manager=MagicMock(spec=ToolManager),
        agent_profile=None,
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
