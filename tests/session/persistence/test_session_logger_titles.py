"""Session logger title management tests (set_title manual and auto transitions, the initial auto title, needs_initial_auto_title, title_source, the async auto refresh, and the manual-title application with its read and write failure and race paths)."""

from __future__ import annotations

import asyncio

import pytest

from chartreux.core.config import SessionLoggingConfig
from chartreux.core.llm_models import LLMMessage, Role
from chartreux.core.session.session_loader import SessionLoader
from chartreux.core.session.session_logger import SessionLogger
from chartreux.core.session_types import AgentStats
from chartreux.core.tools.manager import ToolManager
from tests.conftest import build_test_vibe_config
from tests.session.persistence._session_persistence_support import (
    disabled_session_config as disabled_session_config,
    mock_tool_manager as mock_tool_manager,
    session_config as session_config,
    temp_session_dir as temp_session_dir,
)


class TestSessionLoggerTitleManagement:
    def test_set_title_marks_live_session_title_as_manual(
        self, session_config: SessionLoggingConfig
    ) -> None:
        logger = SessionLogger(session_config, "test-session-123")

        logger._set_title("Manual title")

        assert logger.session_metadata is not None
        assert logger.session_metadata.title == "Manual title"
        assert logger.session_metadata.title_source == "manual"

    def test_set_title_none_returns_live_session_to_auto_mode(
        self, session_config: SessionLoggingConfig
    ) -> None:
        logger = SessionLogger(session_config, "test-session-123")
        logger._set_title("Manual title")

        logger._set_title(None)

        assert logger.session_metadata is not None
        assert logger.session_metadata.title is None
        assert logger.session_metadata.title_source == "auto"

    def test_set_title_rejects_empty_title(
        self, session_config: SessionLoggingConfig
    ) -> None:
        logger = SessionLogger(session_config, "test-session-123")

        with pytest.raises(ValueError, match="Session title cannot be empty."):
            logger._set_title("   ")

    def test_set_title_preserves_live_session_end_time(
        self, session_config: SessionLoggingConfig
    ) -> None:
        logger = SessionLogger(session_config, "test-session-123")
        assert logger.session_metadata is not None
        logger.session_metadata.end_time = "2026-01-01T10:00:00+00:00"

        logger._set_title("Manual title")

        assert logger.session_metadata.end_time == "2026-01-01T10:00:00+00:00"

    def test_set_initial_auto_title_applies_when_no_title_set(
        self, session_config: SessionLoggingConfig
    ) -> None:
        logger = SessionLogger(session_config, "test-session-123")

        applied = logger.set_initial_auto_title("Pretty title")

        assert applied is True
        assert logger.session_metadata is not None
        assert logger.session_metadata.title == "Pretty title"
        assert logger.session_metadata.title_source == "auto"

    def test_set_initial_auto_title_noop_when_title_already_set(
        self, session_config: SessionLoggingConfig
    ) -> None:
        logger = SessionLogger(session_config, "test-session-123")
        logger._set_title("Manual title")

        applied = logger.set_initial_auto_title("Pretty title")

        assert applied is False
        assert logger.session_metadata is not None
        assert logger.session_metadata.title == "Manual title"
        assert logger.session_metadata.title_source == "manual"

    def test_set_initial_auto_title_noop_when_prior_auto_title_set(
        self, session_config: SessionLoggingConfig
    ) -> None:
        logger = SessionLogger(session_config, "test-session-123")
        logger.set_initial_auto_title("First title")

        applied = logger.set_initial_auto_title("Second title")

        assert applied is False
        assert logger.session_metadata is not None
        assert logger.session_metadata.title == "First title"

    def test_set_initial_auto_title_rejects_blank(
        self, session_config: SessionLoggingConfig
    ) -> None:
        logger = SessionLogger(session_config, "test-session-123")

        applied = logger.set_initial_auto_title("   ")

        assert applied is False
        assert logger.session_metadata is not None
        assert logger.session_metadata.title is None

    def test_needs_initial_auto_title_true_when_no_title(
        self, session_config: SessionLoggingConfig
    ) -> None:
        logger = SessionLogger(session_config, "test-session-123")

        assert logger.needs_initial_auto_title() is True

    def test_needs_initial_auto_title_false_after_set_initial_auto_title(
        self, session_config: SessionLoggingConfig
    ) -> None:
        logger = SessionLogger(session_config, "test-session-123")
        logger.set_initial_auto_title("Pretty title")

        assert logger.needs_initial_auto_title() is False

    def test_needs_initial_auto_title_false_after_manual_set_title(
        self, session_config: SessionLoggingConfig
    ) -> None:
        logger = SessionLogger(session_config, "test-session-123")
        logger._set_title("Manual title")

        assert logger.needs_initial_auto_title() is False

    def test_needs_initial_auto_title_true_when_disabled(
        self, disabled_session_config: SessionLoggingConfig
    ) -> None:
        logger = SessionLogger(disabled_session_config, "test-session-123")

        assert logger.needs_initial_auto_title() is True

    def test_title_source_defaults_to_auto(
        self, session_config: SessionLoggingConfig
    ) -> None:
        logger = SessionLogger(session_config, "test-session-123")

        assert logger.title_source == "auto"

    def test_title_source_auto_when_disabled(
        self, disabled_session_config: SessionLoggingConfig
    ) -> None:
        logger = SessionLogger(disabled_session_config, "test-session-123")

        assert logger.title_source == "auto"

    @pytest.mark.asyncio
    async def test_refresh_auto_title_overwrites_prior_auto_title(
        self, session_config: SessionLoggingConfig
    ) -> None:
        logger = SessionLogger(session_config, "test-session-123")
        logger.set_initial_auto_title("First message based title")

        changed = await logger.refresh_auto_title("Concise generated title")

        assert changed is True
        assert logger.title == "Concise generated title"
        assert logger.title_source == "auto"

    @pytest.mark.asyncio
    async def test_refresh_auto_title_never_overrides_manual(
        self, session_config: SessionLoggingConfig
    ) -> None:
        logger = SessionLogger(session_config, "test-session-123")
        logger._set_title("Manual title")

        changed = await logger.refresh_auto_title("Generated title")

        assert changed is False
        assert logger.title == "Manual title"
        assert logger.title_source == "manual"

    @pytest.mark.asyncio
    async def test_refresh_auto_title_noop_when_unchanged(
        self, session_config: SessionLoggingConfig
    ) -> None:
        logger = SessionLogger(session_config, "test-session-123")
        logger.set_initial_auto_title("Same title")

        assert await logger.refresh_auto_title("Same title") is False

    @pytest.mark.asyncio
    async def test_refresh_auto_title_noop_when_blank(
        self, session_config: SessionLoggingConfig
    ) -> None:
        logger = SessionLogger(session_config, "test-session-123")
        logger.set_initial_auto_title("Existing")

        assert await logger.refresh_auto_title("   ") is False
        assert logger.title == "Existing"

    @pytest.mark.asyncio
    async def test_refresh_auto_title_persists_to_disk(
        self, session_config: SessionLoggingConfig, mock_tool_manager: ToolManager
    ) -> None:
        logger = SessionLogger(session_config, "test-session-123")
        await logger.save_interaction(
            [LLMMessage(role=Role.user, content="hi")],
            stats=AgentStats(),
            config=build_test_vibe_config(),
            tool_manager=mock_tool_manager,
        )

        await logger.refresh_auto_title("Persisted generated title")

        assert logger.session_dir is not None
        metadata = SessionLoader.load_metadata(logger.session_dir)
        assert metadata.title == "Persisted generated title"
        assert metadata.title_source == "auto"

    @pytest.mark.asyncio
    async def test_refresh_auto_title_skips_a_title_for_a_reset_session(
        self, session_config: SessionLoggingConfig, mock_tool_manager: ToolManager
    ) -> None:
        # A background generation that outlived a /new or /clear reset carries the
        # old session id; the logger (reset in place) must not persist it.
        logger = SessionLogger(session_config, "test-session-123")
        logger.set_initial_auto_title("Auto title")
        await logger.save_interaction(
            [LLMMessage(role=Role.user, content="hi")],
            stats=AgentStats(),
            config=build_test_vibe_config(),
            tool_manager=mock_tool_manager,
        )

        changed = await logger.refresh_auto_title(
            "Stale title", expected_session_id="a-different-session"
        )

        assert changed is False
        assert logger.title == "Auto title"
        assert logger.session_dir is not None
        metadata = SessionLoader.load_metadata(logger.session_dir)
        assert metadata.title == "Auto title"

    @pytest.mark.asyncio
    async def test_refresh_auto_title_keeps_state_consistent_on_read_failure(
        self,
        session_config: SessionLoggingConfig,
        mock_tool_manager: ToolManager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        logger = SessionLogger(session_config, "test-session-123")
        logger.set_initial_auto_title("Auto title")
        await logger.save_interaction(
            [LLMMessage(role=Role.user, content="hi")],
            stats=AgentStats(),
            config=build_test_vibe_config(),
            tool_manager=mock_tool_manager,
        )

        async def boom(*_args, **_kwargs):
            raise OSError("disk gone")

        monkeypatch.setattr(
            "chartreux.core.session.session_logger.read_safe_async", boom
        )

        with pytest.raises(RuntimeError, match="Failed to read session metadata"):
            await logger.refresh_auto_title("Generated title")

        # The flip never landed, so memory and disk stay consistent.
        assert logger.title == "Auto title"
        assert logger.title_source == "auto"
        assert logger.session_dir is not None
        metadata = SessionLoader.load_metadata(logger.session_dir)
        assert metadata.title == "Auto title"
        assert metadata.title_source == "auto"

    @pytest.mark.asyncio
    async def test_save_interaction_keeps_refreshed_title(
        self, session_config: SessionLoggingConfig, mock_tool_manager: ToolManager
    ) -> None:
        logger = SessionLogger(session_config, "test-session-123")
        await logger.save_interaction(
            [LLMMessage(role=Role.user, content="hi")],
            stats=AgentStats(),
            config=build_test_vibe_config(),
            tool_manager=mock_tool_manager,
        )
        await logger.refresh_auto_title("Generated title")

        await logger.save_interaction(
            [
                LLMMessage(role=Role.user, content="hi"),
                LLMMessage(role=Role.assistant, content="hello"),
            ],
            stats=AgentStats(),
            config=build_test_vibe_config(),
            tool_manager=mock_tool_manager,
        )

        assert logger.session_dir is not None
        metadata = SessionLoader.load_metadata(logger.session_dir)
        assert metadata.title == "Generated title"
        assert metadata.title_source == "auto"

    @pytest.mark.asyncio
    async def test_refresh_auto_title_rechecks_manual_under_lock(
        self, session_config: SessionLoggingConfig
    ) -> None:
        logger = SessionLogger(session_config, "test-session-123")
        logger.set_initial_auto_title("Auto title")

        # Hold the lock so the refresh blocks past its outer guard, then rename
        # before releasing: the refresh must re-check and leave the manual title.
        await logger._save_lock.acquire()
        refresh = asyncio.create_task(logger.refresh_auto_title("Generated title"))
        await asyncio.sleep(0)
        logger._set_title("Manual rename")
        logger._save_lock.release()

        assert await refresh is False
        assert logger.title == "Manual rename"
        assert logger.title_source == "manual"

    @pytest.mark.asyncio
    async def test_apply_manual_title_persists_manual_to_disk(
        self, session_config: SessionLoggingConfig, mock_tool_manager: ToolManager
    ) -> None:
        logger = SessionLogger(session_config, "test-session-123")
        await logger.save_interaction(
            [LLMMessage(role=Role.user, content="hi")],
            stats=AgentStats(),
            config=build_test_vibe_config(),
            tool_manager=mock_tool_manager,
        )

        updated_at = await logger.apply_manual_title("  Reviewed session  ")

        assert isinstance(updated_at, str)
        assert logger.title == "Reviewed session"
        assert logger.title_source == "manual"
        assert logger.session_dir is not None
        metadata = SessionLoader.load_metadata(logger.session_dir)
        assert metadata.title == "Reviewed session"
        assert metadata.title_source == "manual"

    @pytest.mark.asyncio
    async def test_apply_manual_title_rejects_empty(
        self, session_config: SessionLoggingConfig
    ) -> None:
        logger = SessionLogger(session_config, "test-session-123")

        with pytest.raises(ValueError, match="cannot be empty"):
            await logger.apply_manual_title("   ")

    @pytest.mark.asyncio
    async def test_apply_manual_title_blocks_racing_auto_refresh(
        self, session_config: SessionLoggingConfig
    ) -> None:
        logger = SessionLogger(session_config, "test-session-123")
        logger.set_initial_auto_title("Auto title")

        # Rename flips memory to manual under the lock, so a later refresh bails.
        await logger.apply_manual_title("Manual rename")

        assert await logger.refresh_auto_title("Generated title") is False
        assert logger.title == "Manual rename"
        assert logger.title_source == "manual"

    @pytest.mark.asyncio
    async def test_apply_manual_title_keeps_state_consistent_on_read_failure(
        self,
        session_config: SessionLoggingConfig,
        mock_tool_manager: ToolManager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        logger = SessionLogger(session_config, "test-session-123")
        logger.set_initial_auto_title("Auto title")
        await logger.save_interaction(
            [LLMMessage(role=Role.user, content="hi")],
            stats=AgentStats(),
            config=build_test_vibe_config(),
            tool_manager=mock_tool_manager,
        )

        async def boom(*_args, **_kwargs):
            raise OSError("disk gone")

        monkeypatch.setattr(
            "chartreux.core.session.session_logger.read_safe_async", boom
        )

        with pytest.raises(RuntimeError, match="Failed to read session metadata"):
            await logger.apply_manual_title("Manual rename")

        # The flip never landed, so memory and disk stay consistent (auto).
        assert logger.title == "Auto title"
        assert logger.title_source == "auto"
        assert logger.session_dir is not None
        metadata = SessionLoader.load_metadata(logger.session_dir)
        assert metadata.title == "Auto title"
        assert metadata.title_source == "auto"

    @pytest.mark.asyncio
    async def test_apply_manual_title_keeps_state_consistent_on_write_failure(
        self,
        session_config: SessionLoggingConfig,
        mock_tool_manager: ToolManager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        logger = SessionLogger(session_config, "test-session-123")
        logger.set_initial_auto_title("Auto title")
        await logger.save_interaction(
            [LLMMessage(role=Role.user, content="hi")],
            stats=AgentStats(),
            config=build_test_vibe_config(),
            tool_manager=mock_tool_manager,
        )

        async def boom(*_args, **_kwargs):
            raise RuntimeError("disk full")

        monkeypatch.setattr(
            "chartreux.core.session.session_logger.SessionLogger.persist_metadata", boom
        )

        with pytest.raises(RuntimeError, match="disk full"):
            await logger.apply_manual_title("Manual rename")

        # The write failed before the flip, so memory stays auto (not manual),
        # leaving auto-refresh unblocked and memory consistent with disk.
        assert logger.title == "Auto title"
        assert logger.title_source == "auto"

    @pytest.mark.asyncio
    async def test_concurrent_rename_and_auto_refresh_leaves_manual_title(
        self, session_config: SessionLoggingConfig, mock_tool_manager: ToolManager
    ) -> None:
        # The two-writer invariant: a manual rename always wins a race with the
        # background auto-refresh, on disk and in memory, whichever runs first.
        logger = SessionLogger(session_config, "test-session-123")
        logger.set_initial_auto_title("Auto title")
        await logger.save_interaction(
            [LLMMessage(role=Role.user, content="hi")],
            stats=AgentStats(),
            config=build_test_vibe_config(),
            tool_manager=mock_tool_manager,
        )

        await asyncio.gather(
            logger.refresh_auto_title("Generated title"),
            logger.apply_manual_title("Manual rename"),
        )

        assert logger.title == "Manual rename"
        assert logger.title_source == "manual"
        assert logger.session_dir is not None
        metadata = SessionLoader.load_metadata(logger.session_dir)
        assert metadata.title == "Manual rename"
        assert metadata.title_source == "manual"
        # A later auto-refresh must still bail against the manual title.
        assert await logger.refresh_auto_title("Another title") is False
