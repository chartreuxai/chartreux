"""Session logger save interaction tests (transcript rewrite/shrink/regrow, corrupt-transcript quarantine, no-op and empty-conversation saves, tmp-file cleanup and its throttling, and the disk-full soft-failure matrix)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from chartreux.core.agents.models import AgentProfile, AgentSafety
from chartreux.core.config import ChartreuxConfigSchema, SessionLoggingConfig
from chartreux.core.llm_models import LLMMessage, Role
from chartreux.core.session.session_loader import SessionLoader
from chartreux.core.session.session_logger import SessionLogger
from chartreux.core.session_types import AgentStats
from chartreux.core.tools.manager import ToolManager
from tests.conftest import build_test_vibe_config
from tests.session.persistence._session_persistence_support import (
    disabled_session_config as disabled_session_config,
    mock_agent_profile as mock_agent_profile,
    mock_tool_manager as mock_tool_manager,
    mock_vibe_config as mock_vibe_config,
    session_config as session_config,
    temp_session_dir as temp_session_dir,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stage", ["write", "flush", "file_fsync", "replace", "directory_fsync"]
)
async def test_default_disk_full_is_soft(
    session_config: SessionLoggingConfig,
    mock_vibe_config: ChartreuxConfigSchema,
    mock_tool_manager: ToolManager,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    import errno
    import stat
    import tempfile

    logger = SessionLogger(session_config, "disk-full")

    def fail(*args: Any, **kwargs: Any) -> None:
        raise OSError(errno.ENOSPC, "injected full disk")

    if stage in {"write", "flush"}:
        original_temp = tempfile.NamedTemporaryFile

        def failing_temp(**kwargs: Any) -> Any:
            stream = original_temp(**kwargs)
            proxy = MagicMock(wraps=stream)
            proxy.name = stream.name
            proxy.__enter__.return_value = proxy
            proxy.__exit__.side_effect = stream.__exit__
            getattr(proxy, stage).side_effect = fail
            return proxy

        monkeypatch.setattr(tempfile, "NamedTemporaryFile", failing_temp)
    elif stage == "replace":
        monkeypatch.setattr(os, "replace", fail)
    else:
        original_fsync = os.fsync

        def fsync(fd: int) -> None:
            is_dir = stat.S_ISDIR(os.fstat(fd).st_mode)
            if is_dir == (stage == "directory_fsync"):
                fail()
            original_fsync(fd)

        monkeypatch.setattr(os, "fsync", fsync)
    save = logger.save_interaction(
        [LLMMessage(role=Role.user, content="test")],
        AgentStats(),
        mock_vibe_config,
        mock_tool_manager,
        None,
    )
    await save
    assert not logger.persisted


class TestSessionLoggerSaveInteraction:
    @pytest.mark.asyncio
    async def test_save_interaction_disabled(
        self, disabled_session_config: SessionLoggingConfig
    ) -> None:
        """Test that save_interaction returns None when logging is disabled."""
        logger = SessionLogger(disabled_session_config, "test-session")

        result = await logger.save_interaction(
            messages=[],
            stats=AgentStats(),
            config=build_test_vibe_config(),
            tool_manager=MagicMock(),
            agent_profile=AgentProfile(
                name="test",
                display_name="Test",
                description="Test agent",
                safety=AgentSafety.NEUTRAL,
                overrides={},
            ),
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_save_interaction_success(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
        mock_agent_profile: AgentProfile,
    ) -> None:
        """Test that save_interaction successfully saves session data."""
        session_id = "test-session-123"
        logger = SessionLogger(session_config, session_id)

        # Create test messages
        messages = [
            LLMMessage(role=Role.system, content="System prompt"),
            LLMMessage(role=Role.user, content="Hello"),
            LLMMessage(role=Role.assistant, content="Hi there!"),
        ]

        # Create test stats
        stats = AgentStats(
            steps=1, session_prompt_tokens=10, session_completion_tokens=20
        )

        await logger.save_interaction(
            messages=messages,
            stats=stats,
            config=mock_vibe_config,
            tool_manager=mock_tool_manager,
            agent_profile=mock_agent_profile,
        )

        # Verify behavior via file system
        assert logger.session_dir is not None
        messages_file = logger.session_dir / "messages.jsonl"
        metadata_file = logger.session_dir / "meta.json"

        assert messages_file.exists()
        assert metadata_file.exists()

        with open(metadata_file) as f:
            metadata = json.load(f)
            assert metadata["session_id"] == session_id
            assert metadata["total_messages"] == 2
            assert metadata["stats"]["steps"] == stats.steps
            # No title until the background model generates one.
            assert metadata["title"] is None
            assert metadata["title_source"] == "auto"
            assert "system_prompt" in metadata

    @pytest.mark.asyncio
    async def test_save_interaction_system_prompt_in_metadata(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
        mock_agent_profile: AgentProfile,
    ) -> None:
        """Test that system prompt is saved in metadata and not in messages."""
        session_id = "test-session-123"
        logger = SessionLogger(session_config, session_id)

        messages = [
            LLMMessage(role=Role.system, content="System prompt"),
            LLMMessage(role=Role.user, content="Hello"),
            LLMMessage(role=Role.assistant, content="Hi there!"),
        ]

        stats = AgentStats(
            steps=1, session_prompt_tokens=10, session_completion_tokens=20
        )

        await logger.save_interaction(
            messages=messages,
            stats=stats,
            config=mock_vibe_config,
            tool_manager=mock_tool_manager,
            agent_profile=mock_agent_profile,
        )

        assert logger.session_dir is not None
        metadata_file = logger.session_dir / "meta.json"
        assert metadata_file.exists()
        with open(metadata_file) as f:
            metadata = json.load(f)
            assert "system_prompt" in metadata
            assert metadata["system_prompt"]["content"] == "System prompt"
            assert metadata["system_prompt"]["role"] == "system"

        messages_file = logger.session_dir / "messages.jsonl"
        assert messages_file.exists()
        with open(messages_file) as f:
            lines = f.readlines()
            messages_data = [json.loads(line) for line in lines]

            assert len(messages_data) == 2
            assert messages_data[0]["role"] == "user"
            assert messages_data[1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_save_interaction_with_existing_messages(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
        mock_agent_profile: AgentProfile,
    ) -> None:
        """Test that save_interaction correctly handles existing messages."""
        session_id = "test-session-123"
        logger = SessionLogger(session_config, session_id)

        # First save - create initial session
        initial_messages = [
            LLMMessage(role=Role.system, content="System prompt"),
            LLMMessage(role=Role.user, content="Hello"),
            LLMMessage(role=Role.assistant, content="Hi there!"),
        ]

        stats = AgentStats(
            steps=1, session_prompt_tokens=10, session_completion_tokens=20
        )

        await logger.save_interaction(
            messages=initial_messages,
            stats=stats,
            config=mock_vibe_config,
            tool_manager=mock_tool_manager,
            agent_profile=mock_agent_profile,
        )

        # Second save - add more messages
        new_messages = [
            LLMMessage(role=Role.user, content="How are you?"),
            LLMMessage(role=Role.assistant, content="I'm fine, thanks!"),
        ]

        all_messages = initial_messages + new_messages
        updated_stats = AgentStats(
            steps=2, session_prompt_tokens=20, session_completion_tokens=40
        )

        await logger.save_interaction(
            messages=all_messages,
            stats=updated_stats,
            config=mock_vibe_config,
            tool_manager=mock_tool_manager,
            agent_profile=mock_agent_profile,
        )

        # Verify behavior via file system: metadata was updated
        assert logger.session_dir is not None
        metadata_file = logger.session_dir / "meta.json"
        assert metadata_file.exists()
        with open(metadata_file) as f:
            metadata = json.load(f)
            assert metadata["total_messages"] == 4
            assert metadata["stats"]["steps"] == updated_stats.steps

        messages_file = logger.session_dir / "messages.jsonl"
        assert messages_file.exists()
        with open(messages_file) as f:
            lines = f.readlines()
            assert len(lines) == 4

    @pytest.mark.asyncio
    async def test_save_interaction_no_new_messages_is_noop(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
        mock_agent_profile: AgentProfile,
    ) -> None:
        """Test that save_interaction does nothing when there are no new messages."""
        session_id = "test-session-123"
        logger = SessionLogger(session_config, session_id)

        messages = [
            LLMMessage(role=Role.system, content="System prompt"),
            LLMMessage(role=Role.user, content="Hello"),
            LLMMessage(role=Role.assistant, content="Hi there!"),
        ]
        stats = AgentStats(
            steps=1, session_prompt_tokens=10, session_completion_tokens=20
        )

        await logger.save_interaction(
            messages=messages,
            stats=stats,
            config=mock_vibe_config,
            tool_manager=mock_tool_manager,
            agent_profile=mock_agent_profile,
        )

        assert logger.session_dir is not None
        metadata_file = logger.session_dir / "meta.json"
        messages_file = logger.session_dir / "messages.jsonl"

        with open(metadata_file) as f:
            meta_before = json.load(f)
        with open(messages_file) as f:
            lines_before = f.readlines()

        # Call again with same messages: no new messages, should be no-op
        await logger.save_interaction(
            messages=messages,
            stats=stats,
            config=mock_vibe_config,
            tool_manager=mock_tool_manager,
            agent_profile=mock_agent_profile,
        )

        with open(metadata_file) as f:
            meta_after = json.load(f)
        with open(messages_file) as f:
            lines_after = f.readlines()

        assert len(lines_after) == len(lines_before) == 2
        assert lines_after == lines_before
        assert meta_after["total_messages"] == meta_before["total_messages"] == 2
        assert meta_after == meta_before

    @pytest.mark.asyncio
    async def test_save_interaction_no_user_messages(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
        mock_agent_profile: AgentProfile,
    ) -> None:
        """Test that save_interaction handles sessions with no user messages."""
        session_id = "test-session-123"
        logger = SessionLogger(session_config, session_id)

        # Create messages with no user messages (only system and assistant)
        messages = [
            LLMMessage(role=Role.system, content="System prompt"),
            LLMMessage(role=Role.assistant, content="Hi there!"),
        ]

        stats = AgentStats(
            steps=1, session_prompt_tokens=10, session_completion_tokens=20
        )

        await logger.save_interaction(
            messages=messages,
            stats=stats,
            config=mock_vibe_config,
            tool_manager=mock_tool_manager,
            agent_profile=mock_agent_profile,
        )

        # Verify behavior via file system
        assert logger.session_dir is not None
        metadata_file = logger.session_dir / "meta.json"
        assert metadata_file.exists()
        with open(metadata_file) as f:
            metadata = json.load(f)
            assert metadata["session_id"] == session_id
            assert metadata["total_messages"] == 1
            assert metadata["stats"]["steps"] == stats.steps
            assert metadata["title"] is None
            assert metadata["title_source"] == "auto"

        messages_file = logger.session_dir / "messages.jsonl"
        assert messages_file.exists()
        with open(messages_file) as f:
            assert len(f.readlines()) == 1

    @pytest.mark.asyncio
    async def test_save_interaction_preserves_preset_auto_title(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
        mock_agent_profile: AgentProfile,
    ) -> None:
        session_id = "test-session-123"
        logger = SessionLogger(session_config, session_id)
        assert logger.session_metadata is not None

        logger.set_initial_auto_title("Pretty @foo.py title")

        messages = [
            LLMMessage(role=Role.system, content="System prompt"),
            LLMMessage(
                role=Role.user, content="path: file:///abs/foo.py\ncontent: ..."
            ),
            LLMMessage(role=Role.assistant, content="Hi there!"),
        ]
        stats = AgentStats(
            steps=1, session_prompt_tokens=10, session_completion_tokens=20
        )

        await logger.save_interaction(
            messages=messages,
            stats=stats,
            config=mock_vibe_config,
            tool_manager=mock_tool_manager,
            agent_profile=mock_agent_profile,
        )

        assert logger.session_dir is not None
        metadata_file = logger.session_dir / "meta.json"
        with open(metadata_file) as f:
            metadata = json.load(f)

        assert metadata["title"] == "Pretty @foo.py title"
        assert metadata["title_source"] == "auto"

    @pytest.mark.asyncio
    async def test_save_interaction_preserves_manual_title(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
        mock_agent_profile: AgentProfile,
    ) -> None:
        session_id = "test-session-123"
        logger = SessionLogger(session_config, session_id)
        assert logger.session_metadata is not None

        logger._set_title("Manual title")

        messages = [
            LLMMessage(role=Role.system, content="System prompt"),
            LLMMessage(role=Role.user, content="Hello"),
            LLMMessage(role=Role.assistant, content="Hi there!"),
        ]
        stats = AgentStats(
            steps=1, session_prompt_tokens=10, session_completion_tokens=20
        )

        await logger.save_interaction(
            messages=messages,
            stats=stats,
            config=mock_vibe_config,
            tool_manager=mock_tool_manager,
            agent_profile=mock_agent_profile,
        )

        assert logger.session_dir is not None
        metadata_file = logger.session_dir / "meta.json"
        with open(metadata_file) as f:
            metadata = json.load(f)

        assert metadata["title"] == "Manual title"
        assert metadata["title_source"] == "manual"

        messages_file = logger.session_dir / "messages.jsonl"
        assert messages_file.exists()
        with open(messages_file) as f:
            assert len(f.readlines()) == 2

    @pytest.mark.asyncio
    async def test_save_interaction_throttles_tmp_cleanup(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
        mock_agent_profile: AgentProfile,
    ) -> None:
        logger = SessionLogger(session_config, "test-session-123")

        messages = [
            LLMMessage(role=Role.system, content="System prompt"),
            LLMMessage(role=Role.user, content="Hello"),
            LLMMessage(role=Role.assistant, content="Hi there!"),
        ]

        cleanup_spy = MagicMock()
        with (
            patch.object(
                SessionLogger, "_overwrite_messages_sync"
            ) as persist_messages_mock,
            patch.object(
                SessionLogger, "_persist_metadata_sync"
            ) as persist_metadata_mock,
            patch.object(logger, "cleanup_tmp_files", cleanup_spy),
            patch(
                "chartreux.core.session.session_logger.utc_now",
                # a bit brittle, but required for the call-count choregraphy...
                side_effect=[
                    datetime(2026, 1, 1, 10, 0, 0, tzinfo=UTC),
                    datetime(2026, 1, 1, 10, 0, 0, tzinfo=UTC),
                    datetime(2026, 1, 1, 10, 0, 1, tzinfo=UTC),
                    datetime(2026, 1, 1, 10, 0, 1, tzinfo=UTC),
                ],
            ),
        ):
            await logger.save_interaction(
                messages=messages,
                stats=AgentStats(steps=1),
                config=mock_vibe_config,
                tool_manager=mock_tool_manager,
                agent_profile=mock_agent_profile,
            )
            await logger.save_interaction(
                messages=messages,
                stats=AgentStats(steps=2),
                config=mock_vibe_config,
                tool_manager=mock_tool_manager,
                agent_profile=mock_agent_profile,
            )

        assert persist_messages_mock.call_count == 2
        assert persist_metadata_mock.call_count == 2
        assert cleanup_spy.call_count == 1

    def test_quarantine_corrupt_transcript_preserves_existing_timestamp_collision(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        messages_path = tmp_path / "messages.jsonl"
        messages_path.write_text("newly corrupted")
        messages_path.chmod(0o666)
        timestamp = "20250101T000000000000Z"
        existing = tmp_path / f"messages.jsonl.corrupt-{timestamp}"
        existing.write_text("earlier quarantine")
        monkeypatch.setattr(
            "chartreux.core.session.session_logger.utc_now",
            lambda: datetime(2025, 1, 1, tzinfo=UTC),
        )

        with pytest.warns(RuntimeWarning, match="quarantined"):
            quarantine_path = SessionLogger._quarantine_corrupt_transcript(
                messages_path
            )

        assert existing.read_text() == "earlier quarantine"
        assert quarantine_path != existing
        assert quarantine_path.read_text() == "newly corrupted"
        if os.name == "posix":
            assert quarantine_path.stat().st_mode & 0o077 == 0

    @pytest.mark.asyncio
    async def test_save_interaction_quarantines_corrupt_transcript_before_rewrite(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
        mock_agent_profile: AgentProfile,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        logger = SessionLogger(session_config, "corrupt-transcript")
        assert logger.session_dir is not None
        logger.session_dir.mkdir()
        messages_path = logger.session_dir / "messages.jsonl"
        damaged = '{"role":"user","content":"before"}\nnot-json\n'
        messages_path.write_text(damaged)

        with pytest.warns(RuntimeWarning, match="quarantined"):
            await logger.save_interaction(
                [LLMMessage(role=Role.user, content="after")],
                AgentStats(),
                mock_vibe_config,
                mock_tool_manager,
                mock_agent_profile,
            )

        quarantined = list(logger.session_dir.glob("messages.jsonl.corrupt-*"))
        assert len(quarantined) == 1
        assert quarantined[0].read_text() == damaged
        assert [
            json.loads(line)["content"]
            for line in messages_path.read_text().splitlines()
        ] == ["after"]
        assert "transcript was corrupted and quarantined" in caplog.text

    @pytest.mark.asyncio
    async def test_save_interaction_rewrites_log_when_history_shrinks(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
        mock_agent_profile: AgentProfile,
    ) -> None:
        logger = SessionLogger(session_config, "shrink-history")
        stats = AgentStats(steps=1)
        full = [
            LLMMessage(role=Role.system, content="System prompt"),
            LLMMessage(role=Role.user, content="A"),
            LLMMessage(role=Role.assistant, content="response A"),
            LLMMessage(role=Role.user, content="B"),
            LLMMessage(role=Role.assistant, content="response B"),
        ]
        await logger.save_interaction(
            messages=full,
            stats=stats,
            config=mock_vibe_config,
            tool_manager=mock_tool_manager,
            agent_profile=mock_agent_profile,
        )

        await logger.save_interaction(
            messages=full[:3],
            stats=stats,
            config=mock_vibe_config,
            tool_manager=mock_tool_manager,
            agent_profile=mock_agent_profile,
        )

        assert logger.session_dir is not None
        with open(logger.session_dir / "messages.jsonl") as f:
            lines = [json.loads(line) for line in f]
        assert [m["content"] for m in lines] == ["A", "response A"]
        with open(logger.session_dir / "meta.json") as f:
            assert json.load(f)["total_messages"] == 2

    @pytest.mark.asyncio
    async def test_save_interaction_replaces_tail_after_shrink_and_regrow(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
        mock_agent_profile: AgentProfile,
    ) -> None:
        logger = SessionLogger(session_config, "shrink-regrow")
        stats = AgentStats(steps=1)

        async def save(messages: list[LLMMessage]) -> None:
            await logger.save_interaction(
                messages=messages,
                stats=stats,
                config=mock_vibe_config,
                tool_manager=mock_tool_manager,
                agent_profile=mock_agent_profile,
            )

        system = LLMMessage(role=Role.system, content="System prompt")
        first_turn = [
            system,
            LLMMessage(role=Role.user, content="A"),
            LLMMessage(role=Role.assistant, content="response A"),
        ]
        await save([
            *first_turn,
            LLMMessage(role=Role.user, content="B"),
            LLMMessage(role=Role.assistant, content="response B"),
        ])
        await save(first_turn)
        await save([
            *first_turn,
            LLMMessage(role=Role.user, content="B-bis"),
            LLMMessage(role=Role.assistant, content="response B-bis"),
        ])

        assert logger.session_dir is not None
        loaded, metadata = SessionLoader.load_session(logger.session_dir)
        assert [m.content for m in loaded] == [
            "A",
            "response A",
            "B-bis",
            "response B-bis",
        ]
        assert metadata["total_messages"] == 4

    @pytest.mark.asyncio
    async def test_save_interaction_skips_system_only_history_then_reprompt_replaces_it(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
        mock_agent_profile: AgentProfile,
    ) -> None:
        logger = SessionLogger(session_config, "reduced-to-system")
        stats = AgentStats(steps=1)
        system = LLMMessage(role=Role.system, content="System prompt")

        async def save(messages: list[LLMMessage]) -> None:
            await logger.save_interaction(
                messages=messages,
                stats=stats,
                config=mock_vibe_config,
                tool_manager=mock_tool_manager,
                agent_profile=mock_agent_profile,
            )

        await save([
            system,
            LLMMessage(role=Role.user, content="A"),
            LLMMessage(role=Role.assistant, content="response A"),
        ])

        # Reducing the history to only the system prompt is a no-op: the prior
        # log stays in place (never emptied) so the session remains loadable.
        await save([system])
        assert logger.session_dir is not None
        loaded, metadata = SessionLoader.load_session(logger.session_dir)
        assert [m.content for m in loaded] == ["A", "response A"]
        assert metadata["total_messages"] == 2

        # The next real message replaces the stale tail.
        await save([
            system,
            LLMMessage(role=Role.user, content="A-bis"),
            LLMMessage(role=Role.assistant, content="response A-bis"),
        ])

        loaded, metadata = SessionLoader.load_session(logger.session_dir)
        assert [m.content for m in loaded] == ["A-bis", "response A-bis"]
        assert metadata["total_messages"] == 2

    @pytest.mark.asyncio
    async def test_save_interaction_rewrites_log_when_last_message_changes_at_same_count(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
        mock_agent_profile: AgentProfile,
    ) -> None:
        logger = SessionLogger(session_config, "same-count-diff-tail")
        stats = AgentStats(steps=1)
        system = LLMMessage(role=Role.system, content="System prompt")

        async def save(messages: list[LLMMessage]) -> None:
            await logger.save_interaction(
                messages=messages,
                stats=stats,
                config=mock_vibe_config,
                tool_manager=mock_tool_manager,
                agent_profile=mock_agent_profile,
            )

        await save([
            system,
            LLMMessage(role=Role.user, content="A"),
            LLMMessage(role=Role.assistant, content="response A"),
        ])

        # Same message count, but the last message content differs: the count
        # alone would treat this as a no-op, so the fingerprint must catch it.
        await save([
            system,
            LLMMessage(role=Role.user, content="A"),
            LLMMessage(role=Role.assistant, content="response A (edited)"),
        ])

        assert logger.session_dir is not None
        loaded, metadata = SessionLoader.load_session(logger.session_dir)
        assert [m.content for m in loaded] == ["A", "response A (edited)"]
        assert metadata["total_messages"] == 2

    @pytest.mark.asyncio
    async def test_save_interaction_rewrites_log_for_legacy_session_without_fingerprint(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
        mock_agent_profile: AgentProfile,
    ) -> None:
        logger = SessionLogger(session_config, "legacy-no-fingerprint")
        stats = AgentStats(steps=1)
        system = LLMMessage(role=Role.system, content="System prompt")

        async def save(messages: list[LLMMessage]) -> None:
            await logger.save_interaction(
                messages=messages,
                stats=stats,
                config=mock_vibe_config,
                tool_manager=mock_tool_manager,
                agent_profile=mock_agent_profile,
            )

        await save([
            system,
            LLMMessage(role=Role.user, content="A"),
            LLMMessage(role=Role.assistant, content="response A"),
        ])

        # Simulate a session written before fingerprints existed.
        metadata = json.loads(logger.metadata_filepath.read_text(encoding="utf-8"))
        del metadata["last_message_fingerprint"]
        logger.metadata_filepath.write_text(json.dumps(metadata), encoding="utf-8")

        # Same count, edited tail: with no fingerprint the boundary can't be
        # verified, so the log must be fully rewritten rather than no-op'd.
        await save([
            system,
            LLMMessage(role=Role.user, content="A"),
            LLMMessage(role=Role.assistant, content="response A (edited)"),
        ])

        assert logger.session_dir is not None
        loaded, metadata = SessionLoader.load_session(logger.session_dir)
        assert [m.content for m in loaded] == ["A", "response A (edited)"]
        assert metadata["total_messages"] == 2

    @pytest.mark.asyncio
    async def test_save_interaction_persists_empty_conversation_only_when_allowed(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
        mock_agent_profile: AgentProfile,
    ) -> None:
        logger = SessionLogger(session_config, "rewind-to-top")
        stats = AgentStats(steps=1)
        system = LLMMessage(role=Role.system, content="System prompt")

        async def save(
            messages: list[LLMMessage], *, allow_empty: bool = False
        ) -> None:
            await logger.save_interaction(
                messages=messages,
                stats=stats,
                config=mock_vibe_config,
                tool_manager=mock_tool_manager,
                agent_profile=mock_agent_profile,
                allow_empty=allow_empty,
            )

        await save([
            system,
            LLMMessage(role=Role.user, content="A"),
            LLMMessage(role=Role.assistant, content="response A"),
        ])

        # A system-only save without opt-in keeps the prior log intact.
        await save([system])
        assert logger.session_dir is not None
        loaded, _ = SessionLoader.load_session(logger.session_dir)
        assert [m.content for m in loaded] == ["A", "response A"]

        # An in-place rewind to the first message opts in: the emptied log is
        # persisted and loads back as an empty, valid session.
        await save([system], allow_empty=True)
        loaded, metadata = SessionLoader.load_session(logger.session_dir)
        assert loaded == []
        assert metadata["total_messages"] == 0


def create_temp_file_ago(tmp_path: Path, filename: str, minutes_ago: int = 0) -> Path:
    """Create a file with a modification time of `minutes_ago` minutes ago."""
    file = tmp_path / filename
    file.touch()
    old_time = datetime.now() - timedelta(minutes=minutes_ago)
    os.utime(file, (old_time.timestamp(), old_time.timestamp()))
    return file


class TestSessionLoggerCleanupTmpFiles:
    def test_cleanup_tmp_files_disabled(
        self, disabled_session_config: SessionLoggingConfig
    ) -> None:
        """Test that cleanup_tmp_files returns early when logging is disabled."""
        logger = SessionLogger(disabled_session_config, "test-session")

        assert logger.cleanup_tmp_files() is None
        # The early return leaves the disabled logger's state untouched.
        assert logger.save_dir is None
        assert logger.session_dir is None
        assert logger.session_metadata is None

    def test_cleanup_tmp_files_no_tmp_files(
        self, session_config: SessionLoggingConfig
    ) -> None:
        """Test that cleanup_tmp_files handles no tmp files gracefully."""
        session_id = "test-session-123"
        logger = SessionLogger(session_config, session_id)

        assert logger.cleanup_tmp_files() is None
        assert logger.save_dir is not None
        assert list(logger.save_dir.glob("**/*.json.tmp")) == []

    def test_cleanup_tmp_files_deletes_old_files(
        self, session_config: SessionLoggingConfig
    ) -> None:
        """Test that cleanup_tmp_files deletes tmp files older than 5 minutes."""
        session_id = "test-session-123"
        logger = SessionLogger(session_config, session_id)

        assert logger.session_dir is not None
        logger.session_dir.mkdir(parents=True, exist_ok=True)

        old_tmp_file = create_temp_file_ago(
            logger.session_dir, "session-123.json.tmp", 10
        )
        new_tmp_file = create_temp_file_ago(logger.session_dir, "session-123.json")

        logger.cleanup_tmp_files()

        assert not old_tmp_file.exists()
        assert new_tmp_file.exists()

    def test_cleanup_tmp_files_recursive(
        self, session_config: SessionLoggingConfig
    ) -> None:
        """Test that cleanup_tmp_files works recursively in subdirectories."""
        session_id = "test-session-123"
        logger = SessionLogger(session_config, session_id)

        assert logger.session_dir is not None
        logger.session_dir.mkdir(parents=True, exist_ok=True)

        subdir_1 = logger.session_dir / "session-123"
        subdir_1.mkdir()

        old_tmp_file = create_temp_file_ago(subdir_1, "meta.json.tmp", 10)
        new_tmp_file = create_temp_file_ago(subdir_1, "meta.json")

        subdir_2 = logger.session_dir / "session-456"
        subdir_2.mkdir()

        old_tmp_file_2 = create_temp_file_ago(subdir_2, "meta.json.tmp", 10)

        logger.cleanup_tmp_files()

        assert not old_tmp_file.exists()
        assert not old_tmp_file_2.exists()
        assert new_tmp_file.exists()

    def test_cleanup_tmp_files_handles_exceptions(
        self, session_config: SessionLoggingConfig
    ) -> None:
        """Test that cleanup_tmp_files handles exceptions gracefully."""
        session_id = "test-session-123"
        logger = SessionLogger(session_config, session_id)

        assert logger.session_dir is not None
        logger.session_dir.mkdir(parents=True, exist_ok=True)

        old_tmp_file = create_temp_file_ago(logger.session_dir, "meta.json.tmp", 10)
        another_old_tmp_file = create_temp_file_ago(
            logger.session_dir, "meta-002.json.tmp", 10
        )

        # Mock the unlink method to raise an exception for the first file
        original_unlink = Path.unlink

        def mock_unlink(self):
            if str(self) == str(old_tmp_file):
                raise OSError("Mocked error")
            return original_unlink(self)

        with patch.object(Path, "unlink", mock_unlink):
            logger.cleanup_tmp_files()

        assert old_tmp_file.exists()
        assert not another_old_tmp_file.exists()

    def test_maybe_cleanup_tmp_files_throttles_calls(
        self, session_config: SessionLoggingConfig
    ) -> None:
        session_id = "test-session-123"
        logger = SessionLogger(session_config, session_id)

        cleanup_spy = MagicMock()
        with (
            patch.object(logger, "cleanup_tmp_files", cleanup_spy),
            patch(
                "chartreux.core.session.session_logger.utc_now",
                side_effect=[
                    datetime(2026, 1, 1, 10, 0, 0, tzinfo=UTC),
                    datetime(2026, 1, 1, 10, 0, 1, tzinfo=UTC),
                    datetime(2026, 1, 1, 10, 0, 6, tzinfo=UTC),
                ],
            ),
        ):
            logger.maybe_cleanup_tmp_files()
            logger.maybe_cleanup_tmp_files()
            logger.maybe_cleanup_tmp_files()

        assert cleanup_spy.call_count == 2
