"""Session logger metadata tests (git and user metadata collection, the AgentStats cost model, retired experiments, legacy session origin and relocation, and the persisted loops, active model and worktree written into metadata)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from chartreux.core.config import ChartreuxConfigSchema, SessionLoggingConfig
from chartreux.core.llm_models import LLMMessage, Role
from chartreux.core.loop import ScheduledLoop
from chartreux.core.session.session_loader import SessionLoader
from chartreux.core.session.session_logger import SessionLogger
from chartreux.core.session_types import AgentStats, SessionMetadata, WorktreeContext
from chartreux.core.tools.manager import ToolManager
from tests.session.persistence._session_persistence_support import (
    disabled_session_config as disabled_session_config,
    mock_tool_manager as mock_tool_manager,
    mock_vibe_config as mock_vibe_config,
    session_config as session_config,
    temp_session_dir as temp_session_dir,
)


def test_legacy_token_only_stats_are_cost_incomplete() -> None:
    stats = AgentStats.model_validate({
        "session_prompt_tokens": 1_000_000,
        "session_completion_tokens": 1_000_000,
        "input_price_per_million": 10.0,
        "output_price_per_million": 20.0,
    })

    assert stats.has_unknown_cost is True
    assert stats.known_cost_total == 0.0
    assert stats.session_cost is None


class TestSessionLoggerMetadata:
    @patch("chartreux.core.session.session_logger.subprocess.run")
    @patch("chartreux.core.session.session_logger.getpass.getuser")
    def test_session_metadata_initialization(
        self, mock_getuser, mock_subprocess, session_config: SessionLoggingConfig
    ) -> None:
        """Test that session metadata is correctly initialized."""
        # Mock combined git command
        git_mock = MagicMock()
        git_mock.returncode = 0
        git_mock.stdout = "abc123\nmain\n"

        mock_subprocess.return_value = git_mock
        mock_getuser.return_value = "testuser"

        session_id = "test-session-123"
        logger = SessionLogger(session_config, session_id)

        assert logger.session_metadata is not None
        metadata = logger.session_metadata

        assert metadata.session_id == session_id
        assert metadata.start_time == logger.session_start_time
        assert metadata.end_time is None
        assert metadata.git_commit == "abc123"
        assert metadata.git_branch == "main"
        assert metadata.username == "testuser"
        assert "working_directory" in metadata.environment
        assert metadata.environment["working_directory"] == str(Path.cwd())
        assert metadata.title is None
        assert metadata.title_source == "auto"

    @patch("chartreux.core.session.session_logger.subprocess.run")
    @patch("chartreux.core.session.session_logger.getpass.getuser")
    def test_session_metadata_with_git_errors(
        self, mock_getuser, mock_subprocess, session_config: SessionLoggingConfig
    ) -> None:
        """Test that session metadata handles git command errors gracefully."""
        # Mock combined git command to fail
        mock_subprocess.side_effect = FileNotFoundError("git not found")
        mock_getuser.return_value = "testuser"

        session_id = "test-session-123"
        logger = SessionLogger(session_config, session_id)

        assert logger.session_metadata is not None
        metadata = logger.session_metadata

        assert metadata.git_commit is None
        assert metadata.git_branch is None
        assert metadata.username == "testuser"


class TestPersistLoops:
    @pytest.mark.asyncio
    async def test_writes_into_existing_metadata(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
    ) -> None:
        logger = SessionLogger(session_config, "test-session-loops")
        await logger.save_interaction(
            messages=[
                LLMMessage(role=Role.system, content="System prompt"),
                LLMMessage(role=Role.user, content="Hello"),
                LLMMessage(role=Role.assistant, content="Hi there!"),
            ],
            stats=AgentStats(steps=1),
            config=mock_vibe_config,
            tool_manager=mock_tool_manager,
        )

        assert logger.session_metadata is not None
        logger.session_metadata.loops = [
            ScheduledLoop(
                id="aabbccdd",
                interval_seconds=30,
                prompt="ping",
                next_fire_at=12345.0,
                created_at=12000.0,
            )
        ]
        await logger.persist_loops()

        assert logger.session_dir is not None
        with open(logger.session_dir / "meta.json") as f:
            metadata = json.load(f)

        assert metadata["session_id"] == "test-session-loops"
        assert metadata["total_messages"] == 2
        assert metadata["loops"] == [
            {
                "id": "aabbccdd",
                "interval_seconds": 30,
                "prompt": "ping",
                "next_fire_at": 12345.0,
                "created_at": 12000.0,
            }
        ]

    @pytest.mark.asyncio
    async def test_noop_when_metadata_file_missing(
        self, session_config: SessionLoggingConfig
    ) -> None:
        logger = SessionLogger(session_config, "no-meta-session")
        assert logger.session_dir is not None
        # No save_interaction was called -> meta.json does not exist
        assert not (logger.session_dir / "meta.json").exists()

        assert logger.session_metadata is not None
        logger.session_metadata.loops = [
            ScheduledLoop(
                id="aabbccdd",
                interval_seconds=30,
                prompt="ping",
                next_fire_at=1.0,
                created_at=0.0,
            )
        ]
        await logger.persist_loops()

        assert not (logger.session_dir / "meta.json").exists()

    @pytest.mark.asyncio
    async def test_noop_when_logging_disabled(
        self, disabled_session_config: SessionLoggingConfig
    ) -> None:
        logger = SessionLogger(disabled_session_config, "ignored")
        # Should not raise even though session_metadata is None
        assert await logger.persist_loops() is None
        # The no-op leaves the disabled logger's state untouched.
        assert logger.session_metadata is None
        assert logger.session_dir is None

    @pytest.mark.asyncio
    async def test_subsequent_save_interaction_preserves_loops(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
    ) -> None:
        logger = SessionLogger(session_config, "loops-vs-save")
        messages = [
            LLMMessage(role=Role.system, content="System prompt"),
            LLMMessage(role=Role.user, content="Hello"),
            LLMMessage(role=Role.assistant, content="Hi there!"),
        ]
        await logger.save_interaction(
            messages=messages,
            stats=AgentStats(steps=1),
            config=mock_vibe_config,
            tool_manager=mock_tool_manager,
        )

        assert logger.session_metadata is not None
        logger.session_metadata.loops = [
            ScheduledLoop(
                id="aabbccdd",
                interval_seconds=30,
                prompt="ping",
                next_fire_at=12345.0,
                created_at=12000.0,
            )
        ]
        await logger.persist_loops()

        # A subsequent save (e.g. user sends another message) must not
        # overwrite the on-disk loops with the stale in-memory value.
        more_messages = [*messages, LLMMessage(role=Role.user, content="Again")]
        await logger.save_interaction(
            messages=more_messages,
            stats=AgentStats(steps=2),
            config=mock_vibe_config,
            tool_manager=mock_tool_manager,
        )

        assert logger.session_dir is not None
        with open(logger.session_dir / "meta.json") as f:
            metadata = json.load(f)
        assert len(metadata["loops"]) == 1
        assert metadata["loops"][0]["id"] == "aabbccdd"


def test_session_metadata_ignores_retired_experiments(
    session_config: SessionLoggingConfig,
) -> None:
    logger = SessionLogger(session_config, "legacy-experiments")
    assert logger.session_metadata is not None
    data = logger.session_metadata.model_dump()
    data["experiments"] = {"features": {"retired": {"defaultValue": "old"}}}

    restored = SessionMetadata.model_validate(data)

    assert restored.session_id == "legacy-experiments"
    assert "experiments" not in restored.model_dump()


class TestPersistActiveModel:
    @pytest.mark.asyncio
    async def test_persists_before_first_save_and_updates_existing_metadata(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
    ) -> None:
        logger = SessionLogger(session_config, "active-model-session")
        active_model = mock_vibe_config.get_active_model().alias

        first_change = await logger.persist_active_model(active_model)
        repeated_before_save = await logger.persist_active_model(active_model)

        assert logger.session_dir is not None
        metadata_path = logger.session_dir / "meta.json"
        assert not metadata_path.exists()
        assert logger.active_model == active_model

        await logger.save_interaction(
            messages=[
                LLMMessage(role=Role.system, content="System prompt"),
                LLMMessage(role=Role.user, content="Hello"),
            ],
            stats=AgentStats(steps=1),
            config=mock_vibe_config,
            tool_manager=mock_tool_manager,
        )
        second_change = await logger.persist_active_model("changed-model")
        repeated_after_save = await logger.persist_active_model("changed-model")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert first_change is True
        assert repeated_before_save is False
        assert second_change is True
        assert repeated_after_save is False
        assert metadata["config"]["active_model"] == "changed-model"
        assert "models" not in metadata["config"]
        assert "providers" not in metadata["config"]
        assert logger.active_model == "changed-model"


class TestPersistCreatedWorktree:
    @pytest.fixture
    def worktree(self) -> WorktreeContext:
        return WorktreeContext(
            entry_id="worktree-feature-x",
            name="feature-x",
            branch="vibe/feature-x",
            path="/repo/.worktrees/feature-x",
            created_at=1234567890,
        )

    @pytest.mark.asyncio
    async def test_a_fresh_session_carries_the_worktree_into_the_first_save(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
        worktree: WorktreeContext,
    ) -> None:
        # The worktree is recorded before the session has a message, so there is
        # no meta.json to patch yet: only the in-memory metadata can carry it
        # until the first full save writes it out.
        logger = SessionLogger(session_config, "worktree-session")
        assert logger.session_dir is not None
        assert not (logger.session_dir / "meta.json").exists()

        await logger.persist_created_worktree(worktree)

        assert not (logger.session_dir / "meta.json").exists()

        await logger.save_interaction(
            messages=[
                LLMMessage(role=Role.system, content="System prompt"),
                LLMMessage(role=Role.user, content="Hello"),
            ],
            stats=AgentStats(steps=1),
            config=mock_vibe_config,
            tool_manager=mock_tool_manager,
        )

        reloaded = SessionLoader.load_metadata(logger.session_dir)
        assert reloaded.created_worktree == worktree

    @pytest.mark.asyncio
    async def test_an_existing_metadata_file_is_patched_without_a_full_save(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
        worktree: WorktreeContext,
    ) -> None:
        logger = SessionLogger(session_config, "worktree-resumed")
        await logger.save_interaction(
            messages=[
                LLMMessage(role=Role.system, content="System prompt"),
                LLMMessage(role=Role.user, content="Hello"),
            ],
            stats=AgentStats(steps=1),
            config=mock_vibe_config,
            tool_manager=mock_tool_manager,
        )

        await logger.persist_created_worktree(worktree)

        assert logger.session_dir is not None
        reloaded = SessionLoader.load_metadata(logger.session_dir)
        assert reloaded.created_worktree == worktree


def _legacy_metadata(working_directory: Path) -> SessionMetadata:
    """Metadata as written before ``origin_directory`` existed."""
    return SessionMetadata(
        session_id="s1",
        start_time="2026-01-01T00:00:00+00:00",
        end_time=None,
        git_commit=None,
        git_branch=None,
        username="tester",
        environment={"working_directory": str(working_directory)},
    )


def test_a_legacy_session_keeps_its_origin_when_it_first_moves(
    tmp_path: Path, session_config
) -> None:
    # Sessions written before origin_directory existed, and imported ones, carry
    # only the environment entry. Overwriting it on the first move would leave
    # nothing naming where the session started, so it would vanish from that
    # directory: the disappearance this pair of fields exists to prevent.
    repo = tmp_path / "repo"
    worktree = tmp_path / "worktree"
    logger = SessionLogger(session_config, "s1")
    logger.session_metadata = _legacy_metadata(repo)

    logger.relocated_to(worktree)

    metadata = logger.session_metadata.model_dump()
    assert SessionLoader._session_reaches(metadata, repo)
    assert SessionLoader._session_reaches(metadata, worktree)


def test_a_second_move_does_not_overwrite_the_promoted_origin(
    tmp_path: Path, session_config
) -> None:
    repo = tmp_path / "repo"
    first = tmp_path / "first"
    second = tmp_path / "second"
    logger = SessionLogger(session_config, "s1")
    logger.session_metadata = _legacy_metadata(repo)

    logger.relocated_to(first)
    logger.relocated_to(second)

    metadata = logger.session_metadata.model_dump()
    assert SessionLoader._session_reaches(metadata, repo)
    assert SessionLoader._session_reaches(metadata, second)
    assert not SessionLoader._session_reaches(metadata, first)
