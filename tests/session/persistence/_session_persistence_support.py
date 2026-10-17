"""Shared test-support surface for the session logger suite (W10, stage 19i).

The definitions here are the shared module-level surface of
``tests/session/test_session_logger.py``, relocated AST-verbatim in the
19i phase-(i) support-seam commit: the six function-scoped fixtures the
concern clusters share — ``temp_session_dir`` (the dependency closure of
``session_config``; one direct test user in the initialization cluster),
``session_config`` (80 tests request it, spanning the initialization,
metadata, title, save, generation-fence and cursor clusters),
``disabled_session_config`` (8 users across the initialization, title and
save clusters), ``mock_agent_profile`` (40 users), ``mock_tool_manager``
(43 users) and ``mock_vibe_config`` (35 users, each spanning the metadata,
save, title, cursor and generation-fence clusters). The remaining
module-level surface is single-concern and stays in
``test_session_logger.py`` until the phase-(ii) concern moves: the
``create_temp_file_ago`` helper (3 users, tmp-file cleanup), the
``_legacy_metadata`` helper (2 users, legacy session origin), the
``TestTranscriptCursor`` class with its ``_save`` helper method (cursor)
and the ``TestPersistCreatedWorktree`` class with its ``worktree`` fixture
(2 users, its own class). No test bodies or marks live here; the test
population stays in ``test_session_logger.py`` until the phase-(ii)
concern moves. Importer: ``test_session_logger.py`` itself (the seam
enumeration found zero external importers of that file).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from chartreux.core.agents.models import AgentProfile, AgentSafety
from chartreux.core.config import ChartreuxConfigSchema, SessionLoggingConfig
from chartreux.core.tools.manager import ToolManager
from tests.conftest import build_test_vibe_config


@pytest.fixture
def temp_session_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for session logging tests."""
    session_dir = tmp_path / "sessions"
    session_dir.mkdir()
    return session_dir


@pytest.fixture
def session_config(temp_session_dir: Path) -> SessionLoggingConfig:
    """Create a session logging config for testing."""
    return SessionLoggingConfig(
        save_dir=str(temp_session_dir), session_prefix="test", enabled=True
    )


@pytest.fixture
def disabled_session_config() -> SessionLoggingConfig:
    """Create a disabled session logging config for testing."""
    return SessionLoggingConfig(
        save_dir="/tmp/test", session_prefix="test", enabled=False
    )


@pytest.fixture
def mock_agent_profile() -> AgentProfile:
    """Create a mock agent profile for testing."""
    return AgentProfile(
        name="test-agent",
        display_name="Test Agent",
        description="A test agent",
        safety=AgentSafety.NEUTRAL,
        overrides={},
    )


@pytest.fixture
def mock_tool_manager() -> ToolManager:
    """Create a mock tool manager for testing."""
    manager = MagicMock(spec=ToolManager)
    manager.available_tools = {}
    return manager


@pytest.fixture
def mock_vibe_config() -> ChartreuxConfigSchema:
    """Create a mock vibe config for testing."""
    return build_test_vibe_config()
