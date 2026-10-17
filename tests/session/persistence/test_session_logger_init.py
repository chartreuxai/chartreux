"""Session logger initialization tests (enabled and disabled construction, session directory creation and private file modes, session reset, and the logger path and folder properties)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from chartreux.core.config import SessionLoggingConfig
from chartreux.core.session.session_logger import SessionLogger
from chartreux.core.session_types import SessionMetadata
from tests.session.persistence._session_persistence_support import (
    disabled_session_config as disabled_session_config,
    session_config as session_config,
    temp_session_dir as temp_session_dir,
)


class TestSessionLoggerInitialization:
    def test_enabled_session_logger_initialization(
        self, session_config: SessionLoggingConfig
    ) -> None:
        """Test that SessionLogger initializes correctly when enabled."""
        session_id = "test-session-123"
        logger = SessionLogger(session_config, session_id)

        assert logger.enabled is True
        assert logger.session_id == session_id
        assert logger.save_dir == Path(session_config.save_dir)
        assert logger.session_prefix == session_config.session_prefix
        assert logger.session_dir is not None
        assert logger.session_metadata is not None
        assert isinstance(logger.session_metadata, SessionMetadata)

        # Check that session directory was created
        assert logger.session_dir is not None
        assert str(logger.session_dir).startswith(str(session_config.save_dir))

        # Check session directory name format
        dir_name = logger.session_dir.name
        assert dir_name.startswith(f"{session_config.session_prefix}_")
        assert session_id[:8] in dir_name

    def test_disabled_session_logger_initialization(
        self, disabled_session_config: SessionLoggingConfig
    ) -> None:
        """Test that SessionLogger initializes correctly when disabled."""
        session_id = "test-session-123"
        logger = SessionLogger(disabled_session_config, session_id)

        assert logger.enabled is False
        assert logger.session_id == "disabled"
        assert logger.save_dir is None
        assert logger.session_prefix is None
        assert logger.session_dir is None
        assert logger.session_metadata is None

    @pytest.mark.skipif(os.name != "posix", reason="POSIX file modes only")
    def test_new_session_directory_and_messages_are_private(
        self, session_config: SessionLoggingConfig
    ) -> None:
        logger = SessionLogger(session_config, "private-session")
        assert logger.save_dir is not None
        assert logger.session_dir is not None
        logger.session_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
        SessionLogger._persist_messages_sync(
            [{"content": "secret"}], logger.session_dir
        )

        assert logger.session_dir.stat().st_mode & 0o777 == 0o700
        assert logger.messages_filepath.stat().st_mode & 0o777 == 0o600

    @pytest.mark.skipif(os.name != "posix", reason="POSIX file modes only")
    def test_existing_session_directory_and_messages_keep_their_modes(
        self, temp_session_dir: Path
    ) -> None:
        session_dir = temp_session_dir / "existing"
        session_dir.mkdir(mode=0o755)
        session_dir.chmod(0o755)
        initial_mode = session_dir.stat().st_mode & 0o777
        messages = session_dir / "messages.jsonl"
        messages.write_text("{}\n")
        messages.chmod(0o644)

        SessionLogger._persist_messages_sync([{"content": "more"}], session_dir)

        assert session_dir.stat().st_mode & 0o777 == initial_mode
        assert messages.stat().st_mode & 0o777 == 0o644

    def test_private_modes_survive_permissive_umask(
        self, session_config: SessionLoggingConfig
    ) -> None:
        if os.name != "posix":
            pytest.skip("POSIX file modes only")
        old_umask = os.umask(0o022)
        try:
            logger = SessionLogger(session_config, "umask-session")
            assert logger.save_dir is not None
            assert logger.session_dir is not None
            logger.session_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
            SessionLogger._persist_messages_sync(
                [{"content": "secret"}], logger.session_dir
            )
            assert logger.session_dir.stat().st_mode & 0o777 == 0o700
            assert logger.messages_filepath.stat().st_mode & 0o777 == 0o600
        finally:
            os.umask(old_umask)


class TestSessionLoggerResetSession:
    def test_reset_session(self, session_config: SessionLoggingConfig) -> None:
        """Test that reset_session correctly resets session information."""
        session_id = "test-session-123"
        logger = SessionLogger(session_config, session_id)

        # Store original session info
        original_session_id = logger.session_id
        original_metadata = logger.session_metadata

        # Reset session
        new_session_id = "test-session-456"
        logger.reset_session(new_session_id)

        # Verify session was reset
        assert logger.session_id == new_session_id
        assert logger.session_start_time != "N/A"  # Should be a valid timestamp
        assert logger.session_metadata is not None
        assert logger.session_metadata.session_id == new_session_id

        # Verify that metadata was recreated (different object)
        assert logger.session_metadata is not original_metadata

        assert logger.session_id != original_session_id

    def test_reset_session_disabled(
        self, disabled_session_config: SessionLoggingConfig
    ) -> None:
        """Test that reset_session does nothing when logging is disabled."""
        logger = SessionLogger(disabled_session_config, "test-session")

        # Reset session should not raise any errors
        logger.reset_session("new-session")

        # Verify state is unchanged
        assert logger.enabled is False
        assert logger.session_id == "disabled"


class TestSessionLoggerFileOperations:
    def test_save_folder(self, session_config: SessionLoggingConfig) -> None:
        """Test that save_folder creates correct folder name."""
        session_id = "test-session-123"
        logger = SessionLogger(session_config, session_id)

        folder = logger.save_folder

        assert folder.parent == Path(session_config.save_dir)
        assert folder.name.startswith(f"{session_config.session_prefix}_")
        assert session_id[:8] in folder.name

    def test_metadata_filepath(self, session_config: SessionLoggingConfig) -> None:
        """Test that metadata_filepath returns correct path."""
        session_id = "test-session-123"
        logger = SessionLogger(session_config, session_id)

        metadata_file = logger.metadata_filepath

        assert logger.session_dir is not None
        assert metadata_file == logger.session_dir / "meta.json"

    def test_messages_filepath(self, session_config: SessionLoggingConfig) -> None:
        """Test that messages_filepath returns correct path."""
        session_id = "test-session-123"
        logger = SessionLogger(session_config, session_id)

        messages_file = logger.messages_filepath

        assert logger.session_dir is not None
        assert messages_file == logger.session_dir / "messages.jsonl"

    def test_disabled_file_operations_raise_errors(
        self, disabled_session_config: SessionLoggingConfig
    ) -> None:
        """Test that file operations raise errors when logging is disabled."""
        logger = SessionLogger(disabled_session_config, "test-session")

        with pytest.raises(
            RuntimeError,
            match="Cannot get session save folder when logging is disabled",
        ):
            assert logger.save_folder is None

        with pytest.raises(
            RuntimeError,
            match="Cannot get session metadata filepath when logging is disabled",
        ):
            assert logger.metadata_filepath is None

        with pytest.raises(
            RuntimeError,
            match="Cannot get session messages filepath when logging is disabled",
        ):
            assert logger.messages_filepath is None
