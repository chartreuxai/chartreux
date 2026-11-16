from __future__ import annotations

from pathlib import Path

import pytest

from chartreux.app_server._runtime import (
    SessionAuditSinkUnavailableError,
    require_writable_session_audit_sink,
    session_audit_sink_available,
)
from chartreux.core.config import SessionLoggingConfig
from tests.conftest import build_test_vibe_config


def _config_with_logging(save_dir: Path, *, enabled: bool):
    config = build_test_vibe_config()
    return config.model_copy(
        update={
            "session_logging": SessionLoggingConfig(
                save_dir=str(save_dir), enabled=enabled
            )
        }
    )


class TestWritableSessionAuditSink:
    def test_disabled_logging_fails_closed(self, tmp_path: Path) -> None:
        config = _config_with_logging(tmp_path, enabled=False)
        with pytest.raises(SessionAuditSinkUnavailableError):
            require_writable_session_audit_sink(config)
        assert session_audit_sink_available(config) is False

    def test_enabled_writable_dir_passes(self, tmp_path: Path) -> None:
        config = _config_with_logging(tmp_path, enabled=True)
        require_writable_session_audit_sink(config)
        assert session_audit_sink_available(config) is True

    def test_missing_save_dir_fails_closed(self, tmp_path: Path) -> None:
        config = _config_with_logging(tmp_path / "no-such-dir", enabled=True)
        with pytest.raises(SessionAuditSinkUnavailableError):
            require_writable_session_audit_sink(config)
        assert session_audit_sink_available(config) is False
