from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from chartreux.cli.textual_ui.external_editor import ExternalEditor
from chartreux.cli.textual_ui.widgets.messages import ErrorMessage, UserCommandMessage
from tests.conftest import build_test_chartreux_app, build_test_vibe_config


@contextmanager
def mock_suspend():
    """Stand-in for ``App.suspend``, which headless drivers do not support."""
    yield


def _stub_reload(app: Any) -> AsyncMock:
    reload_mock = AsyncMock(
        return_value=SimpleNamespace(
            stripped_history_images=0, launch_metadata_persisted=True
        )
    )
    app.app_server.resources.config.reload = reload_mock
    return reload_mock


@pytest.mark.asyncio
async def test_config_command_creates_missing_file_and_reloads_on_change(
    monkeypatch: pytest.MonkeyPatch, config_dir: Path
) -> None:
    config_file = config_dir / "config.toml"
    config_file.unlink()

    def fake_edit(file_path: Path, *, check: bool = False) -> None:
        file_path.write_text(
            file_path.read_text("utf-8") + '\ntheme = "dark"\n', "utf-8"
        )

    monkeypatch.setattr(ExternalEditor, "edit_file", staticmethod(fake_edit))
    app = build_test_chartreux_app(config=build_test_vibe_config())

    async with app.run_test() as pilot:
        monkeypatch.setattr(app, "suspend", mock_suspend)
        reload_mock = _stub_reload(app)
        handled = await app._handle_command("/config")
        await pilot.pause()

    assert handled is True
    content = config_file.read_text("utf-8")
    assert content.startswith("# Chartreux user configuration.")
    assert 'theme = "dark"' in content
    reload_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_config_command_skips_reload_when_file_unchanged(
    monkeypatch: pytest.MonkeyPatch, config_dir: Path
) -> None:
    config_file = config_dir / "config.toml"
    config_file.write_text('theme = "dark"\n', "utf-8")

    monkeypatch.setattr(
        ExternalEditor, "edit_file", staticmethod(lambda *_a, **_k: None)
    )
    app = build_test_chartreux_app(config=build_test_vibe_config())

    async with app.run_test() as pilot:
        monkeypatch.setattr(app, "suspend", mock_suspend)
        reload_mock = _stub_reload(app)
        handled = await app._handle_command("/config")
        await pilot.pause()
        unchanged = [
            message._content
            for message in app.query(UserCommandMessage)
            if message._content.startswith("Config file unchanged")
        ]

    assert handled is True
    reload_mock.assert_not_awaited()
    assert len(unchanged) == 1
    assert str(config_file) in unchanged[0]


@pytest.mark.asyncio
async def test_config_command_reports_editor_failure(
    monkeypatch: pytest.MonkeyPatch, config_dir: Path
) -> None:
    config_file = config_dir / "config.toml"

    def failing_edit(_file_path: Path, *, check: bool = False) -> None:
        raise OSError("no editor")

    monkeypatch.setattr(ExternalEditor, "edit_file", staticmethod(failing_edit))
    app = build_test_chartreux_app(config=build_test_vibe_config())

    async with app.run_test() as pilot:
        monkeypatch.setattr(app, "suspend", mock_suspend)
        reload_mock = _stub_reload(app)
        handled = await app._handle_command("/config")
        await pilot.pause()
        error = app.query_one(ErrorMessage)

    assert handled is True
    reload_mock.assert_not_awaited()
    error_text = str(error._error)
    assert "Could not open or read config file" in error_text
    assert str(config_file) in error_text


@pytest.mark.asyncio
@pytest.mark.parametrize("dangling", [True, False])
async def test_config_command_does_not_follow_config_symlink(
    monkeypatch: pytest.MonkeyPatch, config_dir: Path, dangling: bool
) -> None:
    config_file = config_dir / "config.toml"
    app = build_test_chartreux_app(config=build_test_vibe_config())
    target = config_dir / "symlink-target.toml"
    config_file.unlink(missing_ok=True)
    if not dangling:
        target.write_text("keep this content")
    config_file.symlink_to(target)

    async with app.run_test() as pilot:
        monkeypatch.setattr(app, "suspend", mock_suspend)
        reload_mock = _stub_reload(app)
        handled = await app._handle_command("/config")
        await pilot.pause()
        errors = list(app.query(ErrorMessage))

    assert handled is True
    assert len(errors) == 1
    if dangling:
        assert not target.exists()
    else:
        assert target.read_text() == "keep this content"
    reload_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_config_command_concurrent_create_preserves_winner(
    monkeypatch: pytest.MonkeyPatch, config_dir: Path
) -> None:
    import os

    import chartreux.cli.textual_ui.app as app_module

    config_file = config_dir / "config.toml"
    config_file.unlink(missing_ok=True)
    original_open = os.open

    def competing_open(
        path: Path | str, flags: int, mode: int = 0o777, **kwargs: Any
    ) -> int:
        if path == config_file.name and "dir_fd" in kwargs:
            config_file.write_text("winner")
        return original_open(path, flags, mode, **kwargs)

    monkeypatch.setattr(app_module.os, "open", competing_open)
    monkeypatch.setattr(
        ExternalEditor, "edit_file", staticmethod(lambda *_a, **_k: None)
    )
    app = build_test_chartreux_app(config=build_test_vibe_config())
    async with app.run_test() as pilot:
        monkeypatch.setattr(app, "suspend", mock_suspend)
        await app._handle_command("/config")
        await pilot.pause()
    assert config_file.read_text() == "winner"


@pytest.mark.asyncio
@pytest.mark.parametrize("issue", ["encoding", "editor"])
async def test_config_command_reports_invalid_input(
    monkeypatch: pytest.MonkeyPatch, config_dir: Path, issue: str
) -> None:
    config_file = config_dir / "config.toml"
    if issue == "encoding":
        config_file.write_bytes(b"\xff")
    else:
        config_file.write_text("original")
        monkeypatch.setenv("VISUAL", "unmatched '")
    app = build_test_chartreux_app(config=build_test_vibe_config())
    async with app.run_test() as pilot:
        monkeypatch.setattr(app, "suspend", mock_suspend)
        reload_mock = _stub_reload(app)
        await app._handle_command("/config")
        await pilot.pause()
        errors = list(app.query(ErrorMessage))
    assert len(errors) == 1
    assert ("UTF-8" if issue == "encoding" else "Invalid editor") in str(
        errors[0]._error
    )
    reload_mock.assert_not_awaited()
