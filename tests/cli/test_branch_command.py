from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from chartreux.cli.textual_ui.widgets.branch_created_message import BranchCreatedMessage
from chartreux.cli.textual_ui.widgets.messages import ErrorMessage
from chartreux.core.config import SessionLoggingConfig
from tests.conftest import (
    build_test_agent_loop,
    build_test_chartreux_app,
    build_test_vibe_config,
)


def _enabled_session_config(save_dir: Path) -> SessionLoggingConfig:
    return SessionLoggingConfig(enabled=True, save_dir=str(save_dir))


def _fork_response(new_session_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        state=SimpleNamespace(session=SimpleNamespace(id=new_session_id))
    )


@pytest.mark.asyncio
async def test_branch_forks_latest_and_shows_resume_hint(tmp_path: Path) -> None:
    config = build_test_vibe_config(session_logging=_enabled_session_config(tmp_path))
    app = build_test_chartreux_app(agent_loop=build_test_agent_loop(config=config))
    captured: dict[str, object] = {}

    async with app.run_test() as pilot:
        sessions = app.app_server.resources.sessions
        old_session_id = app.app_server.session_id

        async def recording_fork(entry_id=None, *, attach=True):
            captured["entry_id"] = entry_id
            captured["attach"] = attach
            return _fork_response("new-branch-session-id")

        sessions.fork = recording_fork  # type: ignore[method-assign]
        handled = await app._handle_command("/branch")
        await pilot.pause()

        assert handled is True
        assert captured == {"entry_id": None, "attach": False}
        message = app.query_one(BranchCreatedMessage)
        assert "new-bran" in message.get_content()
        assert "chartreux --resume new-bran" in message.get_content()
        assert app.app_server.session_id == old_session_id


@pytest.mark.asyncio
async def test_branch_surfaces_errors_without_switching_sessions(
    tmp_path: Path,
) -> None:
    config = build_test_vibe_config(session_logging=_enabled_session_config(tmp_path))
    app = build_test_chartreux_app(agent_loop=build_test_agent_loop(config=config))

    async with app.run_test() as pilot:
        sessions = app.app_server.resources.sessions
        old_session_id = app.app_server.session_id

        async def failing_fork(entry_id=None, *, attach=True):
            raise ConnectionError("app server is unreachable")

        sessions.fork = failing_fork  # type: ignore[method-assign]
        handled = await app._handle_command("/branch")
        await pilot.pause()

        assert handled is True
        assert any(
            "Failed to branch session: app server is unreachable" in str(error._error)
            for error in app.query(ErrorMessage)
        )
        assert not app.query(BranchCreatedMessage)
        assert app.app_server.session_id == old_session_id


@pytest.mark.asyncio
async def test_branch_writes_resumable_detached_session(tmp_path: Path) -> None:
    config = build_test_vibe_config(session_logging=_enabled_session_config(tmp_path))
    app = build_test_chartreux_app(agent_loop=build_test_agent_loop(config=config))

    async with app.run_test() as pilot:
        events = [event async for event in app.app_server.act("hello")]
        assert events

        old_session_id = app.app_server.session_id
        handled = await app._handle_command("/branch")
        await pilot.pause()

        assert handled is True
        message = app.query_one(BranchCreatedMessage)
        new_session_id = message._new_session_id
        assert new_session_id != old_session_id
        assert app.app_server.session_id == old_session_id
        assert new_session_id in {
            session.id
            for session in await app.app_server.resources.sessions.list(
                app.app_server.cwd
            )
        }

        await app.app_server.resume(new_session_id)
        assert app.app_server.session_id == new_session_id
