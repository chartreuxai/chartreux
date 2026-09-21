from __future__ import annotations

import base64
from collections.abc import MutableSequence
from typing import Any

import pytest

import chartreux.app_server._host as host_module
from chartreux.app_server._host import HostRequestHandler
from chartreux.app_server.protocol import SessionListResponse
from chartreux.core.config.harness_files import HarnessFilesManager
from chartreux.core.session.resume_sessions import ResumeSessionInfo
from tests.conftest import build_test_vibe_config


def _session(
    session_id: str, updated_at: str, *, parent_session_id: str | None = None
) -> ResumeSessionInfo:
    return ResumeSessionInfo(
        session_id=session_id,
        cwd="/workspace",
        updated_at=updated_at,
        parent_session_id=parent_session_id,
    )


@pytest.fixture
def session_list_route(
    monkeypatch: pytest.MonkeyPatch,
) -> MutableSequence[ResumeSessionInfo]:
    sessions: MutableSequence[ResumeSessionInfo] = []
    monkeypatch.setattr(
        host_module, "list_local_resume_sessions", lambda _config, _cwd: sessions
    )
    monkeypatch.setattr(
        host_module.SessionLoader, "get_first_user_message", lambda *_args: ""
    )
    monkeypatch.setattr(host_module.last_session_pointer, "load", lambda _config: None)
    return sessions


async def _list_sessions(
    monkeypatch: pytest.MonkeyPatch, **params: Any
) -> SessionListResponse:
    handler = HostRequestHandler(HarnessFilesManager(sources=()))
    config = build_test_vibe_config()

    async def load_config(_cwd: str | None):
        return config

    monkeypatch.setattr(handler, "_load_config", load_config)
    response = (await handler.dispatch("session/list", params)).response
    assert isinstance(response, SessionListResponse)
    return response


@pytest.mark.asyncio
async def test_session_list_continues_after_cursor_session_is_touched(
    monkeypatch: pytest.MonkeyPatch,
    session_list_route: MutableSequence[ResumeSessionInfo],
) -> None:
    session_list_route[:] = [
        _session("newest", "2026-01-03T00:00:00+00:00"),
        _session("cursor", "2026-01-02T00:00:00+00:00"),
        _session("older", "2026-01-01T00:00:00+00:00"),
    ]

    first = await _list_sessions(monkeypatch, limit=2)
    session_list_route[1] = _session("cursor", "2026-01-04T00:00:00+00:00")
    second = await _list_sessions(monkeypatch, limit=2, cursor=first.next_cursor)

    assert [session.id for session in first.items] == ["newest", "cursor"]
    assert [session.id for session in second.items] == ["older"]


@pytest.mark.asyncio
async def test_session_list_continues_after_cursor_session_is_deleted(
    monkeypatch: pytest.MonkeyPatch,
    session_list_route: MutableSequence[ResumeSessionInfo],
) -> None:
    session_list_route[:] = [
        _session("newest", "2026-01-03T00:00:00+00:00"),
        _session("cursor", "2026-01-02T00:00:00+00:00"),
        _session("older", "2026-01-01T00:00:00+00:00"),
    ]

    first = await _list_sessions(monkeypatch, limit=2)
    del session_list_route[1]
    second = await _list_sessions(monkeypatch, limit=2, cursor=first.next_cursor)

    assert [session.id for session in second.items] == ["older"]


@pytest.mark.asyncio
async def test_session_list_cursor_uses_session_id_to_break_equal_timestamps(
    monkeypatch: pytest.MonkeyPatch,
    session_list_route: MutableSequence[ResumeSessionInfo],
) -> None:
    session_list_route[:] = [
        _session("alpha", "2026-01-02T00:00:00+00:00"),
        _session("bravo", "2026-01-02T00:00:00+00:00"),
        _session("older", "2026-01-01T00:00:00+00:00"),
    ]

    first = await _list_sessions(monkeypatch, limit=1)
    second = await _list_sessions(monkeypatch, limit=2, cursor=first.next_cursor)

    assert [session.id for session in first.items] == ["bravo"]
    assert [session.id for session in second.items] == ["alpha", "older"]


@pytest.mark.asyncio
async def test_session_list_cursor_excludes_newer_inserts_and_includes_older_ones(
    monkeypatch: pytest.MonkeyPatch,
    session_list_route: MutableSequence[ResumeSessionInfo],
) -> None:
    session_list_route[:] = [
        _session("newest", "2026-01-04T00:00:00+00:00"),
        _session("cursor", "2026-01-03T00:00:00+00:00"),
        _session("older", "2026-01-01T00:00:00+00:00"),
    ]

    first = await _list_sessions(monkeypatch, limit=2)
    session_list_route.extend([
        _session("newer-insert", "2026-01-05T00:00:00+00:00"),
        _session("older-insert", "2026-01-02T00:00:00+00:00"),
    ])
    second = await _list_sessions(monkeypatch, limit=2, cursor=first.next_cursor)

    assert [session.id for session in second.items] == ["older-insert", "older"]


@pytest.mark.asyncio
async def test_session_list_cursor_preserves_filters_and_continue_pointer(
    monkeypatch: pytest.MonkeyPatch,
    session_list_route: MutableSequence[ResumeSessionInfo],
) -> None:
    session_list_route[:] = [
        _session("root-a", "2026-01-04T00:00:00+00:00"),
        _session("child-a", "2026-01-03T00:00:00+00:00", parent_session_id="root-a"),
        _session("root-b", "2026-01-02T00:00:00+00:00"),
        _session("child-b", "2026-01-01T00:00:00+00:00", parent_session_id="root-b"),
    ]
    monkeypatch.setattr(
        host_module.last_session_pointer, "load", lambda _config: "child-a"
    )

    first = await _list_sessions(monkeypatch, rootSessionId="root-a", limit=1)
    second = await _list_sessions(
        monkeypatch, rootSessionId="root-a", limit=1, cursor=first.next_cursor
    )
    children = await _list_sessions(monkeypatch, parentSessionId="root-a")

    assert [session.id for session in first.items] == ["root-a"]
    assert first.continue_session_id == "child-a"
    assert [session.id for session in second.items] == ["child-a"]
    assert [session.id for session in children.items] == ["child-a"]


@pytest.mark.asyncio
async def test_session_list_rejects_decodable_malformed_cursor_tuple(
    monkeypatch: pytest.MonkeyPatch,
    session_list_route: MutableSequence[ResumeSessionInfo],
) -> None:
    session_list_route[:] = [_session("only", "2026-01-01T00:00:00+00:00")]
    malformed_cursor = base64.urlsafe_b64encode(b"zzzz\0").decode().rstrip("=")

    response = await _list_sessions(monkeypatch, limit=1, cursor=malformed_cursor)

    assert response.items == []
    assert response.next_cursor is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [b"2026-01-02T00:00:00+00:00\0\0", b"2026-01-02T00:00:00+00:00\0only\0junk"],
    ids=["empty-session-id-with-extra-separator", "trailing-component"],
)
async def test_session_list_rejects_cursor_with_additional_nul_separator(
    monkeypatch: pytest.MonkeyPatch,
    session_list_route: MutableSequence[ResumeSessionInfo],
    payload: bytes,
) -> None:
    session_list_route[:] = [_session("only", "2026-01-01T00:00:00+00:00")]
    malformed_cursor = base64.urlsafe_b64encode(payload).decode().rstrip("=")

    response = await _list_sessions(monkeypatch, limit=1, cursor=malformed_cursor)

    assert response.items == []
    assert response.next_cursor is None


@pytest.mark.asyncio
async def test_session_list_cursor_handles_end_of_list_and_malformed_values(
    monkeypatch: pytest.MonkeyPatch,
    session_list_route: MutableSequence[ResumeSessionInfo],
) -> None:
    session_list_route[:] = [_session("only", "2026-01-01T00:00:00+00:00")]

    first = await _list_sessions(monkeypatch, limit=1)
    at_end = await _list_sessions(
        monkeypatch,
        limit=1,
        cursor=host_module._encode_session_cursor(session_list_route[0]),
    )
    malformed = await _list_sessions(monkeypatch, limit=1, cursor="not-a-cursor")

    assert first.next_cursor is None
    assert at_end.items == []
    assert at_end.next_cursor is None
    assert malformed.items == []
    assert malformed.next_cursor is None
