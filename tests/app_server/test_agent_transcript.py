from __future__ import annotations

import asyncio
from dataclasses import replace
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from chartreux.app_server._dispatch import RequestFailure
from chartreux.app_server._handler import (
    CoreRequestHandler,
    _read_authorized_agent_transcript,
)
from chartreux.app_server._session_resources import SessionResource
from chartreux.app_server._sessions import TranscriptReadSnapshot
from chartreux.app_server.protocol import (
    AgentTranscriptGetResponse,
    AgentTranscriptState,
    ProtocolErrorCode,
)
from chartreux.core.session.session_loader import (
    SessionFileContainmentError,
    SessionLoader,
)
from chartreux.core.subagents import UnknownAgentError


def _snapshot(parent: Path, child: Path) -> TranscriptReadSnapshot:
    return TranscriptReadSnapshot(
        agent_id="agent-1",
        child_session_id="child-1",
        child_dir=child,
        parent_dir=parent,
        parent_identity=("parent-1", 1),
        parent_runtime_token=1,
        link_identity=("child-1", "tool-1", "worker", "child"),
    )


def test_authorized_transcript_worker_reads_through_opened_child_directory(
    tmp_path: Path,
) -> None:
    child = tmp_path / "child"
    child.mkdir()
    (child / "meta.json").write_text(json.dumps({"total_messages": 1}))
    (child / "messages.jsonl").write_text(
        json.dumps({"role": "assistant", "message_id": "m-1", "content": "saved"})
        + "\n"
    )

    response = _read_authorized_agent_transcript(_snapshot(tmp_path, child), None, 50)

    assert response.state is AgentTranscriptState.AVAILABLE
    assert [entry.display_text for entry in response.entries or []] == ["saved"]


def test_authorized_transcript_worker_rejects_child_symlink_escape(
    tmp_path: Path,
) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir(exist_ok=True)
    child = tmp_path / "child"
    child.symlink_to(outside, target_is_directory=True)

    with pytest.raises(SessionFileContainmentError):
        _read_authorized_agent_transcript(_snapshot(tmp_path, child), None, 50)


def test_authorized_transcript_worker_rejects_intermediate_symlink_escape(
    tmp_path: Path,
) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    child = outside / "child"
    child.mkdir(parents=True, exist_ok=True)
    (child / "meta.json").write_text(json.dumps({"total_messages": 1}))
    (child / "messages.jsonl").write_text(
        json.dumps({"role": "assistant", "content": "secret"}) + "\n"
    )
    (tmp_path / "nested").symlink_to(outside, target_is_directory=True)

    with pytest.raises(SessionFileContainmentError):
        _read_authorized_agent_transcript(
            _snapshot(tmp_path, tmp_path / "nested" / "child"), None, 50
        )


def test_authorized_transcript_directory_fallback_rejects_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    child = outside / "child"
    child.mkdir(parents=True, exist_ok=True)
    (tmp_path / "nested").symlink_to(outside, target_is_directory=True)
    monkeypatch.delattr(os, "O_NOFOLLOW", raising=False)

    with pytest.raises(SessionFileContainmentError):
        _read_authorized_agent_transcript(
            _snapshot(tmp_path, tmp_path / "nested" / "child"), None, 50
        )


def test_authorized_transcript_worker_rejects_file_symlink_escape(
    tmp_path: Path,
) -> None:
    child = tmp_path / "child"
    child.mkdir()
    outside = tmp_path.parent / f"{tmp_path.name}-outside.json"
    outside.write_text(json.dumps({"total_messages": 0}))
    (child / "meta.json").symlink_to(outside)
    (child / "messages.jsonl").write_text("")

    with pytest.raises(SessionFileContainmentError):
        _read_authorized_agent_transcript(_snapshot(tmp_path, child), None, 50)


def test_authorized_transcript_worker_maps_missing_and_malformed_files_to_no_saved(
    tmp_path: Path,
) -> None:
    child = tmp_path / "child"
    child.mkdir()
    snapshot = _snapshot(tmp_path, child)

    assert _read_authorized_agent_transcript(snapshot, None, 50).state is (
        AgentTranscriptState.NO_SAVED_TRANSCRIPT
    )
    (child / "meta.json").write_text("{")
    (child / "messages.jsonl").write_text('{"role": "user", "content": "saved"}\n')
    assert _read_authorized_agent_transcript(snapshot, None, 50).state is (
        AgentTranscriptState.NO_SAVED_TRANSCRIPT
    )


class _TranscriptSessions:
    def __init__(self, snapshot: TranscriptReadSnapshot) -> None:
        self.snapshot = snapshot
        self.parent_identity = snapshot.parent_identity
        self.parent_runtime_token = snapshot.parent_runtime_token
        self.released = False

    async def resolve_transcript_read(self, agent_id: str) -> TranscriptReadSnapshot:
        assert agent_id == self.snapshot.agent_id
        return self.snapshot

    async def transcript_read_is_current(
        self, snapshot: TranscriptReadSnapshot
    ) -> bool:
        assert snapshot is self.snapshot
        return (
            not self.released
            and self.parent_identity == snapshot.parent_identity
            and self.parent_runtime_token == snapshot.parent_runtime_token
        )

    def switch_parent(self) -> None:
        self.parent_runtime_token += 1

    def release(self) -> None:
        self.released = True

    def change_generation(self) -> None:
        session_id, generation = self.parent_identity
        self.parent_identity = (session_id, generation + 1)

    async def ensure_child(self, _session_id: str) -> bool:
        raise AssertionError("transcript reads must not ensure a child runtime")


def _handler(sessions: _TranscriptSessions) -> CoreRequestHandler:
    handler = object.__new__(CoreRequestHandler)
    test_handler = cast(Any, handler)
    test_handler._agent_loop = SimpleNamespace(
        session_id="parent-1", _session_generation=1
    )
    test_handler._sessions = sessions
    test_handler._require_attached = lambda session_id: (
        None
        if session_id == "parent-1"
        else (_ for _ in ()).throw(AssertionError("unexpected parent"))
    )
    return handler


async def _dispatch(handler: CoreRequestHandler):
    return await handler._dispatch_agent("agent/transcript/get", {"agentId": "agent-1"})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("lifecycle_change", "invalidate"),
    [
        ("parent switch", "switch_parent"),
        ("release", "release"),
        ("generation change", "change_generation"),
    ],
)
async def test_lifecycle_change_while_worker_paused_does_not_disclose_transcript(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_change: str,
    invalidate: str,
) -> None:
    child = tmp_path / "child"
    child.mkdir()
    sessions = _TranscriptSessions(_snapshot(tmp_path, child))
    handler = _handler(sessions)
    started = asyncio.Event()
    release = asyncio.Event()

    def blocked_read(*_args: object) -> AgentTranscriptGetResponse:
        started.set()
        asyncio.run_coroutine_threadsafe(release.wait(), loop).result()
        return AgentTranscriptGetResponse(
            state=AgentTranscriptState.AVAILABLE,
            entries=[],
            oldest_cursor=None,
            has_more=False,
        )

    loop = asyncio.get_running_loop()
    monkeypatch.setattr(
        "chartreux.app_server._handler._read_authorized_agent_transcript", blocked_read
    )
    task = asyncio.create_task(_dispatch(handler))
    await asyncio.wait_for(started.wait(), timeout=1)

    # Each lifecycle change invalidates the snapshot before the worker releases.
    getattr(sessions, invalidate)()
    release.set()
    with pytest.raises(RequestFailure) as exc_info:
        await task
    assert exc_info.value.code is ProtocolErrorCode.CONFLICT
    assert lifecycle_change in {"parent switch", "release", "generation change"}


@pytest.mark.asyncio
async def test_transcript_read_runs_off_the_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    child = tmp_path / "child"
    child.mkdir()
    handler = _handler(_TranscriptSessions(_snapshot(tmp_path, child)))
    started = asyncio.Event()
    release = asyncio.Event()
    loop = asyncio.get_running_loop()

    def blocked_read(*_args: object) -> AgentTranscriptGetResponse:
        started.set()
        asyncio.run_coroutine_threadsafe(release.wait(), loop).result()
        return AgentTranscriptGetResponse(
            state=AgentTranscriptState.NO_SAVED_TRANSCRIPT
        )

    monkeypatch.setattr(
        "chartreux.app_server._handler._read_authorized_agent_transcript", blocked_read
    )
    task = asyncio.create_task(_dispatch(handler))
    await asyncio.wait_for(started.wait(), timeout=1)
    heartbeat = asyncio.create_task(asyncio.sleep(0))
    await asyncio.wait_for(heartbeat, timeout=1)
    release.set()
    response = cast(AgentTranscriptGetResponse, (await task).response)
    assert response.state is AgentTranscriptState.NO_SAVED_TRANSCRIPT


@pytest.mark.asyncio
async def test_transcript_dispatch_rejects_unknown_and_unattached_agents(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot(tmp_path, tmp_path / "child")
    sessions = _TranscriptSessions(snapshot)
    handler = _handler(sessions)

    async def unknown(_agent_id: str) -> TranscriptReadSnapshot:
        raise UnknownAgentError("unknown")

    sessions.resolve_transcript_read = unknown  # type: ignore[method-assign]
    with pytest.raises(RequestFailure) as exc_info:
        await _dispatch(handler)
    assert exc_info.value.code is ProtocolErrorCode.NOT_FOUND

    cast(Any, handler)._require_attached = lambda _session_id: (_ for _ in ()).throw(
        RequestFailure(ProtocolErrorCode.CONFLICT, "Session is not attached")
    )
    with pytest.raises(RequestFailure) as exc_info:
        await _dispatch(handler)
    assert exc_info.value.code is ProtocolErrorCode.CONFLICT


@pytest.mark.asyncio
async def test_transcript_read_reports_no_saved_transcript_without_child_link(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot(tmp_path, tmp_path / "child")
    snapshot = replace(snapshot, child_dir=None, parent_dir=None)
    result = await _dispatch(_handler(_TranscriptSessions(snapshot)))
    response = cast(AgentTranscriptGetResponse, result.response)
    assert response.state is AgentTranscriptState.NO_SAVED_TRANSCRIPT


@pytest.mark.asyncio
async def test_transcript_read_avoids_history_runtime_and_projection_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    child = tmp_path / "child"
    child.mkdir()
    (child / "meta.json").write_text(json.dumps({"total_messages": 1}))
    (child / "messages.jsonl").write_text(
        json.dumps({"role": "assistant", "message_id": "m-1", "content": "saved"})
        + "\n"
    )

    def fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("transcript read used a forbidden live/history path")

    monkeypatch.setattr(SessionLoader, "load_session", fail)
    monkeypatch.setattr("chartreux.app_server._handler.project_history", fail)
    result = await _dispatch(_handler(_TranscriptSessions(_snapshot(tmp_path, child))))
    response = cast(AgentTranscriptGetResponse, result.response)
    assert [entry.display_text for entry in response.entries or []] == ["saved"]


class _BlockingClient:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.late_response = asyncio.Event()

    async def request(self, _method: str, _params: object) -> dict[str, object]:
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            asyncio.create_task(self._record_late_response())
            raise
        return self._response()

    async def _record_late_response(self) -> None:
        await self.release.wait()
        self.late_response.set()

    @staticmethod
    def _response() -> dict[str, object]:
        return {
            "state": "available",
            "entries": [],
            "oldestCursor": None,
            "hasMore": False,
        }


class _Connection:
    def __init__(self, client: _BlockingClient) -> None:
        self.client = client

    async def connect(self) -> _BlockingClient:
        return self.client


@pytest.mark.asyncio
async def test_cancelled_transcript_read_cannot_adopt_a_late_response() -> None:
    client = _BlockingClient()
    state = SimpleNamespace(
        session_id="parent-1",
        projection=SimpleNamespace(history=["history"]),
        attachment="attached",
    )
    resource = SessionResource(_Connection(client), state)  # type: ignore[arg-type]
    initial_history = list(state.projection.history)
    initial_attachment = state.attachment

    task = asyncio.create_task(resource.read_agent_transcript("agent-1"))
    await asyncio.wait_for(client.started.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    client.release.set()
    await asyncio.wait_for(client.late_response.wait(), timeout=1)

    assert state.projection.history == initial_history
    assert state.attachment == initial_attachment


@pytest.mark.asyncio
async def test_transcript_facade_does_not_mutate_parent_state() -> None:
    client = _BlockingClient()
    client.release.set()
    state = SimpleNamespace(
        session_id="parent-1",
        projection=SimpleNamespace(history=["history"], event_id=7),
        attachment="attached",
    )
    resource = SessionResource(_Connection(client), state)  # type: ignore[arg-type]
    initial_history = list(state.projection.history)
    initial_attachment = state.attachment
    initial_event_id = state.projection.event_id

    response = await resource.read_agent_transcript("agent-1")

    assert response.state is AgentTranscriptState.AVAILABLE
    assert state.projection.history == initial_history
    assert state.attachment == initial_attachment
    assert state.projection.event_id == initial_event_id
