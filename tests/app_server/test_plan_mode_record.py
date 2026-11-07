"""The plan-mode record: written over the app server, restored on resume.

The record lives in the server's session metadata (design §7), so entry, the
resume restore, ``/plan cancel``, ``/clear`` and the ``/branch`` fork all cross
the app-server boundary. These tests drive a real app server and a real session
logger: the record has to be on disk, not merely in a wire response, and it has
to reach the client as part of the resume it restores from.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from chartreux.app_server.client import AppServerClient
from chartreux.app_server.protocol import (
    AppServerResponseError,
    ProtocolErrorCode,
    SessionPlanRecord,
    SessionPlanRecordWriteParams,
)
from chartreux.app_server.session import AppServerSession
from chartreux.app_server.transport import memory_transport_pair
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.config.models import SessionLoggingConfig
from chartreux.core.llm_models import LLMMessage, Role
from chartreux.core.session.session_loader import SessionLoader
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.stubs.app_server import attach_test_app_server_session, build_test_app_server

_RECORD = SessionPlanRecord(
    plan_path="/workspaces/chartreux/tasks/plan-abc123/plan.md",
    theme="Fix the widget",
    planner_model="glm-5-3",
    planner_thinking="high",
    pre_entry_active_model="mistral-large-4",
    pre_entry_planner_thinking="low",
)


def _loop(tmp_path: Path, *, enabled: bool = True) -> AgentLoop:
    config = build_test_vibe_config(
        session_logging=SessionLoggingConfig(
            enabled=enabled, save_dir=str(tmp_path / "sessions")
        )
    )
    return build_test_agent_loop(config=config, session_id="initial")


async def _serve(loop: AgentLoop) -> tuple[AppServerClient, AppServerSession]:
    client_transport, server_transport = memory_transport_pair()
    server = build_test_app_server(loop, server_transport)
    client = AppServerClient(client_transport, run_peer=server.serve)
    session = await attach_test_app_server_session(client)
    return client, session


def _sessions(session: AppServerSession) -> Any:
    return session.resources.sessions


@pytest.mark.asyncio
async def test_write_then_resume_carries_the_record_back(tmp_path: Path) -> None:
    loop = _loop(tmp_path)
    await loop.persist_empty_session()
    _, session = await _serve(loop)

    written = await _sessions(session).write_plan_record(_RECORD)
    await session.resume(session.session_id)

    assert written == _RECORD
    assert _sessions(session).resumed_plan_record == _RECORD


@pytest.mark.asyncio
async def test_the_record_reaches_the_session_metadata_file(tmp_path: Path) -> None:
    loop = _loop(tmp_path)
    await loop.persist_empty_session()
    session_dir = loop.session_logger.session_dir
    assert session_dir is not None
    _, session = await _serve(loop)

    await _sessions(session).write_plan_record(_RECORD)

    persisted = SessionLoader.load_metadata(session_dir)
    assert persisted.plan_mode is not None
    assert persisted.plan_mode.plan_path == _RECORD.plan_path
    assert persisted.plan_mode.theme == "Fix the widget"
    assert persisted.plan_mode.planner_model == "glm-5-3"
    assert persisted.plan_mode.pre_entry_active_model == "mistral-large-4"


@pytest.mark.asyncio
async def test_a_resumed_session_without_a_record_adopts_none(tmp_path: Path) -> None:
    loop = _loop(tmp_path)
    await loop.persist_empty_session()
    _, session = await _serve(loop)

    await session.resume(session.session_id)

    assert _sessions(session).resumed_plan_record is None


@pytest.mark.asyncio
async def test_clearing_removes_the_record_from_memory_and_disk(tmp_path: Path) -> None:
    loop = _loop(tmp_path)
    await loop.persist_empty_session()
    session_dir = loop.session_logger.session_dir
    assert session_dir is not None
    _, session = await _serve(loop)
    await _sessions(session).write_plan_record(_RECORD)

    cleared = await _sessions(session).clear_plan_record()
    await session.resume(session.session_id)

    assert cleared is None
    assert _sessions(session).resumed_plan_record is None
    assert SessionLoader.load_metadata(session_dir).plan_mode is None
    assert loop.session_logger.session_metadata is not None
    assert loop.session_logger.session_metadata.plan_mode is None


@pytest.mark.asyncio
async def test_a_write_returns_the_persisted_record_not_the_request(
    tmp_path: Path,
) -> None:
    # Logging disabled means nothing is stored, and the response says so
    # instead of reporting a record that no resume could ever restore.
    loop = _loop(tmp_path, enabled=False)
    assert loop.session_logger.session_metadata is None
    _, session = await _serve(loop)

    written = await _sessions(session).write_plan_record(_RECORD)

    assert written is None


@pytest.mark.asyncio
async def test_a_write_for_another_session_is_refused(tmp_path: Path) -> None:
    loop = _loop(tmp_path)
    await loop.persist_empty_session()
    client, _ = await _serve(loop)

    with pytest.raises(AppServerResponseError) as refused:
        await client.request(
            "session/plan/record/write",
            SessionPlanRecordWriteParams(
                session_id="some-other-session", record=_RECORD
            ),
        )

    assert refused.value.error.code is ProtocolErrorCode.INVALID_PARAMS


@pytest.mark.asyncio
async def test_a_branch_fork_inherits_the_record(tmp_path: Path) -> None:
    # /branch shares the plan document and carries the mode with it: resuming
    # the fork is resuming a planning session (accepted behavior, documented).
    loop = _loop(tmp_path)
    await loop.persist_empty_session()
    loop.messages.append(LLMMessage(role=Role.user, content="Start the plan"))
    _, session = await _serve(loop)
    await _sessions(session).write_plan_record(_RECORD)

    fork = await _sessions(session).fork(attach=False)

    forked_metadata = SessionLoader.load_metadata(
        _session_dir_of(tmp_path, fork.state.session.id)
    )
    assert forked_metadata.plan_mode is not None
    assert forked_metadata.plan_mode.plan_path == _RECORD.plan_path
    assert forked_metadata.plan_mode.theme == "Fix the widget"
    assert forked_metadata.session_id != loop.session_id


def _session_dir_of(tmp_path: Path, session_id: str) -> Path:
    seen: list[str] = []
    for metadata_path in tmp_path.rglob("meta.json"):
        loaded = SessionLoader.load_metadata(metadata_path.parent)
        seen.append(f"{metadata_path.parent.name}={loaded.session_id}")
        if loaded.session_id == session_id:
            return metadata_path.parent
    raise AssertionError(f"no session directory for {session_id}; saw {seen}")
