"""Resuming a session whose committed model left the catalog: the root-session
resume RPC recovers and demands an explicit model choice, while blueprint
resume (fresh session open) fails fast with an actionable error.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import time

import pytest

from chartreux.app_server._projection import committed_model_recovery_issue
from chartreux.app_server._runtime import (
    AgentRuntimeFactory,
    CommittedModelResumeError,
    _RootRuntimeBlueprint,
)
from chartreux.app_server.events import (
    HistoryEntryAdded,
    ServerError,
    TurnCompleted,
    TurnStarted,
)
from chartreux.app_server.models import (
    COMMITTED_MODEL_RECOVERY_ISSUE_FILE,
    PublicMessageEntry,
)
from chartreux.app_server.protocol import (
    AppServerResponseError,
    ClientCapabilities,
    ClientInfo,
    ProtocolErrorCode,
    SessionOptions,
)
from chartreux.core.config import ModelConfig, SessionLoggingConfig
from chartreux.core.config.harness_files import HarnessFilesManager
from chartreux.core.hooks.models import HookConfigResult
from chartreux.core.llm_models import LLMMessage, Role
from chartreux.core.session.session_loader import SessionLoader
from chartreux.core.session_types import LaunchMetadataV2, ScheduledLoop
from chartreux.utils.cache_store import FileSystemCacheStore
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.stubs.app_server import (
    attach_test_app_server_session,
    create_test_app_server_session,
    start_test_app_server,
)
from tests.stubs.fake_config_orchestrator import FakeConfigOrchestrator

_MODEL_A = ModelConfig(name="model-a", provider="mistral", alias="model-a")
_MODEL_B = ModelConfig(name="model-b", provider="mistral", alias="model-b")


def _saved_session_config(tmp_path: Path):
    return build_test_vibe_config(
        active_model="model-b",
        models=[_MODEL_A, _MODEL_B],
        session_logging=SessionLoggingConfig(enabled=True, save_dir=str(tmp_path)),
    )


async def _persist_target_session(tmp_path: Path) -> str:
    saved = build_test_agent_loop(config=_saved_session_config(tmp_path))
    saved.messages.append(LLMMessage(role=Role.user, content="target transcript"))
    await saved.persist_empty_session()
    target_session_id = saved.session_id
    await saved.aclose()
    return target_session_id


def _resuming_config(tmp_path: Path, *, with_target: bool):
    return build_test_vibe_config(
        models=[_MODEL_A, _MODEL_B] if with_target else [_MODEL_A],
        session_logging=SessionLoggingConfig(enabled=True, save_dir=str(tmp_path)),
    )


async def _consume(events):
    return [event async for event in events]


async def _recovery_context(tmp_path: Path, *, target_in_catalog: bool):
    target_session_id = await _persist_target_session(tmp_path)
    # The resuming process pins no model and its catalog lacks model-b.
    source = build_test_agent_loop(
        config=_resuming_config(tmp_path, with_target=target_in_catalog)
    )
    session = await create_test_app_server_session(source)
    return session, source, target_session_id


@pytest.mark.asyncio
async def test_resume_with_missing_committed_model_loads_transcript_and_gates_turns(
    tmp_path: Path,
) -> None:
    session, source, target_session_id = await _recovery_context(
        tmp_path, target_in_catalog=False
    )
    try:
        await session.resume(target_session_id)

        # The transcript is loaded and the unresolvable identity is dropped.
        assert source.session_id == target_session_id
        assert [
            message.content for message in source.messages if message.role is Role.user
        ] == ["target transcript"]
        assert source.committed_model is None

        # The pending choice is surfaced through the runtime snapshot.
        issue = committed_model_recovery_issue(source)
        assert issue is not None
        assert issue.file == COMMITTED_MODEL_RECOVERY_ISSUE_FILE
        runtime = session.resources.runtime
        await runtime.refresh()
        assert any(
            reported.file == COMMITTED_MODEL_RECOVERY_ISSUE_FILE
            for reported in runtime.issues
        )

        # Turns fail with an actionable error until a model is chosen.
        with pytest.raises(AppServerResponseError) as exc_info:
            await _consume(session.act("hello", client_message_id="u1"))
        assert exc_info.value.error.code is ProtocolErrorCode.INVALID_PARAMS
        assert "no longer available" in exc_info.value.error.message
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_resume_with_present_committed_model_is_unaffected(
    tmp_path: Path,
) -> None:
    session, source, target_session_id = await _recovery_context(
        tmp_path, target_in_catalog=True
    )
    try:
        await session.resume(target_session_id)

        assert source.committed_model is not None
        assert source.committed_model.base_model == "model-b"
        assert source.committed_model.provider == "mistral/default"
        assert committed_model_recovery_issue(source) is None
        runtime = session.resources.runtime
        await runtime.refresh()
        assert not any(
            reported.file == COMMITTED_MODEL_RECOVERY_ISSUE_FILE
            for reported in runtime.issues
        )
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_recovery_gate_opens_after_explicit_model_selection(
    tmp_path: Path,
) -> None:
    session, source, target_session_id = await _recovery_context(
        tmp_path, target_in_catalog=False
    )
    try:
        await session.resume(target_session_id)
        # The same write the model picker performs commits the explicit choice.
        await session.resources.config.update({"active_model": "model-a"})

        # The explicit selection unblocks the turn; no silent default is used.
        await _consume(session.act("hello", client_message_id="u1"))
        assert source.committed_model is not None
        assert source.committed_model.base_model == "model-a"
        assert committed_model_recovery_issue(source) is None
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_blueprint_resume_with_unusable_committed_model_fails_fast(
    tmp_path: Path,
) -> None:
    target_session_id = await _persist_target_session(tmp_path)
    blueprint = _RootRuntimeBlueprint(
        config_orchestrator=FakeConfigOrchestrator(
            _resuming_config(tmp_path, with_target=False)
        ),
        harness_files=HarnessFilesManager(),
        options=SessionOptions(cwd=str(tmp_path)),
        client_info=ClientInfo(name="recovery-test", version="1"),
        client_capabilities=ClientCapabilities(),
        hook_config_result=HookConfigResult(hooks=[], issues=[]),
        cache_store=FileSystemCacheStore(tmp_path / "cache.toml"),
    )

    with pytest.raises(CommittedModelResumeError) as exc_info:
        await AgentRuntimeFactory().resume_blueprint(blueprint, target_session_id)
    # The error is actionable: it names the committed model and the reason,
    # and points at the way out.
    message = str(exc_info.value)
    assert "model-b" in message
    assert "missing from the current catalog" in message
    assert "Choose another model" in message
    assert exc_info.value.code == "committed_model_missing"

    # The stored envelope is untouched: the failed resume rewrote nothing.
    session_dir = SessionLoader.find_session_by_id(
        target_session_id, _resuming_config(tmp_path, with_target=False).session_logging
    )
    assert session_dir is not None
    envelope = SessionLoader.load_metadata(session_dir).launch_config
    assert isinstance(envelope, LaunchMetadataV2)
    assert envelope.committed_model.base_model == "model-b"

    # The same failure surfaces as INVALID_PARAMS at the protocol layer.
    resumed_loop = build_test_agent_loop(
        config=_resuming_config(tmp_path, with_target=False)
    )
    with pytest.raises(AppServerResponseError) as exc_info:
        await attach_test_app_server_session(
            start_test_app_server(resumed_loop), resume_session_id=target_session_id
        )
    assert exc_info.value.error.code is ProtocolErrorCode.INVALID_PARAMS
    assert "model-b" in exc_info.value.error.message


@pytest.mark.asyncio
async def test_compact_is_gated_while_model_choice_is_pending(tmp_path: Path) -> None:
    session, source, target_session_id = await _recovery_context(
        tmp_path, target_in_catalog=False
    )
    try:
        await session.resume(target_session_id)

        # Compaction runs an LLM summarization, so it fails with the same
        # actionable error as a turn start instead of summarizing on a
        # silently-committed default.
        with pytest.raises(AppServerResponseError) as exc_info:
            await session.compact()
        assert exc_info.value.error.code is ProtocolErrorCode.INVALID_PARAMS
        assert "no longer available" in exc_info.value.error.message
        assert source.committed_model is None
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_unrelated_config_write_keeps_the_choice_pending(tmp_path: Path) -> None:
    session, source, target_session_id = await _recovery_context(
        tmp_path, target_in_catalog=False
    )
    try:
        await session.resume(target_session_id)

        # An unrelated config write must not silently commit the default or
        # dissolve the pending-choice issue.
        await session.resources.config.update({"theme": "dark"})
        assert source.committed_model is None
        assert committed_model_recovery_issue(source) is not None

        # An explicit active_model write still opens the gate.
        await session.resources.config.update({"active_model": "model-a"})
        assert source.committed_model is not None
        assert source.committed_model.base_model == "model-a"
        assert committed_model_recovery_issue(source) is None
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_enqueue_is_gated_while_model_choice_is_pending(tmp_path: Path) -> None:
    session, source, target_session_id = await _recovery_context(
        tmp_path, target_in_catalog=False
    )
    try:
        await session.resume(target_session_id)

        # Enqueue fails fast with the same actionable error as a turn start;
        # a permissive enqueue would strand the item, because promotion
        # failures inside queue tasks are silent and nothing re-triggers
        # promotion after a later selection.
        with pytest.raises(AppServerResponseError) as exc_info:
            await session.enqueue("queued while pending")
        assert exc_info.value.error.code is ProtocolErrorCode.INVALID_PARAMS
        assert "no longer available" in exc_info.value.error.message
        assert session.turn_queue.items == []

        # After an explicit selection, enqueue promotes and runs normally.
        await session.resources.config.update({"active_model": "model-a"})
        await session.enqueue("queued after choice")
        async with asyncio.timeout(5):
            async for event in session.events():
                if isinstance(event, TurnCompleted):
                    break
        assert source.committed_model is not None
    finally:
        await session.close()


async def _persist_target_session_with_due_loop(tmp_path: Path) -> str:
    saved = build_test_agent_loop(config=_saved_session_config(tmp_path))
    saved.messages.append(LLMMessage(role=Role.user, content="target transcript"))
    metadata = saved.session_logger.session_metadata
    assert metadata is not None
    now = time.time()
    metadata.loops = [
        ScheduledLoop(
            id="scheduled-1",
            interval_seconds=30,
            prompt="scheduled prompt",
            next_fire_at=now - 1,
            created_at=now - 31,
        )
    ]
    await saved.persist_empty_session()
    target_session_id = saved.session_id
    await saved.aclose()
    return target_session_id


@pytest.mark.asyncio
async def test_due_scheduled_loop_backs_off_until_model_choice(tmp_path: Path) -> None:
    target_session_id = await _persist_target_session_with_due_loop(tmp_path)
    source = build_test_agent_loop(config=_resuming_config(tmp_path, with_target=False))
    session = await create_test_app_server_session(source)
    try:
        await session.resume(target_session_id)
        assert source.committed_model is None
        resumed_at = time.time()

        # The due loop neither fires nor spams error notifications: it backs
        # off one full interval instead of failing on every scheduler pass.
        # (One scheduler pass lands within its 1s maximum sleep.)
        events = []
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(1.3):
                async for event in session.events():
                    events.append(event)
        assert not any(isinstance(event, TurnStarted) for event in events)
        assert not any(isinstance(event, ServerError) for event in events)
        metadata = source.session_logger.session_metadata
        assert metadata is not None
        loop = metadata.loops[0]
        assert loop.next_fire_at >= resumed_at + 29

        # After an explicit selection the loop fires normally again.
        await session.resources.config.update({"active_model": "model-a"})
        loop.next_fire_at = time.time() - 1
        fired = []
        async with asyncio.timeout(5):
            async for event in session.events():
                fired.append(event)
                if isinstance(event, TurnCompleted):
                    break
        assert any(isinstance(event, TurnStarted) for event in fired)
        assert any(
            isinstance(event, HistoryEntryAdded)
            and isinstance(event.entry, PublicMessageEntry)
            and event.entry.role == "user"
            and event.entry.text == "scheduled prompt"
            for event in fired
        )
    finally:
        await session.close()
