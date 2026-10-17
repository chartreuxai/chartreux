"""Session logger generation fence tests (launch-envelope persistence and the save lock: stale-save fencing before transcript and metadata, worker draining under the save lock, launch-envelope model validation, envelope survival across interaction saves and failed updates, cancelled lock waits, and in-flight unchanged saves not clearing the dirty envelope)."""

from __future__ import annotations

import asyncio
import json
from threading import Event
from typing import Any

import pytest

from chartreux.core.agents.models import AgentProfile
from chartreux.core.config import ChartreuxConfigSchema, SessionLoggingConfig
from chartreux.core.llm_models import LLMMessage, Role
from chartreux.core.session.session_loader import SessionLoader
from chartreux.core.session.session_logger import SessionLogger
from chartreux.core.session_types import (
    AgentStats,
    CommittedModelIdentity,
    LaunchMetadataV1,
    LaunchMetadataV2,
    LaunchPersonaV1,
)
from chartreux.core.subagents import LaunchConfig, LaunchToolOverride
from chartreux.core.tools.manager import ToolManager
from tests.session.persistence._session_persistence_support import (
    mock_agent_profile as mock_agent_profile,
    mock_tool_manager as mock_tool_manager,
    mock_vibe_config as mock_vibe_config,
    session_config as session_config,
    temp_session_dir as temp_session_dir,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("rewrite", [False, True])
async def test_stale_save_is_fenced_before_transcript_and_metadata(
    session_config: SessionLoggingConfig,
    mock_vibe_config: ChartreuxConfigSchema,
    mock_tool_manager: ToolManager,
    rewrite: bool,
) -> None:
    logger = SessionLogger(session_config, "generation-fence")
    first = LLMMessage(role=Role.user, content="first")
    second = LLMMessage(role=Role.assistant, content="second")

    async def save(messages: list[LLMMessage]) -> None:
        await logger.save_interaction(
            messages, AgentStats(), mock_vibe_config, mock_tool_manager, None
        )

    await save([first, second])
    await logger._save_lock.acquire()
    stale_messages = (
        [LLMMessage(role=Role.user, content="edited")]
        if rewrite
        else [first, second, first]
    )
    stale = asyncio.create_task(save(stale_messages))
    await asyncio.sleep(0)  # snapshot captured; queued behind rewind's lock
    try:
        logger.invalidate_transcript_cursor()
        assert logger.session_metadata is not None
        assert logger.session_dir is not None
        # Commit the rewind while holding the same serialization lock.
        await asyncio.to_thread(
            logger._save_full_verify,
            [first],
            AgentStats(),
            mock_tool_manager,
            None,
            logger.session_dir,
            logger.session_metadata,
            False,
        )
        transcript = logger.messages_filepath.read_bytes()
        metadata = logger.metadata_filepath.read_bytes()
    finally:
        logger._save_lock.release()
    await stale
    assert logger.messages_filepath.read_bytes() == transcript
    assert logger.metadata_filepath.read_bytes() == metadata
    assert [json.loads(line)["content"] for line in transcript.splitlines()] == [
        "first"
    ]


@pytest.mark.asyncio
async def test_repeated_cancellation_drains_worker_under_save_lock(
    session_config: SessionLoggingConfig,
    mock_vibe_config: ChartreuxConfigSchema,
    mock_tool_manager: ToolManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger = SessionLogger(session_config, "cancel-drain")
    started, release = Event(), Event()
    method = "_save_interaction_sync"
    original = getattr(logger, method)
    calls = 0

    def gated(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 1:
            started.set()
            assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(logger, method, gated)

    async def save() -> None:
        await logger.save_interaction(
            [LLMMessage(role=Role.user, content="test")],
            AgentStats(),
            mock_vibe_config,
            mock_tool_manager,
            None,
        )

    first = asyncio.create_task(save())
    assert await asyncio.to_thread(started.wait, 5)
    second = asyncio.create_task(save())
    try:
        for _ in range(3):
            first.cancel()
            await asyncio.sleep(0)
            assert logger._save_lock.locked()
            assert calls == 1
        assert not second.done()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await first
    await second
    assert calls == 2


def test_v1_launch_envelope_rejects_committed_identity_extra() -> None:
    with pytest.raises(ValueError, match="committed_model"):
        LaunchMetadataV1.model_validate({
            "version": 1,
            "profile": "test-agent",
            "overrides": {},
            "persona": {"system_prompt_id": "tests", "instructions": None},
            "committed_model": {
                "base_model": "test",
                "provider": "test-provider",
                "wire_name": "test-wire",
                "catalog_revision": "revision-1",
            },
        })


@pytest.mark.asyncio
async def test_launch_envelope_survives_interaction_saves_and_failed_update(
    session_config: SessionLoggingConfig,
    mock_vibe_config: ChartreuxConfigSchema,
    mock_tool_manager: ToolManager,
    mock_agent_profile: AgentProfile,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dedicated envelope writer is authoritative across every metadata path."""
    logger = SessionLogger(session_config, "launch-envelope")
    initial = LaunchMetadataV2(
        version=2,
        profile="test-agent",
        overrides=LaunchConfig(
            model="test",
            instructions="Frozen role",
            tools={"read_file": LaunchToolOverride(allowlist=["*.py"])},
        ),
        persona=LaunchPersonaV1(system_prompt_id="tests", instructions="Frozen role"),
        committed_model=CommittedModelIdentity(
            base_model="test",
            provider="test-provider",
            wire_name="test-wire",
            catalog_revision="revision-1",
        ),
    )
    await logger.persist_launch_config(initial)
    assert logger.session_dir is not None
    metadata_path = logger.session_dir / "meta.json"
    assert SessionLoader.load_metadata(logger.session_dir).launch_config == initial

    messages = [
        LLMMessage(role=Role.system, content="System prompt"),
        LLMMessage(role=Role.user, content="Hello"),
    ]
    await logger.save_interaction(
        messages, AgentStats(), mock_vibe_config, mock_tool_manager, mock_agent_profile
    )
    assert SessionLoader.load_metadata(logger.session_dir).launch_config == initial

    committed = initial.model_copy(
        update={
            "overrides": LaunchConfig(
                model="test", thinking="high", instructions="Frozen role"
            )
        }
    )
    await logger.persist_launch_config(committed)
    assert SessionLoader.load_metadata(logger.session_dir).launch_config == committed

    async def fail_write(*_args, **_kwargs) -> None:
        raise RuntimeError("injected metadata failure")

    monkeypatch.setattr(SessionLogger, "persist_metadata", fail_write)
    with pytest.raises(RuntimeError, match="injected metadata failure"):
        await logger.persist_launch_config(initial)

    # A post-commit write failure never rolls disk state back to an older envelope.
    assert (
        json.loads(metadata_path.read_text(encoding="utf-8"))["launch_config"][
            "overrides"
        ]["thinking"]
        == "high"
    )


@pytest.mark.asyncio
async def test_cancelled_launch_persistence_lock_wait_keeps_new_envelope_authoritative(
    session_config: SessionLoggingConfig,
    mock_vibe_config: ChartreuxConfigSchema,
    mock_tool_manager: ToolManager,
    mock_agent_profile: AgentProfile,
) -> None:
    logger = SessionLogger(session_config, "launch-envelope-cancel")
    old = LaunchMetadataV1(
        version=1,
        profile="test-agent",
        overrides=LaunchConfig(model="test", thinking="off"),
        persona=LaunchPersonaV1(system_prompt_id="tests", instructions=None),
    )
    new = old.model_copy(
        update={"overrides": LaunchConfig(model="test", thinking="high")}
    )
    await logger.persist_launch_config(old)

    await logger._save_lock.acquire()
    update = asyncio.create_task(logger.persist_launch_config(new))
    await asyncio.sleep(0)
    assert logger.session_metadata is not None
    assert logger.session_metadata.launch_config == new
    update.cancel()
    with pytest.raises(asyncio.CancelledError):
        await update
    logger._save_lock.release()

    await logger.save_interaction(
        [LLMMessage(role=Role.user, content="ordinary save")],
        AgentStats(),
        mock_vibe_config,
        mock_tool_manager,
        mock_agent_profile,
    )
    assert logger.session_dir is not None
    assert SessionLoader.load_metadata(logger.session_dir).launch_config == new


@pytest.mark.asyncio
async def test_inflight_unchanged_save_cannot_clear_new_launch_envelope_dirty_state(
    session_config: SessionLoggingConfig,
    mock_vibe_config: ChartreuxConfigSchema,
    mock_tool_manager: ToolManager,
    mock_agent_profile: AgentProfile,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger = SessionLogger(session_config, "launch-envelope-inflight")
    old = LaunchMetadataV1(
        version=1,
        profile="test-agent",
        overrides=LaunchConfig(model="test", thinking="off"),
        persona=LaunchPersonaV1(system_prompt_id="tests", instructions=None),
    )
    new = old.model_copy(
        update={"overrides": LaunchConfig(model="test", thinking="high")}
    )
    messages = [LLMMessage(role=Role.user, content="unchanged")]
    await logger.persist_launch_config(old)
    await logger.save_interaction(
        messages, AgentStats(), mock_vibe_config, mock_tool_manager, mock_agent_profile
    )

    started, release = Event(), Event()
    original = logger._save_interaction_sync

    def gated_save(*args, **kwargs) -> None:
        started.set()
        assert release.wait(timeout=5)
        original(*args, **kwargs)

    monkeypatch.setattr(logger, "_save_interaction_sync", gated_save)
    ordinary_save = asyncio.create_task(
        logger.save_interaction(
            messages,
            AgentStats(),
            mock_vibe_config,
            mock_tool_manager,
            mock_agent_profile,
        )
    )
    assert await asyncio.to_thread(started.wait, 5)
    dedicated_save = asyncio.create_task(logger.persist_launch_config(new))
    await asyncio.sleep(0)
    dedicated_save.cancel()
    release.set()
    await ordinary_save
    with pytest.raises(asyncio.CancelledError):
        await dedicated_save

    assert logger._launch_config_dirty is True
    assert logger.session_dir is not None
    assert SessionLoader.load_metadata(logger.session_dir).launch_config == old

    await logger.save_interaction(
        messages, AgentStats(), mock_vibe_config, mock_tool_manager, mock_agent_profile
    )
    assert logger._launch_config_dirty is False
    assert SessionLoader.load_metadata(logger.session_dir).launch_config == new
