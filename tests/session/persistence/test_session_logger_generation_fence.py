"""Session logger generation fence tests (launch-envelope persistence and the save lock: stale-save fencing before transcript and metadata, worker draining under the save lock, launch-envelope model validation, envelope survival across interaction saves and failed updates, cancelled lock waits, and in-flight unchanged saves not clearing the dirty envelope).

P4a4 adds the archive trim publication fence (M1 §4 step 6):
``SessionLogger.fence_and_publish_archive_trim`` fences QUEUED STALE SAVES and
publishes the DESCRIPTOR synchronously — one plain ``def`` block, no
intervening await, no reset(), no reset hooks — with TEST-DOUBLE replacement
objects (the P4b replacement model lands later; the real
replacement-publication fence is asserted in P4d's e2e cycle). Save/reset/
cancellation generations are all handled: a stale descriptor never overwrites
the current one.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
import hashlib
import inspect
import json
import os
from pathlib import Path
from threading import Event, get_ident
from typing import Any

import pytest

from chartreux.core.agent_loop._transcript_integrity import (
    ARCHIVE_REFERENCE_ATTRIBUTE,
    ArchivedPrefixAuthority,
    TranscriptIntegrityError,
)
from chartreux.core.config import ChartreuxConfigSchema, SessionLoggingConfig
from chartreux.core.llm_models import (
    FunctionCall,
    LLMMessage,
    PersistedToolResult,
    Role,
    ToolCall,
)
from chartreux.core.message_list import MessageList
from chartreux.core.session.session_loader import SessionLoader
from chartreux.core.session.session_logger import SessionLogger
from chartreux.core.session.transcript_archive import ArchivedPrefixV1
from chartreux.core.session_types import (
    AgentStats,
    CommittedModelIdentity,
    LaunchMetadataV1,
    LaunchMetadataV2,
    LaunchPersonaV1,
    LaunchPersonaV2,
)
from chartreux.core.subagents import LaunchConfig, LaunchToolOverride
from chartreux.core.tools.manager import ToolManager
from tests.session.persistence._session_persistence_support import (
    disabled_session_config as disabled_session_config,
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
            messages, AgentStats(), mock_vibe_config, mock_tool_manager
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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dedicated envelope writer is authoritative across every metadata path."""
    logger = SessionLogger(session_config, "launch-envelope")
    initial = LaunchMetadataV2(
        version=2,
        overrides=LaunchConfig(
            model="test",
            instructions="Frozen role",
            tools={"read_file": LaunchToolOverride(allowlist=["*.py"])},
        ),
        persona=LaunchPersonaV2(
            identity="coder", identity_revision="rev-1", instructions="Frozen role"
        ),
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
        messages, AgentStats(), mock_vibe_config, mock_tool_manager
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
    )
    assert logger.session_dir is not None
    assert SessionLoader.load_metadata(logger.session_dir).launch_config == new


@pytest.mark.asyncio
async def test_inflight_unchanged_save_cannot_clear_new_launch_envelope_dirty_state(
    session_config: SessionLoggingConfig,
    mock_vibe_config: ChartreuxConfigSchema,
    mock_tool_manager: ToolManager,
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
        messages, AgentStats(), mock_vibe_config, mock_tool_manager
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
            messages, AgentStats(), mock_vibe_config, mock_tool_manager
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
        messages, AgentStats(), mock_vibe_config, mock_tool_manager
    )
    assert logger._launch_config_dirty is False
    assert SessionLoader.load_metadata(logger.session_dir).launch_config == new


# ---------------------------------------------------------------------------
# P4a4: the generation fence + synchronous descriptor publication (M1 §4
# step 6). The P4b replacement model does not exist yet, so the publish tests
# use TEST-DOUBLE replacement objects.
# ---------------------------------------------------------------------------


def _fence_user(content: str, message_id: str) -> LLMMessage:
    return LLMMessage(role=Role.user, content=content, message_id=message_id)


def _fence_assistant(content: str, message_id: str) -> LLMMessage:
    return LLMMessage(role=Role.assistant, content=content, message_id=message_id)


def _fence_boundary() -> LLMMessage:
    return LLMMessage(
        role=Role.user,
        content="compaction boundary",
        injected=True,
        context_boundary="compaction",
    )


def _trimmed_test_double(message: LLMMessage, reference: str) -> LLMMessage:
    """A TEST-DOUBLE payload-trim replacement (P4b's model lands later).

    Drops ``reasoning_payloads`` (the declared ``payloads`` trim) and carries
    the private archive reference a P4b archived message would carry.
    """
    replacement = message.model_copy(update={"reasoning_payloads": None})
    object.__setattr__(replacement, ARCHIVE_REFERENCE_ATTRIBUTE, reference)
    return replacement


async def _fence_sealed_logger(
    session_config: SessionLoggingConfig,
    session_id: str,
    mock_vibe_config: ChartreuxConfigSchema,
    mock_tool_manager: ToolManager,
) -> tuple[SessionLogger, MessageList, ArchivedPrefixV1]:
    """A logger whose closed prefix (u1, a1) is sealed (P4a2 receipt).

    The live list is [system, u1, a1, boundary, u2, a2]; the sealed prefix is
    the settled round (u1, a1) strictly before the newest boundary.
    """
    logger = SessionLogger(session_config, session_id)
    messages = MessageList([
        LLMMessage(role=Role.system, content="prompt"),
        _fence_user("u1", "u1"),
        LLMMessage(
            role=Role.assistant,
            content="a1",
            message_id="a1",
            reasoning_payloads=[{"opaque": "payload-1"}],
        ),
        _fence_boundary(),
        _fence_user("u2", "u2"),
        _fence_assistant("a2", "a2"),
    ])
    receipt = await logger.seal_archive(
        messages, AgentStats(steps=1), mock_vibe_config, mock_tool_manager
    )
    return logger, messages, receipt.descriptor


async def _fence_save(
    logger: SessionLogger,
    messages: Any,
    mock_vibe_config: ChartreuxConfigSchema,
    mock_tool_manager: ToolManager,
    *,
    steps: int = 1,
) -> None:
    await logger.save_interaction(
        messages, AgentStats(steps=steps), mock_vibe_config, mock_tool_manager
    )


def _fence_messages_path(logger: SessionLogger) -> Path:
    assert logger.session_dir is not None
    return logger.session_dir / "messages.jsonl"


class TestP4a4FencePublishesDescriptorSynchronously:
    def test_fence_method_is_a_plain_sync_def(self) -> None:
        # The fence + publish must be one synchronous block: no await can
        # intervene between fencing queued stale saves and publishing the
        # descriptor.
        assert not inspect.iscoroutinefunction(
            SessionLogger.fence_and_publish_archive_trim
        )
        assert not asyncio.iscoroutinefunction(
            SessionLogger.fence_and_publish_archive_trim
        )

    @pytest.mark.asyncio
    async def test_queued_stale_save_is_fenced_and_descriptor_published(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
    ) -> None:
        logger, messages, descriptor = await _fence_sealed_logger(
            session_config, "p4a4-fence-stale", mock_vibe_config, mock_tool_manager
        )
        await _fence_save(logger, messages, mock_vibe_config, mock_tool_manager)
        transcript_before = _fence_messages_path(logger).read_bytes()
        # The publish plans against the CURRENT generations.
        save_generation = logger._transcript_cursor_generation
        reset_generation = logger._archive_generation

        # Queue a stale save behind the save lock: its snapshot is captured,
        # then it waits on the lock.
        await logger._save_lock.acquire()
        stale = asyncio.create_task(
            _fence_save(
                logger,
                [_fence_user("stale", "stale")],
                mock_vibe_config,
                mock_tool_manager,
                steps=2,
            )
        )
        await asyncio.sleep(0)  # snapshot captured; queued on the lock
        try:
            # Fence + publish synchronously — the fence needs no lock and no
            # await, so it lands atomically between the queued snapshot and
            # any later save.
            replacements = [
                _trimmed_test_double(messages[1], "ref-u1"),
                _trimmed_test_double(messages[2], "ref-a1"),
            ]
            logger.fence_and_publish_archive_trim(
                messages,
                descriptor,
                replacements,
                save_generation=save_generation,
                reset_generation=reset_generation,
            )
        finally:
            logger._save_lock.release()
        await stale

        # The queued stale save discarded itself under the lock: it wrote
        # nothing to the transcript and did not publish its snapshot.
        assert _fence_messages_path(logger).read_bytes() == transcript_before
        assert logger.archive_descriptor is descriptor
        snapshot = logger.last_save_archive_snapshot
        assert snapshot is not None
        assert snapshot.descriptor is descriptor
        assert snapshot.message_epoch == descriptor.message_epoch
        assert snapshot.is_authoritative
        # The fence moved the save generation and cleared the cursor.
        assert logger._transcript_cursor is None
        assert logger._transcript_cursor_generation == save_generation + 1

    @pytest.mark.asyncio
    async def test_cancelled_in_flight_save_cannot_overwrite_the_published_descriptor(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Cancellation generation: a save cancelled mid-worker drains under
        the save lock; the fence publishes while it drains, and the drained
        save's post-drain publication is discarded by the generation fence.
        """
        logger, messages, descriptor = await _fence_sealed_logger(
            session_config, "p4a4-fence-cancel", mock_vibe_config, mock_tool_manager
        )
        await _fence_save(logger, messages, mock_vibe_config, mock_tool_manager)

        started, release = Event(), Event()
        original = logger._save_interaction_sync
        calls = 0

        def gated(*args: Any, **kwargs: Any) -> Any:
            nonlocal calls
            calls += 1
            if calls == 1:
                started.set()
                assert release.wait(5)
            return original(*args, **kwargs)

        monkeypatch.setattr(logger, "_save_interaction_sync", gated)
        task = asyncio.create_task(
            _fence_save(
                logger,
                [_fence_user("inflight", "inflight")],
                mock_vibe_config,
                mock_tool_manager,
                steps=2,
            )
        )
        assert await asyncio.to_thread(started.wait, 5)
        task.cancel()
        await asyncio.sleep(0)
        await asyncio.sleep(0)  # the save is now draining under the lock
        assert logger._save_lock.locked()

        # The synchronous fence runs during the drain (it takes no lock).
        save_generation = logger._transcript_cursor_generation
        reset_generation = logger._archive_generation
        replacements = [
            _trimmed_test_double(messages[1], "ref-u1"),
            _trimmed_test_double(messages[2], "ref-a1"),
        ]
        logger.fence_and_publish_archive_trim(
            messages,
            descriptor,
            replacements,
            save_generation=save_generation,
            reset_generation=reset_generation,
        )
        published_snapshot = logger.last_save_archive_snapshot
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

        # The drained save republished nothing: the fence's publish is last.
        assert logger.archive_descriptor is descriptor
        assert logger.last_save_archive_snapshot is published_snapshot
        assert published_snapshot is not None and published_snapshot.is_authoritative
        # And its cursor result was discarded by the generation fence.
        assert logger._transcript_cursor is None


class TestP4a4PublishIntegrityConditions:
    @pytest.mark.asyncio
    async def test_publish_is_length_preserving_and_preserves_system_messages(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """M1 §4: no slot insert/delete during trim; system messages preserved."""
        logger, messages, descriptor = await _fence_sealed_logger(
            session_config, "p4a4-length", mock_vibe_config, mock_tool_manager
        )
        before = list(messages)
        system_message = messages[0]
        mutations = {"append": 0, "insert": 0, "reset": 0}
        real_append, real_insert, real_reset = (
            MessageList.append,
            MessageList.insert,
            MessageList.reset,
        )

        def spy_append(self: MessageList, msg: LLMMessage) -> None:
            mutations["append"] += 1
            real_append(self, msg)

        def spy_insert(self: MessageList, i: int, msg: LLMMessage) -> None:
            mutations["insert"] += 1
            real_insert(self, i, msg)

        def spy_reset(self: MessageList, new: list[LLMMessage]) -> None:
            mutations["reset"] += 1
            real_reset(self, new)

        monkeypatch.setattr(MessageList, "append", spy_append)
        monkeypatch.setattr(MessageList, "insert", spy_insert)
        monkeypatch.setattr(MessageList, "reset", spy_reset)

        replacements = [
            _trimmed_test_double(messages[1], "ref-u1"),
            _trimmed_test_double(messages[2], "ref-a1"),
        ]
        logger.fence_and_publish_archive_trim(
            messages,
            descriptor,
            replacements,
            save_generation=logger._transcript_cursor_generation,
            reset_generation=logger._archive_generation,
        )

        assert mutations == {"append": 0, "insert": 0, "reset": 0}
        # Same length, same slots: only the sealed prefix slots were replaced
        # in place (absolute indexes 1 and 2 — the system message at 0 is
        # untouched, the live tail from 3 on is the same objects).
        assert len(messages) == len(before)
        assert messages[0] is system_message
        assert messages[0].role is Role.system
        assert messages[1] is replacements[0]
        assert messages[2] is replacements[1]
        for index in range(3, len(before)):
            assert messages[index] is before[index]

    @pytest.mark.asyncio
    async def test_publish_fires_no_reset_hooks(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """M1 §4: no reset() and no reset hooks in the fence path."""
        logger, messages, descriptor = await _fence_sealed_logger(
            session_config, "p4a4-hooks", mock_vibe_config, mock_tool_manager
        )
        fired: list[bool] = []
        messages.on_reset(lambda: fired.append(True))

        def forbidden_reset(self: MessageList, new: list[LLMMessage]) -> None:
            raise AssertionError("reset() must not be called in the fence path")

        monkeypatch.setattr(MessageList, "reset", forbidden_reset)
        monkeypatch.setattr(MessageList, "reset_preserving_system", forbidden_reset)
        replacements = [
            _trimmed_test_double(messages[1], "ref-u1"),
            _trimmed_test_double(messages[2], "ref-a1"),
        ]
        logger.fence_and_publish_archive_trim(
            messages,
            descriptor,
            replacements,
            save_generation=logger._transcript_cursor_generation,
            reset_generation=logger._archive_generation,
        )
        assert fired == []
        # The descriptor is published on BOTH owners: the list and the logger.
        assert messages.archive_descriptor is descriptor
        assert logger.archive_descriptor is descriptor

    @pytest.mark.asyncio
    async def test_save_after_publish_stays_archive_aware_and_preserves_the_prefix(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
    ) -> None:
        """After the fence publish, ordinary saves serialize only the live
        tail and the archived bytes on disk stay byte-identical (P4a3 row 1
        still holds under a published descriptor).
        """
        logger, messages, descriptor = await _fence_sealed_logger(
            session_config, "p4a4-save-after", mock_vibe_config, mock_tool_manager
        )
        original_prefix = _fence_messages_path(logger).read_bytes()[
            : descriptor.prefix_end_byte
        ]
        replacements = [
            _trimmed_test_double(messages[1], "ref-u1"),
            _trimmed_test_double(messages[2], "ref-a1"),
        ]
        logger.fence_and_publish_archive_trim(
            messages,
            descriptor,
            replacements,
            save_generation=logger._transcript_cursor_generation,
            reset_generation=logger._archive_generation,
        )

        messages.append(_fence_user("u3", "u3"))
        await _fence_save(
            logger, messages, mock_vibe_config, mock_tool_manager, steps=2
        )

        raw = _fence_messages_path(logger).read_bytes()
        assert raw[: descriptor.prefix_end_byte] == original_prefix
        assert (
            hashlib.sha256(raw[: descriptor.prefix_end_byte]).hexdigest()
            == descriptor.prefix_sha256
        )
        records = [json.loads(line) for line in raw.splitlines()]
        assert [record.get("content") for record in records] == [
            "u1",
            "a1",
            "compaction boundary",
            "u2",
            "a2",
            "u3",
        ]
        # The disk bytes are the ORIGINAL archived prefix (the trim lives in
        # RAM only): records[1] still carries reasoning_payloads on disk,
        # while the in-RAM replacement is the declared payloads trim and
        # carries the archive marker.
        assert "reasoning_payloads" in records[1]
        assert records[1]["role"] == "assistant"
        assert messages[2].reasoning_payloads is None
        assert getattr(messages[2], ARCHIVE_REFERENCE_ATTRIBUTE, None) == "ref-a1"
        assert logger.last_save_diagnostic is None

    @pytest.mark.asyncio
    async def test_mutation_inside_the_sealed_prefix_fails_closed_at_the_save_boundary(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
    ) -> None:
        """M1 §4: a mutation inside the sealed prefix is rejected — the marker
        contiguity check fails closed at the save boundary.
        """
        logger, messages, descriptor = await _fence_sealed_logger(
            session_config, "p4a4-mutation", mock_vibe_config, mock_tool_manager
        )
        replacements = [
            _trimmed_test_double(messages[1], "ref-u1"),
            _trimmed_test_double(messages[2], "ref-a1"),
        ]
        logger.fence_and_publish_archive_trim(
            messages,
            descriptor,
            replacements,
            save_generation=logger._transcript_cursor_generation,
            reset_generation=logger._archive_generation,
        )
        # Mutate a sealed slot in place with an UNMARKED message.
        messages.replace(1, _fence_user("mutated", "u1"))
        with pytest.raises(TranscriptIntegrityError) as exc_info:
            await _fence_save(
                logger, messages, mock_vibe_config, mock_tool_manager, steps=2
            )
        kinds = {violation.kind for violation in exc_info.value.violations}
        assert "unauthorized_archived_prefix" in kinds


class TestP4a4StaleDescriptorNeverOverwrites:
    @pytest.mark.asyncio
    async def test_stale_epoch_descriptor_is_refused_and_keeps_the_current_one(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
    ) -> None:
        logger, messages, descriptor = await _fence_sealed_logger(
            session_config, "p4a4-stale-epoch", mock_vibe_config, mock_tool_manager
        )
        assert logger.archive_descriptor is descriptor  # epoch 0
        # The list rebinds: epoch 1. A fresh epoch-1 descriptor publishes fine.
        messages.reset([_fence_user("rewound", "u9")])
        assert messages.message_epoch == 1
        newer = ArchivedPrefixV1(
            session_id=logger.session_id,
            message_epoch=1,
            archived_count=1,
            prefix_end_byte=10,
            prefix_sha256=hashlib.sha256(b"0123456789").hexdigest(),
            boundary_message_id=None,
            trim_mode="payloads",
        )
        logger.fence_and_publish_archive_trim(
            messages,
            newer,
            [_trimmed_test_double(messages[0], "ref")],
            save_generation=logger._transcript_cursor_generation,
            reset_generation=logger._archive_generation,
        )
        assert logger.archive_descriptor is newer
        # The OLD epoch-0 descriptor is stale: it never overwrites.
        with pytest.raises(ValueError, match="stale"):
            logger.fence_and_publish_archive_trim(
                messages,
                descriptor,
                [_trimmed_test_double(messages[0], "ref")],
                save_generation=logger._transcript_cursor_generation,
                reset_generation=logger._archive_generation,
            )
        assert logger.archive_descriptor is newer
        # The retain path refuses it too.
        with pytest.raises(ValueError, match="stale"):
            logger.retain_archive_descriptor(descriptor)
        assert logger.archive_descriptor is newer

    @pytest.mark.asyncio
    async def test_reset_generation_mismatch_refuses_the_publish(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
    ) -> None:
        logger, messages, descriptor = await _fence_sealed_logger(
            session_config, "p4a4-reset-gen", mock_vibe_config, mock_tool_manager
        )
        planned_reset_generation = logger._archive_generation
        # A rebind (same session id: the descriptor survives) starts a new
        # reset generation.
        logger.reset_session(logger.session_id)
        assert logger._archive_generation == planned_reset_generation + 1
        assert logger.archive_descriptor is descriptor
        with pytest.raises(ValueError, match="stale reset generation"):
            logger.fence_and_publish_archive_trim(
                messages,
                descriptor,
                [
                    _trimmed_test_double(messages[1], "ref-u1"),
                    _trimmed_test_double(messages[2], "ref-a1"),
                ],
                save_generation=logger._transcript_cursor_generation,
                reset_generation=planned_reset_generation,
            )
        # Nothing was published: the current authority is untouched.
        assert logger.archive_descriptor is descriptor
        assert messages.archive_descriptor is None

    @pytest.mark.asyncio
    async def test_save_generation_mismatch_refuses_the_publish(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
    ) -> None:
        logger, messages, descriptor = await _fence_sealed_logger(
            session_config, "p4a4-save-gen", mock_vibe_config, mock_tool_manager
        )
        planned_save_generation = logger._transcript_cursor_generation
        planned_reset_generation = logger._archive_generation
        # An invalidation after planning moves the save generation.
        logger.invalidate_transcript_cursor()
        with pytest.raises(ValueError, match="stale save generation"):
            logger.fence_and_publish_archive_trim(
                messages,
                descriptor,
                [
                    _trimmed_test_double(messages[1], "ref-u1"),
                    _trimmed_test_double(messages[2], "ref-a1"),
                ],
                save_generation=planned_save_generation,
                reset_generation=planned_reset_generation,
            )
        assert logger.archive_descriptor is descriptor
        assert messages[1].content == "u1"  # no slot was replaced

    @pytest.mark.asyncio
    async def test_epoch_mismatch_with_the_list_refuses_the_publish(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
    ) -> None:
        logger, messages, descriptor = await _fence_sealed_logger(
            session_config, "p4a4-epoch", mock_vibe_config, mock_tool_manager
        )
        messages.reset([_fence_user("rewound", "u9")])
        with pytest.raises(ValueError, match="does not match the list epoch"):
            logger.fence_and_publish_archive_trim(
                messages,
                descriptor,
                [_trimmed_test_double(messages[0], "ref")],
                save_generation=logger._transcript_cursor_generation,
                reset_generation=logger._archive_generation,
            )
        assert logger.archive_descriptor is descriptor
        assert messages.archive_descriptor is None

    @pytest.mark.asyncio
    async def test_replacement_count_must_match_the_archived_count(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
    ) -> None:
        """No slot insert/delete: one replacement per sealed slot, exactly."""
        logger, messages, descriptor = await _fence_sealed_logger(
            session_config, "p4a4-count", mock_vibe_config, mock_tool_manager
        )
        original_u1 = messages[1]
        with pytest.raises(ValueError, match="exactly one slot"):
            logger.fence_and_publish_archive_trim(
                messages,
                descriptor,
                [_trimmed_test_double(messages[1], "ref-u1")],  # only one of two
                save_generation=logger._transcript_cursor_generation,
                reset_generation=logger._archive_generation,
            )
        assert messages[1] is original_u1
        assert messages.archive_descriptor is None

    @pytest.mark.asyncio
    async def test_publish_rejects_foreign_session_and_disabled_logger(
        self,
        session_config: SessionLoggingConfig,
        disabled_session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
    ) -> None:
        logger, messages, descriptor = await _fence_sealed_logger(
            session_config, "p4a4-foreign", mock_vibe_config, mock_tool_manager
        )
        foreign = ArchivedPrefixV1(
            session_id="some-other-session",
            message_epoch=0,
            archived_count=2,
            prefix_end_byte=10,
            prefix_sha256=descriptor.prefix_sha256,
            boundary_message_id=descriptor.boundary_message_id,
            trim_mode="payloads",
        )
        with pytest.raises(ValueError, match="does not match"):
            logger.fence_and_publish_archive_trim(
                messages,
                foreign,
                [
                    _trimmed_test_double(messages[1], "r1"),
                    _trimmed_test_double(messages[2], "r2"),
                ],
                save_generation=logger._transcript_cursor_generation,
                reset_generation=logger._archive_generation,
            )
        assert logger.archive_descriptor is descriptor

        disabled_logger = SessionLogger(disabled_session_config, "p4a4-disabled")
        with pytest.raises(ValueError, match="disabled"):
            disabled_logger.fence_and_publish_archive_trim(
                messages, descriptor, [], save_generation=0, reset_generation=0
            )


# ---------------------------------------------------------------------------
# T2a: the warm-resume clear-and-reseal proof/transaction — the drain and
# generation fence under the save lock: a paused in-flight save is drained
# before the proof runs, a queued save is fenced THROUGH the synchronous
# rebind in the same lock hold, and disk bytes changed mid-proof fail the
# proof closed with authority retained.
# ---------------------------------------------------------------------------


class TestT2aWarmResumeProofFence:
    @staticmethod
    def _rebound(messages: MessageList) -> list[LLMMessage]:
        """Rebind the list exactly as the warm-resume commit does."""
        loaded = [m for m in messages if m.role is not Role.system]
        messages.reset_preserving_system(list(loaded))
        assert messages.message_epoch == 1
        return loaded

    @pytest.mark.asyncio
    async def test_paused_in_flight_save_is_drained_before_the_proof_runs(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        logger, messages, descriptor = await _fence_sealed_logger(
            session_config, "t2a-inflight", mock_vibe_config, mock_tool_manager
        )
        await _fence_save(logger, messages, mock_vibe_config, mock_tool_manager)
        messages_path = _fence_messages_path(logger)
        raw_before = messages_path.read_bytes()
        prefix_before = raw_before[: descriptor.prefix_end_byte]
        loaded = self._rebound(messages)

        import chartreux.core.session.session_logger as logger_module

        order: list[str] = []
        real_prove = logger_module.prove_rebound_hydration

        def spy_prove(*args: Any, **kwargs: Any) -> Any:
            order.append("proof")
            return real_prove(*args, **kwargs)

        monkeypatch.setattr(logger_module, "prove_rebound_hydration", spy_prove)

        started, release = Event(), Event()
        original = logger._save_interaction_sync

        def gated(*args: Any, **kwargs: Any) -> Any:
            started.set()
            assert release.wait(5)
            result = original(*args, **kwargs)
            order.append("save_done")
            return result

        monkeypatch.setattr(logger, "_save_interaction_sync", gated)

        # A paused in-flight save holds the save lock mid-worker and appends a
        # record the rebound list does not hold.
        extra = loaded + [_fence_user("extra", "extra")]
        save_task = asyncio.create_task(
            _fence_save(logger, extra, mock_vibe_config, mock_tool_manager, steps=2)
        )
        assert await asyncio.to_thread(started.wait, 5)
        txn = asyncio.create_task(
            logger.clear_and_reseal_warm_resume(messages, session_id=logger.session_id)
        )
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        # The transaction waits on the save lock: the proof has NOT run while
        # the in-flight save is paused — no race with the proof.
        assert order == []
        assert not txn.done()
        release.set()
        await save_task
        result = await txn

        # The drain completed first; the proof then saw the save's bytes.
        assert order == ["save_done", "proof"]
        # The save appended a record the rebound list does not hold, so the
        # complete-loaded-records check fails closed.
        assert result.outcome == "refused_proof"
        # Authority retained; the archived prefix bytes are byte-identical and
        # nothing was staged.
        assert logger.archive_descriptor is descriptor
        raw_after = messages_path.read_bytes()
        assert raw_after[: descriptor.prefix_end_byte] == prefix_before
        assert logger.session_dir is not None
        assert not list(logger.session_dir.glob("messages.jsonl.recovered-*"))

    @pytest.mark.asyncio
    async def test_queued_save_is_fenced_through_the_resume_rebind(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        logger, messages, descriptor = await _fence_sealed_logger(
            session_config, "t2a-queued", mock_vibe_config, mock_tool_manager
        )
        await _fence_save(logger, messages, mock_vibe_config, mock_tool_manager)
        messages_path = _fence_messages_path(logger)
        raw_before = messages_path.read_bytes()
        self._rebound(messages)

        import chartreux.core.session.session_logger as logger_module

        started, release = Event(), Event()
        real_prove = logger_module.prove_rebound_hydration

        def gated_prove(*args: Any, **kwargs: Any) -> Any:
            started.set()
            assert release.wait(5)
            return real_prove(*args, **kwargs)

        monkeypatch.setattr(logger_module, "prove_rebound_hydration", gated_prove)

        save_generation = logger._transcript_cursor_generation
        reset_generation = logger._archive_generation
        txn = asyncio.create_task(
            logger.clear_and_reseal_warm_resume(messages, session_id=logger.session_id)
        )
        assert await asyncio.to_thread(started.wait, 5)
        # While the proof runs under the held save lock, queue a stale save:
        # it snapshots the CURRENT generation, then waits on the lock.
        stale = asyncio.create_task(
            _fence_save(
                logger,
                [_fence_user("stale", "stale")],
                mock_vibe_config,
                mock_tool_manager,
                steps=2,
            )
        )
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert not stale.done()
        release.set()
        result = await txn
        await stale

        # The proof held; the authority transition ran, and the generation
        # fence stayed effective THROUGH the synchronous rebind in the same
        # lock hold: the queued save discarded itself under the lock.
        assert result.outcome == "proved_and_transitioned"
        assert logger.archive_descriptor is None  # cleared, never re-pinned
        assert messages.archive_descriptor is None
        assert logger._transcript_cursor_generation == save_generation + 1
        assert logger._archive_generation == reset_generation + 1
        # No bytes written by the fenced save; its snapshot was not published.
        assert messages_path.read_bytes() == raw_before
        snapshot = logger.last_save_archive_snapshot
        assert snapshot is not None
        assert snapshot.descriptor is None
        assert snapshot.message_epoch == 1

    @pytest.mark.asyncio
    async def test_changed_disk_bytes_mid_proof_fail_closed_and_retain_authority(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        logger, messages, descriptor = await _fence_sealed_logger(
            session_config, "t2a-midproof", mock_vibe_config, mock_tool_manager
        )
        await _fence_save(logger, messages, mock_vibe_config, mock_tool_manager)
        messages_path = _fence_messages_path(logger)
        self._rebound(messages)

        import chartreux.core.session.transcript_archive as archive_module

        real_read_chunks = archive_module.read_chunks
        mutated = False

        def mutating_read_chunks(path: Path, *, chunk_size: int = 1024 * 1024) -> Any:
            nonlocal mutated
            if path != messages_path:
                yield from real_read_chunks(path, chunk_size=chunk_size)
                return
            # Unbuffered reads (os.read): a buffered reader would pre-read the
            # whole small file, hiding a mid-proof on-disk change from the
            # in-flight stream. Raw reads see the mutation in later chunks.
            fd = os.open(path, os.O_RDONLY)
            try:
                while True:
                    chunk = os.read(fd, 16)
                    if not chunk:
                        return
                    yield chunk
                    if not mutated:
                        # Change disk bytes MID-PROOF: flip one byte inside
                        # the archived prefix region after the first chunk
                        # was read; the remaining chunks stream changed bytes.
                        mutated = True
                        with open(path, "r+b") as stream:
                            stream.seek(20)
                            current = stream.read(1)
                            stream.seek(20)
                            stream.write(bytes([current[0] ^ 0x01]))
            finally:
                os.close(fd)

        monkeypatch.setattr(archive_module, "read_chunks", mutating_read_chunks)

        save_generation = logger._transcript_cursor_generation
        reset_generation = logger._archive_generation
        result = await logger.clear_and_reseal_warm_resume(
            messages, session_id=logger.session_id
        )

        assert mutated
        # The mid-proof byte change fails the proof closed (prefix hash).
        assert result.outcome == "refused_proof"
        # Authority and originals stay; the fence never moved; nothing staged.
        assert logger.archive_descriptor is descriptor
        assert logger._transcript_cursor_generation == save_generation
        assert logger._archive_generation == reset_generation
        assert logger.session_dir is not None
        assert not list(logger.session_dir.glob("messages.jsonl.recovered-*"))


# ---------------------------------------------------------------------------
# P5 (S1-C, PROTOCOL v2.3): the per-save persistence scan runs in a threaded
# worker while the save lock is held. Parked-worker cases: the scan is off
# the event loop (loop-occupancy proof), a generation bump mid-prep still
# drops at the fence, cancellation during prep frees the lock with disk
# untouched, an append during prep persists one consistent state, an injected
# defect raises the typed TranscriptIntegrityError from the in-lock worker
# BEFORE the fence (corrupt-but-fresh AND corrupt-but-stale), and overlapping
# saves are serialized by construction (no overtaking).
# ---------------------------------------------------------------------------


def _worker_tool_result(call_id: str) -> LLMMessage:
    return LLMMessage(
        role=Role.tool,
        tool_call_id=call_id,
        name="read_file",
        content="file contents",
        tool_result=PersistedToolResult(output={"text": "file contents"}),
    )


def _worker_corrupt_transcript() -> list[LLMMessage]:
    """A duplicate tool result: the save boundary's typed failure shape."""
    return [
        LLMMessage(role=Role.user, content="hi"),
        LLMMessage(
            role=Role.assistant,
            content="Working",
            tool_calls=[
                ToolCall(
                    id="c1",
                    index=0,
                    function=FunctionCall(name="read_file", arguments="{}"),
                )
            ],
        ),
        _worker_tool_result("c1"),
        _worker_tool_result("c1"),
    ]


class _SavePrepScanGate:
    """Test double for the lock-scoped validation scan.

    Records the thread and the save-lock state of every scan. Optionally
    blocks the first scan until released and runs ``mid_scan`` inside the
    blocked scan (mid-prep), then always delegates to the real wrapper.
    """

    def __init__(
        self,
        logger: SessionLogger,
        monkeypatch: pytest.MonkeyPatch,
        *,
        block_first: bool = True,
        mid_scan: Callable[[], None] | None = None,
    ) -> None:
        self.started, self.release, self.finished = Event(), Event(), Event()
        self.scans = 0
        self.scan_threads: list[int] = []
        self.lock_held_at_scan: list[bool] = []
        original = logger._check_transcript_persistence

        def gated(
            messages: Sequence[LLMMessage],
            *,
            archived_prefix: ArchivedPrefixAuthority | None = None,
        ) -> None:
            self.scans += 1
            self.scan_threads.append(get_ident())
            self.lock_held_at_scan.append(logger._save_lock.locked())
            if self.scans == 1:
                self.started.set()
                if mid_scan is not None:
                    mid_scan()
                if block_first:
                    assert self.release.wait(5)
            try:
                original(messages, archived_prefix=archived_prefix)
            finally:
                self.finished.set()

        monkeypatch.setattr(logger, "_check_transcript_persistence", gated)


def _worker_transcript_contents(logger: SessionLogger) -> list[Any]:
    raw = _fence_messages_path(logger).read_bytes()
    return [json.loads(line).get("content") for line in raw.splitlines()]


class TestSavePrepWorker:
    @pytest.mark.asyncio
    async def test_save_prep_worker_scan_runs_off_loop_with_the_save_lock_held(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The validation scan occupies a worker thread, not the event loop:
        while the in-lock scan is gated, other tasks keep running on the loop
        and the save lock stays held across the whole save.
        """
        logger = SessionLogger(session_config, "save-prep-worker-offloop")
        gate = _SavePrepScanGate(logger, monkeypatch)
        loop_thread = get_ident()

        ticks = 0

        async def ticker() -> None:
            nonlocal ticks
            while not gate.release.is_set():
                ticks += 1
                await asyncio.sleep(0)

        task = asyncio.create_task(
            _fence_save(
                logger, [_fence_user("u1", "u1")], mock_vibe_config, mock_tool_manager
            )
        )
        tick_task = asyncio.create_task(ticker())
        assert await asyncio.to_thread(gate.started.wait, 5)
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert ticks > 0  # the event loop stayed free during the in-lock scan
        assert logger._save_lock.locked()
        gate.release.set()
        await task
        await tick_task
        assert gate.scans == 1
        assert gate.scan_threads[0] != loop_thread  # the scan ran off-loop
        assert gate.lock_held_at_scan == [True]
        assert _worker_transcript_contents(logger) == ["u1"]

    @pytest.mark.asyncio
    async def test_save_prep_worker_generation_bump_mid_prep_is_dropped_at_the_fence(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        logger = SessionLogger(session_config, "save-prep-worker-stale")
        await _fence_save(
            logger, [_fence_user("u1", "u1")], mock_vibe_config, mock_tool_manager
        )
        before = _fence_messages_path(logger).read_bytes()
        generation = logger._transcript_cursor_generation

        def bump_mid_prep() -> None:
            # The lock-free generation bump, issued from the scan worker
            # thread mid-prep.
            logger.invalidate_transcript_cursor()

        gate = _SavePrepScanGate(logger, monkeypatch, mid_scan=bump_mid_prep)
        task = asyncio.create_task(
            _fence_save(
                logger,
                [_fence_user("u2", "u2")],
                mock_vibe_config,
                mock_tool_manager,
                steps=2,
            )
        )
        assert await asyncio.to_thread(gate.started.wait, 5)
        assert logger._transcript_cursor_generation == generation + 1
        gate.release.set()
        await task  # no raise: the fence drops the stale save
        assert _fence_messages_path(logger).read_bytes() == before
        assert gate.scans == 1

    @pytest.mark.asyncio
    async def test_save_prep_worker_cancel_during_prep_frees_the_lock_disk_untouched(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        logger = SessionLogger(session_config, "save-prep-worker-cancel")
        await _fence_save(
            logger, [_fence_user("u1", "u1")], mock_vibe_config, mock_tool_manager
        )
        before = _fence_messages_path(logger).read_bytes()
        gate = _SavePrepScanGate(logger, monkeypatch)
        task = asyncio.create_task(
            _fence_save(
                logger,
                [_fence_user("u1", "u1"), _fence_user("u2", "u2")],
                mock_vibe_config,
                mock_tool_manager,
                steps=2,
            )
        )
        assert await asyncio.to_thread(gate.started.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not logger._save_lock.locked()  # the lock freed on cancel
        gate.release.set()
        assert await asyncio.to_thread(gate.finished.wait, 5)  # pure worker drained
        assert _fence_messages_path(logger).read_bytes() == before  # disk untouched
        # A later save still acquires the lock and persists.
        await _fence_save(
            logger,
            [_fence_user("u1", "u1"), _fence_user("u2", "u2")],
            mock_vibe_config,
            mock_tool_manager,
            steps=3,
        )
        assert _worker_transcript_contents(logger) == ["u1", "u2"]

    @pytest.mark.asyncio
    async def test_save_prep_worker_append_during_prep_persists_one_consistent_state(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        logger = SessionLogger(session_config, "save-prep-worker-append")
        messages = MessageList([_fence_user("u1", "u1")])
        gate = _SavePrepScanGate(logger, monkeypatch)
        task = asyncio.create_task(
            _fence_save(logger, messages, mock_vibe_config, mock_tool_manager)
        )
        assert await asyncio.to_thread(gate.started.wait, 5)
        # A concurrent append lands while the first save's scan is gated: the
        # pre-lock snapshot already holds only u1.
        messages.append(_fence_user("u2", "u2"))
        gate.release.set()
        await task
        assert _worker_transcript_contents(logger) == ["u1"]
        # The next save of the live list persists the appended message too:
        # one consistent state per save, never a torn transcript.
        await _fence_save(
            logger, messages, mock_vibe_config, mock_tool_manager, steps=2
        )
        assert _worker_transcript_contents(logger) == ["u1", "u2"]

    @pytest.mark.asyncio
    async def test_save_prep_worker_corrupt_fresh_save_raises_typed_error_from_the_worker(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Corrupt-but-fresh: the typed TranscriptIntegrityError raises from
        the in-lock worker BEFORE the fence — never silently persisted.
        """
        logger = SessionLogger(session_config, "save-prep-worker-corrupt-fresh")
        await _fence_save(
            logger, [_fence_user("u1", "u1")], mock_vibe_config, mock_tool_manager
        )
        before = _fence_messages_path(logger).read_bytes()
        gate = _SavePrepScanGate(logger, monkeypatch, block_first=False)
        loop_thread = get_ident()
        corrupt = _worker_corrupt_transcript()
        with pytest.raises(TranscriptIntegrityError) as exc_info:
            await _fence_save(logger, corrupt, mock_vibe_config, mock_tool_manager)
        assert gate.scans == 1
        assert gate.scan_threads[0] != loop_thread  # raised from the worker
        assert gate.lock_held_at_scan == [True]  # inside the lock, pre-fence
        assert not logger._save_lock.locked()
        assert _fence_messages_path(logger).read_bytes() == before  # not persisted
        # Same exception type and message content as today's on-loop wrapper.
        with pytest.raises(TranscriptIntegrityError) as direct:
            SessionLogger._check_transcript_persistence(corrupt)
        assert str(exc_info.value) == str(direct.value)
        assert "session save boundary" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_save_prep_worker_corrupt_stale_save_raises_never_drops_at_the_fence(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Corrupt-but-stale: the generation bumps mid-prep, yet the save must
        still RAISE from the validation worker — never silently drop at the
        fence and never silently persist.
        """
        logger = SessionLogger(session_config, "save-prep-worker-corrupt-stale")
        await _fence_save(
            logger, [_fence_user("u1", "u1")], mock_vibe_config, mock_tool_manager
        )
        before = _fence_messages_path(logger).read_bytes()
        generation = logger._transcript_cursor_generation

        def bump_mid_prep() -> None:
            logger.invalidate_transcript_cursor()

        gate = _SavePrepScanGate(
            logger, monkeypatch, block_first=False, mid_scan=bump_mid_prep
        )
        corrupt = _worker_corrupt_transcript()
        with pytest.raises(TranscriptIntegrityError) as exc_info:
            await _fence_save(logger, corrupt, mock_vibe_config, mock_tool_manager)
        # The save was already stale when the worker raised: validation
        # precedes the fence, so the typed error wins over the drop.
        assert logger._transcript_cursor_generation == generation + 1
        assert gate.scans == 1
        assert gate.lock_held_at_scan == [True]
        assert _fence_messages_path(logger).read_bytes() == before
        with pytest.raises(TranscriptIntegrityError) as direct:
            SessionLogger._check_transcript_persistence(corrupt)
        assert str(exc_info.value) == str(direct.value)

    @pytest.mark.asyncio
    async def test_save_prep_worker_overlapping_saves_are_serialized_no_overtake(
        self,
        session_config: SessionLoggingConfig,
        mock_vibe_config: ChartreuxConfigSchema,
        mock_tool_manager: ToolManager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Overlapping saves are serialized by the save lock held across prep
        + write: the second save cannot even enter prep — let alone complete
        — before the first.
        """
        logger = SessionLogger(session_config, "save-prep-worker-overlap")
        gate = _SavePrepScanGate(logger, monkeypatch)
        first = asyncio.create_task(
            _fence_save(
                logger, [_fence_user("u1", "u1")], mock_vibe_config, mock_tool_manager
            )
        )
        assert await asyncio.to_thread(gate.started.wait, 5)
        second = asyncio.create_task(
            _fence_save(
                logger,
                [_fence_user("u1", "u1"), _fence_user("u2", "u2")],
                mock_vibe_config,
                mock_tool_manager,
                steps=2,
            )
        )
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert gate.scans == 1  # the second save is queued behind the lock
        assert not second.done()
        gate.release.set()
        await first
        assert not second.done()  # no overtaking: second completes after first
        await second
        assert gate.scans == 2
        assert gate.lock_held_at_scan == [True, True]
        assert _worker_transcript_contents(logger) == ["u1", "u2"]
