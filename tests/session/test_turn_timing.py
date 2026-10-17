from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from chartreux.app_server._agent_transcript import _project_entries
from chartreux.app_server._projection import _persisted_effect_state
from chartreux.app_server._projector import EventProjector
from chartreux.app_server._tool_projection import project_effect_state
from chartreux.app_server._turns import TurnController
from chartreux.app_server.events import ClientProjection
from chartreux.app_server.models import (
    CancelledEffectState,
    FailedEffectState,
    IdleSessionStatus,
    JsonPatchOperation,
    PublicEffectEntry,
    PublicMessageEntry,
    PublicSession,
    PublicSessionState,
    PublicTurn,
    PublicTurnStatus,
)
from chartreux.app_server.protocol import HistoryEntryUpdatedParams, TurnStartParams
from chartreux.core.agent_loop.llm_gateway import TranscriptAppend
from chartreux.core.config import ProviderConfig, SessionLoggingConfig
from chartreux.core.events import (
    AssistantEvent,
    BaseEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from chartreux.core.llm.backend.generic import OpenAIAdapter
from chartreux.core.llm_models import LLMMessage, PersistedToolResult, Role
from chartreux.core.session.session_loader import SessionLoader
from chartreux.core.timing import CompletedTurnTiming, use_elapsed_clock
from chartreux.core.tools.builtins.read_file import ReadFile, ReadFileArgs
from chartreux.core.tools.ui import ToolUIDataAdapter
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [True, False])
async def test_real_turn_total_survives_reload(tmp_path: Path, streaming: bool) -> None:
    config = build_test_vibe_config(
        enabled_tools=[],
        session_logging=SessionLoggingConfig(enabled=True, save_dir=str(tmp_path)),
    )
    agent = build_test_agent_loop(
        config=config,
        backend=FakeBackend(
            [mock_llm_chunk(content="answer")], retries_before_response=2
        ),
        enable_streaming=streaming,
    )
    clock = iter([10.0, 15.0])
    with use_elapsed_clock(lambda: next(clock)):
        _ = [event async for event in agent.act("question")]
    prose = agent.messages[-1]
    assert prose.turn_duration == 5.0
    assert prose.message_id is not None
    assert agent.completed_turn_timing == CompletedTurnTiming(prose.message_id, 5.0)
    session_dir = agent.session_logger.session_dir
    assert session_dir is not None
    loaded, _ = SessionLoader.load_session(session_dir)
    assert loaded[-1].turn_duration == 5.0
    await agent.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["normal", "error", "cancel", "close"])
@pytest.mark.parametrize("retain_last", [True, False])
async def test_last_canonical_prose_owns_total_and_invalidates_logger_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ending: str, retain_last: bool
) -> None:
    config = build_test_vibe_config(
        session_logging=SessionLoggingConfig(enabled=True, save_dir=str(tmp_path))
    )
    agent = build_test_agent_loop(config=config)
    first = LLMMessage(role=Role.assistant, content="first")
    last = LLMMessage(role=Role.assistant, content="partial last")
    foreign = LLMMessage(role=Role.assistant, content="compaction summary")
    now = [1.0]

    async def conversation(*_args: Any, **_kwargs: Any) -> AsyncGenerator[BaseEvent]:
        agent._append_transcript(TranscriptAppend(first, "complete"))
        agent.messages.append(
            LLMMessage(role=Role.user, content="steer", injected=True)
        )
        agent._append_transcript(TranscriptAppend(last, "interrupted"))
        # Copies retained across context changes must resolve by ID, not identity.
        if not retain_last:
            agent.messages.reset([first.model_copy(), foreign])
        else:
            agent.messages.append(foreign)
        agent.messages.append(LLMMessage(role=Role.tool, content="terminal boundary"))
        await agent._save_messages()
        try:
            yield AssistantEvent(content="partial", message_id=last.message_id)
            if ending == "error":
                raise RuntimeError("interrupted")
            if ending == "cancel":
                raise asyncio.CancelledError
        finally:
            # Iterator cleanup represents joined work, and belongs in the total.
            now[0] = 9.0

    monkeypatch.setattr(agent, "_conversation_loop", conversation)
    with use_elapsed_clock(lambda: now[0]):
        events = agent.act("question")
        if ending == "close":
            await anext(events)
            await events.aclose()
        elif ending in {"error", "cancel"}:
            with pytest.raises(
                RuntimeError if ending == "error" else asyncio.CancelledError
            ):
                _ = [event async for event in events]
        else:
            _ = [event async for event in events]
    owner = last if retain_last else first
    timed = [message for message in agent.messages if message.turn_duration is not None]
    assert [(message.message_id, message.turn_duration) for message in timed] == [
        (owner.message_id, 8.0)
    ]
    assert foreign.turn_duration is None
    session_dir = agent.session_logger.session_dir
    assert session_dir is not None
    loaded, _ = SessionLoader.load_session(session_dir)
    assert [
        (m.message_id, m.turn_duration) for m in loaded if m.turn_duration is not None
    ] == [(owner.message_id, 8.0)]
    await agent.aclose()


@pytest.mark.asyncio
async def test_no_prose_does_not_invent_or_retime_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = build_test_agent_loop()
    old = LLMMessage(role=Role.assistant, content="old")
    agent.messages.append(old)

    async def conversation(*_args: Any, **_kwargs: Any) -> AsyncGenerator[BaseEvent]:
        agent._append_transcript(
            TranscriptAppend(
                LLMMessage(role=Role.assistant, reasoning_content="thinking"),
                "complete",
            )
        )
        if False:
            yield AssistantEvent(content="")

    monkeypatch.setattr(agent, "_conversation_loop", conversation)
    with use_elapsed_clock(lambda: 1.0):
        _ = [event async for event in agent.act("question")]
    assert agent.completed_turn_timing is None
    assert all(m.turn_duration is None for m in agent.messages)
    await agent.aclose()


def _projector() -> EventProjector:
    projector = EventProjector("session", "turn")
    projector.project(AssistantEvent(content="answer", message_id="prose"))
    call = ToolCallEvent(
        tool_name="read_file",
        tool_class=ReadFile,
        args=ReadFileArgs(file_path="a"),
        tool_call_id="call",
    )
    projector.project(
        call.model_copy(
            update={
                "presentation": ToolUIDataAdapter(ReadFile).get_call_presentation(call)
            }
        )
    )
    return projector


def _consumer(projector: EventProjector) -> ClientProjection:
    return ClientProjection(
        PublicSessionState(
            event_id=0,
            session=PublicSession(
                id="session", status=IdleSessionStatus(), created_at=0, updated_at=0
            ),
            history=list(projector.history),
        )
    )


def test_completed_timing_patch_at_both_boundaries_and_frozen_content() -> None:
    projector = _projector()
    consumer = _consumer(projector)
    updates = projector.finalize(timing=CompletedTurnTiming("prose", 0.0))
    for update in updates:
        assert isinstance(update.params, HistoryEntryUpdatedParams)
        consumer._update_entry(update.params)
    assert isinstance(consumer.history[0], PublicMessageEntry)
    assert consumer.history[0].turn_duration_ms == 0.0
    for path, value in [("/content", []), ("/id", "changed"), ("/updatedAt", 42)]:
        operations = [JsonPatchOperation(op="replace", path=path, value=value)]
        with pytest.raises(ValueError, match="frozen"):
            projector._patch("prose", operations)
        with pytest.raises(ValueError, match="frozen"):
            consumer._update_entry(
                HistoryEntryUpdatedParams(
                    event_id=1,
                    session_id="session",
                    entry_id="prose",
                    patch=operations,
                    emitted_at=1,
                )
            )


@pytest.mark.parametrize("duration", [None, 0.0, 1.25])
@pytest.mark.parametrize("tag", ["tool_error", "user_cancellation"])
def test_terminal_timing_agrees_live_resume_child(
    duration: float | None, tag: str
) -> None:
    projector = _projector()
    message = LLMMessage(
        role=Role.tool,
        tool_call_id="call",
        name="read_file",
        content=f"<{tag}>interrupted</{tag}>",
        tool_result=PersistedToolResult(
            output={}, duration=duration, cancelled=tag == "user_cancellation"
        ),
    )
    expected = duration * 1000 if duration is not None else None
    live = project_effect_state(
        ToolResultEvent(
            tool_name="read_file",
            tool_class=ReadFile,
            tool_call_id="call",
            error="interrupted",
            cancelled=tag == "user_cancellation",
            duration=duration,
        )
    )
    effect = projector.history[-1]
    assert isinstance(effect, PublicEffectEntry)
    resumed = _persisted_effect_state(effect, message)
    child = _project_entries([message.model_dump(mode="json")])[0]
    assert isinstance(live, FailedEffectState | CancelledEffectState)
    assert isinstance(resumed, FailedEffectState | CancelledEffectState)
    assert isinstance(child.state, FailedEffectState | CancelledEffectState)
    assert (
        live.duration_ms == resumed.duration_ms == child.state.duration_ms == expected
    )


@pytest.mark.parametrize("duration", [0.0, 2.0])
def test_undelivered_cancelled_tool_result_reconciles_and_replays(
    duration: float,
) -> None:
    projector = _projector()
    consumer = _consumer(projector)
    persisted = LLMMessage(
        role=Role.tool,
        tool_call_id="call",
        content="<user_cancellation>interrupted</user_cancellation>",
        tool_result=PersistedToolResult(output={}, duration=duration, cancelled=True),
    )
    updates = projector.finalize(cancelled=True, messages=[persisted])
    for update in updates:
        assert isinstance(update.params, HistoryEntryUpdatedParams)
        consumer._update_entry(update.params)
    effect = consumer.history[-1]
    assert isinstance(effect, PublicEffectEntry)
    assert isinstance(effect.state, CancelledEffectState)
    assert effect.state.duration_ms == duration * 1000
    resumed = _persisted_effect_state(effect, persisted)
    assert isinstance(resumed, CancelledEffectState)
    assert resumed.duration_ms == effect.state.duration_ms
    replay = projector.finalize(
        cancelled=True, messages=[persisted], replay_timing=True
    )
    assert len(replay) == 3
    for update in replay:
        assert isinstance(update.params, HistoryEntryUpdatedParams)
        consumer._update_entry(update.params)


def test_turn_timing_changes_child_revision_digest_and_wire_field() -> None:
    message = LLMMessage(role=Role.assistant, content="answer")
    before = _project_entries([message.model_dump(mode="json")])[0]
    message.turn_duration = 0.0
    after = _project_entries([message.model_dump(mode="json")])[0]
    assert before.digest != after.digest
    assert after.turn_duration_ms == 0.0
    assert after.model_dump(by_alias=True)["turnDurationMs"] == 0.0


def test_provider_payload_excludes_whole_turn_duration() -> None:
    message = LLMMessage(role=Role.assistant, content="answer", turn_duration=10.0)
    provider = ProviderConfig(
        name="test", api_base="https://example.test", api_key_env_var="TEST_KEY"
    )
    wire = OpenAIAdapter()._convert_messages([message], provider)
    assert wire == [{"role": "assistant", "content": "answer"}]


@pytest.mark.asyncio
async def test_recovery_reannounces_timing_and_replaces_copied_stale_entries() -> None:
    projector = _projector()
    agent = build_test_agent_loop()
    agent.completed_turn_timing = CompletedTurnTiming("prose", 4.0)
    stale = list(projector.history)
    consumer = _consumer(projector)
    agent.messages.append(
        LLMMessage(
            role=Role.tool,
            tool_call_id="call",
            tool_result=PersistedToolResult(output={}, duration=2.0, cancelled=True),
        )
    )
    # Ordinary finalization mutated its own entries, then failed during delivery.
    projector.finalize(timing=agent.completed_turn_timing, messages=agent.messages)
    emitted: list[Any] = []

    async def emit(update: Any) -> None:
        emitted.append(update)
        consumer._update_entry(update.params)

    async def noop(*_args: Any) -> None:
        pass

    controller: Any = SimpleNamespace(
        _agent_loop=agent,
        _projector=projector,
        _history=stale,
        _completed_turns=[],
        _cancel_callbacks=lambda _reason: None,
        _emit_projected=emit,
        _publish_turn_completed=noop,
        _emit_status=noop,
    )
    turn = PublicTurn(
        id="turn",
        session_id=agent.session_id,
        status=PublicTurnStatus.IN_PROGRESS,
        started_at=0,
    )
    await TurnController._terminalize_failed_turn(
        controller, turn, PublicTurnStatus.FAILED, RuntimeError("delivery failed")
    )
    assert len(controller._history) == len(stale)
    assert controller._history[0].turn_duration_ms == 4000
    assert controller._history[0] is not stale[0]
    assert len(emitted) == 4
    assert isinstance(consumer.history[0], PublicMessageEntry)
    assert consumer.history[0].turn_duration_ms == 4000
    effect = consumer.history[-1]
    assert isinstance(effect, PublicEffectEntry)
    assert isinstance(effect.state, FailedEffectState)
    assert effect.state.duration_ms == 2000
    await agent.aclose()


def test_tool_timing_alone_changes_child_revision_digest() -> None:
    message = LLMMessage(
        role=Role.tool,
        tool_call_id="call",
        name="read_file",
        content="<tool_error>failed</tool_error>",
        tool_result=PersistedToolResult(output={}),
    )
    before = _project_entries([message.model_dump(mode="json")])[0]
    assert message.tool_result is not None
    message.tool_result.duration = 0.0
    after = _project_entries([message.model_dump(mode="json")])[0]
    assert before.digest != after.digest


@pytest.mark.asyncio
async def test_cancelled_turn_survives_timing_save_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = build_test_agent_loop()
    prose = LLMMessage(role=Role.assistant, content="partial")

    async def conversation(*_args: Any, **_kwargs: Any) -> AsyncGenerator[BaseEvent]:
        agent._append_transcript(TranscriptAppend(prose, "interrupted"))
        yield AssistantEvent(content="partial", message_id=prose.message_id)
        raise asyncio.CancelledError

    async def broken_save() -> None:
        raise OSError("timing save failed")

    monkeypatch.setattr(agent, "_conversation_loop", conversation)
    monkeypatch.setattr(agent, "_save_messages", broken_save)

    async def noop(*_args: Any) -> None:
        pass

    async def announce(_turn: PublicTurn) -> int:
        return agent.stats.context_tokens

    controller: Any = SimpleNamespace(
        _agent_loop=agent,
        _announce_turn_started=announce,
        _inject_queued_contexts=noop,
        _subagent_runner=None,
        _tool_io=None,
        _emit_retrying=noop,
        _clear_retrying=noop,
        _event_sink=None,
        _project_events=noop,
        _emit_stats=noop,
    )
    turn = PublicTurn(
        id="turn",
        session_id=agent.session_id,
        status=PublicTurnStatus.IN_PROGRESS,
        started_at=0,
    )
    status, error, _ = await TurnController._run_turn_phase(
        controller,
        turn,
        "question",
        params=TurnStartParams(session_id=agent.session_id, message=[]),
        images=[],
        input_text=None,
        resources=[],
        queued_contexts=(),
    )
    assert status is PublicTurnStatus.INTERRUPTED
    assert error is None
    assert agent.completed_turn_timing is not None
    assert prose.turn_duration is not None
    await agent.aclose()
