from __future__ import annotations

from collections.abc import AsyncGenerator
from datetime import UTC, datetime
import json
from pathlib import Path
import time

import pytest

from chartreux.app_server._projection import project_message_history
from chartreux.core.config import SessionLoggingConfig
from chartreux.core.llm_models import (
    FunctionCall,
    LLMChunk,
    LLMMessage,
    Role,
    ToolCall,
    use_event_clock,
    use_posting_clock,
)
from chartreux.core.session.session_loader import SessionLoader
from chartreux.core.tools.base import ToolPermission
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend

REASONING_AT = datetime(2024, 5, 6, 7, 8, 9, tzinfo=UTC)
DISPATCH_AT = datetime(2024, 5, 6, 7, 9, 30, tzinfo=UTC)


def epoch_ms(moment: datetime) -> int:
    return int(moment.timestamp() * 1000)


def todo_call(call_id: str = "call_1") -> ToolCall:
    return ToolCall(
        id=call_id,
        index=0,
        function=FunctionCall(name="todo", arguments='{"action": "read"}'),
    )


def sessions_config(tmp_path: Path) -> object:
    return build_test_vibe_config(
        enabled_tools=["todo"],
        tools={"todo": {"permission": ToolPermission.ALWAYS}},
        session_logging=SessionLoggingConfig(
            enabled=True, save_dir=str(tmp_path / "sessions")
        ),
    )


def persisted_records(session_dir: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in (session_dir / "messages.jsonl").read_text().splitlines()
    ]


class AdvancingBackend(FakeBackend):
    """A streaming backend that advances the event clock as its stream ends."""

    def __init__(
        self,
        streams: list[list[LLMChunk]],
        *,
        current: list[datetime],
        after_stream: datetime,
    ) -> None:
        super().__init__(streams)
        self._current = current
        self._after_stream = after_stream

    async def complete_streaming(self, **kwargs: object) -> AsyncGenerator[LLMChunk]:
        async for chunk in super().complete_streaming(**kwargs):
            yield chunk
        self._current[0] = self._after_stream


@pytest.mark.asyncio
async def test_reasoning_and_effect_clocks_are_stamped_at_emission(
    tmp_path: Path,
) -> None:
    current = [REASONING_AT]
    backend = AdvancingBackend(
        [
            [
                mock_llm_chunk(content="", reasoning_content="thinking"),
                mock_llm_chunk(content="Let me look.", tool_calls=[todo_call()]),
            ],
            [mock_llm_chunk(content="Done.")],
        ],
        current=current,
        after_stream=DISPATCH_AT,
    )
    agent = build_test_agent_loop(
        config=sessions_config(tmp_path),  # type: ignore[arg-type]
        backend=backend,
        enable_streaming=True,
        cwd=tmp_path,
    )
    wall_before = int(time.time() * 1000)
    with use_event_clock(lambda: current[0]):
        _ = [event async for event in agent.act("question")]
    await agent.aclose()

    session_dir = agent.session_logger.session_dir
    assert session_dir is not None
    records = persisted_records(session_dir)
    reasoning_record = next(
        record for record in records if record.get("reasoning_content") == "thinking"
    )
    # The reasoning block's own clock: the value read at its emission.
    assert reasoning_record["reasoning_at"] == epoch_ms(REASONING_AT)
    # The effect's clock is captured per event too (the dispatch reading wins
    # over the mid-stream call-event reading), never one shared save stamp.
    tool_calls = reasoning_record["tool_calls"]
    assert isinstance(tool_calls, list)
    assert tool_calls[0]["started_at"] == epoch_ms(DISPATCH_AT)
    # Emission clocks are real clocks captured during the turn, not the
    # wall-clock instant the transcript was saved.
    assert reasoning_record["reasoning_at"] != epoch_ms(DISPATCH_AT)
    assert wall_before > epoch_ms(DISPATCH_AT)

    loaded, _ = SessionLoader.load_session(session_dir)
    loaded_reasoning = next(m for m in loaded if m.reasoning_content)
    assert loaded_reasoning.reasoning_at == epoch_ms(REASONING_AT)
    assert loaded_reasoning.tool_calls is not None
    assert loaded_reasoning.tool_calls[0].started_at == epoch_ms(DISPATCH_AT)

    # The restored history surfaces those same clocks, so a reloaded transcript
    # renders the emission times rather than the save instant.
    projected = project_message_history(agent.session_id, loaded, None)
    projected_reasoning = next(
        entry for entry in projected if entry.type == "reasoning"
    )
    projected_effect = next(entry for entry in projected if entry.type == "effect")
    assert projected_reasoning.event_at == epoch_ms(REASONING_AT)
    assert projected_effect.event_at == epoch_ms(DISPATCH_AT)


@pytest.mark.asyncio
async def test_non_streaming_reasoning_carries_the_gateway_emission_stamp(
    tmp_path: Path,
) -> None:
    backend = FakeBackend([
        [
            mock_llm_chunk(content="", reasoning_content="thinking"),
            mock_llm_chunk(content="answer"),
        ],
        [mock_llm_chunk(content="Done.")],
    ])
    agent = build_test_agent_loop(
        config=sessions_config(tmp_path),  # type: ignore[arg-type]
        backend=backend,
        cwd=tmp_path,
    )
    with use_posting_clock(lambda: REASONING_AT):
        _ = [event async for event in agent.act("question")]
    await agent.aclose()

    session_dir = agent.session_logger.session_dir
    assert session_dir is not None
    records = persisted_records(session_dir)
    reasoning_record = next(
        record for record in records if record.get("reasoning_content") == "thinking"
    )
    assert reasoning_record["reasoning_at"] == epoch_ms(REASONING_AT)


@pytest.mark.asyncio
async def test_legacy_records_restore_without_event_clocks(tmp_path: Path) -> None:
    tmp_path.joinpath("meta.json").write_text(json.dumps({"total_messages": 2}))
    tmp_path.joinpath("messages.jsonl").write_text(
        '{"role":"user","content":"old","message_id":"u"}\n'
        '{"role":"assistant","content":"answer","reasoning_content":"thought",'
        '"message_id":"a","tool_calls":[{"id":"c","index":0,'
        '"function":{"name":"todo","arguments":"{}"}}]}\n'
    )
    loaded, _ = SessionLoader.load_session(tmp_path)
    assistant = next(m for m in loaded if m.role is Role.assistant)
    # Never backfilled: a legacy record restores exactly as it was written.
    assert assistant.reasoning_at is None
    assert assistant.tool_calls is not None
    assert assistant.tool_calls[0].started_at is None


def test_compounded_reasoning_clock_keeps_the_last_emission() -> None:
    first = LLMMessage(
        role=Role.assistant, reasoning_content="think", reasoning_at=1_000
    )
    second = LLMMessage(
        role=Role.assistant, reasoning_content="ing", reasoning_at=2_000
    )
    # Compounding appends the new emission on the right, and that later clock
    # is the one the compounded block keeps.
    assert (first + second).reasoning_at == 2_000
    assert (second + first).reasoning_at == 1_000


def test_event_clock_fields_are_additive_and_optional() -> None:
    message = LLMMessage(role=Role.assistant, content="answer")
    assert message.reasoning_at is None
    assert "reasoning_at" not in message.model_dump(exclude_none=True, mode="json")

    legacy = {"role": "assistant", "content": "answer", "message_id": "a"}
    assert LLMMessage.model_validate(legacy).reasoning_at is None
    # Old readers are not broken either way: unknown keys stay tolerated.
    tolerated = LLMMessage.model_validate({**legacy, "unrecognized_key": 1})
    assert tolerated.reasoning_at is None

    tool_call = ToolCall.model_validate({"id": "c", "function": {"name": "todo"}})
    assert tool_call.started_at is None
    assert "started_at" not in tool_call.model_dump(exclude_none=True, mode="json")
