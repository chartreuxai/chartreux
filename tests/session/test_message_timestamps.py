from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import aclosing
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
from typing import Any

import pytest

from chartreux.core.agent_loop.llm_gateway import (
    CallResources,
    CompletionInputs,
    LLMGateway,
    TranscriptAppend,
)
from chartreux.core.config import ProviderConfig, SessionLoggingConfig
from chartreux.core.events import AssistantEvent, ReasoningEvent, UserMessageEvent
from chartreux.core.llm.backend.anthropic import AnthropicAdapter
from chartreux.core.llm.backend.generic import OpenAIAdapter
from chartreux.core.llm.backend.mistral import MistralMapper
from chartreux.core.llm.backend.openai_responses import OpenAIResponsesAdapter
from chartreux.core.llm_models import (
    FunctionCall,
    LLMChunk,
    LLMMessage,
    Role,
    ToolCall,
    posting_time,
    use_posting_clock,
)
from chartreux.core.session.session_loader import SessionLoader
from chartreux.core.session_types import AgentStats
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend

POSTED = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_posting_sites_round_trip_and_resume(
    tmp_path: Path, streaming: bool
) -> None:
    times = iter([POSTED, POSTED + timedelta(seconds=10)])
    config = build_test_vibe_config(
        enabled_tools=[],
        session_logging=SessionLoggingConfig(
            enabled=True, save_dir=str(tmp_path / "sessions")
        ),
    )
    with use_posting_clock(lambda: next(times)):
        agent = build_test_agent_loop(
            config=config,
            backend=FakeBackend(
                [
                    mock_llm_chunk(content="", reasoning_content="thinking"),
                    mock_llm_chunk(content="answer"),
                    mock_llm_chunk(content="!"),
                ],
                retries_before_response=2,
            ),
            enable_streaming=streaming,
            cwd=tmp_path,
        )
        assert all(message.posted_at is None for message in agent.messages)
        events = [event async for event in agent.act("question")]
    user, assistant = [m for m in agent.messages if m.role is not Role.system]
    assert user.posted_at == POSTED
    assert assistant.posted_at == POSTED + timedelta(seconds=10)
    assert assistant.content == "answer!"
    for event in events:
        if isinstance(event, UserMessageEvent):
            assert event.posted_at == user.posted_at
        if isinstance(event, (AssistantEvent, ReasoningEvent)):
            assert event.posted_at == assistant.posted_at
    session_dir = agent.session_logger.session_dir
    assert session_dir is not None
    await agent.aclose()
    loaded, _ = SessionLoader.load_session(session_dir)
    assert [(m.message_id, m.posted_at) for m in loaded] == [
        (user.message_id, user.posted_at),
        (assistant.message_id, assistant.posted_at),
    ]
    persisted = [
        json.loads(line)
        for line in (session_dir / "messages.jsonl").read_text().splitlines()
    ]
    assert persisted[0]["posted_at"] == "2026-01-02T03:04:05Z"

    def no_clock() -> datetime:
        raise AssertionError("resume must not consult the posting clock")

    with use_posting_clock(no_clock):
        resumed = build_test_agent_loop(config=config, cwd=tmp_path)
        resumed.session_logger.resume_existing_session(agent.session_id, session_dir)
        resumed.messages.reset(loaded)
        assert [m.posted_at for m in resumed.messages] == [
            user.posted_at,
            assistant.posted_at,
        ]
        await resumed.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_provider_timestamp_cannot_override_local_posting_clock(
    streaming: bool,
) -> None:
    chunk = mock_llm_chunk(content="answer")
    chunk.message.posted_at = datetime(2000, 1, 1, tzinfo=UTC)
    agent = build_test_agent_loop(
        backend=FakeBackend([chunk]), enable_streaming=streaming
    )
    with use_posting_clock(lambda: POSTED):
        events = [event async for event in agent.act("question")]
    assert agent.messages[-1].posted_at == POSTED
    assert all(
        event.posted_at == POSTED
        for event in events
        if isinstance(event, (AssistantEvent, ReasoningEvent))
    )
    await agent.aclose()


@pytest.mark.asyncio
async def test_legacy_saved_messages_are_not_backfilled(tmp_path: Path) -> None:
    (tmp_path / "meta.json").write_text(json.dumps({"total_messages": 2}))
    (tmp_path / "messages.jsonl").write_text(
        '{"role":"user","content":"old","message_id":"u"}\n'
        '{"role":"assistant","content":"answer","message_id":"a"}\n'
    )
    loaded, _ = SessionLoader.load_session(tmp_path)
    assert all(message.posted_at is None for message in loaded)
    agent = build_test_agent_loop()
    agent.messages.reset(loaded)
    assert all(message.posted_at is None for message in agent.messages)
    assert all(
        "posted_at" not in m.model_dump(exclude_none=True, mode="json") for m in loaded
    )
    await agent.aclose()


class InterruptedBackend(FakeBackend):
    async def complete_streaming(self, **kwargs: Any) -> AsyncGenerator[LLMChunk]:
        async for chunk in super().complete_streaming(**kwargs):
            yield chunk
        raise RuntimeError("stream interrupted")


@pytest.mark.asyncio
@pytest.mark.parametrize("interruption", ["error", "close", "cancel"])
async def test_interrupted_reasoning_first_keeps_timestamp(interruption: str) -> None:
    backend = InterruptedBackend([
        mock_llm_chunk(content="", reasoning_content="thinking"),
        mock_llm_chunk(content="answer"),
    ])
    config = build_test_vibe_config()
    inputs = CompletionInputs(
        model=config.get_active_model(),
        provider_name="test",
        emits_finish_reason=False,
        messages=(),
        tools=None,
        tool_choice=None,
        extra_headers={},
        metadata={},
        max_tokens=None,
    )
    resources = CallResources(
        backend=backend, stats=AgentStats(), process_message=lambda m: m.model_copy()
    )
    appended: list[TranscriptAppend] = []
    times = iter([POSTED])
    with use_posting_clock(lambda: next(times)):
        async with aclosing(
            LLMGateway().chat_streaming(inputs, resources, transcript=appended.append)
        ) as stream:
            first = await anext(stream)
            assert first.message.posted_at == POSTED
            if interruption == "error":
                with pytest.raises(RuntimeError, match="stream interrupted"):
                    _ = [chunk async for chunk in stream]
            elif interruption == "cancel":
                with pytest.raises(asyncio.CancelledError):
                    await stream.athrow(asyncio.CancelledError())
    assert len(appended) == 1
    assert appended[0].kind == "interrupted"
    assert appended[0].message.reasoning_content == "thinking"
    assert appended[0].message.posted_at == POSTED


class WaitingBackend(FakeBackend):
    def __init__(self) -> None:
        super().__init__([mock_llm_chunk(content="", reasoning_content="thinking")])
        self.waiting = asyncio.Event()

    async def complete_streaming(self, **kwargs: Any) -> AsyncGenerator[LLMChunk]:
        async for chunk in super().complete_streaming(**kwargs):
            yield chunk
        self.waiting.set()
        await asyncio.Event().wait()


@pytest.mark.asyncio
async def test_running_turn_cancellation_retains_canonical_partial() -> None:
    backend = WaitingBackend()
    agent = build_test_agent_loop(backend=backend, enable_streaming=True)
    times = iter([POSTED, POSTED + timedelta(seconds=1)])

    async def consume() -> None:
        async for _ in agent.act("question"):
            pass

    with use_posting_clock(lambda: next(times)):
        task = asyncio.create_task(consume())
        await asyncio.wait_for(backend.waiting.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert agent.messages[-1].reasoning_content == "thinking"
    assert agent.messages[-1].posted_at == POSTED + timedelta(seconds=1)
    await agent.aclose()


@pytest.mark.asyncio
async def test_clock_override_is_inherited_and_restored() -> None:
    with use_posting_clock(lambda: POSTED):

        async def resolve() -> datetime:
            return posting_time()

        assert await asyncio.create_task(resolve()) == POSTED
        with use_posting_clock(lambda: POSTED + timedelta(days=1)):
            assert posting_time() == POSTED + timedelta(days=1)
        assert posting_time() == POSTED
    assert posting_time() != POSTED


@pytest.mark.parametrize("adapter", ["generic", "mistral", "anthropic", "responses"])
def test_stamped_messages_never_leak_into_provider_payloads(adapter: str) -> None:
    messages = [
        LLMMessage(role=Role.system, content="system", posted_at=POSTED),
        LLMMessage(role=Role.user, content="question", posted_at=POSTED),
        LLMMessage(
            role=Role.assistant,
            content="answer",
            posted_at=POSTED,
            tool_calls=[
                ToolCall(
                    id="call",
                    index=0,
                    function=FunctionCall(name="bash", arguments="{}"),
                )
            ],
        ),
        LLMMessage(
            role=Role.tool,
            content="done",
            tool_call_id="call",
            name="bash",
            posted_at=POSTED,
        ),
    ]
    unstamped = [m.model_copy(update={"posted_at": None}) for m in messages]
    provider = ProviderConfig(
        name="test", api_base="https://example.test", api_key_env_var="TEST_KEY"
    )

    def payload(source: list[LLMMessage]) -> str:
        if adapter == "mistral":
            return json.dumps([
                MistralMapper()
                .prepare_message(m)
                .model_dump(mode="json", exclude_none=True)
                for m in source
            ])
        api = {
            "generic": OpenAIAdapter,
            "anthropic": AnthropicAdapter,
            "responses": OpenAIResponsesAdapter,
        }[adapter]()
        request = api.prepare_request(
            model_name="gpt-4.1" if adapter != "anthropic" else "claude-sonnet-4-5",
            messages=source,
            temperature=0.2,
            tools=None,
            max_tokens=None,
            tool_choice=None,
            enable_streaming=False,
            provider=provider,
        )
        return request.body.decode()

    wire = payload(messages)
    assert wire == payload(unstamped)
    assert "posted_at" not in wire and "postedAt" not in wire
    assert "2026-01-02" not in wire
