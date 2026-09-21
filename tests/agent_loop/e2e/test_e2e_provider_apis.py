from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import pytest
import respx

from chartreux.core.config import ChartreuxConfigSchema
from chartreux.core.events import ToolResultEvent
from tests.agent_loop.e2e.conftest import build_e2e_agent_loop
from tests.agent_loop.e2e.providers import anthropic, openai_responses
from tests.agent_loop.e2e.providers.base import (
    ProviderAPI,
    ProviderMocks,
    assistant_text,
)
from tests.backend.data import ANSWER_CONTEXT_TOKENS


@dataclass(frozen=True)
class ProviderScenario:
    """Everything the shared e2e suite needs to exercise one provider.

    Adding a provider is: implement `ProviderMocks`, then append one entry here.
    """

    id: str
    config: Callable[..., ChartreuxConfigSchema]
    api_cls: type[ProviderAPI]
    mocks: ProviderMocks


PROVIDER_SCENARIOS: list[ProviderScenario] = [
    ProviderScenario(
        id="anthropic",
        config=anthropic.e2e_config,
        api_cls=anthropic.API,
        mocks=anthropic.Mocks(),
    ),
    ProviderScenario(
        id="openai-responses",
        config=openai_responses.e2e_config,
        api_cls=openai_responses.API,
        mocks=openai_responses.Mocks(),
    ),
]


@contextmanager
def open_provider_api(
    api_cls: type[ProviderAPI], request: pytest.FixtureRequest
) -> Iterator[ProviderAPI]:
    """Instantiate a provider API and setup respx routers and monkeypatches"""
    api_cls.setup_monkeypatch(request.getfixturevalue("monkeypatch"))
    with respx.mock(base_url=api_cls.base_url, assert_all_called=False) as router:
        api = api_cls()
        api.setup_router(router)
        yield api


@pytest.fixture(params=PROVIDER_SCENARIOS, ids=lambda s: s.id)
def scenario(request: pytest.FixtureRequest) -> ProviderScenario:
    """Parametrized fixture for each provider scenario in the shared e2e suite."""
    return request.param


@pytest.fixture
def provider_api(
    scenario: ProviderScenario, request: pytest.FixtureRequest
) -> Iterator[ProviderAPI]:
    """Returned setup provider API for the current scenario"""
    with open_provider_api(scenario.api_cls, request) as api:
        yield api


class TestProviderCommonBehaviors:
    """Answer / stream / tool-call behaviors every provider shares."""

    @pytest.mark.asyncio
    async def test_agent_answers(
        self, scenario: ProviderScenario, provider_api: ProviderAPI
    ) -> None:
        provider_api.reply(scenario.mocks.answer("pong"))
        agent = build_e2e_agent_loop(config=scenario.config())

        events = [event async for event in agent.act("Reply with exactly: pong")]

        assert assistant_text(events) == "pong"
        assert agent.stats.context_tokens == ANSWER_CONTEXT_TOKENS

    @pytest.mark.asyncio
    async def test_agent_streams(
        self, scenario: ProviderScenario, provider_api: ProviderAPI
    ) -> None:
        provider_api.reply_stream(scenario.mocks.text_stream("pong"))
        agent = build_e2e_agent_loop(config=scenario.config(), enable_streaming=True)

        events = [event async for event in agent.act("Reply with exactly: pong")]

        assert assistant_text(events) == "pong"

    @pytest.mark.asyncio
    async def test_agent_executes_tool_call(
        self, scenario: ProviderScenario, provider_api: ProviderAPI
    ) -> None:
        provider_api.reply(
            scenario.mocks.tool_call("todo", {"action": "read"}),
            scenario.mocks.answer("Your list is empty."),
        )
        agent = build_e2e_agent_loop(config=scenario.config(enabled_tools=["todo"]))

        events = [event async for event in agent.act("What's on my todo list?")]

        assert any(isinstance(e, ToolResultEvent) for e in events)
        assert "Your list is empty." in assistant_text(events)

    @pytest.mark.asyncio
    async def test_agent_captures_reasoning(
        self, scenario: ProviderScenario, provider_api: ProviderAPI
    ) -> None:
        provider_api.reply(scenario.mocks.reasoning_answer("pong", "Let me think."))
        agent = build_e2e_agent_loop(config=scenario.config())

        events = [event async for event in agent.act("Reply with exactly: pong")]

        assert assistant_text(events) == "pong"
        assert any(
            m.reasoning_content and "Let me think." in m.reasoning_content
            for m in agent.messages
        )

    @pytest.mark.asyncio
    async def test_agent_streams_reasoning_then_runs_tool(
        self, scenario: ProviderScenario, provider_api: ProviderAPI
    ) -> None:
        provider_api.reply_streams(
            scenario.mocks.reasoning_tool_call_stream(
                "todo", {"action": "read"}, reasoning="thinking..."
            ),
            scenario.mocks.text_stream("Your list is empty."),
        )
        agent = build_e2e_agent_loop(
            config=scenario.config(enabled_tools=["todo"]), enable_streaming=True
        )

        events = [event async for event in agent.act("What's on my todo list?")]

        assert any(isinstance(e, ToolResultEvent) for e in events)
        assert "Your list is empty." in assistant_text(events)
        assert any(
            m.reasoning_content and "thinking..." in m.reasoning_content
            for m in agent.messages
        )
