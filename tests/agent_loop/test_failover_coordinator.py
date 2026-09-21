from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
import os
from types import MethodType, TracebackType
from typing import cast
from unittest.mock import AsyncMock

import httpx
import pytest

from chartreux.core.agent_loop._loop import AgentLoop, AgentTurnOptions
from chartreux.core.agent_loop.backend_lifetime import BackendLifetime
from chartreux.core.agent_loop.errors import AgentLoopLLMResponseError
from chartreux.core.agent_loop.llm_gateway import TranscriptAppend, _append_interrupted
from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
from chartreux.core.config.models import ModelConfig
from chartreux.core.llm.failures import RequestRetryBudget
from chartreux.core.llm_models import (
    FunctionCall,
    ImageAttachment,
    InlineImageSource,
    LLMChunk,
    LLMMessage,
    Role,
    ToolCall,
)
from chartreux.core.model_catalog.availability import (
    AllDeploymentsUnavailableError,
    ExclusionReason,
)
from chartreux.core.model_catalog.loader import CatalogSnapshot
from chartreux.core.model_catalog.schema import ModelCatalog
from chartreux.core.session_types import CommittedModelIdentity, LaunchMetadataV2
from chartreux.core.subagents import AgentAvailability, AgentSummary, TaskResult
from tests.conftest import build_test_agent_loop
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend, FakeInterruptedStreamingBackend


def _snapshot(*, compaction: bool = False) -> CatalogSnapshot:
    models: dict[str, object] = {
        "base": {
            "deployments": [
                {"provider": "test/first", "name": "first"},
                {"provider": "test/second", "name": "second"},
            ]
        },
        "other": {"deployments": [{"provider": "test/second", "name": "other-second"}]},
    }
    if compaction:
        models["compact"] = {
            "deployments": [
                {"provider": "test/first", "name": "compact-first"},
                {"provider": "test/second", "name": "compact-second"},
            ]
        }
    return CatalogSnapshot(
        ModelCatalog.model_validate({
            "providers": {
                "test/first": {"api_base": "https://first.invalid"},
                "test/second": {"api_base": "https://second.invalid"},
            },
            "models": models,
            "tags": {},
        }),
        "test-revision",
    )


def _agent(*, streaming: bool = False, compaction: bool = False) -> AgentLoop:
    snapshot = _snapshot(compaction=compaction)
    config = ChartreuxConfigSchema.model_validate(
        {"active_model": "base", "compaction_model": "compact" if compaction else ""},
        context={"catalog_snapshot": snapshot},
    ).attach_catalog_snapshot(snapshot)
    return build_test_agent_loop(
        config=config, backend=FakeBackend(), enable_streaming=streaming
    )


def _thinking_agent() -> AgentLoop:
    snapshot = CatalogSnapshot(
        ModelCatalog.model_validate({
            "providers": {
                "test/first": {"api_base": "https://first.invalid"},
                "test/second": {"api_base": "https://second.invalid"},
            },
            "models": {
                "base": {
                    "deployments": [
                        {
                            "provider": "test/first",
                            "name": "first",
                            "supported_thinking_levels": ["off", "high"],
                        },
                        {
                            "provider": "test/second",
                            "name": "second",
                            "supported_thinking_levels": ["off"],
                        },
                    ]
                },
                "compact": {
                    "deployments": [
                        {"provider": "test/first", "name": "compact-first"},
                        {"provider": "test/second", "name": "compact-second"},
                    ]
                },
            },
            "tags": {},
        }),
        "test-revision",
    )
    config = ChartreuxConfigSchema.model_validate(
        {
            "active_model": "base",
            "compaction_model": "compact",
            "thinking_overrides": {"base": "high"},
        },
        context={"catalog_snapshot": snapshot},
    ).attach_catalog_snapshot(snapshot)
    return build_test_agent_loop(config=config, backend=FakeBackend())


def test_failover_uses_attempted_model_thinking_for_compaction() -> None:
    agent = _thinking_agent()
    active_model = agent.config.get_active_model()
    compaction_model = agent.config.get_compaction_model().model_copy(
        update={"thinking": "off"}
    )

    assert [
        candidate.resolved.deployment.provider
        for candidate in agent._failover_candidates(agent.messages, compaction_model)
    ] == ["test/first", "test/second"]
    assert [
        candidate.resolved.deployment.provider
        for candidate in agent._failover_candidates(agent.messages, active_model)
    ] == ["test/first"]


def test_compaction_failover_uses_compactor_thinking_not_conversation_model() -> None:
    snapshot = CatalogSnapshot(
        ModelCatalog.model_validate({
            "providers": {
                "test/mistral": {
                    "api_base": "https://mistral.invalid",
                    "backend": "mistral",
                }
            },
            "models": {
                "base": {
                    "deployments": [{"provider": "test/mistral", "name": "zai-glm-5-3"}]
                },
                "compact": {
                    "deployments": [
                        {
                            "provider": "test/mistral",
                            "name": "mistral-small",
                            "supported_thinking_levels": ["off"],
                        }
                    ],
                    "thinking": "off",
                },
            },
            "tags": {},
        }),
        "test-revision",
    )
    config = ChartreuxConfigSchema.model_validate(
        {"active_model": "base", "compaction_model": "compact"},
        context={"catalog_snapshot": snapshot},
    ).attach_catalog_snapshot(snapshot)
    agent = build_test_agent_loop(config=config, backend=FakeBackend())

    assert [
        candidate.resolved.deployment.provider
        for candidate in agent._failover_candidates(
            agent.messages,
            agent.config.get_compaction_model(),
            call_type="secondary_call",
        )
    ] == ["test/mistral"]


@pytest.mark.asyncio
async def test_compaction_fallback_respects_deployment_thinking_narrowing() -> None:
    snapshot = CatalogSnapshot(
        ModelCatalog.model_validate({
            "providers": {"test/first": {"api_base": "https://first.invalid"}},
            "models": {
                "base": {"deployments": [{"provider": "test/first", "name": "base"}]},
                "compact": {
                    "deployments": [
                        {
                            "provider": "test/first",
                            "name": "compact",
                            "supported_thinking_levels": ["high"],
                        }
                    ],
                    "thinking": "high",
                },
            },
            "tags": {},
        }),
        "test-revision",
    )
    config = ChartreuxConfigSchema.model_validate(
        {"active_model": "base", "compaction_model": "compact"},
        context={"catalog_snapshot": snapshot},
    ).attach_catalog_snapshot(snapshot)
    agent = build_test_agent_loop(config=config, backend=FakeBackend())
    complete = AsyncMock(return_value=mock_llm_chunk(content="<summary>ok</summary>"))
    agent.compaction_manager._complete = complete

    assert await agent.compaction_manager._fallback([], "summarize") == "ok"
    await_args = complete.await_args
    assert await_args is not None
    assert await_args.kwargs["model"].thinking == "high"


def test_failover_projects_only_current_request_context_for_image_compatibility() -> (
    None
):
    snapshot = CatalogSnapshot(
        ModelCatalog.model_validate({
            "providers": {
                "test/images": {"api_base": "https://images.invalid"},
                "test/text": {"api_base": "https://text.invalid"},
            },
            "models": {
                "base": {
                    "deployments": [
                        {
                            "provider": "test/images",
                            "name": "images",
                            "supports_images": True,
                        },
                        {"provider": "test/text", "name": "text"},
                    ]
                }
            },
            "tags": {},
        }),
        "test-revision",
    )
    config = ChartreuxConfigSchema.model_validate(
        {"active_model": "base"}, context={"catalog_snapshot": snapshot}
    ).attach_catalog_snapshot(snapshot)
    agent = build_test_agent_loop(config=config, backend=FakeBackend())
    image = ImageAttachment(
        source=InlineImageSource(data="image"), alias="image.png", mime_type="image/png"
    )
    agent.messages.extend([
        LLMMessage(role=Role.user, content="old image", images=[image]),
        LLMMessage(role=Role.user, content="summary", context_boundary="compaction"),
        LLMMessage(role=Role.user, content="current request"),
    ])

    assert [
        candidate.resolved.deployment.provider
        for candidate in agent._failover_candidates(
            agent.messages, agent.config.get_active_model()
        )
    ] == ["test/images", "test/text"]

    agent.messages.append(
        LLMMessage(role=Role.user, content="live image", images=[image])
    )
    assert [
        candidate.resolved.deployment.provider
        for candidate in agent._failover_candidates(
            agent.messages, agent.config.get_active_model()
        )
    ] == ["test/images"]


def _image_compatibility_agent(*, vision_available: bool) -> AgentLoop:
    snapshot = CatalogSnapshot(
        ModelCatalog.model_validate({
            "providers": {
                "test/text": {"api_base": "https://text.invalid"},
                "test/vision": {"api_base": "https://vision.invalid"},
            },
            "models": {
                "base": {
                    "deployments": [
                        {"provider": "test/text", "name": "text"},
                        {
                            "provider": "test/vision",
                            "name": "vision",
                            "supports_images": vision_available,
                        },
                    ]
                }
            },
            "tags": {},
        }),
        "test-revision",
    )
    config = ChartreuxConfigSchema.model_validate(
        {"active_model": "base"}, context={"catalog_snapshot": snapshot}
    ).attach_catalog_snapshot(snapshot)
    return build_test_agent_loop(config=config, backend=FakeBackend())


def _tool_result_with_image() -> LLMMessage:
    return LLMMessage(
        role=Role.tool,
        content="image tool result",
        images=[
            ImageAttachment(
                source=InlineImageSource(data="image"),
                alias="image.png",
                mime_type="image/png",
            )
        ],
    )


def test_compaction_failover_skips_nonvision_candidate_for_tool_result_images() -> None:
    agent = _image_compatibility_agent(vision_available=True)

    candidates = agent._failover_candidates(
        [_tool_result_with_image()],
        agent.config.get_active_model(),
        call_type="secondary_call",
    )

    assert [candidate.resolved.deployment.provider for candidate in candidates] == [
        "test/vision"
    ]


def test_compaction_failover_reports_no_eligible_deployment_for_tool_result_images() -> (
    None
):
    agent = _image_compatibility_agent(vision_available=False)

    with pytest.raises(AllDeploymentsUnavailableError) as error:
        agent._failover_candidates(
            [_tool_result_with_image()],
            agent.config.get_active_model(),
            call_type="secondary_call",
        )

    assert {exclusion.reason for exclusion in error.value.exclusions} == {
        ExclusionReason.IMAGES_UNSUPPORTED
    }


def _failover_backends(
    agent: AgentLoop, first: FakeBackend, second: FakeBackend
) -> None:
    def backend_for_attempt(
        _self: AgentLoop, model: ModelConfig, _budget: object
    ) -> FakeBackend:
        return {"test/first": first, "test/second": second}[model.provider]

    agent._backend_for_attempt = MethodType(backend_for_attempt, agent)  # type: ignore[method-assign]


def test_committed_active_model_ignores_historical_selection_expression() -> None:
    snapshot = _snapshot()
    config = ChartreuxConfigSchema.model_validate(
        {"active_model": "other"}, context={"catalog_snapshot": snapshot}
    ).attach_catalog_snapshot(snapshot)
    config.attach_committed_model(
        CommittedModelIdentity(
            base_model="base",
            provider="test/first",
            wire_name="first",
            catalog_revision="older-revision",
        )
    )

    assert config.get_active_model().name == "first"
    assert config.get_active_model().provider == "test/first"


@pytest.mark.asyncio
async def test_reload_recommits_changed_selection_and_persists_v2_identity() -> None:
    agent = _agent()
    assert not await agent.config_orchestrator.set_field(
        "/active_model", "other", reason="select another root model"
    )

    await agent.reload_with_initial_messages()

    assert agent.committed_model == CommittedModelIdentity(
        base_model="other",
        provider="test/second",
        wire_name="other-second",
        catalog_revision="test-revision",
    )
    assert agent.config.get_active_model().name == "other-second"
    metadata = agent.session_logger.session_metadata
    assert metadata is not None
    assert isinstance(metadata.launch_config, LaunchMetadataV2)
    assert metadata.launch_config.committed_model == agent.committed_model


@pytest.mark.asyncio
async def test_per_deployment_retry_budget_preserves_time_for_next_provider() -> None:
    agent = _agent()
    budgets: list[RequestRetryBudget] = []
    first = FakeBackend(exception_to_raise=httpx.ConnectError("down"))
    second = FakeBackend([mock_llm_chunk(content="ok")])

    def backend_for_attempt(
        _self: AgentLoop, model: ModelConfig, budget: object
    ) -> FakeBackend:
        budgets.append(cast(RequestRetryBudget, budget))
        return {"test/first": first, "test/second": second}[model.provider]

    agent._backend_for_attempt = MethodType(backend_for_attempt, agent)  # type: ignore[method-assign]
    assert (await agent._chat()).message.content == "ok"
    assert len(budgets) == 2
    assert budgets[0].exhausted is True
    assert budgets[1].remaining > 0


def test_per_deployment_retry_budget_never_extends_parent_deadline() -> None:
    agent = _agent()
    parent = RequestRetryBudget(0.0)

    attempt = agent._attempt_retry_budget(parent, has_alternative=True)

    assert attempt.exhausted is True


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_finalization_failure_rolls_back_publication_and_identity(
    streaming: bool,
) -> None:
    agent = _agent(streaming=streaming)
    original_identity = agent.committed_model
    original_backend = agent._backend_lifetime.active
    agent.config_orchestrator.availability_registry.record_failure("base", "test/first")
    replacement = FakeBackend([mock_llm_chunk(content="ok")])
    _failover_backends(agent, FakeBackend(), replacement)

    def fail_metadata() -> None:
        raise RuntimeError("metadata installation failed")

    agent.install_launch_metadata = fail_metadata  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="metadata installation failed"):
        if streaming:
            _ = [chunk async for chunk in agent._chat_streaming()]
        else:
            await agent._chat()

    assert agent.committed_model == original_identity
    assert agent._backend_lifetime.active is original_backend


@pytest.mark.asyncio
async def test_reload_prepares_backend_from_retained_committed_deployment() -> None:
    agent = _agent()
    _failover_backends(
        agent,
        FakeBackend(exception_to_raise=httpx.ConnectError("down")),
        FakeBackend([mock_llm_chunk(content="ok")]),
    )
    await agent._chat()
    captured_models: list[ModelConfig] = []

    def backend_factory(
        _self: AgentLoop, config: ChartreuxConfigSchema | None = None
    ) -> FakeBackend:
        if config is None:
            raise RuntimeError("reload should supply its target config")
        captured_models.append(config.get_active_model())
        return FakeBackend()

    agent.backend_factory = MethodType(backend_factory, agent)  # type: ignore[method-assign]
    target = ChartreuxConfigSchema.model_validate(
        {"active_model": "base"}, context={"catalog_snapshot": _snapshot()}
    ).attach_catalog_snapshot(_snapshot())

    _ = agent._prepare_reload(target, False)

    assert captured_models[-1].provider == "test/second"
    assert captured_models[-1].name == "second"


@pytest.mark.asyncio
async def test_permanent_probe_failure_releases_recovery_claim() -> None:
    agent = _agent()
    registry = agent.config_orchestrator.availability_registry
    registry.initial_cooldown = 0
    registry.record_failure("base", "test/first")
    permanent = httpx.HTTPStatusError(
        "auth",
        request=httpx.Request("GET", "https://x"),
        response=httpx.Response(401, request=httpx.Request("GET", "https://x")),
    )
    _failover_backends(agent, FakeBackend(exception_to_raise=permanent), FakeBackend())
    with pytest.raises(RuntimeError, match="API error"):
        await agent._chat()
    assert registry.admission("base", "test/first")[:2] == (True, True)


@pytest.mark.asyncio
async def test_streaming_input_failure_rolls_back_publication_and_identity() -> None:
    agent = _agent(streaming=True)
    original_identity = agent.committed_model
    original_backend = agent._injected_backend

    def fail_inputs(**_kwargs: object) -> object:
        raise ValueError("input construction failed")

    agent._completion_inputs = fail_inputs  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="input construction failed"):
        [chunk async for chunk in agent._chat_streaming()]
    assert agent.committed_model == original_identity
    with agent._backend_lifetime.borrow() as current:
        assert current is original_backend


@pytest.mark.asyncio
async def test_coordinator_is_used_by_chat_complete_and_streaming_with_one_budget() -> (
    None
):
    agent = _agent()
    called: list[tuple[bool, str | None]] = []

    async def attempt(**kwargs: object) -> LLMChunk:
        called.append((
            bool(kwargs["transcript"]),
            cast(str | None, kwargs["call_type"]),
        ))
        return mock_llm_chunk(content="ok")

    agent._attempt_nonstreaming = AsyncMock(side_effect=attempt)  # type: ignore[method-assign]
    await agent._chat(call_type="main_call")
    await agent._complete(
        model=agent.config.get_active_model(),
        messages=(),
        tools=None,
        tool_choice=None,
        call_type="secondary_call",
    )
    assert called == [(True, "main_call"), (False, "secondary_call")]

    streaming = _agent(streaming=True)
    first = FakeBackend(exception_to_raise=httpx.ConnectError("down"))
    second = FakeBackend([mock_llm_chunk(content="replayed")])
    _failover_backends(streaming, first, second)
    assert [chunk.message.content async for chunk in streaming._chat_streaming()] == [
        "replayed"
    ]
    assert streaming._completion_providers[-1] == ("test/first", "test/second")


@pytest.mark.asyncio
async def test_eligible_failure_switches_and_permanent_failure_stops() -> None:
    agent = _agent()
    first = FakeBackend(exception_to_raise=httpx.ConnectError("down"))
    second = FakeBackend([mock_llm_chunk(content="ok")])
    _failover_backends(agent, first, second)
    result = await agent._chat()
    assert result.message.content == "ok"
    committed = agent.committed_model
    if committed is None:
        raise RuntimeError("expected a committed model after successful chat")
    assert committed.provider == "test/second"

    stopped = _agent()
    permanent = httpx.HTTPStatusError(
        "auth",
        request=httpx.Request("GET", "https://x"),
        response=httpx.Response(401, request=httpx.Request("GET", "https://x")),
    )
    _failover_backends(
        stopped, FakeBackend(exception_to_raise=permanent), FakeBackend()
    )
    with pytest.raises(RuntimeError, match="API error"):
        await stopped._chat()
    assert stopped._completion_providers[-1] == ("test/first",)


@pytest.mark.asyncio
async def test_streaming_replays_before_semantic_delta_but_not_after_content_reasoning_or_tool_call() -> (
    None
):
    replay = _agent(streaming=True)
    first = _InterruptingBackend([mock_llm_chunk(content="", prompt_tokens=1)])
    second = FakeBackend([mock_llm_chunk(content="replayed")])
    _failover_backends(replay, first, second)
    assert [chunk.message.content async for chunk in replay._chat_streaming()] == [
        "",
        "replayed",
    ]

    interrupted = _agent(streaming=True)
    first_partial = _InterruptingBackend([mock_llm_chunk(content="partial")])
    _failover_backends(
        interrupted, first_partial, FakeBackend([mock_llm_chunk(content="never")])
    )
    with pytest.raises(RuntimeError, match="API error"):
        [chunk async for chunk in interrupted._chat_streaming()]
    committed = interrupted.committed_model
    if committed is None:
        raise RuntimeError("expected a committed model after interrupted chat")
    assert committed.provider == "test/first"
    assert (
        interrupted.config_orchestrator.availability_registry.cooldown_until(
            "base", "test/first"
        )
        is not None
    )


@pytest.mark.asyncio
async def test_partial_stream_transcript_keeps_producing_deployment_identity() -> None:
    agent = _agent(streaming=True)
    previous_identity = agent.committed_model
    if previous_identity is None:
        raise RuntimeError("expected initial committed model")
    registry = agent.config_orchestrator.availability_registry
    registry.record_failure("base", "test/first")
    outcomes: list[TranscriptAppend] = []
    append_transcript = agent._append_transcript

    def capture(outcome: TranscriptAppend) -> None:
        outcomes.append(outcome)
        append_transcript(outcome)

    agent._append_transcript = capture  # type: ignore[method-assign]
    _failover_backends(
        agent,
        FakeBackend(),
        FakeInterruptedStreamingBackend([mock_llm_chunk(content="from second")]),
    )

    with pytest.raises(RuntimeError, match="API error"):
        _ = [chunk async for chunk in agent._chat_streaming()]

    assert agent.committed_model == previous_identity
    assert outcomes[-1].kind == "interrupted"
    assert outcomes[-1].committed_model == CommittedModelIdentity(
        base_model="base",
        provider="test/second",
        wire_name="second",
        catalog_revision="test-revision",
    )
    assert agent.messages[-1].content == "from second"
    assert agent.messages[-1].deployment_identity == {
        "base_model": "base",
        "provider": "test/second",
        "wire_name": "second",
        "catalog_revision": "test-revision",
    }


@pytest.mark.asyncio
async def test_attempt_rejects_missing_prepared_backend_before_publication() -> None:
    agent = _agent()
    published = False

    def missing_backend(_self: AgentLoop, _model: ModelConfig, _budget: object) -> None:
        return None

    def publish(_replacement: object) -> object:
        nonlocal published
        published = True
        raise AssertionError("publication must be unreachable")

    agent._backend_for_attempt = MethodType(missing_backend, agent)  # type: ignore[method-assign]
    agent._backend_lifetime.publish_reversible = publish  # type: ignore[method-assign]

    with pytest.raises(
        AgentLoopLLMResponseError, match="produced no backend for publication"
    ):
        await agent._chat()

    assert published is False

    agent = _agent()
    agent.committed_model = CommittedModelIdentity(
        base_model="base",
        provider="test/first",
        wire_name="first",
        catalog_revision="test-revision",
    )
    registry = agent.config_orchestrator.availability_registry
    registry.record_failure("base", "test/first")
    _failover_backends(
        agent, FakeBackend([mock_llm_chunk(content="recovered")]), FakeBackend()
    )

    _ = [
        event
        async for event in agent.act(
            "retry", turn_options=AgentTurnOptions(user_initiated_retry=True)
        )
    ]

    assert agent._completion_providers[-1] == ("test/first",)
    assert registry.cooldown_until("base", "test/first") is None


@pytest.mark.asyncio
async def test_cancellation_rolls_back_publication_and_identity_and_closes_attempts_once() -> (
    None
):
    agent = _agent()
    original = agent.committed_model
    cancelling = FakeBackend(exception_to_raise=asyncio.CancelledError())
    _failover_backends(agent, cancelling, FakeBackend())
    with pytest.raises(asyncio.CancelledError):
        await agent._chat()
    assert agent.committed_model == original
    await agent.aclose()


@pytest.mark.asyncio
async def test_switch_visibility_identity_metadata_and_destination_compaction_model(
    caplog: pytest.LogCaptureFixture,
) -> None:
    agent = _agent(compaction=True)
    first = FakeBackend(exception_to_raise=httpx.ConnectError("down"))
    second = FakeBackend([mock_llm_chunk(content="ok")])
    _failover_backends(agent, first, second)
    await agent._chat()
    identity = agent.committed_model
    assert identity == CommittedModelIdentity(
        base_model="base",
        provider="test/second",
        wire_name="second",
        catalog_revision="test-revision",
    )
    assert agent.completion_metadata_since((0, 0))["providers_used"] == [
        ["test/first", "test/second"]
    ]
    assert agent.completion_metadata_since((0, 0))["switch_notices"] == [
        {
            "base_model": "base",
            "old_provider": "test/first",
            "new_provider": "test/second",
            "reason": "connection",
        }
    ]
    assert "Model deployment switched" in caplog.text
    assert (
        agent._attempt_model(
            agent._failover_candidates(agent.messages, agent.config.get_active_model())[
                0
            ].resolved,
            call_type="secondary_call",
        ).name
        == "compact-second"
    )


def test_alternative_forces_immediate_failover_but_standalone_keeps_retry_budget() -> (
    None
):
    agent = _agent()
    parent = RequestRetryBudget(10.0)

    failover_attempt = agent._attempt_retry_budget(parent, has_alternative=True)
    standalone_attempt = agent._attempt_retry_budget(parent, has_alternative=False)

    assert failover_attempt.exhausted is True
    assert standalone_attempt is parent


@pytest.mark.asyncio
async def test_superseded_publication_completion_is_idempotent_and_keeps_newer() -> (
    None
):
    backends = [_ClosingBackend() for _ in range(3)]
    lifetime = BackendLifetime(backends[0])
    older = lifetime.publish_reversible(backends[1])
    newer = lifetime.publish_reversible(backends[2])

    assert older.rollback() is False
    older.finalize(whole_turn_active=False)
    assert lifetime.active is backends[2]
    newer.finalize(whole_turn_active=False)
    lifetime.drain(whole_turn_active=False)
    await lifetime.aclose()

    assert [backend.closes for backend in backends] == [1, 1, 1]


@pytest.mark.asyncio
async def test_teardown_closes_backends_owned_by_unfinished_publications() -> None:
    backends = [_ClosingBackend() for _ in range(3)]
    lifetime = BackendLifetime(backends[0])
    lifetime.publish_reversible(backends[1])
    lifetime.publish_reversible(backends[2])

    await lifetime.aclose()

    assert [backend.closes for backend in backends] == [1, 1, 1]


@pytest.mark.asyncio
async def test_metadata_restoration_failure_still_releases_recovery_probe() -> None:
    agent = _agent()
    registry = agent.config_orchestrator.availability_registry
    registry.initial_cooldown = 0
    registry.record_failure("base", "test/first")
    candidate = agent._failover_candidates(
        agent.messages, agent.config.get_active_model()
    )[0]
    publication = agent._backend_lifetime.publish_reversible(FakeBackend())

    def fail_metadata() -> None:
        raise RuntimeError("metadata restoration failed")

    agent.install_launch_metadata = fail_metadata  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="metadata restoration failed"):
        agent._rollback_attempt(publication, agent.committed_model, candidate)

    assert registry.admission("base", "test/first")[:2] == (True, True)
    await agent.aclose()


@pytest.mark.asyncio
async def test_outer_stream_aclose_releases_attempt_resources_and_preserves_partial_output() -> (
    None
):
    agent = _agent(streaming=True)
    registry = agent.config_orchestrator.availability_registry
    registry.initial_cooldown = 0
    registry.record_failure("base", "test/first")
    backend = FakeBackend([mock_llm_chunk(content="partial", prompt_tokens=1)])
    _failover_backends(agent, backend, FakeBackend())

    stream = agent._chat_streaming()
    assert (await anext(stream)).message.content == "partial"
    await stream.aclose()

    assert agent._backend_lifetime._borrow_counts == {}
    assert registry.admission("base", "test/first")[:2] == (True, True)
    assert agent.messages[-1].content == "partial"
    assert agent.stats.session_prompt_tokens == 1
    assert agent.stats.session_completion_tokens == 5
    assert agent.stats.session_cost is None
    await agent.aclose()


@pytest.mark.asyncio
async def test_backend_attempts_close_once_and_fd_count_stays_flat() -> None:
    backends = [_ClosingBackend() for _ in range(4)]
    before = len(os.listdir("/proc/self/fd"))
    lifetime = BackendLifetime(backends[0])
    for replacement in backends[1:]:
        lifetime.publish_reversible(replacement).finalize(whole_turn_active=False)
        lifetime.drain(whole_turn_active=False)
        await asyncio.sleep(0)
    await lifetime.aclose()
    assert len(os.listdir("/proc/self/fd")) - before <= 5
    assert [backend.closes for backend in backends] == [1, 1, 1, 1]


def test_failover_visibility_is_accepted_by_task_results_and_agent_summaries() -> None:
    metadata = {
        "switch_notices": [{"base_model": "base"}],
        "providers_used": [["p1", "p2"]],
    }
    result = TaskResult(response="ok", turns_used=1, completed=True, metadata=metadata)
    summary = AgentSummary(
        agent_id="agent",
        profile="worker",
        availability=AgentAvailability.IDLE,
        current_run_id=None,
        current_run_status=None,
        effective_model="test/second",
        base_model="base",
        active_provider="test/second",
    )
    assert result.metadata == metadata
    assert (summary.base_model, summary.active_provider) == ("base", "test/second")


@pytest.mark.asyncio
async def test_attempt_prepares_inputs_before_publication() -> None:
    agent = _agent()
    published = False

    def fail_inputs(**_kwargs: object) -> object:
        raise ValueError("input construction failed")

    def publish(_replacement: object) -> object:
        nonlocal published
        published = True
        raise AssertionError("publication must be unreachable")

    agent._completion_inputs = fail_inputs  # type: ignore[method-assign]
    agent._backend_lifetime.publish_reversible = publish  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="input construction failed"):
        await agent._chat()
    assert published is False


@pytest.mark.asyncio
async def test_preparation_failure_releases_recovery_probe_claim() -> None:
    agent = _agent()
    registry = agent.config_orchestrator.availability_registry
    registry.initial_cooldown = 0
    registry.record_failure("base", "test/first")

    def fail_backend(_self: AgentLoop, _model: ModelConfig, _budget: object) -> object:
        raise RuntimeError("backend construction failed")

    agent._backend_for_attempt = MethodType(fail_backend, agent)  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="backend construction failed"):
        await agent._chat()
    assert registry.admission("base", "test/first")[:2] == (True, True)


def test_interrupted_transcript_preserves_reasoning_and_incomplete_tool_calls() -> None:
    outcomes: list[TranscriptAppend] = []
    tool_call = ToolCall(
        id="partial", function=FunctionCall(name="todo", arguments='{"task":')
    )
    chunk = LLMChunk(
        message=LLMMessage(
            role=Role.assistant,
            content="",
            reasoning_content="unfinished thought",
            tool_calls=[tool_call],
        )
    )

    _append_interrupted(outcomes.append, chunk)

    assert outcomes[0].kind == "interrupted"
    assert outcomes[0].message.reasoning_content == "unfinished thought"
    assert outcomes[0].message.tool_calls == [tool_call]


@pytest.mark.asyncio
async def test_reload_metadata_failure_rolls_back_identity_and_backend() -> None:
    agent = _agent()
    original_identity = agent.committed_model
    original_backend = agent.backend
    target = ChartreuxConfigSchema.model_validate(
        {"active_model": "other"}, context={"catalog_snapshot": _snapshot()}
    ).attach_catalog_snapshot(_snapshot())
    prepared = agent._prepare_reload(target, False)

    def fail_metadata() -> None:
        raise RuntimeError("metadata installation failed")

    agent.install_launch_metadata = fail_metadata  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="metadata installation failed"):
        agent._commit_reload(prepared, reset_middleware=False)
    assert agent.committed_model == original_identity
    assert agent.backend is original_backend


@pytest.mark.asyncio
async def test_chat_and_streaming_share_attempt_transaction() -> None:
    agent = _agent(streaming=True)
    modes: list[bool] = []
    original = agent._attempt_completion

    async def recording(**kwargs: object) -> AsyncGenerator[LLMChunk]:
        modes.append(cast(bool, kwargs["streaming"]))
        async for chunk in original(**kwargs):  # type: ignore[arg-type]
            yield chunk

    agent._attempt_completion = recording  # type: ignore[method-assign]
    _failover_backends(
        agent,
        FakeBackend([mock_llm_chunk(content="nonstream")]),
        FakeBackend([mock_llm_chunk(content="stream")]),
    )
    await agent._chat()
    _ = [chunk async for chunk in agent._chat_streaming()]
    assert modes == [False, True]


class _ClosingBackend(FakeBackend):
    def __init__(self) -> None:
        super().__init__()
        self.closes = 0

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.closes += 1


class _InterruptingBackend(FakeBackend):
    async def complete_streaming(self, **kwargs: object) -> AsyncGenerator[LLMChunk]:
        async for chunk in super().complete_streaming(**kwargs):  # type: ignore[arg-type]
            yield chunk
        raise httpx.ConnectError("down")
