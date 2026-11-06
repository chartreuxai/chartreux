from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import pytest

from chartreux.app_server._projection import (
    project_config,
    project_config_view,
    project_history,
    project_mcp,
    project_session_log,
    project_stats,
)
from chartreux.app_server._projector import EventProjector
from chartreux.app_server.models import (
    AgentStatsSnapshot,
    AnsweredCallbackState,
    CancelledCallbackState,
    CancelledEffectState,
    CompletedEffectState,
    EffectCallDisplay,
    ExpiredCallbackState,
    FailedEffectState,
    GenericEffectDetail,
    MCPSourceStatus,
    OpenCallbackState,
    PublicCallbackEntry,
    PublicCheckpointEntry,
    PublicEffectEntry,
    PublicEntryGenerationStatus,
    PublicHistoryEntry,
    PublicMessageEntry,
    PublicReasoningEntry,
    QuestionChoice,
    ResourceContentBlock,
    RunningEffectState,
    TextContentBlock,
    UserAnswer,
    UserInputCallbackDetail,
    UserInputCallbackOutput,
    UserQuestion,
    UserQuestionRequest,
    UserQuestionResult,
)
from chartreux.core.config import (
    ChartreuxConfigSchema,
    MCPStdio,
    SessionLoggingConfig,
    ThinkingLevel,
)
from chartreux.core.dispatch.route_models import PackageTier, PurposeRoute, RouteTarget
from chartreux.core.llm_models import (
    FunctionCall,
    ImageAttachment,
    InlineImageSource,
    LLMMessage,
    Role,
    ToolCall,
)
from chartreux.core.model_catalog.defaults import SHIPPED_CATALOG
from chartreux.core.model_catalog.loader import CatalogSnapshot, merge_catalog_overlay
from chartreux.core.session.transcript_archive import ArchivedPrefixV1
from chartreux.core.utils import CANCELLATION_TAG, TOOL_ERROR_TAG
from chartreux.user_content import UserDisplayContent, UserResourceLink
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.stubs.fake_mcp_registry import FakeMCPRegistry


def _history_with_tool_output(tool_name: str, output: str):
    agent_loop = build_test_agent_loop(
        config=build_test_vibe_config(enabled_tools=["task"])
    )
    agent_loop.messages.reset([
        LLMMessage(role=Role.system, content="system"),
        LLMMessage(role=Role.user, content="delegate"),
        LLMMessage(
            role=Role.assistant,
            content="",
            reasoning_content="Thinking",
            reasoning_message_id="reasoning-1",
            tool_calls=[
                ToolCall(
                    id="tool-1",
                    function=FunctionCall(
                        name=tool_name,
                        arguments=json.dumps({"task": "inspect", "agent": "worker"}),
                    ),
                )
            ],
        ),
        LLMMessage(
            role=Role.tool, name=tool_name, tool_call_id="tool-1", content=output
        ),
    ])
    return project_history(agent_loop)


@pytest.mark.parametrize(
    ("tag", "state_type"),
    [(CANCELLATION_TAG, CancelledEffectState), (TOOL_ERROR_TAG, FailedEffectState)],
)
def test_persisted_tagged_tool_result_is_terminal_and_stripped(
    tag: str, state_type: type[CancelledEffectState | FailedEffectState]
) -> None:
    history = _history_with_tool_output("task", f"<{tag}>Stopped</{tag}>")

    reasoning = next(
        entry for entry in history if isinstance(entry, PublicReasoningEntry)
    )
    effect = next(entry for entry in history if isinstance(entry, PublicEffectEntry))
    assert reasoning.generation_status == "completed"
    assert isinstance(effect.state, state_type)
    assert effect.state.output_text == "Stopped"
    assert effect.generation_status == "completed"


def test_persisted_known_tool_output_is_not_treated_as_typed_result() -> None:
    history = _history_with_tool_output(
        "task", "response: done\nturns_used: 1\ncompleted: True"
    )

    effect = next(entry for entry in history if isinstance(entry, PublicEffectEntry))
    assert isinstance(effect.state, CompletedEffectState)
    assert effect.state.output is None
    assert effect.state.output_text.startswith("response: done")


def test_config_view_redacts_persistence_paths() -> None:
    agent_loop = build_test_agent_loop()

    config = project_config(agent_loop)

    assert "sessionLogging" not in config.model_dump(mode="json", by_alias=True)


def test_stats_projection_includes_cached_token_counts() -> None:
    agent_loop = build_test_agent_loop()
    agent_loop.stats.session_cached_tokens = 42
    agent_loop.stats.last_turn_cached_tokens = 7
    agent_loop.stats.input_price_per_million = 1.0
    agent_loop.stats.cached_input_price_per_million = 0.1

    stats = project_stats(agent_loop)
    serialized = stats.model_dump(mode="json", by_alias=True)

    assert stats.session_cached_tokens == 42
    assert stats.last_turn_cached_tokens == 7
    assert stats.cached_input_price_per_million == 0.1
    assert serialized["sessionCachedTokens"] == 42
    assert serialized["lastTurnCachedTokens"] == 7


def test_stats_projection_preserves_cost_ledger_fields() -> None:
    agent_loop = build_test_agent_loop()
    agent_loop.stats.known_cost_total = 1.25
    agent_loop.stats.has_unknown_cost = True

    stats = project_stats(agent_loop)

    assert stats.known_cost_total == 1.25
    assert stats.has_unknown_cost is True
    assert stats.session_cost is None


def test_snapshot_token_only_legacy_stats_have_no_complete_cost() -> None:
    snapshot = AgentStatsSnapshot.model_validate({"session_prompt_tokens": 100})

    assert snapshot.has_unknown_cost is True
    assert snapshot.session_cost is None


def test_snapshot_explicit_zero_ledger_is_a_complete_zero_cost() -> None:
    snapshot = AgentStatsSnapshot(known_cost_total=0.0, has_unknown_cost=False)

    assert snapshot.session_cost == 0.0


def test_snapshot_session_cost_discounts_cached_tokens() -> None:
    snapshot = AgentStatsSnapshot(
        session_prompt_tokens=1_000_000,
        session_completion_tokens=0,
        session_cached_tokens=400_000,
        input_price_per_million=1.0,
        cached_input_price_per_million=0.1,
        known_cost_total=0.64,
    )
    # 600k * $1/M + 400k * $0.1/M = $0.64
    assert snapshot.session_cost == pytest.approx(0.64)


def test_snapshot_session_cost_bills_cached_at_input_rate_when_unset() -> None:
    snapshot = AgentStatsSnapshot(
        session_prompt_tokens=1_000_000,
        session_completion_tokens=0,
        session_cached_tokens=400_000,
        input_price_per_million=1.0,
        known_cost_total=1.0,
    )
    assert snapshot.session_cost == pytest.approx(1.0)


def test_snapshot_session_cost_never_negative_when_cached_exceeds_prompt() -> None:
    snapshot = AgentStatsSnapshot(
        session_prompt_tokens=100_000,
        session_completion_tokens=0,
        session_cached_tokens=190_000,
        input_price_per_million=1.0,
        cached_input_price_per_million=0.1,
        known_cost_total=0.01,
    )
    assert snapshot.session_cost == pytest.approx(0.01)


def test_config_view_flags_pinned_active_model() -> None:
    from chartreux.core.config import ModelConfig
    from chartreux.core.config.layers.overrides import OverridesLayer

    models = [
        ModelConfig(name="model-a", provider="mistral", alias="alpha"),
        ModelConfig(name="model-b", provider="mistral", alias="beta"),
    ]
    agent_loop = build_test_agent_loop(
        config=build_test_vibe_config(models=models, active_model="beta")
    )
    agent_loop.config_orchestrator.insert_layer(
        OverridesLayer(data={"active_model": "beta"}),
        len(agent_loop.config_orchestrator.layers),
    )
    agent_loop.config_orchestrator.rebuild()

    config = project_config(agent_loop)

    assert config.active_model_pinned is True
    assert config.active_model.alias == "beta"


def test_config_view_reports_unpinned_default_model() -> None:
    from chartreux.core.config import ModelConfig

    models = [
        ModelConfig(name="model-a", provider="mistral", alias="alpha"),
        ModelConfig(name="model-b", provider="mistral", alias="beta"),
    ]
    agent_loop = build_test_agent_loop(
        config=build_test_vibe_config(models=models, active_model="")
    )

    config = project_config(agent_loop)

    assert config.active_model_pinned is False
    # The unpinned state resolves to the shipped catalog default for display.
    assert config.default_model_alias == "glm-5-3"
    assert config.active_model.alias == "glm-5-3"


def test_config_view_default_alias_never_follows_the_active_model() -> None:
    """``default_model_alias`` names the configured default, never the pin.

    It used to be read off the active model unless the caller passed
    ``active_model_pinned``, so any projection that omitted the flag -- every
    unified-harness runtime snapshot -- advertised the user's own pinned model
    as the "Default (currently ...)" hint.
    """
    from chartreux.core.config import ModelConfig

    models = [
        ModelConfig(name="model-a", provider="mistral", alias="alpha"),
        ModelConfig(name="model-b", provider="mistral", alias="beta"),
    ]
    pinned_config = build_test_vibe_config(models=models, active_model="beta")

    pinned = project_config_view(pinned_config, active_model_pinned=True)
    # A caller that forgets the flag still must not echo the pin back as the
    # default; only ``active_model_pinned`` itself may differ.
    unflagged = project_config_view(pinned_config)
    unpinned = project_config_view(
        build_test_vibe_config(models=models, active_model="")
    )

    assert pinned.active_model.alias == "beta"
    assert pinned.default_model_alias == "glm-5-3"
    assert pinned.default_model_display_name == "mistral/zai-glm-5-3"
    assert unflagged.default_model_alias == "glm-5-3"
    assert unpinned.active_model.alias == "glm-5-3"
    assert unpinned.default_model_alias == "glm-5-3"


def test_config_view_hydrates_display_name_from_alias() -> None:
    from chartreux.core.config import ModelConfig

    models = [
        ModelConfig(name="model-a", provider="mistral", alias="alpha"),
        ModelConfig(
            name="zai-glm-5-2",
            provider="mistral",
            alias="glm-5-2",
            display_name="glm-5.2 (Mistral Hosted)",
        ),
    ]
    agent_loop = build_test_agent_loop(
        config=build_test_vibe_config(models=models, active_model="alpha")
    )

    config = project_config(agent_loop)

    # UI renders canonical provider/wire display names from the catalog.
    assert "mistral/model-a" in [model.display_name for model in config.models]
    assert config.model_display_name("glm-5-2") == "mistral/zai-glm-5-2"
    assert config.default_model_display_name == "mistral/zai-glm-5-3"
    # An alias that names no configured model is echoed back unchanged.
    assert config.model_display_name("unknown") == "unknown"


def test_config_view_projects_the_live_route_table() -> None:
    """The wire carries the SAME inspection the browser and doctor render.

    The live table is projected server-side, so a reload that swaps the table
    reaches an already-open browser through the ordinary config resource
    instead of being re-derived by the client.
    """
    from chartreux.core.model_catalog.loader import CatalogSnapshot
    from tests.conftest import cost_tier_catalog

    routes = {
        "review.custom": PurposeRoute(
            targets=(
                RouteTarget(
                    model="premium-model",
                    identity="reviewer",
                    tier=PackageTier.SUBSTANTIVE,
                    seat="analytical",
                ),
                RouteTarget(
                    model="missing-model",
                    identity="coder",
                    tier=PackageTier.MECHANICAL,
                    seat="execution",
                    overlay="review-execution",
                ),
            ),
            requires_distinct_models=2,
        )
    }
    config = build_test_vibe_config(
        orchestration={"routes": routes}, active_model="budget-model"
    ).attach_catalog_snapshot(CatalogSnapshot(cost_tier_catalog(), "fixture"))

    view = project_config_view(config)

    assert view.routes_error is None
    (route,) = view.routes
    assert route.purpose == "review.custom"
    assert route.requires_distinct_models == 2
    # One target cannot resolve: no count is claimed, so launch diversity is
    # UNKNOWN while the structural reload guard still passes.
    assert route.resolved_distinct_models is None
    assert route.reload_status == "pass"
    assert route.launch_status == "unknown"

    analytical, execution = route.targets
    assert analytical.seat == "analytical"
    assert analytical.identity == "reviewer"
    assert analytical.tier == PackageTier.SUBSTANTIVE.value
    assert analytical.binding == "premium-model"
    assert analytical.preview == "premium-model"
    assert analytical.error is None
    assert execution.seat == "execution"
    assert execution.overlay == "review-execution"
    assert execution.binding == "missing-model"
    assert execution.preview is None
    assert "missing-model" in (execution.error or "")


def test_config_view_carries_the_route_inspection_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An inspection that cannot start reports why instead of fabricating rows."""
    from chartreux.core.dispatch import route_inspection

    monkeypatch.setattr(
        route_inspection,
        "inspect_config_routes",
        lambda config: route_inspection.RouteTableReport((), "catalog unattached"),
    )

    view = project_config_view(build_test_vibe_config())

    assert view.routes == []
    assert view.routes_error == "catalog unattached"


_OTHER_PROVIDER: dict[str, object] = {"api_base": "https://other.test/v1"}
_OTHER_MODEL: dict[str, object] = {
    "deployments": [{"provider": "other-openai", "name": "other-model"}]
}


@pytest.mark.parametrize(
    ("overlay", "allowed_models"),
    [
        pytest.param(
            {
                "providers": {"other-openai": _OTHER_PROVIDER},
                "models": {"other-model": _OTHER_MODEL},
            },
            ["other-model"],
            id="default-excluded-by-allowlist",
        ),
        pytest.param(
            {
                "providers": {
                    "other-openai": _OTHER_PROVIDER,
                    "mistral": {"disabled": True},
                },
                "models": {"other-model": _OTHER_MODEL},
            },
            [],
            id="default-provider-disabled",
        ),
    ],
)
def test_config_view_keeps_active_model_when_default_is_unavailable(
    overlay: dict[str, object], allowed_models: list[str]
) -> None:
    catalog = merge_catalog_overlay(SHIPPED_CATALOG, overlay)
    snapshot = CatalogSnapshot(catalog, "test")
    schema = ChartreuxConfigSchema.model_validate(
        {"active_model": "other-model", "allowed_models": allowed_models},
        context={"catalog_snapshot": snapshot},
    ).attach_catalog_snapshot(snapshot)

    config = project_config_view(schema)

    assert config.active_model.alias == "other-model"
    assert "other-model" in [model.alias for model in config.models]
    assert config.default_model_alias == "none"
    assert config.default_model_display_name == "none"


@pytest.mark.asyncio
async def test_session_log_is_persisted_only_after_it_is_saved(tmp_path: Path) -> None:
    agent_loop = build_test_agent_loop(
        config=build_test_vibe_config(
            session_logging=SessionLoggingConfig(enabled=True, save_dir=str(tmp_path))
        )
    )

    fresh = project_session_log(agent_loop)
    await agent_loop.persist_empty_session()
    saved = project_session_log(agent_loop)
    agent_loop.session_logger.reset_session("replacement")
    replacement = project_session_log(agent_loop)

    assert fresh.persisted is False
    assert fresh.path is None
    assert saved.persisted is True
    assert saved.path is not None
    assert replacement.persisted is False
    assert replacement.path is None


def test_persisted_user_message_preserves_images_and_display_metadata() -> None:
    agent_loop = build_test_agent_loop()
    metadata = UserDisplayContent(
        version="1.0.0",
        host="mistral-vscode",
        content=[{"type": "workspace_mention", "name": "app.py"}],
    )
    agent_loop.messages.reset([
        LLMMessage(
            role=Role.user,
            content="Look at app.py",
            message_id="user-1",
            images=[
                ImageAttachment(
                    source=InlineImageSource(data="aW1hZ2U="),
                    alias="diagram.png",
                    mime_type="image/png",
                )
            ],
            user_display_content=metadata,
        )
    ])

    entry = project_history(agent_loop)[0]

    assert isinstance(entry, PublicMessageEntry)
    assert [image.alias for image in entry.images] == ["diagram.png"]
    assert entry.user_display_content == metadata


def test_persisted_user_message_preserves_structured_resources() -> None:
    agent_loop = build_test_agent_loop()
    agent_loop.messages.reset([
        LLMMessage(
            role=Role.user,
            content=(
                "Review the reference\n\n"
                "Resource: Specification\nURI: file:///workspace/spec.md"
            ),
            input_text="Review the reference",
            resources=[
                UserResourceLink(
                    uri="file:///workspace/spec.md",
                    media_type="text/markdown",
                    title="Specification",
                )
            ],
        )
    ])

    entry = project_history(agent_loop)[0]

    assert isinstance(entry, PublicMessageEntry)
    assert entry.text == "Review the reference"
    resource = next(
        block for block in entry.content if isinstance(block, ResourceContentBlock)
    )
    assert resource.resource.uri == "file:///workspace/spec.md"
    assert resource.resource.media_type == "text/markdown"


def test_project_mcp_discovery_error_marks_server_unavailable() -> None:
    server = MCPStdio(name="broken", transport="stdio", command="fake-cmd")
    config = build_test_vibe_config(mcp_servers=[server])
    registry = FakeMCPRegistry()
    registry.sync_active_servers([server])
    agent_loop = build_test_agent_loop(config=config, mcp_registry=registry)

    state = project_mcp(agent_loop, discovery_errors={"broken": "no binary found"})

    (source,) = [s for s in state.sources if s.name == "broken"]
    assert source.status is MCPSourceStatus.UNAVAILABLE


def test_project_mcp_discovery_error_does_not_affect_healthy_server() -> None:
    healthy = MCPStdio(name="healthy", transport="stdio", command="fake-cmd")
    broken = MCPStdio(name="broken", transport="stdio", command="missing-cmd")
    config = build_test_vibe_config(mcp_servers=[healthy, broken])
    registry = FakeMCPRegistry()
    registry.sync_active_servers([healthy, broken])
    agent_loop = build_test_agent_loop(config=config, mcp_registry=registry)

    state = project_mcp(agent_loop, discovery_errors={"broken": "no binary found"})

    statuses = {s.name: s.status for s in state.sources}
    assert statuses["broken"] is MCPSourceStatus.UNAVAILABLE
    assert statuses["healthy"] is not MCPSourceStatus.UNAVAILABLE


def test_persisted_compaction_boundary_projects_completed_checkpoint() -> None:
    agent_loop = build_test_agent_loop()
    agent_loop.messages.reset([
        LLMMessage(role=Role.user, content="Before", message_id="user-1"),
        LLMMessage(
            role=Role.user,
            content="Compacted context",
            injected=True,
            message_id="compaction-1",
            context_boundary="compaction",
        ),
        LLMMessage(role=Role.assistant, content="After", message_id="assistant-1"),
    ])

    history = project_history(agent_loop)

    checkpoint = next(
        entry for entry in history if isinstance(entry, PublicCheckpointEntry)
    )
    assert checkpoint.id == "checkpoint:compaction:compaction-1"
    assert checkpoint.kind == "compaction"
    assert checkpoint.message == "Context compacted"
    assert checkpoint.generation_status == "completed"


def _open_question_callback() -> tuple[EventProjector, PublicCallbackEntry]:
    projector = EventProjector("session-1", "turn-1")
    projector.start_effect(
        "tool-1",
        title="ask_user_question",
        detail=GenericEffectDetail(
            tool_name="ask_user_question",
            input=None,
            display=EffectCallDisplay(
                summary="Ask a question",
                verb="Asking",
                message="Ask a question",
                settled_verb="Asked",
                settled_message="Ask a question",
                status_text="Asking",
            ),
        ),
    )
    projector.open_callback(
        "callback-1",
        UserInputCallbackDetail(
            request=UserQuestionRequest(
                questions=[
                    UserQuestion(
                        question="Ship it?",
                        options=[
                            QuestionChoice(label="Yes"),
                            QuestionChoice(label="No"),
                        ],
                    )
                ]
            ),
            related_entry_id="tool-1",
        ),
        "Ship it?",
    )
    entry = next(
        item for item in projector.history if isinstance(item, PublicCallbackEntry)
    )
    return projector, entry


def _question_answer() -> UserInputCallbackOutput:
    return UserInputCallbackOutput(
        result=UserQuestionResult(
            answers=[UserAnswer(question="Ship it?", answer="Yes")]
        )
    )


def test_close_callback_publishes_terminal_cancelled_state_and_unblocks_effect() -> (
    None
):
    projector, entry = _open_question_callback()
    assert isinstance(entry.state, OpenCallbackState)

    updates = projector.close_callback("callback-1", "App server closed")

    assert [update.method for update in updates] == [
        "history/entryUpdated",
        "history/entryUpdated",
    ]
    closed = next(
        item for item in projector.history if isinstance(item, PublicCallbackEntry)
    )
    assert isinstance(closed.state, CancelledCallbackState)
    assert closed.state.reason == "App server closed"
    assert closed.generation_status is PublicEntryGenerationStatus.COMPLETED
    effect = next(
        item for item in projector.history if isinstance(item, PublicEffectEntry)
    )
    assert isinstance(effect.state, RunningEffectState)


def test_close_callback_is_idempotent_and_never_overwrites_an_answer() -> None:
    projector, _ = _open_question_callback()
    projector.resolve_callback("callback-1", _question_answer())

    assert projector.close_callback("callback-1", "App server closed") == []
    answered = next(
        item for item in projector.history if isinstance(item, PublicCallbackEntry)
    )
    assert isinstance(answered.state, AnsweredCallbackState)


def test_close_callback_never_overwrites_an_expired_callback() -> None:
    projector, _ = _open_question_callback()
    projector.expire_callback("callback-1", "unanswered_timeout")

    assert projector.close_callback("callback-1", "App server closed") == []
    expired = next(
        item for item in projector.history if isinstance(item, PublicCallbackEntry)
    )
    assert isinstance(expired.state, ExpiredCallbackState)


def test_duplicate_close_publishes_exactly_one_terminal_state() -> None:
    projector, _ = _open_question_callback()

    assert len(projector.close_callback("callback-1", "Turn ended")) == 2
    assert projector.close_callback("callback-1", "Turn ended") == []
    assert projector.close_callback("unknown-callback", "Turn ended") == []


def _mistral_config(wire_name: str, thinking: ThinkingLevel):
    from chartreux.core.config import ModelConfig

    return build_test_vibe_config(
        active_model="active",
        models=[
            ModelConfig(
                name=wire_name, provider="mistral", alias="active", thinking=thinking
            )
        ],
    )


def _anthropic_config():
    from chartreux.core.config import ModelConfig, ProviderConfig
    from chartreux.core.llm_models import Backend

    provider = ProviderConfig(
        name="claude",
        api_base="https://claude.test/v1",
        api_key_env_var="ANTHROPIC_API_KEY",
        backend=Backend.GENERIC,
        api_style="anthropic",
    )
    return build_test_vibe_config(
        active_model="claude",
        providers=[provider],
        models=[
            ModelConfig(
                name="claude-wire", provider="claude", alias="claude", thinking="off"
            )
        ],
    )


def test_effective_thinking_respects_deployment_subset() -> None:
    from chartreux.core.config import ModelConfig

    config = build_test_vibe_config(
        active_model="active",
        models=[
            ModelConfig(
                name="mistral-large-4",
                provider="mistral",
                alias="active",
                thinking="medium",
                supported_thinking_levels=["low", "high"],
            )
        ],
    )
    effective = project_config(build_test_agent_loop(config=config)).effective_thinking
    assert effective is not None
    assert effective.effective == "high"
    assert effective.api_effort == "high"
    assert effective.matches_requested is False


def test_effective_thinking_forced_anthropic_history_uses_declared_level() -> None:
    from chartreux.core.config import ModelConfig, ProviderConfig
    from chartreux.core.llm_models import Backend

    provider = ProviderConfig(
        name="claude",
        api_base="https://claude.test/v1",
        api_key_env_var="ANTHROPIC_API_KEY",
        backend=Backend.GENERIC,
        api_style="anthropic",
    )
    config = build_test_vibe_config(
        active_model="claude",
        providers=[provider],
        models=[
            ModelConfig(
                name="claude-wire",
                provider="claude",
                alias="claude",
                thinking="off",
                supported_thinking_levels=["off", "max"],
            )
        ],
    )
    agent_loop = build_test_agent_loop(config=config)
    agent_loop.messages.append(
        LLMMessage(
            role=Role.assistant,
            content="Answered",
            reasoning_payloads=[{"type": "thinking", "thinking": "considering"}],
        )
    )

    effective = project_config(agent_loop).effective_thinking

    assert effective is not None
    assert effective.semantic == "medium"
    assert effective.effective == "max"
    assert effective.api_effort == "max"
    assert effective.matches_requested is False


def test_effective_thinking_glm_medium_is_api_effort_high() -> None:
    """GLM-5.3 does not declare ``medium``; it resolves to the nearest ``high``."""
    agent_loop = build_test_agent_loop(config=_mistral_config("zai-glm-5-3", "medium"))

    effective = project_config(agent_loop).effective_thinking

    assert effective is not None
    assert effective.requested == "medium"
    assert effective.semantic == "medium"
    assert effective.effective == "high"
    assert effective.api_effort == "high"
    assert effective.known is True
    assert effective.matches_requested is False


def test_effective_thinking_mistral_large_4_low_is_api_effort_low() -> None:
    """Mistral Large 4 declares low/medium/high reasoning efforts."""
    agent_loop = build_test_agent_loop(config=_mistral_config("mistral-large-4", "low"))

    effective = project_config(agent_loop).effective_thinking

    assert effective is not None
    assert effective.requested == "low"
    assert effective.effective == "low"
    assert effective.api_effort == "low"
    assert effective.known is True
    assert effective.matches_requested is True


def test_effective_thinking_unlisted_mistral_model_uses_the_provider_table() -> None:
    """A model without its own entry keeps the provider's legacy wire mapping."""
    agent_loop = build_test_agent_loop(config=_mistral_config("zai-glm-future", "low"))

    effective = project_config(agent_loop).effective_thinking

    assert effective is not None
    assert effective.api_effort == "none"
    assert effective.known is True
    assert effective.matches_requested is False


def test_effective_thinking_equal_case_matches_the_requested_level() -> None:
    agent_loop = build_test_agent_loop(
        config=_mistral_config("mistral-large-4", "high")
    )

    effective = project_config(agent_loop).effective_thinking

    assert effective is not None
    assert effective.requested == "high"
    assert effective.api_effort == "high"
    assert effective.known is True
    assert effective.matches_requested is True


def test_effective_thinking_anthropic_off_without_history_stays_off() -> None:
    agent_loop = build_test_agent_loop(config=_anthropic_config())

    effective = project_config(agent_loop).effective_thinking

    assert effective is not None
    assert effective.requested == "off"
    assert effective.semantic == "off"
    assert effective.api_effort is None
    assert effective.known is True
    assert effective.matches_requested is True


def test_effective_thinking_anthropic_off_with_thinking_history_forces_medium() -> None:
    """Retained Anthropic thinking keeps the model thinking on."""
    agent_loop = build_test_agent_loop(config=_anthropic_config())
    agent_loop.messages.append(
        LLMMessage(
            role=Role.assistant,
            content="Answered",
            reasoning_payloads=[{"type": "thinking", "thinking": "considering"}],
        )
    )

    effective = project_config(agent_loop).effective_thinking

    assert effective is not None
    assert effective.requested == "off"
    assert effective.semantic == "medium"
    assert effective.api_effort == "medium"
    assert effective.known is True
    assert effective.matches_requested is False


def test_effective_thinking_reflects_appended_history_on_the_next_projection() -> None:
    """The estimate is recomputed from the loop, never cached at startup."""
    agent_loop = build_test_agent_loop(config=_anthropic_config())
    before = project_config(agent_loop).effective_thinking
    assert before is not None and before.matches_requested is True

    agent_loop.messages.append(
        LLMMessage(
            role=Role.assistant,
            content="Answered",
            reasoning_payloads=[{"type": "thinking", "thinking": "considering"}],
        )
    )
    after = project_config(agent_loop).effective_thinking

    assert after is not None
    assert after.semantic == "medium"
    assert after.matches_requested is False


def test_effective_thinking_is_unknown_without_a_committed_deployment() -> None:
    agent_loop = build_test_agent_loop(
        config=_mistral_config("mistral-large-4", "high")
    )
    agent_loop.committed_model = None

    effective = project_config(agent_loop).effective_thinking

    assert effective is not None
    assert effective.requested == "high"
    assert effective.known is False
    assert effective.matches_requested is None


def test_config_view_without_conversation_context_has_no_estimate() -> None:
    """A sessionless config read must not fabricate a next-request estimate."""
    view = project_config_view(build_test_vibe_config())

    assert view.effective_thinking is None


def test_config_view_reports_each_deployment_declared_thinking_levels() -> None:
    """Clients offer exactly the declared levels, in vocabulary order."""
    from chartreux.core.config import ModelConfig

    config = build_test_vibe_config(
        active_model="glm",
        models=[
            ModelConfig(name="zai-glm-5-3", provider="mistral", alias="glm"),
            ModelConfig(name="mistral-large-4", provider="mistral", alias="large"),
        ],
    )

    view = project_config_view(config)

    declared = {
        model.alias: model.thinking_levels
        for model in view.models
        if model.alias in {"glm", "large"}
    }
    assert declared == {
        "glm": ["low", "high", "max"],
        "large": ["low", "medium", "high"],
    }


def _sealed_transcript(
    directory: Path,
    records: list[dict[str, object]],
    *,
    session_id: str,
    message_epoch: int,
) -> ArchivedPrefixV1:
    """Persist the records and describe their byte prefix."""
    payload = b"".join(json.dumps(record).encode("utf-8") + b"\n" for record in records)
    (directory / "messages.jsonl").write_bytes(payload)
    return ArchivedPrefixV1(
        session_id=session_id,
        message_epoch=message_epoch,
        archived_count=len(records),
        prefix_end_byte=len(payload),
        prefix_sha256=hashlib.sha256(payload).hexdigest(),
        boundary_message_id="boundary-1",
        trim_mode="tombstones",
    )


def _archived_history(
    directory: Path, records: list[dict[str, object]], live: list[LLMMessage]
) -> list[PublicHistoryEntry]:
    agent_loop = build_test_agent_loop()
    agent_loop.session_logger.session_dir = directory
    agent_loop.messages.reset(live)
    agent_loop.messages.set_archive_descriptor(
        _sealed_transcript(
            directory,
            records,
            session_id=agent_loop.session_id,
            message_epoch=agent_loop.messages.message_epoch,
        )
    )
    return project_history(agent_loop)


def _live_history(live: list[LLMMessage]) -> list[PublicHistoryEntry]:
    agent_loop = build_test_agent_loop()
    agent_loop.messages.reset(live)
    return project_history(agent_loop)


def _visible(entries: list[PublicHistoryEntry]) -> list[str]:
    visible: list[str] = []
    for entry in entries:
        if isinstance(entry, PublicMessageEntry):
            block = entry.content[0]
            visible.append(
                block.text or "" if isinstance(block, TextContentBlock) else ""
            )
        elif isinstance(entry, PublicReasoningEntry):
            visible.append(f"reasoning:{entry.text}")
        elif isinstance(entry, PublicEffectEntry):
            visible.append(f"effect:{entry.title}")
    return visible


_SEALED_RECORDS: list[dict[str, object]] = [
    {"role": "user", "message_id": "user-1", "content": "one"},
    {
        "role": "assistant",
        "message_id": "assistant-1",
        "content": "two",
        "tool_calls": [
            {"id": "call-1", "function": {"name": "read_file", "arguments": "{}"}}
        ],
    },
    {
        "role": "tool",
        "message_id": "tool-1",
        "tool_call_id": "call-1",
        "content": "file body",
    },
]


def _trimmed_live_messages() -> list[LLMMessage]:
    """The RAM view after a trim: the sealed prefix lost its tool structure."""
    return [
        LLMMessage(role=Role.system, content="system"),
        LLMMessage(role=Role.user, message_id="user-1", content="one"),
        LLMMessage(role=Role.assistant, message_id="assistant-1", content="two"),
        LLMMessage(role=Role.tool, message_id="tool-1", tool_call_id="call-1"),
        LLMMessage(role=Role.user, message_id="user-2", content="three"),
    ]


def test_history_reconstruction_reads_the_sealed_prefix_from_disk(
    tmp_path: Path,
) -> None:
    """Archived turns come from the persisted originals, not the trimmed RAM
    copies, and the live tail still follows them: no turn is hidden and no
    turn is projected twice.
    """
    entries = _archived_history(tmp_path, _SEALED_RECORDS, _trimmed_live_messages())

    assert _visible(entries) == ["one", "two", "effect:read_file", "three"]


def test_history_without_archive_state_keeps_the_live_projection() -> None:
    """The unarchived path is untouched: it still projects the live copies,
    degraded tool structure included — which is exactly why the archived path
    reads the persisted originals instead.
    """
    assert _visible(_live_history(_trimmed_live_messages())) == [
        "one",
        "two",
        "effect:tool",
        "three",
    ]


def test_history_ignores_a_transcript_it_has_no_authority_for(tmp_path: Path) -> None:
    """A descriptor naming another session never authorizes a disk read."""
    agent_loop = build_test_agent_loop()
    agent_loop.session_logger.session_dir = tmp_path
    agent_loop.messages.reset(_trimmed_live_messages())
    agent_loop.messages.set_archive_descriptor(
        _sealed_transcript(
            tmp_path,
            _SEALED_RECORDS,
            session_id="another-session",
            message_epoch=agent_loop.messages.message_epoch,
        )
    )

    assert _visible(project_history(agent_loop)) == [
        "one",
        "two",
        "effect:tool",
        "three",
    ]


def test_history_degrades_to_the_live_view_when_the_prefix_is_corrupt(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """An unverifiable prefix warns and projects the live view: every turn the
    RAM copies still carry stays visible, and nothing raises.
    """
    agent_loop = build_test_agent_loop()
    agent_loop.session_logger.session_dir = tmp_path
    agent_loop.messages.reset(_trimmed_live_messages())
    descriptor = _sealed_transcript(
        tmp_path,
        [dict(record) for record in _SEALED_RECORDS],
        session_id=agent_loop.session_id,
        message_epoch=agent_loop.messages.message_epoch,
    )
    transcript = tmp_path / "messages.jsonl"
    transcript.write_bytes(transcript.read_bytes().replace(b"one", b"ONE"))
    agent_loop.messages.set_archive_descriptor(descriptor)

    with caplog.at_level(logging.WARNING):
        entries = project_history(agent_loop)

    # The changed disk bytes (``ONE``) never reach the transcript.
    assert _visible(entries) == ["one", "two", "effect:tool", "three"]
    assert "Sealed transcript prefix is unreadable" in caplog.text


def test_history_keeps_sealed_turns_the_live_list_no_longer_holds(
    tmp_path: Path,
) -> None:
    """The disk is the record of the conversation: a sealed turn the live list
    no longer holds is still projected rather than silently dropped.
    """
    live = [
        LLMMessage(role=Role.system, content="system"),
        LLMMessage(role=Role.user, message_id="user-1", content="one"),
    ]

    entries = _archived_history(tmp_path, _SEALED_RECORDS, live)

    assert _visible(entries) == ["one", "two", "effect:read_file"]
