from __future__ import annotations

from typing import Any

import pytest

from chartreux.agents import AgentSafety, AgentType
from chartreux.core.agents.launch import FrozenPersona, resolve_launch
from chartreux.core.agents.models import AgentProfile
from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
from chartreux.core.config.models import ModelConfig, ProviderConfig
from chartreux.core.llm_models import Backend, LLMMessage, Role
from chartreux.core.session_types import CommittedModelIdentity
from chartreux.core.subagents import (
    ImmutableLaunchPersonaError,
    InvalidLaunchModelError,
    InvalidLaunchPromptError,
    InvalidLaunchToolError,
    LaunchConfig,
    LaunchConfigError,
    LaunchToolOverride,
    MissingAgentProfileError,
)
from chartreux.core.tools.models import ToolPermission
from tests.conftest import build_test_vibe_config
from tests.stubs.fake_config_orchestrator import FakeConfigOrchestrator


@pytest.fixture
def profile() -> AgentProfile:
    return AgentProfile(
        name="worker",
        display_name="Worker",
        description="test",
        safety=AgentSafety.NEUTRAL,
        agent_type=AgentType.SUBAGENT,
        instructions="profile instructions",
    )


@pytest.fixture
def config() -> ChartreuxConfigSchema:
    provider = ProviderConfig(
        name="provider", api_base="https://example.test", backend=Backend.GENERIC
    )
    return build_test_vibe_config(
        active_model="small",
        providers=[provider],
        models=[
            ModelConfig(
                name="small",
                alias="small",
                provider="provider",
                thinking="off",
                supported_thinking_levels=["off", "low", "high"],
            ),
            ModelConfig(
                name="large",
                alias="large",
                provider="provider",
                thinking="high",
                supported_thinking_levels=["off", "high"],
            ),
        ],
    )


def _config_with_style(style: str) -> ChartreuxConfigSchema:
    provider = ProviderConfig(
        name="provider",
        api_base="https://example.test",
        backend=Backend.GENERIC,
        api_style=style,  # type: ignore[arg-type]
    )
    return build_test_vibe_config(
        active_model="small",
        providers=[provider],
        models=[ModelConfig(name="small", alias="small", provider="provider")],
    )


def resolve(
    config: ChartreuxConfigSchema,
    profile: AgentProfile,
    launch: LaunchConfig | None = None,
    **kwargs: Any,
):
    return resolve_launch(
        profile_name="worker",
        config=launch,
        parent_orchestrator=FakeConfigOrchestrator(config),
        tool_inventory={"bash": object(), "read_file": object()},
        profile_lookup=lambda _: profile,
        **kwargs,
    )


def test_retained_resolution_preserves_captured_profile_selection(
    config: ChartreuxConfigSchema, profile: AgentProfile
) -> None:
    parent = config.model_copy(update={"active_model": "large"})
    original = AgentProfile(**{
        **profile.__dict__,
        "overrides": {"active_model": "small"},
    })
    initial = resolve(parent, original)
    retained = resolve_launch(
        profile_name=None,
        config=None,
        parent_orchestrator=initial.orchestrator,
        tool_inventory={"bash": object()},
        retained_profile=AgentProfile(**{
            **profile.__dict__,
            "overrides": {"active_model": "large"},
        }),
        accumulated_overrides=initial.semantic_overrides,
        frozen_persona=initial.persona,
    )
    assert (initial.effective_model.alias, retained.effective_model.alias) == (
        "small",
        "small",
    )


def test_explicit_model_is_exact_and_empty_config_inherits(
    config: ChartreuxConfigSchema, profile: AgentProfile
) -> None:
    assert resolve(config, profile).effective_model.alias == "small"
    assert (
        resolve(config, profile, LaunchConfig(model="large")).effective_model.alias
        == "large"
    )


@pytest.mark.parametrize(
    ("launch", "error", "field"),
    [
        ({"model": "missing"}, InvalidLaunchModelError, "config.model"),
        (
            {"system_prompt_id": "missing-prompt"},
            InvalidLaunchPromptError,
            "config.system_prompt_id",
        ),
        (
            {"enabled_tools": ["missing*"]},
            InvalidLaunchToolError,
            "config.enabled_tools",
        ),
        (
            {"tools": {"missing": {"permission": "always"}}},
            InvalidLaunchToolError,
            "config.tools.missing",
        ),
    ],
)
def test_invalid_launches_are_typed_after_only_profile_lookup(
    config: ChartreuxConfigSchema,
    profile: AgentProfile,
    launch: dict[str, object],
    error: type[LaunchConfigError],
    field: str,
) -> None:
    calls = 0

    def factory(_: str) -> AgentProfile:
        nonlocal calls
        calls += 1
        return profile

    with pytest.raises(error) as raised:
        resolve_launch(
            profile_name="worker",
            config=LaunchConfig.model_validate(launch),
            parent_orchestrator=FakeConfigOrchestrator(config),
            tool_inventory={"bash": object()},
            profile_lookup=factory,
        )
    assert (raised.value.field, calls) == (field, 1)


def test_unknown_and_disallowed_catalog_models_are_typed(
    config: ChartreuxConfigSchema, profile: AgentProfile
) -> None:
    with pytest.raises(InvalidLaunchModelError, match="config.model"):
        resolve(config, profile, LaunchConfig(model="missing"))
    with pytest.raises(InvalidLaunchModelError, match="config.model"):
        resolve(
            config.model_copy(update={"allowed_models": ["small"]}),
            profile,
            LaunchConfig(model="large"),
        )


def test_thinking_validation_honors_catalog_declaration(
    config: ChartreuxConfigSchema, profile: AgentProfile
) -> None:
    assert (
        resolve(config, profile, LaunchConfig(thinking="low")).effective_thinking
        == "low"
    )


def test_accumulated_launch_config_deep_merges_tools_without_mutating_input(
    config: ChartreuxConfigSchema, profile: AgentProfile
) -> None:
    first = LaunchConfig(
        model="large",
        enabled_tools=["bash"],
        tools={
            "bash": LaunchToolOverride(
                allowlist=["git *"], permission=ToolPermission.ASK
            )
        },
    )
    candidate = resolve(
        config,
        profile,
        LaunchConfig(
            thinking="high",
            tools={"bash": LaunchToolOverride(permission=ToolPermission.ALWAYS)},
        ),
        accumulated_overrides=first,
    )
    assert candidate.semantic_overrides.model == "large"
    assert candidate.semantic_overrides.enabled_tools == ["bash"]
    assert candidate.semantic_overrides.tools is not None
    assert candidate.semantic_overrides.tools["bash"].allowlist == ["git *"]
    assert candidate.semantic_overrides.tools["bash"].permission == "always"
    assert first.tools is not None and first.tools["bash"].permission == "ask"


def test_retained_persona_cannot_change_but_identical_value_is_idempotent(
    config: ChartreuxConfigSchema, profile: AgentProfile
) -> None:
    persona = FrozenPersona("cli", "frozen")
    with pytest.raises(ImmutableLaunchPersonaError) as raised:
        resolve(
            config,
            profile,
            LaunchConfig(instructions="secret"),
            retained_profile=profile,
            frozen_persona=persona,
        )
    assert "secret" not in str(raised.value)
    candidate = resolve(
        config,
        profile,
        LaunchConfig(instructions="frozen", system_prompt_id="cli"),
        retained_profile=profile,
        frozen_persona=persona,
    )
    assert candidate.persona == persona


def test_tool_authority_ceiling_rejects_explicit_selection(
    config: ChartreuxConfigSchema, profile: AgentProfile
) -> None:
    with pytest.raises(InvalidLaunchToolError, match="config.enabled_tools"):
        resolve(
            config,
            profile,
            LaunchConfig(enabled_tools=["bash"]),
            authorized_tool_names=frozenset(),
        )


@pytest.mark.parametrize(
    ("style", "payload", "reasoning_content"),
    [
        ("openai-responses", {"type": "reasoning", "id": "r"}, None),
        ("openai-responses", {"type": "thinking"}, "portable"),
        ("anthropic", {"type": "thinking", "thinking": "prior"}, None),
    ],
)
def test_provider_reasoning_history_replayability_matrix(
    config: ChartreuxConfigSchema,
    profile: AgentProfile,
    style: str,
    payload: dict[str, str],
    reasoning_content: str | None,
) -> None:
    adapted = _config_with_style(style)
    history = [
        LLMMessage(
            role=Role.assistant,
            reasoning_payloads=[payload],
            reasoning_content=reasoning_content,
        )
    ]
    assert resolve(adapted, profile, history=history).effective_model.alias == "small"


def test_anthropic_history_normalizes_omitted_launch_thinking(
    config: ChartreuxConfigSchema, profile: AgentProfile
) -> None:
    anthropic = _config_with_style("anthropic")
    history = [
        LLMMessage(
            role=Role.assistant,
            reasoning_payloads=[{"type": "thinking", "thinking": "prior"}],
        )
    ]
    assert (
        resolve(
            anthropic,
            profile,
            accumulated_overrides=LaunchConfig(thinking="off"),
            history=history,
        ).effective_thinking
        == "medium"
    )


def test_child_launch_inherits_parent_committed_deployment(
    config: ChartreuxConfigSchema, profile: AgentProfile
) -> None:
    parent_identity = CommittedModelIdentity(
        base_model="large",
        provider="provider/default",
        wire_name="large",
        catalog_revision="test-fixture",
    )

    candidate = resolve(config, profile, committed_model=parent_identity)

    assert candidate.committed_model == parent_identity
    assert candidate.effective_model.name == "large"


def test_new_child_explicit_profile_model_beats_parent_committed_identity(
    config: ChartreuxConfigSchema, profile: AgentProfile
) -> None:
    parent_identity = CommittedModelIdentity(
        base_model="large",
        provider="provider/default",
        wire_name="large",
        catalog_revision="test-fixture",
    )
    selected = AgentProfile(**{
        **profile.__dict__,
        "name": "specialist",
        "overrides": {"active_model": "small"},
    })

    candidate = resolve_launch(
        profile_name="specialist",
        config=None,
        parent_orchestrator=FakeConfigOrchestrator(config),
        tool_inventory={"bash": object()},
        profile_lookup=lambda _: selected,
        committed_model=parent_identity,
    )

    assert candidate.effective_model.alias == "small"
    assert candidate.committed_model.base_model == "small"


def test_tag_launch_thinking_uses_resolved_base_override(
    config: ChartreuxConfigSchema, profile: AgentProfile
) -> None:
    assert config.catalog_snapshot is not None
    config.attach_catalog_snapshot(
        config.catalog_snapshot.__class__(
            config.catalog_snapshot.catalog.model_copy(
                update={"tags": {"panel": ("large",)}}
            ),
            config.catalog_snapshot.revision,
        )
    )

    candidate = resolve(config, profile, LaunchConfig(model="@panel", thinking="high"))

    assert candidate.effective_model.alias == "large"
    assert candidate.effective_thinking == "high"


def test_missing_profile_is_typed(config: ChartreuxConfigSchema) -> None:
    with pytest.raises(MissingAgentProfileError, match="agent"):
        resolve_launch(
            profile_name="missing",
            config=None,
            parent_orchestrator=FakeConfigOrchestrator(config),
            tool_inventory={},
            profile_lookup=lambda _: (_ for _ in ()).throw(ValueError()),
        )
