from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from chartreux.agents import AgentSafety, AgentType
from chartreux.core.agents.launch import FrozenPersona, resolve_launch
from chartreux.core.agents.models import AgentProfile
from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
from chartreux.core.config.layers.overrides import OverridesLayer
from chartreux.core.config.models import ModelConfig, ProviderConfig, ThinkingLevel
from chartreux.core.llm_models import Backend, LLMMessage, Role
from chartreux.core.model_catalog.schema import RoleDefinition
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
    parent_orchestrator = kwargs.pop("parent_orchestrator", None)
    return resolve_launch(
        profile_name="worker",
        config=launch,
        parent_orchestrator=parent_orchestrator or FakeConfigOrchestrator(config),
        tool_inventory={"bash": object(), "read_file": object()},
        profile_lookup=lambda _: profile,
        **kwargs,
    )


def _with_role(
    config: ChartreuxConfigSchema,
    name: str,
    models: tuple[str, ...],
    thinking: ThinkingLevel | None = None,
) -> ChartreuxConfigSchema:
    assert config.catalog_snapshot is not None
    assert len(models) == 1
    model = models[0]
    config.attach_catalog_snapshot(
        config.catalog_snapshot.__class__(
            config.catalog_snapshot.catalog.model_copy(
                update={
                    "roles": {
                        **config.catalog_snapshot.catalog.roles,
                        name: RoleDefinition(
                            model=model,
                            thinking=thinking
                            or ("high" if model == "large" else "off"),
                        ),
                    }
                }
            ),
            config.catalog_snapshot.revision,
        )
    )
    return config


def test_retained_resolution_preserves_committed_model_over_profile_role(
    config: ChartreuxConfigSchema, profile: AgentProfile
) -> None:
    config = _with_role(config, "specialist", ("small",))
    parent = config.model_copy(update={"active_model": "large"})
    initial = resolve(
        parent, AgentProfile(**{**profile.__dict__, "role": "specialist"})
    )
    retained = resolve_launch(
        profile_name=None,
        config=None,
        parent_orchestrator=initial.orchestrator,
        tool_inventory={"bash": object()},
        retained_profile=AgentProfile(**{**profile.__dict__, "role": "specialist"}),
        accumulated_overrides=initial.semantic_overrides,
        frozen_persona=initial.persona,
        committed_model=initial.committed_model,
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


def test_task_launch_unknown_model_lists_canonical_choices(
    config: ChartreuxConfigSchema, profile: AgentProfile
) -> None:
    with pytest.raises(InvalidLaunchModelError) as raised:
        resolve(config, profile, LaunchConfig(model="luna"))
    assert raised.value.field == "config.model"
    assert "Unknown model expression 'luna'" in str(raised.value)
    assert "Valid canonical models: glm-5-3, large, mistral-large-4, small" in str(
        raised.value
    )

    config = _with_role(config, "specialist", ("small",))
    with pytest.raises(InvalidLaunchModelError) as raised:
        resolve(config, profile, LaunchConfig(model="@luna"))
    assert "@specialist" in str(raised.value)


def test_launch_model_schema_describes_accepted_expression() -> None:
    assert LaunchConfig.model_json_schema()["properties"]["model"]["description"] == (
        "canonical model name or `@role`"
    )


def test_thinking_validation_honors_catalog_declaration(
    config: ChartreuxConfigSchema, profile: AgentProfile
) -> None:
    assert (
        resolve(config, profile, LaunchConfig(thinking="low")).effective_thinking
        == "low"
    )


def test_launch_tool_override_rejects_removed_ask_permission() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="'always' or 'never'"):
        LaunchToolOverride.model_validate({"permission": "ask"})


def test_accumulated_launch_config_deep_merges_tools_without_mutating_input(
    config: ChartreuxConfigSchema, profile: AgentProfile
) -> None:
    first = LaunchConfig(
        model="large",
        enabled_tools=["bash"],
        tools={
            "bash": LaunchToolOverride(
                allowlist=["git *"], permission=ToolPermission.NEVER
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
    assert first.tools is not None and first.tools["bash"].permission == "never"


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
        provider="provider",
        wire_name="large",
        catalog_revision="test-fixture",
    )

    candidate = resolve(config, profile, committed_model=parent_identity)

    assert candidate.committed_model.base_model == parent_identity.base_model
    assert candidate.committed_model.provider == parent_identity.provider
    assert candidate.committed_model.thinking == "high"
    assert candidate.effective_model.name == "large"


def test_new_child_profile_role_beats_parent_committed_identity(
    config: ChartreuxConfigSchema, profile: AgentProfile
) -> None:
    config = _with_role(config, "specialist", ("small",))
    parent_identity = CommittedModelIdentity(
        base_model="large",
        provider="provider",
        wire_name="large",
        catalog_revision="test-fixture",
    )
    selected = AgentProfile(**{
        **profile.__dict__,
        "name": "specialist",
        "role": "specialist",
    })

    candidate = resolve_launch(
        profile_name="specialist",
        config=None,
        parent_orchestrator=FakeConfigOrchestrator(config),
        tool_inventory={"bash": object()},
        profile_lookup=lambda _: selected,
        committed_model=parent_identity,
    )

    assert candidate.orchestrator.config.active_model == "@specialist"
    assert candidate.effective_model.alias == "small"
    assert candidate.committed_model.base_model == "small"


def test_default_profile_role_beats_parent_committed_identity(
    config: ChartreuxConfigSchema, profile: AgentProfile
) -> None:
    config = _with_role(config, "small", ("small",))
    parent_identity = CommittedModelIdentity(
        base_model="large",
        provider="provider",
        wire_name="large",
        catalog_revision="test-fixture",
    )
    default_worker = AgentProfile(**{**profile.__dict__, "role": "small"})

    candidate = resolve_launch(
        profile_name=None,
        config=None,
        parent_orchestrator=FakeConfigOrchestrator(config),
        tool_inventory={"bash": object()},
        profile_lookup=lambda _: default_worker,
        committed_model=parent_identity,
    )

    assert candidate.orchestrator.config.active_model == "@small"
    assert candidate.effective_model.alias == "small"
    assert candidate.committed_model.base_model == "small"


def test_role_launch_thinking_uses_resolved_base_override(
    config: ChartreuxConfigSchema, profile: AgentProfile
) -> None:
    config = _with_role(config, "panel", ("large",))

    candidate = resolve(config, profile, LaunchConfig(model="@panel", thinking="high"))

    assert candidate.effective_model.alias == "large"
    assert candidate.effective_thinking == "high"


def test_shared_model_roles_keep_distinct_thinking_and_explicit_override(
    config: ChartreuxConfigSchema, profile: AgentProfile
) -> None:
    _with_role(config, "brief", ("small",), "off")
    _with_role(config, "deep", ("small",), "high")
    parent = FakeConfigOrchestrator(config)
    parent.insert_layer(
        OverridesLayer(data={"thinking_overrides": {"small": "low"}}), 0
    )
    parent.rebuild()

    brief = resolve(
        config, profile, LaunchConfig(model="@brief"), parent_orchestrator=parent
    )
    deep = resolve(
        config, profile, LaunchConfig(model="@deep"), parent_orchestrator=parent
    )
    explicit = resolve(
        config,
        profile,
        LaunchConfig(model="@brief", thinking="high"),
        parent_orchestrator=parent,
    )

    assert (brief.effective_thinking, deep.effective_thinking) == ("off", "high")
    assert (brief.committed_model.thinking, deep.committed_model.thinking) == (
        "off",
        "high",
    )
    assert brief.orchestrator.config.get_active_model().thinking == "off"
    assert deep.orchestrator.config.get_active_model().thinking == "high"
    assert explicit.effective_thinking == "high"
    assert explicit.committed_model.thinking == "high"
    assert explicit.orchestrator.config.get_active_model().thinking == "high"


def test_explicit_thinking_can_repair_unavailable_preset_level_for_spawn(
    config: ChartreuxConfigSchema, profile: AgentProfile
) -> None:
    _with_role(config, "unavailable", ("small",), "max")

    with pytest.raises(InvalidLaunchModelError, match="thinking"):
        resolve(config, profile, LaunchConfig(model="@unavailable"))

    repaired = resolve(
        config, profile, LaunchConfig(model="@unavailable", thinking="high")
    )
    assert repaired.effective_thinking == "high"
    assert repaired.committed_model.thinking == "high"
    assert repaired.orchestrator.config.get_active_model().thinking == "high"


def test_missing_profile_is_typed(config: ChartreuxConfigSchema) -> None:
    with pytest.raises(MissingAgentProfileError, match="agent"):
        resolve_launch(
            profile_name="missing",
            config=None,
            parent_orchestrator=FakeConfigOrchestrator(config),
            tool_inventory={},
            profile_lookup=lambda _: (_ for _ in ()).throw(ValueError()),
        )


def test_bound_role_launch_ignores_later_catalog_edits(config, profile):
    from chartreux.core.dispatch.presets import ORCHESTRATED_PRESET
    from chartreux.core.dispatch.session import bind_policy

    _with_role(config, "bound", ("small",), "off")
    slot = ORCHESTRATED_PRESET.slots["implementor"].model_copy(
        update={"role": "@bound"}
    )
    snapshot = config.catalog_snapshot
    config.attach_catalog_snapshot(
        replace(
            snapshot,
            dispatch=ORCHESTRATED_PRESET.model_copy(
                update={"slots": {"implementor": slot}}
            ),
        )
    )
    parent = FakeConfigOrchestrator(config)
    bound = bind_policy(config)
    parent.bound_dispatch_policy = bound
    _with_role(config, "bound", ("large",), "high")
    child = resolve(
        config, profile, LaunchConfig(model="@bound"), parent_orchestrator=parent
    )
    assert child.committed_model.base_model == "small"
    assert child.effective_thinking == "off"
    assert child.orchestrator.bound_dispatch_policy == bound
    override = resolve(
        config,
        profile,
        LaunchConfig(model="@bound", thinking="high"),
        parent_orchestrator=parent,
    )
    assert override.committed_model.base_model == "small"
    assert override.effective_thinking == "high"
    retained = resolve(
        config,
        profile,
        retained_profile=profile,
        committed_model=child.committed_model,
        parent_orchestrator=child.orchestrator,
    )
    assert retained.committed_model == child.committed_model
    changed_profile = AgentProfile(**{**profile.__dict__, "role": "bound"})
    restored = resolve(
        config,
        changed_profile,
        parent_orchestrator=parent,
        frozen_persona=child.persona,
        committed_model=child.committed_model,
    )
    assert restored.committed_model == child.committed_model
    assert restored.orchestrator.config.get_active_model().thinking == "off"
