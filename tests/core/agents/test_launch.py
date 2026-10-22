from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from chartreux.core.agents.launch import FrozenPersona, resolve_launch
from chartreux.core.agents.manager import resolve_identity_candidate
from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
from chartreux.core.config.harness_files import get_harness_files_manager
from chartreux.core.config.layers.overrides import OverridesLayer
from chartreux.core.config.models import ModelConfig, ProviderConfig, ThinkingLevel
from chartreux.core.dispatch.session import bind_policy_bundle
from chartreux.core.llm_models import Backend, LLMMessage, Role
from chartreux.core.model_catalog.schema import RoleDefinition
from chartreux.core.policy.composition import ExecutionContext
from chartreux.core.session_types import CommittedModelIdentity
from chartreux.core.skills.manager import SkillManager
from chartreux.core.subagents import (
    ImmutableLaunchPersonaError,
    InvalidLaunchModelError,
    InvalidLaunchToolError,
    LaunchConfig,
    LaunchConfigError,
    LaunchToolOverride,
    TaskArgs,
)
from chartreux.core.tools.models import ToolPermission
from tests.conftest import build_test_vibe_config
from tests.stubs.fake_config_orchestrator import FakeConfigOrchestrator


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
    config: ChartreuxConfigSchema, launch: LaunchConfig | None = None, **kwargs: Any
):
    parent_orchestrator = kwargs.pop("parent_orchestrator", None)
    # The delegation ceiling is a required argument (S2 round 2 INFO): the
    # helper's inventory is {"bash", "read_file"}, so that is its ceiling
    # unless a caller passes a narrower one explicitly.
    kwargs.setdefault("authorized_tool_names", frozenset({"bash", "read_file"}))
    return resolve_launch(
        identity="coder",
        config=launch,
        parent_orchestrator=parent_orchestrator or FakeConfigOrchestrator(config),
        tool_inventory={"bash": object(), "read_file": object()},
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


def test_retained_resolution_preserves_committed_model_over_role(
    config: ChartreuxConfigSchema,
) -> None:
    config = _with_role(config, "specialist", ("small",))
    parent = config.model_copy(update={"active_model": "large"})
    initial = resolve(parent, LaunchConfig(model="@specialist"))
    retained = resolve_launch(
        identity="coder",
        config=None,
        parent_orchestrator=initial.orchestrator,
        tool_inventory={"bash": object()},
        authorized_tool_names=frozenset({"bash"}),
        accumulated_overrides=initial.semantic_overrides,
        frozen_persona=initial.persona,
        committed_model=initial.committed_model,
    )
    assert (initial.effective_model.alias, retained.effective_model.alias) == (
        "small",
        "small",
    )


def test_ceremony_parent_child_resolves_its_own_workflow_and_strategy() -> None:
    """A ceremony parent's coder child never inherits the parent's
    workflow/strategy: the child resolves its OWN selections (the shipped
    defaults), and the bounded-child composition + S8 lint validate the child
    on those selections. The bounded-child composition omits the workflow
    body, so an inherited ceremony selection would make the lint reject the
    child (DispatchLintError).
    """
    provider = ProviderConfig(
        name="provider", api_base="https://example.test", backend=Backend.GENERIC
    )
    parent_config = build_test_vibe_config(
        agent={
            "identity": "orchestrator",
            "workflow": "ceremony",
            "orchestration_strategy": "builtin:default",
        },
        providers=[provider],
        models=[
            ModelConfig(
                name="small",
                alias="small",
                provider="provider",
                thinking="off",
                supported_thinking_levels=["off", "low", "high"],
            )
        ],
    )
    candidate = resolve_identity_candidate(
        TaskArgs(task="work", identity="coder"),
        parent_orchestrator=FakeConfigOrchestrator(parent_config),
        tool_inventory={},
        authorized_tool_names=frozenset(),
    )
    child = candidate.orchestrator.config.agent
    assert child.identity == "coder"
    # The child's own selections — the shipped defaults, not the parent's.
    assert child.workflow == "minimal"
    assert child.orchestration_strategy == "none"
    # The bounded-child composition binds on the child's own selections.
    _bundle, activated = bind_policy_bundle(
        candidate.orchestrator.config,
        execution_context=ExecutionContext.BOUNDED_CHILD,
        skill_manager=SkillManager(lambda: candidate.orchestrator.config),
        scratchpad_dir=None,
        cwd=Path.cwd(),
        harness_files=get_harness_files_manager(),
        role_instructions=None,
    )
    assert activated.manifest.workflow.selected == "minimal"


def test_explicit_model_is_exact_and_empty_config_inherits(
    config: ChartreuxConfigSchema,
) -> None:
    assert resolve(config).effective_model.alias == "small"
    assert resolve(config, LaunchConfig(model="large")).effective_model.alias == "large"


@pytest.mark.parametrize(
    ("launch", "error", "field"),
    [
        ({"model": "missing"}, InvalidLaunchModelError, "config.model"),
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
def test_invalid_launches_are_typed(
    config: ChartreuxConfigSchema,
    launch: dict[str, object],
    error: type[LaunchConfigError],
    field: str,
) -> None:
    with pytest.raises(error) as raised:
        resolve_launch(
            identity="coder",
            config=LaunchConfig.model_validate(launch),
            parent_orchestrator=FakeConfigOrchestrator(config),
            tool_inventory={"bash": object()},
            authorized_tool_names=frozenset({"bash"}),
        )
    assert raised.value.field == field


def test_unknown_and_disallowed_catalog_models_are_typed(
    config: ChartreuxConfigSchema,
) -> None:
    with pytest.raises(InvalidLaunchModelError, match="config.model"):
        resolve(config, LaunchConfig(model="missing"))
    with pytest.raises(InvalidLaunchModelError, match="config.model"):
        resolve(
            config.model_copy(update={"allowed_models": ["small"]}),
            LaunchConfig(model="large"),
        )


def test_task_launch_unknown_model_lists_canonical_choices(
    config: ChartreuxConfigSchema,
) -> None:
    with pytest.raises(InvalidLaunchModelError) as raised:
        resolve(config, LaunchConfig(model="luna"))
    assert raised.value.field == "config.model"
    assert "Unknown model expression 'luna'" in str(raised.value)
    assert "Valid canonical models: glm-5-3, large, mistral-large-4, small" in str(
        raised.value
    )

    config = _with_role(config, "specialist", ("small",))
    with pytest.raises(InvalidLaunchModelError) as raised:
        resolve(config, LaunchConfig(model="@luna"))
    assert "@specialist" in str(raised.value)


def test_launch_model_schema_describes_accepted_expression() -> None:
    assert LaunchConfig.model_json_schema()["properties"]["model"]["description"] == (
        "canonical model name or `@role`"
    )


def test_thinking_validation_honors_catalog_declaration(
    config: ChartreuxConfigSchema,
) -> None:
    assert resolve(config, LaunchConfig(thinking="low")).effective_thinking == "low"


def test_launch_tool_override_rejects_removed_ask_permission() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="'always' or 'never'"):
        LaunchToolOverride.model_validate({"permission": "ask"})


def test_accumulated_launch_config_deep_merges_tools_without_mutating_input(
    config: ChartreuxConfigSchema,
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
    config: ChartreuxConfigSchema,
) -> None:
    persona = FrozenPersona("frozen")
    with pytest.raises(ImmutableLaunchPersonaError) as raised:
        resolve(config, LaunchConfig(instructions="secret"), frozen_persona=persona)
    assert "secret" not in str(raised.value)
    candidate = resolve(
        config, LaunchConfig(instructions="frozen"), frozen_persona=persona
    )
    assert candidate.persona == persona


def test_tool_authority_ceiling_rejects_explicit_selection(
    config: ChartreuxConfigSchema,
) -> None:
    with pytest.raises(InvalidLaunchToolError, match="config.enabled_tools"):
        resolve(
            config,
            LaunchConfig(enabled_tools=["bash"]),
            authorized_tool_names=frozenset(),
        )


def _resolve_with_identity(
    config: ChartreuxConfigSchema,
    identity: str,
    launch: LaunchConfig | None,
    authorized: frozenset[str],
):
    return resolve_launch(
        identity=identity,
        config=launch,
        parent_orchestrator=FakeConfigOrchestrator(config),
        tool_inventory={"bash": object(), "read_file": object(), "edit": object()},
        authorized_tool_names=authorized,
    )


def test_child_identity_posture_is_a_second_admission_ceiling(
    config: ChartreuxConfigSchema,
) -> None:
    """The parent's delegation set admits edit (the first ceiling passes);
    the reviewer child's personal posture rejects it (the second ceiling).

    This is the documented v0.5.7 policy reversal at admission: launch-level
    tool overrides can never loosen the child identity posture.
    """
    with pytest.raises(InvalidLaunchToolError, match="identity posture"):
        _resolve_with_identity(
            config,
            "reviewer",
            LaunchConfig(enabled_tools=["edit"]),
            frozenset({"bash", "read_file", "edit"}),
        )


def test_child_identity_posture_rejects_loosening_tool_patch(
    config: ChartreuxConfigSchema,
) -> None:
    with pytest.raises(InvalidLaunchToolError, match="identity posture"):
        _resolve_with_identity(
            config,
            "reviewer",
            LaunchConfig(
                tools={"edit": LaunchToolOverride(permission=ToolPermission.ALWAYS)}
            ),
            frozenset({"bash", "read_file", "edit"}),
        )
    # Narrowing stays allowed: a never-patch on a posture-excluded tool is
    # not a loosening attempt, so admission accepts it.
    candidate = _resolve_with_identity(
        config,
        "reviewer",
        LaunchConfig(
            tools={"edit": LaunchToolOverride(permission=ToolPermission.NEVER)}
        ),
        frozenset({"bash", "read_file", "edit"}),
    )
    assert candidate.identity == "reviewer"


def test_coder_child_tool_overrides_pass_both_ceilings(
    config: ChartreuxConfigSchema,
) -> None:
    candidate = _resolve_with_identity(
        config,
        "coder",
        LaunchConfig(enabled_tools=["edit", "read_file"]),
        frozenset({"bash", "read_file", "edit"}),
    )
    assert candidate.identity == "coder"


def test_launch_installs_the_resolved_identity_binding(
    config: ChartreuxConfigSchema,
) -> None:
    for identity in ("coder", "advisor", "reviewer", "user:wp3-agent"):
        candidate = _resolve_with_identity(config, identity, None, frozenset())
        assert candidate.orchestrator.config.agent.identity == identity
        assert candidate.identity == identity


def test_inherited_runtime_identity_cannot_disagree_with_the_child_binding(
    config: ChartreuxConfigSchema,
) -> None:
    """A parent runtime identity override never re-postures a launched child.

    The child's identity is bound at launch resolution and admission
    validates that binding. An inherited runtime override sitting above the
    launch-override layer would otherwise make a reviewer child execute as
    the parent's coder identity while admission validated reviewer
    (S2 identity disagreement): children never inherit the parent's identity
    selection.
    """
    parent = FakeConfigOrchestrator(config)
    parent.insert_layer(OverridesLayer(data={"agent": {"identity": "coder"}}), 0)
    parent.rebuild()
    assert parent.config.agent.identity == "coder"  # the parent runs as coder

    candidate = resolve_launch(
        identity="reviewer",
        config=None,
        parent_orchestrator=parent,
        tool_inventory={"bash": object(), "read_file": object(), "edit": object()},
        authorized_tool_names=frozenset({"bash", "read_file", "edit"}),
    )
    # The runtime enforces the validated binding: reviewer, not the parent's
    # inherited coder selection.
    assert candidate.orchestrator.config.agent.identity == "reviewer"
    # Admission and runtime derive from the same binding: the reviewer
    # posture rejects a write tool the coder identity would have admitted.
    with pytest.raises(InvalidLaunchToolError, match="identity posture"):
        resolve_launch(
            identity="reviewer",
            config=LaunchConfig(enabled_tools=["edit"]),
            parent_orchestrator=parent,
            tool_inventory={"bash": object(), "read_file": object(), "edit": object()},
            authorized_tool_names=frozenset({"bash", "read_file", "edit"}),
        )


def test_explicit_identity_binds_admission_and_runtime_together(
    config: ChartreuxConfigSchema,
) -> None:
    """The resolved identity reference is the single binding.

    Admission and the runtime posture BOTH derive from the same resolved
    binding: a launch that resolves the reviewer identity is validated and
    enforced as reviewer, so the admission posture and the enforced runtime
    posture can never disagree (S2 identity disagreement, peer-seat case).
    """
    candidate = _resolve_with_identity(
        config, "reviewer", None, frozenset({"bash", "read_file", "edit"})
    )
    # The resolved reference is the binding at runtime...
    assert candidate.orchestrator.config.agent.identity == "reviewer"
    # ...and admission validates the same binding: the reviewer posture
    # rejects a write tool.
    with pytest.raises(InvalidLaunchToolError, match="identity posture"):
        _resolve_with_identity(
            config,
            "reviewer",
            LaunchConfig(enabled_tools=["edit"]),
            frozenset({"bash", "read_file", "edit"}),
        )


def test_builtin_qualified_identity_normalizes_the_admission_posture(
    config: ChartreuxConfigSchema,
) -> None:
    """An explicit ``builtin:``-qualified selection gets the builtin posture.

    The runtime identity lookup normalizes the ``builtin:`` prefix
    (``ToolManager._identity_posture``); admission applies the same
    normalization in ``_child_personal_posture``. Without it, a qualified
    selection of ``builtin:reviewer`` would get an UNRESTRICTED admission
    posture while the runtime enforced reviewer — the second admission
    ceiling would be void for qualified selections (S2 round 2 warning:
    admission builtin: normalization).
    """
    candidate = _resolve_with_identity(
        config, "builtin:reviewer", None, frozenset({"bash", "read_file", "edit"})
    )
    # The qualified selection is the resolved binding at runtime...
    assert candidate.orchestrator.config.agent.identity == "builtin:reviewer"
    # ...and admission derives the SAME reviewer posture from it: the
    # non-mutating ceiling rejects a write tool exactly as it does for the
    # bare reviewer selection.
    with pytest.raises(InvalidLaunchToolError, match="identity posture"):
        _resolve_with_identity(
            config,
            "builtin:reviewer",
            LaunchConfig(enabled_tools=["edit"]),
            frozenset({"bash", "read_file", "edit"}),
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
    assert resolve(adapted, history=history).effective_model.alias == "small"


def test_anthropic_history_normalizes_omitted_launch_thinking(
    config: ChartreuxConfigSchema,
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
            accumulated_overrides=LaunchConfig(thinking="off"),
            history=history,
        ).effective_thinking
        == "medium"
    )


def test_child_launch_inherits_parent_committed_deployment(
    config: ChartreuxConfigSchema,
) -> None:
    parent_identity = CommittedModelIdentity(
        base_model="large",
        provider="provider",
        wire_name="large",
        catalog_revision="test-fixture",
    )

    candidate = resolve(config, committed_model=parent_identity)

    assert candidate.committed_model.base_model == parent_identity.base_model
    assert candidate.committed_model.provider == parent_identity.provider
    assert candidate.committed_model.thinking == "high"
    assert candidate.effective_model.name == "large"


def test_new_child_role_beats_parent_committed_identity(
    config: ChartreuxConfigSchema,
) -> None:
    config = _with_role(config, "specialist", ("small",))
    parent_identity = CommittedModelIdentity(
        base_model="large",
        provider="provider",
        wire_name="large",
        catalog_revision="test-fixture",
    )

    candidate = resolve_launch(
        identity="coder",
        config=LaunchConfig(model="@specialist"),
        parent_orchestrator=FakeConfigOrchestrator(config),
        tool_inventory={"bash": object()},
        authorized_tool_names=frozenset({"bash"}),
        committed_model=parent_identity,
    )

    assert candidate.orchestrator.config.active_model == "@specialist"
    assert candidate.effective_model.alias == "small"
    assert candidate.committed_model.base_model == "small"


def test_explicit_model_beats_parent_committed_identity(
    config: ChartreuxConfigSchema,
) -> None:
    config = _with_role(config, "small", ("small",))
    parent_identity = CommittedModelIdentity(
        base_model="large",
        provider="provider",
        wire_name="large",
        catalog_revision="test-fixture",
    )

    candidate = resolve_launch(
        identity="coder",
        config=LaunchConfig(model="@small"),
        parent_orchestrator=FakeConfigOrchestrator(config),
        tool_inventory={"bash": object()},
        authorized_tool_names=frozenset({"bash"}),
        committed_model=parent_identity,
    )

    assert candidate.orchestrator.config.active_model == "@small"
    assert candidate.effective_model.alias == "small"
    assert candidate.committed_model.base_model == "small"


def test_role_launch_thinking_uses_resolved_base_override(
    config: ChartreuxConfigSchema,
) -> None:
    config = _with_role(config, "panel", ("large",))

    candidate = resolve(config, LaunchConfig(model="@panel", thinking="high"))

    assert candidate.effective_model.alias == "large"
    assert candidate.effective_thinking == "high"


def test_shared_model_roles_keep_distinct_thinking_and_explicit_override(
    config: ChartreuxConfigSchema,
) -> None:
    _with_role(config, "brief", ("small",), "off")
    _with_role(config, "deep", ("small",), "high")
    parent = FakeConfigOrchestrator(config)
    parent.insert_layer(
        OverridesLayer(data={"thinking_overrides": {"small": "low"}}), 0
    )
    parent.rebuild()

    brief = resolve(config, LaunchConfig(model="@brief"), parent_orchestrator=parent)
    deep = resolve(config, LaunchConfig(model="@deep"), parent_orchestrator=parent)
    explicit = resolve(
        config,
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
    config: ChartreuxConfigSchema,
) -> None:
    _with_role(config, "unavailable", ("small",), "max")

    with pytest.raises(InvalidLaunchModelError, match="thinking"):
        resolve(config, LaunchConfig(model="@unavailable"))

    repaired = resolve(config, LaunchConfig(model="@unavailable", thinking="high"))
    assert repaired.effective_thinking == "high"
    assert repaired.committed_model.thinking == "high"
    assert repaired.orchestrator.config.get_active_model().thinking == "high"


def test_committed_model_launch_ignores_later_catalog_edits(
    config: ChartreuxConfigSchema,
) -> None:
    # The dispatch roster freeze is retired with the session-policy bind
    # (package 16b); the launch-level freeze is the committed model, captured
    # at launch. A later catalog edit rebinds the role for NEW launches but
    # never rewrites a committed one.
    _with_role(config, "bound", ("small",), "off")
    parent = FakeConfigOrchestrator(config)
    child = resolve(config, LaunchConfig(model="@bound"), parent_orchestrator=parent)
    assert child.committed_model.base_model == "small"
    assert child.effective_thinking == "off"
    _with_role(config, "bound", ("large",), "high")
    # A fresh launch resolves the current role, with an explicit thinking
    # override on top of it.
    override = resolve(
        config,
        LaunchConfig(model="@bound", thinking="high"),
        parent_orchestrator=parent,
    )
    assert override.committed_model.base_model == "large"
    assert override.effective_thinking == "high"
    # Retained and restored resolutions keep the committed model.
    retained = resolve(
        config,
        committed_model=child.committed_model,
        parent_orchestrator=child.orchestrator,
    )
    assert retained.committed_model == child.committed_model
    restored = resolve(
        config,
        parent_orchestrator=parent,
        frozen_persona=child.persona,
        committed_model=child.committed_model,
    )
    assert restored.committed_model == child.committed_model
    assert restored.orchestrator.config.get_active_model().thinking == "off"
