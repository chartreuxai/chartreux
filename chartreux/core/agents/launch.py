from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import copy
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from chartreux.core.agents.identities import AgentIdentity, get_builtin_identity
from chartreux.core.agents.registry import build_child_orchestrator
from chartreux.core.config.agent_selection import AgentSelectionConfig
from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
from chartreux.core.config.models import ModelConfig, ThinkingLevel
from chartreux.core.config.orchestrator import ConfigOrchestrator
from chartreux.core.dispatch.route_models import ReplacementCapability
from chartreux.core.llm.backend.anthropic import REASONING_BLOCK_TYPES
from chartreux.core.llm.thinking_levels import get_thinking_levels
from chartreux.core.llm_models import LLMMessage
from chartreux.core.recovery_models import PackageTier
from chartreux.core.session_types import CommittedModelIdentity
from chartreux.core.subagents import (
    ImmutableLaunchPersonaError,
    InvalidLaunchModelError,
    InvalidLaunchThinkingError,
    InvalidLaunchToolError,
    LaunchConfig,
    LaunchToolOverride,
)
from chartreux.core.tools.manager import identity_allows_name
from chartreux.core.tools.models import ToolPermission
from chartreux.core.utils import name_matches

if TYPE_CHECKING:
    from chartreux.core.tools.base import BaseTool


@dataclass(frozen=True, slots=True)
class FrozenPersona:
    """The instruction overlay fixed for a child conversation.

    The persona is keyed on the identity revision (design §7): the identity
    binding and its revision are committed at launch (``LaunchPersonaV2``),
    and the frozen instructions are the additive per-launch overlay (plus the
    matched route's overlay body). The retired ``system_prompt_id`` axis no
    longer participates.
    """

    instructions: str | None


@dataclass(frozen=True, slots=True)
class LaunchAuthorityInputs:
    """Pre-resource inventory snapshot for later authority-ceiling validation."""

    known_tool_names: frozenset[str]


@dataclass(frozen=True, slots=True)
class LaunchCandidate:
    """Validated, resource-free input for child construction or reconfiguration."""

    identity: str
    config_inputs: Mapping[str, Any]
    orchestrator: ConfigOrchestrator[ChartreuxConfigSchema]
    semantic_overrides: LaunchConfig
    persona: FrozenPersona
    effective_model: ModelConfig
    committed_model: CommittedModelIdentity
    effective_thinking: ThinkingLevel
    authority_inputs: LaunchAuthorityInputs
    purpose: str | None = None
    """The purpose label whose route was resolved; None for a direct launch."""
    effective_tier: PackageTier = PackageTier.SUBSTANTIVE
    """The effective package tier resolved for the launch (the route target's
    explicit tier, or the identity/purpose default for a direct launch); the
    recovery boundary reads it through the child loop's launch metadata."""
    replacement_capability: ReplacementCapability = ReplacementCapability.NONE
    """The model-level replacement-capability marking resolved for the launch
    (the route target's explicit marking) — a separate axis from the tier."""


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _deep_merge(base: dict[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _merge_semantic_overrides(
    accumulated: LaunchConfig | None, supplied: LaunchConfig | None
) -> LaunchConfig:
    """Merge only fields supplied by the caller, retaining Pydantic omission state."""
    data = (
        accumulated.model_dump(exclude_unset=True, mode="python")
        if accumulated is not None
        else {}
    )
    if supplied is None:
        return LaunchConfig.model_validate(data)

    for field_name in supplied.model_fields_set:
        value = getattr(supplied, field_name)
        if field_name != "tools":
            data[field_name] = copy.deepcopy(value)
            continue

        tools = copy.deepcopy(data.get("tools", {}))
        for tool_name, patch in (value or {}).items():
            previous = tools.get(tool_name)
            merged = (
                previous.model_dump(exclude_unset=True, mode="python")
                if isinstance(previous, LaunchToolOverride)
                else dict(previous or {})
            )
            merged.update(patch.model_dump(exclude_unset=True, mode="python"))
            tools[tool_name] = merged
        data["tools"] = tools
    return LaunchConfig.model_validate(data)


def _resolved_model(
    config: ChartreuxConfigSchema,
    expression: str,
    committed: CommittedModelIdentity | None = None,
    candidate_filter: Callable[[Any], bool] | None = None,
    thinking: ThinkingLevel | None = None,
) -> tuple[ModelConfig, CommittedModelIdentity]:
    if config.catalog_snapshot is not None:
        from chartreux.core.model_catalog.resolver import (
            ModelResolutionError,
            resolver_for,
        )

        try:
            resolver = resolver_for(config)
            resolved = (
                resolver.resolve_committed(
                    committed, allowed_models=config.allowed_models
                )
                if committed is not None
                else resolver.resolve(
                    expression,
                    allowed_models=config.allowed_models,
                    candidate_filter=candidate_filter,
                    thinking_override=thinking,
                )
            )
            selected_thinking = (
                thinking
                or resolved.thinking
                or (committed.thinking if committed is not None else None)
                or config.thinking_overrides.get(resolved.base_model)
            )
            model = resolved.materialize(
                auto_compact_threshold=config.auto_compact_threshold,
                thinking=selected_thinking,
            )
        except ModelResolutionError as exc:
            raise InvalidLaunchModelError("config.model", str(exc)) from exc
        return model, resolved.identity.model_copy(update={"thinking": model.thinking})
    raise InvalidLaunchModelError(
        "config.model", "No model catalog is attached to this configuration"
    )


def _validate_thinking(
    config: ChartreuxConfigSchema,
    model: ModelConfig,
    thinking: ThinkingLevel,
    *,
    explicit: bool,
) -> None:
    try:
        provider = config.get_provider_for_model(model)
    except ValueError as exc:
        raise InvalidLaunchModelError(
            "config.model", "Configured model provider is unavailable"
        ) from exc

    levels = get_thinking_levels(str(provider.backend), provider.api_style, model.name)
    declared = model.supported_thinking_levels
    if levels is None:
        if declared is not None:
            raise InvalidLaunchThinkingError(
                "config.thinking",
                "Declared thinking levels cannot be encoded by provider",
            )
        if explicit:
            raise InvalidLaunchThinkingError(
                "config.thinking", "Selected provider cannot encode thinking settings"
            )
        return

    if declared is not None and any(level not in levels for level in declared):
        raise InvalidLaunchThinkingError(
            "config.thinking",
            "Declared thinking level is unsupported by the selected model",
        )
    if thinking not in levels:
        message = (
            "Selected model cannot disable thinking"
            if thinking == "off"
            else "Configured thinking level is unsupported by the selected model"
        )
        raise InvalidLaunchThinkingError("config.thinking", message)
    if explicit and declared is not None and thinking not in declared:
        raise InvalidLaunchThinkingError(
            "config.thinking", "Thinking level is unsupported by the selected model"
        )


def _history_has_anthropic_thinking(history: Sequence[LLMMessage]) -> bool:
    return any(
        block.get("type") in REASONING_BLOCK_TYPES
        for message in history
        for block in message.reasoning_payloads or ()
    )


def _validate_history(
    config: ChartreuxConfigSchema, model: ModelConfig, history: Sequence[LLMMessage]
) -> None:
    if any(message.images for message in history) and not model.supports_images:
        raise InvalidLaunchModelError(
            "config.model", "Retained history contains images unsupported by the model"
        )

    provider = config.get_provider_for_model(model)
    style = provider.api_style
    for message in history:
        payloads = message.reasoning_payloads or ()
        if not payloads:
            continue
        if style == "anthropic":
            incompatible = any(
                block.get("type") not in REASONING_BLOCK_TYPES for block in payloads
            )
            # The Anthropic mapper ignores generic reasoning text, so only its
            # native replayable blocks are lossless.
            if incompatible:
                raise InvalidLaunchModelError(
                    "config.model",
                    "Retained history contains incompatible provider reasoning blocks",
                )
        elif style == "openai-responses":
            incompatible = any(block.get("type") != "reasoning" for block in payloads)
            if incompatible and not message.reasoning_content:
                raise InvalidLaunchModelError(
                    "config.model",
                    "Retained history contains non-convertible provider reasoning blocks",
                )
        elif not message.reasoning_content:
            # Generic adapters deliberately discard opaque provider payloads;
            # require the portable text projection before switching.
            raise InvalidLaunchModelError(
                "config.model",
                "Retained history contains non-convertible provider reasoning blocks",
            )


def _effective_thinking(
    config: ChartreuxConfigSchema,
    model: ModelConfig,
    thinking: ThinkingLevel,
    history: Sequence[LLMMessage],
) -> ThinkingLevel:
    provider = config.get_provider_for_model(model)
    if (
        thinking == "off"
        and provider.api_style == "anthropic"
        and _history_has_anthropic_thinking(history)
    ):
        return "medium"
    return thinking


def _validate_tool_patterns(
    patterns: list[str], names: frozenset[str], field: str
) -> None:
    for pattern in patterns:
        if not any(name_matches(name, [pattern]) for name in names):
            raise InvalidLaunchToolError(field, "Tool selection matches no known tool")


def _validate_tools(
    overrides: LaunchConfig,
    names: frozenset[str],
    authorized_names: frozenset[str],
    child_identity: AgentIdentity | None = None,
) -> None:
    """Validate launch tool overrides against two independent ceilings.

    The first ceiling is the parent's delegated authority (``authorized_names``):
    granted authority never widens. The second ceiling is the CHILD identity's
    personal posture: a launch override can never loosen it (v0.5.7 policy
    reversal — overrides used to be able to widen against profile-level
    restrictions; the ToolManager enforces the posture below the config layer,
    and this guard rejects loosening attempts at admission). The identity's
    execution carve-out is part of that ceiling, not a loosening of it: a
    carved-out name is admitted exactly as the manager's runtime check admits
    it. Narrowing overrides (``disabled_tools``, ``permission = "never"``
    patches) stay allowed on both axes.
    """
    if "enabled_tools" in overrides.model_fields_set:
        patterns = overrides.enabled_tools or []
        _validate_tool_patterns(patterns, names, "config.enabled_tools")
        for pattern in patterns:
            for name in names:
                if name_matches(name, [pattern]):
                    if name not in authorized_names:
                        raise InvalidLaunchToolError(
                            "config.enabled_tools",
                            f"Tool '{name}' exceeds parent authority",
                        )
                    if child_identity is not None and not identity_allows_name(
                        child_identity, name
                    ):
                        raise InvalidLaunchToolError(
                            "config.enabled_tools",
                            f"Tool '{name}' exceeds the child identity posture",
                        )
    if "disabled_tools" in overrides.model_fields_set:
        _validate_tool_patterns(
            overrides.disabled_tools or [], names, "config.disabled_tools"
        )
    if "tools" not in overrides.model_fields_set:
        return
    for name, patch in (overrides.tools or {}).items():
        if name not in names:
            raise InvalidLaunchToolError(f"config.tools.{name}", "Unknown tool")
        if name not in authorized_names:
            raise InvalidLaunchToolError(
                f"config.tools.{name}", f"Tool '{name}' exceeds parent authority"
            )
        if (
            child_identity is not None
            and patch.permission is not None
            and patch.permission != ToolPermission.NEVER
            and not identity_allows_name(child_identity, name)
        ):
            raise InvalidLaunchToolError(
                f"config.tools.{name}",
                f"Tool '{name}' exceeds the child identity posture",
            )
        # Re-validate the narrow public schema here so this boundary remains safe
        # when semantic state is reconstructed outside TaskArgs.
        try:
            LaunchToolOverride.model_validate(
                patch.model_dump(exclude_unset=True, mode="python")
            )
        except ValidationError as exc:
            raise InvalidLaunchToolError(
                f"config.tools.{name}", "Invalid tool override"
            ) from exc


def _child_identity(identity: str) -> AgentIdentity | None:
    """The child's resolved built-in identity (None = custom, unrestricted).

    The admission ceiling and the enforced posture derive from the SAME
    resolved identity binding the launch committed, so they always agree. The
    runtime normalizes a ``builtin:`` qualification before its identity lookup
    (``ToolManager._identity_posture``); admission applies the same
    normalization, so an explicitly qualified selection cannot widen the
    admission ceiling while the runtime enforces the builtin posture (S2
    round 2: admission builtin: normalization). The identity carries the
    execution carve-out alongside its posture, so admission and the runtime
    enforce the same ceiling.
    """
    return get_builtin_identity(identity.removeprefix("builtin:"))


def resolve_launch(  # noqa: PLR0913, PLR0914, PLR0915
    *,
    identity: str,
    config: LaunchConfig | None,
    tool_inventory: Mapping[str, type[BaseTool]] | Mapping[str, object],
    parent_orchestrator: ConfigOrchestrator[ChartreuxConfigSchema] | None = None,
    authorized_tool_names: frozenset[str],
    history: Sequence[LLMMessage] = (),
    accumulated_overrides: LaunchConfig | None = None,
    frozen_persona: FrozenPersona | None = None,
    committed_model: CommittedModelIdentity | None = None,
    purpose: str | None = None,
    effective_tier: PackageTier = PackageTier.SUBSTANTIVE,
    replacement_capability: ReplacementCapability = ReplacementCapability.NONE,
) -> LaunchCandidate:
    """Resolve launch semantics without constructing any child-owned resource.

    ``config`` and ``accumulated_overrides`` are copied; neither caller object is
    modified. ``identity`` is the resolved identity reference the launch
    committed (fresh launches: package-8 resolution; retained launches: the
    record's committed binding) — admission validates it and the runtime
    enforces it, so the installed selection and the validated binding can
    never disagree (S2 identity disagreement).
    """
    semantic = _merge_semantic_overrides(accumulated_overrides, config)
    explicit_model = config is not None and "model" in config.model_fields_set
    inputs = semantic.model_dump(exclude_unset=True, mode="python")
    # Persona is loop-owned rather than a mutable configuration-layer concern.
    inputs.pop("instructions", None)
    # WP5 replacement-claim lineage (work_package_id, predecessor_agent_id,
    # package_tier) is loop-owned claim metadata, not a configuration-layer
    # concern: it rides the LaunchConfig (semantic_overrides) into the loop's
    # launch metadata and the agent record — never into the child's
    # config-layer build, whose source validation rejects unknown keys. The
    # demanding route's difficulty category (P1-c2) is the same kind of
    # admission metadata: it rides the run record and the acknowledgment.
    inputs.pop("work_package_id", None)
    inputs.pop("predecessor_agent_id", None)
    inputs.pop("package_tier", None)
    inputs.pop("difficulty", None)
    if "model" in inputs:
        inputs["active_model"] = inputs.pop("model")
    if committed_model is not None and not explicit_model:
        inputs["active_model"] = committed_model.base_model
    # The child's resolved identity selection rides the launch-override layer:
    # the child never inherits the parent's identity selection (the builder
    # strips inherited runtime selections before this layer is installed).
    # The child likewise resolves its OWN workflow/strategy selections (the
    # shipped defaults) instead of inheriting the parent's conversation
    # workflow/strategy (design §7): the bounded-child composition omits the
    # workflow/strategy bodies, so an inherited ceremony selection would make
    # the S8 lint validate the child as ceremony and reject the launch.
    child_selection = AgentSelectionConfig()
    inputs["agent"] = {
        "identity": identity,
        "workflow": child_selection.workflow,
        "orchestration_strategy": child_selection.orchestration_strategy,
    }
    selected_thinking = inputs.pop("thinking", None)

    source = parent_orchestrator
    if source is None:
        raise InvalidLaunchModelError(
            "config", "Authoritative parent configuration assembly is unavailable"
        )
    # The child identity's personal posture is the second admission ceiling
    # for tool overrides, independent of the parent's delegation set.
    child_identity = _child_identity(identity)
    orchestrator = build_child_orchestrator(
        source, inputs, strip_runtime_thinking=explicit_model
    )
    staged_config = orchestrator.config
    if frozen_persona is not None:
        if (
            "instructions" in semantic.model_fields_set
            and semantic.instructions != frozen_persona.instructions
        ):
            raise ImmutableLaunchPersonaError(
                "config.instructions", "Retained persona cannot be changed"
            )
        persona = frozen_persona
    else:
        persona = FrozenPersona(
            instructions=(
                semantic.instructions
                if "instructions" in semantic.model_fields_set
                else None
            )
        )

    orchestrator = build_child_orchestrator(
        source, inputs, strip_runtime_thinking=explicit_model
    )
    staged_config = orchestrator.config

    selected_alias = staged_config.active_model
    if not selected_alias:
        selected_alias = "@main"
    explicit_thinking = "thinking" in semantic.model_fields_set

    def assignment_candidate(candidate: Any) -> bool:
        from chartreux.core.model_catalog.availability import compatibility_exclusion

        if not orchestrator.availability_registry.is_available(
            candidate.base_model, candidate.deployment.provider
        ):
            return False
        candidate_thinking = (
            selected_thinking
            or candidate.thinking
            or (committed_model.thinking if committed_model is not None else None)
            or staged_config.thinking_overrides.get(candidate.base_model)
        )
        candidate_model = candidate.materialize(
            auto_compact_threshold=staged_config.auto_compact_threshold,
            thinking=candidate_thinking,
        )
        candidate_thinking = candidate_model.thinking
        assert candidate_thinking is not None
        return (
            compatibility_exclusion(
                config=staged_config,
                model=candidate_model,
                history=history,
                thinking=candidate_thinking,
                thinking_explicit=explicit_thinking,
            )
            is None
        )

    model, resolved_identity = _resolved_model(
        staged_config,
        selected_alias,
        committed_model if committed_model is not None and not explicit_model else None,
        None
        if committed_model is not None and not explicit_model
        else assignment_candidate,
        selected_thinking,
    )
    # Explicit spawn thinking is a child-only override. Role thinking remains
    # attached to the selected preset and committed identity.
    if selected_thinking is not None:
        inputs["thinking_overrides"] = {model.alias: selected_thinking}
        orchestrator = build_child_orchestrator(
            source, inputs, strip_runtime_thinking=explicit_model
        )
        staged_config = orchestrator.config
    thinking = model.thinking
    assert thinking is not None
    _validate_thinking(
        staged_config,
        model,
        thinking,
        explicit=explicit_thinking or resolved_identity.thinking is not None,
    )
    _validate_history(staged_config, model, history)

    known_tools = frozenset(tool_inventory)
    # The admission ceiling is REQUIRED: it is the parent's delegation set,
    # passed explicitly by every production caller. Defaulting to "all known
    # tools" here would be a latent fail-open (S2 round 2 INFO).
    _validate_tools(semantic, known_tools, authorized_tool_names, child_identity)
    effective_thinking = _effective_thinking(staged_config, model, thinking, history)
    staged_config.attach_committed_model(resolved_identity)
    return LaunchCandidate(
        identity=identity,
        config_inputs=_freeze(inputs),
        orchestrator=orchestrator,
        semantic_overrides=semantic.model_copy(deep=True),
        persona=persona,
        effective_model=model,
        committed_model=resolved_identity,
        effective_thinking=effective_thinking,
        authority_inputs=LaunchAuthorityInputs(known_tool_names=known_tools),
        purpose=purpose,
        effective_tier=effective_tier,
        replacement_capability=replacement_capability,
    )
