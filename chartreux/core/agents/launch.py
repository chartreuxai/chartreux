from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import copy
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from chartreux.core.agents.models import AgentProfile
from chartreux.core.agents.registry import build_child_orchestrator
from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
from chartreux.core.config.models import ModelConfig, ThinkingLevel
from chartreux.core.config.orchestrator import ConfigOrchestrator
from chartreux.core.llm.backend.anthropic import REASONING_BLOCK_TYPES
from chartreux.core.llm.thinking_levels import get_thinking_levels
from chartreux.core.llm_models import LLMMessage
from chartreux.core.prompts import load_system_prompt
from chartreux.core.session_types import CommittedModelIdentity
from chartreux.core.subagents import (
    ImmutableLaunchPersonaError,
    InvalidLaunchModelError,
    InvalidLaunchPromptError,
    InvalidLaunchThinkingError,
    InvalidLaunchToolError,
    LaunchConfig,
    LaunchToolOverride,
    MissingAgentProfileError,
)
from chartreux.core.utils import name_matches

if TYPE_CHECKING:
    from chartreux.core.agents.manager import AgentManager
    from chartreux.core.tools.base import BaseTool


@dataclass(frozen=True, slots=True)
class FrozenPersona:
    """The prompt identity and role instructions fixed for a child conversation."""

    system_prompt_id: str
    instructions: str | None


@dataclass(frozen=True, slots=True)
class LaunchAuthorityInputs:
    """Pre-resource inventory snapshot for later authority-ceiling validation."""

    known_tool_names: frozenset[str]


@dataclass(frozen=True, slots=True)
class LaunchCandidate:
    """Validated, resource-free input for child construction or reconfiguration."""

    profile: AgentProfile
    config_inputs: Mapping[str, Any]
    orchestrator: ConfigOrchestrator[ChartreuxConfigSchema]
    semantic_overrides: LaunchConfig
    persona: FrozenPersona
    effective_model: ModelConfig
    committed_model: CommittedModelIdentity
    effective_thinking: ThinkingLevel
    authority_inputs: LaunchAuthorityInputs


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
                )
            )
        except ModelResolutionError as exc:
            raise InvalidLaunchModelError("config.model", str(exc)) from exc
        return (
            resolved.materialize(
                auto_compact_threshold=config.auto_compact_threshold,
                thinking=config.thinking_overrides.get(resolved.base_model),
            ),
            resolved.identity,
        )
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
    overrides: LaunchConfig, names: frozenset[str], authorized_names: frozenset[str]
) -> None:
    if "enabled_tools" in overrides.model_fields_set:
        patterns = overrides.enabled_tools or []
        _validate_tool_patterns(patterns, names, "config.enabled_tools")
        for pattern in patterns:
            for name in names:
                if name_matches(name, [pattern]) and name not in authorized_names:
                    raise InvalidLaunchToolError(
                        "config.enabled_tools",
                        f"Tool '{name}' exceeds parent authority",
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


def _resolve_profile(
    profile_name: str | None,
    retained_profile: AgentProfile | None,
    agent_manager: AgentManager | None,
    profile_lookup: Callable[[str], AgentProfile] | None,
) -> AgentProfile:
    if retained_profile is not None:
        if profile_name is None:
            return retained_profile
        if profile_name != retained_profile.name:
            raise MissingAgentProfileError(
                "agent",
                f"Requested profile '{profile_name}' differs from retained agent",
            )
        return retained_profile
    if profile_name is None:
        profile_name = "worker"
    try:
        if profile_lookup is not None:
            return profile_lookup(profile_name)
        if agent_manager is None:
            raise ValueError
        return agent_manager.get_agent(profile_name)
    except ValueError as exc:
        raise MissingAgentProfileError(
            "agent", f"Selected agent profile '{profile_name}' is unavailable"
        ) from exc


def resolve_launch(  # noqa: PLR0913, PLR0914, PLR0915
    *,
    profile_name: str | None,
    config: LaunchConfig | None,
    tool_inventory: Mapping[str, type[BaseTool]] | Mapping[str, object],
    parent_orchestrator: ConfigOrchestrator[ChartreuxConfigSchema] | None = None,
    authorized_tool_names: frozenset[str] | None = None,
    history: Sequence[LLMMessage] = (),
    agent_manager: AgentManager | None = None,
    profile_lookup: Callable[[str], AgentProfile] | None = None,
    retained_profile: AgentProfile | None = None,
    accumulated_overrides: LaunchConfig | None = None,
    frozen_persona: FrozenPersona | None = None,
    committed_model: CommittedModelIdentity | None = None,
) -> LaunchCandidate:
    """Resolve launch semantics without constructing any child-owned resource.

    ``config`` and ``accumulated_overrides`` are copied; neither caller object is
    modified.  A retained call is selected by supplying ``retained_profile``.
    """
    profile = _resolve_profile(
        profile_name, retained_profile, agent_manager, profile_lookup
    )
    semantic = _merge_semantic_overrides(accumulated_overrides, config)
    explicit_model = config is not None and "model" in config.model_fields_set
    explicit_profile = retained_profile is None and profile.role is not None
    inputs = semantic.model_dump(exclude_unset=True, mode="python")
    # Persona is loop-owned rather than a mutable configuration-layer concern.
    inputs.pop("instructions", None)
    if "model" in inputs:
        inputs["active_model"] = inputs.pop("model")
    if committed_model is not None and not explicit_model and not explicit_profile:
        inputs["active_model"] = committed_model.base_model
    selected_thinking = inputs.pop("thinking", None)

    source = parent_orchestrator
    if source is None:
        raise InvalidLaunchModelError(
            "config", "Authoritative parent configuration assembly is unavailable"
        )
    profile_overrides = None if retained_profile is not None else profile.overrides
    profile_role = None if retained_profile is not None else profile.role
    orchestrator = build_child_orchestrator(
        source, profile_overrides, inputs, profile_role=profile_role
    )
    staged_config = orchestrator.config
    profile_prompt = staged_config.system_prompt_id
    if frozen_persona is not None:
        if (
            "instructions" in semantic.model_fields_set
            and semantic.instructions != frozen_persona.instructions
        ):
            raise ImmutableLaunchPersonaError(
                "config.instructions", "Retained persona cannot be changed"
            )
        if (
            "system_prompt_id" in semantic.model_fields_set
            and semantic.system_prompt_id != frozen_persona.system_prompt_id
        ):
            raise ImmutableLaunchPersonaError(
                "config.system_prompt_id", "Retained persona cannot be changed"
            )
        persona = frozen_persona
    else:
        persona = FrozenPersona(
            system_prompt_id=(
                semantic.system_prompt_id
                if (
                    "system_prompt_id" in semantic.model_fields_set
                    and semantic.system_prompt_id is not None
                )
                else profile_prompt
            ),
            instructions=(
                semantic.instructions
                if "instructions" in semantic.model_fields_set
                else profile.instructions
            ),
        )

    try:
        load_system_prompt(persona.system_prompt_id)
    except (OSError, ValueError) as exc:
        raise InvalidLaunchPromptError(
            "config.system_prompt_id", "System prompt is unavailable"
        ) from exc

    # The frozen persona, rather than changed parent/profile defaults, is the
    # prompt identity consumed by both the retained loop and staged renderer.
    inputs["system_prompt_id"] = persona.system_prompt_id
    orchestrator = build_child_orchestrator(
        source, profile_overrides, inputs, profile_role=profile_role
    )
    staged_config = orchestrator.config

    selected_alias = staged_config.active_model
    if not selected_alias:
        selected_alias = staged_config.resolve_default_model_alias()
    explicit_thinking = "thinking" in semantic.model_fields_set

    def assignment_candidate(candidate: Any) -> bool:
        from chartreux.core.model_catalog.availability import compatibility_exclusion

        if not orchestrator.availability_registry.is_available(
            candidate.base_model, candidate.deployment.provider
        ):
            return False
        candidate_model = candidate.materialize(
            auto_compact_threshold=staged_config.auto_compact_threshold,
            thinking=(
                selected_thinking
                if selected_thinking is not None
                else staged_config.thinking_overrides.get(candidate.base_model)
            ),
        )
        candidate_thinking = (
            selected_thinking
            if selected_thinking is not None
            else staged_config.thinking_overrides.get(
                candidate.base_model, candidate_model.thinking
            )
        )
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

    model, identity = _resolved_model(
        staged_config,
        selected_alias,
        committed_model
        if committed_model is not None and not explicit_model and not explicit_profile
        else None,
        None
        if committed_model is not None and not explicit_model and not explicit_profile
        else assignment_candidate,
    )
    if selected_thinking is not None:
        # Roles resolve at assignment; overrides are keyed by their resolved base.
        inputs["thinking_overrides"] = {model.alias: selected_thinking}
        orchestrator = build_child_orchestrator(
            source, profile_overrides, inputs, profile_role=profile_role
        )
        staged_config = orchestrator.config
        model, identity = _resolved_model(
            staged_config,
            selected_alias,
            committed_model
            if committed_model is not None
            and not explicit_model
            and not explicit_profile
            else None,
            None
            if committed_model is not None
            and not explicit_model
            and not explicit_profile
            else assignment_candidate,
        )
    thinking = staged_config.thinking_overrides.get(model.alias, model.thinking)
    assert thinking is not None
    _validate_thinking(staged_config, model, thinking, explicit=explicit_thinking)
    _validate_history(staged_config, model, history)

    known_tools = frozenset(tool_inventory)
    authorized_tools = (
        known_tools if authorized_tool_names is None else authorized_tool_names
    )
    _validate_tools(semantic, known_tools, authorized_tools)
    effective_thinking = _effective_thinking(staged_config, model, thinking, history)
    return LaunchCandidate(
        profile=profile,
        config_inputs=_freeze(inputs),
        orchestrator=orchestrator,
        semantic_overrides=semantic.model_copy(deep=True),
        persona=persona,
        effective_model=model,
        committed_model=identity,
        effective_thinking=effective_thinking,
        authority_inputs=LaunchAuthorityInputs(known_tool_names=known_tools),
    )
