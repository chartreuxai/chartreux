"""Shared deployment availability and committed-base eligibility resolution."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
import time

from chartreux.core.config.models import ModelConfig, ThinkingLevel
from chartreux.core.llm.backend.anthropic import REASONING_BLOCK_TYPES
from chartreux.core.llm.failures import FailureInfo
from chartreux.core.llm.thinking_levels import get_thinking_levels
from chartreux.core.llm_models import LLMMessage
from chartreux.core.model_catalog.loader import CatalogSnapshot
from chartreux.core.model_catalog.resolver import ModelResolver, ResolvedModel
from chartreux.core.session_types import CommittedModelIdentity


class ExclusionReason(StrEnum):
    COOLDOWN = "cooldown"
    IMAGES_UNSUPPORTED = "images-unsupported"
    THINKING_UNSUPPORTED = "thinking-unsupported"
    REASONING_UNSUPPORTED = "reasoning-unsupported"
    COMPACTION_INCOMPATIBLE = "compaction-incompatible"
    DISABLED = "disabled"
    ALLOWLIST_EXCLUDED = "allowlist-excluded"


@dataclass(frozen=True, slots=True)
class DeploymentExclusion:
    base_model: str
    provider: str
    wire_name: str
    reason: ExclusionReason
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class EligibleDeployment:
    resolved: ResolvedModel
    recovery_probe: bool = False
    recovery_probe_token: int | None = None


@dataclass(frozen=True, slots=True)
class EligibilityResult:
    candidates: tuple[EligibleDeployment, ...]
    exclusions: tuple[DeploymentExclusion, ...]


class AllDeploymentsUnavailableError(ValueError):
    """No deployment of the committed base can serve the current completion."""

    code = "all_deployments_unavailable"

    def __init__(
        self, base_model: str, exclusions: Sequence[DeploymentExclusion]
    ) -> None:
        super().__init__(
            f"All deployments for base model {base_model!r} are unavailable"
        )
        self.base_model = base_model
        self.exclusions = tuple(exclusions)


@dataclass(slots=True)
class _Cooldown:
    until: float
    probe_in_flight: bool = False
    probe_token: int | None = None


class AvailabilityRegistry:
    """Root-owned mutable cooldown state shared by every runtime child."""

    def __init__(
        self,
        *,
        initial_cooldown: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if initial_cooldown < 0:
            raise ValueError("initial_cooldown must be non-negative")
        self.initial_cooldown = initial_cooldown
        self._clock = clock
        self._cooldowns: dict[tuple[str, str], _Cooldown] = {}
        self._next_probe_token = 0

    def record_failure(
        self, base_model: str, provider: str, failure: FailureInfo | None = None
    ) -> None:
        """Start or extend cooldown and release an outstanding recovery probe."""
        delay = max(
            self.initial_cooldown,
            failure.retry_after if failure is not None and failure.retry_after else 0.0,
        )
        key = (base_model, provider)
        until = self._clock() + delay
        current = self._cooldowns.get(key)
        self._cooldowns[key] = _Cooldown(
            until=max(until, current.until if current is not None else until)
        )

    def record_success(self, base_model: str, provider: str) -> None:
        """Clear cooldown after a successful normal attempt or recovery probe."""
        self._cooldowns.pop((base_model, provider), None)

    def reset_base(self, base_model: str) -> None:
        """Clear every deployment cooldown for a user-requested base recovery."""
        for key in [key for key in self._cooldowns if key[0] == base_model]:
            del self._cooldowns[key]

    def is_available(self, base_model: str, provider: str) -> bool:
        """Check assignment-time availability without claiming a recovery probe."""
        state = self._cooldowns.get((base_model, provider))
        return state is None or (
            self._clock() >= state.until and not state.probe_in_flight
        )

    def admission(
        self, base_model: str, provider: str
    ) -> tuple[bool, bool, int | None]:
        """Return admission, probe status, and an owner token for expired probes."""
        state = self._cooldowns.get((base_model, provider))
        if state is None:
            return True, False, None
        if self._clock() < state.until or state.probe_in_flight:
            return False, False, None
        self._next_probe_token += 1
        state.probe_in_flight = True
        state.probe_token = self._next_probe_token
        return True, True, state.probe_token

    def release_probe(self, base_model: str, provider: str, token: int) -> None:
        """Release an unattempted probe only when its owner token still matches."""
        state = self._cooldowns.get((base_model, provider))
        if state is not None and state.probe_token == token:
            state.probe_in_flight = False
            state.probe_token = None

    def cooldown_until(self, base_model: str, provider: str) -> float | None:
        state = self._cooldowns.get((base_model, provider))
        return state.until if state is not None else None


def compatibility_exclusion(
    *,
    config: object,
    model: ModelConfig,
    history: Sequence[LLMMessage],
    thinking: ThinkingLevel,
    thinking_explicit: bool,
    images_only: bool = False,
) -> tuple[ExclusionReason, str] | None:
    """Return why a concrete deployment cannot serve the exact request."""
    if any(message.images for message in history) and not model.supports_images:
        return (
            ExclusionReason.IMAGES_UNSUPPORTED,
            "Retained history contains images unsupported by the model",
        )
    provider = config.get_provider_for_model(model)  # type: ignore[attr-defined]
    style = provider.api_style
    for message in () if images_only else history:
        payloads = message.reasoning_payloads or ()
        if not payloads:
            continue
        if style == "anthropic":
            if any(
                block.get("type") not in REASONING_BLOCK_TYPES for block in payloads
            ):
                return (
                    ExclusionReason.REASONING_UNSUPPORTED,
                    "Retained history contains incompatible provider reasoning blocks",
                )
        elif style == "openai-responses":
            if (
                any(block.get("type") != "reasoning" for block in payloads)
                and not message.reasoning_content
            ):
                return (
                    ExclusionReason.REASONING_UNSUPPORTED,
                    "Retained history contains non-convertible provider reasoning blocks",
                )
        elif not message.reasoning_content:
            return (
                ExclusionReason.REASONING_UNSUPPORTED,
                "Retained history contains non-convertible provider reasoning blocks",
            )

    levels = get_thinking_levels(str(provider.backend), provider.api_style, model.name)
    declared = model.supported_thinking_levels
    incompatible = not images_only and (
        (levels is None and (declared is not None or thinking_explicit))
        or (
            levels is not None
            and declared is not None
            and any(level not in levels for level in declared)
        )
        or (levels is not None and thinking not in levels)
        or (declared is not None and thinking not in declared)
    )
    if incompatible:
        return (
            ExclusionReason.THINKING_UNSUPPORTED,
            "Configured thinking level is unsupported by the selected model",
        )
    return None


def eligible_deployments(  # noqa: PLR0912, PLR0914
    *,
    snapshot: CatalogSnapshot,
    committed: CommittedModelIdentity,
    registry: AvailabilityRegistry,
    config: object,
    history: Sequence[LLMMessage] = (),
    thinking: ThinkingLevel,
    thinking_explicit: bool = True,
    compaction_base: str | None = None,
    compaction_only: bool = False,
    allowed_models: Sequence[str] = (),
) -> EligibilityResult:
    """Resolve ordered candidates for exactly the committed canonical base."""
    from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema

    if not isinstance(config, ChartreuxConfigSchema):
        raise TypeError("config must be a ChartreuxConfigSchema")
    definition = snapshot.catalog.models.get(committed.base_model)
    if definition is None or definition.disabled:
        raise AllDeploymentsUnavailableError(committed.base_model, ())

    resolver = ModelResolver(snapshot)
    candidates: list[EligibleDeployment] = []
    exclusions: list[DeploymentExclusion] = []
    canonical_compaction_base = (
        resolver.canonicalize(compaction_base) if compaction_base else None
    )
    compaction_definition = (
        snapshot.catalog.models.get(canonical_compaction_base)
        if canonical_compaction_base
        else None
    )

    for deployment in definition.deployments:
        provider = snapshot.catalog.providers.get(deployment.provider)
        exclusion: DeploymentExclusion | None = None
        if deployment.disabled or provider is None or provider.disabled:
            exclusion = DeploymentExclusion(
                committed.base_model,
                deployment.provider,
                deployment.name,
                ExclusionReason.DISABLED,
            )
        elif not resolver._allowed(
            committed.base_model, definition, deployment, allowed_models
        ):
            exclusion = DeploymentExclusion(
                committed.base_model,
                deployment.provider,
                deployment.name,
                ExclusionReason.ALLOWLIST_EXCLUDED,
            )
        elif compaction_base and (
            compaction_definition is None
            or compaction_definition.disabled
            or not any(
                item.provider == deployment.provider
                and not item.disabled
                and not snapshot.catalog.providers[item.provider].disabled
                for item in compaction_definition.deployments
            )
        ):
            exclusion = DeploymentExclusion(
                committed.base_model,
                deployment.provider,
                deployment.name,
                ExclusionReason.COMPACTION_INCOMPATIBLE,
            )
        else:
            admitted, probe, probe_token = registry.admission(
                committed.base_model, deployment.provider
            )
            if not admitted:
                exclusion = DeploymentExclusion(
                    committed.base_model,
                    deployment.provider,
                    deployment.name,
                    ExclusionReason.COOLDOWN,
                )
            else:
                resolved = ResolvedModel(
                    committed.base_model,
                    definition,
                    deployment,
                    provider,
                    snapshot.revision,
                )
                model = resolved.materialize(
                    auto_compact_threshold=config.auto_compact_threshold,
                    thinking=thinking,
                )
                problem = compatibility_exclusion(
                    config=config,
                    model=model,
                    history=history,
                    thinking=thinking,
                    thinking_explicit=thinking_explicit,
                    images_only=compaction_only,
                )
                if problem is not None:
                    reason, detail = problem
                    exclusion = DeploymentExclusion(
                        committed.base_model,
                        deployment.provider,
                        deployment.name,
                        reason,
                        detail,
                    )
                elif compaction_definition is not None:
                    compaction_deployment = next(
                        item
                        for item in compaction_definition.deployments
                        if item.provider == deployment.provider and not item.disabled
                    )
                    compaction_resolved = ResolvedModel(
                        canonical_compaction_base or "",
                        compaction_definition,
                        compaction_deployment,
                        snapshot.catalog.providers[compaction_deployment.provider],
                        snapshot.revision,
                    )
                    compaction_thinking = (
                        thinking
                        if compaction_only
                        else config.thinking_overrides.get(
                            canonical_compaction_base or "",
                            compaction_definition.thinking,
                        )
                    )
                    compaction_model = compaction_resolved.materialize(
                        auto_compact_threshold=config.auto_compact_threshold,
                        thinking=compaction_thinking,
                    )
                    compaction_problem = compatibility_exclusion(
                        config=config,
                        model=compaction_model,
                        history=history,
                        thinking=compaction_model.thinking or "off",
                        thinking_explicit=False,
                    )
                    if compaction_problem is not None:
                        _, detail = compaction_problem
                        exclusion = DeploymentExclusion(
                            committed.base_model,
                            deployment.provider,
                            deployment.name,
                            ExclusionReason.COMPACTION_INCOMPATIBLE,
                            detail,
                        )
                if exclusion is None:
                    candidates.append(EligibleDeployment(resolved, probe, probe_token))
                elif probe:
                    assert probe_token is not None
                    registry.release_probe(
                        committed.base_model, deployment.provider, probe_token
                    )
        if exclusion is not None:
            exclusions.append(exclusion)

    if not candidates:
        raise AllDeploymentsUnavailableError(committed.base_model, exclusions)
    return EligibilityResult(tuple(candidates), tuple(exclusions))
