"""Canonical model-expression resolution for sessions and child launches."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from chartreux.core.config.models import ModelConfig, ThinkingLevel
from chartreux.core.llm.thinking_levels import get_thinking_levels
from chartreux.core.model_catalog.loader import CatalogSnapshot
from chartreux.core.model_catalog.schema import (
    BaseModelDefinition,
    DeploymentDefinition,
    ProviderDefinition,
)
from chartreux.core.session_types import CommittedModelIdentity
from chartreux.core.utils.matching import name_matches

if TYPE_CHECKING:
    from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema


class ModelResolutionError(ValueError):
    """A selection error that callers can present without guessing a fallback."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _supports_thinking(
    provider: ProviderDefinition, deployment: DeploymentDefinition, thinking: str
) -> bool:
    """Check declared deployment limits and known backend wire capabilities."""
    declared = deployment.supported_thinking_levels
    if declared is not None and thinking not in declared:
        return False
    levels = get_thinking_levels(provider.backend, provider.api_style, deployment.name)
    return levels is None or thinking in levels


@dataclass(frozen=True)
class ResolvedModel:
    """A canonical base and concrete catalog deployment."""

    base_model: str
    definition: BaseModelDefinition
    deployment: DeploymentDefinition
    provider: ProviderDefinition
    catalog_revision: str
    thinking: str | None = None

    @property
    def identity(self) -> CommittedModelIdentity:
        return CommittedModelIdentity(
            base_model=self.base_model,
            provider=self.deployment.provider,
            wire_name=self.deployment.name,
            catalog_revision=self.catalog_revision,
            thinking=self.thinking,
        )

    def materialize(
        self,
        *,
        auto_compact_threshold: int,
        thinking: str | None = None,
        validate_thinking: bool = True,
    ) -> ModelConfig:
        """Produce the legacy backend input, always retaining the wire name."""
        selected_thinking = (
            thinking
            if thinking is not None
            else self.thinking or self.definition.thinking
        )
        if validate_thinking and not _supports_thinking(
            self.provider, self.deployment, selected_thinking
        ):
            raise ModelResolutionError(
                "thinking_unsupported",
                f"Thinking level {selected_thinking!r} is unsupported by "
                f"{self.deployment.provider}/{self.deployment.name}; edit the preset thinking level",
            )
        prices = self.deployment.prices
        return ModelConfig(
            name=self.deployment.name,
            provider=self.deployment.provider,
            alias=self.base_model,
            display_name=f"{self.deployment.provider}/{self.deployment.name}",
            temperature=(
                self.definition.temperature
                if self.definition.temperature is not None
                else 0.2
            ),
            input_price=prices.input if prices.input is not None else 0.0,
            output_price=prices.output if prices.output is not None else 0.0,
            cached_input_price=prices.cached_input,
            input_price_known=prices.input is not None,
            output_price_known=prices.output is not None,
            cached_input_price_known=prices.cached_input is not None,
            thinking=cast(ThinkingLevel, selected_thinking),
            supported_thinking_levels=cast(
                list[ThinkingLevel] | None, self.deployment.supported_thinking_levels
            ),
            supports_images=self.deployment.supports_images,
            auto_compact_threshold=int(
                self.deployment.auto_compact_threshold
                if self.deployment.auto_compact_threshold is not None
                else auto_compact_threshold
            ),
        )


class ModelResolver:
    """Resolve singleton presets, selecting a deployment of their exact model."""

    def __init__(self, snapshot: CatalogSnapshot) -> None:
        self.snapshot = snapshot

    def eligible_models(self, allowed_models: Sequence[str] = ()) -> set[str]:
        """Canonical bases with an enabled, permitted deployment."""
        return {
            base
            for base, definition in self.snapshot.catalog.models.items()
            if not definition.disabled
            and any(
                not deployment.disabled
                and not self.snapshot.catalog.providers[deployment.provider].disabled
                and self._allowed(base, definition, deployment, allowed_models)
                for deployment in definition.deployments
            )
        }

    def eligible_roles(self, allowed_models: Sequence[str] = ()) -> list[str]:
        eligible = self.eligible_models(allowed_models)
        return sorted(
            name
            for name, role in self.snapshot.catalog.roles.items()
            if role.model in eligible
        )

    def canonicalize(
        self, expression: str, *, allowed_models: Sequence[str] = ()
    ) -> str:
        """Return the one base represented by a non-role expression."""
        if not isinstance(expression, str) or not expression:
            raise ModelResolutionError(
                "invalid_expression", "Model expression must be a non-empty string"
            )
        if expression == "@" or expression.startswith("@"):
            if expression == "@":
                raise ModelResolutionError(
                    "reserved_at", "'@' is reserved for a named role"
                )
            raise ModelResolutionError(
                "role_not_scalar", "A role expression selects a named preset"
            )
        if expression in self.snapshot.catalog.models:
            return expression
        if "/" in expression:
            raise ModelResolutionError(
                "invalid_expression",
                "Provider-qualified model expressions are unsupported; use a canonical base name",
            )
        options = ", ".join(sorted(self.eligible_models(allowed_models))) or "none"
        raise ModelResolutionError(
            "unknown_model",
            f"Unknown model expression {expression!r}. Valid canonical models: {options}",
        )

    def expression_bases(
        self, expression: str, *, allowed_models: Sequence[str] = ()
    ) -> tuple[str, ...]:
        """Expand one scalar expression into canonical base names."""
        if not isinstance(expression, str):
            raise ModelResolutionError(
                "invalid_expression",
                "Model expression must be a bare string; fallback arrays are unsupported",
            )
        if expression == "@":
            raise ModelResolutionError(
                "reserved_at", "'@' is reserved for a named role"
            )
        if expression.startswith("@"):
            definition = self.snapshot.catalog.roles.get(expression[1:])
            if definition is None:
                options = (
                    ", ".join(
                        f"@{name}" for name in self.eligible_roles(allowed_models)
                    )
                    or "none"
                )
                raise ModelResolutionError(
                    "unknown_role",
                    f"Unknown model role {expression!r}. Valid roles: {options}",
                )
            return (definition.model,)
        return (self.canonicalize(expression, allowed_models=allowed_models),)

    def resolve(
        self,
        expression: str,
        *,
        allowed_models: Sequence[str] = (),
        candidate_filter: Callable[[ResolvedModel], bool] | None = None,
        thinking_override: str | None = None,
    ) -> ResolvedModel:
        """Resolve a single canonical model, selecting an eligible deployment."""
        role = (
            self.snapshot.catalog.roles.get(expression[1:])
            if isinstance(expression, str) and expression.startswith("@")
            else None
        )
        last_error: ModelResolutionError | None = None
        for base in self.expression_bases(expression, allowed_models=allowed_models):
            if base not in self.snapshot.catalog.models:
                raise ModelResolutionError(
                    "preset_model_missing",
                    f"Preset {expression!r} selects missing model {base!r}; edit its model and thinking fields",
                )
            definition = self.snapshot.catalog.models[base]
            if definition.disabled:
                last_error = ModelResolutionError(
                    "model_disabled", f"Model {base!r} is disabled"
                )
                continue
            deployments = list(definition.deployments)
            eligible = [
                deployment
                for deployment in deployments
                if not deployment.disabled
                and not self.snapshot.catalog.providers[deployment.provider].disabled
            ]
            if not eligible:
                last_error = ModelResolutionError(
                    "all_deployments_disabled",
                    f"All deployments for model {base!r} are disabled",
                )
                continue
            allowed = [
                deployment
                for deployment in eligible
                if self._allowed(base, definition, deployment, allowed_models)
            ]
            if not allowed:
                last_error = ModelResolutionError(
                    "allowlist_excluded",
                    f"Model {base!r} has no deployment permitted by allowed_models",
                )
                continue
            thinking_compatible = False
            for deployment in allowed:
                if role is not None and not _supports_thinking(
                    self.snapshot.catalog.providers[deployment.provider],
                    deployment,
                    thinking_override
                    if thinking_override is not None
                    else role.thinking,
                ):
                    continue
                thinking_compatible = True
                resolved = ResolvedModel(
                    base,
                    definition,
                    deployment,
                    self.snapshot.catalog.providers[deployment.provider],
                    self.snapshot.revision,
                    role.thinking if role is not None else None,
                )
                if candidate_filter is None or candidate_filter(resolved):
                    return resolved
            last_error = ModelResolutionError(
                "thinking_unsupported"
                if role is not None and not thinking_compatible
                else "no_compatible_deployment",
                (
                    f"Preset {expression!r} has no deployment of model {base!r} "
                    f"supporting thinking level "
                    f"{(thinking_override if thinking_override is not None else role.thinking)!r}; "
                    "edit its thinking level"
                    if role is not None and not thinking_compatible
                    else f"Model {base!r} has no compatible available deployment"
                ),
            )
        if last_error is not None:
            raise last_error
        raise ModelResolutionError("no_eligible_model", "No eligible model was found")

    def resolve_committed(
        self, identity: CommittedModelIdentity, *, allowed_models: Sequence[str] = ()
    ) -> ResolvedModel:
        """Validate and materialize a stored deployment without re-selecting it."""
        definition = self.snapshot.catalog.models.get(identity.base_model)
        if definition is None:
            raise ModelResolutionError(
                "committed_model_missing",
                f"Committed base model {identity.base_model!r} is missing from the current catalog",
            )
        if definition.disabled:
            raise ModelResolutionError(
                "committed_model_disabled",
                f"Committed base model {identity.base_model!r} is disabled in the current catalog",
            )
        deployment = next(
            (
                item
                for item in definition.deployments
                if item.provider == identity.provider
                and item.name == identity.wire_name
            ),
            None,
        )
        label = f"{identity.provider}/{identity.wire_name}"
        if deployment is None:
            raise ModelResolutionError(
                "committed_deployment_missing",
                f"Committed deployment {label!r} for {identity.base_model!r} is missing from the current catalog",
            )
        provider = self.snapshot.catalog.providers.get(identity.provider)
        if deployment.disabled or provider is None or provider.disabled:
            raise ModelResolutionError(
                "committed_deployment_disabled",
                f"Committed deployment {label!r} for {identity.base_model!r} is disabled in the current catalog",
            )
        if not self._allowed(
            identity.base_model, definition, deployment, allowed_models
        ):
            raise ModelResolutionError(
                "allowlist_excluded",
                f"Committed deployment {label!r} is not permitted by allowed_models",
            )
        if identity.thinking is not None and not _supports_thinking(
            provider, deployment, identity.thinking
        ):
            raise ModelResolutionError(
                "committed_thinking_unsupported",
                f"Committed thinking level {identity.thinking!r} is no longer "
                f"supported by {label!r}; edit the preset thinking level",
            )
        return ResolvedModel(
            identity.base_model,
            definition,
            deployment,
            provider,
            self.snapshot.revision,
            identity.thinking,
        )

    def resolve_precedence(
        self,
        *,
        task: str | None = None,
        profile: str | None = None,
        parent_concrete: str | None = None,
        default: str | None = None,
        allowed_models: Sequence[str] = (),
    ) -> ResolvedModel:
        for expression in (task, profile, parent_concrete, default):
            if expression is not None and expression != "":
                return self.resolve(expression, allowed_models=allowed_models)
        raise ModelResolutionError("no_selection", "No model selection was supplied")

    def canonicalize_thinking_overrides(
        self, overrides: dict[str, Any]
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for expression, value in overrides.items():
            base = self.canonicalize(expression)
            if base in result:
                raise ModelResolutionError(
                    "conflicting_thinking_override",
                    f"Multiple thinking overrides resolve to {base!r}",
                )
            result[base] = value
        return result

    @staticmethod
    def _allowed(
        base: str,
        _definition: BaseModelDefinition,
        deployment: DeploymentDefinition,
        patterns: Sequence[str],
    ) -> bool:
        if not patterns:
            return True
        candidates = (base, f"{deployment.provider}/{deployment.name}")
        return any(name_matches(candidate, list(patterns)) for candidate in candidates)


def resolver_for(config: ChartreuxConfigSchema) -> ModelResolver:
    """Retrieve the catalog resolver attached by the authoritative builder."""
    snapshot = config.catalog_snapshot
    if snapshot is None:
        raise ModelResolutionError(
            "catalog_unattached",
            "No validated model catalog is attached to this configuration",
        )
    return ModelResolver(snapshot)
