"""Map discovered provider wire IDs to effective catalog identities.

This module is deliberately persistence-free: callers render its explicit outcomes
and turn a chosen outcome into a catalog change batch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from chartreux.core.llm.thinking_levels import get_thinking_levels
from chartreux.core.model_catalog.schema import (
    BaseModelDefinition,
    DeploymentDefinition,
    ModelCatalog,
    Prices,
    ProviderDefinition,
)


@dataclass(frozen=True, slots=True)
class CatalogMatch:
    """One base-model deployment that matched a discovered wire ID."""

    base_name: str
    definition: BaseModelDefinition
    deployment: DeploymentDefinition


@dataclass(frozen=True, slots=True)
class NewModelProposal:
    """Conservative definitions for a discovered model absent from the catalog."""

    base_name: str
    definition: BaseModelDefinition
    deployment: DeploymentDefinition


@dataclass(frozen=True, slots=True)
class MatchOutcome:
    """Facts the UI needs to present an identity decision without guessing."""

    kind: Literal[
        "existing",
        "base_exists_other_provider",
        "occupied_slot",
        "new_model",
        "multiple_matches",
    ]
    wire_name: str
    matches: tuple[CatalogMatch, ...] = ()
    existing_base: str | None = None
    proposed: NewModelProposal | None = None
    proposed_base_exists: bool = False
    provider_slot_occupied: bool = False


def match_discovered_model(
    catalog: ModelCatalog,
    provider_id: str,
    wire_name: str,
    *,
    provider: ProviderDefinition | None = None,
) -> MatchOutcome:
    """Classify one exact wire ID against the effective catalog.

    Exact deployment matches take precedence. A caller adding a new provider may
    provider may supply its unsaved definition so new-model thinking defaults use
    that provider's backend and API style.
    """
    matches = tuple(
        CatalogMatch(base_name, definition, deployment)
        for base_name, definition in catalog.models.items()
        for deployment in definition.deployments
        if deployment.provider == provider_id and deployment.name == wire_name
    )
    if len(matches) > 1:
        return MatchOutcome("multiple_matches", wire_name, matches=matches)
    if matches:
        return MatchOutcome(
            "existing", wire_name, matches=matches, existing_base=matches[0].base_name
        )

    existing_definition = catalog.models.get(wire_name)
    if existing_definition is not None:
        slot = next(
            (
                deployment
                for deployment in existing_definition.deployments
                if deployment.provider == provider_id
            ),
            None,
        )
        if slot is not None:
            return MatchOutcome(
                "occupied_slot",
                wire_name,
                existing_base=wire_name,
                proposed_base_exists=True,
                provider_slot_occupied=True,
            )
        return MatchOutcome(
            "base_exists_other_provider",
            wire_name,
            existing_base=wire_name,
            proposed_base_exists=True,
        )

    effective_provider = provider or catalog.providers.get(provider_id)
    return MatchOutcome(
        "new_model",
        wire_name,
        proposed=_new_model_proposal(provider_id, wire_name, effective_provider),
    )


def _new_model_proposal(
    provider_id: str, wire_name: str, provider: ProviderDefinition | None
) -> NewModelProposal:
    levels = get_thinking_levels(
        provider.backend if provider is not None else "generic",
        provider.api_style if provider is not None else "openai",
        wire_name,
    )
    thinking = "off" if levels is not None and "off" in levels else "medium"
    deployment = DeploymentDefinition(
        provider=provider_id, name=wire_name, prices=Prices(), supports_images=False
    )
    return NewModelProposal(
        wire_name,
        BaseModelDefinition(thinking=thinking, deployments=(deployment,)),
        deployment,
    )
