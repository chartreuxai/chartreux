"""Persistence-free Provider Settings draft and catalog patch derivation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Literal

from chartreux.core.llm.thinking_levels import get_thinking_levels
from chartreux.core.model_catalog.contracts import (
    ApiStyle,
    CatalogChanges,
    DiscoveryError,
    DiscoveryResult,
    ModelEdits,
    OptionalEdit,
)
from chartreux.core.model_catalog.loader import CatalogSnapshot
from chartreux.core.model_catalog.matching import MatchOutcome, match_discovered_model
from chartreux.core.model_catalog.schema import (
    DeploymentDefinition,
    ModelCatalog,
    ProviderDefinition,
)

type CredentialStatusResolver = Callable[[str], str | None]
type CollisionDecision = Literal["add_existing", "separate"]


@dataclass(frozen=True, slots=True)
class ConnectionDraft:
    """Editable connection fields; provider identity and credentials are separate."""

    api_base: str
    api_style: ApiStyle
    api_key_env_var: str
    backend: str
    reasoning_field_name: str
    extra_headers: Mapping[str, str]

    @classmethod
    def from_definition(cls, definition: ProviderDefinition) -> ConnectionDraft:
        return cls(
            definition.api_base,
            definition.api_style,
            definition.api_key_env_var,
            definition.backend,
            definition.reasoning_field_name,
            dict(definition.extra_headers),
        )


@dataclass(frozen=True, slots=True)
class PendingModel:
    wire_name: str
    canonical_name: str
    enabled: bool = False
    decision: CollisionDecision | None = None
    edits: ModelEdits = field(default_factory=ModelEdits)


@dataclass(frozen=True, slots=True)
class Validation:
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    unresolved_roles: tuple[str, ...] = ()
    unusable_roles: tuple[str, ...] = ()


@dataclass(slots=True)
class ManagementState:
    """One catalog draft shared by provider and canonical-model views."""

    snapshot: CatalogSnapshot
    provider_id: str
    staged_providers: dict[str, ProviderDefinition] = field(default_factory=dict)
    connections: dict[str, ConnectionDraft] = field(default_factory=dict)
    enabled_by_deployment: dict[tuple[str, str], bool] = field(default_factory=dict)
    deployment_edits: dict[tuple[str, str], ModelEdits] = field(default_factory=dict)
    canonical_edits: dict[str, dict[str, OptionalEdit[object]]] = field(
        default_factory=dict
    )
    role_presets: dict[str, tuple[str, str]] = field(default_factory=dict)
    pending_by_provider: dict[str, dict[str, PendingModel]] = field(
        default_factory=dict
    )
    discovery_generations: dict[str, int] = field(default_factory=dict)
    discoveries: dict[str, DiscoveryResult | DiscoveryError | None] = field(
        default_factory=dict
    )

    @classmethod
    def from_snapshot(
        cls, snapshot: CatalogSnapshot, provider_id: str
    ) -> ManagementState:
        return cls(snapshot, provider_id)

    def for_provider(self, provider_id: str) -> ManagementState:
        """Switch the focused provider without discarding any catalog edits."""
        if provider_id not in self.catalog.providers:
            raise KeyError(provider_id)
        self.provider_id = provider_id
        return self

    def stage_provider(self, provider_id: str, definition: ProviderDefinition) -> None:
        """Keep every new provider in the one pending catalog transaction."""
        self.staged_providers[provider_id] = definition
        catalog = self.catalog.model_copy(
            update={"providers": {**self.catalog.providers, provider_id: definition}}
        )
        self.snapshot = CatalogSnapshot(
            catalog, self.snapshot.revision, self.snapshot.overlaid_providers
        )
        self.provider_id = provider_id
        self.connections[provider_id] = ConnectionDraft.from_definition(definition)

    @property
    def connection(self) -> ConnectionDraft:
        return self.connections.get(
            self.provider_id,
            ConnectionDraft.from_definition(self.catalog.providers[self.provider_id]),
        )

    @connection.setter
    def connection(self, value: ConnectionDraft) -> None:
        self.connections[self.provider_id] = value

    @property
    def enabled(self) -> dict[str, bool]:
        return {
            name: value
            for (name, provider), value in self.enabled_by_deployment.items()
            if provider == self.provider_id
        }

    @enabled.setter
    def enabled(self, values: dict[str, bool]) -> None:
        self.enabled_by_deployment = {
            key: value
            for key, value in self.enabled_by_deployment.items()
            if key[1] != self.provider_id
        }
        self.enabled_by_deployment.update({
            (name, self.provider_id): value for name, value in values.items()
        })

    @property
    def edits(self) -> dict[str, ModelEdits]:
        edits = {
            name: value
            for (name, provider), value in self.deployment_edits.items()
            if provider == self.provider_id
        }
        for name in self.configured():
            canonical = self.canonical_edits.get(name)
            if canonical:
                edits[name] = self._with_canonical_edits(
                    edits.get(name, ModelEdits()), canonical
                )
        return edits

    @property
    def pending(self) -> dict[str, PendingModel]:
        pending = self.pending_by_provider.setdefault(self.provider_id, {})
        for wire_name, item in tuple(pending.items()):
            canonical = self.canonical_edits.get(item.canonical_name)
            if canonical:
                pending[wire_name] = replace(
                    item, edits=self._with_canonical_edits(item.edits, canonical)
                )
        return pending

    @pending.setter
    def pending(self, values: dict[str, PendingModel]) -> None:
        self.pending_by_provider[self.provider_id] = values

    @property
    def discovery_generation(self) -> int:
        return self.discovery_generations.get(self.provider_id, 0)

    @discovery_generation.setter
    def discovery_generation(self, value: int) -> None:
        self.discovery_generations[self.provider_id] = value

    @property
    def discovery(self) -> DiscoveryResult | DiscoveryError | None:
        return self.discoveries.get(self.provider_id)

    @discovery.setter
    def discovery(self, value: DiscoveryResult | DiscoveryError | None) -> None:
        self.discoveries[self.provider_id] = value

    @property
    def catalog(self) -> ModelCatalog:
        return self.snapshot.catalog

    def configured(self) -> dict[str, DeploymentDefinition]:
        return {
            name: deployment
            for name, definition in self.catalog.models.items()
            for deployment in definition.deployments
            if deployment.provider == self.provider_id
        }

    def model_rows(self) -> tuple[tuple[str, str, bool, bool], ...]:
        """Canonical name, wire ID, checked state, discovered state; union of both lists."""
        discovered = (
            {item.wire_id for item in self.discovery.models}
            if isinstance(self.discovery, DiscoveryResult)
            else set()
        )
        rows = [
            (
                name,
                deployment.name,
                self.enabled.get(name, not deployment.disabled),
                deployment.name in discovered,
            )
            for name, deployment in self.configured().items()
        ]
        rows.extend(
            (item.canonical_name, item.wire_name, item.enabled, True)
            for item in self.pending.values()
        )
        known_wires = {wire for _name, wire, _enabled, _found in rows}
        rows.extend((wire, wire, False, True) for wire in discovered - known_wires)
        return tuple(sorted(rows))

    def toggle(self, canonical_name: str, enabled: bool) -> None:
        if canonical_name not in self.configured():
            raise KeyError(canonical_name)
        self.enabled_by_deployment[canonical_name, self.provider_id] = enabled

    def set_edits(self, canonical_name: str, edits: ModelEdits) -> None:
        if canonical_name not in self.configured():
            raise KeyError(canonical_name)
        self.deployment_edits[canonical_name, self.provider_id] = edits
        self._accept_canonical_edits(canonical_name, edits)

    def set_pending_edits(self, wire_name: str, edits: ModelEdits) -> None:
        """Replace a pending model's edits and accept its canonical metadata edits."""
        item = self.pending.get(wire_name)
        if item is None:
            raise KeyError(wire_name)
        self.pending[wire_name] = replace(item, edits=edits)
        self._accept_canonical_edits(item.canonical_name, edits)

    @staticmethod
    def _with_canonical_edits(
        edits: ModelEdits, canonical: Mapping[str, OptionalEdit[object]]
    ) -> ModelEdits:
        return replace(edits, **canonical)

    def _accept_canonical_edits(self, canonical_name: str, edits: ModelEdits) -> None:
        canonical = self.canonical_edits.get(canonical_name, {})
        for field_name in ("thinking", "temperature"):
            edit = getattr(edits, field_name)
            if edit.state != "untouched":
                canonical = self.canonical_edits.setdefault(canonical_name, canonical)
                canonical[field_name] = edit
        if canonical:
            for key, existing in tuple(self.deployment_edits.items()):
                if key[0] == canonical_name:
                    self.deployment_edits[key] = self._with_canonical_edits(
                        existing, canonical
                    )
            for pending in self.pending_by_provider.values():
                for wire_name, item in tuple(pending.items()):
                    if item.canonical_name == canonical_name:
                        pending[wire_name] = replace(
                            item,
                            edits=self._with_canonical_edits(item.edits, canonical),
                        )

    def select(self, wire_name: str, *, canonical_name: str | None = None) -> None:
        outcome = match_discovered_model(self.catalog, self.provider_id, wire_name)
        if outcome.kind == "existing":
            assert outcome.existing_base is not None
            self.toggle(outcome.existing_base, True)
            return
        self.pending[wire_name] = PendingModel(
            wire_name, canonical_name or outcome.existing_base or wire_name, True
        )

    def set_connection(self, connection: ConnectionDraft) -> None:
        if connection != self.connection:
            self.connection = connection
            self.begin_discovery()

    def begin_discovery(self, provider_id: str | None = None) -> tuple[str, int]:
        """Start a request tied to the provider view that launched it."""
        provider_id = provider_id or self.provider_id
        if provider_id not in self.catalog.providers:
            raise KeyError(provider_id)
        generation = self.discovery_generations.get(provider_id, 0) + 1
        self.discovery_generations[provider_id] = generation
        self.discoveries[provider_id] = None
        return provider_id, generation

    def accept_discovery(
        self,
        provider_id: str,
        generation: int,
        result: DiscoveryResult | DiscoveryError,
    ) -> bool:
        if (
            provider_id not in self.catalog.providers
            or generation != self.discovery_generations.get(provider_id, 0)
        ):
            return False
        self.discoveries[provider_id] = result
        return True

    def _provider_patch(self) -> dict[str, object]:
        current = self.catalog.providers[self.provider_id]
        return {
            key: value
            for key, value in (
                ("api_base", self.connection.api_base),
                ("api_style", self.connection.api_style),
                ("api_key_env_var", self.connection.api_key_env_var),
                ("backend", self.connection.backend),
                ("reasoning_field_name", self.connection.reasoning_field_name),
                ("extra_headers", dict(self.connection.extra_headers)),
            )
            if getattr(current, key) != value
        }

    def _target(self, name: str, deployment: DeploymentDefinition) -> dict[str, object]:
        patch: dict[str, object] = {
            "provider": self.provider_id,
            "name": deployment.name,
        }
        enabled = self.enabled.get(name, not deployment.disabled)
        if enabled != (not deployment.disabled):
            patch["disabled"] = not enabled
        return patch

    @staticmethod
    def _price_patch(
        deployment: DeploymentDefinition | None, edits: ModelEdits
    ) -> dict[str, float]:
        prices = deployment.prices if deployment is not None else None
        values: dict[str, float] = {}
        for name, edit in (
            ("input", edits.input_price),
            ("output", edits.output_price),
            ("cached_input", edits.cached_input_price),
        ):
            previous = getattr(prices, name) if prices is not None else None
            value = (
                edit.value
                if edit.state == "set"
                else None
                if edit.state == "cleared"
                else previous
            )
            if value is not None:
                values[name] = value
        return values

    @staticmethod
    def _edit_value(edit: OptionalEdit[object]) -> object:
        return edit.value if edit.state == "set" else None

    def _canonical_edit_patches(self) -> dict[str, dict[str, object]]:
        """Build canonical metadata from the latest model-keyed authority."""
        result: dict[str, dict[str, object]] = {}
        for name, fields in self.canonical_edits.items():
            definition = self.catalog.models.get(name)
            if definition is None and not any(
                item.enabled and item.canonical_name == name
                for pending in self.pending_by_provider.values()
                for item in pending.values()
            ):
                continue
            patch: dict[str, object] = {}
            for field_name, edit in fields.items():
                target = self._edit_value(edit)
                if definition is None or getattr(definition, field_name) != target:
                    patch[field_name] = target
            if patch:
                result[name] = patch
        return result

    def _role_patches(self) -> dict[str, dict[str, object]]:
        return {
            name: {"model": model, "thinking": thinking}
            for name, (model, thinking) in self.role_presets.items()
            if (model, thinking)
            != (self.catalog.roles[name].model, self.catalog.roles[name].thinking)
        }

    def preset(self, role: str) -> tuple[str, str]:
        definition = self.catalog.roles[role]
        return self.role_presets.get(role, (definition.model, definition.thinking))

    def set_role_preset(self, role: str, model: str, thinking: str) -> None:
        """Stage one canonical model and thinking level for a named preset."""
        if role not in self.catalog.roles:
            raise ValueError(f"Unknown preset {role}.")
        if not model or "@" in model:
            raise ValueError("Choose one canonical model for the preset.")
        if thinking not in {"off", "low", "medium", "high", "max"}:
            raise ValueError(f"Unknown thinking level {thinking} for {role}.")
        self.role_presets[role] = (model, thinking)

    def preset_thinking_levels(self, model: str) -> tuple[str, ...]:
        """Return levels supported by enabled deployments of one canonical model."""
        definition = self.catalog.models.get(model)
        deployments = [
            (deployment.provider, deployment.name, deployment.supported_thinking_levels)
            for deployment in (definition.deployments if definition else ())
            if self.enabled_by_deployment.get(
                (model, deployment.provider), not deployment.disabled
            )
        ]
        deployments.extend(
            (provider, item.wire_name, None)
            for provider, pending in self.pending_by_provider.items()
            for item in pending.values()
            if item.enabled and item.canonical_name == model
        )
        available: set[str] = set()
        for provider_id, wire_name, declared in deployments:
            provider = self.catalog.providers[provider_id]
            if provider.disabled:
                continue
            connection = self.connections.get(provider_id)
            levels = get_thinking_levels(
                connection.backend if connection else provider.backend,
                connection.api_style if connection else provider.api_style,
                wire_name,
            )
            supported = set(declared or ("off", "low", "medium", "high", "max"))
            if levels is not None:
                supported.intersection_update(levels)
            available.update(supported)
        return tuple(
            level
            for level in ("off", "low", "medium", "high", "max")
            if level in available
        )

    def validate_preset(
        self, role: str, credential_resolver: CredentialStatusResolver | None
    ) -> str | None:
        model, thinking = self.preset(role)
        if model not in self.catalog.models and not any(
            item.enabled and item.canonical_name == model
            for pending in self.pending_by_provider.values()
            for item in pending.values()
        ):
            return f"Preset {role}: configure model {model} or choose another model."
        reason = self._preset_readiness(model, thinking, credential_resolver)
        return (
            f"Preset {role}: {reason}; repair it or choose another model/level."
            if reason
            else None
        )

    def _pending_outcome(self, item: PendingModel) -> MatchOutcome:
        return match_discovered_model(self.catalog, self.provider_id, item.wire_name)

    def _apply_pending_canonical_defaults(
        self, models: dict[str, dict[str, object]], provider_id: str, item: PendingModel
    ) -> None:
        if not item.enabled:
            return
        outcome = match_discovered_model(self.catalog, provider_id, item.wire_name)
        previous = self.catalog.models.get(item.canonical_name)
        canonical = models.setdefault(item.canonical_name, {})
        if previous is None and outcome.proposed is not None:
            canonical.setdefault("thinking", outcome.proposed.definition.thinking)
        authority = self.canonical_edits.get(item.canonical_name, {})
        for field_name in ("thinking", "temperature"):
            if field_name in authority:
                continue
            edit = getattr(item.edits, field_name)
            if edit.state == "untouched":
                continue
            target = self._edit_value(edit)
            current = getattr(previous, field_name) if previous is not None else None
            if current != target:
                canonical[field_name] = target
        if not canonical:
            models.pop(item.canonical_name, None)

    def _build_provider(self) -> CatalogChanges:
        patches: dict[str, dict[str, object]] = {}
        for name, deployment in self.configured().items():
            patch = self._target(name, deployment)
            edits = self.edits.get(name, ModelEdits())
            if any(
                edit.state != "untouched"
                for edit in (
                    edits.input_price,
                    edits.output_price,
                    edits.cached_input_price,
                )
            ):
                prices = self._price_patch(deployment, edits)
                if prices != deployment.prices.model_dump(exclude_none=True):
                    patch["prices"] = prices
            for field_name in ("supports_images", "auto_compact_threshold"):
                edit = getattr(edits, field_name)
                if edit.state == "untouched":
                    continue
                target = self._edit_value(edit)
                if getattr(deployment, field_name) != target:
                    patch[field_name] = target
            if any(
                key in patch
                for key in (
                    "disabled",
                    "prices",
                    "supports_images",
                    "auto_compact_threshold",
                )
            ):
                patches[name] = {"deployments": [patch]}
        for item in self.pending.values():
            if not item.enabled:
                continue
            canonical = item.canonical_name
            previous = self.catalog.models.get(canonical)
            deployment = (
                next(
                    (
                        dep
                        for dep in previous.deployments
                        if dep.provider == self.provider_id
                    ),
                    None,
                )
                if previous is not None
                else None
            )
            patch = patches.setdefault(canonical, {"deployments": []})
            entry: dict[str, object] = {
                "provider": self.provider_id,
                "name": item.wire_name,
            }
            if deployment is not None and deployment.disabled:
                entry["disabled"] = False
            if any(
                edit.state != "untouched"
                for edit in (
                    item.edits.input_price,
                    item.edits.output_price,
                    item.edits.cached_input_price,
                )
            ):
                entry["prices"] = self._price_patch(deployment, item.edits)
            for field_name in ("supports_images", "auto_compact_threshold"):
                edit = getattr(item.edits, field_name)
                if edit.state != "untouched":
                    target = self._edit_value(edit)
                    if deployment is None or getattr(deployment, field_name) != target:
                        entry[field_name] = target
            entries = patch["deployments"]
            assert isinstance(entries, list)
            entries.append(entry)
        return CatalogChanges(self.provider_id, self._provider_patch(), patches)

    def _build(self) -> CatalogChanges:
        focused = self.provider_id
        providers: dict[str, Mapping[str, object]] = {}
        models: dict[str, dict[str, object]] = {}
        for provider in self.catalog.providers:
            self.provider_id = provider
            part = self._build_provider()
            if part.provider:
                providers[provider] = part.provider
            for name, patch in part.models.items():
                combined = models.setdefault(name, {})
                for key, value in patch.items():
                    if key == "deployments":
                        combined.setdefault(key, []).extend(value)  # type: ignore[union-attr]
                    else:
                        combined[key] = value
        for provider_id, definition in self.staged_providers.items():
            providers[provider_id] = {
                **definition.model_dump(),
                **providers.get(provider_id, {}),
            }
        for provider_id, pending in self.pending_by_provider.items():
            for item in pending.values():
                self._apply_pending_canonical_defaults(models, provider_id, item)
        for name, patch in self._canonical_edit_patches().items():
            models.setdefault(name, {}).update(patch)
        self.provider_id = focused
        return CatalogChanges(
            focused,
            providers.pop(focused, {}),
            models,
            self._role_patches() or None,
            providers,
        )

    @property
    def dirty(self) -> bool:
        changes = self._build()
        return bool(changes.provider_patches or changes.models or changes.roles)

    def _usable(
        self,
        name: str,
        *,
        after: bool = False,
        credential_resolver: CredentialStatusResolver | None = None,
    ) -> bool:
        definition = self.catalog.models.get(name)
        if definition is not None and definition.disabled:
            return False

        def ready(provider_id: str) -> bool:
            provider = self.catalog.providers[provider_id]
            connection = self.connections.get(provider_id)
            env_var = (
                connection.api_key_env_var
                if after and connection
                else provider.api_key_env_var
            )
            return not provider.disabled and (
                not env_var
                or credential_resolver is None
                or bool(credential_resolver(env_var))
            )

        if after and any(
            item.enabled and item.canonical_name == name and ready(provider_id)
            for provider_id, pending in self.pending_by_provider.items()
            for item in pending.values()
        ):
            return True
        return definition is not None and any(
            ready(dep.provider)
            and (
                self.enabled_by_deployment.get((name, dep.provider), not dep.disabled)
                if after
                else not dep.disabled
            )
            for dep in definition.deployments
        )

    def _preset_readiness(
        self,
        model: str,
        thinking: str,
        credential_resolver: CredentialStatusResolver | None,
    ) -> str | None:
        """Check one deployment against credentials and thinking together."""
        definition = self.catalog.models.get(model)
        if definition is not None and definition.disabled:
            return "the canonical model is disabled"
        deployments: list[tuple[str, str, tuple[str, ...] | None, bool]] = [
            (
                deployment.provider,
                deployment.name,
                deployment.supported_thinking_levels,
                self.enabled_by_deployment.get(
                    (model, deployment.provider), not deployment.disabled
                ),
            )
            for deployment in (definition.deployments if definition else ())
        ]
        deployments.extend(
            (provider, item.wire_name, None, item.enabled)
            for provider, pending in self.pending_by_provider.items()
            for item in pending.values()
            if item.canonical_name == model
        )
        unavailable_credential: str | None = None
        unsupported = False
        for provider_id, wire_name, supported, enabled in deployments:
            if not enabled:
                continue
            provider = self.catalog.providers[provider_id]
            if provider.disabled:
                continue
            connection = self.connections.get(provider_id)
            env_var = (
                connection.api_key_env_var if connection else provider.api_key_env_var
            )
            if (
                env_var
                and credential_resolver is not None
                and not credential_resolver(env_var)
            ):
                unavailable_credential = env_var
                continue
            levels = get_thinking_levels(
                connection.backend if connection else provider.backend,
                connection.api_style if connection else provider.api_style,
                wire_name,
            )
            if (supported is not None and thinking not in supported) or (
                levels is not None and thinking not in levels
            ):
                unsupported = True
                continue
            return None
        if unavailable_credential:
            return f"credential {unavailable_credential} is unavailable"
        if unsupported:
            return (
                f"thinking {thinking} is unsupported by enabled deployments of {model}"
            )
        return "no enabled deployment is available"

    def _collision_errors(self) -> list[str]:
        errors: list[str] = []
        seen: set[str] = set()
        for item in self.pending.values():
            if not item.enabled:
                continue
            name = item.canonical_name
            if not name or "@" in name:
                errors.append(f"Choose a valid canonical name for {item.wire_name}.")
            if name in seen:
                errors.append(f"Duplicate pending canonical name: {name}.")
            seen.add(name)
            outcome = self._pending_outcome(item)
            if outcome.kind in {"existing", "multiple_matches"}:
                errors.append(f"{item.wire_name} already matches a catalog deployment.")
            elif outcome.kind == "base_exists_other_provider":
                if item.decision == "add_existing" and name != outcome.existing_base:
                    errors.append(
                        f"Choose the existing canonical name for {item.wire_name}."
                    )
                elif item.decision == "separate" and name == outcome.existing_base:
                    errors.append(
                        f"Choose a separate canonical name for {item.wire_name}."
                    )
                elif item.decision is None:
                    errors.append(f"Resolve the collision for {item.wire_name}.")
            elif outcome.kind == "occupied_slot" and name == outcome.existing_base:
                errors.append(
                    f"{name} already has an existing provider deployment; choose another canonical name."
                )
            if name in self.catalog.models and (
                any(
                    dep.provider == self.provider_id
                    for dep in self.catalog.models[name].deployments
                )
                or item.decision != "add_existing"
            ):
                errors.append(f"Canonical name {name} is already occupied.")
        return errors

    def validate(
        self,
        active_expression: str | None = None,
        *,
        mode: Literal["management", "onboarding"] = "management",
        credential_resolver: CredentialStatusResolver | None = None,
        completion: bool = True,
    ) -> Validation:
        """Validate structural edits; additionally check preset readiness at Finish."""
        errors: list[str] = []
        warnings: list[str] = []
        if completion and mode == "onboarding" and credential_resolver is None:
            errors.append("Credential status is required to finish setup.")
        for edits in self.deployment_edits.values():
            for field_name in ("thinking", "supports_images"):
                if getattr(edits, field_name).state == "cleared":
                    errors.append(f"{field_name} cannot be cleared.")
        for pending in self.pending_by_provider.values():
            for item in pending.values():
                if item.enabled:
                    for field_name in ("thinking", "supports_images"):
                        if getattr(item.edits, field_name).state == "cleared":
                            errors.append(f"{field_name} cannot be cleared.")
        changes = self._build()
        unresolved: list[str] = []
        unusable: list[str] = []
        if completion:
            for role in self.catalog.roles:
                model, thinking = self.preset(role)
                if model not in self.catalog.models and model not in changes.models:
                    unresolved.append(role)
                    errors.append(
                        f"Preset {role}: configure model {model} or choose another model."
                    )
                    continue
                reason = self._preset_readiness(model, thinking, credential_resolver)
                if reason is not None:
                    unusable.append(role)
                    errors.append(
                        f"Preset {role}: {reason}; repair it or choose another model/level."
                    )
        focused = self.provider_id
        for provider in self.catalog.providers:
            self.provider_id = provider
            errors.extend(self._collision_errors())
        self.provider_id = focused
        return Validation(
            tuple(errors), tuple(warnings), tuple(unresolved), tuple(unusable)
        )

    def changes(
        self,
        active_expression: str | None = None,
        *,
        mode: Literal["management", "onboarding"] = "management",
        credential_resolver: CredentialStatusResolver | None = None,
    ) -> CatalogChanges:
        validation = self.validate(
            active_expression,
            mode=mode,
            credential_resolver=credential_resolver,
            completion=False,
        )
        if validation.errors:
            raise ValueError("; ".join(validation.errors))
        return self._build()


def credential_status(env_var: str, resolver: CredentialStatusResolver) -> str:
    """The host resolves credentials; no environment or secret is read here."""
    return (
        "No Authentication"
        if not env_var
        else "Key Set"
        if resolver(env_var)
        else "Key Required"
    )


def has_usable_deployment(catalog: ModelCatalog, base: str) -> bool:
    definition = catalog.models[base]
    return not definition.disabled and any(
        not dep.disabled and not catalog.providers[dep.provider].disabled
        for dep in definition.deployments
    )
