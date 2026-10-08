"""Conversation-owned dispatch snapshots, independent of saved catalog edits."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, field_serializer, field_validator

from chartreux.core.dispatch.lint import RosterShape
from chartreux.core.dispatch.schema import DispatchMode, DispatchPolicy
from chartreux.core.session_types import CommittedModelIdentity
from chartreux.observability.logging import logger

if TYPE_CHECKING:
    from chartreux.core.config import ChartreuxConfigSchema


class SessionPolicyError(ValueError):
    """A persisted binding cannot safely be resumed."""


class BoundDispatchPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal[1] = 1
    policy: DispatchPolicy
    bindings: Mapping[str, CommittedModelIdentity]
    diagnostics: tuple[str, ...] = ()
    activation_mode: Literal[DispatchMode.STANDALONE] | None = None

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> BoundDispatchPolicy:
        return self

    @field_validator("bindings")
    @classmethod
    def freeze_bindings(
        cls, value: Mapping[str, CommittedModelIdentity]
    ) -> Mapping[str, CommittedModelIdentity]:
        return MappingProxyType(dict(value))

    @field_serializer("bindings")
    def serialize_bindings(
        self, value: Mapping[str, CommittedModelIdentity]
    ) -> dict[str, Any]:
        return {
            name: identity.model_dump(mode="json") for name, identity in value.items()
        }

    @property
    def render_policy(self) -> DispatchPolicy:
        if self.activation_mode is not None:
            return self.policy.model_copy(update={"mode": self.activation_mode})
        return self.policy

    @property
    def roster(self) -> RosterShape:
        return RosterShape(
            frozenset(
                (item.base_model, item.thinking)
                for name, item in self.bindings.items()
                if not any(f"slot {name!r}:" in note for note in self.diagnostics)
            ),
            {
                name: (item.base_model, item.thinking)
                for name, item in self.bindings.items()
            },
            {
                name: "bound slot unavailable; see dispatch diagnostics"
                for name in self.policy.slots
                if name not in self.bindings
                or any(f"slot {name!r}:" in note for note in self.diagnostics)
            },
        )

    def identity_for(self, expression: str) -> CommittedModelIdentity | None:
        matches = [
            self.bindings[name]
            for name, slot in self.policy.slots.items()
            if slot.role == expression and name in self.bindings
        ]
        return matches[0] if matches else None


def bind_policy(
    config: ChartreuxConfigSchema, *, legacy: bool = False
) -> BoundDispatchPolicy:
    from chartreux.core.dispatch.presets import DEFAULT_DISPATCH_MODE, SHIPPED_PRESETS
    from chartreux.core.model_catalog.defaults import SHIPPED_CATALOG
    from chartreux.core.model_catalog.loader import CatalogSnapshot
    from chartreux.core.model_catalog.resolver import (
        ModelResolutionError,
        ModelResolver,
    )

    snapshot = config.catalog_snapshot or CatalogSnapshot(SHIPPED_CATALOG, "session")
    policy = (
        snapshot.dispatch
        if config.catalog_snapshot
        else SHIPPED_PRESETS[DEFAULT_DISPATCH_MODE]
    )
    diagnostics = list(snapshot.dispatch_diagnostics)
    bindings = {}
    resolver = ModelResolver(snapshot)
    for name, slot in policy.slots.items():
        try:
            resolved = resolver.resolve(slot.role, allowed_models=config.allowed_models)
            model = resolved.materialize(
                auto_compact_threshold=config.auto_compact_threshold
            )
            bindings[name] = resolved.identity.model_copy(
                update={"thinking": model.thinking}
            )
        except ModelResolutionError as exc:
            diagnostics.append(
                f"Dispatch slot {name!r}: {slot.role} unavailable: {exc}; launches fail closed."
            )
    if legacy:
        diagnostics.append(
            "Legacy session has no dispatch policy snapshot; resolved live policy on resume."
        )
    for note in diagnostics:
        logger.warning("%s", note)
    return BoundDispatchPolicy(
        policy=policy, bindings=bindings, diagnostics=tuple(diagnostics)
    )


def resume_policy(
    config: ChartreuxConfigSchema,
    saved: dict[str, Any] | None,
    profiles: Mapping[str, object],
) -> BoundDispatchPolicy:
    if saved is None:
        return bind_policy(config, legacy=True)
    if saved.get("version") != 1:
        raise SessionPolicyError(
            f"Unsupported dispatch policy snapshot version: {saved.get('version')!r}"
        )
    bound = BoundDispatchPolicy.model_validate(saved)
    from chartreux.core.model_catalog.defaults import SHIPPED_CATALOG
    from chartreux.core.model_catalog.loader import CatalogSnapshot

    snapshot = config.catalog_snapshot or CatalogSnapshot(SHIPPED_CATALOG, "resume")
    for name in bound.bindings:
        slot = bound.policy.slots.get(name)
        if slot is None:
            raise SessionPolicyError(f"Unknown bound dispatch slot: {name}")
        if slot.profile not in profiles:
            raise SessionPolicyError(f"Bound dispatch profile removed: {slot.profile}")
        if snapshot is not None and slot.role[1:] not in snapshot.catalog.roles:
            raise SessionPolicyError(f"Bound dispatch role removed: {slot.role}")
    # Deployment/credential unavailability is an activation failure, not a
    # reference deletion: preserve the original snapshot and visibly degrade.
    if snapshot is not None:
        from chartreux.core.model_catalog.resolver import (
            ModelResolutionError,
            ModelResolver,
        )

        # Activation failures describe this resolution attempt, not history.
        # Policy recovery and legacy provenance remain part of the frozen policy.
        diagnostics = [
            note for note in bound.diagnostics if not note.startswith("Dispatch slot ")
        ]
        resolver = ModelResolver(snapshot)
        for name, slot in bound.policy.slots.items():
            if name in bound.bindings:
                continue
            try:
                resolver.resolve(slot.role, allowed_models=config.allowed_models)
            except ModelResolutionError as exc:
                diagnostics.append(
                    f"Dispatch slot {name!r}: {slot.role} unavailable: {exc}; launches fail closed."
                )
            else:
                diagnostics.append(
                    f"Dispatch slot {name!r}: no committed binding in this session; launches fail closed. Start a new session to bind the repaired role."
                )
        for name, identity in bound.bindings.items():
            try:
                resolved = ModelResolver(snapshot).resolve_committed(
                    identity, allowed_models=config.allowed_models
                )
                model = resolved.materialize(
                    auto_compact_threshold=config.auto_compact_threshold
                )
                provider = snapshot.catalog.providers[model.provider]
                from chartreux.utils.api_keys import resolve_api_key

                if provider.api_key_env_var and not resolve_api_key(
                    provider.api_key_env_var
                ):
                    diagnostics.append(
                        f"Dispatch slot {name!r}: credential unavailable for {model.provider}; launches fail closed."
                    )
            except ModelResolutionError as exc:
                diagnostics.append(
                    f"Dispatch slot {name!r}: bound model unavailable: {exc}; launches fail closed."
                )
        for note in diagnostics:
            logger.warning("%s", note)
        return bound.model_copy(
            update={"diagnostics": tuple(dict.fromkeys(diagnostics))}
        )
    return bound
