from __future__ import annotations

from collections.abc import Awaitable, Callable
import copy
from typing import Any, cast
from uuid import UUID, uuid4

from jsonpatch import apply_patch as json_apply_patch
from jsonpointer import JsonPointer

from chartreux.core.config import ChartreuxConfigSchema, RawConfig
from chartreux.core.config._restrictions import SourceRestrictions
from chartreux.core.config.builder import ConfigBuilder
from chartreux.core.config.event_bus import EventBus
from chartreux.core.config.layer import ConfigLayer
from chartreux.core.config.layers.default import DefaultConfigLayer
from chartreux.core.config.layers.overrides import OverridesLayer
from chartreux.core.config.layers.user import UserConfigLayer
from chartreux.core.config.orchestrator import ConfigOrchestrator, _changed_keys_between
from chartreux.core.config.patch import PatchOp, ensure_parent_paths
from chartreux.core.config.types import ConfigChangeEvent, ConflictStrategy
from chartreux.core.model_catalog.availability import AvailabilityRegistry
from chartreux.core.utils.concurrency import run_sync


class FakeConfigOrchestrator[C: ChartreuxConfigSchema](ConfigOrchestrator[C]):
    """In-memory test double that holds a config verbatim, skipping the layered
    ConfigOrchestrator machinery (builder, bus, layer stack).

    Reads return exactly the config the test built. Writes to a persisted layer
    are mirrored through a real default-plus-user layer stack so sparse writes
    have production merge semantics; writes targeting the in-memory overrides
    layer stay session-local.

    The verbatim config is treated as the base layer; additional in-memory
    layers (e.g. the agent-profile layer) may be inserted and are folded onto it
    by ``rebuild`` using production merge semantics.
    """

    def __init__(self, config: C) -> None:
        self._base_config = config
        self._config = config
        self._extra_layers: list[ConfigLayer[RawConfig]] = []
        self._bus = EventBus()
        self._policy_owner = uuid4()
        self._accepted_token = uuid4()
        self._availability_registry = AvailabilityRegistry()

    def copy(self, *, config: C | None = None) -> FakeConfigOrchestrator[C]:
        base_config = self._base_config if config is None else config
        clone = FakeConfigOrchestrator(base_config.model_copy(deep=True))
        clone._policy_owner = self._policy_owner
        clone._availability_registry = self._availability_registry
        clone._extra_layers = [copy.deepcopy(layer) for layer in self._extra_layers]
        clone.rebuild()
        return clone

    @property
    def policy_owner(self) -> UUID:
        # Model writer ownership without inventing provenance for verbatim config.
        return self._policy_owner

    def _copy_for_child(self) -> FakeConfigOrchestrator[C]:
        child = self.copy()
        child._policy_owner = uuid4()
        return child

    @property
    def layers(self) -> tuple[ConfigLayer[RawConfig], ...]:
        return tuple(self._extra_layers)

    def insert_layer(self, layer: ConfigLayer[RawConfig], index: int) -> None:
        self._extra_layers.insert(index, layer)

    def remove_layer(self, index: int) -> ConfigLayer[RawConfig]:
        return self._extra_layers.pop(index)

    def rebuild(self) -> None:
        builder = ConfigBuilder(
            type(self._base_config), catalog_snapshot=self._base_config.catalog_snapshot
        )
        base = DefaultConfigLayer(schema=type(self._base_config))
        builder.add_layer(base)
        builder.add_layers(list(self._extra_layers))
        # This test double supplies a fixture for the actual builtin layer; it
        # does not grant catalog authority to runtime or profile overrides.
        result = run_sync(
            builder.build(
                layer_overrides={
                    base.name: RawConfig.model_validate(
                        self._base_config.model_dump(mode="json")
                    )
                }
            )
        )
        # The verbatim base is already validated (e.g. active_model fixed up), so
        # re-validation won't reproduce its warnings; carry them forward to match
        # the real orchestrator, which re-merges raw layer snapshots.
        for warning in self._base_config.validation_warnings:
            if warning not in result.validation_warnings:
                result._validation_warnings.append(warning)
        self._config = result

    def _publish(self, before: dict[str, Any], reason: str) -> None:
        after = self._config.model_dump(mode="json")
        if changed := _changed_keys_between(before, after):
            self._bus.publish(
                ConfigChangeEvent(
                    changed_keys=changed, before=before, after=after, reason=reason
                )
            )

    @property
    def config(self) -> C:
        return self._config

    @property
    def restrictions(self) -> tuple[SourceRestrictions, ...]:
        # Verbatim snapshots have no source provenance. Source-policy tests must
        # construct the production orchestrator instead of this test double.
        return ()

    @property
    def writable_layer_name(self) -> str:
        return UserConfigLayer().name

    async def load_persistence_layer(
        self, *, target_layer: str | None = None
    ) -> RawConfig:
        if target_layer is not None and target_layer != "user-toml":
            return await self.get_layer(target_layer).load()
        return await UserConfigLayer().load()

    def persisted_active_model(self) -> str:
        # The verbatim fake has no layer stack, so the held config's value is
        # exactly what the test declared as the user's pin.
        return self._base_config.active_model

    async def set_field(
        self,
        path: str,
        value: Any,
        reason: str = "No reason",
        *,
        target_layer: str | None = None,
        preflight: Callable[[C], Awaitable[None]] | None = None,
    ) -> list[BaseException]:
        data = self._base_config.model_dump()
        _set_pointer_in_place(data, path, value)
        candidate = self.copy()
        candidate._base_config = cast(
            C,
            type(self._base_config)
            .model_validate(data)
            .attach_catalog_snapshot(self._base_config.catalog_snapshot),
        )
        candidate.rebuild()
        if preflight is not None:
            await preflight(candidate.config)
        if target_layer != OverridesLayer.NAME:
            orchestrator = await self._persistence_orchestrator()
            await orchestrator.set_field(path, value, reason)
        before = self._config.model_dump(mode="json")
        self._base_config = cast(
            C,
            type(self._base_config)
            .model_validate(data)
            .attach_catalog_snapshot(self._base_config.catalog_snapshot),
        )
        self.rebuild()
        self._publish(before, reason)
        return []

    async def mutate_field(
        self,
        path: str,
        mutate: Callable[[Any], Any],
        reason: str = "No reason",
        *,
        default: Any = None,
        target_layer: str | None = None,
        preflight: Callable[[C], Awaitable[None]] | None = None,
    ) -> list[BaseException]:
        raw = (await self.load_persistence_layer()).model_dump()
        current = JsonPointer(path).resolve(raw, default=copy.deepcopy(default))
        return await self.set_field(
            path,
            mutate(copy.deepcopy(current)),
            reason,
            target_layer=target_layer,
            preflight=preflight,
        )

    async def apply_session_patch(
        self,
        operations: list[PatchOp],
        *,
        reason: str,
        preflight: Callable[[C], Awaitable[None]],
        apply: Callable[[C], None],
    ) -> list[BaseException]:
        if any(op.target_layer_name not in {None, "overrides"} for op in operations):
            raise ValueError("Session writes require the session target")
        staged = self.copy()
        failures = await staged.apply_patch(
            [
                op.model_copy(update={"target_layer_name": "overrides"})
                for op in operations
            ],
            reason=reason,
        )
        if failures:
            return failures
        await preflight(staged.config)
        apply(staged.config)
        before = self.config.model_dump(mode="json")
        self._base_config = staged._base_config
        self._config = staged._config
        self._extra_layers = staged._extra_layers
        self._publish(before, reason)
        return []

    async def apply_patch(
        self,
        operations: list[PatchOp],
        reason: str = "No reason",
        *,
        on_conflict: ConflictStrategy = ConflictStrategy.CANCEL,
        preflight: Callable[[C], Awaitable[None]] | None = None,
    ) -> list[BaseException]:
        data = ensure_parent_paths(self._base_config.model_dump(), operations)
        data = json_apply_patch(
            data, [op.to_json_patch() for op in operations], in_place=False
        )
        candidate = self.copy()
        candidate._base_config = cast(
            C,
            type(self._base_config)
            .model_validate(data)
            .attach_catalog_snapshot(self._base_config.catalog_snapshot),
        )
        candidate.rebuild()
        if preflight is not None:
            await preflight(candidate.config)
        before = self._config.model_dump(mode="json")
        persistent_operations = [
            operation
            for operation in operations
            if operation.target_layer_name != OverridesLayer.NAME
        ]
        if persistent_operations:
            orchestrator = await self._persistence_orchestrator()
            failures = await orchestrator.apply_patch(
                persistent_operations, reason, on_conflict=on_conflict
            )
            if failures:
                return failures
        self._base_config = cast(
            C,
            type(self._base_config)
            .model_validate(data)
            .attach_catalog_snapshot(self._base_config.catalog_snapshot),
        )
        self.rebuild()
        self._publish(before, reason)
        return []

    async def reload(
        self,
        *,
        preflight: Callable[[C], Awaitable[None]] | None = None,
        apply: Callable[[C], None] | None = None,
    ) -> None:
        if preflight is not None:
            await preflight(self._config)
        if apply is not None:
            apply(self._config)

    async def _persistence_orchestrator(self) -> ConfigOrchestrator[C]:
        layer = UserConfigLayer()
        return await ConfigOrchestrator.create(
            schema=type(self._config),
            layers=[DefaultConfigLayer(schema=type(self._config)), layer],
            default_layer_resolver=lambda: layer,
        )


def _set_pointer_in_place(root: dict[str, Any], path: str, value: Any) -> None:
    parts = JsonPointer(path).parts
    target: Any = root
    for part in parts[:-1]:
        if not isinstance(target.get(part), dict):
            target[part] = {}
        target = target[part]
    target[parts[-1]] = value
