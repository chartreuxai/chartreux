from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Awaitable, Callable, Mapping
import copy
from dataclasses import replace
from typing import Any, Literal
from uuid import UUID, uuid4

from jsonpatch import JsonPatchException, apply_patch
from jsonpointer import JsonPointer, JsonPointerException
from pydantic import ValidationError

from chartreux.core.config._catalog import (
    CATALOG_DEFINITION_FIELDS,
    validate_catalog_scope,
)
from chartreux.core.config._credential_authority import (
    CREDENTIAL_ENV_FIELD,
    validate_credential_env_source,
)
from chartreux.core.config._restrictions import ConfigCandidate, SourceRestrictions
from chartreux.core.config._root_authority import ROOTS_FIELD, validate_root_source
from chartreux.core.config._source_validation import validate_source
from chartreux.core.config.builder import ConfigBuilder
from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
from chartreux.core.config.event_bus import EventBus
from chartreux.core.config.layer import ConfigLayer, LayerNotLoadedError, RawConfig
from chartreux.core.config.layers._base import BaseTomlConfigLayer
from chartreux.core.config.layers.project import ProjectConfigLayer
from chartreux.core.config.layers.user import UserConfigLayer
from chartreux.core.config.patch import (
    AddOperationPatch,
    ConfigPatch,
    PatchOp,
    ensure_parent_paths,
    resolve_upsert_op,
)
from chartreux.core.config.schema import ConfigSchema
from chartreux.core.config.types import (
    ConcurrencyConflictError,
    ConfigChangeCallback,
    ConfigChangeEvent,
    ConfigDurabilityError,
    ConfigSaveResult,
    ConflictStrategy,
)
from chartreux.core.model_catalog.availability import AvailabilityRegistry
from chartreux.core.model_catalog.loader import CatalogSnapshot, load_catalog
from chartreux.core.utils.concurrency import run_sync


class ConfigPatchValidationError(Exception):
    """Raised when the merged-config preflight rejects a patch."""

    def __init__(self, detail: str | None = None) -> None:
        super().__init__(
            detail
            or "Config patch failed preflight validation against the merged config; "
            "fix the patch payload and retry"
        )


def _safe_validation_detail(exc: ValidationError) -> str | None:
    errors = exc.errors(include_input=False, include_context=False)
    if not errors or any(
        error["type"]
        not in {"unknown_thinking_override", "catalog_scope", "legacy_catalog"}
        for error in errors
    ):
        return None
    return "; ".join(str(error["msg"]) for error in errors)


class DefaultLayerResolutionError(Exception):
    """Raised when a patch needs implicit routing but no valid default is available."""


type DefaultLayerResolver = Callable[[], ConfigLayer[RawConfig]]


class ConfigOrchestrator[S: ConfigSchema]:  # noqa: PLR0904
    """Single entry point for config management."""

    def __init__(
        self,
        builder: ConfigBuilder[S],
        candidate: ConfigCandidate[S],
        default_layer_resolver: DefaultLayerResolver,
        bus: EventBus | None = None,
        catalog_loader: Callable[[], CatalogSnapshot] | None = None,
        availability_registry: AvailabilityRegistry | None = None,
    ) -> None:
        self._builder = builder
        self._snapshot = candidate
        self._builder.accepted_root_sources = candidate.restrictions
        self._default_layer_resolver = default_layer_resolver
        self._bus = bus if bus is not None else EventBus()
        self._catalog_loader = catalog_loader
        self._availability_registry = availability_registry or AvailabilityRegistry()
        self._mutation_lock = asyncio.Lock()
        self._accepted_token = uuid4()
        self._can_persist = True

    @property
    def availability_registry(self) -> AvailabilityRegistry:
        """Root-owned deployment availability state shared by all copies."""
        return self._availability_registry

    @property
    def accepted_token(self) -> object:
        """Opaque, instance-local accepted revision (including ordinary edits)."""
        return self._accepted_token

    async def _prepare_root_replacement(
        self, *, source: str, roots: dict[str, list[str]], expected_token: object
    ) -> ConfigOrchestrator[S]:
        """Prepare a session-only root overlay on an accepted actual user source.

        The caller must prepare runtime/tree authority before using the existing
        ``_commit_policy_replacement`` API. Names select sources, not privileges.
        No installed user source is synthesized, and inherited ownership is not
        claimed. An installed user layer with an absent file is valid.
        """
        async with self._mutation_lock:
            self._check_policy_token(expected_token)
            current = next(
                (item for item in self.restrictions if item.layer_name == source), None
            )
            layer = next((item for item in self.layers if item.name == source), None)
            if (
                current is None
                or current.kind != "source"
                or type(layer) is not UserConfigLayer
                or current.locator != layer.source_locator
            ):
                raise ValueError(
                    "Root replacement requires an accepted actual user source"
                )
            if (
                current.identity is None
                or current.identity.owner != self.policy_owner
                or current.identity != self._builder.source_identities.get(source)
            ):
                raise ValueError(
                    "Root replacement requires the same owned source identity"
                )
            canonical = validate_root_source({ROOTS_FIELD: roots}, layer=layer)
            staged = self.copy()
            replacement = replace(
                current,
                authorized_roots=canonical,
                replaces_roots=True,
                store_fingerprint=None,
            )
            staged._builder.policy_replacements[source] = replacement
            # Only this explicit operation may change the expected root projection.
            # Other sources retain their accepted symlink/identity guards.
            staged._builder.accepted_root_sources = tuple(
                replacement if item.identity == current.identity else item
                for item in self.restrictions
            )
            staged._snapshot = await staged._builder.build_candidate()
            self._check_policy_token(expected_token)
            return staged

    async def _stage_policy_replacement(
        self, *, source: str, tools: dict[str, Any], expected_token: object
    ) -> ConfigOrchestrator[S]:
        """Private session-only staging for the loop; not a generic patch route.

        The caller must prepare all retained runtime authority before committing.
        An empty tools mapping, or explicit empty lists, removes only this source's
        contributions. Profile-local mode belongs to the profile switch lifecycle.
        """
        async with self._mutation_lock:
            self._check_policy_token(expected_token)
            current = next(
                (r for r in self.restrictions if r.layer_name == source), None
            )
            if current is None or current.kind != "source":
                raise ValueError(
                    "Policy replacement requires an accepted non-mode source"
                )
            if current.identity is None:
                raise ValueError(
                    "Policy source provenance unavailable; replacement blocked"
                )
            staged = self.copy()
            replacement = SourceRestrictions.from_raw(
                {"tools": tools},
                root_layer=staged.get_layer(source),
                identity=staged._builder.claim_source(source),
                layer_name=source,
                locator=current.locator,
                kind="source",
                store_fingerprint=None,
                replacement=True,
            )
            previous_override = self._builder.policy_replacements.get(source)
            replacement = replace(
                replacement,
                authorized_roots=current.authorized_roots,
                replaces_roots=bool(
                    previous_override and previous_override.replaces_roots
                ),
            )
            staged._builder.policy_replacements[source] = replacement
            staged._snapshot = await staged._builder.build_candidate()
            self._check_policy_token(expected_token)
            return staged

    async def _stage_policy_refresh(
        self,
        *,
        previous: SourceRestrictions,
        replacement: SourceRestrictions,
        expected_token: object,
    ) -> ConfigOrchestrator[S]:
        """Refresh a copied contribution by identity, never claim child ownership."""
        async with self._mutation_lock:
            self._check_policy_token(expected_token)
            staged = self.copy()
            for source in self.restrictions:
                if source.identity is None:
                    raise ValueError("Policy source provenance unavailable")
                if source.identity != previous.identity:
                    continue
                if source.identity.owner == self.policy_owner:
                    raise ValueError("Cannot refresh an independently owned source")
                if (
                    replacement.identity != previous.identity
                    or replacement.layer_name != previous.layer_name
                    or replacement.locator != previous.locator
                ):
                    raise ValueError("Root refresh source identity changed")
                staged._builder.policy_replacements[source.layer_name] = replacement
                staged._builder.source_identities[source.layer_name] = source.identity
                if replacement.replaces_roots:
                    # Explicit owner replacement alone advances this copied source's
                    # expected projection. All other recanonicalization guards stay.
                    staged._builder.accepted_root_sources = tuple(
                        replacement if item.identity == source.identity else item
                        for item in staged._builder.accepted_root_sources
                    )
            staged._snapshot = await staged._builder.build_candidate()
            self._check_policy_token(expected_token)
            return staged

    def _check_policy_commit(
        self, expected_token: object, staged: ConfigOrchestrator[S]
    ) -> None:
        self._check_policy_token(expected_token)
        if self._mutation_lock.locked():
            raise RuntimeError("Configuration mutation is in progress")
        live_layers = [(type(layer), layer.name) for layer in self.layers]
        staged_layers = [(type(layer), layer.name) for layer in staged.layers]
        if live_layers != staged_layers or self.policy_owner != staged.policy_owner:
            raise ValueError("Stale policy layer preparation")

    def _check_policy_token(self, expected_token: object) -> None:
        if expected_token is not self._accepted_token:
            raise ValueError("Stale accepted policy revision")

    def _commit_policy_replacement(
        self, staged: ConfigOrchestrator[S], *, expected_token: object
    ) -> None:
        """Synchronous core commit after whole-tree validation by the server."""
        self._check_policy_commit(expected_token, staged)
        # Policy edits are session-only overlays, not source reloads. Keep live
        # layer caches/identities intact: invoking layer adoption here could fail
        # after another tree participant has already published its authority.
        self._builder.policy_replacements = staged._builder.policy_replacements
        self._builder.source_identities = staged._builder.source_identities
        self._snapshot = staged._snapshot
        self._builder.accepted_root_sources = staged._snapshot.restrictions
        self._accepted_token = staged._accepted_token

    def copy(self, *, config: S | None = None) -> ConfigOrchestrator[S]:
        """Return an independent in-memory copy of this orchestrator.

        The builder and its layers are deep-copied so writes on the copy never
        touch the original. The default-layer resolver is rebound to the copied
        layers, and the copy starts with a fresh event bus so it does not
        inherit the original's subscribers. An already validated merged config
        may be supplied for consumers that need to stage against a pending
        snapshot while preserving the accepted source restrictions.
        """
        builder = self._builder.copy()
        default_layer_name = self._default_layer_resolver().name
        layers_by_name = {layer.name: layer for layer in builder.layers}
        snapshot = copy.deepcopy(self._snapshot)
        catalog_snapshot = (
            config.catalog_snapshot
            if isinstance(config, ChartreuxConfigSchema)
            else (
                self._snapshot.config.catalog_snapshot
                if isinstance(self._snapshot.config, ChartreuxConfigSchema)
                else None
            )
        )
        if config is not None:
            snapshot = replace(snapshot, config=config)
        if (
            isinstance(snapshot.config, ChartreuxConfigSchema)
            and catalog_snapshot is not None
        ):
            snapshot.config.attach_catalog_snapshot(catalog_snapshot)
        copied = type(self)(
            builder,
            snapshot,
            lambda: layers_by_name[default_layer_name],
            bus=None,
            catalog_loader=self._catalog_loader,
            availability_registry=self._availability_registry,
        )
        copied._can_persist = self._can_persist
        return copied

    @property
    def policy_owner(self) -> UUID:
        """In-memory writer identity, preserved by staging/fork copies."""
        return self._builder.policy_owner

    def _copy_for_child(self) -> ConfigOrchestrator[S]:
        """Factory-only new writer; copied source authority retains its owner.

        Profile installation and explicit source replacement claim fresh child
        contributions. Copying settings (including session logging edits) does not.
        """
        child = self.copy()
        child._builder.policy_owner = uuid4()
        child._can_persist = False
        return child

    @classmethod
    async def create(
        cls,
        *,
        schema: type[S],
        layers: list[ConfigLayer[RawConfig]],
        default_layer_resolver: DefaultLayerResolver,
        bus: EventBus | None = None,
        catalog_snapshot: CatalogSnapshot | None = None,
        catalog_loader: Callable[[], CatalogSnapshot] | None = None,
    ) -> ConfigOrchestrator[S]:
        """Build an orchestrator from a schema and an ordered list of layers."""
        if catalog_snapshot is None and issubclass(schema, ChartreuxConfigSchema):
            catalog_snapshot = load_catalog()
        builder = ConfigBuilder[S](schema, catalog_snapshot=catalog_snapshot)
        builder.add_layers(layers)
        candidate = await builder.build_candidate()
        return cls(
            builder,
            candidate,
            default_layer_resolver,
            bus,
            catalog_loader=catalog_loader,
        )

    @property
    def config(self) -> S:
        return self._snapshot.config

    @property
    def restrictions(self) -> tuple[SourceRestrictions, ...]:
        """Source authority from the same accepted snapshot as ordinary config."""
        return self._snapshot.restrictions

    async def preview_candidate(
        self,
        *,
        force_load: bool = False,
        layer_overrides: Mapping[str, RawConfig] | None = None,
    ) -> ConfigCandidate[S]:
        """Prepare a config/policy candidate without publishing or changing caches.

        This is not a policy edit or acceptance operation. Runtime enforcement
        must not read from a preview; accepted-snapshot wiring is separate.
        """
        async with self._mutation_lock:
            builder = self._builder.copy()
            return await builder.build_candidate(
                force_load=force_load, layer_overrides=layer_overrides
            )

    def rebuild(self) -> None:
        """Re-merge the layer stack synchronously and install the result."""
        builder = self._builder.copy()
        candidate = run_sync(builder.build_candidate())
        self._accept_candidate(builder, candidate)

    def _accept_candidate(
        self, builder: ConfigBuilder[S], candidate: ConfigCandidate[S]
    ) -> None:
        # Preserve layer identity: resolvers and runtime workspace owners retain
        # references to installed layers. Only adopt staged caches after all
        # validation/preparation succeeds; publication itself has no await.
        for live, staged in zip(self._builder.layers, builder.layers, strict=True):
            live._accept_loaded_state(staged)
        self._builder.policy_replacements = builder.policy_replacements.copy()
        self._builder.source_identities = builder.source_identities.copy()
        self._builder.catalog_snapshot = builder.catalog_snapshot
        self._snapshot = candidate
        self._builder.accepted_root_sources = candidate.restrictions
        self._accepted_token = uuid4()

    @property
    def layers(self) -> tuple[ConfigLayer[RawConfig], ...]:
        """Active layers, lowest to highest priority. Read-only view."""
        return tuple(self._builder.layers)

    @property
    def writable_layer_name(self) -> str:
        """Name of the layer that implicit writes are routed to."""
        return self._resolve_default_layer_name()

    def get_layer(self, name: str) -> ConfigLayer[RawConfig]:
        for layer in self._builder.layers:
            if layer.name == name:
                return layer
        raise KeyError(f"No layer named {name!r}")

    def insert_layer(self, layer: ConfigLayer[RawConfig], index: int) -> None:
        """Insert a layer at *index* (0 = lowest priority). Rebuild to apply."""
        self._builder.insert_layer(layer, index)

    def remove_layer(self, index: int) -> ConfigLayer[RawConfig]:
        """Remove and return the layer at *index*. Rebuild to apply."""
        return self._builder.remove_layer(index)

    def replace_or_append_layer(self, name: str, layer: ConfigLayer[RawConfig]) -> None:
        """Replace the layer named *name* in place, or append it when absent."""
        index = next(
            (i for i, existing in enumerate(self.layers) if existing.name == name), None
        )
        if index is None:
            self.insert_layer(layer, len(self.layers))
            return
        self.remove_layer(index)
        self.insert_layer(layer, index)

    async def load_persistence_layer(
        self, *, target_layer: str | None = None
    ) -> RawConfig:
        layer = (
            self.get_layer(target_layer)
            if target_layer is not None
            else self._default_layer_resolver()
        )
        # Inspection must not replace an accepted source cache behind the runtime.
        return await copy.deepcopy(layer).load()

    def persisted_active_model(self) -> str:
        data = self._default_layer_resolver().cached_data
        if data is None:
            return ""
        value = getattr(data, "active_model", "")
        return value if isinstance(value, str) else ""

    async def reload(
        self,
        *,
        preflight: Callable[[S], Awaitable[None]] | None = None,
        apply: Callable[[S], None] | None = None,
    ) -> None:
        """Force-reload all layers and atomically replace the config snapshot."""
        async with self._mutation_lock:
            await self._reload_locked(preflight=preflight, apply=apply)

    async def _reload_locked(
        self,
        *,
        preflight: Callable[[S], Awaitable[None]] | None = None,
        apply: Callable[[S], None] | None = None,
    ) -> None:
        builder = self._builder.copy()
        # Catalog authority is staged with settings and becomes visible only at the
        # same synchronous accepted-snapshot publication boundary.
        if self._catalog_loader is not None:
            builder.catalog_snapshot = self._catalog_loader()
        candidate = await builder.build_candidate(force_load=True)
        if preflight is not None:
            await preflight(candidate.config)
        if apply is not None:
            apply(candidate.config)
        self._accept_candidate(builder, candidate)

    async def save(
        self,
        operations: list[PatchOp],
        *,
        target: Literal["user", "project"],
        expected_revision: str,
        reason: str,
        preflight: Callable[[ConfigCandidate[S]], Awaitable[None]] | None = None,
        apply: Callable[[ConfigCandidate[S]], None] | None = None,
    ) -> ConfigSaveResult:
        """Prepare and save exactly one source; never mirror it into session state.

        The server must reserve the idle session tree through this entire call.
        ``preflight`` prepares consumers without mutation. ``apply`` publishes
        prepared consumers synchronously and must retain old state on failure.
        Disk is never rolled back after replacement. Values are deliberately not
        included in this result: delivery adapters own their redacted projection.
        """
        async with self._mutation_lock:
            try:
                operations = self._canonicalize_thinking_operations(operations)
                if not self._can_persist or not expected_revision:
                    raise ValueError(
                        "Explicit source revision and root session required"
                    )
                layer_type = {"user": UserConfigLayer, "project": ProjectConfigLayer}[
                    target
                ]
                layers = [
                    layer for layer in self.layers if isinstance(layer, layer_type)
                ]
                if len(layers) != 1:
                    raise ValueError("Save requires one installed target")
                layer = layers[0]
                assert isinstance(layer, (UserConfigLayer, ProjectConfigLayer))
                if any(
                    op.target_layer_name not in {None, layer.name} for op in operations
                ):
                    raise ValueError("Mixed save targets are not supported")
                routed = [
                    op.model_copy(update={"target_layer_name": layer.name})
                    for op in operations
                ]
                staged = self.copy()
                staged_layer = staged.get_layer(layer.name)
                await staged_layer.load(force=True)
                if staged_layer.fingerprint != expected_revision:
                    return ConfigSaveResult(
                        target, "not_saved", "unchanged", error="conflict"
                    )
                staged._validate_catalog_patch_scope(routed)
                await staged._preview_patch(routed)
                raw = (await staged_layer.load()).model_dump()
                patched = staged_layer.validate_output(
                    apply_patch(
                        ensure_parent_paths(raw, routed),
                        [op.to_json_patch() for op in routed],
                        in_place=False,
                    )
                )
                staged_layer._stage_loaded_data(patched, expected_revision)
                candidate = await staged._builder.build_candidate()
                if preflight is not None:
                    await preflight(candidate)
            except Exception:
                return ConfigSaveResult(
                    target, "not_saved", "unchanged", error="validation"
                )

            persistence: Literal["saved", "durability_uncertain"] = "saved"
            # Once admitted, the writer must finish before either lock or the
            # caller's runtime reservation is released. Cancellation prevents
            # publication, not an already-running filesystem replacement.
            writer = asyncio.create_task(
                layer.save_checked(patched, expected_revision=expected_revision)
            )
            cancelled = False
            try:
                while True:
                    try:
                        revision = await asyncio.shield(writer)
                        break
                    except asyncio.CancelledError:
                        cancelled = True
                        if writer.cancelled():
                            raise
            except ConcurrencyConflictError:
                return ConfigSaveResult(
                    target, "not_saved", "unchanged", error="conflict"
                )
            except ConfigDurabilityError as exc:
                revision = exc.revision
                persistence = "durability_uncertain"
            except Exception:
                return ConfigSaveResult(target, "not_saved", "unchanged", error="write")

            if cancelled:
                return ConfigSaveResult(
                    target, persistence, "unchanged", revision, "cancelled"
                )
            assert isinstance(staged_layer, BaseTomlConfigLayer)
            staged_layer._stage_saved_data(patched, revision)
            return await self._publish_saved_candidate(
                staged,
                result=ConfigSaveResult(target, persistence, "unchanged", revision),
                reason=reason,
                apply=apply,
            )

    async def _publish_saved_candidate(
        self,
        staged: ConfigOrchestrator[S],
        *,
        result: ConfigSaveResult,
        reason: str,
        apply: Callable[[ConfigCandidate[S]], None] | None,
    ) -> ConfigSaveResult:
        """Finish a replaced source while the caller still holds mutation admission."""
        try:
            candidate = await staged._builder.build_candidate()
            if apply is not None:
                apply(candidate)
        except asyncio.CancelledError:
            return replace(result, error="cancelled")
        except Exception:
            return replace(result, application="failed", error="application")
        before = self.config.model_dump(mode="json")
        self._accept_candidate(staged._builder, candidate)
        after = self.config.model_dump(mode="json")
        result = replace(result, application="applied")
        try:
            self._bus.publish(
                ConfigChangeEvent(
                    changed_keys=_changed_keys_between(before, after),
                    before=before,
                    after=after,
                    reason=reason,
                )
            )
        except (Exception, asyncio.CancelledError):
            # Subscribers observe an already accepted commit. A failed
            # notification cannot undo it or turn it into a failed save.
            return replace(result, error="notification")
        return result

    async def set_field(
        self,
        path: str,
        value: Any,
        reason: str = "No reason",
        *,
        target_layer: str | None = None,
        preflight: Callable[[S], Awaitable[None]] | None = None,
    ) -> list[BaseException]:
        return await self.apply_patch(
            [AddOperationPatch(path=path, value=value, target_layer_name=target_layer)],
            reason=reason,
            preflight=preflight,
        )

    async def mutate_field(
        self,
        path: str,
        mutate: Callable[[Any], Any],
        reason: str = "No reason",
        *,
        default: Any = None,
        target_layer: str | None = None,
        preflight: Callable[[S], Awaitable[None]] | None = None,
    ) -> list[BaseException]:
        """Read, transform, preflight, and persist one field under the mutation lock."""
        async with self._mutation_lock:
            layer_name = target_layer or self._resolve_default_layer_name()
            try:
                raw = (await self.get_layer(layer_name).load()).model_dump()
            except KeyError as exc:
                return [exc]
            current = JsonPointer(path).resolve(raw, default=copy.deepcopy(default))
            value = mutate(copy.deepcopy(current))
            return await self._apply_patch_locked(
                [
                    AddOperationPatch(
                        path=path, value=value, target_layer_name=layer_name
                    )
                ],
                reason,
                on_conflict=ConflictStrategy.CANCEL,
                preflight=preflight,
            )

    async def upsert_field(
        self,
        path: str,
        *,
        key_field: str,
        value: dict[str, Any],
        reason: str = "No reason",
        target_layer: str | None = None,
    ) -> list[BaseException]:
        """Insert or replace one entry in a persisted config list section.

        *path* is a JSON Pointer to the list field (e.g. ``/providers``);
        *key_field* identifies an entry within that list (e.g. ``name``).
        When an entry with the same key already exists it is replaced in
        place, otherwise the value is appended (or the section is created
        when empty).
        """
        async with self._mutation_lock:
            layer_name = target_layer or self._resolve_default_layer_name()
            raw: dict[str, Any] = (await self.get_layer(layer_name).load()).model_dump()
            existing = JsonPointer(path).resolve(raw, default=[])
            operation = resolve_upsert_op(
                existing, path, key_field, value, target_layer_name=layer_name
            )
            return await self._apply_patch_locked(
                [operation], reason, on_conflict=ConflictStrategy.CANCEL, preflight=None
            )

    async def apply_session_patch(
        self,
        operations: list[PatchOp],
        *,
        reason: str,
        preflight: Callable[[S], Awaitable[None]],
        apply: Callable[[S], None],
    ) -> list[BaseException]:
        """Publish session-only edits, then notify the original bus.

        Cancellation during preparation leaves accepted state unchanged. Once
        synchronous publication starts, subscriber cancellation or failure cannot
        undo the edit or report it as rejected.
        """
        if any(op.target_layer_name not in {None, "overrides"} for op in operations):
            raise ValueError("Session writes require the session target")
        async with self._mutation_lock:
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
            candidate = await staged.preview_candidate()
            await preflight(candidate.config)
            before = self.config.model_dump(mode="json")
            apply(candidate.config)
            self._accept_candidate(staged._builder, candidate)
            after = self.config.model_dump(mode="json")
            if changed_keys := _changed_keys_between(before, after):
                try:
                    self._bus.publish(
                        ConfigChangeEvent(
                            changed_keys=changed_keys,
                            before=before,
                            after=after,
                            reason=reason,
                        )
                    )
                except (Exception, asyncio.CancelledError):
                    # Notification is observational: publication already succeeded.
                    pass
        return []

    async def apply_patch(
        self,
        operations: list[PatchOp],
        reason: str,
        *,
        on_conflict: ConflictStrategy = ConflictStrategy.CANCEL,
        preflight: Callable[[S], Awaitable[None]] | None = None,
    ) -> list[BaseException]:
        """Apply patch operations layer by layer.

        The merged-config preflight is a cheap sanity check only. Once patching
        begins, writes are not atomic across layers. Invalid patch requests
        still raise, but per-layer write failures are returned in the result.
        """
        if not operations:
            return []

        async with self._mutation_lock:
            return await self._apply_patch_locked(
                operations, reason, on_conflict=on_conflict, preflight=preflight
            )

    async def _apply_patch_locked(
        self,
        operations: list[PatchOp],
        reason: str,
        *,
        on_conflict: ConflictStrategy,
        preflight: Callable[[S], Awaitable[None]] | None,
    ) -> list[BaseException]:
        operations = self._canonicalize_thinking_operations(operations)
        if not self._can_persist and any(
            isinstance(
                self.get_layer(
                    op.target_layer_name or self._resolve_default_layer_name()
                ),
                BaseTomlConfigLayer,
            )
            for op in operations
        ):
            return [ValueError("Child sessions cannot persist configuration")]
        self._validate_catalog_patch_scope(operations)
        self._validate_patch_shape(operations)
        try:
            candidate = await self._preview_patch(operations)
        except (KeyError, LayerNotLoadedError) as exc:
            detail = self.thinking_patch_failure_detail([
                (op.path, op.target_layer_name) for op in operations
            ])
            return [ConfigPatchValidationError(detail)] if detail else [exc]
        if preflight is not None:
            await preflight(candidate)

        before = self.config.model_dump(mode="json")

        operations_by_layer: dict[str, list[PatchOp]] = defaultdict(list)
        default_layer_name: str | None = None
        for op in operations:
            layer_name = op.target_layer_name
            if layer_name is None:
                if default_layer_name is None:
                    default_layer_name = self._resolve_default_layer_name()
                layer_name = default_layer_name

            operations_by_layer[layer_name].append(op)

        tasks = []
        for layer_name, layer_operations in operations_by_layer.items():
            tasks.append(
                asyncio.create_task(
                    self._apply_patch_to_layer(
                        layer_name=layer_name,
                        layer_operations=list(layer_operations),
                        reason=reason,
                        on_conflict=on_conflict,
                    )
                )
            )
        results = await asyncio.gather(*tasks, return_exceptions=True)
        failures: list[BaseException] = [
            r for r in results if isinstance(r, BaseException)
        ]
        has_success = any(not isinstance(r, BaseException) for r in results)
        if failures and (
            detail := self.thinking_patch_failure_detail([
                (op.path, op.target_layer_name) for op in operations
            ])
        ):
            failures = [ConfigPatchValidationError(detail) for _ in failures]
        if not has_success:
            return failures

        # Ordinary edits rebuild from accepted caches plus the changed layers.
        # Only explicit reload may accept unrelated backing-source changes.
        builder = self._builder.copy()
        rebuilt = await builder.build_candidate()
        self._accept_candidate(builder, rebuilt)
        after = self.config.model_dump(mode="json")
        changed_keys = _changed_keys_between(before, after)
        if has_success and changed_keys:
            self._bus.publish(
                ConfigChangeEvent(
                    changed_keys=changed_keys, before=before, after=after, reason=reason
                )
            )

        return failures

    def _canonicalize_thinking_operations(
        self, operations: list[PatchOp]
    ) -> list[PatchOp]:
        """Persist catalog thinking overrides under canonical base-model keys."""
        if not isinstance(self.config, ChartreuxConfigSchema):
            return operations
        snapshot = self.config.catalog_snapshot
        if snapshot is None:
            return operations
        from chartreux.core.model_catalog.resolver import ModelResolver

        resolver = ModelResolver(snapshot)
        canonical_paths: set[str] = set()
        normalized: list[PatchOp] = []
        for operation in operations:
            parts = JsonPointer(operation.path).parts
            value = operation.to_json_patch().get("value")
            if tuple(parts) == ("thinking_overrides",) and isinstance(value, dict):
                value = resolver.canonicalize_thinking_overrides(value)
                operation = operation.model_copy(update={"value": value})
            elif len(parts) > 1 and parts[0] == "thinking_overrides":
                base = resolver.canonicalize(parts[1])
                path = "/thinking_overrides/" + base.replace("~", "~0").replace(
                    "/", "~1"
                )
                if path in canonical_paths:
                    raise ValueError(f"Multiple thinking overrides resolve to {base!r}")
                canonical_paths.add(path)
                operation = operation.model_copy(update={"path": path})
            normalized.append(operation)
        return normalized

    def thinking_patch_failure_detail(
        self, paths: list[tuple[str, str | None]]
    ) -> str | None:
        """Bounded source/field diagnostics, never JSON pointers or patch values."""
        if not any(
            path in {"", "/thinking_overrides"}
            or path.startswith("/thinking_overrides/")
            for path, _ in paths
        ):
            return None
        sources = {layer.name for layer in self.layers}
        fields = type(self.config).model_fields
        labels: set[str] = set()
        for path, target in paths:
            source = target or self.writable_layer_name
            source = source if source in sources else "unknown"
            field = (
                path.split("/", 2)[1] if path.startswith("/") else "thinking_overrides"
            )
            field = field if field in fields else "configuration"
            labels.add(f"field '{field}' in source '{source}'")
        return "Config patch failed for " + "; ".join(sorted(labels))

    def _validate_catalog_patch_scope(self, operations: list[PatchOp]) -> None:
        if not isinstance(self.config, ChartreuxConfigSchema):
            return
        # Authorize the operation, not just its resulting document: removals and
        # set-then-remove batches must not erase evidence of a forbidden write.
        for operation in operations:
            try:
                layer = self.get_layer(
                    operation.target_layer_name or self._resolve_default_layer_name()
                )
            except (KeyError, DefaultLayerResolutionError):
                continue  # Existing routing diagnostics handle unknown targets.
            try:
                parts = JsonPointer(operation.path).parts
            except JsonPointerException:
                continue  # Shape validation handles malformed pointers.
            fields = set(parts[:1])
            if not parts:
                if layer.cached_data is not None:
                    fields.update(layer.cached_data.model_dump())
                value = operation.to_json_patch().get("value")
                if isinstance(value, dict):
                    fields.update(value)
            if CREDENTIAL_ENV_FIELD in fields:
                # Generic patches (including client config_write with a user target)
                # cannot prove user intent. Edit the user file directly instead.
                raise ConfigPatchValidationError(
                    f"Field '{CREDENTIAL_ENV_FIELD}' requires an actual user source "
                    "edited outside generic config writes"
                )
            if ROOTS_FIELD in fields:
                source_label = "user" if type(layer) is UserConfigLayer else "non-user"
                raise ConfigPatchValidationError(
                    f"Field '{ROOTS_FIELD}' in source '{source_label}' requires "
                    "explicit root policy replacement"
                )
            try:
                validate_catalog_scope(
                    type(self.config),
                    fields,
                    layer=layer,
                    source=f"{layer.name} ({layer.source_locator})",
                )
            except ValidationError as exc:
                detail = _safe_validation_detail(exc)
            else:
                continue
            raise ConfigPatchValidationError(detail) from None

    def _validate_patch_shape(self, operations: list[PatchOp]) -> None:
        if isinstance(self.config, ChartreuxConfigSchema):
            # Chartreux validates every prospective sparse source in the preview.
            # Applying these operations to the effective document instead would
            # mistake sparse catalog replacements (including {}) for full ones.
            return
        try:
            self._builder.validate(
                apply_patch(
                    ensure_parent_paths(self.config.model_dump(), operations),
                    patch=[operation.to_json_patch() for operation in operations],
                    in_place=False,
                )
            )
        except (JsonPatchException, JsonPointerException, ValidationError) as exc:
            if isinstance(exc, ValidationError):
                detail = _safe_validation_detail(exc)
                if detail is not None:
                    # Preserve source provenance from the merged candidate preview.
                    return
            detail = self.thinking_patch_failure_detail([
                (op.path, op.target_layer_name) for op in operations
            ])
            if detail is None:
                raise ConfigPatchValidationError() from exc
        else:
            return
        raise ConfigPatchValidationError(detail) from None

    async def _preview_patch(self, operations: list[PatchOp]) -> S:
        operations_by_layer: dict[str, list[PatchOp]] = defaultdict(list)
        default_layer_name: str | None = None
        for operation in operations:
            layer_name = operation.target_layer_name
            if layer_name is None:
                if default_layer_name is None:
                    default_layer_name = self._resolve_default_layer_name()
                layer_name = default_layer_name
            operations_by_layer[layer_name].append(operation)

        overrides: dict[str, RawConfig] = {}
        try:
            for layer_name, layer_operations in operations_by_layer.items():
                layer = self.get_layer(layer_name)
                if layer.cached_data is None or layer.fingerprint is None:
                    raise LayerNotLoadedError(layer_name)
                raw = layer.cached_data.model_dump()
                patched = apply_patch(
                    ensure_parent_paths(raw, layer_operations),
                    patch=[operation.to_json_patch() for operation in layer_operations],
                    in_place=False,
                )
                overrides[layer_name] = layer.validate_output(patched)
                if isinstance(self.config, ChartreuxConfigSchema):
                    validate_credential_env_source(raw, layer=layer)
                    validate_credential_env_source(patched, layer=layer)
                    validate_root_source(raw, layer=layer)
                    validate_root_source(patched, layer=layer)
                    validate_source(
                        type(self.config),
                        patched,
                        source=f"{layer.name} ({layer.source_locator})",
                    )
                source = next(
                    (r for r in self.restrictions if r.layer_name == layer_name), None
                )
                if source is not None and source.kind == "source":
                    before_policy = SourceRestrictions.from_raw(
                        raw,
                        root_layer=layer,
                        layer_name=layer_name,
                        locator=source.locator,
                        kind=source.kind,
                        store_fingerprint=layer.fingerprint,
                    )
                    after_policy = SourceRestrictions.from_raw(
                        patched,
                        root_layer=layer,
                        layer_name=layer_name,
                        locator=source.locator,
                        kind=source.kind,
                        store_fingerprint=None,
                    )
                    if (
                        source.identity is not None
                        and source.identity.owner != self.policy_owner
                        and after_policy.tools
                        and any(
                            op.path in {"", "/tools"} or op.path.startswith("/tools/")
                            for op in layer_operations
                        )
                    ):
                        # A merged copied layer cannot prove which restrictions
                        # this writer intended to assert, even for equal values.
                        raise ValueError(
                            "Copied source policy requires explicit policy replacement"
                        )
                    if (
                        layer_name in self._builder.policy_replacements
                        and after_policy.tools != before_policy.tools
                    ):
                        raise ValueError(
                            "Session-replaced source policy requires explicit replacement"
                        )
                    replacement = self._builder.policy_replacements.get(layer_name)
                    projected_roots = (
                        validate_root_source(
                            replacement.replace_in(patched), layer=layer
                        )
                        if replacement is not None
                        else after_policy.authorized_roots
                    )
                    if (
                        not after_policy.preserves(before_policy)
                        or projected_roots != source.authorized_roots
                    ):
                        raise ValueError(
                            "Generic config edits cannot weaken source policy; "
                            "explicit policy replacement is required"
                        )
            candidate = await self._builder.copy().build_candidate(
                layer_overrides=overrides
            )
        except (JsonPatchException, JsonPointerException, ValidationError) as exc:
            detail = (
                _safe_validation_detail(exc)
                if isinstance(exc, ValidationError)
                else None
            )
            detail = detail or self.thinking_patch_failure_detail([
                (op.path, op.target_layer_name) for op in operations
            ])
            if detail is None and isinstance(self.config, ChartreuxConfigSchema):
                labels = {
                    f"field '{parts[0]}' in source '{op.target_layer_name or self.writable_layer_name}'"
                    for op in operations
                    if (parts := JsonPointer(op.path).parts)
                    and parts[0] in CATALOG_DEFINITION_FIELDS
                }
                if labels:
                    detail = "Config patch failed for " + "; ".join(sorted(labels))
            if detail is None:
                raise ConfigPatchValidationError() from exc
        else:
            return candidate.config
        raise ConfigPatchValidationError(detail) from None

    async def _apply_patch_to_layer(
        self,
        *,
        layer_name: str,
        layer_operations: list[PatchOp],
        reason: str,
        on_conflict: ConflictStrategy,
    ) -> None:
        layer = self.get_layer(layer_name)
        if not self._can_persist and isinstance(layer, BaseTomlConfigLayer):
            raise ValueError("Child sessions cannot persist configuration")
        if layer.fingerprint is None:
            raise LayerNotLoadedError(layer_name)
        await layer.apply(
            ConfigPatch(
                *layer_operations, fingerprint=layer.fingerprint, reason=reason
            ),
            on_conflict=on_conflict,
        )

    def _resolve_default_layer_name(self) -> str:
        layer = self._default_layer_resolver()
        if layer not in self._builder.layers:
            raise DefaultLayerResolutionError(
                f"Default layer resolver returned unknown layer {layer.name!r}"
            )
        return layer.name

    def subscribe(
        self, callback: ConfigChangeCallback, *, keys: set[str] | None = None
    ) -> Callable[[], None]:
        """Register a listener and return a callable that unsubscribes it.

        Args:
            callback: Invoked with the event on every matching config change.
            keys: Slash-separated config paths to filter on (e.g. {"models/active"}).
                A path matches its ancestors and descendants but not partial
                segments ("model" never matches "models"). None subscribes to
                every change (wildcard).
        """
        return self._bus.subscribe(callback, keys=keys)


_MISSING = object()


def _changed_keys_between(
    before: dict[str, Any], after: dict[str, Any]
) -> frozenset[str]:
    changed: set[str] = set()
    _collect_changed_keys(before, after, (), changed)
    return frozenset(changed)


def _collect_changed_keys(
    before: Any, after: Any, path: tuple[str, ...], changed: set[str]
) -> None:
    if before == after:
        return

    if isinstance(before, dict) and isinstance(after, dict):
        for key in sorted(before.keys() | after.keys()):
            _collect_changed_keys(
                before.get(key, _MISSING),
                after.get(key, _MISSING),
                (*path, key),
                changed,
            )
        return

    changed.add("/".join(path))
