from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Callable, Mapping
import copy
from dataclasses import dataclass
from typing import Any, cast
from uuid import uuid4

from pydantic import BeforeValidator
from pydantic.fields import FieldInfo

from chartreux.core.config._catalog import validate_catalog_scope
from chartreux.core.config._credential_authority import validate_credential_env_source
from chartreux.core.config._restrictions import (
    ConfigCandidate,
    PolicySourceIdentity,
    SourceRestrictions,
)
from chartreux.core.config._root_authority import validate_root_source
from chartreux.core.config._source_validation import validate_source
from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
from chartreux.core.config.layer import (
    ConfigLayer,
    EmptyLayerError,
    RawConfig,
    UntrustedLayerError,
)
from chartreux.core.config.layers.agent_profile import AgentProfileLayer
from chartreux.core.config.layers.launch_overrides import LaunchOverridesLayer
from chartreux.core.config.schema import (
    ConfigFragment,
    ConfigSchema,
    MergeFieldMetadata,
)
from chartreux.core.model_catalog.loader import CatalogSnapshot
from chartreux.core.utils.merge import MergeStrategy


class ConfigMergeError(ValueError):
    def __init__(
        self, field_name: str, layer_name: str, expected_type: str, value: Any
    ) -> None:
        actual_type = "dictionary" if isinstance(value, dict) else type(value).__name__
        message = (
            f"Invalid configuration: {field_name} from {layer_name} must be a "
            f"{expected_type}, not a {actual_type}."
        )
        if field_name == "mcp_servers" and isinstance(value, dict):
            message += " Use [[mcp_servers]] instead of [mcp_servers.<name>]."
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class _LayerData:
    name: str
    data: dict[str, Any]


class ConfigBuilder[S: ConfigSchema]:
    """Collects layers and merges them into an immutable Config[S]."""

    def __init__(
        self, schema: type[S], *, catalog_snapshot: CatalogSnapshot | None = None
    ) -> None:
        self._schema = schema
        self.catalog_snapshot = catalog_snapshot
        self._layers: list[ConfigLayer[RawConfig]] = []
        self._lock = asyncio.Lock()
        # Session-only, source-targeted replacements; never persisted to layers.
        self.policy_replacements: dict[str, SourceRestrictions] = {}
        self.policy_owner = uuid4()
        self.source_identities: dict[str, PolicySourceIdentity] = {}
        self.accepted_root_sources: tuple[SourceRestrictions, ...] = ()

    def claim_source(self, name: str) -> PolicySourceIdentity:
        """Explicit source installation/write, never called by candidate previews."""
        identity = self.source_identities.get(name)
        if identity is None or identity.owner != self.policy_owner:
            identity = PolicySourceIdentity(self.policy_owner)
            self.source_identities[name] = identity
        return identity

    def add_layer(self, layer: ConfigLayer[RawConfig]) -> None:
        self.claim_source(layer.name)
        self._layers.append(layer)

    def add_layers(self, layers: list[ConfigLayer[RawConfig]]) -> None:
        for layer in layers:
            self.add_layer(layer)

    def insert_layer(self, layer: ConfigLayer[RawConfig], index: int) -> None:
        self.claim_source(layer.name)
        self._layers.insert(index, layer)

    def remove_layer(self, index: int) -> ConfigLayer[RawConfig]:
        return self._layers.pop(index)

    @property
    def layers(self) -> list[ConfigLayer[RawConfig]]:
        return self._layers

    def copy(self) -> ConfigBuilder[S]:
        """Return a new builder for the same schema with deep-copied layers."""
        new_builder = ConfigBuilder(
            self._schema, catalog_snapshot=self.catalog_snapshot
        )
        new_builder._layers = [copy.deepcopy(layer) for layer in self._layers]
        new_builder.policy_owner = self.policy_owner
        new_builder.source_identities = self.source_identities.copy()
        new_builder.policy_replacements = self.policy_replacements.copy()
        new_builder.accepted_root_sources = self.accepted_root_sources
        return new_builder

    def validate(self, data: dict[str, Any]) -> S:
        return self._schema.model_validate(data)

    async def build(
        self,
        force_load: bool = False,
        *,
        layer_overrides: Mapping[str, RawConfig] | None = None,
    ) -> S:
        """Merge all layers and return a validated schema.

        Untrusted and empty layers are skipped.
        Pass ``force_load=True`` to bypass caching.
        ``layer_overrides`` previews already-validated layer values without
        mutating the layer or its backing store.
        """
        if issubclass(self._schema, ChartreuxConfigSchema):
            return (
                await self.build_candidate(force_load, layer_overrides=layer_overrides)
            ).config
        async with self._lock:
            internal_layers = self._layers.copy()
            overrides = layer_overrides or {}

            layer_dicts: list[_LayerData] = []
            for layer in internal_layers:
                try:
                    data = overrides.get(layer.name)
                    if data is None:
                        data = await layer.load(force=force_load)
                    raw = data.model_dump()
                    if raw:
                        layer_dicts.append(_LayerData(name=layer.name, data=raw))
                except (UntrustedLayerError, EmptyLayerError):
                    continue

            merged, origins, thinking_origins = self._merge_fields(
                self._schema, layer_dicts
            )
            config = self._validate_merged(
                merged,
                origins,
                thinking_origins,
                catalog_snapshot=self._catalog_for_layers(layer_dicts),
            )
            for live, staged in zip(self._layers, internal_layers, strict=True):
                live._accept_loaded_state(staged)
            return config

    async def build_candidate(
        self,
        force_load: bool = False,
        *,
        layer_overrides: Mapping[str, RawConfig] | None = None,
    ) -> ConfigCandidate[S]:
        """Prepare config and source restrictions without accepting either.

        Ordinary and restriction projections use the same loaded data, never a
        second source read. Generic schemas need not use Chartreux's tool schema.
        """
        if not issubclass(self._schema, ChartreuxConfigSchema):
            return ConfigCandidate(
                await self.build(force_load, layer_overrides=layer_overrides), ()
            )
        async with self._lock:
            overrides = layer_overrides or {}
            layers = [copy.deepcopy(layer) for layer in self._layers]
            names = [layer.name for layer in layers]
            if len(set(names)) != len(names):
                raise ValueError("Restriction candidates require unique layer names")
            if unknown := overrides.keys() - set(names):
                raise ValueError(
                    f"Unknown candidate layers: {', '.join(sorted(unknown))}"
                )
            layer_dicts: list[_LayerData] = []
            restrictions: list[SourceRestrictions] = []
            for layer in layers:
                try:
                    loaded = await layer.load(force=force_load)
                except (UntrustedLayerError, EmptyLayerError):
                    continue
                validate_credential_env_source(loaded.model_dump(), layer=layer)
                validate_root_source(loaded.model_dump(), layer=layer)
                validate_catalog_scope(
                    self._schema,
                    loaded.model_dump(),
                    layer=layer,
                    source=f"{layer.name} ({layer.source_locator})",
                )
                data = overrides.get(layer.name, loaded)
                raw = data.model_dump()
                validate_credential_env_source(raw, layer=layer)
                validate_root_source(raw, layer=layer)
                validate_catalog_scope(
                    self._schema,
                    raw,
                    layer=layer,
                    source=f"{layer.name} ({layer.source_locator})",
                )
                validate_source(
                    self._schema, raw, source=f"{layer.name} ({layer.source_locator})"
                )
                if replacement := self.policy_replacements.get(layer.name):
                    # Validate even shadowed backing-source policy before overlaying.
                    SourceRestrictions.from_raw(
                        raw,
                        root_layer=layer,
                        layer_name=layer.name,
                        locator=layer.source_locator,
                        kind="source",
                        store_fingerprint=layer.fingerprint,
                    )
                    if replacement.locator != layer.source_locator:
                        raise ValueError("Policy replacement source locator changed")
                    if replacement.replaces_roots and (
                        replacement.identity is None
                        or replacement.identity
                        != self.source_identities.get(layer.name)
                    ):
                        raise ValueError("Root replacement source identity changed")
                    raw = replacement.replace_in(raw)
                roots = validate_root_source(raw, layer=layer)
                # Compare the effective source projection, not shadowed disk roots.
                # Canonical overlay paths must still resist symlink reinterpretation.
                if not force_load or (replacement and replacement.replaces_roots):
                    for accepted in self.accepted_root_sources:
                        if (
                            accepted.layer_name == layer.name
                            and roots != accepted.authorized_roots
                        ):
                            raise ValueError(
                                "Authorized root interpretation changed; explicit reload required"
                            )
                if not isinstance(layer, LaunchOverridesLayer):
                    restrictions.append(
                        SourceRestrictions.from_raw(
                            self._restriction_projection(layer, raw, replacement),
                            root_layer=layer,
                            layer_name=layer.name,
                            identity=self.source_identities.get(layer.name),
                            replaces_roots=bool(
                                replacement and replacement.replaces_roots
                            ),
                            locator=layer.source_locator,
                            kind="mode"
                            if isinstance(layer, AgentProfileLayer)
                            else "source",
                            store_fingerprint=(
                                None
                                if layer.name in overrides
                                or layer.name in self.policy_replacements
                                else layer.fingerprint
                            ),
                        )
                    )
                if raw:
                    layer_dicts.append(_LayerData(name=layer.name, data=raw))
            merged, origins, thinking_origins = self._merge_fields(
                self._schema, layer_dicts
            )
            candidate = ConfigCandidate(
                self._validate_merged(
                    merged,
                    origins,
                    thinking_origins,
                    catalog_snapshot=self._catalog_for_layers(layer_dicts),
                ),
                tuple(restrictions),
            )
            for live, staged in zip(self._layers, layers, strict=True):
                live._accept_loaded_state(staged)
            return candidate

    @staticmethod
    def _restriction_projection(
        layer: ConfigLayer[RawConfig],
        raw: dict[str, Any],
        replacement: SourceRestrictions | None,
    ) -> dict[str, Any]:
        """Keep only ordinary child-profile NEVER values out of irrevocable policy."""
        if not isinstance(layer, AgentProfileLayer) or replacement is not None:
            return raw
        projected = copy.deepcopy(raw)
        for settings in projected.get("tools", {}).values():
            if settings.get("permission") == "never":
                settings["permission"] = "ask"
        return projected

    def _merge_fields(
        self, schema: type[S], layer_dicts: list[_LayerData]
    ) -> tuple[dict[str, Any], dict[str, str], dict[str, str]]:
        accumulated: dict[str, Any] = defaultdict(dict)
        origins: dict[str, str] = {}
        thinking_origins: dict[str, str] = {}

        for ld in layer_dicts:
            for key, value in ld.data.items():
                if key not in schema.model_fields:
                    continue

                field_info = schema.model_fields[key]
                annotation = field_info.annotation
                if annotation is None:
                    continue

                is_fragment = isinstance(annotation, type) and issubclass(
                    annotation, ConfigFragment
                )
                if is_fragment:
                    if not isinstance(value, dict):
                        continue

                    merged_fragment = False
                    for fragment_key, fragment_value in value.items():
                        if fragment_key not in annotation.model_fields:
                            continue

                        fragment_field = annotation.model_fields[fragment_key]
                        fragment_meta = MergeFieldMetadata.from_field(fragment_field)
                        if fragment_meta is None:
                            continue

                        fragment_value = self._apply_model_before_validators(
                            fragment_key, fragment_field, fragment_value
                        )
                        self._validate_merge_value(
                            f"{key}.{fragment_key}",
                            ld.name,
                            fragment_meta.merge_strategy,
                            fragment_value,
                        )
                        accumulated[key][fragment_key] = (
                            fragment_meta.merge_strategy.apply(
                                accumulated[key].get(fragment_key),
                                fragment_value,
                                key_fn=self._make_key_fn(fragment_meta),
                            )
                        )
                        merged_fragment = True
                    if merged_fragment:
                        origins[key] = ld.name
                    continue

                meta = MergeFieldMetadata.from_field(field_info)
                if meta is None:
                    continue

                value = self._apply_model_before_validators(key, field_info, value)
                if key == "mcp_servers" and isinstance(value, dict):
                    # TOML [mcp_servers.<name>] yields a dict; the schema wants
                    # a list of tables. REPLACE passes the dict through to
                    # pydantic, so flag it here with actionable guidance.
                    raise ConfigMergeError(key, ld.name, "list", value)
                if key == "thinking_overrides" and isinstance(value, dict):
                    thinking_origins.update({alias: ld.name for alias in value})
                accumulated[key] = meta.merge_strategy.apply(
                    accumulated.get(key), value, key_fn=self._make_key_fn(meta)
                )
                origins[key] = ld.name

        return accumulated, origins, thinking_origins

    def _validate_merge_value(
        self, field_name: str, layer_name: str, strategy: MergeStrategy, value: Any
    ) -> None:
        if value is None:
            return
        if strategy in {MergeStrategy.CONCAT, MergeStrategy.UNION}:
            if isinstance(value, list) or (isinstance(value, dict) and not value):
                return
            raise ConfigMergeError(field_name, layer_name, "list", value)
        if strategy in {
            MergeStrategy.MERGE,
            MergeStrategy.DEEP_MERGE,
        } and not isinstance(value, dict):
            raise ConfigMergeError(field_name, layer_name, "dictionary", value)

    def _catalog_for_layers(
        self, layer_dicts: list[_LayerData]
    ) -> CatalogSnapshot | None:
        """Catalog authority is external to ordinary configuration layers."""
        return self.catalog_snapshot

    def _validate_merged(
        self,
        merged: dict[str, Any],
        origins: dict[str, str],
        thinking_origins: dict[str, str],
        *,
        catalog_snapshot: CatalogSnapshot | None,
    ) -> S:
        context = {
            "origins": origins,
            "item_origins": {"thinking_overrides": thinking_origins},
            "catalog_snapshot": catalog_snapshot,
        }
        config = self._schema.model_validate(merged, context=context)
        config._origins = origins
        if catalog_snapshot is not None and isinstance(config, ChartreuxConfigSchema):
            config.attach_catalog_snapshot(catalog_snapshot)
            from chartreux.core.model_catalog.resolver import ModelResolver

            resolver = ModelResolver(catalog_snapshot)
            canonical = resolver.canonicalize_thinking_overrides(
                config.thinking_overrides
            )
            object.__setattr__(config, "thinking_overrides", canonical)
        return config

    def _apply_model_before_validators(
        self, field_name: str, field_info: FieldInfo, value: Any
    ) -> Any:
        if field_name != "models":
            return value

        for item in field_info.metadata:
            if isinstance(item, BeforeValidator):
                func = cast(Callable[[Any], Any], item.func)
                value = func(value)
        return value

    def _make_key_fn(
        self, merge_field_meta: MergeFieldMetadata
    ) -> Callable[[Any], str] | None:
        merge_key = merge_field_meta.merge_key
        if merge_key is None:
            return None

        return lambda item: item[merge_key]
