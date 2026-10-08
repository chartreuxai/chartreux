from __future__ import annotations

from pathlib import Path
from typing import cast
import warnings

from pydantic import ValidationError
import pytest

from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
from chartreux.core.config.layers.default import DefaultConfigLayer
from chartreux.core.config.layers.overrides import OverridesLayer
from chartreux.core.config.orchestrator import ConfigOrchestrator
from chartreux.core.dispatch import (
    DEFAULT_DISPATCH_MODE,
    ORCHESTRATED_PRESET,
    SHIPPED_PRESETS,
)
from chartreux.core.model_catalog.defaults import SHIPPED_CATALOG
from chartreux.core.model_catalog.loader import (
    CatalogLoadError,
    CatalogSnapshot,
    load_catalog,
    merge_catalog_overlay,
    merge_dispatch_overlay,
)
from chartreux.core.model_catalog.resolver import ModelResolutionError, ModelResolver
from chartreux.core.model_catalog.schema import ModelCatalog, Prices, ProviderDefinition


def _minimal() -> dict[str, object]:
    return {
        "providers": {"test-provider": {"api_base": "https://example.test"}},
        "models": {
            "base": {"deployments": [{"provider": "test-provider", "name": "wire"}]}
        },
        "roles": {},
    }


@pytest.mark.parametrize(
    "threshold",
    [0.75, 1.5, 0, -1, True, False, float("nan"), float("inf"), float("-inf"), "1.5"],
)
def test_deployment_threshold_rejects_non_positive_whole_integers(
    threshold: object,
) -> None:
    raw = _minimal()
    raw["models"]["base"]["deployments"][0]["auto_compact_threshold"] = threshold  # type: ignore[index]
    with pytest.raises(ValidationError, match="positive whole integer"):
        ModelCatalog.model_validate(raw)


@pytest.mark.parametrize("threshold", [None, 1, 400000, 400000.0])
@pytest.mark.parametrize("global_threshold", [0, 12345])
def test_deployment_threshold_and_global_fallback_resolve(
    threshold: int | float | None, global_threshold: int
) -> None:
    raw = _minimal()
    raw["models"]["base"]["deployments"][0]["auto_compact_threshold"] = threshold  # type: ignore[index]
    catalog = ModelCatalog.model_validate(raw)
    deployment_threshold = catalog.models["base"].deployments[0].auto_compact_threshold
    assert deployment_threshold is None or type(deployment_threshold) is int
    resolver = ModelResolver(CatalogSnapshot(catalog, "test"))
    resolved = resolver.resolve("base")
    assert resolved.materialize(
        auto_compact_threshold=global_threshold
    ).auto_compact_threshold == (
        global_threshold if threshold is None else int(threshold)
    )


def test_1_schema_accepts_minimal_valid_catalog() -> None:
    catalog = ModelCatalog.model_validate(_minimal())
    assert catalog.models["base"].deployments[0].name == "wire"


def test_2_schema_forbids_unknown_fields() -> None:
    raw = _minimal()
    for table, field in (
        (raw["providers"]["test-provider"], "provider_extra"),  # type: ignore[index]
        (raw["models"]["base"]["deployments"][0], "deployment_extra"),  # type: ignore[index]
        (raw["models"]["base"], "base_extra"),  # type: ignore[index]
        (raw, "catalog_extra"),
    ):
        table[field] = True  # type: ignore[index]
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            ModelCatalog.model_validate(raw)
        del table[field]  # type: ignore[index]


def test_3_catalog_is_frozen() -> None:
    catalog = ModelCatalog.model_validate(_minimal())
    with pytest.raises(ValidationError, match="frozen"):
        catalog.models = {}  # type: ignore[misc]


def test_4_duplicate_deployment_provider_is_rejected() -> None:
    raw = _minimal()
    raw["models"]["base"]["deployments"].append({  # type: ignore[index]
        "provider": "test-provider",
        "name": "other",
    })
    with pytest.raises(ValidationError, match="existing provider deployment"):
        ModelCatalog.model_validate(raw)


def test_5_roles_and_reserved_at_are_validated() -> None:
    raw = _minimal()
    raw["models"] = {"bad@name": raw["models"]["base"]}  # type: ignore[index]
    with pytest.raises(ValidationError, match="reserved"):
        ModelCatalog.model_validate(raw)
    raw = _minimal()
    raw["roles"] = {"bad@role": {"model": "base", "thinking": "medium"}}
    with pytest.raises(ValidationError, match="reserved"):
        ModelCatalog.model_validate(raw)


def test_6_provider_name_and_unknown_provider_are_rejected() -> None:
    for invalid in ("", "  ", "bad/name", "bad@name", "bad\nname", "bad\x7fname"):
        raw = _minimal()
        raw["providers"] = {invalid: {"api_base": "https://example.test"}}
        with pytest.raises(ValidationError, match="Provider name"):
            ModelCatalog.model_validate(raw)

    raw = _minimal()
    raw["providers"] = {" test-provider ": raw["providers"]["test-provider"]}  # type: ignore[index]
    raw["models"]["base"]["deployments"][0]["provider"] = " test-provider "  # type: ignore[index]
    catalog = ModelCatalog.model_validate(raw)
    assert list(catalog.providers) == ["test-provider"]
    assert catalog.models["base"].deployments[0].provider == "test-provider"

    raw = _minimal()
    raw["providers"][" test-provider "] = raw["providers"]["test-provider"]  # type: ignore[index]
    with pytest.raises(ValidationError, match="collision"):
        ModelCatalog.model_validate(raw)

    for invalid in ("bad/name", "bad@name", "bad\nname", "  "):
        raw = _minimal()
        raw["models"]["base"]["deployments"][0]["provider"] = invalid  # type: ignore[index]
        with pytest.raises(ValidationError, match="Provider name"):
            ModelCatalog.model_validate(raw)

    raw = _minimal()
    raw["models"]["base"]["deployments"][0]["provider"] = "other-provider"  # type: ignore[index]
    with pytest.raises(ValidationError, match="unknown provider"):
        ModelCatalog.model_validate(raw)


def test_provider_names_preserve_case_and_inner_spaces() -> None:
    raw = _minimal()
    raw["providers"] = {"My Gateway": {"api_base": "https://example.test"}}
    raw["models"]["base"]["deployments"][0]["provider"] = "My Gateway"  # type: ignore[index]
    assert list(ModelCatalog.model_validate(raw).providers) == ["My Gateway"]


def test_7_legacy_role_lists_and_missing_preset_fields_are_rejected() -> None:
    raw = _minimal()
    raw["roles"] = {"role": {"models": ["base", "base"]}}
    with pytest.raises(ValidationError, match="one 'model'.*'thinking'"):
        ModelCatalog.model_validate(raw)
    raw = _minimal()
    raw["models"]["base"]["deployments"] = []  # type: ignore[index]
    with pytest.raises(ValidationError):
        ModelCatalog.model_validate(raw)
    raw = _minimal()
    raw["roles"] = {"role": {"model": "base"}}
    with pytest.raises(ValidationError, match="thinking"):
        ModelCatalog.model_validate(raw)


def test_8_disabled_entries_are_valid() -> None:
    raw = _minimal()
    raw["providers"]["test-provider"]["disabled"] = True  # type: ignore[index]
    raw["models"]["base"]["disabled"] = True  # type: ignore[index]
    raw["models"]["base"]["deployments"][0]["disabled"] = True  # type: ignore[index]
    assert ModelCatalog.model_validate(raw).models["base"].disabled is True


def test_9_unknown_and_free_prices_are_distinct() -> None:
    unknown, free = Prices(), Prices(input=0.0, output=0.0, cached_input=0.0)
    assert unknown.input is None
    assert free.input == 0.0
    assert unknown != free


def test_10_sparse_provider_overlay_inherits_shipped_fields() -> None:
    catalog = merge_catalog_overlay(
        SHIPPED_CATALOG, {"providers": {"mistral": {"emits_finish_reason": False}}}
    )
    provider = catalog.providers["mistral"]
    assert provider.emits_finish_reason is False
    assert provider.api_base == SHIPPED_CATALOG.providers["mistral"].api_base


def test_11_overlay_replaces_scalars_and_deployment_lists() -> None:
    catalog = merge_catalog_overlay(
        SHIPPED_CATALOG,
        {
            "models": {
                "glm-5-3": {
                    "thinking": "low",
                    "deployments": [{"provider": "mistral", "name": "glm-5-3"}],
                }
            }
        },
    )
    assert catalog.models["glm-5-3"].thinking == "low"
    assert [item.name for item in catalog.models["glm-5-3"].deployments] == ["glm-5-3"]


def test_12_deployment_overlay_inherits_by_provider_and_adds_provider() -> None:
    catalog = merge_catalog_overlay(
        SHIPPED_CATALOG,
        {
            "providers": {"test-second": {"api_base": "https://second.test"}},
            "models": {
                "glm-5-3": {
                    "deployments": [
                        {"provider": "mistral", "supports_images": True},
                        {"provider": "test-second", "name": "glm-second"},
                    ]
                }
            },
        },
    )
    deployments = catalog.models["glm-5-3"].deployments
    assert [(item.provider, item.name) for item in deployments] == [
        ("mistral", "zai-glm-5-3"),
        ("test-second", "glm-second"),
    ]
    assert deployments[0].supports_images is True


def test_13_overlay_can_disable_shipped_entries() -> None:
    catalog = merge_catalog_overlay(
        SHIPPED_CATALOG, {"models": {"glm-5-3": {"disabled": True}}}
    )
    assert catalog.models["glm-5-3"].disabled is True


def test_14_missing_models_toml_loads_shipped_defaults(tmp_path: Path) -> None:
    snapshot = load_catalog(tmp_path / "models.toml")
    assert snapshot.catalog == SHIPPED_CATALOG


def test_15_invalid_models_toml_raises_typed_error(tmp_path: Path) -> None:
    path = tmp_path / "models.toml"
    path.write_text("[providers.bad]\nunknown = true\n")
    with pytest.raises(CatalogLoadError, match="Invalid model catalog"):
        load_catalog(path)


def test_16_shipped_defaults_validate_and_have_verified_wire_names() -> None:
    catalog = ModelCatalog.model_validate(SHIPPED_CATALOG.model_dump())
    assert all(
        deployment.provider in catalog.providers
        for model in catalog.models.values()
        for deployment in model.deployments
    )
    # The shipped catalog is neutral and publicly reachable only: the Mistral
    # public provider and models it actually serves. Personal setups (local
    # proxies, private pins) belong in the user models.toml overlay.
    assert set(catalog.providers) == {"mistral"}
    assert catalog.providers["mistral"].api_base == "https://api.mistral.ai/v1"
    assert set(catalog.models) == {"glm-5-3"}
    assert {name: model.thinking for name, model in catalog.models.items()} == {
        "glm-5-3": "high"
    }
    assert catalog.roles["orchestrator"].model == "glm-5-3"
    assert {
        name: (role.model, role.thinking) for name, role in catalog.roles.items()
    } == {
        "orchestrator": ("glm-5-3", "high"),
        "large": ("glm-5-3", "high"),
        "medium": ("glm-5-3", "medium"),
        "small": ("glm-5-3", "low"),
    }
    assert {
        name: (
            model.deployments[0].provider,
            model.deployments[0].name,
            model.deployments[0].prices.input,
            model.deployments[0].prices.output,
            model.deployments[0].prices.cached_input,
            model.deployments[0].auto_compact_threshold,
        )
        for name, model in catalog.models.items()
    } == {"glm-5-3": ("mistral", "zai-glm-5-3", 1.4, 4.4, 0.14, 400000)}
    assert {
        deployment.name
        for model in catalog.models.values()
        for deployment in model.deployments
    } == {"zai-glm-5-3"}


def test_shipped_catalog_json_dump_round_trips() -> None:
    resolved = ModelResolver(CatalogSnapshot(SHIPPED_CATALOG, "test")).resolve(
        "glm-5-3"
    )
    assert resolved.materialize(auto_compact_threshold=200000).temperature == 1.0
    dumped = SHIPPED_CATALOG.model_dump_json()

    assert ModelCatalog.model_validate_json(dumped) == SHIPPED_CATALOG


@pytest.mark.parametrize(
    ("preset", "thinking"),
    [
        ("orchestrator", "high"),
        ("large", "high"),
        ("medium", "medium"),
        ("small", "low"),
    ],
)
def test_shipped_presets_resolve_their_own_thinking(preset: str, thinking: str) -> None:
    resolver = ModelResolver(CatalogSnapshot(SHIPPED_CATALOG, "test"))

    resolved = resolver.resolve(f"@{preset}")

    assert resolved.base_model == "glm-5-3"
    assert resolved.thinking == thinking
    assert resolved.materialize(auto_compact_threshold=200000).thinking == thinking


def test_provider_definition_extra_headers_serialize_without_warnings() -> None:
    provider = ProviderDefinition(
        api_base="https://example.test", extra_headers={"Authorization": "test"}
    )

    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always")
        dumped = provider.model_dump()
        dumped_json = provider.model_dump_json()

    assert not captured_warnings
    assert dumped["extra_headers"] == {"Authorization": "test"}
    assert ProviderDefinition.model_validate_json(dumped_json) == provider


def test_17_resolver_expands_scalar_models_and_roles_only() -> None:
    raw = _minimal()
    raw["roles"] = {"preferred": {"model": "base", "thinking": "medium"}}
    resolver = ModelResolver(CatalogSnapshot(ModelCatalog.model_validate(raw), "test"))
    assert resolver.expression_bases("base") == ("base",)
    assert resolver.expression_bases("@preferred") == ("base",)
    with pytest.raises(ModelResolutionError) as error:
        resolver.expression_bases(cast(str, ["base"]))
    assert error.value.code == "invalid_expression"
    with pytest.raises(ModelResolutionError) as error:
        resolver.expression_bases("alias")
    assert error.value.code == "unknown_model"
    with pytest.raises(ModelResolutionError, match="reserved"):
        resolver.expression_bases("@")


def test_roles_can_share_a_model_with_distinct_thinking() -> None:
    raw = _minimal()
    raw["roles"] = {
        "planner": {"model": "base", "thinking": "low"},
        "reviewer": {"model": "base", "thinking": "high"},
    }
    resolver = ModelResolver(CatalogSnapshot(ModelCatalog.model_validate(raw), "test"))
    planner = resolver.resolve("@planner")
    reviewer = resolver.resolve("@reviewer")
    assert planner.base_model == reviewer.base_model == "base"
    assert planner.materialize(auto_compact_threshold=100).thinking == "low"
    assert reviewer.materialize(auto_compact_threshold=100).thinking == "high"
    assert (
        reviewer.materialize(auto_compact_threshold=100, thinking="medium").thinking
        == "medium"
    )
    assert planner.identity.thinking == "low"
    assert reviewer.identity.thinking == "high"


def test_incomplete_preset_is_structurally_valid_but_resolution_guides_repair() -> None:
    raw = _minimal()
    raw["roles"] = {"planner": {"model": "pending", "thinking": "high"}}
    resolver = ModelResolver(CatalogSnapshot(ModelCatalog.model_validate(raw), "test"))
    with pytest.raises(
        ModelResolutionError, match="missing model.*edit its model and thinking"
    ) as error:
        resolver.resolve("@planner")
    assert error.value.code == "preset_model_missing"


def test_invalid_preset_thinking_and_legacy_overlay_are_actionable() -> None:
    raw = _minimal()
    raw["roles"] = {"planner": {"model": "base", "thinking": "absurd"}}
    with pytest.raises(ValidationError, match="known thinking level"):
        ModelCatalog.model_validate(raw)
    with pytest.raises(
        ValueError, match="roles.planner.models is obsolete.*model.*thinking"
    ):
        merge_catalog_overlay(
            SHIPPED_CATALOG, {"roles": {"planner": {"models": ["glm-5-3"]}}}
        )


def test_preset_thinking_must_be_supported_by_selected_deployment() -> None:
    raw = _minimal()
    raw["models"]["base"]["deployments"][0]["supported_thinking_levels"] = ["low"]  # type: ignore[index]
    raw["roles"] = {"planner": {"model": "base", "thinking": "high"}}
    resolver = ModelResolver(CatalogSnapshot(ModelCatalog.model_validate(raw), "test"))
    with pytest.raises(
        ModelResolutionError, match="supporting thinking level"
    ) as error:
        resolver.resolve("@planner")
    assert error.value.code == "thinking_unsupported"
    overridden = resolver.resolve("@planner", thinking_override="low")
    assert (
        overridden.materialize(auto_compact_threshold=100, thinking="low").thinking
        == "low"
    )


def test_committed_preset_pair_revalidates_after_deployment_capability_change() -> None:
    raw = _minimal()
    raw["roles"] = {"planner": {"model": "base", "thinking": "high"}}
    committed = (
        ModelResolver(CatalogSnapshot(ModelCatalog.model_validate(raw), "before"))
        .resolve("@planner")
        .identity
    )
    raw["models"]["base"]["deployments"][0]["supported_thinking_levels"] = ["low"]  # type: ignore[index]
    changed = ModelResolver(CatalogSnapshot(ModelCatalog.model_validate(raw), "after"))
    with pytest.raises(ModelResolutionError, match="Committed thinking level") as error:
        changed.resolve_committed(committed)
    assert error.value.code == "committed_thinking_unsupported"


def test_preset_candidate_filter_rejection_is_not_a_thinking_error() -> None:
    raw = _minimal()
    raw["roles"] = {"planner": {"model": "base", "thinking": "high"}}
    resolver = ModelResolver(CatalogSnapshot(ModelCatalog.model_validate(raw), "test"))
    with pytest.raises(ModelResolutionError) as error:
        resolver.resolve("@planner", candidate_filter=lambda _candidate: False)
    assert error.value.code == "no_compatible_deployment"


def test_18_resolver_materializes_wire_name_and_filters_deployments() -> None:
    raw = _minimal()
    raw["providers"]["second-provider"] = {"api_base": "https://second.test"}  # type: ignore[index]
    raw["models"]["base"]["thinking"] = "high"  # type: ignore[index]
    raw["models"]["base"]["temperature"] = 0.7  # type: ignore[index]
    raw["models"]["base"]["deployments"].append(  # type: ignore[index]
        {"provider": "second-provider", "name": "second-wire"}
    )
    resolver = ModelResolver(CatalogSnapshot(ModelCatalog.model_validate(raw), "test"))
    selected = resolver.resolve("base", allowed_models=["second-provider/*"])
    model = selected.materialize(auto_compact_threshold=123)
    assert (model.name, model.provider, model.thinking, model.temperature) == (
        "second-wire",
        "second-provider",
        "high",
        0.7,
    )
    assert model.auto_compact_threshold == 123
    with pytest.raises(ModelResolutionError, match="permitted"):
        resolver.resolve("base", allowed_models=["none"])


def test_resolver_unknown_explicit_name_is_typed() -> None:
    resolver = ModelResolver(CatalogSnapshot(SHIPPED_CATALOG, "test"))
    with pytest.raises(ModelResolutionError) as error:
        resolver.resolve("typo-model")
    assert error.value.code == "unknown_model"
    assert "Valid canonical models: glm-5-3" in str(error.value)


def test_resolver_unknown_role_lists_valid_roles() -> None:
    resolver = ModelResolver(CatalogSnapshot(SHIPPED_CATALOG, "test"))
    with pytest.raises(ModelResolutionError) as error:
        resolver.resolve("@not-a-role")
    assert error.value.code == "unknown_role"
    assert "@orchestrator" in str(error.value)
    assert "@medium" in str(error.value)


def test_deployment_priority_disabled_skip_and_all_disabled() -> None:
    raw = _minimal()
    raw["providers"]["second-provider"] = {"api_base": "https://second.test"}  # type: ignore[index]
    raw["models"]["base"]["deployments"] = [  # type: ignore[index]
        {"provider": "test-provider", "name": "first", "disabled": True},
        {"provider": "second-provider", "name": "second"},
    ]
    catalog = ModelCatalog.model_validate(raw)
    resolver = ModelResolver(CatalogSnapshot(catalog, "test"))
    assert resolver.resolve("base").deployment.name == "second"

    disabled = catalog.model_copy(
        update={
            "models": {
                "base": catalog.models["base"].model_copy(
                    update={
                        "deployments": [
                            deployment.model_copy(update={"disabled": True})
                            for deployment in catalog.models["base"].deployments
                        ]
                    }
                )
            }
        }
    )
    with pytest.raises(ModelResolutionError) as error:
        ModelResolver(CatalogSnapshot(disabled, "disabled")).resolve("base")
    assert error.value.code == "all_deployments_disabled"


def test_thinking_overrides_require_canonical_keys() -> None:
    resolver = ModelResolver(CatalogSnapshot(SHIPPED_CATALOG, "test"))
    assert resolver.canonicalize_thinking_overrides({"glm-5-3": "low"}) == {
        "glm-5-3": "low"
    }
    with pytest.raises(ModelResolutionError, match="Unknown model"):
        resolver.canonicalize_thinking_overrides({"glm-alias": "low"})


def test_real_config_consumer_materializes_wire_name_and_base_label() -> None:
    snapshot = CatalogSnapshot(SHIPPED_CATALOG, "revision")
    config = ChartreuxConfigSchema.model_validate(
        {"active_model": "glm-5-3"}, context={"catalog_snapshot": snapshot}
    ).attach_catalog_snapshot(snapshot)
    model = config.get_active_model()
    assert model.alias == "glm-5-3"
    assert model.name == "zai-glm-5-3"
    assert model.provider == "mistral"


def test_glm_canonical_name_keeps_exact_deployment_wire_name() -> None:
    snapshot = CatalogSnapshot(SHIPPED_CATALOG, "revision")
    config = ChartreuxConfigSchema.model_validate(
        {"active_model": "glm-5-3"}, context={"catalog_snapshot": snapshot}
    ).attach_catalog_snapshot(snapshot)
    assert config.get_active_model().name == "zai-glm-5-3"


def test_catalog_snapshot_is_deeply_immutable() -> None:
    raw = _minimal()
    raw["providers"]["test-provider"]["extra_headers"] = {"Authorization": "test"}  # type: ignore[index]
    raw["models"]["base"]["deployments"][0]["supported_thinking_levels"] = [  # type: ignore[index]
        "low"
    ]
    raw["roles"] = {"role": {"model": "base", "thinking": "low"}}
    catalog = ModelCatalog.model_validate(raw)

    with pytest.raises(TypeError):
        catalog.providers["other/provider"] = catalog.providers["test-provider"]  # type: ignore[index]
    with pytest.raises(TypeError):
        catalog.models["other"] = catalog.models["base"]  # type: ignore[index]
    with pytest.raises(TypeError):
        catalog.roles["other"] = catalog.roles["role"]  # type: ignore[index]
    with pytest.raises(TypeError):
        catalog.providers["test-provider"].extra_headers["Other"] = "value"  # type: ignore[index]
    with pytest.raises(TypeError):
        catalog.models["base"].deployments[0] = catalog.models["base"].deployments[0]  # type: ignore[index]
    with pytest.raises(TypeError):
        catalog.models["base"].deployments[0].supported_thinking_levels[0] = "high"  # type: ignore[index]
    with pytest.raises(ValidationError, match="frozen"):
        catalog.roles["role"].model = "other"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("providers", "test-provider", "api_base"), "", "HTTP"),
        (("providers", "test-provider", "api_base"), "not-a-url", "HTTP"),
        (("providers", "test-provider", "api_key_env_var"), "BAD-NAME", "environment"),
        (("models", "base", "thinking"), "invalid", "known thinking"),
        (("models", "base", "deployments", 0, "prices", "input"), -1, "non-negative"),
        (
            ("models", "base", "deployments", 0, "prices", "output"),
            float("nan"),
            "non-negative",
        ),
        (("models", "base", "deployments", 0, "auto_compact_threshold"), 0, "positive"),
        (
            ("models", "base", "deployments", 0, "supported_thinking_levels"),
            ["invalid"],
            "unknown thinking",
        ),
    ],
)
def test_catalog_validation_rejects_malformed_runtime_metadata(
    path: tuple[str | int, ...], value: object, message: str
) -> None:
    raw = _minimal()
    target: object = raw
    for key in path[:-1]:
        if key == "prices":
            target = target.setdefault(key, {})  # type: ignore[attr-defined]
        else:
            target = target[key]  # type: ignore[index]
    target[path[-1]] = value  # type: ignore[index]

    with pytest.raises(ValidationError, match=message):
        ModelCatalog.model_validate(raw)


@pytest.mark.asyncio
async def test_orchestrator_copy_reattaches_supplied_and_source_catalog_snapshots() -> (
    None
):
    layer = OverridesLayer(data={})
    source = await ConfigOrchestrator.create(
        schema=ChartreuxConfigSchema,
        layers=[DefaultConfigLayer(schema=ChartreuxConfigSchema), layer],
        default_layer_resolver=lambda: layer,
    )
    supplied_snapshot = CatalogSnapshot(
        ModelCatalog.model_validate(_minimal()), "supplied"
    )
    supplied = ChartreuxConfigSchema.model_validate({}).attach_catalog_snapshot(
        supplied_snapshot
    )

    assert source.copy().config.catalog_snapshot is source.config.catalog_snapshot
    assert source.copy(config=supplied).config.catalog_snapshot is supplied_snapshot


def test_role_overlay_merges_per_key_and_rejects_legacy_names() -> None:
    catalog = merge_catalog_overlay(
        SHIPPED_CATALOG, {"roles": {"orchestrator": {"thinking": "medium"}}}
    )
    assert catalog.roles["orchestrator"].model == "glm-5-3"
    assert catalog.roles["orchestrator"].thinking == "medium"
    assert (
        catalog.roles["orchestrator"].description
        == SHIPPED_CATALOG.roles["orchestrator"].description
    )
    assert catalog.roles["large"] == SHIPPED_CATALOG.roles["large"]
    with pytest.raises(ValueError, match=r"migration required.*\[roles\]"):
        merge_catalog_overlay(SHIPPED_CATALOG, {"tags": {}})
    with pytest.raises(ValueError, match="remove aliases"):
        merge_catalog_overlay(SHIPPED_CATALOG, {"models": {"glm-5-3": {"aliases": []}}})


def test_disabled_preset_model_does_not_fall_back_to_another_model() -> None:
    catalog = merge_catalog_overlay(
        SHIPPED_CATALOG,
        {
            "providers": {"backup": {"api_base": "https://backup.test/v1"}},
            "models": {
                "glm-5-3": {"disabled": True},
                "backup-model": {
                    "deployments": [{"provider": "backup", "name": "backup-model"}]
                },
            },
            "roles": {"priority": {"model": "glm-5-3", "thinking": "high"}},
        },
    )
    with pytest.raises(ModelResolutionError, match="disabled"):
        ModelResolver(CatalogSnapshot(catalog, "test")).resolve("@priority")


def test_unknown_expression_options_only_list_eligible_models_and_roles() -> None:
    raw = _minimal()
    raw["providers"]["disabled-provider"] = {  # type: ignore[index]
        "api_base": "https://disabled.test",
        "disabled": True,
    }
    raw["models"].update({  # type: ignore[union-attr]
        "off": {
            "disabled": True,
            "deployments": [{"provider": "test-provider", "name": "off"}],
        },
        "blocked": {
            "deployments": [{"provider": "disabled-provider", "name": "blocked"}]
        },
        "restricted": {
            "deployments": [{"provider": "test-provider", "name": "restricted"}]
        },
    })
    raw["roles"] = {
        "usable": {"model": "base", "thinking": "medium"},
        "disabled-only": {"model": "off", "thinking": "medium"},
        "restricted-only": {"model": "restricted", "thinking": "medium"},
    }
    resolver = ModelResolver(CatalogSnapshot(ModelCatalog.model_validate(raw), "test"))
    for expression, expected in (
        ("unknown", "Valid canonical models: base"),
        ("@unknown", "Valid roles: @usable"),
    ):
        with pytest.raises(ModelResolutionError) as error:
            resolver.resolve(expression, allowed_models=["base"])
        assert expected in str(error.value)
        assert "off" not in str(error.value)
        assert "blocked" not in str(error.value)
        assert "restricted" not in str(error.value)
    assert resolver.resolve("@usable", allowed_models=["base"]).base_model == "base"


def test_allowlist_exclusion_does_not_change_preset_model() -> None:
    catalog = merge_catalog_overlay(
        SHIPPED_CATALOG,
        {
            "providers": {"backup": {"api_base": "https://backup.test/v1"}},
            "models": {
                "backup-model": {
                    "deployments": [{"provider": "backup", "name": "backup-model"}]
                }
            },
            "roles": {"priority": {"model": "glm-5-3", "thinking": "high"}},
        },
    )
    with pytest.raises(ModelResolutionError, match="permitted"):
        ModelResolver(CatalogSnapshot(catalog, "test")).resolve(
            "@priority", allowed_models=["backup-model"]
        )


def test_catalog_overlay_accepts_dispatch_table_without_catalog_authority() -> None:
    catalog = merge_catalog_overlay(
        SHIPPED_CATALOG, {"dispatch": {"mode": "standalone"}}
    )
    assert catalog == SHIPPED_CATALOG
    with pytest.raises(ValueError, match="unknown fields"):
        merge_catalog_overlay(SHIPPED_CATALOG, {"dispatching": {}})


def test_snapshot_default_dispatch_is_the_shipped_default() -> None:
    snapshot = CatalogSnapshot(SHIPPED_CATALOG, "test")
    assert snapshot.dispatch == SHIPPED_PRESETS[DEFAULT_DISPATCH_MODE]
    assert snapshot.dispatch_diagnostics == ()


def test_load_catalog_resolves_dispatch_and_revision(tmp_path: Path) -> None:
    plain = tmp_path / "plain.toml"
    plain.write_text('[roles.small]\nthinking = "low"\n')
    dispatched = tmp_path / "dispatched.toml"
    dispatched.write_text(
        '[roles.small]\nthinking = "low"\n\n[dispatch]\nmode = "orchestrated"\n'
    )

    base = load_catalog(plain)
    selected = load_catalog(dispatched)

    assert base.catalog == selected.catalog
    assert base.dispatch == SHIPPED_PRESETS[DEFAULT_DISPATCH_MODE]
    assert selected.dispatch == ORCHESTRATED_PRESET
    assert selected.dispatch.mode == DEFAULT_DISPATCH_MODE.ORCHESTRATED
    assert selected.revision != base.revision
    assert selected.dispatch_diagnostics == ()


def test_dispatch_overlay_sparse_merge_inherits_shipped_preset() -> None:
    policy = merge_dispatch_overlay({"slots": {"implementor": {"role": "@small"}}})
    assert policy.slots["implementor"].role == "@small"
    assert policy.slots["implementor"].purposes == ("implementation",)
    assert policy.slots["advisor"].role == "@large"
    assert policy.identity == SHIPPED_PRESETS[DEFAULT_DISPATCH_MODE].identity
