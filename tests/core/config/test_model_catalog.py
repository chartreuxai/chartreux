from __future__ import annotations

from pathlib import Path
from typing import cast

from pydantic import ValidationError
import pytest

from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
from chartreux.core.config.layers.default import DefaultConfigLayer
from chartreux.core.config.layers.overrides import OverridesLayer
from chartreux.core.config.orchestrator import ConfigOrchestrator
from chartreux.core.model_catalog.defaults import SHIPPED_CATALOG
from chartreux.core.model_catalog.loader import (
    CatalogLoadError,
    CatalogSnapshot,
    load_catalog,
    merge_catalog_overlay,
)
from chartreux.core.model_catalog.resolver import ModelResolutionError, ModelResolver
from chartreux.core.model_catalog.schema import ModelCatalog, Prices


def _minimal() -> dict[str, object]:
    return {
        "providers": {"test/provider": {"api_base": "https://example.test"}},
        "models": {
            "base": {"deployments": [{"provider": "test/provider", "name": "wire"}]}
        },
        "roles": {},
    }


def test_1_schema_accepts_minimal_valid_catalog() -> None:
    catalog = ModelCatalog.model_validate(_minimal())
    assert catalog.models["base"].deployments[0].name == "wire"


def test_2_schema_forbids_unknown_fields() -> None:
    raw = _minimal()
    for table, field in (
        (raw["providers"]["test/provider"], "provider_extra"),  # type: ignore[index]
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
        "provider": "test/provider",
        "name": "other",
    })
    with pytest.raises(ValidationError, match="one deployment"):
        ModelCatalog.model_validate(raw)


def test_5_roles_and_reserved_at_are_validated() -> None:
    raw = _minimal()
    raw["models"] = {"bad@name": raw["models"]["base"]}  # type: ignore[index]
    with pytest.raises(ValidationError, match="reserved"):
        ModelCatalog.model_validate(raw)
    raw = _minimal()
    raw["roles"] = {"bad@role": {"models": ["base"]}}
    with pytest.raises(ValidationError, match="reserved"):
        ModelCatalog.model_validate(raw)


def test_6_provider_id_and_unknown_provider_are_rejected() -> None:
    raw = _minimal()
    raw["providers"] = {"unqualified": {"api_base": "https://example.test"}}
    with pytest.raises(ValidationError, match="must contain"):
        ModelCatalog.model_validate(raw)
    raw = _minimal()
    raw["models"]["base"]["deployments"][0]["provider"] = "other/provider"  # type: ignore[index]
    with pytest.raises(ValidationError, match="unknown provider"):
        ModelCatalog.model_validate(raw)


def test_7_duplicate_role_members_and_empty_lists_are_rejected() -> None:
    raw = _minimal()
    raw["roles"] = {"role": {"models": ["base", "base"]}}
    with pytest.raises(ValidationError, match="Duplicate role"):
        ModelCatalog.model_validate(raw)
    raw = _minimal()
    raw["models"]["base"]["deployments"] = []  # type: ignore[index]
    with pytest.raises(ValidationError):
        ModelCatalog.model_validate(raw)
    raw = _minimal()
    raw["roles"] = {"role": {"models": []}}
    with pytest.raises(ValidationError, match="at least 1 item"):
        ModelCatalog.model_validate(raw)


def test_8_disabled_entries_are_valid() -> None:
    raw = _minimal()
    raw["providers"]["test/provider"]["disabled"] = True  # type: ignore[index]
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
        SHIPPED_CATALOG,
        {"providers": {"mistral/default": {"emits_finish_reason": False}}},
    )
    provider = catalog.providers["mistral/default"]
    assert provider.emits_finish_reason is False
    assert provider.api_base == SHIPPED_CATALOG.providers["mistral/default"].api_base


def test_11_overlay_replaces_scalars_and_deployment_lists() -> None:
    catalog = merge_catalog_overlay(
        SHIPPED_CATALOG,
        {
            "models": {
                "glm-5-3": {
                    "thinking": "low",
                    "deployments": [{"provider": "mistral/default", "name": "glm-5-3"}],
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
            "providers": {"test/second": {"api_base": "https://second.test"}},
            "models": {
                "glm-5-3": {
                    "deployments": [
                        {"provider": "mistral/default", "supports_images": True},
                        {"provider": "test/second", "name": "glm-second"},
                    ]
                }
            },
        },
    )
    deployments = catalog.models["glm-5-3"].deployments
    assert [(item.provider, item.name) for item in deployments] == [
        ("mistral/default", "zai-glm-5-3"),
        ("test/second", "glm-second"),
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
    assert catalog.models["glm-5-3"].thinking == "medium"
    assert catalog.models["gpt-6-astra"].thinking == "low"
    assert catalog.models["gpt-5.6-luna"].thinking == "high"
    assert catalog.models["gpt-5.6-sol"].thinking == "medium"
    assert catalog.models["gpt-5.6-terra"].thinking == "medium"
    assert catalog.roles["orchestrator"].models == ("glm-5-3",)
    assert {
        deployment.name
        for model in catalog.models.values()
        for deployment in model.deployments
    } == {"zai-glm-5-3", "gpt-6-astra", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra"}


def test_shipped_catalog_json_dump_round_trips() -> None:
    dumped = SHIPPED_CATALOG.model_dump_json()

    assert ModelCatalog.model_validate_json(dumped) == SHIPPED_CATALOG


def test_17_resolver_expands_scalar_models_and_roles_only() -> None:
    raw = _minimal()
    raw["roles"] = {"preferred": {"models": ["base"]}}
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


def test_18_resolver_materializes_wire_name_and_filters_deployments() -> None:
    raw = _minimal()
    raw["providers"]["second/provider"] = {"api_base": "https://second.test"}  # type: ignore[index]
    raw["models"]["base"]["thinking"] = "high"  # type: ignore[index]
    raw["models"]["base"]["temperature"] = 0.7  # type: ignore[index]
    raw["models"]["base"]["deployments"].append(  # type: ignore[index]
        {"provider": "second/provider", "name": "second-wire"}
    )
    resolver = ModelResolver(CatalogSnapshot(ModelCatalog.model_validate(raw), "test"))
    selected = resolver.resolve("base", allowed_models=["second/*"])
    model = selected.materialize(auto_compact_threshold=123)
    assert (model.name, model.provider, model.thinking, model.temperature) == (
        "second-wire",
        "second/provider",
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


def test_deployment_priority_disabled_skip_and_all_disabled() -> None:
    raw = _minimal()
    raw["providers"]["second/provider"] = {"api_base": "https://second.test"}  # type: ignore[index]
    raw["models"]["base"]["deployments"] = [  # type: ignore[index]
        {"provider": "test/provider", "name": "first", "disabled": True},
        {"provider": "second/provider", "name": "second"},
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
    assert model.provider == "mistral/default"


def test_glm_canonical_name_keeps_exact_deployment_wire_name() -> None:
    snapshot = CatalogSnapshot(SHIPPED_CATALOG, "revision")
    config = ChartreuxConfigSchema.model_validate(
        {"active_model": "glm-5-3"}, context={"catalog_snapshot": snapshot}
    ).attach_catalog_snapshot(snapshot)
    assert config.get_active_model().name == "zai-glm-5-3"


def test_catalog_snapshot_is_deeply_immutable() -> None:
    raw = _minimal()
    raw["providers"]["test/provider"]["extra_headers"] = {"Authorization": "test"}  # type: ignore[index]
    raw["models"]["base"]["deployments"][0]["supported_thinking_levels"] = [  # type: ignore[index]
        "low"
    ]
    raw["roles"] = {"role": {"models": ["base"]}}
    catalog = ModelCatalog.model_validate(raw)

    with pytest.raises(TypeError):
        catalog.providers["other/provider"] = catalog.providers["test/provider"]  # type: ignore[index]
    with pytest.raises(TypeError):
        catalog.models["other"] = catalog.models["base"]  # type: ignore[index]
    with pytest.raises(TypeError):
        catalog.roles["other"] = catalog.roles["role"]  # type: ignore[index]
    with pytest.raises(TypeError):
        catalog.providers["test/provider"].extra_headers["Other"] = "value"  # type: ignore[index]
    with pytest.raises(TypeError):
        catalog.models["base"].deployments[0] = catalog.models["base"].deployments[0]  # type: ignore[index]
    with pytest.raises(TypeError):
        catalog.models["base"].deployments[0].supported_thinking_levels[0] = "high"  # type: ignore[index]
    with pytest.raises(TypeError):
        catalog.roles["role"].models[0] = "other"  # type: ignore[index]


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("providers", "test/provider", "api_base"), "", "HTTP"),
        (("providers", "test/provider", "api_base"), "not-a-url", "HTTP"),
        (("providers", "test/provider", "api_key_env_var"), "BAD-NAME", "environment"),
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
        SHIPPED_CATALOG, {"roles": {"orchestrator": {"models": ["glm-5-3"]}}}
    )
    assert catalog.roles["orchestrator"].models == ("glm-5-3",)
    assert (
        catalog.roles["orchestrator"].description
        == SHIPPED_CATALOG.roles["orchestrator"].description
    )
    assert catalog.roles["advisor"] == SHIPPED_CATALOG.roles["advisor"]
    with pytest.raises(ValueError, match=r"migration required.*\[roles\]"):
        merge_catalog_overlay(SHIPPED_CATALOG, {"tags": {}})
    with pytest.raises(ValueError, match="remove aliases"):
        merge_catalog_overlay(SHIPPED_CATALOG, {"models": {"glm-5-3": {"aliases": []}}})


def test_role_priority_skips_disabled_first_member() -> None:
    catalog = merge_catalog_overlay(
        SHIPPED_CATALOG,
        {
            "models": {"glm-5-3": {"disabled": True}},
            "roles": {"priority": {"models": ["glm-5-3", "gpt-6-astra"]}},
        },
    )
    assert (
        ModelResolver(CatalogSnapshot(catalog, "test")).resolve("@priority").base_model
        == "gpt-6-astra"
    )


def test_role_priority_skips_allowlist_excluded_first_member() -> None:
    catalog = merge_catalog_overlay(
        SHIPPED_CATALOG, {"roles": {"priority": {"models": ["glm-5-3", "gpt-6-astra"]}}}
    )
    assert (
        ModelResolver(CatalogSnapshot(catalog, "test"))
        .resolve("@priority", allowed_models=["gpt-6-astra"])
        .base_model
        == "gpt-6-astra"
    )
