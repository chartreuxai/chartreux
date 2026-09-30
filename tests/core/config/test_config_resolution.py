from __future__ import annotations

from pathlib import Path

import pytest

from chartreux.core.config.builder import ConfigBuilder
from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
from chartreux.core.config.layers.default import DefaultConfigLayer
from chartreux.core.config.layers.overrides import OverridesLayer
from chartreux.core.config.layers.project import ProjectConfigLayer
from chartreux.core.config.layers.user import UserConfigLayer
from chartreux.core.config.models import MissingAPIKeyError
from chartreux.core.config.orchestrator import ConfigOrchestrator
from chartreux.core.config.patch import AddOperationPatch
from chartreux.core.model_catalog.defaults import SHIPPED_CATALOG
from chartreux.core.model_catalog.loader import (
    CatalogSnapshot,
    load_catalog,
    merge_catalog_overlay,
)
from chartreux.core.model_catalog.schema import ModelCatalog
from chartreux.core.trusted_folders import TrustedFoldersManager

pytestmark = pytest.mark.asyncio


async def _build(
    *layers: object, snapshot: CatalogSnapshot | None = None
) -> ChartreuxConfigSchema:
    builder = ConfigBuilder(
        ChartreuxConfigSchema, catalog_snapshot=snapshot or load_catalog()
    )
    builder.add_layers([DefaultConfigLayer(schema=ChartreuxConfigSchema), *layers])  # type: ignore[arg-type]
    return await builder.build()


def _two_model_snapshot() -> CatalogSnapshot:
    """A snapshot with a second resolvable model; the shipped catalog has one."""
    catalog = merge_catalog_overlay(
        SHIPPED_CATALOG,
        {
            "providers": {"second": {"api_base": "https://second.test/v1"}},
            "models": {
                "second-model": {
                    "deployments": [{"provider": "second", "name": "second-model"}]
                }
            },
        },
    )
    return CatalogSnapshot(catalog, "test-two-models")


async def test_shipped_catalog_resolves_selection_to_wire_name_and_display_provider() -> (
    None
):
    config = await _build(OverridesLayer(data={"active_model": "glm-5-3"}))
    model = config.get_active_model()
    assert (model.alias, model.name, model.provider) == (
        "glm-5-3",
        "zai-glm-5-3",
        "mistral",
    )


async def test_orchestrator_preset_never_skips_missing_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "chartreux.core.config.chartreux_schema.resolve_api_key", lambda _env: None
    )
    catalog = merge_catalog_overlay(
        SHIPPED_CATALOG,
        {
            "providers": {
                "keyless": {
                    "api_base": "http://127.0.0.1:11434/v1",
                    "api_key_env_var": "",
                }
            },
            "models": {
                "local": {"deployments": [{"provider": "keyless", "name": "local"}]}
            },
            "roles": {"orchestrator": {"model": "glm-5-3", "thinking": "high"}},
        },
    )
    config = await _build(snapshot=CatalogSnapshot(catalog, "credential-preset"))
    assert config.resolve_default_model_alias() == "glm-5-3"
    assert config.get_active_model().provider == "mistral"
    with pytest.raises(MissingAPIKeyError, match="MISTRAL_API_KEY"):
        config.require_active_provider_api_key()


async def test_orchestrator_uses_first_enabled_deployment_of_selected_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "chartreux.core.config.chartreux_schema.resolve_api_key", lambda _env: None
    )
    raw = SHIPPED_CATALOG.model_dump()
    raw["providers"]["keyless"] = {
        "api_base": "http://127.0.0.1:11434/v1",
        "api_key_env_var": "",
    }
    raw["models"]["glm-5-3"]["deployments"] = [
        *raw["models"]["glm-5-3"]["deployments"],
        {"provider": "keyless", "name": "glm-local"},
    ]
    catalog = ModelCatalog.model_validate(raw)
    config = await _build(snapshot=CatalogSnapshot(catalog, "deployment-selection"))
    assert config.get_active_model().provider == "mistral"


async def test_semantic_orchestrator_without_ready_candidate_keeps_missing_key_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "chartreux.core.config.chartreux_schema.resolve_api_key", lambda _env: None
    )
    config = await _build(snapshot=CatalogSnapshot(SHIPPED_CATALOG, "missing-key"))
    assert config.get_active_model().provider == "mistral"
    with pytest.raises(MissingAPIKeyError, match="MISTRAL_API_KEY"):
        config.require_active_provider_api_key()


@pytest.mark.parametrize("layer_kind", ["user", "project"])
@pytest.mark.parametrize(
    ("field_text", "error_hint"),
    [
        ('active_model = "glm-5-3"', "roles.orchestrator.*models.toml"),
        ('[thinking_overrides]\nglm-5-3 = "low"', "models.toml.*session"),
    ],
)
async def test_persisted_model_overrides_rejected_with_preset_guidance(
    layer_kind: str, field_text: str, error_hint: str, tmp_path: Path
) -> None:
    if layer_kind == "user":
        path = tmp_path / "user.toml"
        path.write_text(f"{field_text}\n")
        layer = UserConfigLayer(path=path)
    else:
        root = tmp_path / "project"
        path = root / ".chartreux" / "config.toml"
        path.parent.mkdir(parents=True)
        path.write_text(f"{field_text}\n")
        trust = TrustedFoldersManager()
        trust.trust_for_session(root)
        layer = ProjectConfigLayer(path=root, trust_store=trust)
    with pytest.raises(ValueError, match=error_hint):
        await _build(layer)


async def test_session_model_override_wins_without_changing_default() -> None:
    snapshot = _two_model_snapshot()
    config = await _build(
        OverridesLayer(data={"active_model": "second-model"}), snapshot=snapshot
    )
    assert config.get_active_model().alias == "second-model"
    assert config.resolve_default_model_alias() == "glm-5-3"


@pytest.mark.parametrize("expression", ["missing", "@missing", "provider/missing"])
async def test_unknown_selection_is_preserved_and_fails_only_at_resolution(
    expression: str,
) -> None:
    config = await _build(OverridesLayer(data={"active_model": expression}))
    assert config.active_model == expression
    with pytest.raises(ValueError):
        config.get_active_model()


async def test_thinking_overrides_are_layer_merged_and_materialized_without_catalog_mutation() -> (
    None
):
    config = await _build(
        OverridesLayer(data={"thinking_overrides": {"glm-5-3": "low"}})
    )
    assert config.get_active_model().thinking == "low"
    assert config.catalog_snapshot is not None
    assert config.catalog_snapshot.catalog.models["glm-5-3"].thinking == "high"


@pytest.mark.parametrize(
    "field", ["active_model", "compaction_model", "allowed_models"]
)
async def test_selection_fields_replace_instead_of_merging(field: str) -> None:
    lower = {
        field: "glm-5-3" if field != "allowed_models" else ["glm-5-3", "other-model"]
    }
    higher = {field: "other-model" if field != "allowed_models" else ["other-model"]}
    config = await _build(
        OverridesLayer(data=lower, name="lower"),
        OverridesLayer(data=higher, name="higher"),
    )
    assert getattr(config, field) == higher[field]


async def test_runtime_selection_patch_stays_in_session_layer(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    user = UserConfigLayer(path=path, name="user")
    session = OverridesLayer(data={}, name="session")
    orch = await ConfigOrchestrator.create(
        schema=ChartreuxConfigSchema,
        layers=[DefaultConfigLayer(schema=ChartreuxConfigSchema), user, session],
        default_layer_resolver=lambda: user,
    )
    assert not await orch.set_field(
        "/thinking_overrides/glm-5-3", "low", target_layer="session"
    )
    assert not await orch.set_field("/active_model", "glm-5-3", target_layer="session")
    assert not path.exists()
    assert orch.config.get_active_model().thinking == "low"


async def test_save_replaces_top_level_list_and_reload_keeps_project_user_resolution(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.toml"
    path.write_text('disabled_agents = ["first", "other"]\n')
    user = UserConfigLayer(path=path, name="user")
    orch = await ConfigOrchestrator.create(
        schema=ChartreuxConfigSchema,
        layers=[DefaultConfigLayer(schema=ChartreuxConfigSchema), user],
        default_layer_resolver=lambda: user,
    )
    revision = user.fingerprint
    assert revision is not None
    result = await orch.save(
        [AddOperationPatch(path="/disabled_agents", value=["first"])],
        target="user",
        expected_revision=revision,
        reason="test",
    )
    assert (result.persistence, result.application) == ("saved", "applied")
    await orch.reload()
    assert orch.config.disabled_agents == ["first"]


@pytest.mark.parametrize("field", ["models", "providers"])
async def test_legacy_tables_fail_before_selection_layer_can_shadow_them(
    field: str, tmp_path: Path
) -> None:
    path = tmp_path / "config.toml"
    path.write_text(f"{field} = []\n")
    with pytest.raises(Exception, match="chartreux models migrate"):
        await _build(
            UserConfigLayer(path=path),
            OverridesLayer(data={"active_model": "other-model"}),
        )
