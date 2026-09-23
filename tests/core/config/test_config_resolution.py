from __future__ import annotations

from pathlib import Path
import tomllib

import pytest

from chartreux.core.config.builder import ConfigBuilder
from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
from chartreux.core.config.layers.default import DefaultConfigLayer
from chartreux.core.config.layers.environment import EnvironmentLayer
from chartreux.core.config.layers.overrides import OverridesLayer
from chartreux.core.config.layers.project import ProjectConfigLayer
from chartreux.core.config.layers.user import UserConfigLayer
from chartreux.core.config.orchestrator import ConfigOrchestrator
from chartreux.core.config.patch import AddOperationPatch
from chartreux.core.model_catalog.loader import load_catalog
from chartreux.core.trusted_folders import TrustedFoldersManager

pytestmark = pytest.mark.asyncio


async def _build(*layers: object) -> ChartreuxConfigSchema:
    builder = ConfigBuilder(ChartreuxConfigSchema, catalog_snapshot=load_catalog())
    builder.add_layers([DefaultConfigLayer(schema=ChartreuxConfigSchema), *layers])  # type: ignore[arg-type]
    return await builder.build()


async def test_shipped_catalog_resolves_selection_to_wire_name_and_display_provider() -> (
    None
):
    config = await _build(OverridesLayer(data={"active_model": "glm-5-3"}))
    model = config.get_active_model()
    assert (model.alias, model.name, model.provider) == (
        "glm-5-3",
        "zai-glm-5-3",
        "mistral/default",
    )


@pytest.mark.parametrize(
    ("user", "project", "environment", "expected"),
    [
        ("glm-5-3", None, None, "glm-5-3"),
        ("glm-5-3", "gpt-6-astra", None, "gpt-6-astra"),
        ("glm-5-3", "gpt-6-astra", "glm-5-3", "glm-5-3"),
    ],
)
async def test_selection_precedence_across_user_project_and_environment(
    user: str,
    project: str | None,
    environment: str | None,
    expected: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_path = tmp_path / "user.toml"
    user_path.write_text(f'active_model = "{user}"\n')
    layers: list[object] = [UserConfigLayer(path=user_path)]
    if project is not None:
        root = tmp_path / "project"
        path = root / ".chartreux" / "config.toml"
        path.parent.mkdir(parents=True)
        path.write_text(f'active_model = "{project}"\n')
        trust = TrustedFoldersManager()
        trust.trust_for_session(root)
        layers.append(ProjectConfigLayer(path=root, trust_store=trust))
    if environment is not None:
        monkeypatch.setenv("CHARTREUX_ACTIVE_MODEL", environment)
        layers.append(EnvironmentLayer(schema=ChartreuxConfigSchema))
    assert (await _build(*layers)).get_active_model().alias == expected


async def test_untrusted_project_cannot_override_user_selection(tmp_path: Path) -> None:
    user = tmp_path / "user.toml"
    user.write_text('active_model = "glm-5-3"\n')
    root = tmp_path / "project"
    project = root / ".chartreux" / "config.toml"
    project.parent.mkdir(parents=True)
    project.write_text('active_model = "gpt-6-astra"\n')
    config = await _build(
        UserConfigLayer(path=user),
        ProjectConfigLayer(path=root, trust_store=TrustedFoldersManager()),
    )
    assert config.get_active_model().alias == "glm-5-3"


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
        field: "glm-5-3" if field != "allowed_models" else ["glm-5-3", "gpt-6-astra"]
    }
    higher = {field: "gpt-6-astra" if field != "allowed_models" else ["gpt-6-astra"]}
    config = await _build(
        OverridesLayer(data=lower, name="lower"),
        OverridesLayer(data=higher, name="higher"),
    )
    assert getattr(config, field) == higher[field]


async def test_runtime_selection_patch_persists_only_selection_table(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.toml"
    user = UserConfigLayer(path=path, name="user")
    session = OverridesLayer(data={}, name="session")
    orch = await ConfigOrchestrator.create(
        schema=ChartreuxConfigSchema,
        layers=[DefaultConfigLayer(schema=ChartreuxConfigSchema), user, session],
        default_layer_resolver=lambda: user,
    )
    assert not await orch.set_field("/thinking_overrides/glm-5-3", "low")
    assert not await orch.set_field("/active_model", "glm-5-3")
    persisted = tomllib.loads(path.read_text())
    assert persisted == {
        "active_model": "glm-5-3",
        "thinking_overrides": {"glm-5-3": "low"},
    }
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
    path.write_text(f'{field} = []\nactive_model = "glm-5-3"\n')
    with pytest.raises(Exception, match="chartreux models migrate"):
        await _build(
            UserConfigLayer(path=path),
            OverridesLayer(data={"active_model": "gpt-6-astra"}),
        )
