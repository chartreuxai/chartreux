from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError
import pytest

from chartreux.core.config._catalog import validate_catalog_scope
from chartreux.core.config.builder import ConfigBuilder
from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
from chartreux.core.config.harness_files import HarnessFilesManager
from chartreux.core.config.layer import LayerImplementationError
from chartreux.core.config.layers.default import DefaultConfigLayer
from chartreux.core.config.layers.environment import EnvironmentLayer
from chartreux.core.config.layers.overrides import OverridesLayer
from chartreux.core.config.layers.project import ProjectConfigLayer
from chartreux.core.config.layers.user import UserConfigLayer
from chartreux.core.config.orchestrator import (
    ConfigOrchestrator,
    ConfigPatchValidationError,
)
from chartreux.core.config.patch import (
    AddOperationPatch,
    RemoveOperationPatch,
    ReplaceOperationPatch,
)
from chartreux.core.model_catalog.loader import CatalogLoadError, load_catalog
from chartreux.core.trusted_folders import TrustedFoldersManager

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize("field", ["models", "providers"])
@pytest.mark.parametrize("value", [{}, [], "private-catalog-payload"])
async def test_non_catalog_layers_reject_legacy_tables_before_payload_decoding(
    field: str, value: object, tmp_path: Path
) -> None:
    layers = [
        OverridesLayer(data={field: value}),
        ProjectConfigLayer(path=tmp_path, trust_store=TrustedFoldersManager()),
        EnvironmentLayer(schema=ChartreuxConfigSchema),
    ]
    for layer in layers:
        with pytest.raises(ValidationError, match="chartreux models migrate") as raised:
            validate_catalog_scope(
                ChartreuxConfigSchema, {field}, layer=layer, source=layer.name
            )
        assert "private" not in raised.value.json()


@pytest.mark.parametrize("field", ["models", "providers"])
async def test_user_and_project_config_tables_are_migration_errors_not_catalog_authority(
    field: str, tmp_path: Path
) -> None:
    user_path = tmp_path / "user.toml"
    user_path.write_text(f"{field} = []\ntheme = 'user'\n")
    project_root = tmp_path / "project"
    project_path = project_root / ".chartreux" / "config.toml"
    project_path.parent.mkdir(parents=True)
    project_path.write_text(f"{field} = []\ntheme = 'project'\n")
    trust = TrustedFoldersManager()
    trust.trust_for_session(project_root)
    for layer in [
        UserConfigLayer(path=user_path),
        ProjectConfigLayer(path=project_root, trust_store=trust),
    ]:
        builder = ConfigBuilder(ChartreuxConfigSchema)
        builder.add_layers([DefaultConfigLayer(schema=ChartreuxConfigSchema), layer])
        with pytest.raises(ValidationError, match="chartreux models migrate"):
            await builder.build()


async def test_models_toml_is_the_only_catalog_overlay_and_config_remains_selections_only(
    tmp_path: Path,
) -> None:
    catalog_path = tmp_path / "models.toml"
    catalog_path.write_text("""
[models.glm-5-3]
thinking = "low"
[models.glm-5-3.deployments]
""")
    # A malformed overlay has a typed catalog error; config.toml is not consulted.
    with pytest.raises(CatalogLoadError):
        load_catalog(catalog_path)
    valid = tmp_path / "valid-models.toml"
    valid.write_text('[models.glm-5-3]\nthinking = "low"\n')
    assert load_catalog(valid).catalog.models["glm-5-3"].thinking == "low"


@pytest.mark.parametrize("field", ["models", "providers"])
@pytest.mark.parametrize("operation", ["set", "add", "replace", "remove", "root"])
async def test_catalog_patch_attempts_do_not_mutate_accepted_config_or_disk(
    field: str, operation: str, tmp_path: Path
) -> None:
    path = tmp_path / "config.toml"
    before_bytes = b'# retained\ntheme = "original"\n'
    path.write_bytes(before_bytes)
    user = UserConfigLayer(path=path, name="user")
    session = OverridesLayer(data={}, name="session")
    orch = await ConfigOrchestrator.create(
        schema=ChartreuxConfigSchema,
        layers=[DefaultConfigLayer(schema=ChartreuxConfigSchema), user, session],
        default_layer_resolver=lambda: session,
    )
    before = orch.config, orch.accepted_token
    if operation == "set":
        action = lambda: orch.set_field(f"/{field}", {})
    elif operation == "add":
        action = lambda: orch.apply_patch(
            [AddOperationPatch(path=f"/{field}", value={})], "test"
        )
    elif operation == "replace":
        action = lambda: orch.apply_patch(
            [ReplaceOperationPatch(path=f"/{field}", value={})], "test"
        )
    elif operation == "remove":
        action = lambda: orch.apply_patch(
            [RemoveOperationPatch(path=f"/{field}")], "test"
        )
    else:
        action = lambda: orch.apply_patch(
            [ReplaceOperationPatch(path="", value={field: {}})], "test"
        )
    with pytest.raises(ConfigPatchValidationError, match="chartreux models migrate"):
        await action()
    assert orch.config is before[0]
    assert orch.accepted_token is before[1]
    assert path.read_bytes() == before_bytes


async def test_failed_legacy_reload_preserves_last_accepted_selection_state(
    tmp_path: Path,
) -> None:
    session = OverridesLayer(data={"theme": "accepted"})
    orch = await ConfigOrchestrator.create(
        schema=ChartreuxConfigSchema,
        layers=[DefaultConfigLayer(schema=ChartreuxConfigSchema), session],
        default_layer_resolver=lambda: session,
    )
    before = orch.config, orch.accepted_token
    session._data["models"] = {}  # prospective invalid source, not accepted cache
    with pytest.raises(ValidationError, match="chartreux models migrate"):
        await orch.reload()
    assert (orch.config, orch.accepted_token) == before


async def test_project_discovery_write_keeps_selection_authority_at_discovered_file(
    tmp_path: Path,
) -> None:
    ancestor = tmp_path / ".chartreux" / "config.toml"
    ancestor.parent.mkdir()
    ancestor.write_text('theme = "ancestor"\n')
    child = tmp_path / "child"
    child.mkdir()
    trust = TrustedFoldersManager()
    trust.trust_for_session(tmp_path)
    project = ProjectConfigLayer(path=child, trust_store=trust)
    orch = await ConfigOrchestrator.create(
        schema=ChartreuxConfigSchema,
        layers=[DefaultConfigLayer(schema=ChartreuxConfigSchema), project],
        default_layer_resolver=lambda: project,
    )
    assert orch.config.theme == "ancestor"
    assert not await orch.set_field("/theme", "project")
    assert 'theme = "project"' in ancestor.read_text()
    assert not (child / ".chartreux").exists()


async def test_environment_catalog_name_is_redacted_before_settings_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ChArTrEuX_MODELS__private_alias", "private-invalid-json")
    layer = EnvironmentLayer(schema=ChartreuxConfigSchema)
    with pytest.raises(LayerImplementationError) as raised:
        await layer.load()
    assert "private" not in str(raised.value)
    assert layer.cached_data is None


@pytest.mark.parametrize("value", [{}, [], "dispatch-payload"])
async def test_dispatch_tables_are_rejected_before_payload_decoding(
    value: object, tmp_path: Path
) -> None:
    layers = [
        OverridesLayer(data={"dispatch": value}),
        ProjectConfigLayer(path=tmp_path, trust_store=TrustedFoldersManager()),
        EnvironmentLayer(schema=ChartreuxConfigSchema),
    ]
    for layer in layers:
        with pytest.raises(
            ValidationError, match="The \\[dispatch\\] table is retired"
        ) as raised:
            validate_catalog_scope(
                ChartreuxConfigSchema, {"dispatch"}, layer=layer, source=layer.name
            )
        assert "[orchestration.routes.<purpose>]" in str(raised.value)
        assert "dispatch-payload" not in raised.value.json()


async def test_user_and_project_config_dispatch_tables_are_rejected(
    tmp_path: Path,
) -> None:
    user_path = tmp_path / "user.toml"
    user_path.write_text('[dispatch]\nmode = "standalone"\ntheme = "user"\n')
    project_root = tmp_path / "project"
    project_path = project_root / ".chartreux" / "config.toml"
    project_path.parent.mkdir(parents=True)
    project_path.write_text('[dispatch]\nmode = "standalone"\n')
    trust = TrustedFoldersManager()
    trust.trust_for_session(project_root)
    for layer in [
        UserConfigLayer(path=user_path),
        ProjectConfigLayer(path=project_root, trust_store=trust),
    ]:
        builder = ConfigBuilder(ChartreuxConfigSchema)
        builder.add_layers([DefaultConfigLayer(schema=ChartreuxConfigSchema), layer])
        with pytest.raises(
            ValidationError, match="The \\[dispatch\\] table is retired"
        ) as raised:
            await builder.build()
        assert "[orchestration.routes.<purpose>]" in str(raised.value)


async def test_runtime_dispatch_table_is_rejected_and_keeps_accepted_state() -> None:
    session = OverridesLayer(data={"theme": "accepted"})
    orch = await ConfigOrchestrator.create(
        schema=ChartreuxConfigSchema,
        layers=[DefaultConfigLayer(schema=ChartreuxConfigSchema), session],
        default_layer_resolver=lambda: session,
    )
    before = orch.config, orch.accepted_token
    session._data["dispatch"] = {"mode": "standalone"}
    with pytest.raises(
        ValidationError, match="The \\[dispatch\\] table is retired"
    ) as raised:
        await orch.reload()
    assert "[orchestration.routes.<purpose>]" in str(raised.value)
    assert (orch.config, orch.accepted_token) == before


async def test_environment_dispatch_path_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHARTREUX_DISPATCH__MODE", "standalone")
    layer = EnvironmentLayer(schema=ChartreuxConfigSchema)
    with pytest.raises(LayerImplementationError) as raised:
        await layer.load()
    assert "The [dispatch] table is retired" in str(raised.value.__cause__)
    assert "[orchestration.routes.<purpose>]" in str(raised.value.__cause__)
    assert layer.cached_data is None


async def test_dispatch_patch_attempt_is_rejected_without_mutating_disk(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.toml"
    before_bytes = b'# retained\ntheme = "original"\n'
    path.write_bytes(before_bytes)
    user = UserConfigLayer(path=path, name="user")
    session = OverridesLayer(data={}, name="session")
    orch = await ConfigOrchestrator.create(
        schema=ChartreuxConfigSchema,
        layers=[DefaultConfigLayer(schema=ChartreuxConfigSchema), user, session],
        default_layer_resolver=lambda: session,
    )
    with pytest.raises(
        ConfigPatchValidationError, match="The \\[dispatch\\] table is retired"
    ):
        await orch.set_field("/dispatch", {"mode": "standalone"})
    assert path.read_bytes() == before_bytes


async def test_retired_profile_files_are_detected_and_never_load(
    tmp_path: Path,
) -> None:
    # agents/*.toml profiles are retired (package 16a): the directory is
    # detected loudly and the file is left untouched, but nothing loads it —
    # a [dispatch] table inside a profile file can never reach a config
    # source (dispatch authority lives only in the user models.toml overlay).
    project_root = tmp_path / "project"
    agents_dir = project_root / ".chartreux" / "agents"
    agents_dir.mkdir(parents=True)
    profile_path = agents_dir / "dispatching.toml"
    profile_bytes = b'[dispatch]\nmode = "standalone"\n'
    profile_path.write_bytes(profile_bytes)
    session = OverridesLayer(data={})
    orch = await ConfigOrchestrator.create(
        schema=ChartreuxConfigSchema,
        layers=[DefaultConfigLayer(schema=ChartreuxConfigSchema), session],
        default_layer_resolver=lambda: session,
    )

    manager = HarnessFilesManager(
        sources=("user",), cwd=project_root, _additional_dirs=(project_root,)
    )
    assert agents_dir in manager.retired_agent_dirs
    assert profile_path.read_bytes() == profile_bytes
    # The orchestrator built cleanly: the profile file was never consulted, so
    # its dispatch table smuggled no authority into the merged config.
    assert "dispatch" not in orch.config.model_dump()
