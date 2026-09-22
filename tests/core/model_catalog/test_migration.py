from __future__ import annotations

import base64
import multiprocessing
from pathlib import Path
import sys
import tomllib

from pydantic import ValidationError
import pytest

from chartreux.cli.entrypoint import main
from chartreux.core.config.builder import ConfigBuilder
from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
from chartreux.core.config.layers.user import UserConfigLayer
from chartreux.core.config.models import ProviderConfig
from chartreux.core.config.orchestrator import (
    ConfigOrchestrator,
    ConfigPatchValidationError,
)
from chartreux.core.model_catalog.loader import load_catalog
from chartreux.core.model_catalog.migration import (
    MigrationConflictError,
    MigrationError,
    apply_migration,
    plan_migration,
    run_models_cli,
)
from chartreux.setup.auth.api_key_persistence import apply_provider_to_config

_LEGACY = b"""active_model = "friendly"\ncompaction_model = "friendly"\ntheme = "dark"\n[thinking_overrides]\nfriendly = "low"\n[[providers]]\nname = "test"\napi_base = "https://example.test/v1"\napi_key_env_var = "TEST_API_KEY"\n[[models]]\nname = "base"\nprovider = "test"\nalias = "friendly"\ntemperature = 0.3\nthinking = "low"\n"""

# These provider and model fields are emitted by main's create_default_config(),
# including the empty Vertex-era provider values.
_MAIN_GENERATED_CONFIG = b"""active_model = ""\ncompaction_model = ""\n[[providers]]\nname = "mistral"\napi_base = "https://api.mistral.ai/v1"\napi_key_env_var = "MISTRAL_API_KEY"\napi_style = "openai"\nbackend = "mistral"\nreasoning_field_name = "reasoning_content"\nemits_finish_reason = true\nproject_id = ""\nregion = ""\n[providers.extra_headers]\n[[models]]\nname = "glm-5-2"\nprovider = "mistral"\nalias = "glm-5-2"\ntemperature = 0.2\ninput_price = 1.4\noutput_price = 4.4\ncached_input_price = 0.14\nthinking = "high"\nsupports_images = false\nauto_compact_threshold = 400000\n"""


def _paths(tmp_path: Path) -> tuple[Path, Path]:
    return tmp_path / "config.toml", tmp_path / "models.toml"


def _apply_in_process(
    config_path: Path,
    catalog_path: Path,
    catalog_written: multiprocessing.Queue[None] | None,
    release_first: multiprocessing.Queue[None] | None,
    result: multiprocessing.Queue[tuple[str, str]],
) -> None:
    from chartreux.core.model_catalog import migration

    original_write = migration._atomic_write

    def delayed_write(path: Path, data: bytes) -> None:
        original_write(path, data)
        if path == catalog_path and catalog_written is not None:
            catalog_written.put(None)
            assert release_first is not None
            release_first.get(timeout=5)

    if catalog_written is not None:
        migration._atomic_write = delayed_write
    try:
        result.put((
            "success",
            apply_migration(plan_migration(config_path, catalog_path)),
        ))
    except Exception as exc:
        result.put((type(exc).__name__, str(exc)))


def test_migration_preview_makes_no_changes_and_shows_plan(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    config, catalog = _paths(tmp_path)
    config.write_bytes(_LEGACY)
    monkeypatch.setenv("CHARTREUX_HOME", str(tmp_path))

    run_models_cli(["migrate"])

    assert config.read_bytes() == _LEGACY
    assert not catalog.exists()
    assert "Preview: would write" in capsys.readouterr().out


def test_migration_apply_round_trip_writes_catalog_and_canonicalizes_selections(
    tmp_path: Path,
) -> None:
    config, catalog = _paths(tmp_path)
    config.write_bytes(_LEGACY)

    apply_migration(plan_migration(config, catalog))

    migrated = tomllib.loads(catalog.read_text())
    assert (
        migrated["providers"]["test/default"]["api_base"] == "https://example.test/v1"
    )
    assert "aliases" not in migrated["models"]["base"]
    assert migrated["models"]["base"]["deployments"][0]["name"] == "base"
    cleaned = config.read_text()
    assert "providers" not in cleaned and "[[models]]" not in cleaned
    assert 'active_model = "base"' in cleaned
    assert 'compaction_model = "base"' in cleaned
    assert 'base = "low"' in cleaned


def test_migration_canonicalizes_exact_aliases_in_allowed_models(
    tmp_path: Path,
) -> None:
    config, catalog = _paths(tmp_path)
    config.write_bytes(b'allowed_models = ["friendly"]\n' + _LEGACY)

    apply_migration(plan_migration(config, catalog))

    assert tomllib.loads(config.read_text())["allowed_models"] == ["base"]


def test_migration_reports_alias_matching_allowlist_patterns(tmp_path: Path) -> None:
    config, catalog = _paths(tmp_path)
    config.write_bytes(b'allowed_models = ["friend*", "re:friendly"]\n' + _LEGACY)

    plan = plan_migration(config, catalog)

    assert tomllib.loads(plan.cleaned_config.decode())["allowed_models"] == [
        "friend*",
        "re:friendly",
    ]
    assert plan.warnings == (
        "allowed_models pattern 'friend*' may reference removed aliases; use canonical names ['base'].",
        "allowed_models pattern 're:friendly' may reference removed aliases; use canonical names ['base'].",
    )


def test_migration_tags_only_config_migrates_tags_to_roles(tmp_path: Path) -> None:
    config, catalog = _paths(tmp_path)
    config.write_text('[tags]\nworker = ["glm-5-2"]\n')

    apply_migration(plan_migration(config, catalog))

    assert tomllib.loads(catalog.read_text())["roles"]["worker"] == {
        "description": "",
        "models": ["glm-5-3"],
    }
    assert "tags" not in tomllib.loads(config.read_text())


def test_migration_warns_about_retargeted_references_and_orphaned_preserved_tables(
    tmp_path: Path,
) -> None:
    config, catalog = _paths(tmp_path)
    config.write_text(
        'active_model = "glm-5-2"\n'
        "[tags]\n"
        'worker = ["glm-5-2"]\n'
        "[[providers]]\n"
        'name = "mistral"\n'
        'api_base = "https://api.mistral.ai/v1"\n'
        "[[models]]\n"
        'name = "glm-5-2"\n'
        'provider = "mistral"\n'
        'alias = "legacy-glm"\n'
    )

    plan = plan_migration(config, catalog)

    assert plan.warnings == (
        "active_model selection 'glm-5-2' was retargeted to 'glm-5-3'.",
        "role 'worker' member 'glm-5-2' was retargeted to 'glm-5-3'.",
        "Preserved user model table 'glm-5-2' is no longer referenced after canonicalization.",
    )


def test_migration_rejects_role_member_without_a_model_table(tmp_path: Path) -> None:
    config, catalog = _paths(tmp_path)
    config.write_text('[tags]\nworker = ["mistral-small"]\n')

    with pytest.raises(MigrationError, match="unknown model"):
        plan_migration(config, catalog)


def test_migration_preserves_explicit_removed_shipped_model_but_canonicalizes_references(
    tmp_path: Path,
) -> None:
    config, catalog = _paths(tmp_path)
    config.write_text(
        'active_model = "glm-5-2"\n'
        'compaction_model = "glm-5-2"\n'
        'allowed_models = ["glm-5-2"]\n'
        "[thinking_overrides]\n"
        'glm-5-2 = "high"\n'
        "[tags]\n"
        'worker = ["glm-5-2"]\n'
        "[[providers]]\n"
        'name = "mistral"\n'
        'api_base = "https://api.mistral.ai/v1"\n'
        "[[models]]\n"
        'name = "glm-5-2"\n'
        'provider = "mistral"\n'
        'alias = "legacy-glm"\n'
    )

    apply_migration(plan_migration(config, catalog))

    migrated = tomllib.loads(catalog.read_text())
    assert "glm-5-2" in migrated["models"]
    assert migrated["roles"]["worker"]["models"] == ["glm-5-3"]
    cleaned = tomllib.loads(config.read_text())
    assert cleaned["active_model"] == "glm-5-3"
    assert cleaned["compaction_model"] == "glm-5-3"
    assert cleaned["allowed_models"] == ["glm-5-3"]
    assert cleaned["thinking_overrides"] == {"glm-5-3": "high"}


def test_migration_preserves_explicit_removed_mistral_small_model(
    tmp_path: Path,
) -> None:
    config, catalog = _paths(tmp_path)
    config.write_text(
        "[tags]\n"
        'worker = ["mistral-small"]\n'
        "[[providers]]\n"
        'name = "mistral"\n'
        'api_base = "https://api.mistral.ai/v1"\n'
        "[[models]]\n"
        'name = "mistral-small-latest"\n'
        'provider = "mistral"\n'
        'alias = "mistral-small"\n'
    )

    apply_migration(plan_migration(config, catalog))

    migrated = tomllib.loads(catalog.read_text())
    assert "mistral-small-latest" in migrated["models"]
    assert migrated["roles"]["worker"]["models"] == ["mistral-small-latest"]


def test_migration_main_generated_default_config_drops_empty_vertex_fields(
    tmp_path: Path,
) -> None:
    config, catalog = _paths(tmp_path)
    config.write_bytes(_MAIN_GENERATED_CONFIG)

    apply_migration(plan_migration(config, catalog))

    migrated = tomllib.loads(catalog.read_text())
    provider = migrated["providers"]["mistral/default"]
    assert "project_id" not in provider and "region" not in provider
    assert tomllib.loads(config.read_text())["active_model"] == ""


def test_migration_nonempty_legacy_project_id_is_actionable_conflict(
    tmp_path: Path,
) -> None:
    config, catalog = _paths(tmp_path)
    config.write_bytes(
        _MAIN_GENERATED_CONFIG.replace(b'project_id = ""', b'project_id = "my-project"')
    )

    with pytest.raises(MigrationConflictError, match=r"project_id.*my-project"):
        plan_migration(config, catalog)


@pytest.mark.parametrize(("wire_name", "base_name"), (("zai-glm-5-3", "glm-5-3"),))
def test_migration_reconciles_shipped_wire_name_and_preserves_overrides(
    tmp_path: Path, wire_name: str, base_name: str
) -> None:
    config, catalog = _paths(tmp_path)
    config.write_text(
        "[[providers]]\n"
        'name = "mistral"\n'
        'api_base = "https://api.mistral.ai/v1"\n'
        "[[models]]\n"
        f'name = "{wire_name}"\n'
        'provider = "mistral"\n'
        'alias = "custom"\n'
        "temperature = 0.7\n"
        'thinking = "low"\n'
        "input_price = 1.2\n"
        "output_price = 3.4\n"
        "cached_input_price = 0.5\n"
    )

    apply_migration(plan_migration(config, catalog))

    migrated = tomllib.loads(catalog.read_text())["models"]
    assert wire_name not in migrated
    assert migrated[base_name]["thinking"] == "low"
    assert migrated[base_name]["temperature"] == 0.7
    assert "aliases" not in migrated[base_name]
    assert migrated[base_name]["deployments"] == [
        {
            "provider": "mistral/default",
            "prices": {"input": 1.2, "output": 3.4, "cached_input": 0.5},
        }
    ]


def test_models_migrate_help_has_no_duplicate_migrate_path(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit, match="0"):
        run_models_cli(["migrate", "--help"])

    assert "usage: chartreux models migrate" in capsys.readouterr().out


def test_migration_cli_apply_reenters_lock_in_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config, catalog = _paths(tmp_path)
    config.write_bytes(_LEGACY)
    monkeypatch.setenv("CHARTREUX_HOME", str(tmp_path))

    run_models_cli(["migrate", "--apply"])

    captured = capsys.readouterr()
    assert "Migrated catalog to" in captured.out
    assert "Migration already in progress." not in captured.err
    assert catalog.exists()


def test_migration_existing_models_toml_is_conflict_without_overwrite(
    tmp_path: Path,
) -> None:
    config, catalog = _paths(tmp_path)
    config.write_bytes(_LEGACY)
    catalog.write_text("[providers]\n")

    with pytest.raises(MigrationConflictError, match="already exists"):
        apply_migration(plan_migration(config, catalog))

    assert catalog.read_text() == "[providers]\n"
    assert config.read_bytes() == _LEGACY


def test_migration_rerun_finishes_interrupted_apply_without_duplication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, catalog = _paths(tmp_path)
    config.write_bytes(_LEGACY)
    plan = plan_migration(config, catalog)
    from chartreux.core.model_catalog import migration

    original_write = migration._atomic_write
    calls = 0

    def interrupt(path: Path, data: bytes) -> None:
        nonlocal calls
        calls += 1
        original_write(path, data)
        if calls == 3:
            raise KeyboardInterrupt

    monkeypatch.setattr(migration, "_atomic_write", interrupt)
    with pytest.raises(KeyboardInterrupt):
        apply_migration(plan)
    monkeypatch.setattr(migration, "_atomic_write", original_write)

    assert catalog.exists()
    assert config.read_bytes() == _LEGACY
    assert (
        apply_migration(plan)
        == "Recovered interrupted migration; config.toml cleanup completed."
    )
    assert not (tmp_path / ".models-migration-recovery.toml").exists()
    assert len(tomllib.loads(catalog.read_text())["models"]) == 1


def test_migration_recovery_rejects_changed_models_toml(tmp_path: Path) -> None:
    config, catalog = _paths(tmp_path)
    config.write_bytes(_LEGACY)
    plan = plan_migration(config, catalog)
    from chartreux.core.model_catalog import migration

    original_write = migration._atomic_write

    def interrupt_after_marker(path: Path, data: bytes) -> None:
        original_write(path, data)
        if path.name == ".models-migration-recovery.toml":
            raise KeyboardInterrupt

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(migration, "_atomic_write", interrupt_after_marker)
        with pytest.raises(KeyboardInterrupt):
            apply_migration(plan)

    catalog.write_text("[providers.replaced/default]\n")

    with pytest.raises(MigrationConflictError, match="changed models.toml"):
        apply_migration(plan)

    assert config.read_bytes() == _LEGACY
    assert (tmp_path / ".models-migration-recovery.toml").exists()


@pytest.mark.parametrize("cleaned_payload", (b"\xff", b"[broken"))
def test_migration_recovery_rejects_invalid_cleaned_config_payload(
    tmp_path: Path, cleaned_payload: bytes
) -> None:
    config, catalog = _paths(tmp_path)
    config.write_bytes(_LEGACY)
    plan = plan_migration(config, catalog)
    from chartreux.core.model_catalog import migration

    catalog.write_bytes(migration.tomli_w.dumps(plan.catalog).encode())
    marker = tmp_path / ".models-migration-recovery.toml"
    marker.write_bytes(
        migration._recovery_marker_data(plan).replace(
            base64.b64encode(plan.cleaned_config), base64.b64encode(cleaned_payload)
        )
    )

    with pytest.raises(MigrationError, match=str(plan.backup_path)):
        apply_migration(plan)

    assert config.read_bytes() == _LEGACY
    assert marker.exists()


def test_migration_apply_replans_from_current_config(tmp_path: Path) -> None:
    config, catalog = _paths(tmp_path)
    config.write_bytes(_LEGACY)
    preview = plan_migration(config, catalog)
    edited = _LEGACY.replace(b'theme = "dark"\n', b'theme = "dark"\nuser_edit = true\n')
    config.write_bytes(edited)

    apply_migration(preview)

    assert tomllib.loads(config.read_text())["user_edit"] is True
    assert b"user_edit = true" in preview.backup_path.read_bytes()


def test_migration_recovery_rejects_valid_cleaned_payload_with_wrong_digest(
    tmp_path: Path,
) -> None:
    config, catalog = _paths(tmp_path)
    config.write_bytes(_LEGACY)
    plan = plan_migration(config, catalog)
    from chartreux.core.model_catalog import migration

    catalog.write_bytes(migration.tomli_w.dumps(plan.catalog).encode())
    marker = tmp_path / ".models-migration-recovery.toml"
    marker.write_bytes(
        migration._recovery_marker_data(plan).replace(
            base64.b64encode(plan.cleaned_config),
            base64.b64encode(b"user_edit = true\n"),
        )
    )

    with pytest.raises(MigrationConflictError, match="changed cleaned config payload"):
        apply_migration(plan)

    assert config.read_bytes() == _LEGACY
    assert marker.exists()


def test_migration_fsyncs_directory_after_each_apply_phase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, catalog = _paths(tmp_path)
    config.write_bytes(_LEGACY)
    plan = plan_migration(config, catalog)
    from chartreux.core.model_catalog import migration

    operations: list[str] = []
    original_write = migration._atomic_write

    def record_write(path: Path, data: bytes) -> None:
        original_write(path, data)
        operations.append(f"write:{path.name}")

    monkeypatch.setattr(migration, "_atomic_write", record_write)
    monkeypatch.setattr(
        migration,
        "_fsync_directory",
        lambda directory: operations.append(f"sync:{directory}"),
    )

    apply_migration(plan)

    assert operations[:8] == [
        f"write:{plan.backup_path.name}",
        f"sync:{tmp_path}",
        "write:models.toml",
        f"sync:{tmp_path}",
        "write:.models-migration-recovery.toml",
        f"sync:{tmp_path}",
        "write:config.toml",
        f"sync:{tmp_path}",
    ]
    assert operations[-1] == f"sync:{tmp_path}"


def test_migration_backup_precedes_mutation_and_preserves_original_bytes(
    tmp_path: Path,
) -> None:
    config, catalog = _paths(tmp_path)
    config.write_bytes(_LEGACY)
    plan = plan_migration(config, catalog)

    apply_migration(plan)

    assert plan.backup_path.read_bytes() == _LEGACY


def test_migration_ambiguous_inference_identity_is_conflict(tmp_path: Path) -> None:
    config, catalog = _paths(tmp_path)
    config.write_bytes(
        _LEGACY
        + b'[[models]]\nname = "base"\nprovider = "other"\nalias = "other-friendly"\ntemperature = 0.9\n'
    )

    with pytest.raises(MigrationConflictError, match="Ambiguous legacy identity"):
        plan_migration(config, catalog)


def test_migration_ambiguous_thinking_override_aliases_are_conflict(
    tmp_path: Path,
) -> None:
    config, catalog = _paths(tmp_path)
    config.write_bytes(
        _LEGACY.replace(b'friendly = "low"', b'friendly = "low"\nbase = "high"')
    )

    with pytest.raises(MigrationConflictError, match="multiple thinking overrides"):
        plan_migration(config, catalog)


def test_migration_preserves_unrelated_settings(tmp_path: Path) -> None:
    config, catalog = _paths(tmp_path)
    config.write_bytes(_LEGACY + b'\n[tools.bash]\ndenylist = ["pwd"]\n')

    apply_migration(plan_migration(config, catalog))

    assert 'theme = "dark"' in config.read_text()
    assert tomllib.loads(config.read_text())["tools"]["bash"] == {"denylist": ["pwd"]}


def test_migration_does_not_rename_unrelated_quoted_alias_key(tmp_path: Path) -> None:
    config, catalog = _paths(tmp_path)
    config.write_bytes(_LEGACY + b'\n[unrelated]\n"friendly" = "preserve"\n')

    apply_migration(plan_migration(config, catalog))

    assert tomllib.loads(config.read_text())["unrelated"]["friendly"] == "preserve"
    assert "base" not in tomllib.loads(config.read_text())["unrelated"]


def test_migration_special_character_alias_round_trips_as_valid_toml(
    tmp_path: Path,
) -> None:
    config, catalog = _paths(tmp_path)
    config.write_text(
        'active_model = "friendly\\"\\\\path"\n'
        'compaction_model = "friendly\\"\\\\path"\n'
        "[thinking_overrides]\n"
        '"friendly\\"\\\\path" = "low"\n'
        '[[providers]]\nname = "test"\napi_base = "https://example.test/v1"\n'
        '[[models]]\nname = "base"\nprovider = "test"\n'
        'alias = "friendly\\"\\\\path"\n'
    )

    apply_migration(plan_migration(config, catalog))

    cleaned = tomllib.loads(config.read_text())
    assert cleaned["active_model"] == "base"
    assert cleaned["thinking_overrides"] == {"base": "low"}
    assert "aliases" not in tomllib.loads(catalog.read_text())["models"]["base"]


def test_migration_concurrent_apply_rejects_second_process(tmp_path: Path) -> None:
    config, catalog = _paths(tmp_path)
    config.write_bytes(_LEGACY)
    context = multiprocessing.get_context("fork")
    catalog_written = context.Queue()
    release_first = context.Queue()
    first_result = context.Queue()
    second_result = context.Queue()

    first = context.Process(
        target=_apply_in_process,
        args=(config, catalog, catalog_written, release_first, first_result),
    )
    first.start()
    catalog_written.get(timeout=5)
    second = context.Process(
        target=_apply_in_process, args=(config, catalog, None, None, second_result)
    )
    second.start()
    second.join(timeout=5)
    release_first.put(None)
    first.join(timeout=5)

    assert not first.is_alive() and not second.is_alive()
    assert second.exitcode == 0
    assert second_result.get(timeout=1) == (
        "MigrationConflictError",
        "Migration already in progress.",
    )
    assert first_result.get(timeout=1)[0] == "success"


@pytest.mark.parametrize(
    "interrupted_name",
    (
        "config.toml.pre-model-catalog-migration.bak",
        "models.toml",
        ".models-migration-recovery.toml",
        "config.toml",
    ),
)
def test_migration_interruption_at_each_write_step_finishes_forward(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interrupted_name: str
) -> None:
    config, catalog = _paths(tmp_path)
    config.write_bytes(_LEGACY)
    plan = plan_migration(config, catalog)
    from chartreux.core.model_catalog import migration

    original_write = migration._atomic_write

    def interrupt(path: Path, data: bytes) -> None:
        original_write(path, data)
        if path.name == interrupted_name:
            raise KeyboardInterrupt

    monkeypatch.setattr(migration, "_atomic_write", interrupt)
    with pytest.raises(KeyboardInterrupt):
        apply_migration(plan)
    monkeypatch.setattr(migration, "_atomic_write", original_write)

    apply_migration(plan)
    assert not (tmp_path / ".models-migration-recovery.toml").exists()
    assert tomllib.loads(config.read_text())["active_model"] == "base"


@pytest.mark.asyncio
async def test_legacy_catalog_tables_produce_actionable_load_error(
    tmp_path: Path,
) -> None:
    config, _ = _paths(tmp_path)
    config.write_bytes(_LEGACY)
    builder = ConfigBuilder(ChartreuxConfigSchema)
    builder.add_layer(UserConfigLayer(path=config))

    with pytest.raises(ValidationError, match=r"chartreux models migrate"):
        await builder.build()


@pytest.mark.asyncio
async def test_ordinary_patch_rejects_catalog_tables_with_migration_message(
    tmp_path: Path,
) -> None:
    from chartreux.core.config.layers.default import DefaultConfigLayer
    from chartreux.core.config.layers.overrides import OverridesLayer

    session = OverridesLayer(data={})
    orchestrator = await ConfigOrchestrator.create(
        schema=ChartreuxConfigSchema,
        layers=[DefaultConfigLayer(schema=ChartreuxConfigSchema), session],
        default_layer_resolver=lambda: session,
    )
    with pytest.raises(ConfigPatchValidationError, match=r"chartreux models migrate"):
        await orchestrator.set_field("/models", {})


def test_api_key_persistence_writes_provider_to_models_toml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CHARTREUX_HOME", str(tmp_path))
    provider = ProviderConfig(
        name="test", api_base="https://example.test", api_key_env_var="TEST_KEY"
    )

    assert __import__("asyncio").run(apply_provider_to_config(None, provider))

    data = tomllib.loads((tmp_path / "models.toml").read_text())
    assert data["providers"]["test/default"]["api_base"] == "https://example.test"
    assert not (tmp_path / "config.toml").exists()


def test_entrypoint_invokes_migration_before_normal_bootstrap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config, _ = _paths(tmp_path)
    config.write_bytes(_LEGACY)
    monkeypatch.setenv("CHARTREUX_HOME", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["chartreux", "models", "migrate"])

    main()

    assert "Preview: would write" in capsys.readouterr().out


def test_m1_reconciled_deployment_keeps_explicit_legacy_overrides(
    tmp_path: Path,
) -> None:
    config, catalog = _paths(tmp_path)
    config.write_text(
        '[[providers]]\nname = "mistral"\napi_base = "https://api.mistral.ai/v1"\n'
        '[[models]]\nname = "glm-5-2"\nprovider = "mistral"\nalias = "legacy"\n'
        "supports_images = true\nauto_compact_threshold = 123\n"
        'supported_thinking_levels = ["low"]\ninput_price = 0\n'
    )

    apply_migration(plan_migration(config, catalog))

    deployment = load_catalog(catalog).catalog.models["glm-5-2"].deployments[0]
    assert deployment.supports_images and deployment.auto_compact_threshold == 123
    assert deployment.supported_thinking_levels == ("low",)
    assert deployment.prices.input == 0


def test_m3_reconciles_each_deployment_and_materializes_new_provider_slot(
    tmp_path: Path,
) -> None:
    config, catalog = _paths(tmp_path)
    config.write_text(
        '[[providers]]\nname = "mistral"\napi_base = "https://api.mistral.ai/v1"\n'
        '[[providers]]\nname = "second"\napi_base = "https://second.test/v1"\n'
        '[[models]]\nname = "glm-5-2"\nprovider = "mistral"\nalias = "glm-5-2"\n'
        '[[models]]\nname = "glm-5-2"\nprovider = "second"\nalias = "glm-5-2"\n'
    )

    apply_migration(plan_migration(config, catalog))

    deployments = tomllib.loads(catalog.read_text())["models"]["glm-5-2"]["deployments"]
    assert {entry["provider"] for entry in deployments} == {
        "mistral/default",
        "second/default",
    }
    second = next(
        entry for entry in deployments if entry["provider"] == "second/default"
    )
    assert second["name"] == "glm-5-2" and second["supports_images"] is False


def test_m4_former_alias_wire_name_migrates_as_new_base(tmp_path: Path) -> None:
    config, catalog = _paths(tmp_path)
    config.write_text(
        '[[providers]]\nname = "mistral"\napi_base = "https://api.mistral.ai/v1"\n'
        '[[models]]\nname = "zai-glm-latest"\nprovider = "mistral"\nalias = "legacy-zai"\n'
    )

    apply_migration(plan_migration(config, catalog))

    assert load_catalog(catalog).catalog.models["zai-glm-latest"]


def test_m5_sparse_main_model_override_merges_onto_shipped_base(tmp_path: Path) -> None:
    _, catalog = _paths(tmp_path)
    catalog.write_text("[models.glm-5-3]\ntemperature = 0.7\n")

    definition = load_catalog(catalog).catalog.models["glm-5-3"]
    assert definition.temperature == 0.7
    assert definition.deployments[0].name == "zai-glm-5-3"


def test_m6_recovery_checks_config_digest_and_finishes_cleaned_empty_config(
    tmp_path: Path,
) -> None:
    config, catalog = _paths(tmp_path)
    config.write_text(
        '[[providers]]\nname = "test"\napi_base = "https://test.invalid"\n'
        '[[models]]\nname = "base"\nprovider = "test"\nalias = "base"\n'
    )
    plan = plan_migration(config, catalog)
    from chartreux.core.model_catalog import migration

    catalog.write_bytes(migration.tomli_w.dumps(plan.catalog).encode())
    marker = tmp_path / ".models-migration-recovery.toml"
    marker.write_bytes(migration._recovery_marker_data(plan))
    config.write_text("user_edit = true\n")
    with pytest.raises(MigrationConflictError, match="changed config.toml"):
        apply_migration(plan)

    config.write_bytes(plan.cleaned_config)
    assert apply_migration(plan).startswith("Recovered interrupted migration")
    assert config.read_bytes() == b"" and not marker.exists()


def test_migration_converts_legacy_tags_to_roles(tmp_path: Path) -> None:
    config, catalog = _paths(tmp_path)
    config.write_bytes(_LEGACY + b'\n[tags]\npreferred = ["friendly"]\n')

    apply_migration(plan_migration(config, catalog))

    assert tomllib.loads(catalog.read_text())["roles"] == {
        "preferred": {"description": "", "models": ["base"]}
    }


def test_m2_legacy_aliases_are_not_written_to_catalog(tmp_path: Path) -> None:
    config, catalog = _paths(tmp_path)
    config.write_text(
        '[[providers]]\nname = "mistral"\napi_base = "https://api.mistral.ai/v1"\n'
        '[[models]]\nname = "zai-glm-5-3"\nprovider = "mistral"\nalias = "my-glm"\n'
    )

    apply_migration(plan_migration(config, catalog))

    assert "aliases" not in load_catalog(catalog).catalog.models["glm-5-3"].model_dump()
