from __future__ import annotations

from pathlib import Path
import tomllib

import pytest
import tomli_w

from chartreux.core.config import build_default_orchestrator
from chartreux.core.config.harness_files import (
    init_harness_files_manager,
    reset_harness_files_manager,
)
from chartreux.core.trusted_folders import trusted_folders_manager
from tests.conftest import get_base_config


def _write_user_config(config_dir: Path, extra: dict[str, object]) -> None:
    data = get_base_config()
    data.update(extra)
    with (config_dir / "config.toml").open("wb") as f:
        tomli_w.dump(data, f)


def _write_project_config(
    tmp_working_directory: Path, extra: dict[str, object]
) -> Path:
    project_vibe_dir = tmp_working_directory / ".chartreux"
    project_vibe_dir.mkdir(parents=True, exist_ok=True)
    project_config = project_vibe_dir / "config.toml"
    with project_config.open("wb") as f:
        tomli_w.dump(extra, f)
    return project_config


def _project_vibe_dir(tmp_working_directory: Path) -> Path:
    return tmp_working_directory / ".chartreux"


@pytest.mark.asyncio
async def test_loading_preserves_unmigrated_policy_fingerprints(
    config_dir: Path,
) -> None:
    path = config_dir / "config.toml"
    path.write_text('[tools.read]\npermission = "never"\n')

    orchestrator = await build_default_orchestrator()

    assert path.read_text() == '[tools.read]\npermission = "never"\n'
    layer = orchestrator.get_layer("user-toml")
    source = next(r for r in orchestrator.restrictions if r.layer_name == layer.name)
    assert source.store_fingerprint is not None
    assert source.store_fingerprint == layer.fingerprint
    assert "read" in (await layer.load()).model_dump()["tools"]
    config_before = orchestrator.config.model_dump()
    restrictions_before = orchestrator.restrictions
    await orchestrator.reload()
    assert orchestrator.config.model_dump() == config_before
    assert orchestrator.restrictions == restrictions_before


class TestBothTomlLayersInstalled:
    @pytest.mark.asyncio
    async def test_project_overrides_user_scalar(
        self, config_dir: Path, tmp_working_directory: Path
    ) -> None:
        _write_user_config(config_dir, {"theme": "user-theme"})
        _write_project_config(tmp_working_directory, {"theme": "project-theme"})
        trusted_folders_manager.add_trusted(_project_vibe_dir(tmp_working_directory))

        orch = await build_default_orchestrator()

        assert orch.config.theme == "project-theme"
        # Project still wins the merge, but ordinary writes stay in the session.
        assert orch.writable_layer_name == "overrides"

    @pytest.mark.asyncio
    async def test_discovered_project_config_does_not_capture_writes(
        self, config_dir: Path, tmp_working_directory: Path
    ) -> None:
        _write_user_config(config_dir, {"theme": "user-theme"})
        project_config = _write_project_config(
            tmp_working_directory, {"theme": "project-theme"}
        )
        trusted_folders_manager.add_trusted(_project_vibe_dir(tmp_working_directory))

        orch = await build_default_orchestrator()
        user_before = (config_dir / "config.toml").read_bytes()
        project_before = project_config.read_bytes()
        assert await orch.set_field("/enable_notifications", False) == []
        assert orch.config.enable_notifications is False
        assert project_config.read_bytes() == project_before
        assert (config_dir / "config.toml").read_bytes() == user_before

    @pytest.mark.asyncio
    async def test_user_only_field_survives_when_project_absent(
        self, config_dir: Path, tmp_working_directory: Path
    ) -> None:
        _write_user_config(config_dir, {"theme": "user-theme"})
        assert not _project_vibe_dir(tmp_working_directory).exists()

        orch = await build_default_orchestrator()

        assert orch.config.theme == "user-theme"
        assert orch.writable_layer_name == "overrides"

    @pytest.mark.asyncio
    async def test_concat_merge_concatenates_disabled_tools(
        self, config_dir: Path, tmp_working_directory: Path
    ) -> None:
        _write_user_config(config_dir, {"disabled_tools": ["user-tool"]})
        _write_project_config(
            tmp_working_directory, {"disabled_tools": ["project-tool"]}
        )
        trusted_folders_manager.add_trusted(_project_vibe_dir(tmp_working_directory))

        orch = await build_default_orchestrator()

        assert orch.config.disabled_tools == ["user-tool", "project-tool"]

    @pytest.mark.asyncio
    async def test_untrusted_project_falls_back_to_user_only(
        self, config_dir: Path, tmp_working_directory: Path
    ) -> None:
        _write_user_config(config_dir, {"theme": "user-theme"})
        _write_project_config(tmp_working_directory, {"theme": "project-theme"})
        # Deliberately do NOT trust the project .chartreux dir. The untrusted project
        # layer loads empty (so effective config equals the user config) and the
        # write target remains the session layer.
        orch = await build_default_orchestrator()

        assert orch.config.theme == "user-theme"
        assert orch.writable_layer_name == "overrides"

    @pytest.mark.asyncio
    async def test_no_project_file_write_target_is_session(
        self, config_dir: Path, tmp_working_directory: Path
    ) -> None:
        _write_user_config(config_dir, {"theme": "user-theme"})
        assert not _project_vibe_dir(tmp_working_directory).exists()

        orch = await build_default_orchestrator()

        assert orch.config.theme == "user-theme"
        assert orch.writable_layer_name == "overrides"

    @pytest.mark.asyncio
    async def test_project_only_no_file_creates_project_config_on_write(
        self, config_dir: Path, tmp_working_directory: Path
    ) -> None:
        reset_harness_files_manager()
        init_harness_files_manager("project")
        try:
            project_config = _project_vibe_dir(tmp_working_directory) / "config.toml"
            assert not project_config.exists()

            orch = await build_default_orchestrator()

            assert orch.writable_layer_name == "overrides"
            failures = await orch.set_field("/theme", "session-theme")
            assert failures == []
            assert orch.config.theme == "session-theme"
            assert not project_config.exists()
            from chartreux.core.config.patch import AddOperationPatch

            revision = orch.get_layer("project-toml").fingerprint
            assert revision is not None
            result = await orch.save(
                [AddOperationPatch(path="/theme", value="persisted-theme")],
                target="project",
                expected_revision=revision,
                reason="explicit save",
            )
            assert (result.persistence, result.application) == ("saved", "applied")
            assert orch.config.theme == "session-theme"
            with project_config.open("rb") as f:
                assert tomllib.load(f)["theme"] == "persisted-theme"
        finally:
            reset_harness_files_manager()
            init_harness_files_manager("user", "project")

    @pytest.mark.asyncio
    async def test_no_persistent_sources_writes_to_runtime_overrides(
        self, config_dir: Path
    ) -> None:
        reset_harness_files_manager()
        init_harness_files_manager()
        try:
            orch = await build_default_orchestrator()

            assert orch.writable_layer_name == "overrides"
            failures = await orch.set_field("/theme", "runtime-theme")

            assert failures == []
            assert orch.config.theme == "runtime-theme"
            with (config_dir / "config.toml").open("rb") as f:
                assert tomllib.load(f).get("theme") != "runtime-theme"
        finally:
            reset_harness_files_manager()
            init_harness_files_manager("user", "project")

    @pytest.mark.asyncio
    async def test_launched_from_home_does_not_double_user_config(
        self, config_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_user_config(config_dir, {"disabled_tools": ["user-tool"]})
        # cwd == user home, so cwd/.chartreux is the user config dir itself.
        monkeypatch.chdir(config_dir.parent)

        orch = await build_default_orchestrator()

        assert orch.config.disabled_tools == ["user-tool"]
        layer_names = {layer.name for layer in orch.layers}
        assert "project-toml" not in layer_names
        assert orch.writable_layer_name == "overrides"

    @pytest.mark.asyncio
    async def test_both_layers_present_in_stack(
        self, config_dir: Path, tmp_working_directory: Path
    ) -> None:
        _write_user_config(config_dir, {"theme": "user-theme"})
        _write_project_config(tmp_working_directory, {"theme": "project-theme"})
        trusted_folders_manager.add_trusted(_project_vibe_dir(tmp_working_directory))

        orch = await build_default_orchestrator()

        layer_names = {layer.name for layer in orch.layers}
        assert {"user-toml", "project-toml"}.issubset(layer_names)
        # Order: user before project.
        ordered = [layer.name for layer in orch.layers]
        assert ordered.index("user-toml") < ordered.index("project-toml")
