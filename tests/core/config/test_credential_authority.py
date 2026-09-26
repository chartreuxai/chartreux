from __future__ import annotations

from pathlib import Path

import pytest

from chartreux.core.config import ChartreuxConfigSchema
from chartreux.core.config._credential_authority import (
    CREDENTIAL_ENV_FIELD,
    validate_credential_env_source,
)
from chartreux.core.config.layer import LayerImplementationError, RawConfig
from chartreux.core.config.layers.default import DefaultConfigLayer
from chartreux.core.config.layers.environment import EnvironmentLayer
from chartreux.core.config.layers.overrides import OverridesLayer
from chartreux.core.config.layers.project import ProjectConfigLayer
from chartreux.core.config.layers.user import UserConfigLayer
from chartreux.core.config.orchestrator import (
    ConfigOrchestrator,
    ConfigPatchValidationError,
)
from chartreux.core.config.patch import AddOperationPatch
from chartreux.core.trusted_folders import trusted_folders_manager


async def _orchestrator(
    tmp_path: Path,
) -> tuple[ConfigOrchestrator[ChartreuxConfigSchema], Path]:
    project_dir = tmp_path / "project"
    config_dir = project_dir / ".chartreux"
    config_dir.mkdir(parents=True, exist_ok=True)
    trusted_folders_manager.add_trusted(config_dir)
    project = ProjectConfigLayer(path=project_dir)
    user_path = tmp_path / "user.toml"
    user_path.write_text('credential_env_passthrough = ["USER_ONLY"]\n')
    user = UserConfigLayer(path=user_path)
    orchestrator = await ConfigOrchestrator.create(
        schema=ChartreuxConfigSchema,
        layers=[DefaultConfigLayer(schema=ChartreuxConfigSchema), project, user],
        default_layer_resolver=lambda: project,
    )
    return orchestrator, config_dir / "config.toml"


@pytest.mark.asyncio
async def test_project_disk_definition_rejected_even_when_user_shadows(
    tmp_path: Path,
) -> None:
    orchestrator, project_path = await _orchestrator(tmp_path)
    assert orchestrator.config.credential_env_passthrough == ["USER_ONLY"]
    original, token = orchestrator.config, orchestrator.accepted_token
    project_path.write_text('credential_env_passthrough = ["PROJECT_SECRET"]\n')
    with pytest.raises(ValueError, match="actual user source") as caught:
        await orchestrator.reload()
    assert "PROJECT_SECRET" not in str(caught.value)
    assert caught.value.__context__ is None
    assert orchestrator.config is original
    assert orchestrator.accepted_token is token
    with pytest.raises(ValueError, match="actual user source"):
        await _orchestrator(tmp_path)  # disk load rejects the same project file


@pytest.mark.asyncio
async def test_project_staged_candidate_rejected_without_publication(
    tmp_path: Path,
) -> None:
    orchestrator, _ = await _orchestrator(tmp_path)
    original, token = orchestrator.config, orchestrator.accepted_token
    with pytest.raises(ValueError, match="actual user source"):
        await orchestrator.preview_candidate(
            layer_overrides={
                "project-toml": RawConfig.model_validate({
                    CREDENTIAL_ENV_FIELD: ["PROJECT_SECRET"]
                })
            }
        )
    assert orchestrator.config is original
    assert orchestrator.accepted_token is token
    assert orchestrator.config.credential_env_passthrough == ["USER_ONLY"]


@pytest.mark.asyncio
async def test_user_disk_reload_and_empty_default(tmp_path: Path) -> None:
    orchestrator, _ = await _orchestrator(tmp_path)
    user = orchestrator.get_layer("user-toml")
    assert orchestrator.config.credential_env_passthrough == ["USER_ONLY"]
    assert isinstance(user, UserConfigLayer)
    (tmp_path / "user.toml").write_text('credential_env_passthrough = ["UPDATED"]\n')
    await orchestrator.reload()
    assert orchestrator.config.credential_env_passthrough == ["UPDATED"]
    empty = DefaultConfigLayer(schema=ChartreuxConfigSchema)
    assert (await empty.load()).model_dump()[CREDENTIAL_ENV_FIELD] == []
    validate_credential_env_source({CREDENTIAL_ENV_FIELD: []}, layer=empty)


@pytest.mark.parametrize("name", ["user-toml", "user", "default"])
def test_forged_source_and_empty_nonuser_rejected(name: str) -> None:
    with pytest.raises(ValueError, match="actual user source"):
        validate_credential_env_source(
            {CREDENTIAL_ENV_FIELD: []}, layer=OverridesLayer(name=name, data={})
        )


def test_user_subclass_cannot_claim_authority(tmp_path: Path) -> None:
    class FakeUser(UserConfigLayer):
        pass

    with pytest.raises(ValueError, match="actual user source"):
        validate_credential_env_source(
            {CREDENTIAL_ENV_FIELD: ["SECRET"]},
            layer=FakeUser(path=tmp_path / "fake.toml"),
        )


@pytest.mark.asyncio
async def test_generic_user_patch_and_save_rejected(tmp_path: Path) -> None:
    orchestrator, _ = await _orchestrator(tmp_path)
    user = orchestrator.get_layer("user-toml")
    operation = AddOperationPatch(
        path=f"/{CREDENTIAL_ENV_FIELD}", value=["SECRET"], target_layer_name=user.name
    )
    with pytest.raises(ConfigPatchValidationError, match=CREDENTIAL_ENV_FIELD):
        await orchestrator.apply_patch([operation], reason="model")
    assert user.fingerprint is not None
    result = await orchestrator.save(
        [operation], target="user", expected_revision=user.fingerprint, reason="client"
    )
    assert result.persistence == "not_saved"
    assert orchestrator.config.credential_env_passthrough == ["USER_ONLY"]


@pytest.mark.asyncio
async def test_environment_definition_rejected_before_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import patch

    monkeypatch.setenv("CHARTREUX_CREDENTIAL_ENV_PASSTHROUGH", "secret-invalid-json")
    layer = EnvironmentLayer(schema=ChartreuxConfigSchema)
    with patch("chartreux.core.config.layers.environment.EnvSettingsSource") as decoder:
        with pytest.raises(LayerImplementationError) as caught:
            await layer.load()
    decoder.assert_not_called()
    assert caught.value.__cause__ is not None
    assert CREDENTIAL_ENV_FIELD in str(caught.value.__cause__)
    assert "secret-invalid-json" not in str(caught.value.__cause__)
