"""Dangerous-command grant authority: grants-specific coverage only.

The shared authority pairs (unverified-contribution provenance, name/subclass/
default-layer authority, canonical projection immutability, definition
sanitization, colliding project keys, builder acceptance) and the
source-integration pairs migrated to ``_grant_config_harness.py`` in stage 12a;
this module keeps the grants-specific concerns: non-user layer contributions,
programmatic patch routes, symlink reinterpretation, and the settings-catalog
exclusion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from chartreux.core.config import ChartreuxConfigSchema
from chartreux.core.config._dangerous_command_authority import GRANTS_FIELD
from chartreux.core.config._restrictions import SourceRestrictions
from chartreux.core.config.layers.agent_profile import AgentProfileLayer
from chartreux.core.config.layers.default import DefaultConfigLayer
from chartreux.core.config.layers.launch_overrides import LaunchOverridesLayer
from chartreux.core.config.layers.overrides import OverridesLayer
from chartreux.core.config.layers.project import ProjectConfigLayer
from chartreux.core.config.layers.user import UserConfigLayer
from chartreux.core.config.orchestrator import (
    ConfigOrchestrator,
    ConfigPatchValidationError,
)
from chartreux.core.config.patch import AddOperationPatch
from chartreux.core.config.settings_catalog import EDITABLE_SETTINGS, EXCLUDED_SETTINGS
from chartreux.core.trusted_folders import trusted_folders_manager

PUSH = "git.push"


def user_source(
    orchestrator: ConfigOrchestrator[ChartreuxConfigSchema],
) -> SourceRestrictions:
    return next(r for r in orchestrator.restrictions if r.layer_name == "renamed-user")


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["project", "agent-profile", "session", "launch"])
async def test_non_user_layer_contributions_rejected(tmp_path: Path, kind: str) -> None:
    grants = {GRANTS_FIELD: {str(tmp_path): [PUSH]}}
    layers: list[Any] = [DefaultConfigLayer(schema=ChartreuxConfigSchema)]
    if kind == "project":
        project_dir = tmp_path / "project"
        config_dir = project_dir / ".chartreux"
        config_dir.mkdir(parents=True)
        (config_dir / "config.toml").write_text(
            f'[{GRANTS_FIELD}]\n"{tmp_path}" = ["{PUSH}"]\n'
        )
        trusted_folders_manager.trust_for_session(config_dir)
        layers.append(ProjectConfigLayer(path=project_dir))
    elif kind == "agent-profile":
        layers.append(AgentProfileLayer(data=grants))
    elif kind == "session":
        layers.append(OverridesLayer(name="session", data=grants))
    else:
        layers.append(LaunchOverridesLayer(data=grants))
    with pytest.raises(ValueError, match="actual user source"):
        await ConfigOrchestrator.create(
            schema=ChartreuxConfigSchema,
            layers=layers,
            default_layer_resolver=lambda: layers[-1],
        )


async def _noop_preflight(config: ChartreuxConfigSchema) -> None:
    del config


def _noop_apply(config: ChartreuxConfigSchema) -> None:
    del config


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["apply_patch", "apply_session_patch"])
async def test_programmatic_patch_routes_cannot_touch_grants(
    tmp_path: Path, method: str
) -> None:
    path = tmp_path / "user.toml"
    path.write_text(f'[{GRANTS_FIELD}]\n"{tmp_path}" = ["{PUSH}"]\n')
    user = UserConfigLayer(path=path, name="renamed-user")
    session = OverridesLayer(data={})
    orchestrator = await ConfigOrchestrator.create(
        schema=ChartreuxConfigSchema,
        layers=[DefaultConfigLayer(schema=ChartreuxConfigSchema), user, session],
        default_layer_resolver=lambda: session,
    )
    operation = AddOperationPatch(
        path=f"/{GRANTS_FIELD}", value={str(tmp_path): [PUSH]}
    )
    before = path.read_bytes()
    accepted = user_source(orchestrator).dangerous_command_grants
    if method == "apply_patch":
        with pytest.raises(ConfigPatchValidationError, match=GRANTS_FIELD):
            await orchestrator.apply_patch([operation], reason="programmatic")
    else:
        with pytest.raises(ConfigPatchValidationError, match=GRANTS_FIELD):
            await orchestrator.apply_session_patch(
                [operation],
                reason="programmatic",
                preflight=_noop_preflight,
                apply=_noop_apply,
            )
    assert path.read_bytes() == before
    assert user_source(orchestrator).dangerous_command_grants == accepted


@pytest.mark.asyncio
async def test_symlink_reinterpretation_rejected_until_explicit_reload(
    tmp_path: Path,
) -> None:
    alias = tmp_path / "alias"
    target = tmp_path / "target"
    target.mkdir()
    alias.symlink_to(target, target_is_directory=True)
    path = tmp_path / "user.toml"
    path.write_text(f'[{GRANTS_FIELD}]\n"{alias}" = ["{PUSH}"]\n')
    user = UserConfigLayer(path=path, name="renamed-user")
    orchestrator = await ConfigOrchestrator.create(
        schema=ChartreuxConfigSchema,
        layers=[DefaultConfigLayer(schema=ChartreuxConfigSchema), user],
        default_layer_resolver=lambda: user,
    )
    accepted = user_source(orchestrator).dangerous_command_grants
    assert accepted[0].project == target
    other = tmp_path / "other"
    other.mkdir()
    alias.unlink()
    alias.symlink_to(other, target_is_directory=True)
    before = orchestrator.restrictions
    with pytest.raises(ValueError, match="grant interpretation changed"):
        orchestrator.rebuild()
    with pytest.raises(ValueError, match="grant interpretation changed"):
        await orchestrator.preview_candidate()
    assert orchestrator.restrictions is before
    # Explicit reload is the one sanctioned way to adopt the new interpretation.
    await orchestrator.reload()
    assert user_source(orchestrator).dangerous_command_grants[0].project == other


def test_grants_field_is_excluded_from_the_settings_catalog() -> None:
    assert GRANTS_FIELD in EXCLUDED_SETTINGS
    assert GRANTS_FIELD not in {item.path for item in EDITABLE_SETTINGS}
