from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any, Literal
from unittest.mock import patch

import pytest

from chartreux.core.config import ChartreuxConfigSchema
from chartreux.core.config._dangerous_command_authority import (
    GRANTS_FIELD,
    validate_grant_source,
)
from chartreux.core.config._restrictions import SourceRestrictions
from chartreux.core.config.builder import ConfigBuilder
from chartreux.core.config.layer import LayerImplementationError, RawConfig
from chartreux.core.config.layers.agent_profile import AgentProfileLayer
from chartreux.core.config.layers.default import DefaultConfigLayer
from chartreux.core.config.layers.environment import EnvironmentLayer
from chartreux.core.config.layers.launch_overrides import LaunchOverridesLayer
from chartreux.core.config.layers.overrides import OverridesLayer
from chartreux.core.config.layers.project import ProjectConfigLayer
from chartreux.core.config.layers.user import UserConfigLayer
from chartreux.core.config.orchestrator import (
    ConfigOrchestrator,
    ConfigPatchValidationError,
)
from chartreux.core.config.patch import (
    AddOperationPatch,
    PatchOp,
    RemoveOperationPatch,
    ReplaceOperationPatch,
)
from chartreux.core.config.settings_catalog import EDITABLE_SETTINGS, EXCLUDED_SETTINGS
from chartreux.core.trusted_folders import trusted_folders_manager

PUSH = "git.push"


async def user_orchestrator(
    tmp_path: Path, *, grants: bool = True
) -> tuple[ConfigOrchestrator[ChartreuxConfigSchema], Path]:
    path = tmp_path / "user.toml"
    path.write_text(f'[{GRANTS_FIELD}]\n"{tmp_path}" = ["{PUSH}"]\n' if grants else "")
    user = UserConfigLayer(path=path, name="renamed-user")
    orchestrator = await ConfigOrchestrator.create(
        schema=ChartreuxConfigSchema,
        layers=[DefaultConfigLayer(schema=ChartreuxConfigSchema), user],
        default_layer_resolver=lambda: user,
    )
    return orchestrator, path


def user_source(
    orchestrator: ConfigOrchestrator[ChartreuxConfigSchema],
) -> SourceRestrictions:
    return next(r for r in orchestrator.restrictions if r.layer_name == "renamed-user")


def test_unverified_grant_contribution_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="grant source provenance unavailable"):
        SourceRestrictions.from_raw(
            {GRANTS_FIELD: {str(tmp_path): [PUSH]}},
            layer_name="user-toml",
            locator="user-toml",
            kind="source",
            store_fingerprint=None,
        )


@pytest.mark.parametrize(
    "name", ["user-toml", "user", "default", "environment", "agent-profile"]
)
@pytest.mark.parametrize("empty", [False, True])
def test_names_never_authorize_grants(tmp_path: Path, name: str, empty: bool) -> None:
    data = {GRANTS_FIELD: {} if empty else {str(tmp_path): [PUSH]}}
    with pytest.raises(ValueError, match="actual user source"):
        validate_grant_source(data, layer=OverridesLayer(name=name, data=data))


def test_only_empty_actual_default_is_tolerated(tmp_path: Path) -> None:
    layer = DefaultConfigLayer(schema=ChartreuxConfigSchema, name="renamed")
    assert validate_grant_source({GRANTS_FIELD: {}}, layer=layer) == ()
    with pytest.raises(ValueError, match="actual user source"):
        validate_grant_source({GRANTS_FIELD: {str(tmp_path): []}}, layer=layer)


def test_subclass_cannot_claim_user_authority(tmp_path: Path) -> None:
    class FakeUserLayer(UserConfigLayer):
        pass

    with pytest.raises(ValueError, match="actual user source"):
        validate_grant_source(
            {GRANTS_FIELD: {}}, layer=FakeUserLayer(path=tmp_path / "user.toml")
        )


def test_projection_canonical_immutable_and_separate_from_merge(tmp_path: Path) -> None:
    project = tmp_path / "project"
    data = {GRANTS_FIELD: {str(project): [PUSH, PUSH]}}
    layer = UserConfigLayer(path=tmp_path / "user.toml", name="renamed")
    source = SourceRestrictions.from_raw(
        data,
        layer_name=layer.name,
        locator=layer.source_locator,
        kind="source",
        store_fingerprint="accepted-source",
        root_layer=layer,
    )
    assert source.dangerous_command_grants == validate_grant_source(data, layer=layer)
    item = source.dangerous_command_grants[0]
    assert item.project == project and item.operations == (PUSH,)
    assert source.project_grants() == {str(project): [PUSH]}
    frozen_field = "operations"
    with pytest.raises(FrozenInstanceError):
        setattr(item, frozen_field, ())
    data[GRANTS_FIELD][str(project)].clear()
    config = ChartreuxConfigSchema.model_validate({
        GRANTS_FIELD: {str(project): [PUSH]}
    })
    # Mutating the effective (mergeable) schema dict creates no accepted authority.
    config.dangerous_command_grants_by_project.clear()
    config.dangerous_command_grants_by_project[str(tmp_path / "other")] = [PUSH]
    assert item.operations == (PUSH,)
    assert not source.preserves(
        SourceRestrictions.from_raw(
            {},
            layer_name=layer.name,
            locator=layer.source_locator,
            kind="source",
            store_fingerprint=None,
        )
    )


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        {"relative-project": [PUSH]},
        {"/project": ["git.push.force"]},
        {"/project": ["git.push.delete"]},
        {"/project": ["git.checkout"]},
        {"/project": ["git.push.unregistered"]},
        {"/project": [42]},
        {"/project": PUSH},
        {"/project": [PUSH, "git.push.force"]},
    ],
)
def test_invalid_definitions_are_sanitized(tmp_path: Path, value: Any) -> None:
    with pytest.raises(ValueError) as caught:
        validate_grant_source(
            {GRANTS_FIELD: value}, layer=UserConfigLayer(path=tmp_path / "user.toml")
        )
    assert str(caught.value) == (
        "Invalid dangerous_command_grants_by_project definition (source: user)"
    )
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_colliding_canonical_project_keys_rejected(tmp_path: Path) -> None:
    alias = tmp_path / "alias"
    project = tmp_path / "project"
    project.mkdir()
    alias.symlink_to(project, target_is_directory=True)
    with pytest.raises(ValueError, match="Invalid dangerous_command_grants"):
        validate_grant_source(
            {GRANTS_FIELD: {str(alias): [], str(project): []}},
            layer=UserConfigLayer(path=tmp_path / "user.toml"),
        )


@pytest.mark.asyncio
async def test_builder_accepts_real_user_grants(tmp_path: Path) -> None:
    path = tmp_path / "user.toml"
    path.write_text(f'[{GRANTS_FIELD}]\n"{tmp_path}" = ["{PUSH}"]\n')
    builder = ConfigBuilder(ChartreuxConfigSchema)
    builder.add_layers([
        DefaultConfigLayer(schema=ChartreuxConfigSchema),
        UserConfigLayer(path=path),
    ])
    candidate = await builder.build_candidate()
    assert candidate.restrictions[-1].dangerous_command_grants[0].project == tmp_path
    assert candidate.restrictions[-1].dangerous_command_grants[0].operations == (PUSH,)
    assert (await builder.build()).dangerous_command_grants_by_project == {
        str(tmp_path): [PUSH]
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["build", "build_candidate"])
@pytest.mark.parametrize("value", [{}, None, [], {"/private-project": ["secret"]}])
@pytest.mark.parametrize("prospective", [False, True])
async def test_forbidden_sources_fail_before_shadow_and_cache(
    method: Literal["build", "build_candidate"], value: Any, prospective: bool
) -> None:
    spoof = OverridesLayer(
        name="user-toml", data={} if prospective else {GRANTS_FIELD: value}
    )
    builder = ConfigBuilder(ChartreuxConfigSchema)
    builder.add_layers([spoof, DefaultConfigLayer(schema=ChartreuxConfigSchema)])
    overrides = {
        spoof.name: RawConfig.model_validate(
            {GRANTS_FIELD: value} if prospective else {}
        )
    }
    with pytest.raises(ValueError, match="actual user source") as caught:
        await getattr(builder, method)(layer_overrides=overrides)
    assert "secret" not in str(caught.value)
    assert "/private-project" not in str(caught.value)
    assert caught.value.__context__ is None
    assert all(layer.cached_data is None for layer in builder.layers)
    assert all(layer.fingerprint is None for layer in builder.layers)


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["build", "build_candidate"])
async def test_malformed_real_user_cannot_be_hidden_by_override(
    tmp_path: Path, method: Literal["build", "build_candidate"]
) -> None:
    path = tmp_path / "private-source.toml"
    path.write_text(f'[{GRANTS_FIELD}]\n"/private-project" = ["git.push.force"]\n')
    user = UserConfigLayer(path=path)
    builder = ConfigBuilder(ChartreuxConfigSchema)
    builder.add_layers([DefaultConfigLayer(schema=ChartreuxConfigSchema), user])
    with pytest.raises(ValueError, match=GRANTS_FIELD) as caught:
        await getattr(builder, method)(layer_overrides={user.name: RawConfig()})
    assert caught.value.__context__ is None
    assert "private" not in str(caught.value)
    assert "git.push.force" not in str(caught.value)
    assert user.cached_data is None
    assert builder.layers[0].cached_data is None


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


@pytest.mark.asyncio
@pytest.mark.parametrize("suffix", ["", "__/private-project"])
@pytest.mark.parametrize("value", ["", "{}", "null", "secret-malformed-json"])
async def test_environment_grants_rejected_before_decode(
    monkeypatch: pytest.MonkeyPatch, suffix: str, value: str
) -> None:
    monkeypatch.setenv(f"CHARTREUX_{GRANTS_FIELD.upper()}{suffix}", value)
    layer = EnvironmentLayer(schema=ChartreuxConfigSchema)
    with patch("chartreux.core.config.layers.environment.EnvSettingsSource") as decoder:
        with pytest.raises(LayerImplementationError) as caught:
            await layer.load()
    decoder.assert_not_called()
    cause = caught.value.__cause__
    assert cause is not None
    assert GRANTS_FIELD in str(cause)
    assert "secret" not in str(cause)
    assert "private" not in str(cause)
    assert cause.__context__ is None
    assert layer.cached_data is None
    assert layer.fingerprint is None


@pytest.mark.asyncio
@pytest.mark.parametrize("save", [False, True])
@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize(
    "operation",
    ["set", "replace", "remove", "document", "erase-document", "set-remove"],
)
async def test_ordinary_user_patch_cannot_change_grants(
    tmp_path: Path, save: bool, existing: bool, operation: str
) -> None:
    orchestrator, path = await user_orchestrator(tmp_path, grants=existing)
    layer = orchestrator.get_layer("renamed-user")
    grants = {str(tmp_path): [PUSH]}
    ops: list[PatchOp]
    match operation:
        case "set":
            ops = [AddOperationPatch(path=f"/{GRANTS_FIELD}", value=grants)]
        case "replace":
            ops = [ReplaceOperationPatch(path=f"/{GRANTS_FIELD}", value=grants)]
        case "remove":
            ops = [RemoveOperationPatch(path=f"/{GRANTS_FIELD}")]
        case "document":
            ops = [AddOperationPatch(path="", value={GRANTS_FIELD: grants})]
        case "erase-document":
            if not existing:
                pytest.skip("No grant authority exists to remove")
            ops = [AddOperationPatch(path="", value={})]
        case _:
            ops = [
                AddOperationPatch(path=f"/{GRANTS_FIELD}", value=grants),
                RemoveOperationPatch(path=f"/{GRANTS_FIELD}"),
            ]
    ops = [op.model_copy(update={"target_layer_name": layer.name}) for op in ops]
    before = path.read_bytes()
    config, policy, token, cache = (
        orchestrator.config,
        orchestrator.restrictions,
        orchestrator.accepted_token,
        layer.cached_data,
    )
    if save:
        assert layer.fingerprint is not None
        result = await orchestrator.save(
            ops, target="user", expected_revision=layer.fingerprint, reason="ordinary"
        )
        assert result.persistence == "not_saved"
        assert result.error == "validation"
    else:
        with pytest.raises(ConfigPatchValidationError, match=GRANTS_FIELD) as caught:
            await orchestrator.apply_patch(ops, reason="ordinary")
        assert PUSH not in str(caught.value)
        assert caught.value.__context__ is None
    assert path.read_bytes() == before
    assert orchestrator.config is config
    assert orchestrator.restrictions is policy
    assert orchestrator.accepted_token is token
    assert layer.cached_data is cache


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


async def _noop_preflight(config: ChartreuxConfigSchema) -> None:
    del config


def _noop_apply(config: ChartreuxConfigSchema) -> None:
    del config


@pytest.mark.asyncio
@pytest.mark.parametrize("preview", [False, True])
async def test_bad_reload_or_preview_preserves_accepted_snapshot(
    tmp_path: Path, preview: bool
) -> None:
    orchestrator, path = await user_orchestrator(tmp_path)
    layer = orchestrator.get_layer("renamed-user")
    config, policy, token, cache = (
        orchestrator.config,
        orchestrator.restrictions,
        orchestrator.accepted_token,
        layer.cached_data,
    )
    with pytest.raises(ValueError, match=GRANTS_FIELD) as caught:
        if preview:
            await orchestrator.preview_candidate(
                layer_overrides={
                    layer.name: RawConfig.model_validate({
                        GRANTS_FIELD: {"/private": ["secret"]}
                    })
                }
            )
        else:
            path.write_text(f'[{GRANTS_FIELD}]\n"/private" = ["secret"]\n')
            await orchestrator.reload()
    assert caught.value.__context__ is None
    assert "private" not in str(caught.value)
    assert "secret" not in str(caught.value)
    assert orchestrator.config is config
    assert orchestrator.restrictions is policy
    assert orchestrator.accepted_token is token
    assert layer.cached_data is cache


@pytest.mark.parametrize("value", [{}, None, []])
def test_empty_unattributed_assertions_fail_closed(value: Any) -> None:
    with pytest.raises(ValueError, match="provenance"):
        SourceRestrictions.from_raw(
            {GRANTS_FIELD: value},
            layer_name="user-toml",
            locator="user-toml",
            kind="source",
            store_fingerprint=None,
        )


@pytest.mark.asyncio
async def test_real_user_grants_are_accepted_and_immutable(tmp_path: Path) -> None:
    orchestrator, _ = await user_orchestrator(tmp_path)
    grants = orchestrator.restrictions[-1].dangerous_command_grants
    assert grants[0].project == tmp_path
    assert grants[0].operations == (PUSH,)
    # Mutating the effective schema dict creates no accepted authority.
    orchestrator.config.dangerous_command_grants_by_project.clear()
    orchestrator.config.dangerous_command_grants_by_project[str(tmp_path / "other")] = [
        PUSH
    ]
    assert orchestrator.restrictions[-1].dangerous_command_grants == grants
    assert await orchestrator.set_field("/theme", "dark", reason="ordinary") == []
    assert orchestrator.restrictions[-1].dangerous_command_grants == grants
    staged = await orchestrator._stage_policy_replacement(
        source="renamed-user", tools={}, expected_token=orchestrator.accepted_token
    )
    assert staged.restrictions[-1].dangerous_command_grants == grants
    child = orchestrator._copy_for_child()
    refreshed = await child._stage_policy_refresh(
        previous=orchestrator.restrictions[-1],
        replacement=staged.restrictions[-1],
        expected_token=child.accepted_token,
    )
    assert refreshed.restrictions[-1].dangerous_command_grants == grants


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
