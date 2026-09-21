from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from chartreux.core.config import ChartreuxConfigSchema
from chartreux.core.config._restrictions import SourceRestrictions
from chartreux.core.config._root_authority import ROOTS_FIELD
from chartreux.core.config.layer import ConfigLayer, RawConfig
from chartreux.core.config.layers.agent_profile import AgentProfileLayer
from chartreux.core.config.layers.default import DefaultConfigLayer
from chartreux.core.config.layers.overrides import OverridesLayer
from chartreux.core.config.layers.user import UserConfigLayer
from chartreux.core.config.orchestrator import (
    ConfigOrchestrator,
    ConfigPatchValidationError,
)
from chartreux.core.config.patch import AddOperationPatch


async def make_orchestrator(
    tmp_path: Path, *, persisted: bool = True
) -> tuple[ConfigOrchestrator[ChartreuxConfigSchema], Path]:
    path = tmp_path / "user.toml"
    if persisted:
        path.write_text(
            f'[{ROOTS_FIELD}]\n"{tmp_path}" = ["{tmp_path / "old"}"]\n'
            '[tools.bash]\ndenylist = ["user-denial"]\n'
        )
    user = UserConfigLayer(path=path, name="renamed-user")
    session = OverridesLayer(name="session", data={})
    independent = OverridesLayer(
        name="independent", data={"tools": {"bash": {"denylist": ["other-denial"]}}}
    )
    orchestrator = await ConfigOrchestrator.create(
        schema=ChartreuxConfigSchema,
        layers=[
            DefaultConfigLayer(schema=ChartreuxConfigSchema),
            user,
            session,
            independent,
        ],
        default_layer_resolver=lambda: session,
    )
    return orchestrator, path


def user_source(
    orchestrator: ConfigOrchestrator[ChartreuxConfigSchema],
) -> SourceRestrictions:
    return next(r for r in orchestrator.restrictions if r.layer_name == "renamed-user")


async def install(
    orchestrator: ConfigOrchestrator[ChartreuxConfigSchema], roots: dict[str, list[str]]
) -> None:
    token = orchestrator.accepted_token
    staged = await orchestrator._prepare_root_replacement(
        source="renamed-user", roots=roots, expected_token=token
    )
    orchestrator._commit_policy_replacement(staged, expected_token=token)


@pytest.mark.asyncio
@pytest.mark.parametrize("persisted", [False, True])
async def test_overlay_survives_all_rebuilds_without_writes(
    tmp_path: Path, persisted: bool
) -> None:
    orchestrator, path = await make_orchestrator(tmp_path, persisted=persisted)
    disk = path.read_bytes() if persisted else None
    original = user_source(orchestrator)
    independent = orchestrator.restrictions[-1]
    roots = {str(tmp_path): [str(tmp_path / "new"), str(tmp_path / "new")]}
    token = orchestrator.accepted_token
    staged = await orchestrator._prepare_root_replacement(
        source="renamed-user", roots=roots, expected_token=token
    )
    assert user_source(orchestrator) is original
    canonical = {str(tmp_path): [str(tmp_path / "new")]}
    assert user_source(staged).project_roots() == canonical
    orchestrator._commit_policy_replacement(staged, expected_token=token)
    roots.clear()
    assert user_source(orchestrator).identity == original.identity
    assert user_source(orchestrator).tools == original.tools
    assert orchestrator.restrictions[-1] == independent

    for expected in (canonical, {}):
        if not expected:
            await install(orchestrator, {})
        assert (
            await orchestrator.set_field("/active_model", "", reason="ordinary") == []
        )
        # Previewing an ordinary user-targeted edit exercises the shadowed backing
        # roots without writing the file (session writes above are in memory).
        await orchestrator._preview_patch([
            AddOperationPatch(
                path="/active_model", value="", target_layer_name="renamed-user"
            )
        ])
        orchestrator.replace_or_append_layer(
            "agent-profile", AgentProfileLayer(data={"active_model": ""})
        )
        orchestrator.rebuild()
        assert orchestrator.config.authorized_roots_by_project == expected
        assert user_source(orchestrator).project_roots() == expected
        copied = orchestrator.copy()
        copied.rebuild()
        assert user_source(copied).project_roots() == expected
        assert (await copied._builder.build()).authorized_roots_by_project == expected
        assert (
            await copied.preview_candidate(force_load=True)
        ).config.authorized_roots_by_project == expected
        token = orchestrator.accepted_token
        tool_stage = await orchestrator._stage_policy_replacement(
            source="renamed-user", tools={}, expected_token=token
        )
        orchestrator._commit_policy_replacement(tool_stage, expected_token=token)
        orchestrator.rebuild()
        assert user_source(orchestrator).project_roots() == expected
        assert user_source(orchestrator).tools == ()
        assert (
            next(r for r in orchestrator.restrictions if r.layer_name == "independent")
            == independent
        )
        assert (path.read_bytes() if path.exists() else None) == disk


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "roots", [None, [], {"relative": []}, {"/project": [42]}, {"/project": "secret"}]
)
async def test_malformed_input_is_atomic(tmp_path: Path, roots: Any) -> None:
    orchestrator, path = await make_orchestrator(tmp_path)
    before = orchestrator.config, orchestrator.restrictions, orchestrator.accepted_token
    cache = orchestrator.get_layer("renamed-user").cached_data
    disk = path.read_bytes()
    with pytest.raises(ValueError, match="Invalid authorized") as caught:
        await orchestrator._prepare_root_replacement(
            source="renamed-user", roots=roots, expected_token=before[2]
        )
    assert caught.value.__context__ is None
    assert "secret" not in str(caught.value)
    assert orchestrator.config is before[0]
    assert orchestrator.restrictions is before[1]
    assert orchestrator.accepted_token is before[2]
    assert orchestrator.get_layer("renamed-user").cached_data is cache
    assert path.read_bytes() == disk


@pytest.mark.asyncio
async def test_stale_prepare_commit_and_inherited_owner_rejected(
    tmp_path: Path,
) -> None:
    orchestrator, path = await make_orchestrator(tmp_path)
    disk = path.read_bytes()
    old = orchestrator.accepted_token
    staged = await orchestrator._prepare_root_replacement(
        source="renamed-user", roots={}, expected_token=old
    )
    await orchestrator.set_field("/active_model", "", reason="ordinary")
    before = orchestrator.restrictions
    with pytest.raises(ValueError, match="Stale"):
        orchestrator._commit_policy_replacement(staged, expected_token=old)
    with pytest.raises(ValueError, match="Stale"):
        await orchestrator._prepare_root_replacement(
            source="renamed-user", roots={}, expected_token=old
        )
    child = orchestrator._copy_for_child()
    with pytest.raises(ValueError, match="same owned source"):
        await child._prepare_root_replacement(
            source="renamed-user", roots={}, expected_token=child.accepted_token
        )
    assert orchestrator.restrictions is before
    assert path.read_bytes() == disk


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["absent", "spoof", "subclass"])
async def test_source_names_are_not_privileges(tmp_path: Path, kind: str) -> None:
    class FakeUserLayer(UserConfigLayer):
        pass

    layer: ConfigLayer[RawConfig]
    if kind == "subclass":
        layer = FakeUserLayer(path=tmp_path / "user.toml")
    else:
        layer = OverridesLayer(
            name="user-toml" if kind == "spoof" else "session", data={}
        )
    orchestrator = await ConfigOrchestrator.create(
        schema=ChartreuxConfigSchema,
        layers=[DefaultConfigLayer(schema=ChartreuxConfigSchema), layer],
        default_layer_resolver=lambda: layer,
    )
    before = orchestrator.restrictions
    with pytest.raises(ValueError, match="actual user source"):
        await orchestrator._prepare_root_replacement(
            source="user-toml", roots={}, expected_token=orchestrator.accepted_token
        )
    assert orchestrator.restrictions is before
    assert not (tmp_path / "user.toml").exists()


@pytest.mark.asyncio
async def test_ordinary_root_edits_and_symlink_reinterpretation_still_rejected(
    tmp_path: Path,
) -> None:
    orchestrator, _ = await make_orchestrator(tmp_path)
    root = tmp_path / "canonical"
    await install(orchestrator, {str(tmp_path): [str(root)]})
    before = orchestrator.restrictions
    with pytest.raises(ConfigPatchValidationError, match=ROOTS_FIELD):
        await orchestrator.set_field(f"/{ROOTS_FIELD}", {}, reason="ordinary")
    target = tmp_path / "other"
    target.mkdir()
    root.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="interpretation changed"):
        orchestrator.rebuild()
    with pytest.raises(ValueError, match="interpretation changed"):
        await orchestrator._stage_policy_replacement(
            source="renamed-user", tools={}, expected_token=orchestrator.accepted_token
        )
    with pytest.raises(ValueError, match="interpretation changed"):
        await orchestrator.reload()
    assert orchestrator.restrictions is before
    # The dedicated explicit operation alone may accept the newly resolved path.
    await install(orchestrator, {str(tmp_path): [str(root)]})
    assert user_source(orchestrator).project_roots() == {str(tmp_path): [str(target)]}


@pytest.mark.asyncio
async def test_explicit_owner_root_refresh_advances_only_copied_source(
    tmp_path: Path,
) -> None:
    parent, path = await make_orchestrator(tmp_path)
    child = parent._copy_for_child()
    previous = user_source(parent)
    disk = path.read_bytes()
    before = child.restrictions
    staged_parent = await parent._prepare_root_replacement(
        source="renamed-user",
        roots={str(tmp_path): [str(tmp_path / "new")]},
        expected_token=parent.accepted_token,
    )
    replacement = user_source(staged_parent)
    token = child.accepted_token
    staged_child = await child._stage_policy_refresh(
        previous=previous, replacement=replacement, expected_token=token
    )
    assert child.restrictions is before
    assert user_source(staged_child).project_roots() == replacement.project_roots()
    child._commit_policy_replacement(staged_child, expected_token=token)
    child.rebuild()
    assert user_source(child).project_roots() == replacement.project_roots()
    assert user_source(child).identity == previous.identity
    assert path.read_bytes() == disk
    with pytest.raises(ValueError, match="identity changed"):
        await child._stage_policy_refresh(
            previous=replacement,
            replacement=replace(replacement, locator="different"),
            expected_token=child.accepted_token,
        )
    target = tmp_path / "target"
    target.mkdir()
    (tmp_path / "new").symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="interpretation changed"):
        child.rebuild()


def test_tool_only_projection_preserves_roots_and_explicit_empty_revokes(
    tmp_path: Path,
) -> None:
    data = {
        ROOTS_FIELD: {str(tmp_path): [str(tmp_path / "old")]},
        "tools": {"bash": {"denylist": ["old"]}},
    }
    tool = SourceRestrictions.from_raw(
        {"tools": {}},
        layer_name="user",
        locator="user",
        kind="source",
        store_fingerprint=None,
        replacement=True,
    )
    assert tool.replace_in(data)[ROOTS_FIELD] == data[ROOTS_FIELD]
    assert replace(tool, replaces_roots=True).replace_in(data)[ROOTS_FIELD] == {}
    assert data["tools"]["bash"]["denylist"] == ["old"]
