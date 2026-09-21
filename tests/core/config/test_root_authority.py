from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest

from chartreux.core.config import ChartreuxConfigSchema
from chartreux.core.config._restrictions import SourceRestrictions
from chartreux.core.config._root_authority import ROOTS_FIELD, validate_root_source
from chartreux.core.config.builder import ConfigBuilder
from chartreux.core.config.layers.default import DefaultConfigLayer
from chartreux.core.config.layers.overrides import OverridesLayer
from chartreux.core.config.layers.user import UserConfigLayer
from chartreux.core.workspace import Workspace


def test_unverified_root_contribution_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="root.*provenance"):
        SourceRestrictions.from_raw(
            {"authorized_roots_by_project": {str(tmp_path): [str(tmp_path.parent)]}},
            layer_name="user-toml",
            locator="user-toml",
            kind="source",
            store_fingerprint=None,
        )


@pytest.mark.parametrize(
    "name", ["user-toml", "user", "default", "environment", "agent-profile"]
)
@pytest.mark.parametrize("empty", [False, True])
def test_names_never_authorize_roots(tmp_path: Path, name: str, empty: bool) -> None:
    data = {ROOTS_FIELD: {} if empty else {str(tmp_path): [str(tmp_path.parent)]}}
    with pytest.raises(ValueError, match="actual user source"):
        validate_root_source(data, layer=OverridesLayer(name=name, data=data))


def test_only_empty_actual_default_is_tolerated(tmp_path: Path) -> None:
    layer = DefaultConfigLayer(schema=ChartreuxConfigSchema, name="renamed")
    assert validate_root_source({ROOTS_FIELD: {}}, layer=layer) == ()
    with pytest.raises(ValueError, match="actual user source"):
        validate_root_source({ROOTS_FIELD: {str(tmp_path): []}}, layer=layer)


def test_subclass_cannot_claim_user_authority(tmp_path: Path) -> None:
    class FakeUserLayer(UserConfigLayer):
        pass

    with pytest.raises(ValueError, match="actual user source"):
        validate_root_source(
            {ROOTS_FIELD: {}}, layer=FakeUserLayer(path=tmp_path / "user.toml")
        )


def test_projection_canonical_immutable_and_separate_from_merge(tmp_path: Path) -> None:
    project, root = tmp_path / "project", tmp_path / "extra"
    data = {ROOTS_FIELD: {str(project): [str(root), str(root)]}}
    layer = UserConfigLayer(path=tmp_path / "user.toml", name="renamed")
    source = SourceRestrictions.from_raw(
        data,
        layer_name=layer.name,
        locator=layer.source_locator,
        kind="source",
        store_fingerprint="accepted-source",
        root_layer=layer,
    )
    assert source.authorized_roots == validate_root_source(data, layer=layer)
    item = source.authorized_roots[0]
    assert item.project == project and item.roots == (root,)
    frozen_field = "roots"
    with pytest.raises(FrozenInstanceError):
        setattr(item, frozen_field, ())
    data[ROOTS_FIELD][str(project)].clear()
    config = ChartreuxConfigSchema.model_validate({
        ROOTS_FIELD: {str(project): [str(root)]}
    })
    config.authorized_roots_by_project.clear()
    assert item.roots == (root,)
    assert not Workspace.for_session(project).allows(root)
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
        {"relative": []},
        {"/project": ["relative"]},
        {"/project": [42]},
        {"/project": "secret-marker"},
    ],
)
def test_invalid_definitions_are_sanitized(tmp_path: Path, value: Any) -> None:
    with pytest.raises(ValueError) as caught:
        validate_root_source(
            {ROOTS_FIELD: value}, layer=UserConfigLayer(path=tmp_path / "user.toml")
        )
    assert str(caught.value) == (
        "Invalid authorized_roots_by_project definition (source: user)"
    )
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_colliding_canonical_project_keys_rejected(tmp_path: Path) -> None:
    alias = tmp_path / "alias"
    project = tmp_path / "project"
    project.mkdir()
    alias.symlink_to(project, target_is_directory=True)
    with pytest.raises(ValueError, match="Invalid authorized"):
        validate_root_source(
            {ROOTS_FIELD: {str(alias): [], str(project): []}},
            layer=UserConfigLayer(path=tmp_path / "user.toml"),
        )


@pytest.mark.asyncio
async def test_builder_accepts_real_user_roots(tmp_path: Path) -> None:
    path = tmp_path / "user.toml"
    path.write_text(
        f'[authorized_roots_by_project]\n"{tmp_path}" = ["{tmp_path.parent}"]\n'
    )
    builder = ConfigBuilder(ChartreuxConfigSchema)
    builder.add_layers([
        DefaultConfigLayer(schema=ChartreuxConfigSchema),
        UserConfigLayer(path=path),
    ])
    candidate = await builder.build_candidate()
    assert candidate.restrictions[-1].authorized_roots[0].project == tmp_path
    assert candidate.restrictions[-1].authorized_roots[0].roots == (tmp_path.parent,)
    assert (await builder.build()).authorized_roots_by_project == {
        str(tmp_path): [str(tmp_path.parent)]
    }
