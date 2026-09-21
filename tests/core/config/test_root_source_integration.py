from __future__ import annotations

from pathlib import Path
from typing import Any, Literal
from unittest.mock import patch

import pytest

from chartreux.core.config import ChartreuxConfigSchema
from chartreux.core.config._restrictions import SourceRestrictions
from chartreux.core.config._root_authority import ROOTS_FIELD
from chartreux.core.config.builder import ConfigBuilder
from chartreux.core.config.layer import LayerImplementationError, RawConfig
from chartreux.core.config.layers.default import DefaultConfigLayer
from chartreux.core.config.layers.environment import EnvironmentLayer
from chartreux.core.config.layers.overrides import OverridesLayer
from chartreux.core.config.layers.user import UserConfigLayer
from chartreux.core.config.orchestrator import (
    ConfigOrchestrator,
    ConfigPatchValidationError,
)
from chartreux.core.config.patch import AddOperationPatch, PatchOp, RemoveOperationPatch


async def user_orchestrator(
    tmp_path: Path, *, roots: bool = True
) -> tuple[ConfigOrchestrator[ChartreuxConfigSchema], Path]:
    path = tmp_path / "user.toml"
    path.write_text(
        f'[{ROOTS_FIELD}]\n"{tmp_path}" = ["{tmp_path / "extra"}"]\n' if roots else ""
    )
    user = UserConfigLayer(path=path, name="renamed-user")
    orchestrator = await ConfigOrchestrator.create(
        schema=ChartreuxConfigSchema,
        layers=[DefaultConfigLayer(schema=ChartreuxConfigSchema), user],
        default_layer_resolver=lambda: user,
    )
    return orchestrator, path


@pytest.mark.asyncio
async def test_real_user_roots_are_accepted_and_immutable(tmp_path: Path) -> None:
    orchestrator, _ = await user_orchestrator(tmp_path)
    roots = orchestrator.restrictions[-1].authorized_roots
    assert roots[0].project == tmp_path
    assert roots[0].roots == (tmp_path / "extra",)
    orchestrator.config.authorized_roots_by_project.clear()
    assert orchestrator.restrictions[-1].authorized_roots == roots
    assert await orchestrator.set_field("/active_model", "", reason="ordinary") == []
    assert orchestrator.restrictions[-1].authorized_roots == roots
    staged = await orchestrator._stage_policy_replacement(
        source="renamed-user", tools={}, expected_token=orchestrator.accepted_token
    )
    assert staged.restrictions[-1].authorized_roots == roots
    child = orchestrator._copy_for_child()
    refreshed = await child._stage_policy_refresh(
        previous=orchestrator.restrictions[-1],
        replacement=staged.restrictions[-1],
        expected_token=child.accepted_token,
    )
    assert refreshed.restrictions[-1].authorized_roots == roots


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["build", "build_candidate"])
@pytest.mark.parametrize("value", [{}, None, [], {"/private-project": ["secret"]}])
@pytest.mark.parametrize("prospective", [False, True])
async def test_forbidden_sources_fail_before_shadow_and_cache(
    method: Literal["build", "build_candidate"], value: Any, prospective: bool
) -> None:
    spoof = OverridesLayer(
        name="user-toml", data={} if prospective else {ROOTS_FIELD: value}
    )
    builder = ConfigBuilder(ChartreuxConfigSchema)
    builder.add_layers([spoof, DefaultConfigLayer(schema=ChartreuxConfigSchema)])
    overrides = {
        spoof.name: RawConfig.model_validate(
            {ROOTS_FIELD: value} if prospective else {}
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
    path.write_text(f'[{ROOTS_FIELD}]\n"/private-project" = ["secret"]\n')
    user = UserConfigLayer(path=path)
    builder = ConfigBuilder(ChartreuxConfigSchema)
    builder.add_layers([DefaultConfigLayer(schema=ChartreuxConfigSchema), user])
    with pytest.raises(ValueError, match=ROOTS_FIELD) as caught:
        await getattr(builder, method)(layer_overrides={user.name: RawConfig()})
    assert caught.value.__context__ is None
    assert "private" not in str(caught.value)
    assert "secret" not in str(caught.value)
    assert user.cached_data is None
    assert builder.layers[0].cached_data is None


@pytest.mark.asyncio
@pytest.mark.parametrize("suffix", ["", "__/private-project"])
@pytest.mark.parametrize("value", ["", "{}", "null", "secret-malformed-json"])
async def test_environment_roots_rejected_before_decode(
    monkeypatch: pytest.MonkeyPatch, suffix: str, value: str
) -> None:
    monkeypatch.setenv(f"CHARTREUX_{ROOTS_FIELD.upper()}{suffix}", value)
    layer = EnvironmentLayer(schema=ChartreuxConfigSchema)
    with patch("chartreux.core.config.layers.environment.EnvSettingsSource") as decoder:
        with pytest.raises(LayerImplementationError) as caught:
            await layer.load()
    decoder.assert_not_called()
    cause = caught.value.__cause__
    assert cause is not None
    assert ROOTS_FIELD in str(cause)
    assert "secret" not in str(cause)
    assert "private" not in str(cause)
    assert cause.__context__ is None
    assert layer.cached_data is None
    assert layer.fingerprint is None


@pytest.mark.asyncio
@pytest.mark.parametrize("save", [False, True])
@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize(
    "operation", ["set", "remove", "document", "erase-document", "set-remove"]
)
async def test_ordinary_user_patch_cannot_change_roots(
    tmp_path: Path, save: bool, existing: bool, operation: str
) -> None:
    orchestrator, path = await user_orchestrator(tmp_path, roots=existing)
    layer = orchestrator.get_layer("renamed-user")
    roots = {str(tmp_path): [str(tmp_path / "new-secret-root")]}
    ops: list[PatchOp]
    match operation:
        case "set":
            ops = [AddOperationPatch(path=f"/{ROOTS_FIELD}", value=roots)]
        case "remove":
            ops = [RemoveOperationPatch(path=f"/{ROOTS_FIELD}")]
        case "document":
            ops = [AddOperationPatch(path="", value={ROOTS_FIELD: roots})]
        case "erase-document":
            if not existing:
                pytest.skip("No root authority exists to remove")
            ops = [AddOperationPatch(path="", value={})]
        case _:
            ops = [
                AddOperationPatch(path=f"/{ROOTS_FIELD}", value=roots),
                RemoveOperationPatch(path=f"/{ROOTS_FIELD}"),
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
        with pytest.raises(ConfigPatchValidationError, match=ROOTS_FIELD) as caught:
            await orchestrator.apply_patch(ops, reason="ordinary")
        assert "new-secret-root" not in str(caught.value)
        assert caught.value.__context__ is None
    assert path.read_bytes() == before
    assert orchestrator.config is config
    assert orchestrator.restrictions is policy
    assert orchestrator.accepted_token is token
    assert layer.cached_data is cache


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
    with pytest.raises(ValueError, match=ROOTS_FIELD) as caught:
        if preview:
            await orchestrator.preview_candidate(
                layer_overrides={
                    layer.name: RawConfig.model_validate({
                        ROOTS_FIELD: {"/private": ["secret"]}
                    })
                }
            )
        else:
            path.write_text(f'[{ROOTS_FIELD}]\n"/private" = ["secret"]\n')
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
            {ROOTS_FIELD: value},
            layer_name="user-toml",
            locator="user-toml",
            kind="source",
            store_fingerprint=None,
        )
