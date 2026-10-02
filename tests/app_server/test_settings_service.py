from __future__ import annotations

from dataclasses import asdict
from typing import cast
from unittest.mock import AsyncMock

import pytest

from chartreux.app_server._runtime_resources import ConfigResource
from chartreux.app_server.protocol import (
    SettingDescriptorWire,
    SettingLeafWire,
    SettingsReadResponse,
)
from chartreux.core.config.settings_catalog import VISIBLE_SETTINGS
from chartreux.ui.settings_service import SettingsService


def snapshot(
    *,
    revision: str | None = "one",
    saved: bool = False,
    origin: str = "default",
    value: bool = True,
) -> SettingsReadResponse:
    from chartreux.core.config.settings_catalog import EDITABLE_BY_PATH

    return SettingsReadResponse(
        user_layer="user-toml",
        user_revision=revision,
        backing_settings={
            path: SettingDescriptorWire.model_validate(asdict(EDITABLE_BY_PATH[path]))
            for path in ("enabled_tools", "disabled_tools")
        },
        catalog=[
            SettingDescriptorWire.model_validate(asdict(item))
            for item in VISIBLE_SETTINGS
        ],
        fields=[
            SettingLeafWire(
                path="session_logging.enabled",
                effective_value=value,
                saved_explicit=saved,
                saved_value=False if saved else None,
                origin=origin,
            )
        ],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("persistence", ["saved", "durability_uncertain", "not_saved"])
@pytest.mark.parametrize("application", ["unchanged", "applied", "failed"])
async def test_save_outcomes(persistence: str, application: str) -> None:
    config = AsyncMock()
    config.read_settings.side_effect = [
        snapshot(),
        snapshot(saved=True, origin="user-toml", value=False),
    ]
    config.write.return_value.persistence = persistence
    config.write.return_value.application = application
    config.write.return_value.failures = []
    config.write.return_value.fields = ["[redacted]"]
    config.write.return_value.saved_values = {"session_logging": "[redacted]"}
    apply_ui = AsyncMock()
    service = SettingsService(cast(ConfigResource, config), apply_ui=apply_ui)
    outcome = await service.save({"session_logging.enabled": False}, "one")
    assert (outcome.persistence, outcome.application) == (persistence, application)
    assert (
        outcome.snapshot is not None
        if persistence != "not_saved"
        else outcome.snapshot is None
    )
    assert config.read_settings.await_count == (2 if persistence != "not_saved" else 1)
    assert apply_ui.await_count == (
        1 if application == "applied" and persistence != "not_saved" else 0
    )
    ops = config.write.await_args.args[0]
    assert len(ops) == 1 and ops[0].path == "/session_logging/enabled"
    assert ops[0].op == "set" and ops[0].target_layer == "user-toml"
    assert config.write.await_args.kwargs["expected_revision"] == "one"


@pytest.mark.asyncio
async def test_ui_update_failure_preserves_saved_revision() -> None:
    config = AsyncMock()
    after = snapshot(revision="two", saved=True, origin="user-toml", value=False)
    config.read_settings.side_effect = [snapshot(), after]
    config.write.return_value.persistence = "saved"
    config.write.return_value.application = "applied"
    apply_ui = AsyncMock(side_effect=RuntimeError("UI failed"))
    outcome = await SettingsService(
        cast(ConfigResource, config), apply_ui=apply_ui
    ).save({"session_logging.enabled": False}, "one")
    assert outcome.persistence == "saved"
    assert outcome.application == "failed"
    assert outcome.error == "ui_update_failed"
    assert outcome.snapshot is after
    assert after.user_revision == "two"


@pytest.mark.asyncio
async def test_remove_override_and_shadowed_save() -> None:
    config = AsyncMock()
    config.read_settings.side_effect = [
        snapshot(saved=True, origin="overrides"),
        snapshot(saved=False, origin="overrides"),
    ]
    config.write.return_value.persistence = "saved"
    config.write.return_value.application = "applied"
    outcome = await SettingsService(cast(ConfigResource, config)).save(
        {"session_logging.enabled": None}, "one"
    )
    assert outcome.shadowed == ()
    assert outcome.snapshot and not outcome.snapshot.fields[0].saved_explicit
    assert config.write.await_args.args[0][0].op == "remove"

    config.read_settings.side_effect = [
        snapshot(origin="overrides"),
        snapshot(saved=True, origin="overrides"),
    ]
    outcome = await SettingsService(cast(ConfigResource, config)).save(
        {"session_logging.enabled": False}, "one"
    )
    assert outcome.shadowed == ("session_logging.enabled",)


@pytest.mark.asyncio
async def test_equal_to_default_is_explicit_set_not_omitted() -> None:
    config = AsyncMock()
    before = snapshot(value=False)
    after = snapshot(saved=True, origin="user-toml", value=False)
    config.read_settings.side_effect = [before, after]
    config.write.return_value.persistence = "saved"
    config.write.return_value.application = "applied"
    outcome = await SettingsService(cast(ConfigResource, config)).save(
        {"session_logging.enabled": False}, "one"
    )
    assert outcome.snapshot and outcome.snapshot.fields[0].saved_explicit
    op = config.write.await_args.args[0][0]
    assert op.op == "set" and op.value is False


@pytest.mark.asyncio
async def test_list_leaf_is_one_json_pointer_set_and_validates_elements() -> None:
    config = AsyncMock()
    before = snapshot()
    after = snapshot()
    for snap in (before, after):
        snap.fields.append(
            SettingLeafWire(
                path="enabled_tools",
                effective_value=[],
                saved_explicit=False,
                origin="default",
            )
        )
    config.read_settings.side_effect = [before, after]
    config.write.return_value.persistence = "saved"
    config.write.return_value.application = "applied"
    service = SettingsService(cast(ConfigResource, config))
    outcome = await service.save({"enabled_tools": ["re:^foo", "bar*"]}, "one")
    assert outcome.persistence == "saved"
    ops = config.write.await_args.args[0]
    assert len(ops) == 1
    assert ops[0].path == "/enabled_tools" and ops[0].value == ["re:^foo", "bar*"]
    config.read_settings.side_effect = None
    config.read_settings.return_value = before
    with pytest.raises(ValueError):
        await service.save({"enabled_tools": [False]}, "one")


@pytest.mark.asyncio
async def test_inventory_reset_removes_both_backing_leaves() -> None:
    config = AsyncMock()
    before, after = snapshot(), snapshot()
    for snap, saved in ((before, True), (after, False)):
        for path in ("enabled_tools", "disabled_tools"):
            snap.fields.append(
                SettingLeafWire(
                    path=path,
                    effective_value=[],
                    saved_explicit=saved,
                    saved_value=[] if saved else None,
                    origin="user-toml" if saved else "default",
                )
            )
    config.read_settings.side_effect = [before, after]
    config.write.return_value.persistence = "saved"
    config.write.return_value.application = "applied"
    outcome = await SettingsService(cast(ConfigResource, config)).save(
        {"enabled_tools": None, "disabled_tools": None}, "one"
    )
    assert outcome.snapshot is not None
    assert all(not field.saved_explicit for field in outcome.snapshot.fields[1:])
    assert [(op.path, op.op) for op in config.write.await_args.args[0]] == [
        ("/enabled_tools", "remove"),
        ("/disabled_tools", "remove"),
    ]


@pytest.mark.asyncio
async def test_inventory_batch_writes_both_backing_leaves_in_one_revision() -> None:
    config = AsyncMock()
    before, after = snapshot(), snapshot()
    for snap in (before, after):
        for path in ("enabled_tools", "disabled_tools"):
            snap.fields.append(
                SettingLeafWire(
                    path=path,
                    effective_value=[],
                    saved_explicit=False,
                    origin="default",
                )
            )
    config.read_settings.side_effect = [before, after]
    config.write.return_value.persistence = "saved"
    config.write.return_value.application = "applied"
    service = SettingsService(cast(ConfigResource, config))
    outcome = await service.save({"enabled_tools": [], "disabled_tools": []}, "one")
    assert outcome.persistence == "saved"
    ops = config.write.await_args.args[0]
    assert [(op.path, op.value) for op in ops] == [
        ("/enabled_tools", []),
        ("/disabled_tools", []),
    ]
    assert config.write.await_count == 1
    assert config.write.await_args.kwargs["expected_revision"] == "one"
    with pytest.raises(ValueError, match="Unknown settings leaf"):
        await SettingsService(
            cast(
                ConfigResource, AsyncMock(read_settings=AsyncMock(return_value=before))
            )
        ).save({"inventory_tools": []}, "one")

    config = AsyncMock()
    config.read_settings.return_value = snapshot(revision="new")
    service = SettingsService(cast(ConfigResource, config))
    assert (
        await service.save({"session_logging.enabled": False}, "old")
    ).error == "conflict"
    config.read_settings.return_value = snapshot(revision=None)
    assert (await service.read()).view_only
    assert (
        await service.save({"session_logging.enabled": False}, "old")
    ).error == "view_only"
    config.write.assert_not_awaited()
