"""Collapsed first-run setup with the real atomic catalog writer."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tomllib

import pytest
from textual.widgets import OptionList

from chartreux.core.model_catalog.loader import load_catalog
from chartreux.ui.providers.management_state import ManagementState
from tests.integration.test_provider_workbench_boundaries import Services, add_provider
from tests.ui.providers._workbench_support import Host, press_option, wait_until


@pytest.mark.asyncio
async def test_empty_home_keyless_setup_finishes_without_role_questions(
    tmp_path: Path,
) -> None:
    services = Services(tmp_path)
    screen = services.screen()
    screen.mode = "onboarding"
    screen.credential_resolver = lambda _: None
    host = Host(screen)
    async with host.run_test(size=(80, 24)) as pilot:
        await add_provider(pilot, screen, "local")
        rows = screen.query_one("#wb-presets", OptionList)
        assert not any(str(option.id).startswith("preset:") for option in rows.options)
        assert screen.state is not None
        assert all(
            screen.state.preset(role)[0] == "local-wire"
            for role in screen.state.catalog.roles
        )
        assert not screen.state.validate(
            mode="onboarding", credential_resolver=lambda _: None
        ).errors
        await press_option(
            pilot, screen.query_one("#wb-presets-actions", OptionList), "finish"
        )
        await wait_until(pilot, lambda: bool(host.results))
        assert host.results[-1].status == "completed"
    saved = load_catalog(tmp_path / "models.toml")
    assert all(role.model == "local-wire" for role in saved.catalog.roles.values())
    # The catalog owns connections and presets; it carries no dispatch policy.
    assert "dispatch" not in tomllib.loads((tmp_path / "models.toml").read_text())


@pytest.mark.asyncio
async def test_summary_customize_cancel_back_and_failed_save(tmp_path: Path) -> None:
    services = Services(tmp_path)
    screen = services.screen()
    screen.mode = "onboarding"
    screen.credential_resolver = lambda _: None
    host = Host(screen)
    async with host.run_test(size=(80, 24)) as pilot:
        await add_provider(pilot, screen, "local")
        assert screen.state is not None
        original = screen.state.preset("main")
        rows = screen.query_one("#wb-presets", OptionList)
        await press_option(pilot, rows, "customize")
        await press_option(pilot, rows, "preset:main")
        screen._preset_draft = ("local-wire", "low")
        await press_option(
            pilot, screen.query_one("#wb-preset-editor-actions", OptionList), "cancel"
        )
        assert screen.state.preset("main") == original
        await pilot.press("escape")
        assert rows.get_option("customize")
        assert not any(str(option.id).startswith("preset:") for option in rows.options)
        before = (tmp_path / "models.toml").read_bytes()
        services.fail_once = True
        await press_option(
            pilot, screen.query_one("#wb-presets-actions", OptionList), "finish"
        )
        await wait_until(pilot, lambda: not screen._busy)
        assert not host.results
        assert screen.state.dirty
        assert (tmp_path / "models.toml").read_bytes() == before
        await press_option(
            pilot, screen.query_one("#wb-presets-actions", OptionList), "finish"
        )
        await wait_until(pilot, lambda: bool(host.results))
        assert host.results[-1].status == "completed"


@pytest.mark.asyncio
async def test_back_and_cancel_summary_keep_explicit_provider_saves(
    tmp_path: Path,
) -> None:
    services = Services(tmp_path)
    screen = services.screen()
    screen.mode = "onboarding"
    screen.credential_resolver = lambda _: None
    host = Host(screen)
    async with host.run_test(size=(80, 24)) as pilot:
        await add_provider(pilot, screen, "local")
        before = (tmp_path / "models.toml").read_bytes()
        await pilot.press("escape")
        assert screen.query_one("#wb-providers").has_focus
        for _ in range(4):
            await pilot.press("escape")
            if screen._confirm:
                break
        assert screen._confirm == "close"
        screen.action_confirm_yes()
        await wait_until(pilot, lambda: bool(host.results))
        assert host.results[-1].status == "cancelled"
        assert host.results[-1].changed
        assert (tmp_path / "models.toml").read_bytes() == before
        assert "local" in load_catalog(tmp_path / "models.toml").catalog.providers


def test_completion_requires_every_defined_preset_role(tmp_path: Path) -> None:
    snapshot = load_catalog(tmp_path / "models.toml")
    roles = dict(snapshot.catalog.roles)
    roles["draft"] = roles["scout"].model_copy(update={"model": "not-configured"})
    state = ManagementState.from_snapshot(
        replace(snapshot, catalog=snapshot.catalog.model_copy(update={"roles": roles})),
        "mistral",
    )
    validation = state.validate(
        mode="onboarding", credential_resolver=lambda _: "ready"
    )
    assert "draft" in validation.unresolved_roles
    assert validation.errors
    state.set_role_preset("scout", "not-configured", "low")
    validation = state.validate(
        mode="onboarding", credential_resolver=lambda _: "ready"
    )
    assert "scout" in validation.unresolved_roles


@pytest.mark.asyncio
async def test_legacy_dispatch_table_neither_blocks_nor_rewrites_setup(
    tmp_path: Path,
) -> None:
    path = tmp_path / "models.toml"
    path.write_text('[dispatch]\nmode = "invalid"\n')
    services = Services(tmp_path)
    screen = services.screen()
    screen.mode = "onboarding"
    screen.initial_view = "presets"
    host = Host(screen)
    async with host.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await press_option(
            pilot, screen.query_one("#wb-presets-actions", OptionList), "finish"
        )
        await wait_until(pilot, lambda: bool(host.results))
        assert host.results[-1].status == "completed"
    # Setup no longer owns dispatch policy: the legacy table is neither a
    # completion gate nor silently rewritten by the catalog writer.
    assert path.read_text() == '[dispatch]\nmode = "invalid"\n'
