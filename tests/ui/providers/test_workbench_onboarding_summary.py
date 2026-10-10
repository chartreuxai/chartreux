"""Collapsed first-run setup with the real atomic catalog writer."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tomllib

import pytest
from textual.widgets import OptionList

from chartreux.core.dispatch.lint import lint_catalog, roster_shape
from chartreux.core.dispatch.presets import SHIPPED_PRESETS
from chartreux.core.dispatch.schema import DispatchMode
from chartreux.core.model_catalog.loader import load_catalog
from chartreux.core.model_catalog.resolver import ModelResolver
from chartreux.ui.providers.management_state import ManagementState
from tests.integration.test_provider_workbench_boundaries import Services, add_provider
from tests.ui.providers.test_workbench import Host, press_option, wait_until


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", list(DispatchMode))
async def test_empty_home_keyless_setup_finishes_without_role_questions(
    tmp_path: Path, mode: DispatchMode
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
        assert screen.state._dispatch.mode == DispatchMode.STANDALONE
        assert "Standalone — direct implementation" in str(
            rows.get_option("mode:standalone").prompt
        )
        assert all(
            screen.state.preset(role)[0] == "local-wire"
            for role in screen.state.catalog.roles
        )
        assert not screen.state.validate(
            mode="onboarding", credential_resolver=lambda _: None
        ).errors
        await press_option(pilot, rows, f"mode:{mode.value}")
        await press_option(
            pilot, screen.query_one("#wb-presets-actions", OptionList), "finish"
        )
        await wait_until(pilot, lambda: bool(host.results))
        assert host.results[-1].status == "completed"
    saved = load_catalog(tmp_path / "models.toml")
    assert saved.dispatch.mode == mode
    assert not [
        item
        for item in lint_catalog(saved.dispatch, saved.catalog)
        if item.severity == "error"
    ]
    resolver = ModelResolver(saved)
    assert roster_shape(
        resolver.resolve(slot.role) for slot in saved.dispatch.slots.values()
    ).single_model
    # No explicit slot assignments are needed for one canonical model.
    assert "slots" not in tomllib.loads((tmp_path / "models.toml").read_text()).get(
        "dispatch", {}
    )


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
        original = screen.state.preset("orchestrator")
        rows = screen.query_one("#wb-presets", OptionList)
        await press_option(pilot, rows, "customize")
        await press_option(pilot, rows, "preset:orchestrator")
        screen._preset_draft = ("local-wire", "low")
        await press_option(
            pilot, screen.query_one("#wb-preset-editor-actions", OptionList), "cancel"
        )
        assert screen.state.preset("orchestrator") == original
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


def test_completion_checks_dispatch_roles_not_unreferenced_drafts(
    tmp_path: Path,
) -> None:
    snapshot = load_catalog(tmp_path / "models.toml")
    roles = dict(snapshot.catalog.roles)
    roles["draft"] = roles["scout"].model_copy(update={"model": "not-configured"})
    state = ManagementState.from_snapshot(
        replace(snapshot, catalog=snapshot.catalog.model_copy(update={"roles": roles})),
        "mistral",
    )
    assert not state.validate(
        mode="onboarding", credential_resolver=lambda _: "ready"
    ).errors
    state.set_role_preset("scout", "not-configured", "low")
    assert (
        "scout"
        in state.validate(
            mode="onboarding", credential_resolver=lambda _: "ready"
        ).unresolved_roles
    )


def test_mode_save_preserves_custom_slots_and_uses_new_preset(tmp_path: Path) -> None:
    snapshot = load_catalog(tmp_path / "models.toml")
    slots = dict(snapshot.dispatch.slots)
    slots["mechanical"] = slots["mechanical"].model_copy(update={"role": "@heavy"})
    snapshot = replace(
        snapshot, dispatch=snapshot.dispatch.model_copy(update={"slots": slots})
    )
    state = ManagementState.from_snapshot(snapshot, "mistral")
    state.dispatch_mode = DispatchMode.ORCHESTRATED
    changes = state.changes()
    assert changes.dispatch is not None
    assert changes.dispatch["mode"] == "orchestrated"
    assert state._dispatch.slots["mechanical"].role == "@heavy"
    assert (
        state._dispatch.instructions
        == SHIPPED_PRESETS[DispatchMode.ORCHESTRATED].instructions
    )


@pytest.mark.asyncio
async def test_invalid_dispatch_requires_explicit_repair_not_incidental_seeding(
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
        assert not host.results
        assert screen._feedback_kind == "error"
        assert screen._customize_presets
        assert path.read_text() == '[dispatch]\nmode = "invalid"\n'
        await press_option(
            pilot, screen.query_one("#wb-presets", OptionList), "mode:orchestrated"
        )
        await press_option(
            pilot, screen.query_one("#wb-presets-actions", OptionList), "finish"
        )
        await wait_until(pilot, lambda: bool(host.results))
        assert host.results[-1].status == "completed"
    assert load_catalog(path).dispatch.mode == DispatchMode.ORCHESTRATED
    assert not load_catalog(path).dispatch_diagnostics
