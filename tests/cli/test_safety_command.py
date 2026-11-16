"""v0.8.0 A-P1 chunk 2: the ``/safety`` command surface, the startup-flag
rejection, and the DANGEROUS ceremony screens/banner.
"""

from __future__ import annotations

import asyncio

import pytest
from textual.app import App
from textual.widgets import Button, Input, Static

from chartreux.app_server._model import validate_wire
from chartreux.app_server.protocol import (
    SessionSafetyReadParams,
    SessionSafetyReadResponse,
    SessionSafetyTransitionParams,
    SessionSafetyTransitionResponse,
)
from chartreux.cli.commands import CommandRegistry
import chartreux.cli.entrypoint as entrypoint
from chartreux.cli.textual_ui.screens.safety_dangerous import (
    DANGEROUS_PHRASE,
    DangerousBanner,
    DangerousConfirmScreen,
    DangerousWarningScreen,
)


class TestSafetyCommandRegistration:
    def test_safety_alias_resolves(self) -> None:
        registry = CommandRegistry()
        assert registry.get_command_name("/safety") == "safety"

    def test_safety_command_is_side_channel_with_handler(self) -> None:
        registry = CommandRegistry()
        result = registry.parse_command("/safety dangerous")
        assert result is not None
        name, cmd, args = result
        assert name == "safety"
        assert cmd.handler == "_safety_command"
        assert cmd.side_channel is True
        assert args == "dangerous"

    def test_safety_bare_parses_with_empty_args(self) -> None:
        registry = CommandRegistry()
        result = registry.parse_command("/safety")
        assert result is not None
        assert result[0] == "safety"
        assert result[2] == ""


class TestDangerousStartupRejection:
    @pytest.mark.parametrize(
        "argv",
        [
            ["--dangerous"],
            ["--unguarded"],
            ["--bypass-tool-permissions"],
            ["--safety"],
            ["--safety-level"],
            ["--safety-mode"],
            ["--safety=dangerous"],
        ],
    )
    def test_level_naming_flags_exit_1(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        argv: list[str],
    ) -> None:
        for var in entrypoint._DANGEROUS_STARTUP_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        with pytest.raises(SystemExit) as exc_info:
            entrypoint._reject_dangerous_startup_enablement(argv)
        assert exc_info.value.code == 1
        assert "interactive ceremony" in capsys.readouterr().err

    @pytest.mark.parametrize("var", sorted(entrypoint._DANGEROUS_STARTUP_ENV_VARS))
    def test_dangerous_env_var_exits_1(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        var: str,
    ) -> None:
        for other in entrypoint._DANGEROUS_STARTUP_ENV_VARS:
            monkeypatch.delenv(other, raising=False)
        monkeypatch.setenv(var, "dangerous")
        with pytest.raises(SystemExit) as exc_info:
            entrypoint._reject_dangerous_startup_enablement([])
        assert exc_info.value.code == 1
        assert var in capsys.readouterr().err

    def test_benign_args_and_non_dangerous_env_pass(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CHARTREUX_SAFETY_LEVEL", "safe")
        for other in entrypoint._DANGEROUS_STARTUP_ENV_VARS:
            if other != "CHARTREUX_SAFETY_LEVEL":
                monkeypatch.delenv(other, raising=False)
        # No SystemExit: programmatic and ordinary flags are untouched.
        entrypoint._reject_dangerous_startup_enablement([
            "-p",
            "hello",
            "--model",
            "mock-provider/mock-model",
        ])


class TestSafetyWireModels:
    def test_transition_params_round_trip(self) -> None:
        params = SessionSafetyTransitionParams(session_id="s1", to="dangerous")
        assert (
            validate_wire(SessionSafetyTransitionParams, params.model_dump()) == params
        )

    def test_transition_response_defaults_reason(self) -> None:
        response = SessionSafetyTransitionResponse(level="safe", transitioned=False)
        assert response.reason == ""
        assert (
            validate_wire(SessionSafetyTransitionResponse, response.model_dump())
            == response
        )

    def test_read_models_round_trip(self) -> None:
        params = SessionSafetyReadParams(session_id="s1")
        response = SessionSafetyReadResponse(level="safe")
        assert validate_wire(SessionSafetyReadParams, params.model_dump()) == params
        assert (
            validate_wire(SessionSafetyReadResponse, response.model_dump()) == response
        )


class _ScreenApp(App[None]):
    """Minimal host so a modal screen can be pushed under run_test."""


def _push(app: _ScreenApp, screen) -> tuple[asyncio.Event, dict[str, object]]:
    """Push a modal screen and return (done-event, result-holder)."""
    done = asyncio.Event()
    holder: dict[str, object] = {}

    def _cb(result: object) -> None:
        holder["result"] = result
        done.set()

    app.push_screen(screen, _cb)
    return done, holder


@pytest.mark.asyncio
async def test_warning_screen_rejects_wrong_phrase_and_escape_then_accepts_phrase() -> (
    None
):
    app = _ScreenApp()
    async with app.run_test(size=(80, 24)) as pilot:
        done, holder = _push(app, DangerousWarningScreen())
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, DangerousWarningScreen)
        phrase_input = screen.query_one("#dangerous-phrase", Input)
        error = screen.query_one("#dangerous-warning-error", Static)

        phrase_input.value = "not the phrase"
        phrase_input.focus()
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert not done.is_set()
        assert "does not match" in str(error.render())

        # Escape is a priority no-op: the screen stays open.
        await pilot.press("escape")
        await pilot.pause()
        assert not done.is_set()
        assert app.screen is screen

        phrase_input.value = DANGEROUS_PHRASE
        await pilot.press("enter")
        await asyncio.wait_for(done.wait(), 5)
        assert holder["result"] == "phrase_ok"


@pytest.mark.asyncio
async def test_warning_screen_phrase_match_is_exact() -> None:
    app = _ScreenApp()
    async with app.run_test(size=(80, 24)) as pilot:
        done, holder = _push(app, DangerousWarningScreen())
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, DangerousWarningScreen)
        phrase_input = screen.query_one("#dangerous-phrase", Input)
        phrase_input.focus()
        await pilot.pause()
        phrase_input.value = DANGEROUS_PHRASE.lower()
        await pilot.press("enter")
        await pilot.pause()
        assert not done.is_set()
        # Cancel the ceremony so the screen settles.
        screen.dismiss(None)
        await asyncio.wait_for(done.wait(), 5)
        assert holder["result"] is None


@pytest.mark.asyncio
async def test_confirm_screen_buttons_and_escape() -> None:
    app = _ScreenApp()
    async with app.run_test(size=(80, 24)) as pilot:
        done, holder = _push(app, DangerousConfirmScreen())
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, DangerousConfirmScreen)
        await pilot.click("#dangerous-confirm-yes")
        await asyncio.wait_for(done.wait(), 5)
        assert holder["result"] is True

    app = _ScreenApp()
    async with app.run_test(size=(80, 24)) as pilot:
        done, holder = _push(app, DangerousConfirmScreen())
        await pilot.pause()
        await pilot.click("#dangerous-confirm-no")
        await asyncio.wait_for(done.wait(), 5)
        assert holder["result"] is False

    app = _ScreenApp()
    async with app.run_test(size=(80, 24)) as pilot:
        done, holder = _push(app, DangerousConfirmScreen())
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, DangerousConfirmScreen)
        assert screen.query_one("#dangerous-confirm-yes", Button)
        await pilot.press("escape")
        await asyncio.wait_for(done.wait(), 5)
        assert holder["result"] is False


def test_dangerous_banner_is_non_dismissible_marker() -> None:
    banner = DangerousBanner()
    assert banner.id == "dangerous-banner"
    assert "DANGEROUS MODE ACTIVE" in str(banner.render())
