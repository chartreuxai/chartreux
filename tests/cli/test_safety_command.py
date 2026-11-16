"""v0.8.0 A-P1 chunk 2: the ``/safety`` command surface, the startup-flag
rejection, and the DANGEROUS direct transition/banner (v0.8.2 W2 removed the
typed-phrase/confirmation ceremony screens).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.screen import ModalScreen

from chartreux.app_server._model import validate_wire
from chartreux.app_server.protocol import (
    SessionSafetyReadParams,
    SessionSafetyReadResponse,
    SessionSafetyTransitionParams,
    SessionSafetyTransitionResponse,
)
from chartreux.cli.commands import CommandRegistry
import chartreux.cli.entrypoint as entrypoint
from chartreux.cli.textual_ui.screens import safety_dangerous
from chartreux.cli.textual_ui.screens.safety_dangerous import DangerousBanner
from chartreux.core.config.models import SessionLoggingConfig
from tests.conftest import build_test_chartreux_app, build_test_vibe_config


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
        assert "`/safety dangerous`" in capsys.readouterr().err

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


class TestCeremonyRemoval:
    """v0.8.2 W2: the typed-phrase/confirmation modal screens are gone."""

    def test_modal_screens_and_phrase_are_gone(self) -> None:
        assert not hasattr(safety_dangerous, "DangerousWarningScreen")
        assert not hasattr(safety_dangerous, "DangerousConfirmScreen")
        assert not hasattr(safety_dangerous, "DANGEROUS_PHRASE")
        assert not hasattr(safety_dangerous, "WARNING_TEXT")

    def test_modal_screens_are_not_importable(self) -> None:
        with pytest.raises(ImportError):
            from chartreux.cli.textual_ui.screens.safety_dangerous import (
                DangerousConfirmScreen,  # type: ignore[reportAttributeAccessIssue] # noqa: F401
            )
        with pytest.raises(ImportError):
            from chartreux.cli.textual_ui.screens.safety_dangerous import (
                DangerousWarningScreen,  # type: ignore[reportAttributeAccessIssue] # noqa: F401
            )


def _dangerous_capable_config(tmp_path: Path):
    """Test config with a writable session audit sink (server-side gate for
    entering DANGEROUS fails closed without one).
    """
    return build_test_vibe_config(
        session_logging=SessionLoggingConfig(enabled=True, save_dir=str(tmp_path))
    )


@pytest.mark.asyncio
async def test_safety_command_dangerous_transitions_directly(tmp_path: Path) -> None:
    """``/safety dangerous`` reaches DANGEROUS with no modal on the stack."""
    app = build_test_chartreux_app(config=_dangerous_capable_config(tmp_path))
    async with app.run_test(size=(100, 30)) as pilot:
        await app._session_ready.wait()
        await app.app_server.resources.runtime.wait_until_ready()
        await pilot.pause()
        assert await app.app_server.safety_level() == "safe"
        await app._safety_command("dangerous")
        await pilot.pause()
        assert await app.app_server.safety_level() == "dangerous"
        # Direct transition: no ceremony modal was pushed.
        assert not isinstance(app.screen, ModalScreen)
        assert len(app.screen_stack) == 1
        # Banner and tint mounted.
        assert app.query("#dangerous-banner")
        assert app.query_one("#input-container").has_class("safety-dangerous")


@pytest.mark.asyncio
async def test_idle_shift_tab_cycle_reaches_dangerous_directly(tmp_path: Path) -> None:
    """Pin: idle Shift+Tab cycling into dangerous opens no modal."""
    app = build_test_chartreux_app(config=_dangerous_capable_config(tmp_path))
    async with app.run_test(size=(100, 30)) as pilot:
        await app._session_ready.wait()
        await app.app_server.resources.runtime.wait_until_ready()
        await pilot.pause()
        await app._cycle_safety_target("dangerous")
        await pilot.pause()
        assert await app.app_server.safety_level() == "dangerous"
        assert not isinstance(app.screen, ModalScreen)
        assert len(app.screen_stack) == 1
        assert app.query("#dangerous-banner")


@pytest.mark.asyncio
async def test_dangerous_transition_remains_idle_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The idle-only gate survives the ceremony removal on both entry paths."""
    app = build_test_chartreux_app(config=_dangerous_capable_config(tmp_path))
    async with app.run_test(size=(100, 30)) as pilot:
        await app._session_ready.wait()
        await app.app_server.resources.runtime.wait_until_ready()
        await pilot.pause()
        monkeypatch.setattr(app, "_agent_job_active", lambda: True)
        await app._safety_command("dangerous")
        await app._cycle_safety_target("dangerous")
        await pilot.pause()
        assert await app.app_server.safety_level() == "safe"
        assert not app.query("#dangerous-banner")


def test_dangerous_banner_is_non_dismissible_marker() -> None:
    banner = DangerousBanner()
    assert banner.id == "dangerous-banner"
    assert "DANGEROUS MODE ACTIVE" in str(banner.render())
