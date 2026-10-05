from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from textual.widgets import Static

from chartreux.app_server.events import AgentsUpdate
from chartreux.cli.textual_ui.quit_manager import ExitConsequencesScreen
from chartreux.cli.textual_ui.screens.settings import SettingsOptionList, SettingsScreen
from chartreux.cli.textual_ui.screens.usage import UsageScreen
from chartreux.cli.textual_ui.widgets.debug_console import _LogView
from chartreux.cli.textual_ui.widgets.log_level_picker import LogLevelPickerApp
from chartreux.cli.textual_ui.widgets.mcp_app import MCPApp
from chartreux.cli.textual_ui.widgets.question_app import QuestionApp
from chartreux.ui.providers.workbench import ProviderWorkbenchScreen
from tests.cli.test_mcp_app import _source, _state
from tests.cli.textual_ui.test_agent_transcript_app import _agent
from tests.cli.textual_ui.test_app_server_requests import _question_callback
from tests.cli.textual_ui.test_message_queue_ui import _blocked_app, _wait_until
from tests.cli.textual_ui.test_usage_screen import snapshot
from tests.stubs.fake_config_orchestrator import FakeConfigOrchestrator


async def _start_turn(app, backend, pilot) -> None:
    await app._session_ready.wait()
    await app.app_server.resources.runtime.wait_until_ready()
    await app._handle_user_message("keep working")
    assert await _wait_until(pilot, backend.started.is_set, timeout=5)
    assert app.app_server.turn_active


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (120, 36)])
async def test_console_escape_dismissal_preserves_running_turn_and_restores_focus(
    size,
) -> None:
    app, backend = _blocked_app()
    async with app.run_test(size=size) as pilot:
        await _start_turn(app, backend, pilot)
        opener = app.screen.focused
        assert opener is not None
        focusable = opener.can_focus
        await app.action_toggle_debug_console()
        await pilot.pause()
        console = app._debug_console
        assert console is not None
        console.query_one("#debug-console-log").focus()
        await pilot.pause()
        await pilot.press("escape")
        assert await _wait_until(pilot, lambda: app._debug_console is None)
        assert app.app_server.turn_active
        assert not app.app_server.turn_queue.paused
        assert app._interrupt_operation is None
        assert opener.can_focus == focusable
        assert app.screen.focused is opener
        # Once the local owner is gone, plain Escape still interrupts the turn.
        await pilot.press("escape")
        assert await _wait_until(
            pilot, lambda: not app.app_server.turn_active, timeout=5
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (120, 36)])
async def test_focused_console_escape_precedes_expanded_agent_browser(size) -> None:
    app, backend = _blocked_app()
    async with app.run_test(size=size) as pilot:
        await _start_turn(app, backend, pilot)
        await app._handle_turn_event(AgentsUpdate([_agent("one")]))
        await pilot.press("ctrl+shift+a")
        bar = app._agent_bar
        assert bar is not None and bar.expanded
        await app.action_toggle_debug_console()
        console = app._debug_console
        assert console is not None
        console.query_one("#debug-console-log").focus()
        await pilot.pause()
        assert console.owns_interaction
        queue = app.app_server.turn_queue.model_copy(deep=True)

        await pilot.press("escape")
        assert await _wait_until(pilot, lambda: app._debug_console is None)
        assert bar.expanded
        assert app.app_server.turn_active
        assert app.app_server.turn_queue == queue
        assert app._interrupt_operation is None
        assert not backend.release.is_set()

        await pilot.press("escape")
        assert await _wait_until(pilot, lambda: not bar.expanded)
        assert app.app_server.turn_active
        assert app.app_server.turn_queue == queue
        assert app._interrupt_operation is None
        assert not backend.release.is_set()
        backend.release.set()


@pytest.mark.asyncio
async def test_visible_dock_does_not_steal_composer_escape() -> None:
    app, backend = _blocked_app()
    async with app.run_test(size=(120, 36)) as pilot:
        await _start_turn(app, backend, pilot)
        await app.action_toggle_debug_console()
        console = app._debug_console
        assert console is not None
        assert app._chat_input_container is not None
        app._chat_input_container.focus_input()
        await pilot.pause()
        assert not console.owns_interaction
        await pilot.press("escape")
        assert await _wait_until(
            pilot, lambda: not app.app_server.turn_active, timeout=5
        )
        assert app._debug_console is not None


@pytest.mark.asyncio
async def test_console_keyboard_focus_cycle_and_live_resize_preserve_work() -> None:
    app, backend = _blocked_app()
    async with app.run_test(size=(120, 36)) as pilot:
        await _start_turn(app, backend, pilot)
        await app.action_toggle_debug_console()
        await pilot.pause()
        console = app._debug_console
        assert console is not None
        # Screen's focus traversal bypasses Widget.focus; this must still beat
        # the composer's blur-refocus policy.
        app.screen.focus_next("#debug-console-log")
        await pilot.pause()
        assert console.owns_interaction
        await pilot.press("tab")
        assert not console.owns_interaction
        assert app.query_one("#input").can_focus
        app.screen.focus_next("#debug-console-log")
        await pilot.pause()
        await pilot.resize_terminal(80, 24)
        await pilot.pause()
        assert console.owns_interaction
        assert console.has_class("-fullscreen")
        await pilot.resize_terminal(120, 36)
        await pilot.pause()
        assert console.owns_interaction
        await pilot.press("escape")
        assert await _wait_until(pilot, lambda: app._debug_console is None)
        assert app.app_server.turn_active
        assert app.query_one("#input").can_focus
        backend.release.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (120, 36)])
async def test_exit_modal_precedes_console_and_does_not_interrupt_work(size) -> None:
    app, backend = _blocked_app()
    async with app.run_test(size=size) as pilot:
        await _start_turn(app, backend, pilot)
        await app.action_toggle_debug_console()
        await pilot.pause()
        console = app._debug_console
        assert console is not None
        console.query_one("#debug-console-log").focus()
        await pilot.pause()
        app._request_intentional_exit("Ctrl+D")
        await pilot.pause()
        assert isinstance(app.screen, ExitConsequencesScreen)
        await pilot.press("ctrl+c", "ctrl+d")
        assert isinstance(app.screen, ExitConsequencesScreen)
        await pilot.press("escape")
        await pilot.pause()
        assert not isinstance(app.screen, ExitConsequencesScreen)
        assert app._debug_console is console
        assert app.app_server.turn_active
        await pilot.press("escape")
        assert await _wait_until(pilot, lambda: app._debug_console is None)
        assert app.app_server.turn_active
        await pilot.press("escape")
        assert await _wait_until(
            pilot, lambda: not app.app_server.turn_active, timeout=5
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (120, 36)])
async def test_console_pointer_copy_and_close_use_keyboard_actions(
    size, monkeypatch
) -> None:
    app, backend = _blocked_app()
    async with app.run_test(size=size) as pilot:
        await _start_turn(app, backend, pilot)
        await app.action_toggle_debug_console()
        await pilot.pause()
        console = app._debug_console
        assert console is not None
        view = console.query_one("#debug-console-log", _LogView)
        view.write_line("whole entry " * 20, scroll_end=False)
        view._select_line(len(view._lines) - 1)
        copied = []
        monkeypatch.setattr(app, "copy_to_clipboard", copied.append)
        console.action_copy()
        expected = copied.pop()
        footer = console.query_one("#debug-console-footer", Static)
        await pilot.pause()
        # Find the rendered action spans, including wrapping in the narrow dock.
        for action in ("copy", "close"):
            clicked = False
            for y in range(footer.size.height):
                strip = footer.render_line(y)
                x = 0
                for segment in strip._segments:
                    if segment.style and segment.style.meta.get("@click") == action:
                        await pilot.click(footer, offset=(x, y))
                        clicked = True
                        break
                    x += segment.cell_length
                if clicked:
                    break
            assert clicked, f"{action} action must be pointer reachable"
            if action == "copy":
                assert copied == [expected]
        assert await _wait_until(pilot, lambda: app._debug_console is None)
        assert app.app_server.turn_active
        backend.release.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (120, 36)])
@pytest.mark.parametrize(
    "command,surface_type",
    [
        ("/settings", SettingsScreen),
        ("/mcp", MCPApp),
        ("/providers", ProviderWorkbenchScreen),
        ("/usage", UsageScreen),
        ("/log-level", LogLevelPickerApp),
    ],
)
async def test_secondary_escape_unwinds_one_owner_before_interrupting_parent(
    size, command, surface_type, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def preview_candidate(self: FakeConfigOrchestrator):
        return SimpleNamespace(config=self.config)

    monkeypatch.setattr(FakeConfigOrchestrator, "preview_candidate", preview_candidate)
    app, backend = _blocked_app()
    async with app.run_test(size=size) as pilot:
        await app._session_ready.wait()
        await app.app_server.resources.runtime.wait_until_ready()
        monkeypatch.setattr(
            app.app_server.resources.mcp,
            "read",
            AsyncMock(return_value=_state(_source("gmail"), _source("slack"))),
        )
        monkeypatch.setattr(
            app.app_server.resources.usage, "read", AsyncMock(return_value=snapshot())
        )
        await app.action_toggle_debug_console()
        console = app._debug_console
        assert console is not None
        # Providers deliberately cannot be opened during work. Exercise the
        # overlapping state by starting a real server turn after opening it.
        if command != "/providers":
            await _start_turn(app, backend, pilot)
        assert await app._handle_command(command)

        def surface_open() -> bool:
            return isinstance(app.screen, surface_type) or bool(app.query(surface_type))

        assert await _wait_until(pilot, surface_open)
        surface = (
            app.screen
            if isinstance(app.screen, surface_type)
            else app.query_one(surface_type)
        )
        if command == "/providers":
            await _start_turn(app, backend, pilot)
        queue = app.app_server.turn_queue.model_copy(deep=True)

        def assert_work_preserved() -> None:
            assert app.app_server.turn_active
            assert app.app_server.turn_queue == queue
            assert app._interrupt_operation is None
            assert app._debug_console is console
            assert not backend.release.is_set()

        if isinstance(surface, SettingsScreen):
            await pilot.press(*"log_level")
            options = surface.query_one(SettingsOptionList)
            assert options._query == "log_level"
            await pilot.press("escape")
            assert options._query == ""
            assert app.screen is surface
            assert_work_preserved()
        elif isinstance(surface, MCPApp):
            await pilot.press("shift+tab", "g", "tab", "enter")
            assert surface._viewing_name == "gmail"
            await pilot.press("escape")
            assert surface._viewing_name is None and surface._query == "g"
            assert_work_preserved()
            await pilot.press("escape")
            assert surface._query == ""
            assert_work_preserved()
        elif isinstance(surface, ProviderWorkbenchScreen):
            await pilot.press("enter")
            assert surface.state is not None
            await pilot.press("escape")
            assert surface.state is None and app.screen is surface
            assert_work_preserved()
        elif isinstance(surface, LogLevelPickerApp):
            original = surface._session_level
            await pilot.press("enter", "space", "escape")
            assert surface._confirming_discard
            await pilot.press("escape")
            assert not surface._confirming_discard
            assert_work_preserved()
            # Remove the unpersisted draft to permit a clean root dismissal.
            surface._session_level = original

        # Resize while the secondary owner, rather than the dock, owns focus.
        other_size = (120, 36) if size == (80, 24) else (80, 24)
        await pilot.resize_terminal(*other_size)
        await pilot.pause()
        assert_work_preserved()
        # Return the fullscreen console to a dock while the secondary surface
        # still owns focus, before handing interaction back to the composer.
        await pilot.resize_terminal(120, 36)
        await pilot.pause()
        await pilot.press("escape")
        assert await _wait_until(pilot, lambda: not surface_open())
        assert_work_preserved()
        assert app._chat_input_container is not None
        app._chat_input_container.focus_input()
        await pilot.pause()
        await pilot.press("escape")
        assert await _wait_until(
            pilot, lambda: not app.app_server.turn_active, timeout=5
        )
        assert app._debug_console is console


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (120, 36)])
async def test_question_escape_cancels_only_local_question_before_parent(size) -> None:
    app, backend = _blocked_app()
    async with app.run_test(size=size) as pilot:
        await _start_turn(app, backend, pilot)
        request = _question_callback("navigation").detail.request
        question = asyncio.create_task(app._request_local_user_input(request))
        try:
            assert await _wait_until(pilot, lambda: bool(app.query(QuestionApp)))
            await pilot.press("escape")
            result = await asyncio.wait_for(question, 5)
            assert result.cancelled
            assert not app.query(QuestionApp)
            assert app.app_server.turn_active
            assert app._interrupt_operation is None
            await pilot.press("escape")
            assert await _wait_until(
                pilot, lambda: not app.app_server.turn_active, timeout=5
            )
        finally:
            if not question.done():
                question.cancel()
                await asyncio.gather(question, return_exceptions=True)
