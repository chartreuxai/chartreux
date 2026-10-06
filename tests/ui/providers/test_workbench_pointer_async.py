"""Pointer ownership and controlled asynchronous workbench acceptance."""

from __future__ import annotations

import asyncio
from copy import deepcopy
import threading
from unittest.mock import AsyncMock, Mock

import pytest
from textual import events
from textual.widgets import Input, OptionList, SelectionList
from textual.widgets.option_list import Option

from chartreux.core.model_catalog.contracts import (
    CatalogValidationError,
    DiscoveryItem,
    DiscoveryResult,
)
from chartreux.core.model_catalog.presets import FULLY_CUSTOM
from chartreux.ui.providers.workbench import WorkbenchView
from tests.ui.providers.test_workbench import Host, expand, press_option, setup


async def click_hint(pilot, screen, key: str) -> None:
    start = next(start for start, _end, target in screen._hint_targets if target == key)
    await pilot.click("#wb-hint", offset=(start, 0))


@pytest.mark.asyncio
async def test_protocol_pointer_selects_without_accepting_then_accepts_selected() -> (
    None
):
    screen, services = setup()
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await expand(pilot, screen)
        assert screen.state is not None
        await press_option(pilot, screen.query_one("#wb-actions", OptionList), "style")
        protocol = screen.query_one("#wb-protocol", OptionList)
        before = screen.state.connection
        await pilot.click("#wb-protocol", offset=(4, 1))
        assert screen._protocol_picker and screen._protocol_value == "openai-responses"
        assert screen.state.connection == before
        await pilot.press("down")
        assert protocol.highlighted_option is not None
        assert protocol.highlighted_option.id == "anthropic"
        await click_hint(pilot, screen, "Enter")
        assert not screen._protocol_picker
        assert screen.state.connection.api_style == "openai-responses"
        assert screen.query_one("#wb-actions").has_focus
        assert not services.writes


@pytest.mark.asyncio
@pytest.mark.parametrize("owner", ["busy", "confirmation", "help", "editor"])
@pytest.mark.parametrize("method", ["keyboard", "pointer", "message"])
async def test_underlay_activation_is_blocked_before_focus(owner, method) -> None:
    screen, services = setup()
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await expand(pilot, screen)
        assert screen.state is not None
        operations = screen.query_one("#wb-provider-operations", OptionList)
        operations.highlighted = next(
            i for i, row in enumerate(operations.options) if row.id == "models"
        )
        operations.focus()
        await pilot.pause()
        if owner == "busy":
            screen._busy = True
        elif owner == "confirmation":
            screen._confirm = "discard"
        elif owner == "help":
            screen.action_help()
        else:
            screen._select_action("base")
        screen._update_help()
        before = deepcopy(screen.state.changes())
        spy = Mock(wraps=screen._select_action)
        screen._select_action = spy
        if method == "keyboard":
            operations.action_select()
        elif method == "pointer":
            await pilot.click(
                "#wb-provider-operations", offset=(3, operations.highlighted)
            )
        else:
            assert operations.highlighted_option is not None
            assert operations.highlighted is not None
            screen.on_option_list_option_selected(
                OptionList.OptionSelected(
                    operations, operations.highlighted_option, operations.highlighted
                )
            )
        await pilot.pause()
        spy.assert_not_called()
        assert screen.state.changes() == before
        assert not services.writes
        screen._busy = False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("view", "group", "key", "command"),
    [
        (WorkbenchView.PROVIDERS, "wb-root-actions", "\x00models", "_open_catalog"),
        (WorkbenchView.ACTIONS, "wb-provider-operations", "models", "_select_action"),
        (
            WorkbenchView.CONNECTION,
            "wb-connection-actions",
            "continue",
            "_connection_action",
        ),
        (
            WorkbenchView.MODELS,
            "wb-models-actions",
            "continue-presets",
            "_advance_models",
        ),
        (WorkbenchView.CATALOG, "wb-catalog-actions", "apply", "_select_catalog"),
        (WorkbenchView.DETAIL, "wb-detail-actions", "save-detail", "_save_detail"),
        (
            WorkbenchView.PRESETS,
            "wb-presets-actions",
            "finish",
            "_select_preset_action",
        ),
        (
            WorkbenchView.PRESET_EDITOR,
            "wb-preset-editor-actions",
            "apply",
            "_select_preset_editor",
        ),
    ],
)
@pytest.mark.parametrize("method", ["keyboard", "pointer", "double-pointer", "footer"])
async def test_moved_action_targets_share_dispatch(
    view, group, key, command, method
) -> None:
    screen, services = setup()
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await expand(pilot, screen)
        assert screen.state is not None
        widget = screen.query_one(f"#{group}", OptionList)
        widget.clear_options()
        widget.add_option(Option("Acceptance action", id=key))
        screen._view = view
        # Dispatch fixture: keep the real mounted control and handlers, replacing
        # only the destination command so every action can be exercised uniformly.
        screen._update_help = Mock()
        widget.display = True
        widget.disabled = False
        widget.highlighted = 0
        widget.focus()
        spy = Mock()
        setattr(screen, command, spy)
        await pilot.pause()
        if method == "keyboard":
            await pilot.press("enter")
        elif method in {"pointer", "double-pointer"}:
            await pilot.click(
                f"#{group}", offset=(4, 0), times=2 if method == "double-pointer" else 1
            )
        else:
            # Retain the actual NoMarkupStatic footer and route its Enter target.
            await click_hint(pilot, screen, "Enter")
        await pilot.pause()
        spy.assert_called_once()
        assert not services.writes


@pytest.mark.asyncio
@pytest.mark.parametrize("owner", ["actions", "help", "editor", "confirmation"])
async def test_discovery_completion_preserves_owner_draft_and_bookmarks(owner) -> None:
    screen, services = setup()
    started, release = asyncio.Event(), asyncio.Event()

    async def discover(*_args):
        started.set()
        await release.wait()
        return DiscoveryResult((DiscoveryItem("new-wire"),))

    screen.discovery_service = AsyncMock(side_effect=discover)
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await expand(pilot, screen)
        assert screen.state is not None
        screen._stage = "models"
        screen._select_action("models")
        await pilot.press("tab")
        assert screen.query_one("#wb-models-actions").has_focus
        task = asyncio.create_task(screen._discover(screen.state))
        await started.wait()
        if owner == "help":
            screen.action_help()
            screen.query_one("#wb-help").focus()
        elif owner == "editor":
            screen._select_action("manual")
            screen.query_one("#wb-input", Input).value = "unsaved-wire"
        elif owner == "confirmation":
            screen._confirm = "discard"
            screen._update_help()
        await pilot.pause()
        expected_focus = {
            "actions": "wb-models-actions",
            "help": "wb-help",
            "editor": "wb-input",
            "confirmation": "wb-confirm-actions",
        }[owner]
        assert screen.focused is screen.query_one(f"#{expected_focus}")
        assert screen._help_open == (owner == "help")
        assert (screen._editing == "manual") == (owner == "editor")
        assert bool(screen._confirm) == (owner == "confirmation")
        focused = screen.focused
        before = deepcopy(screen.state.changes())
        positions = dict(screen._positions)
        release.set()
        await task
        await pilot.pause()
        assert screen.focused is focused
        assert screen.state.changes() == before
        assert all(
            screen._positions.get(key) == value
            for key, value in positions.items()
            if key[0] != WorkbenchView.PROVIDERS
        )
        assert not services.writes
        if owner == "editor":
            assert screen.query_one("#wb-input", Input).value == "unsaved-wire"


@pytest.mark.asyncio
async def test_deferred_focus_does_not_cross_transition_or_minimum_warning() -> None:
    screen, _services = setup()
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await expand(pilot, screen)
        assert screen.state is not None
        screen._defer_focus(screen.query_one("#wb-actions"))
        screen._open_presets()
        await pilot.pause()
        assert screen.query_one("#wb-presets").has_focus
        await pilot.resize_terminal(47, 24)
        screen._defer_focus(screen.query_one("#wb-presets"))
        await pilot.pause()
        assert not screen.query_one("#workbench").display
        assert (
            screen.focused is None
            or not screen.focused.has_focus
            or any(not ancestor.display for ancestor in screen.focused.ancestors)
        )
        await pilot.resize_terminal(80, 24)
        assert screen.view == WorkbenchView.PRESETS


@pytest.mark.asyncio
async def test_resize_family_preserves_radio_draft_and_return_route() -> None:
    screen, services = setup()
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await expand(pilot, screen)
        assert screen.state is not None
        await press_option(pilot, screen.query_one("#wb-actions", OptionList), "style")
        await pilot.press("down", "space", "down")
        navigation = list(screen._navigation)
        for size in [(120, 36), (48, 24), (47, 24), (48, 23), (80, 24)]:
            await pilot.resize_terminal(*size)
            await pilot.pause()
            assert screen._protocol_value == "openai-responses"
            assert screen._protocol_picker
            assert screen._navigation == navigation
            option = screen.query_one("#wb-protocol", OptionList).highlighted_option
            assert option is not None and option.id == "anthropic"
        await pilot.press("enter")
        assert screen.state.connection.api_style == "openai-responses"
        assert not services.writes


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "transition", ["provider", "stage", "generation", "close", "resize"]
)
async def test_controlled_discovery_transition_adoption(transition) -> None:
    screen, services = setup()
    started, release = asyncio.Event(), asyncio.Event()

    async def discover(*_args):
        started.set()
        await release.wait()
        return DiscoveryResult((DiscoveryItem("new-wire"),))

    screen.discovery_service = AsyncMock(side_effect=discover)
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await expand(pilot, screen)
        assert screen.state is not None
        state = screen.state
        task = asyncio.create_task(screen._discover(state))
        await started.wait()
        if transition == "provider":
            screen._expand("two")
        elif transition == "stage":
            screen._stage = "models"
        elif transition == "generation":
            state.begin_discovery()
        elif transition == "close":
            await screen.app.pop_screen()
        else:
            await pilot.resize_terminal(47, 24)
        payload = deepcopy(state.changes())
        focused = screen.focused
        positions = dict(screen._positions)
        release.set()
        await task
        await pilot.pause()
        assert state.changes() == payload
        assert screen.focused is focused
        if transition != "resize":
            assert screen._positions == positions
            assert "new-wire" not in {
                wire for _name, wire, _enabled, _found in state.model_rows()
            }
        else:
            assert "new-wire" in {
                wire for _name, wire, _enabled, _found in state.model_rows()
            }
            assert not screen.query_one("#workbench").display
        assert not services.writes


@pytest.mark.asyncio
@pytest.mark.parametrize("result", ["saved", "validation", "revision", "exception"])
async def test_delayed_pointer_save_repetition_retains_draft_and_opener(result) -> None:
    screen, services = setup()
    entered, release = threading.Event(), threading.Event()
    original = services.apply_changes
    calls = []

    def delayed(changes):
        calls.append(changes)
        entered.set()
        assert release.wait(timeout=5)
        if result in {"validation", "revision"}:
            return CatalogValidationError(
                "Revision conflict" if result == "revision" else "Rejected"
            )
        if result == "exception":
            raise RuntimeError("write failed")
        return original(changes)

    services.apply_changes = delayed  # type: ignore[method-assign]
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await expand(pilot, screen)
        assert screen.state is not None
        await press_option(pilot, screen.query_one("#wb-actions", OptionList), "base")
        screen.query_one("#wb-input", Input).value = "https://changed.test"
        await pilot.press("enter")
        operations = screen.query_one("#wb-provider-operations", OptionList)
        operations.focus()
        operations.highlighted = next(
            i for i, option in enumerate(operations.options) if option.id == "apply"
        )
        await pilot.pause()
        payload = deepcopy(screen.state.changes())
        # Repeated requests in one turn, before the save coroutine is scheduled.
        operations.action_select()
        operations.action_select()
        await pilot.pause()
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            assert screen._busy and len(calls) == 1 and not services.writes
            await pilot.click(
                "#wb-provider-operations", offset=(4, operations.highlighted)
            )
            await pilot.press("enter", "escape", "tab")
            assert (
                screen.state.changes() == payload and screen.state.provider_id == "one"
            )
            assert not screen._dismissed and screen.view == WorkbenchView.ACTIONS
        finally:
            release.set()
        assert screen._commit_task is not None
        await screen._commit_task
        await pilot.pause()
        assert len(calls) == 1 and not screen._busy
        assert len(services.writes) == (1 if result == "saved" else 0)
        if result != "saved":
            assert screen.state.changes() == payload
            assert operations.has_focus
            assert operations.highlighted_option is not None
            assert operations.highlighted_option.id == "apply"


@pytest.mark.asyncio
async def test_pointer_help_back_and_editor_acceptance_are_local() -> None:
    screen, services = setup()
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await expand(pilot, screen)
        await press_option(pilot, screen.query_one("#wb-actions", OptionList), "base")
        editor = screen.query_one("#wb-input", Input)
        editor.value = "https://pointer.test"
        await pilot.click("#wb-help", offset=(2, 0))
        assert screen._help_open and screen._editing == "base"
        await click_hint(pilot, screen, "Esc")
        assert not screen._help_open and editor.has_focus
        await click_hint(pilot, screen, "Enter")
        assert screen._editing is None and screen.state is not None
        assert screen.state.connection.api_base == "https://pointer.test"
        assert not services.writes


@pytest.mark.asyncio
async def test_pointer_wheel_remains_local_without_activation() -> None:
    screen, services = setup()
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        providers = screen.query_one("#wb-providers", OptionList)
        providers.clear_options()
        providers.add_options(
            Option(f"Provider {i}", id=f"provider-{i}") for i in range(100)
        )
        providers.highlighted = 0
        providers.focus()
        await pilot.pause()
        actions = screen.query_one("#wb-root-actions", OptionList)
        action_cursor = actions.highlighted
        await pilot._post_mouse_events(
            [events.MouseScrollDown], providers, offset=(4, 2), times=5
        )
        await pilot.pause()
        assert (
            providers.scroll_y > 0
            and providers.highlighted == 0
            and providers.has_focus
        )
        assert actions.scroll_y == 0 and actions.highlighted == action_cursor
        assert screen.view == WorkbenchView.PROVIDERS
        assert not services.writes and not services.keys


@pytest.mark.asyncio
@pytest.mark.parametrize("owner", ["models", "help", "invalid-editor"])
async def test_resize_family_retains_membership_draft_disclosure_error_and_opener(
    owner,
) -> None:
    screen, services = setup()
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await expand(pilot, screen)
        assert screen.state is not None
        await press_option(pilot, screen.query_one("#wb-actions", OptionList), "base")
        screen.query_one("#wb-input", Input).value = "https://draft.test"
        await pilot.press("enter")
        screen._select_action("models")
        models = screen.query_one("#wb-models", SelectionList)
        if owner == "help":
            screen.action_help()
            screen.query_one("#wb-help").focus()
        elif owner == "invalid-editor":
            screen._select_action("key")
            screen.query_one("#wb-input", Input).value = ""
            await pilot.press("enter")
            assert screen._editing == "key"
        await pilot.pause()
        focused = screen.focused
        assert focused is not None
        identity = models.highlighted
        membership = models.selected
        payload = deepcopy(screen.state.changes())
        route = list(screen._navigation)
        help_open, editing = screen._help_open, screen._editing
        editor = screen.query_one("#wb-input", Input)
        text, error = editor.value, str(screen.query_one("#wb-field-error").render())
        for size in [(120, 36), (48, 24), (47, 24), (48, 23), (80, 24)]:
            await pilot.resize_terminal(*size)
            await pilot.pause()
            assert screen.state.changes() == payload
            assert models.selected == membership and models.highlighted == identity
            assert screen._navigation == route
            assert screen._help_open == help_open and screen._editing == editing
            assert editor.value == text
            assert str(screen.query_one("#wb-field-error").render()) == error
        assert screen.focused is focused
        assert not services.writes


@pytest.mark.asyncio
@pytest.mark.parametrize("next_key", ["f1", "escape"])
async def test_queued_model_toggle_precedes_owner_transition(next_key) -> None:
    screen, services = setup()
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await expand(pilot, screen)
        assert screen.state is not None
        screen._select_action("models")
        await pilot.pause()
        models = screen.query_one("#wb-models", SelectionList)
        models.highlighted = 0
        state = screen.state
        before = deepcopy(state.changes())
        # Send both driver keys without Pilot.press's inter-key idle barrier.
        driver = pilot.app._driver
        assert driver is not None
        driver.send_message(events.Key("space", " "))
        driver.send_message(events.Key(next_key, None))
        await pilot.pause()
        assert screen._help_open == (next_key == "f1")
        assert not next(
            enabled for name, _, enabled, _ in state.model_rows() if name == "a"
        )
        assert state.dirty and state.changes() != before
        screen._refresh_models()
        await pilot.pause()
        assert "a" not in models.selected
        assert not services.writes


@pytest.mark.asyncio
@pytest.mark.parametrize("retained", [False, True])
async def test_connection_resize_rebuilds_draft_not_management_fields(retained) -> None:
    screen, services = setup()
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        if retained:
            await expand(pilot, screen)
        screen._choose_preset(FULLY_CUSTOM)
        await pilot.pause()
        screen._connection_action("name")
        screen.query_one("#wb-input", Input).value = "resize-draft"
        await pilot.press("enter")
        screen._connection_action("base")
        screen.query_one("#wb-input", Input).value = "https://resize.test/v1"
        await pilot.press("enter")
        draft = deepcopy(screen._add)
        fields = screen.query_one("#wb-actions", OptionList)
        fields.highlighted = fields.get_option_index("env")
        for size in [(120, 36), (47, 24), (80, 24)]:
            await pilot.resize_terminal(*size)
            await pilot.pause()
            assert screen._add == draft
            assert [row.id for row in fields.options] == [
                "name",
                "base",
                "style",
                "env",
                "key",
            ]
            assert "resize-draft" in str(fields.get_option("name").prompt)
            assert "resize.test" in str(fields.get_option("base").prompt)
            assert fields.highlighted_option and fields.highlighted_option.id == "env"
        assert not services.writes


@pytest.mark.asyncio
async def test_confirmation_pointer_focus_and_underlay_wheel_are_confined() -> None:
    screen, services = setup()
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await expand(pilot, screen)
        fields = screen.query_one("#wb-actions", OptionList)
        fields.add_options(
            Option(f"Long underlay {i}", id=f"extra-{i}") for i in range(40)
        )
        fields.highlighted = 0
        screen._confirm = "discard"
        screen._update_help()
        await pilot.pause()
        confirm = screen.query_one("#wb-confirm-actions", OptionList)
        assert confirm.has_focus
        await pilot.click("#wb-actions", offset=(2, 0))
        assert confirm.has_focus
        await pilot.press("down")
        assert fields.highlighted == 0
        before = fields.scroll_y
        fields.post_message(
            events.MouseScrollDown(fields, 2, 0, 0, 0, 0, False, False, False)
        )
        await pilot.pause()
        assert fields.scroll_y == before
        assert confirm.has_focus
        assert not services.writes


@pytest.mark.asyncio
async def test_scheduled_discovery_binds_requester_before_worker_start() -> None:
    screen, services = setup()
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await expand(pilot, screen)
        assert screen.state is not None
        state = screen.state
        scheduled = []
        screen.run_worker = Mock(
            side_effect=lambda request, **_: scheduled.append(request)
        )
        screen.discovery_service = AsyncMock(
            return_value=DiscoveryResult((DiscoveryItem("new-wire"),))
        )
        screen._select_action("discover")
        screen._expand("two")
        await scheduled.pop()()
        assert screen.discovery_service.call_args.args[0].provider_id == "one"
        assert "new-wire" not in {wire for _, wire, _, _ in state.model_rows()}
        await pilot.pause()
        assert screen._feedback_kind != "running"
        assert "Discovering" not in str(screen._feedback())
        assert "Discovering" not in str(screen.query_one("#wb-help").render())
        screen._expand("one")
        screen._select_action("discover")
        await scheduled.pop()()
        assert "new-wire" in {wire for _, wire, _, _ in state.model_rows()}
        assert not services.writes


@pytest.mark.asyncio
async def test_discovery_stage_back_drops_result_and_running_feedback() -> None:
    screen, _services = setup()
    started, release = asyncio.Event(), asyncio.Event()

    async def discover(*_args):
        started.set()
        await release.wait()
        return DiscoveryResult((DiscoveryItem("new-wire"),))

    screen.discovery_service = AsyncMock(side_effect=discover)
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await expand(pilot, screen)
        assert screen.state is not None
        state = screen.state
        screen._stage = "models"
        screen._select_action("models")
        task = asyncio.create_task(screen._discover(state))
        await started.wait()
        await pilot.press("escape", "escape")
        assert screen._stage == "connection"
        release.set()
        await task
        await pilot.pause()
        assert "new-wire" not in {wire for _, wire, _, _ in state.model_rows()}
        assert screen._feedback_kind != "running"
        assert "Discovering" not in str(screen._feedback())


@pytest.mark.asyncio
async def test_pointer_plain_focus_updates_help_context() -> None:
    screen, _services = setup()
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await pilot.press("tab")
        assert screen.query_one("#wb-root-actions").has_focus
        await pilot.click("#wb-providers", offset=(3, 4))
        await pilot.pause()
        assert screen.query_one("#wb-providers").has_focus
        assert screen._help_focus_id == "wb-providers"


@pytest.mark.asyncio
async def test_field_error_sibling_has_error_color_only_while_invalid() -> None:
    from textual.color import Color

    screen, _services = setup()
    async with Host(screen).run_test(size=(80, 24)) as pilot:
        await expand(pilot, screen)
        screen._select_action("key")
        await pilot.pause()
        await pilot.press("enter")
        error = screen.query_one("#wb-field-error")
        assert error.has_class("has-error")
        assert error.styles.height is not None
        assert error.styles.color == Color.parse(
            screen.app.get_css_variables()["error"]
        )
        screen.query_one("#wb-input", Input).value = "new-key"
        await pilot.pause()
        assert not error.has_class("has-error")
        assert str(error.render()) == ""
