from __future__ import annotations

from unittest.mock import AsyncMock, Mock
from weakref import WeakKeyDictionary

import pytest

from chartreux.cli.textual_ui.handlers.event_handler import EventHandler
from chartreux.cli.textual_ui.widgets.status_message import IndicatorState
from chartreux.cli.textual_ui.widgets.tools import ToolGroup, ToolResultMessage
from chartreux.cli.textual_ui.windowing.history import build_history_widgets
from chartreux.core.events import ToolCallEvent, ToolResultEvent
from tests.stubs.app_server import CoreEventProjection
from tests.stubs.fake_tool import FakeTool, FakeToolArgs


def _call_event(call_id: str) -> ToolCallEvent:
    return ToolCallEvent(
        tool_name="stub_tool",
        tool_class=FakeTool,
        args=FakeToolArgs(),
        tool_call_id=call_id,
    )


def _error_result(call_id: str) -> ToolResultEvent:
    return ToolResultEvent(
        tool_name="stub_tool",
        tool_class=FakeTool,
        result=None,
        error="boom",
        tool_call_id=call_id,
    )


def _ok_result(call_id: str) -> ToolResultEvent:
    return ToolResultEvent(
        tool_name="stub_tool", tool_class=FakeTool, result=None, tool_call_id=call_id
    )


def _make_handler() -> tuple[EventHandler, AsyncMock, CoreEventProjection]:
    mount_callback = AsyncMock()
    handler = EventHandler(
        mount_callback=mount_callback, get_tools_collapsed=lambda: False
    )
    return handler, mount_callback, CoreEventProjection()


def _last_result_widget(mount_callback: AsyncMock) -> ToolResultMessage:
    for call in reversed(mount_callback.call_args_list):
        widget = call.args[0]
        if isinstance(widget, ToolResultMessage):
            return widget
    raise AssertionError("no ToolResultMessage was mounted")


@pytest.mark.asyncio
async def test_error_result_is_registered_as_pending() -> None:
    handler, mount_callback, projection = _make_handler()

    await projection.dispatch(_call_event("a"), handler.handle_event)
    await projection.dispatch(_error_result("a"), handler.handle_event)

    assert len(handler._pending_error_results) == 1


@pytest.mark.asyncio
async def test_later_success_keeps_earlier_error_muted() -> None:
    handler, mount_callback, projection = _make_handler()

    await projection.dispatch(_call_event("a"), handler.handle_event)
    await projection.dispatch(_error_result("a"), handler.handle_event)
    result_widget = _last_result_widget(mount_callback)
    result_widget.escalate_error = Mock()

    await projection.dispatch(_call_event("b"), handler.handle_event)
    await projection.dispatch(_ok_result("b"), handler.handle_event)

    result_widget.escalate_error.assert_not_called()
    assert handler._pending_error_results == []


@pytest.mark.asyncio
async def test_group_summary_and_individual_error_muting_are_separate() -> None:
    handler, mount_callback, projection = _make_handler()

    await projection.dispatch(_call_event("a"), handler.handle_event)
    await projection.dispatch(_error_result("a"), handler.handle_event)
    failed_result = _last_result_widget(mount_callback)
    await projection.dispatch(_call_event("b"), handler.handle_event)
    await projection.dispatch(_ok_result("b"), handler.handle_event)

    group = handler.current_tool_group
    assert group is not None
    assert group.header._last_state is IndicatorState.SUCCESS
    assert failed_result._should_escalate is False


def test_restored_failed_then_successful_call_mutes_the_failed_indicator() -> None:
    projection = CoreEventProjection()
    projection.project(_call_event("a"))
    projection.project(_error_result("a"))
    projection.project(_call_event("b"))
    projection.project(_ok_result("b"))

    group = build_history_widgets(
        projection.history,
        start_index=0,
        history_widget_indices=WeakKeyDictionary(),
        tools_collapsed=True,
    )[0]

    assert isinstance(group, ToolGroup)
    failed_result = group.content_container.children[1]
    assert isinstance(failed_result, ToolResultMessage)
    assert failed_result._should_escalate is False
    assert group.header._last_state is IndicatorState.SUCCESS


def test_restored_terminal_failure_escalates() -> None:
    projection = CoreEventProjection()
    projection.project(_call_event("a"))
    projection.project(_error_result("a"))

    group = build_history_widgets(
        projection.history,
        start_index=0,
        history_widget_indices=WeakKeyDictionary(),
        tools_collapsed=True,
    )[0]

    assert isinstance(group, ToolGroup)
    failed_result = group.content_container.children[1]
    assert isinstance(failed_result, ToolResultMessage)
    assert failed_result._should_escalate is True


@pytest.mark.asyncio
async def test_turn_end_escalates_error() -> None:
    handler, mount_callback, projection = _make_handler()

    await projection.dispatch(_call_event("a"), handler.handle_event)
    await projection.dispatch(_error_result("a"), handler.handle_event)
    result_widget = _last_result_widget(mount_callback)
    result_widget.escalate_error = Mock()

    handler.escalate_unresolved_errors()

    result_widget.escalate_error.assert_called_once()
    assert handler._pending_error_results == []


@pytest.mark.asyncio
async def test_successful_result_is_not_pending() -> None:
    handler, mount_callback, projection = _make_handler()

    await projection.dispatch(_call_event("a"), handler.handle_event)
    await projection.dispatch(_ok_result("a"), handler.handle_event)

    assert handler._pending_error_results == []


@pytest.mark.asyncio
async def test_streaming_arg_update_before_result_does_not_register_error() -> None:
    handler, _, projection = _make_handler()

    # Same tool_call_id re-emitted as a streaming arg update, before any result.
    await projection.dispatch(_call_event("a"), handler.handle_event)
    await projection.dispatch(_call_event("a"), handler.handle_event)

    assert handler._pending_error_results == []


@pytest.mark.asyncio
async def test_parallel_errors_escalated_together_at_turn_end() -> None:
    handler, mount_callback, projection = _make_handler()

    await projection.dispatch(_call_event("a"), handler.handle_event)
    await projection.dispatch(_call_event("b"), handler.handle_event)
    await projection.dispatch(_error_result("a"), handler.handle_event)
    await projection.dispatch(_error_result("b"), handler.handle_event)
    mocks: list[Mock] = []
    for widget in handler._pending_error_results:
        mock = Mock()
        widget.escalate_error = mock
        mocks.append(mock)
    assert len(mocks) == 2

    handler.escalate_unresolved_errors()

    for mock in mocks:
        mock.assert_called_once()
    assert handler._pending_error_results == []


@pytest.mark.asyncio
async def test_cancel_holds_muted_square_for_in_flight_call() -> None:
    handler, _, projection = _make_handler()

    await projection.dispatch(_call_event("a"), handler.handle_event)
    tool_call = handler.tool_calls["a"]
    show_muted = Mock()
    stop_spinning = Mock()
    tool_call.show_muted = show_muted
    tool_call.stop_spinning = stop_spinning

    handler.stop_current_tool_call(cancelled=True)

    show_muted.assert_called_once()
    stop_spinning.assert_not_called()


@pytest.mark.asyncio
async def test_turn_error_shows_red_cross_for_in_flight_call() -> None:
    handler, _, projection = _make_handler()

    await projection.dispatch(_call_event("a"), handler.handle_event)
    tool_call = handler.tool_calls["a"]
    show_muted = Mock()
    stop_spinning = Mock()
    tool_call.show_muted = show_muted
    tool_call.stop_spinning = stop_spinning

    handler.stop_current_tool_call(success=False)

    stop_spinning.assert_called_once()
    show_muted.assert_not_called()


@pytest.mark.asyncio
async def test_cancel_does_not_escalate_pending_errors() -> None:
    handler, mount_callback, projection = _make_handler()

    await projection.dispatch(_call_event("a"), handler.handle_event)
    await projection.dispatch(_error_result("a"), handler.handle_event)
    result_widget = _last_result_widget(mount_callback)
    result_widget.escalate_error = Mock()

    handler.stop_current_tool_call(cancelled=True)

    result_widget.escalate_error.assert_not_called()
    assert handler._pending_error_results == []
