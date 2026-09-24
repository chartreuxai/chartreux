from __future__ import annotations

from unittest.mock import AsyncMock
from weakref import WeakKeyDictionary

import pytest
from textual.widget import Widget

from chartreux.app_server._shell import shell_effect_detail
from chartreux.app_server.models import (
    PublicEffectEntry,
    PublicEntryGenerationStatus,
    RunningEffectState,
)
from chartreux.cli.textual_ui.handlers.event_handler import EventHandler
from chartreux.cli.textual_ui.widgets.messages import ReasoningMessage
from chartreux.cli.textual_ui.widgets.status_message import IndicatorState
from chartreux.cli.textual_ui.widgets.tools import (
    ToolCallMessage,
    ToolGroup,
    ToolResultMessage,
)
from chartreux.cli.textual_ui.windowing.history import build_history_widgets
from chartreux.core.events import (
    AssistantEvent,
    ReasoningEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from chartreux.core.tools.builtins.edit import Edit, EditArgs
from chartreux.core.tools.builtins.write_file import WriteFile, WriteFileArgs
from tests.conftest import build_test_chartreux_app, build_test_vibe_config
from tests.stubs.app_server import CoreEventProjection
from tests.stubs.fake_tool import FakeTool, FakeToolArgs


def _call_event(call_id: str, tool_name: str = "stub_tool") -> ToolCallEvent:
    if tool_name == "edit":
        return ToolCallEvent(
            tool_name=tool_name,
            tool_class=Edit,
            args=EditArgs(file_path="app.py", old_string="old", new_string="new"),
            tool_call_id=call_id,
        )
    if tool_name == "write_file":
        return ToolCallEvent(
            tool_name=tool_name,
            tool_class=WriteFile,
            args=WriteFileArgs(file_path="app.py", content="content"),
            tool_call_id=call_id,
        )
    return ToolCallEvent(
        tool_name=tool_name,
        tool_class=FakeTool,
        args=FakeToolArgs(),
        tool_call_id=call_id,
    )


def _make_handler(
    *, show_thinking: bool = True
) -> tuple[EventHandler, AsyncMock, CoreEventProjection]:
    mount_callback = AsyncMock()
    handler = EventHandler(
        mount_callback=mount_callback,
        get_tools_collapsed=lambda: False,
        get_show_thinking=lambda: show_thinking,
    )
    return handler, mount_callback, CoreEventProjection()


def _mounted_groups(mount_callback: AsyncMock) -> list[ToolGroup]:
    return [
        call.args[0]
        for call in mount_callback.call_args_list
        if call.args and isinstance(call.args[0], ToolGroup)
    ]


def _mount_call_for(mount_callback: AsyncMock, widget: object):
    for call in mount_callback.call_args_list:
        if call.args and call.args[0] is widget:
            return call
    raise AssertionError("widget was never mounted")


@pytest.mark.asyncio
async def test_consecutive_tool_calls_share_one_group() -> None:
    handler, mount_callback, projection = _make_handler()

    await projection.dispatch(_call_event("a"), handler.handle_event)
    await projection.dispatch(_call_event("b"), handler.handle_event)

    groups = _mounted_groups(mount_callback)
    assert len(groups) == 1
    assert handler.current_tool_group is groups[0]


@pytest.mark.asyncio
async def test_reasoning_and_following_tool_call_share_group() -> None:
    handler, mount_callback, projection = _make_handler()

    await projection.dispatch(ReasoningEvent(content="thinking"), handler.handle_event)
    await projection.dispatch(_call_event("a"), handler.handle_event)

    assert len(_mounted_groups(mount_callback)) == 1


@pytest.mark.asyncio
async def test_assistant_text_breaks_group_into_two() -> None:
    handler, mount_callback, projection = _make_handler()

    await projection.dispatch(_call_event("a"), handler.handle_event)
    await projection.dispatch(AssistantEvent(content="done"), handler.handle_event)

    assert handler.current_tool_group is None

    await projection.dispatch(_call_event("b"), handler.handle_event)

    assert len(_mounted_groups(mount_callback)) == 2


@pytest.mark.asyncio
async def test_edit_mounts_standalone_and_breaks_open_group() -> None:
    handler, mount_callback, projection = _make_handler()

    await projection.dispatch(_call_event("a"), handler.handle_event)
    await projection.dispatch(_call_event("e", tool_name="edit"), handler.handle_event)

    assert handler.current_tool_group is None

    edit_widget = handler.tool_calls["e"]
    edit_call = _mount_call_for(mount_callback, edit_widget)
    assert "container" not in edit_call.kwargs
    assert "after" not in edit_call.kwargs


@pytest.mark.asyncio
async def test_edit_after_edit_does_not_open_a_group() -> None:
    handler, mount_callback, projection = _make_handler()

    await projection.dispatch(_call_event("e1", tool_name="edit"), handler.handle_event)
    await projection.dispatch(
        _call_event("e2", tool_name="write_file"), handler.handle_event
    )

    assert _mounted_groups(mount_callback) == []
    assert handler.current_tool_group is None


@pytest.mark.asyncio
@pytest.mark.parametrize("standalone_tool", ["edit", "write_file"])
async def test_edit_and_write_break_groups_in_live_and_restored_history(
    standalone_tool: str,
) -> None:
    app = build_test_chartreux_app(config=build_test_vibe_config())
    projection = CoreEventProjection()

    async with app.run_test() as pilot:
        handler = app.event_handler
        assert handler is not None
        await projection.dispatch(_call_event("before"), handler.handle_event)
        await projection.dispatch(
            _call_event("standalone", tool_name=standalone_tool), handler.handle_event
        )
        await projection.dispatch(_call_event("after"), handler.handle_event)
        await pilot.pause()

        live_widgets = list(app._messages_area.children)
        restored_widgets = build_history_widgets(
            projection.history,
            start_index=0,
            history_widget_indices=WeakKeyDictionary(),
            tools_collapsed=True,
        )

        expected = [ToolGroup, ToolCallMessage, ToolGroup]
        assert [type(widget) for widget in live_widgets] == expected
        assert [
            type(widget)
            for widget in restored_widgets
            if not isinstance(widget, ToolResultMessage)
        ] == expected


def _manual_shell_entry(entry_id: str) -> PublicEffectEntry:
    return PublicEffectEntry(
        id=entry_id,
        session_id="session-1",
        created_at=1,
        updated_at=1,
        generation_status=PublicEntryGenerationStatus.IN_PROGRESS,
        title="shell",
        detail=shell_effect_detail("pwd"),
        state=RunningEffectState(),
    )


@pytest.mark.asyncio
async def test_manual_shell_breaks_groups_and_mounts_standalone() -> None:
    handler, mount_callback, projection = _make_handler()

    await projection.dispatch(_call_event("before"), handler.handle_event)
    await handler._handle_entry_added(_manual_shell_entry("manual"), None)
    await projection.dispatch(_call_event("after"), handler.handle_event)

    groups = _mounted_groups(mount_callback)
    manual = handler.tool_calls["manual"]
    manual_mount = _mount_call_for(mount_callback, manual)
    assert len(groups) == 2
    assert manual_mount.kwargs.get("container") is None
    assert manual_mount.kwargs.get("after") is None

    handler, _, projection = _make_handler()

    await projection.dispatch(_call_event("a"), handler.handle_event)
    assert handler.current_tool_group is not None

    handler.stop_current_tool_call()

    assert handler.current_tool_group is None


@pytest.mark.asyncio
async def test_hidden_thinking_node_is_not_displayed() -> None:
    handler, _, projection = _make_handler(show_thinking=False)

    await projection.dispatch(ReasoningEvent(content="secret"), handler.handle_event)

    msg = handler.current_streaming_reasoning
    assert msg is not None
    assert msg.display is False


@pytest.mark.asyncio
async def test_live_thinking_toggle_collapses_only_empty_groups() -> None:
    config = build_test_vibe_config(show_thinking_nodes=True)
    app = build_test_chartreux_app(config=config)

    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        handler = app.event_handler
        assert handler is not None
        projection = CoreEventProjection()

        await projection.dispatch(
            ReasoningEvent(content="reasoning only"), handler.handle_event
        )
        await projection.dispatch(
            AssistantEvent(content="separator"), handler.handle_event
        )
        await projection.dispatch(
            ReasoningEvent(content="reasoning with tool"), handler.handle_event
        )
        await projection.dispatch(_call_event("a"), handler.handle_event)

        groups = list(app._messages_area.query(ToolGroup))
        reasoning_nodes = list(app._messages_area.query(ReasoningMessage))
        assert len(groups) == 2
        assert len(reasoning_nodes) == 2

        app.config.show_thinking_nodes = False
        app._apply_thinking_visibility()

        assert all(node.display is False for node in reasoning_nodes)
        assert groups[0].display is False
        assert groups[1].display is True
        assert handler.tool_calls["a"].display is True


@pytest.mark.asyncio
async def test_grouped_tool_call_mounts_into_the_group() -> None:
    handler, mount_callback, projection = _make_handler()

    await projection.dispatch(_call_event("a"), handler.handle_event)

    group = handler.current_tool_group
    assert group is not None
    call_widget = handler.tool_calls["a"]
    mount_call = _mount_call_for(mount_callback, call_widget)
    # First child of an empty group is mounted through the group's body widget.
    assert mount_call.kwargs.get("container") is group.content_container
    assert not isinstance(call_widget, ToolGroup)
    assert isinstance(call_widget, ToolCallMessage)


def _result_event(call_id: str, *, error: str | None = None) -> ToolResultEvent:
    return ToolResultEvent(
        tool_name="stub_tool",
        tool_class=FakeTool,
        result=None,
        error=error,
        tool_call_id=call_id,
    )


@pytest.mark.asyncio
async def test_out_of_order_completion_uses_latest_timeline_call_for_group_status() -> (
    None
):
    handler, _, projection = _make_handler()

    await projection.dispatch(_call_event("a"), handler.handle_event)
    await projection.dispatch(_call_event("b"), handler.handle_event)
    group = handler.current_tool_group
    assert group is not None

    await projection.dispatch(_result_event("b"), handler.handle_event)
    await projection.dispatch(_result_event("a", error="failed"), handler.handle_event)

    # The second call remains the latest timeline entry even though it completed
    # before the first, so its success controls the folded summary.
    assert group.header._last_state is IndicatorState.SUCCESS


@pytest.mark.asyncio
async def test_ctrl_o_toggles_groups_and_honors_local_expansion() -> None:
    app = build_test_chartreux_app(config=build_test_vibe_config())
    first = ToolGroup()
    second = ToolGroup()

    async with app.run_test() as pilot:
        await app._messages_area.mount(first, second)
        await pilot.pause()
        assert first.is_collapsed and second.is_collapsed

        await app.action_toggle_tool()
        assert not first.is_collapsed and not second.is_collapsed
        assert app._tools_collapsed is False

        # Locally expanding every group leaves Ctrl+O in the collapse direction.
        first.set_collapsed(False)
        second.set_collapsed(False)
        await app.action_toggle_tool()
        assert first.is_collapsed and second.is_collapsed
        assert app._tools_collapsed is True


@pytest.mark.asyncio
async def test_live_and_restored_tool_groups_have_matching_structure() -> None:
    app = build_test_chartreux_app(config=build_test_vibe_config())
    projection = CoreEventProjection()

    async with app.run_test() as pilot:
        handler = app.event_handler
        assert handler is not None
        await projection.dispatch(_call_event("a"), handler.handle_event)
        await projection.dispatch(_call_event("b"), handler.handle_event)
        await projection.dispatch(_result_event("b"), handler.handle_event)
        await projection.dispatch(
            _result_event("a", error="failed"), handler.handle_event
        )
        await pilot.pause()

        live_group = app._messages_area.query_one(ToolGroup)
        assert live_group.header._spinner_timer is None
        handler.stop_current_tool_call()
        restored_group = build_history_widgets(
            projection.history,
            start_index=0,
            history_widget_indices=WeakKeyDictionary(),
            tools_collapsed=True,
        )[0]

        assert isinstance(restored_group, ToolGroup)
        assert live_group.header.get_content() == restored_group.header.get_content()
        assert live_group.header._last_state is restored_group.header._last_state
        assert [type(child) for child in live_group.content_container.children] == [
            type(child) for child in restored_group.content_container.children
        ]


@pytest.mark.asyncio
async def test_restored_group_supports_local_expansion_and_ctrl_o() -> None:
    app = build_test_chartreux_app(config=build_test_vibe_config())
    projection = CoreEventProjection()
    projection.project(_call_event("a"))
    projection.project(_result_event("a"))
    restored_group = build_history_widgets(
        projection.history,
        start_index=0,
        history_widget_indices=WeakKeyDictionary(),
        tools_collapsed=True,
        expansion_state=app._tool_group_expansion_state,
    )[0]

    assert isinstance(restored_group, ToolGroup)
    async with app.run_test() as pilot:
        await app._messages_area.mount(restored_group)
        await pilot.pause()
        await pilot.click(".tool-group-header")
        await pilot.pause()
        assert not restored_group.is_collapsed

        await app.action_toggle_tool()
        assert restored_group.is_collapsed
        assert app._tools_collapsed


@pytest.mark.asyncio
async def test_tool_call_anchors_live_until_turn_completion() -> None:
    app = build_test_chartreux_app(config=build_test_vibe_config())

    async with app.run_test():
        handler = app.event_handler
        assert handler is not None
        handler._tool_call_anchors["call"] = Widget()

        # Stream finalization also occurs between tool activity and must not
        # release anchors needed by later tool-hook events in the same turn.
        await handler.finalize_streaming()
        assert "call" in handler._tool_call_anchors

        handler.offer_retry()
        assert handler.begin_retry()
        await app._finalize_turn_ui(notify_complete=False)
        assert "call" in handler._tool_call_anchors

        await app._finalize_turn_ui()
        assert not handler._tool_call_anchors
