from __future__ import annotations

from typing import cast

from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.pilot import Pilot
from textual.widget import Widget
from textual.worker import WorkerState

from chartreux.cli.textual_ui.handlers.event_handler import EventHandler
from chartreux.cli.textual_ui.widgets.tool_widgets import EditResultWidget
from chartreux.core.events import ToolCallEvent, ToolResultEvent
from chartreux.core.tools.builtins.edit import Edit, EditArgs, EditResult
from tests.snapshots.snap_compare import SnapCompare
from tests.stubs.app_server import CoreEventProjection


class _SnapshotApp(App):
    CSS_PATH = "../../chartreux/cli/textual_ui/app.tcss"

    def __init__(self) -> None:
        super().__init__()
        self._scroll: VerticalScroll | None = None
        self._handler: EventHandler | None = None

    def compose(self) -> ComposeResult:
        self._scroll = VerticalScroll(id="messages")
        yield self._scroll

    def on_mount(self) -> None:
        async def mount_callback(
            widget: Widget,
            *,
            after: Widget | None = None,
            before: Widget | None = None,
            container: Widget | None = None,
        ) -> None:
            if self._scroll is None:
                return
            if after is not None and after.parent is not None:
                await cast(Widget, after.parent).mount(widget, after=after)
            elif container is not None:
                await container.mount(widget)
            else:
                await self._scroll.mount(widget)

        self._handler = EventHandler(
            mount_callback=mount_callback, get_tools_collapsed=lambda: False
        )

    async def populate(self) -> None:
        raise NotImplementedError


class EditResultApp(_SnapshotApp):
    async def populate(self) -> None:
        if self._scroll is None or self._handler is None:
            return

        projection = CoreEventProjection()
        call_id = "edit_success"
        old_string = '    return f"hello {name}"'
        new_string = '    return f"Hello, {name}!"'
        await projection.dispatch(
            ToolCallEvent(
                tool_call_id=call_id,
                tool_name="edit",
                tool_class=Edit,
                args=EditArgs(
                    file_path="/repo/app.py",
                    old_string=old_string,
                    new_string=new_string,
                ),
            ),
            self._handler.handle_event,
        )
        result = EditResult(
            file="/repo/app.py",
            message="The file has been updated successfully.",
            old_string=old_string,
            new_string=new_string,
        )
        result._ui_occurrences = [
            (
                1,
                "\n".join([
                    "def greet(name: str) -> str:",
                    old_string,
                    "",
                    'print(greet("world"))',
                ]),
                "\n".join([
                    "def greet(name: str) -> str:",
                    new_string,
                    "",
                    'print(greet("world"))',
                ]),
            )
        ]
        await projection.dispatch(
            ToolResultEvent(
                tool_name="edit", tool_class=Edit, result=result, tool_call_id=call_id
            ),
            self._handler.handle_event,
        )


async def _populated_edit(pilot: Pilot) -> None:
    app = cast(_SnapshotApp, pilot.app)
    await app.populate()
    await pilot.pause(0.3)
    widget = pilot.app.query_one(EditResultWidget)
    assert widget._render_worker is not None
    assert widget._render_worker.state == WorkerState.SUCCESS


def test_snapshot_edit_result(snap_compare: SnapCompare) -> None:
    assert snap_compare(
        "test_ui_snapshot_overflow_sections.py:EditResultApp",
        terminal_size=(80, 16),
        run_before=_populated_edit,
    )
