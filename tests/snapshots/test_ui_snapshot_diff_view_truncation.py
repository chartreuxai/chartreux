from __future__ import annotations

from textual.containers import VerticalScroll
from textual.pilot import Pilot

from chartreux.app_server.models import FileWriteEffectOutput
from chartreux.cli.textual_ui.widgets.tool_widgets import WriteFileResultWidget
from chartreux.core.tools.builtins.write_file import WriteFileArgs
from tests.snapshots.base_snapshot_test_app import BaseSnapshotTestApp
from tests.snapshots.snap_compare import SnapCompare

WF_CONTENT_LONG = "\n".join(f"line_{i:03d} = {i * 7}" for i in range(1, 101))
WF_CONTENT_SHORT = "line_001 = 7"


def _write_result(args: WriteFileArgs) -> WriteFileResultWidget:
    return WriteFileResultWidget(
        FileWriteEffectOutput(file_path=args.file_path, content=args.content),
        success=True,
        message="written",
    )


# Keep historical snapshot IDs while exercising the retained result renderer.
class WriteApprovalLongContentApp(BaseSnapshotTestApp):
    async def on_ready(self) -> None:
        args = WriteFileArgs(file_path="src/example.py", content=WF_CONTENT_LONG)
        await self._mount_and_scroll(_write_result(args))


class WriteApprovalShortContentApp(BaseSnapshotTestApp):
    async def on_ready(self) -> None:
        args = WriteFileArgs(file_path="src/example.py", content=WF_CONTENT_SHORT)
        await self._mount_and_scroll(_write_result(args))


def test_snapshot_write_approval_long_content_bottom_lines_hidden(
    snap_compare: SnapCompare,
) -> None:
    async def run_before(pilot: Pilot) -> None:
        await pilot.pause(0.3)
        scroll = pilot.app.query_one("#chat", VerticalScroll)
        scroll.scroll_end(animate=False, immediate=True)
        await pilot.pause(0.2)

    assert snap_compare(
        "test_ui_snapshot_diff_view_truncation.py:WriteApprovalLongContentApp",
        terminal_size=(100, 30),
        run_before=run_before,
    )


def test_snapshot_write_approval_short_content(snap_compare: SnapCompare) -> None:
    async def run_before(pilot: Pilot) -> None:
        await pilot.pause(0.3)

    assert snap_compare(
        "test_ui_snapshot_diff_view_truncation.py:WriteApprovalShortContentApp",
        terminal_size=(100, 30),
        run_before=run_before,
    )


def test_snapshot_write_approval_long_content_after_resize(
    snap_compare: SnapCompare,
) -> None:
    async def run_before(pilot: Pilot) -> None:
        await pilot.pause(0.3)
        await pilot.resize_terminal(120, 40)
        await pilot.pause(0.2)
        scroll = pilot.app.query_one("#chat", VerticalScroll)
        scroll.scroll_end(animate=False, immediate=True)
        await pilot.pause(0.2)

    assert snap_compare(
        "test_ui_snapshot_diff_view_truncation.py:WriteApprovalLongContentApp",
        terminal_size=(100, 30),
        run_before=run_before,
    )
