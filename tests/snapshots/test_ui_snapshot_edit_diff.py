from __future__ import annotations

from textual.pilot import Pilot
from textual.worker import WorkerState

from chartreux.app_server.models import FileEditEffectOccurrence, FileEditEffectOutput
from chartreux.cli.textual_ui.widgets.tool_widgets import EditResultWidget
from tests.snapshots.base_snapshot_test_app import BaseSnapshotTestApp
from tests.snapshots.snap_compare import SnapCompare


def _edit_result(
    *, file: str, occurrences: list[FileEditEffectOccurrence]
) -> EditResultWidget:
    return EditResultWidget(
        FileEditEffectOutput(file=file, occurrences=occurrences),
        success=True,
        message="updated",
    )


# Keep historical snapshot IDs while exercising the retained result renderer.
class EditApprovalApp(BaseSnapshotTestApp):
    _diff_theme: str = "tokyo-night"

    async def on_ready(self) -> None:
        await super().on_ready()
        self.theme = self._diff_theme
        await self._mount_and_scroll(
            _edit_result(
                file="src/example.py",
                occurrences=[
                    FileEditEffectOccurrence(
                        start_line=4,
                        old_text="MAX_USERS = 100\nTIMEOUT = 30",
                        new_text="MAX_USERS = 200\nTIMEOUT = 30",
                    )
                ],
            )
        )


class EditApprovalAnsiApp(EditApprovalApp):
    _diff_theme = "ansi-dark"


class EditReplaceAllApprovalApp(BaseSnapshotTestApp):
    _diff_theme: str = "tokyo-night"

    async def on_ready(self) -> None:
        await super().on_ready()
        self.theme = self._diff_theme
        await self._mount_and_scroll(
            _edit_result(
                file="src/counter.py",
                occurrences=[
                    FileEditEffectOccurrence(
                        start_line=2, old_text="    count = 0", new_text="    count = 1"
                    ),
                    FileEditEffectOccurrence(
                        start_line=9,
                        old_text="        count = 0  # start over",
                        new_text="        count = 1  # start over",
                    ),
                ],
            )
        )


LONG_OLD = "    message = " + " + ".join(f'"word_{i}"' for i in range(40))
LONG_NEW = "    message = " + " + ".join(f'"token_{i}"' for i in range(40))


class EditOverflowApprovalApp(BaseSnapshotTestApp):
    _diff_theme: str = "tokyo-night"

    async def on_ready(self) -> None:
        await super().on_ready()
        self.theme = self._diff_theme
        await self._mount_and_scroll(
            _edit_result(
                file="src/message.py",
                occurrences=[
                    FileEditEffectOccurrence(
                        start_line=2,
                        old_text=f"{LONG_OLD}\n    return message",
                        new_text=f"{LONG_NEW}\n    return message.upper()",
                    )
                ],
            )
        )


def test_snapshot_edit_approval_diff(snap_compare: SnapCompare) -> None:
    async def run_before(pilot: Pilot) -> None:
        await pilot.pause(0.3)
        widget = pilot.app.query_one(EditResultWidget)
        worker = widget._render_worker
        assert worker is not None and worker.state == WorkerState.SUCCESS

    assert snap_compare(
        "test_ui_snapshot_edit_diff.py:EditApprovalApp",
        terminal_size=(100, 30),
        run_before=run_before,
    )


def test_snapshot_edit_approval_diff_ansi(snap_compare: SnapCompare) -> None:
    async def run_before(pilot: Pilot) -> None:
        await pilot.pause(0.3)
        widget = pilot.app.query_one(EditResultWidget)
        worker = widget._render_worker
        assert worker is not None and worker.state == WorkerState.SUCCESS

    assert snap_compare(
        "test_ui_snapshot_edit_diff.py:EditApprovalAnsiApp",
        terminal_size=(100, 30),
        run_before=run_before,
    )


def test_snapshot_edit_approval_diff_replace_all(snap_compare: SnapCompare) -> None:
    async def run_before(pilot: Pilot) -> None:
        await pilot.pause(0.3)
        widget = pilot.app.query_one(EditResultWidget)
        worker = widget._render_worker
        assert worker is not None and worker.state == WorkerState.SUCCESS

    assert snap_compare(
        "test_ui_snapshot_edit_diff.py:EditReplaceAllApprovalApp",
        terminal_size=(100, 30),
        run_before=run_before,
    )


def test_snapshot_edit_approval_diff_horizontal_overflow(
    snap_compare: SnapCompare,
) -> None:
    async def run_before(pilot: Pilot) -> None:
        await pilot.pause(0.3)
        widget = pilot.app.query_one(EditResultWidget)
        worker = widget._render_worker
        assert worker is not None and worker.state == WorkerState.SUCCESS

    assert snap_compare(
        "test_ui_snapshot_edit_diff.py:EditOverflowApprovalApp",
        terminal_size=(100, 30),
        run_before=run_before,
    )
