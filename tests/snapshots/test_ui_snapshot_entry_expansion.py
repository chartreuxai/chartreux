from __future__ import annotations

from typing import cast

from textual.pilot import Pilot

from chartreux.cli.textual_ui.widgets.collapsible import CollapsibleSection
from chartreux.cli.textual_ui.widgets.tools import ToolGroup
from tests.snapshots.snap_compare import SnapCompare
from tests.snapshots.test_ui_snapshot_session_resume import (
    SnapshotTestAppWithResumedSession,
)


class EntryExpansionSnapshotApp(SnapshotTestAppWithResumedSession):
    pass


async def _resume_expand_and_rebuild(pilot: Pilot) -> None:
    app = cast(EntryExpansionSnapshotApp, pilot.app)
    await pilot.pause(0.5)
    group = app._messages_area.query_one(ToolGroup)
    group.set_collapsed(False)
    await pilot.pause()
    section = app._messages_area.query_one(CollapsibleSection)
    section.set_collapsed(False)
    await pilot.pause()
    before = app.export_screenshot()
    await app._rebuild_transcript_from_current_session()
    await pilot.pause(0.5)
    assert not app._messages_area.query_one(ToolGroup).is_collapsed
    assert not app._messages_area.query_one(CollapsibleSection).is_collapsed
    assert app.export_screenshot() == before


def test_resume_change_expansion_rebuild_snapshot(snap_compare: SnapCompare) -> None:
    assert snap_compare(
        "test_ui_snapshot_entry_expansion.py:EntryExpansionSnapshotApp",
        terminal_size=(120, 36),
        run_before=_resume_expand_and_rebuild,
    )
