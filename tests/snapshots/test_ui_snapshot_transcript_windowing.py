from __future__ import annotations

import asyncio
from typing import cast

import pytest
from textual.pilot import Pilot

from chartreux.app_server.models import (
    PublicEntryGenerationStatus,
    PublicMessageEntry,
    TextContentBlock,
)
from chartreux.cli.textual_ui.app import ChatScroll
from tests.conftest import build_test_agent_loop
from tests.snapshots.base_snapshot_test_app import BaseSnapshotTestApp, default_config
from tests.snapshots.snap_compare import SnapCompare
from tests.stubs.app_server import create_test_app_server_session


class TranscriptWindowingSnapshotApp(BaseSnapshotTestApp):
    def __init__(self) -> None:
        agent_loop = build_test_agent_loop(config=default_config())
        super().__init__(agent_loop=agent_loop)

        async def start_session():
            session = await create_test_app_server_session(agent_loop)
            session._state.projection.state.history = [
                PublicMessageEntry(
                    id=f"snapshot-window-{index:03d}",
                    session_id="snapshot-window",
                    turn_id=f"snapshot-turn-{index:03d}",
                    created_at=index,
                    updated_at=index,
                    generation_status=PublicEntryGenerationStatus.COMPLETED,
                    role="user",
                    content=[
                        TextContentBlock(
                            text=(
                                f"Transcript window {index:03d}: stable content for remount parity.\n"
                                * 3
                            )
                        )
                    ],
                    source="harness",
                )
                for index in range(20)
            ]
            return session

        self._start_app_server = start_session


async def _resume_scroll_up_down(pilot: Pilot) -> None:
    app = cast(TranscriptWindowingSnapshotApp, pilot.app)
    await pilot.pause(0.5)
    chat = app.query_one("#chat", ChatScroll)
    chat.scroll_to(y=0, animate=False, force=True, immediate=True)
    await pilot.pause(0.1)
    before = app.export_screenshot()
    initial_roots = {
        unit.id: tuple(unit.mounted_roots)
        for unit in app._transcript.units.values()
        if unit.mounted_roots
    }
    chat.scroll_end(animate=False, immediate=True)
    await pilot.pause(0.1)
    first = app._transcript.units[next(iter(initial_roots))]
    assert first.mounted_roots and first.mounted_roots[-1].region.bottom < chat.region.y
    await app._transcript.evict_unit(first.id)
    chat.scroll_to(y=0, animate=False, force=True, immediate=True)
    await pilot.pause(0.1)
    task = app._transcript._reconcile_task
    if task is not None:
        await asyncio.wait_for(asyncio.shield(task), 15)
    chat.scroll_to(y=0, animate=False, force=True, immediate=True)
    await pilot.pause(0.1)
    assert any(
        unit.mounted_roots and tuple(unit.mounted_roots) != roots
        for unit_id, roots in initial_roots.items()
        if (unit := app._transcript.units[unit_id])
    ), "scroll round-trip did not remount any transcript units"
    assert app.export_screenshot() == before


@pytest.mark.timeout(90)
def test_resume_scroll_up_down_snapshot(snap_compare: SnapCompare) -> None:
    assert snap_compare(
        "test_ui_snapshot_transcript_windowing.py:TranscriptWindowingSnapshotApp",
        terminal_size=(120, 36),
        run_before=_resume_scroll_up_down,
    )
