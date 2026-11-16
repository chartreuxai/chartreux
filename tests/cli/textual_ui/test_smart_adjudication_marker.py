"""v0.8.2 W7: transcript markers for classifier-consulted adjudications.

Green "allowed by smart approve" on override, red "denied by smart
approve" on uphold/cap/classifier-error, rendered once per effect entry
id; decided-call replays carry no marker and never re-render.
"""

from __future__ import annotations

import pytest

from chartreux.app_server._effect_models import GenericEffectDetail
from chartreux.app_server.events import HistoryEntryUpdated
from chartreux.app_server.models import (
    CompletedEffectState,
    PublicEffectEntry,
    PublicEntryGenerationStatus,
    SkippedEffectState,
)
from chartreux.cli.textual_ui.widgets.messages import UserCommandMessage
from chartreux.utils.tool_presentation import EffectCallDisplay, EffectResultDisplay
from tests.conftest import build_test_chartreux_app

pytestmark = pytest.mark.asyncio


def _effect_entry(entry_id: str, state: object) -> PublicEffectEntry:
    return PublicEffectEntry(
        id=entry_id,
        session_id="sess",
        created_at=1,
        updated_at=2,
        generation_status=PublicEntryGenerationStatus.COMPLETED,
        title="bash",
        detail=GenericEffectDetail(
            tool_name="bash",
            display=EffectCallDisplay(summary="bash ls", status_text="ran"),
        ),
        state=state,  # type: ignore[arg-type]
    )


def _completed_entry(entry_id: str, message: str) -> PublicEffectEntry:
    return _effect_entry(
        entry_id,
        CompletedEffectState(
            display=EffectResultDisplay(success=True, message=message), output_text=""
        ),
    )


def _skipped_entry(entry_id: str, reason: str) -> PublicEffectEntry:
    return _effect_entry(
        entry_id,
        SkippedEffectState(
            reason=reason, display=EffectResultDisplay(success=False, message=reason)
        ),
    )


def _event(entry: PublicEffectEntry) -> HistoryEntryUpdated:
    return HistoryEntryUpdated(previous=entry, entry=entry, patch=[])


async def test_override_marker_renders_once_with_allowed_variant() -> None:
    app = build_test_chartreux_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await app._session_ready.wait()
        entry = _completed_entry(
            "call-override-1", "Ran bash\nSMART classifier override: requested by user"
        )
        await app._maybe_render_smart_adjudication_marker(_event(entry), entry)
        # Repeated HistoryEntryUpdated for the same entry id must not re-render.
        await app._maybe_render_smart_adjudication_marker(_event(entry), entry)
        await pilot.pause()
        markers = list(app.query(UserCommandMessage))
        assert len(markers) == 1
        assert markers[0].has_class("smart-approve-allowed")
        assert "allowed by smart approve" in markers[0]._content
        assert "requested by user" in markers[0]._content


async def test_uphold_marker_renders_with_denied_variant_on_skipped_entry() -> None:
    app = build_test_chartreux_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await app._session_ready.wait()
        entry = _skipped_entry(
            "call-uphold-1",
            "Shell path is outside the authorized workspace\n"
            "SMART classifier uphold: accident risk",
        )
        await app._maybe_render_smart_adjudication_marker(_event(entry), entry)
        await app._maybe_render_smart_adjudication_marker(_event(entry), entry)
        await pilot.pause()
        markers = list(app.query(UserCommandMessage))
        assert len(markers) == 1
        assert markers[0].has_class("smart-approve-denied")
        assert "denied by smart approve" in markers[0]._content
        assert "accident risk" in markers[0]._content


async def test_entry_without_marker_renders_nothing() -> None:
    app = build_test_chartreux_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await app._session_ready.wait()
        # A decided-call replay returns the original denial: no marker text.
        entry = _skipped_entry(
            "call-replay-1", "Shell path is outside the authorized workspace"
        )
        await app._maybe_render_smart_adjudication_marker(_event(entry), entry)
        await pilot.pause()
        assert list(app.query(UserCommandMessage)) == []
