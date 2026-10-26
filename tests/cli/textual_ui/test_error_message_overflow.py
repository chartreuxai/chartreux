"""ErrorMessage oversized-payload containment (v0.5.9 incident fix).

A fatal provider error can embed a multi-kilobyte single-line JSON excerpt.
Before the fix, ErrorMessage rendered the raw payload with ``height: auto``
and no cap anywhere in its styles, so Textual's fold-wrap chopped the
unbreakable token into width-sized lines and the widget grew to ~59 rows in
an 18-row chat viewport, obscuring every subsequent turn.

The fix: an error estimated taller than ``_ERROR_MAX_ROWS`` renders as a
one-line ellipsized summary plus the house ``+N lines`` disclosure; the
expanded body is capped by the ``.error-overflow-body`` max-height, so even
``collapsed=False`` can never reclaim the viewport. Upstream,
``_resolve_turn_error_message`` truncates megabyte payloads to
``_MAX_TURN_ERROR_CHARS``.

No live provider calls: everything here mounts real widgets with synthetic
incident-shaped payloads.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from textual.app import App, ComposeResult

from chartreux.app_server.models import PublicError, TurnErrorCode
from chartreux.app_server.session import AppServerTurnError
from chartreux.cli.textual_ui.turn_supervisor import (
    _MAX_TURN_ERROR_CHARS,
    TurnSupervisor,
)
from chartreux.cli.textual_ui.widgets.collapsible import (
    DisclosureHeader,
    OverflowCollapsibleSection,
)
from chartreux.cli.textual_ui.widgets.messages import AssistantMessage, ErrorMessage
from tests.conftest import build_test_chartreux_app


def _incident_payload() -> str:
    """The incident shape: one unbreakable multi-kilobyte JSON token."""
    excerpt = json.dumps({
        "error": "model_terminated",
        "payload_summary": "x" * 4000,
        "request_id": "req_0123456789abcdef0123456789abcdef",
    })
    return (
        "API request failed with status 502: upstream terminated the connection "
        "before a response was received. Provider body: " + excerpt
    )


@pytest.mark.asyncio
async def test_error_message_height_capped_for_long_payload() -> None:
    """The incident payload mounts as a one-line summary plus the "+N lines"
    disclosure; expanding stays within the cap and the disclosure is
    keyboard-operable (DisclosureHeader tab stop, Enter/Space toggles).
    """
    app = build_test_chartreux_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()

        error = ErrorMessage(_incident_payload(), collapsed=True)
        await app._mount_and_scroll(error)
        await pilot.pause()

        section = error.query_one(OverflowCollapsibleSection)
        assert section.is_collapsed
        # Collapsed: one summary row + the disclosure row (head is nowrap,
        # so the 4000-char token can never wrap it taller).
        assert error.region.height <= 3

        section.set_collapsed(False)
        await pilot.pause()
        assert not section.is_collapsed
        # Expanded: summary + capped body (_ERROR_MAX_ROWS=12) + disclosure.
        assert error.region.height <= 15

        section.set_collapsed(True)
        await pilot.pause()
        assert section.is_collapsed
        assert error.region.height <= 3


@pytest.mark.asyncio
async def test_error_disclosure_is_keyboard_operable() -> None:
    """The disclosure header is a tab stop: Enter toggles it both ways."""

    class _ErrorHost(App[None]):
        def compose(self) -> ComposeResult:
            yield ErrorMessage(_incident_payload(), collapsed=True)

    app = _ErrorHost()
    async with app.run_test(size=(80, 24)) as pilot:
        header = app.query_one(DisclosureHeader)
        assert header.can_focus
        header.focus()
        await pilot.press("enter")
        await pilot.pause()
        assert app.query_one(OverflowCollapsibleSection).is_collapsed is False
        await pilot.press("enter")
        await pilot.pause()
        assert app.query_one(OverflowCollapsibleSection).is_collapsed is True


@pytest.mark.asyncio
async def test_error_message_does_not_obscure_subsequent_turns() -> None:
    """After the capped error, a follow-up AssistantMessage is reachable:
    its region intersects the visible screen region after scroll_end, and
    the chat scroll range covers its bottom edge.
    """
    app = build_test_chartreux_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()

        error = ErrorMessage(_incident_payload(), collapsed=True)
        await app._mount_and_scroll(error)
        await pilot.pause()

        follow_up = AssistantMessage("Follow-up turn text after the error.")
        await app._mount_and_scroll(follow_up)
        await pilot.pause()
        chat = app.query_one("#chat")
        chat.scroll_end(animate=False)
        await pilot.pause()

        follow_region = follow_up.region
        chat_region = chat.region
        intersects = (
            follow_region.y + follow_region.height > chat_region.y
            and follow_region.y < chat_region.y + chat_region.height
        )
        assert intersects, "follow-up turn must intersect the visible region"
        bottom_offset = follow_region.y + follow_region.height - chat_region.y
        assert chat.scroll_y + chat_region.height >= bottom_offset - 0.5, (
            "the scroll range must cover the follow-up's bottom edge"
        )


@pytest.mark.asyncio
async def test_error_message_height_stable_across_new_turns() -> None:
    """The error widget's height is identical across three subsequent mounts."""
    app = build_test_chartreux_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()

        error = ErrorMessage(_incident_payload(), collapsed=True)
        await app._mount_and_scroll(error)
        await pilot.pause()
        initial_height = error.region.height

        for i in range(3):
            await app._mount_and_scroll(AssistantMessage(f"Subsequent turn {i}."))
            await pilot.pause()
            assert error.region.height == initial_height


def test_turn_error_message_truncates_long_payload() -> None:
    """_resolve_turn_error_message bounds a megabyte provider excerpt: the
    first line (provider, model, error head) survives verbatim, the tail is
    replaced by an explicit truncation marker, and short messages pass
    through untouched.
    """
    huge = "API error from acme (model: big): " + json.dumps({
        "body_excerpt": "y" * 1_000_000
    })
    error = AppServerTurnError(
        PublicError(message=huge, code=TurnErrorCode.INTERNAL_ERROR)
    )
    message = TurnSupervisor._resolve_turn_error_message(MagicMock(), error)

    assert message.startswith("API error from acme (model: big): ")
    assert "truncated" in message
    assert len(message) <= _MAX_TURN_ERROR_CHARS + 100


def test_turn_error_message_keeps_short_payload_verbatim() -> None:
    error = AppServerTurnError(
        PublicError(
            message="API error from acme (model: big): upstream 502",
            code=TurnErrorCode.INTERNAL_ERROR,
        )
    )
    message = TurnSupervisor._resolve_turn_error_message(MagicMock(), error)
    assert message == "API error from acme (model: big): upstream 502"
