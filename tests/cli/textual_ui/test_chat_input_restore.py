"""Failed-send draft restoration (design row 17): the draft, never a merge.

A refused send restores what the user submitted as the input's draft. The three
properties pinned here are the ones the PTY artifact ``out/stdio-p0b`` broke:
the caret sits after the restored text, the restored draft keeps its mode and
attachment tokens, and text typed while the send was failing is never
concatenated with or overwritten by it.
"""

from __future__ import annotations

import pytest
import pytest_asyncio

from chartreux.cli.textual_ui.widgets.chat_input import ChatInputContainer
from tests.conftest import build_test_chartreux_app
from tests.snapshots.snapshot_event_loop import install_snapshot_wake


@pytest_asyncio.fixture(autouse=True)
async def _snapshot_event_loop_wake() -> None:
    install_snapshot_wake()


@pytest.mark.asyncio
async def test_restored_draft_keeps_its_attachment_and_caret_at_the_end() -> None:
    app = build_test_chartreux_app()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.1)
        container = app.query_one(ChatInputContainer)
        draft = "look at @/tmp/draft-shot.png"

        assert container.restore_draft(draft)

        assert container.value == draft
        widget = container.input_widget
        assert widget is not None
        assert widget.get_cursor_offset() == len(widget.text)
        # The caret is the whole point: the next keystrokes follow the draft
        # instead of being prepended to it.
        widget.insert(" and this")
        assert container.value == draft + " and this"


@pytest.mark.asyncio
async def test_restored_draft_keeps_its_input_mode() -> None:
    app = build_test_chartreux_app()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.1)
        container = app.query_one(ChatInputContainer)

        assert container.restore_draft("!echo restored")

        widget = container.input_widget
        assert widget is not None
        assert container.value == "!echo restored"
        assert not widget.is_default_mode
        assert widget.get_cursor_offset() == len("echo restored")


@pytest.mark.asyncio
async def test_typed_since_text_is_never_overwritten_or_concatenated() -> None:
    app = build_test_chartreux_app()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.1)
        container = app.query_one(ChatInputContainer)
        container.value = "typed while the send was failing"

        assert not container.restore_draft("the refused prompt")

        assert container.value == "typed while the send was failing"
