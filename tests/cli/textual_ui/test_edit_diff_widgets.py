from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import patch

import pytest
from textual.app import App, ComposeResult

from chartreux.app_server.models import (
    CompletedEffectState,
    EffectCallDisplay,
    EffectResultDisplay,
    FileEditEffectDetail,
    FileEditEffectOccurrence,
    FileEditEffectOutput,
    FileWriteEffectDetail,
    FileWriteEffectOutput,
    PublicEffectEntry,
    PublicEntryGenerationStatus,
)
from chartreux.cli.textual_ui.widgets.collapsible import HeaderCollapsibleSection
from chartreux.cli.textual_ui.widgets.diff_rendering import (
    DiffOccurrence,
    render_edit_diff,
)
from chartreux.cli.textual_ui.widgets.entry_expansion import EntryExpansionState
from chartreux.cli.textual_ui.widgets.tool_widgets import (
    EditResultWidget,
    WriteFileResultWidget,
    edit_result_stats,
    effect_result_is_collapsible,
)
from chartreux.cli.textual_ui.widgets.tools import ToolCallMessage, ToolResultMessage


def _result_widget() -> EditResultWidget:
    return EditResultWidget(
        FileEditEffectOutput(
            file="example.py", old_string="value = 1", new_string="value = 2"
        ),
        success=True,
        message="updated",
    )


def _edit_entry(entry_id: str = "edit") -> PublicEffectEntry:
    output = FileEditEffectOutput(
        file="example.py",
        occurrences=[
            FileEditEffectOccurrence(
                start_line=1, old_text="value = 1", new_text="value = 2"
            )
        ],
    )
    return PublicEffectEntry(
        id=entry_id,
        session_id="session",
        turn_id="turn",
        created_at=1,
        updated_at=2,
        generation_status=PublicEntryGenerationStatus.COMPLETED,
        title="tool",
        detail=FileEditEffectDetail(
            tool_name="edit",
            display=EffectCallDisplay(
                summary="Editing example.py",
                message="example.py",
                verb="Editing",
                settled_verb="Edited",
                settled_message="example.py",
                status_text="Editing",
            ),
        ),
        state=CompletedEffectState(
            output=output.model_dump(mode="json", by_alias=True),
            output_text="The file has been updated successfully.",
            display=EffectResultDisplay(
                success=True, verb="Edited", message="example.py"
            ),
        ),
    )


def _write_entry(entry_id: str = "write") -> PublicEffectEntry:
    output = FileWriteEffectOutput(file_path="example.py", content="value = 1\n")
    return PublicEffectEntry(
        id=entry_id,
        session_id="session",
        turn_id="turn",
        created_at=1,
        updated_at=2,
        generation_status=PublicEntryGenerationStatus.COMPLETED,
        title="tool",
        detail=FileWriteEffectDetail(
            tool_name="write_file",
            display=EffectCallDisplay(
                summary="Writing example.py",
                message="example.py",
                verb="Writing",
                settled_verb="Created",
                settled_message="example.py",
                status_text="Writing",
            ),
        ),
        state=CompletedEffectState(
            output=output.model_dump(mode="json", by_alias=True),
            output_text="File written successfully.",
            display=EffectResultDisplay(
                success=True, verb="Created", message="example.py"
            ),
        ),
    )


class _ResultApp(App[None]):
    def __init__(self, entry: PublicEffectEntry, *, state: EntryExpansionState) -> None:
        super().__init__()
        self._call = ToolCallMessage(entry)
        self._result = ToolResultMessage(entry, self._call, expansion_state=state)

    def compose(self) -> ComposeResult:
        yield self._call
        yield self._result


def _header_suffix_text(section: HeaderCollapsibleSection) -> str:
    suffix = section.query(".status-indicator-suffix")
    return " ".join(str(widget.render()) for widget in suffix)


def test_edit_and_write_results_join_the_collapsible_family() -> None:
    assert EditResultWidget.COLLAPSIBLE is True
    assert WriteFileResultWidget.COLLAPSIBLE is True
    assert effect_result_is_collapsible(_edit_entry().detail) is True
    assert effect_result_is_collapsible(_write_entry().detail) is True


def _output_dump(output: FileEditEffectOutput) -> Any:
    # The header consumes the effect state's JSON dump, not the parsed model.
    return output.model_dump(mode="json", by_alias=True)


def test_edit_header_stats_count_diff_rows_not_snippet_lines() -> None:
    # The legacy pair renders one removed and one added row.
    output = FileEditEffectOutput(
        file="example.py", old_string="value = 1", new_string="value = 2"
    )
    assert edit_result_stats(_output_dump(output)) == "+1/-1"


def test_edit_header_stats_count_every_occurrence() -> None:
    output = FileEditEffectOutput(
        file="example.py",
        occurrences=[
            FileEditEffectOccurrence(start_line=1, old_text="a = 1", new_text="a = 2"),
            FileEditEffectOccurrence(start_line=9, old_text="a = 1", new_text="a = 2"),
        ],
    )
    assert edit_result_stats(_output_dump(output)) == "+2/-2"


def test_edit_header_stats_ignore_context_rows() -> None:
    # Six-line snippet, one changed line: context rows are not counted.
    output = FileEditEffectOutput(
        file="example.py",
        occurrences=[
            FileEditEffectOccurrence(
                start_line=1, old_text="a\nb\nc\nd\ne\nf", new_text="a\nb\nX\nd\ne\nf"
            )
        ],
    )
    assert edit_result_stats(_output_dump(output)) == "+1/-1"


def test_edit_result_without_a_diff_has_no_stats() -> None:
    assert edit_result_stats(None) == ""


def test_edit_header_stats_match_the_expanded_diff_rows() -> None:
    occurrences = [
        DiffOccurrence(2, "x = 100", "x = 200"),
        DiffOccurrence(7, "y = 1\nz = 2", "y = 1\nz = 3"),
    ]
    output = FileEditEffectOutput(
        file="example.py",
        occurrences=[
            FileEditEffectOccurrence(
                start_line=item.start_line,
                old_text=item.old_lines,
                new_text=item.new_lines,
            )
            for item in occurrences
        ],
    )
    lines = render_edit_diff(occurrences, "py", ansi=False, dark=True)
    added = sum(line.css_class == "diff-added" for line in lines)
    removed = sum(line.css_class == "diff-removed" for line in lines)
    assert edit_result_stats(_output_dump(output)) == f"+{added}/-{removed}"


@pytest.mark.asyncio
async def test_edit_result_is_collapsed_by_default_and_builds_no_diff() -> None:
    entry = _edit_entry()
    state = EntryExpansionState()
    app = _ResultApp(entry, state=state)

    async with app.run_test() as pilot:
        await pilot.pause()
        assert app._result._is_collapsible
        section = app._result.query_one(HeaderCollapsibleSection)
        assert section.is_collapsed
        # Nothing heavy is constructed while collapsed: no diff body, no DiffView.
        assert not app._result.query(EditResultWidget)
        assert not app._result.query("DiffView")
        # The section header replaces the settled call row.
        assert app._call.display is False
        # Registration makes the entry a session-local expansion choice.
        assert "edit" in state.entry_ids
        assert state.is_collapsed("edit")
        assert _header_suffix_text(section) == "+1/-1"


@pytest.mark.asyncio
async def test_expanded_edit_result_renders_the_diff_and_binds_its_border() -> None:
    entry = _edit_entry()
    app = _ResultApp(entry, state=EntryExpansionState())

    async with app.run_test() as pilot:
        await pilot.pause()
        section = app._result.query_one(HeaderCollapsibleSection)
        section.set_collapsed(False)
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        widget = app._result.query_one(EditResultWidget)
        assert widget._occurrences == [DiffOccurrence(1, "value = 1", "value = 2")]
        assert app._result._border is not None
        assert app._result._border._row_colors == {
            0: "not dim $error",
            1: "not dim $success",
        }

        section.set_collapsed(True)
        await pilot.pause()
        assert not app._result.query(EditResultWidget)

        # Re-expanding rebuilds the body from the factory and rebinds its border.
        section.set_collapsed(False)
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        rebuilt = app._result.query_one(EditResultWidget)
        assert rebuilt is not widget
        assert app._result._result_widget is rebuilt
        assert app._result._border is not None
        assert app._result._border._row_colors == {
            0: "not dim $error",
            1: "not dim $success",
        }


@pytest.mark.asyncio
async def test_write_result_collapsed_header_keeps_the_plain_call_line() -> None:
    entry = _write_entry()
    app = _ResultApp(entry, state=EntryExpansionState())

    async with app.run_test() as pilot:
        await pilot.pause()
        section = app._result.query_one(HeaderCollapsibleSection)
        assert section.is_collapsed
        assert app._result.query_one(".status-indicator-text").render() == "example.py"
        assert _header_suffix_text(section) == ""
        assert app._call.display is False

        section.set_collapsed(False)
        await pilot.pause()
        assert app._result.query_one(WriteFileResultWidget)


def test_occurrence_only_result_builds_ordered_diff_widget() -> None:
    result = EditResultWidget(
        FileEditEffectOutput(
            file="example.py",
            occurrences=[
                FileEditEffectOccurrence(
                    start_line=1, old_text="alpha\n", new_text="beta\n"
                ),
                FileEditEffectOccurrence(
                    start_line=1, old_text="beta\n", new_text="gamma\n"
                ),
            ],
        ),
        success=True,
        message="updated",
    )

    assert result._occurrences == [
        DiffOccurrence(1, "alpha\n", "beta\n"),
        DiffOccurrence(1, "beta\n", "gamma\n"),
    ]


@pytest.mark.asyncio
async def test_restyle_keeps_existing_diff_until_background_render_finishes() -> None:
    widget = _result_widget()

    class _App(App[None]):
        def compose(self) -> ComposeResult:
            yield widget

    async with _App().run_test() as pilot:
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        original_lines = widget._diff_view._lines
        assert not widget._diff_view._ansi
        replacement_lines = render_edit_diff(
            [DiffOccurrence(1, "value = 2", "value = 3")], "py", ansi=True, dark=True
        )
        render_started = asyncio.Event()
        allow_render = asyncio.Event()

        async def delayed_render(*args, **kwargs):
            render_started.set()
            await allow_render.wait()
            return replacement_lines

        with patch(
            "chartreux.cli.textual_ui.widgets.tool_widgets.render_edit_diff_async",
            side_effect=delayed_render,
        ):
            worker = widget.request_diff_render(ansi=True, dark=True)
            assert worker is not None
            await render_started.wait()

            assert widget._diff_view._lines is original_lines
            assert widget._diff_view._ansi
            assert widget._diff_view._dark

            allow_render.set()
            await worker.wait()

        assert widget._diff_view._lines is replacement_lines


@pytest.mark.asyncio
async def test_result_uses_theme_requested_before_worker_starts() -> None:
    widget = _result_widget()

    class _App(App[None]):
        def compose(self) -> ComposeResult:
            yield widget

    async with _App().run_test() as pilot:
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        worker = widget.request_diff_render(ansi=True, dark=True)
        assert worker is not None
        assert widget.request_diff_render(ansi=False, dark=False) is worker
        await worker.wait()

        assert not widget._diff_view._ansi
        assert not widget._diff_view._dark
        assert widget._diff_view._lines == render_edit_diff(
            widget._occurrences, "py", ansi=False, dark=False
        )


@pytest.mark.asyncio
async def test_latest_diff_render_wins_while_an_older_render_is_running() -> None:
    widget = _result_widget()

    class _App(App[None]):
        def compose(self) -> ComposeResult:
            yield widget

    async with _App().run_test() as pilot:
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        first_lines = render_edit_diff(
            [DiffOccurrence(1, "first = 1", "first = 2")], "py", ansi=True, dark=True
        )
        latest_lines = render_edit_diff(
            [DiffOccurrence(1, "latest = 1", "latest = 2")],
            "py",
            ansi=False,
            dark=False,
        )
        first_started = asyncio.Event()
        latest_started = asyncio.Event()
        release_first = asyncio.Event()
        release_latest = asyncio.Event()

        async def delayed_render(*args, ansi: bool, **kwargs):
            if ansi:
                first_started.set()
                await release_first.wait()
                return first_lines
            latest_started.set()
            await release_latest.wait()
            return latest_lines

        with patch(
            "chartreux.cli.textual_ui.widgets.tool_widgets.render_edit_diff_async",
            side_effect=delayed_render,
        ):
            worker = widget.request_diff_render(ansi=True, dark=True)
            assert worker is not None
            await first_started.wait()

            assert widget.request_diff_render(ansi=False, dark=False) is worker
            release_first.set()
            await latest_started.wait()
            release_latest.set()
            await worker.wait()

        assert widget._diff_view._lines is latest_lines
        assert not widget._diff_view._ansi
        assert not widget._diff_view._dark
