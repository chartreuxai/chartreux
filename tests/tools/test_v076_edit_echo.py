"""P6a gates: tuned edit-echo bounds and line-keyed hidden-site anchors.

The bounds were tuned against the recorded root-session edit corpus (3,572
edit results across 25 sessions): rendered context p50 1.25KB / p95 2.74KB /
max 5.64KB; hunk counts {1: 3544, 2: 8, 3: 11, 4: 7, 5: 1, 8: 1}. The window
bound now binds the observed tail (1.0% of windows clipped, first-site lines
always kept) and sites past the hunk cap are named by their current line
number in a trailing note bounded at ``EDIT_DIFF_NOTE_MAX_BYTES``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chartreux.core.tools.base import BaseToolState
from chartreux.core.tools.builtins._edit_result_context import (
    EDIT_DIFF_MAX_SITES,
    EDIT_DIFF_NOTE_MAX_BYTES,
    EDIT_WINDOW_MAX_BYTES,
    AppliedSite,
    find_applied_sites,
    render_applied_diff,
    render_context_window,
)
from chartreux.core.tools.builtins.edit import Edit, EditArgs, EditConfig
from tests.mock.utils import collect_result


def _make_edit() -> Edit:
    return Edit(config_getter=lambda: EditConfig(), state=BaseToolState())


def _hidden_note(diff: str) -> str:
    """Return the trailing hidden-site note, including its leading ellipsis."""
    marker = "\n… "
    index = diff.index(marker)
    return diff[index + 1 :]


def test_tuned_echo_bounds_are_pinned() -> None:
    # Measured against the root-session edit corpus: the window bound sits at
    # the context tail (p95 2.74KB, 1.0% clipped) and the hunk cap stays at the
    # measured knee (99.2% of edits are single-site).
    assert EDIT_WINDOW_MAX_BYTES == 2 * 1024
    assert EDIT_DIFF_MAX_SITES == 8
    assert EDIT_DIFF_NOTE_MAX_BYTES <= EDIT_WINDOW_MAX_BYTES


def test_context_window_honors_tuned_byte_bound_and_keeps_site_anchor() -> None:
    lines = [f"L{n}" + "x" * 400 for n in range(1, 6)]
    lines[2] = "site" + "y" * 400
    modified = "\n".join(lines) + "\n"

    text = render_context_window(
        modified, first_site=AppliedSite(line=3, old_text="old", new_text="site")
    )

    assert text is not None
    header, body = text.split("\n", 1)
    assert header == "Current content, lines 1–5:"
    marker = f"… [window truncated at {EDIT_WINDOW_MAX_BYTES} bytes]"
    assert marker in body
    # The truncated body stays within the tuned bound; only the marker sits on
    # top of it.
    assert len(body.encode("utf-8")) <= EDIT_WINDOW_MAX_BYTES + len(
        marker.encode("utf-8")
    )
    # The first site's line is mandatory, so the anchor survives the cut.
    assert "3→site" in body


def test_hidden_sites_are_named_by_current_line_number() -> None:
    sites = [
        AppliedSite(line=line, old_text="old", new_text="new") for line in range(1, 13)
    ]

    diff = render_applied_diff(sites)

    assert diff.count("@@ line") == EDIT_DIFF_MAX_SITES
    note = _hidden_note(diff)
    assert note.startswith("… 4 more replaced site(s) not shown (total 12).")
    assert "Hidden lines: 9, 10, 11, 12." in note
    assert "read_file on the named file" in note


def test_hidden_anchors_track_current_lines_after_insertions() -> None:
    original = "a\n" + "".join("tgt\nb\n" for _ in range(10))
    modified = "a\n" + "".join("T1\nT2\nb\n" for _ in range(10))

    sites = find_applied_sites(original, modified, "tgt", "T1\nT2", replace_all=True)

    assert [site.line for site in sites] == [2, 5, 8, 11, 14, 17, 20, 23, 26, 29]
    note = _hidden_note(render_applied_diff(sites))
    assert "Hidden lines: 26, 29." in note
    # The anchors point at the current text, not the pre-edit coordinates.
    current = modified.splitlines()
    assert current[26 - 1].startswith("T1")
    assert current[29 - 1].startswith("T1")


def test_hidden_anchors_are_not_lost_under_the_tuned_bounds() -> None:
    sites = [
        AppliedSite(line=line, old_text="old", new_text="new")
        for line in range(100, 148)
    ]

    note = _hidden_note(render_applied_diff(sites))

    expected = ", ".join(str(line) for line in range(108, 148))
    assert f"Hidden lines: {expected}." in note
    assert "(+" not in note


def test_hidden_note_itself_stays_within_the_echo_bound() -> None:
    sites = [
        AppliedSite(line=line * 7, old_text="old", new_text="new")
        for line in range(1, 501)
    ]

    note = _hidden_note(render_applied_diff(sites))

    assert len(note.encode("utf-8")) <= EDIT_DIFF_NOTE_MAX_BYTES
    assert note.startswith("… 492 more replaced site(s) not shown (total 500).")
    assert "Hidden lines: " in note
    # A genuinely unbounded hidden set is elided with an explicit remainder.
    assert "(+" in note
    assert "read_file on the named file" in note


@pytest.mark.asyncio
async def test_edit_tool_publishes_line_keyed_hidden_anchors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "f.txt"
    path.write_text("".join(f"tgt {index}\n" for index in range(12)))

    result = await collect_result(
        _make_edit().run(
            EditArgs(
                file_path="f.txt", old_string="tgt", new_string="TGT", replace_all=True
            )
        )
    )

    assert result.replaced_count == 12
    assert result.applied_diff is not None
    assert result.applied_diff.count("@@ line") == EDIT_DIFF_MAX_SITES
    note = _hidden_note(result.applied_diff)
    assert note.startswith("… 4 more replaced site(s) not shown (total 12).")
    assert "Hidden lines: 9, 10, 11, 12." in note
    current = path.read_text().splitlines()
    for line in (9, 10, 11, 12):
        assert current[line - 1] == f"TGT {line - 1}"
