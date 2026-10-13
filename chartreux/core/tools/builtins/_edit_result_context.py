"""Post-edit evidence for the edit tool's model-facing result.

After an edit lands, the model re-locates in the file because the old result
carried no post-edit evidence: the UI-only occurrence hints are computed from
the *pre*-replacement text (their line numbers go stale after any
newline-changing edit) and are excluded from the model dump. This module
computes the evidence that is published on ``EditResult`` instead, all of it
derived from the file as it stands *after* the applied edit:

- ``applied_diff`` — what the edit actually changed: every applied replacement
  site rendered as whole lines from the file (the replaced segment, then the
  replacement), each headed by the site's *current* (post-edit) line number.
  With ``replace_all`` every site is named; the rendering is capped at
  ``EDIT_DIFF_MAX_SITES`` hunks, ``EDIT_DIFF_MAX_LINES_PER_SIDE`` lines and
  ``EDIT_DIFF_MAX_BYTES_PER_SIDE`` bytes per side, with an elision marker
  stating what was cut.
- ``context`` — a numbered window of the current content around the first
  applied site, rendered in the same ``N→line`` style as ``read_file`` so the
  model can correlate it with reads. Line numbers point at the current text
  even after newline insertion/deletion and after ``replace_all``. The window
  is capped at ``EDIT_WINDOW_MAX_LINES`` lines, ``EDIT_WINDOW_MAX_LINE_BYTES``
  bytes per line (oversized lines are elided with a byte marker) and
  ``EDIT_WINDOW_MAX_BYTES`` bytes of rendered text; the first site's lines are
  always included.
- ``revision`` — the post-edit file state as a full-scope
  :class:`~chartreux.core.file_revision.FileRevision` (blake2b-128 over the
  raw bytes), computed by a full-scan re-read through the tool's own read
  path (:func:`chartreux.utils.io.read_lines_hashed`), never from host-side
  metadata and never from a second in-memory copy. Client-mediated writes
  (the tool I/O port serves the write, so the host may not have the file at
  all) cannot produce a revision and degrade gracefully to ``None`` — a
  documented degradation, not an error. A failed re-read after a successful
  host write also degrades to ``None`` rather than failing the edit.
- ``replaced_count`` — how many replacement sites were applied.

The window and diff are computed from the exact post-replacement content the
tool wrote (identical to what an independent re-read decodes back), so their
line numbers always agree with the file on disk. Failed edits raise before a
result is built and therefore publish no evidence at all.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

from chartreux.core.file_revision import FileRevision
from chartreux.utils.io import read_lines_hashed_async

#: Lines of context shown before/after the first edit site in the window.
EDIT_WINDOW_CONTEXT_LINES = 2
#: Hard cap on window lines (the replaced span counts toward it).
EDIT_WINDOW_MAX_LINES = 16
#: Soft cap on the rendered window text, in bytes.
EDIT_WINDOW_MAX_BYTES = 8 * 1024
#: Per-line cap inside the window; longer lines are elided with a marker.
EDIT_WINDOW_MAX_LINE_BYTES = 512
#: Max site hunks rendered in the applied diff.
EDIT_DIFF_MAX_SITES = 8
#: Max old/new lines rendered per site hunk.
EDIT_DIFF_MAX_LINES_PER_SIDE = 12
#: Max bytes rendered per side (old or new) of one site hunk.
EDIT_DIFF_MAX_BYTES_PER_SIDE = 4 * 1024


class AppliedSite(NamedTuple):
    """One applied replacement, located in current (post-edit) coordinates.

    ``line`` is the current line number of the replacement's first line;
    ``old_text``/``new_text`` are the replaced segment and the replacement,
    each expanded to whole lines of the pre-edit and post-edit file
    respectively (empty when that side is an empty segment).
    """

    line: int
    old_text: str
    new_text: str


class EditEvidence(NamedTuple):
    """The post-edit evidence published on the model-facing ``EditResult``."""

    applied_diff: str
    context: str | None
    revision: FileRevision | None
    replaced_count: int


def _expand_whole_lines(text: str, start: int, end: int) -> str:
    """Expand the segment ``text[start:end]`` to whole lines.

    A segment ending on a line boundary keeps its trailing newline (the
    expansion covers exactly the lines the segment touches); otherwise the
    expansion runs to the end of the segment's last line.
    """
    line_start = text.rfind("\n", 0, start) + 1
    if end > start and text[end - 1] == "\n":
        return text[line_start:end]
    line_end = text.find("\n", end)
    if line_end == -1:
        line_end = len(text)
    return text[line_start:line_end]


def _truncate_to_bytes(text: str, budget: int) -> tuple[str, bool]:
    """Return ``(prefix, complete)`` for up to ``budget`` UTF-8 bytes.

    The prefix never splits a character, so no content is lost at the cut.
    """
    if len(text.encode("utf-8")) <= budget:
        return text, True
    low, high = 0, len(text)
    while low < high:
        mid = (low + high + 1) // 2
        if len(text[:mid].encode("utf-8")) <= budget:
            low = mid
        else:
            high = mid - 1
    return text[:low], False


def find_applied_sites(
    original: str, modified: str, old_string: str, new_string: str, *, replace_all: bool
) -> list[AppliedSite]:
    """Locate every applied replacement in current (post-edit) coordinates.

    Sites mirror ``str.replace`` semantics: non-overlapping occurrences in
    file order, all of them for ``replace_all`` and only the first otherwise.
    Each site's current line number accounts for the newline delta contributed
    by every earlier site, so it stays correct after newline insertion or
    deletion and across multiple replacements in one call.
    """
    if not old_string:
        # Populating an empty file: a single site at the top of the file.
        if not new_string:
            return []
        return [
            AppliedSite(
                line=1,
                old_text="",
                new_text=_expand_whole_lines(modified, 0, len(new_string)),
            )
        ]
    char_shift = len(new_string) - len(old_string)
    line_shift = new_string.count("\n") - old_string.count("\n")
    sites: list[AppliedSite] = []
    pos = original.find(old_string)
    while pos != -1:
        index = len(sites)
        post_pos = pos + index * char_shift
        sites.append(
            AppliedSite(
                line=original.count("\n", 0, pos) + 1 + index * line_shift,
                old_text=_expand_whole_lines(original, pos, pos + len(old_string)),
                new_text=(
                    ""
                    if not new_string
                    else _expand_whole_lines(
                        modified, post_pos, post_pos + len(new_string)
                    )
                ),
            )
        )
        if not replace_all:
            break
        pos = original.find(old_string, pos + len(old_string))
    return sites


def _bounded_side(text: str) -> list[str]:
    """Cap one diff side (old or new) at the stated line and byte bounds.

    Trailing lines past the line cap are dropped first, then lines past the
    byte cap; a single line that overflows the byte cap alone keeps a
    byte-safe prefix. Elided content is reported in a trailing marker line.
    """
    if not text:
        return []
    all_lines = text.splitlines()
    shown = all_lines[:EDIT_DIFF_MAX_LINES_PER_SIDE]
    elided_lines = len(all_lines) - len(shown)
    elided_bytes = 0
    while shown:
        if len("\n".join(shown).encode("utf-8")) <= EDIT_DIFF_MAX_BYTES_PER_SIDE:
            break
        elided_bytes += len(shown.pop().encode("utf-8")) + 1
        elided_lines += 1
    if not shown:
        prefix, _complete = _truncate_to_bytes(text, EDIT_DIFF_MAX_BYTES_PER_SIDE)
        shown = [prefix]
        elided_bytes = len(text.encode("utf-8")) - len(prefix.encode("utf-8"))
    notes = []
    if elided_lines:
        notes.append(f"{elided_lines} line(s)")
    if elided_bytes:
        notes.append(f"{elided_bytes} byte(s)")
    if notes:
        shown.append(f"… [+{', '.join(notes)}]")
    return shown


def render_applied_diff(sites: list[AppliedSite]) -> str:
    """Render the applied change(s) as a bounded ``-``/``+`` hunk list.

    Every site is named (its current line number heads the hunk) up to
    ``EDIT_DIFF_MAX_SITES``; further sites are summarized in a trailing note
    so ``replace_all`` evidence always accounts for every applied site.
    """
    blocks: list[str] = []
    for site in sites[:EDIT_DIFF_MAX_SITES]:
        lines = [f"@@ line {site.line} (current) @@"]
        lines.extend(f"- {line}" for line in _bounded_side(site.old_text))
        lines.extend(f"+ {line}" for line in _bounded_side(site.new_text))
        blocks.append("\n".join(lines))
    diff = "\n".join(blocks)
    hidden = len(sites) - min(len(sites), EDIT_DIFF_MAX_SITES)
    if hidden > 0:
        diff += f"\n… {hidden} more replaced site(s) not shown (total {len(sites)})."
    return diff


def _bounded_window_line(line: str) -> tuple[str, int]:
    """Cap one window line at ``EDIT_WINDOW_MAX_LINE_BYTES`` bytes."""
    encoded = len(line.encode("utf-8"))
    if encoded <= EDIT_WINDOW_MAX_LINE_BYTES:
        return line, 0
    prefix, _complete = _truncate_to_bytes(line, EDIT_WINDOW_MAX_LINE_BYTES)
    return prefix, encoded - len(prefix.encode("utf-8"))


def render_context_window(
    modified: str, *, first_site: AppliedSite | None
) -> str | None:
    """Render the numbered post-edit window around the first applied site.

    The window shows the current content (``modified`` — exactly what the
    tool wrote) with current line numbers, in ``read_file``'s ``N→line``
    style. ``None`` when there is no site to anchor the window.
    """
    if first_site is None:
        return None
    lines = modified.splitlines()
    if not lines:
        return "The file is now empty."
    span = first_site.new_text.count("\n") + 1 if first_site.new_text else 1
    start = max(1, first_site.line - EDIT_WINDOW_CONTEXT_LINES)
    end = min(len(lines), first_site.line + span - 1 + EDIT_WINDOW_CONTEXT_LINES)
    window = lines[start - 1 : end][:EDIT_WINDOW_MAX_LINES]
    site_offset = min(first_site.line - start, len(window) - 1)
    rendered: list[str] = []
    used = 0
    for offset, line in enumerate(window):
        number = start + offset
        text, elided = _bounded_window_line(line)
        if elided:
            text = f"{text} …[+{elided} bytes]"
        entry = f"{str(number).rjust(9)}\u2192{text}"
        cost = len(entry.encode("utf-8")) + 1
        mandatory = offset <= site_offset + span - 1
        if rendered and not mandatory and used + cost > EDIT_WINDOW_MAX_BYTES:
            rendered.append(f"… [window truncated at {EDIT_WINDOW_MAX_BYTES} bytes]")
            break
        rendered.append(entry)
        used += cost
    header = f"Current content, lines {start}\u2013{start + len(window) - 1}:"
    return f"{header}\n" + "\n".join(rendered)


async def _read_post_edit_revision(file_path: Path) -> FileRevision | None:
    """Full-scope content revision of the post-edit file, via the read path.

    A full scan through the same read machinery ``read_file`` uses computes
    the blake2b-128 identifier over every raw byte now on disk, plus the
    byte and line totals — never host-side metadata. Any read failure
    degrades to ``None``: the edit already succeeded, so its evidence must
    not fail with it.
    """
    try:
        read = await read_lines_hashed_async(
            file_path,
            start_line=1,
            limit=1,
            max_bytes=EDIT_WINDOW_MAX_BYTES,
            full_scan=True,
        )
    except OSError:
        return None
    if read.revision is None or read.raw_bytes is None:
        return None
    return FileRevision(
        revision=read.revision, size_bytes=read.raw_bytes, line_count=read.total_lines
    )


async def build_edit_evidence(
    *,
    file_path: Path,
    original: str,
    modified: str,
    old_string: str,
    new_string: str,
    replace_all: bool,
    host_written: bool,
) -> EditEvidence:
    """Compute the post-edit evidence published in the model-facing result.

    The diff and the context window are computed from ``modified`` — the
    exact post-replacement content the tool wrote. The revision is computed
    only for host-written files, by re-reading the file through the tool's
    own read path; client-mediated writes (``host_written=False``) leave it
    ``None``.
    """
    sites = find_applied_sites(
        original, modified, old_string, new_string, replace_all=replace_all
    )
    revision = await _read_post_edit_revision(file_path) if host_written else None
    return EditEvidence(
        applied_diff=render_applied_diff(sites),
        context=render_context_window(modified, first_site=sites[0] if sites else None),
        revision=revision,
        replaced_count=len(sites),
    )
