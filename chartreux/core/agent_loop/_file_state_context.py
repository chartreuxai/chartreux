"""Working-set file state wiring: state blocks, compaction replay, spawn handoff.

This module connects the standalone workspace-state pieces
(:mod:`chartreux.core.file_state`, :mod:`chartreux.core.file_outline`,
:mod:`chartreux.core.file_revision`) to the agent loop. A worker that has
already observed a file should not re-read it just to learn where its symbols
live, and it should learn about *other* workers' edits to its working set
without re-reading either. The context records one observation per successful
authorized file interaction (read / edit / write results), publishes every
observed revision to the dispatcher-level revision registry, and renders
*state blocks* — bounded, append-only messages that carry a changed path, its
new revision, and the current symbol outline.

State-block strategy (cache guard): a block is an injected user message
appended as the LAST message of the transcript, exactly once per working-set
revision change. Blocks are immutable: nothing in this module ever rewrites,
reorders, or removes an appended block, so the request prefix the provider
caches stays valid across turns. The bounded count is carried by the STORE,
not by message history: the store tracks at most ``MAX_TRACKED_FILES`` paths
and is trimmed only at compaction boundaries, where the compaction envelope
rewrites the model-facing prefix anyway (a prefix rewrite the cache already
absorbs). Blocks are deliberately NEVER evicted from message history between
compactions — dropping the oldest block would rewrite the message prefix and
bust the prefix cache mid-session — so block history is bounded by compaction
itself, exactly like every other message in the transcript. The one place a
drop is cache-safe is the compaction boundary, and there the store's
``trim_to_cap`` performs it.

Membership: a path enters the working set only through (a) the spawn handoff —
the parent's explicitly named working set, deep-copied at child creation — or
(b) a successful authorized file interaction the session itself performed.
There is no repository crawling. Denied or unauthorized reads never produce a
tool result, so they enroll nothing; sensitive paths (per the observing tool's
configured patterns, defaulting to ``DEFAULT_SENSITIVE_PATTERNS``) enroll their
path and revision but never an outline. Client-mediated reads (the tool I/O
port) and bounded window reads report no revision: they enroll the path and
its outline but are *non-vouching* — they never trigger a block, never
publish to the registry, and never erase a vouched baseline (the stored
revision and outline survive, so external-bump detection stays armed).

Compaction: the store rides the compaction envelope structurally (replayed on
resume), and each compaction boundary additionally renders the bounded
working-set summary — paths, last-seen revisions, current outlines — into the
boundary message's *content*, the model-visible part of the boundary (the
envelope itself is not sent to the provider). The model therefore sees the
working set and its revisions right after compaction, without re-reading the
files.

Publication ordering: the loop observes tool results after awaited post-tool
finalization, so an older read result can be published after a newer revision
is already known. Every vouched observation therefore publishes to the
registry with a per-path sequence (numbered per session at this seam), and the
registry rejects a publication whose sequence is behind the path's stored one
— a superseded, delayed publication never regresses tree truth.

Degradation is first-class throughout: a missing registry (a loop constructed
outside the dispatcher) disables external-bump reporting but keeps
own-observation blocks; unreadable, non-Python, unparsable, or oversized files
carry no outline; every observation path is bounded (the outline caps its
rendered size) and never raises into the turn.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import os
from typing import Protocol

from pydantic import BaseModel

from chartreux.core.file_outline import FileOutline, outline_file
from chartreux.core.file_revision import FileRevision
from chartreux.core.file_state import FileStateEntry, FileStateStore
from chartreux.core.llm_models import LLMMessage, Role
from chartreux.core.tools.builtins.edit import EditResult
from chartreux.core.tools.builtins.read_file import ReadFileResult
from chartreux.core.tools.builtins.write_file import WriteFileResult
from chartreux.core.tools.utils import (
    DEFAULT_SENSITIVE_PATTERNS,
    matches_sensitive_pattern,
)

__all__ = [
    "FILE_STATE_BLOCK_TAG",
    "MAX_HANDOFF_BYTES",
    "FileStateContext",
    "RevisionRegistryPort",
    "render_file_state_block",
    "render_file_state_compaction_summary",
    "render_file_state_handoff",
    "render_file_state_summary",
]

#: Tag wrapping every working-set state block (and the spawn handoff).
FILE_STATE_BLOCK_TAG = "file_state"

#: Upper bound on a rendered whole-working-set summary (the spawn handoff and
#: the compaction-boundary section); the file list is truncated (with a note)
#: rather than allowed to grow without bound.
MAX_HANDOFF_BYTES = 32 * 1024

_BLOCK_OPEN = f"<{FILE_STATE_BLOCK_TAG} "
_HANDOFF_OPEN = f"<{FILE_STATE_BLOCK_TAG}_handoff>"
_HANDOFF_CLOSE = f"</{FILE_STATE_BLOCK_TAG}_handoff>"
_COMPACTION_OPEN = f"<{FILE_STATE_BLOCK_TAG}_summary>"
_COMPACTION_CLOSE = f"</{FILE_STATE_BLOCK_TAG}_summary>"
_BLOCK_CLOSE = f"</{FILE_STATE_BLOCK_TAG}>"
_BLOCK_NOTE = (
    "Working-set file state block (metadata only — re-read the file for its content)."
)
_EXTERNAL_NOTE = (
    "Another worker (or an outside change) published this revision after this "
    "session last observed the file; the outline below is the current on-disk state."
)
_UNKNOWN_NOTE = (
    "The observing read could not provide a content revision (client-mediated "
    "read); the outline below describes the observed content."
)
_NO_OUTLINE_NOTE = (
    "No symbol outline is available (the file is not Python, does not parse, is "
    "oversized, or is restricted); re-read the file for its content."
)


class RevisionRegistryPort(Protocol):
    """Structural port for the dispatcher-level revision registry.

    The concrete implementation
    (:class:`chartreux.app_server._file_revision_registry.FileRevisionRegistry`)
    lives in the app server; core consumes it through this protocol so the
    session tree can share one instance without a core -> app_server import.
    """

    def publish(
        self, path: str, revision: FileRevision, *, sequence: int | None = None
    ) -> None:
        """Publish the last-known revision for ``path``.

        ``sequence`` is the publication's per-path ordering token (numbered
        at the observation seam); the registry rejects a publication whose
        sequence is behind the path's stored one — a superseded, delayed
        observation — instead of overwriting newer truth. ``None`` carries
        no ordering claim and is always accepted.
        """
        ...

    def known_revision(self, path: str) -> FileRevision | None:
        """Return the last-known revision for ``path``, or ``None``."""
        ...

    def external_bump(
        self, path: str, last_seen: FileRevision | str | None
    ) -> FileRevision | None:
        """Return the tree-wide revision when it moved past ``last_seen``."""
        ...


@dataclass(slots=True)
class _PendingBlock:
    """One rendered-but-not-yet-appended state block."""

    path: str
    revision: FileRevision | None
    outline: FileOutline | None
    external: bool


def _revision_label(revision: FileRevision | None) -> str:
    return revision.revision if revision is not None else "unknown"


def _outline_lines(outline: FileOutline | None) -> list[str]:
    if outline is None:
        return []
    return outline.render().splitlines()


def _is_sensitive(path: str, patterns: Sequence[str]) -> bool:
    effective = list(patterns) or list(DEFAULT_SENSITIVE_PATTERNS)
    return matches_sensitive_pattern(path, effective)


def _read_revision(
    revision: str | None, *, line_count: int | None
) -> FileRevision | None:
    """Build a revision from a read result's revision string.

    ``size_bytes=0`` is the read-discovered sentinel ("size unknown"): reads
    report the content revision but not the consumed byte count. Every
    comparison (:meth:`FileRevision.matches`) uses the revision string and
    scope, so read- and edit-computed revisions compare equal for equal
    content.
    """
    if revision is None:
        return None
    return FileRevision(revision=revision, size_bytes=0, line_count=line_count)


def _revision_changed(
    known: FileRevision | None, observed: FileRevision | None
) -> bool:
    """True when an observation moves the stored revision of a path.

    An observation without revision information (``observed is None``, a
    client-mediated or windowed read) is never a change: it appends nothing
    and, per the store contract, never disturbs the stored baseline.
    """
    if observed is None:
        return False
    if known is None:
        return True
    return not known.matches(observed)


def render_file_state_block(
    *,
    path: str,
    revision: FileRevision | None,
    outline: FileOutline | None,
    external: bool,
) -> str:
    """Render one state block: changed path, new revision, bounded outline."""
    source = "external" if external else "self"
    lines = [
        f"{_BLOCK_OPEN}path={path!r} revision={_revision_label(revision)!r} "
        f"source={source!r}>",
        _BLOCK_NOTE,
    ]
    if revision is None:
        lines.append(_UNKNOWN_NOTE)
    elif external:
        lines.append(_EXTERNAL_NOTE)
    outline_lines = _outline_lines(outline)
    if outline_lines:
        lines.append(
            "Current symbol outline (line ranges refer to the current file state):"
        )
        lines.extend(outline_lines)
    else:
        lines.append(_NO_OUTLINE_NOTE)
    lines.append(_BLOCK_CLOSE)
    return "\n".join(lines)


def render_file_state_summary(
    store: FileStateStore,
    *,
    open_tag: str,
    close_tag: str,
    heading: str,
    truncation_note: str,
    max_bytes: int = MAX_HANDOFF_BYTES,
) -> str | None:
    """Render a bounded whole-working-set summary, or ``None`` if empty.

    The summary lists each tracked path with its last-seen revision and
    current outline — metadata only. The rendered size is bounded by
    ``max_bytes``; the file list is truncated with ``truncation_note`` rather
    than allowed to grow without bound.
    """
    if not store.entries:
        return None
    lines = [open_tag, heading]
    budget = max_bytes - sum(len(line) + 1 for line in lines)
    omitted = 0
    for path in store.paths():
        entry = store.entries[path]
        item_lines = [f"- {path} (revision {_revision_label(entry.revision)}):"]
        outline_lines = _outline_lines(entry.outline)
        if outline_lines:
            item_lines.extend(f"    {line}" for line in outline_lines)
        else:
            item_lines.append("    no symbol outline")
        cost = sum(len(line) + 1 for line in item_lines)
        if cost > budget:
            omitted += 1
            continue
        budget -= cost
        lines.extend(item_lines)
    if omitted:
        lines.append(f"- … and {omitted} more tracked files ({truncation_note})")
    lines.append(close_tag)
    return "\n".join(lines)


def render_file_state_handoff(store: FileStateStore) -> str | None:
    """Render the spawn handoff for a whole working set, or ``None`` if empty.

    The handoff lists each tracked path with its last-seen revision and current
    outline — metadata only. The rendered size is bounded by
    :data:`MAX_HANDOFF_BYTES`; the file list is truncated with a note rather
    than allowed to grow without bound.
    """
    return render_file_state_summary(
        store,
        open_tag=_HANDOFF_OPEN,
        close_tag=_HANDOFF_CLOSE,
        heading=(
            "Working-set handoff at spawn: files this session tree has already "
            "observed, with their last-seen content revisions and current symbol "
            "outlines. Metadata only — re-read a file for its content."
        ),
        truncation_note="handoff budget reached",
    )


def render_file_state_compaction_summary(store: FileStateStore) -> str | None:
    """Render the working-set summary for a compaction boundary, or ``None``.

    The compaction boundary rewrites the model-facing prefix, so the bounded
    working-set metadata (paths, last-seen revisions, current outlines) is
    rendered into the boundary message's content once, at that boundary: the
    model sees which files are in the working set and at what revisions
    without re-reading them. The shape mirrors the spawn handoff and is
    bounded by :data:`MAX_HANDOFF_BYTES`.
    """
    return render_file_state_summary(
        store,
        open_tag=_COMPACTION_OPEN,
        close_tag=_COMPACTION_CLOSE,
        heading=(
            "Working-set file state at this compaction: files this session has "
            "already observed, with their last-seen content revisions and "
            "current symbol outlines. Metadata only — re-read a file for its "
            "content. Unchanged files produce no new state block, so this list "
            "is the authoritative working set as of the compaction."
        ),
        truncation_note="summary budget reached",
    )


class FileStateContext:
    """Wiring between tool results, the working-set store, and the registry.

    The context owns the session-side :class:`FileStateStore` and an optional
    :class:`RevisionRegistryPort`. Observations update the store and publish to
    the registry immediately; the rendered block messages are produced by
    :meth:`flush_blocks` at the loop's append points (end of a tool batch, start
    of an LLM turn), so a block is always the last message at append time.
    """

    def __init__(
        self,
        *,
        store: FileStateStore | None = None,
        registry: RevisionRegistryPort | None = None,
    ) -> None:
        self.store = store if store is not None else FileStateStore()
        self.registry = registry
        self._pending: list[_PendingBlock] = []
        # Per-path publication sequences (session-lifetime ordering tokens
        # for registry publications; see _next_sequence). Deliberately not
        # part of the working-set store: a history clear keeps them so
        # post-clear publications are not falsely superseded.
        self._publish_sequences: dict[str, int] = {}

    # -- observation intake -------------------------------------------------

    def observe_tool_result(
        self,
        tool_name: str,
        result: BaseModel,
        *,
        sensitive_patterns: Sequence[str] = (),
    ) -> None:
        """Record one successful tool result (read / edit / write).

        Only the file-interaction tools enroll paths; anything else is a no-op.
        The result models are matched structurally by type, so an unknown tool
        or an unexpected result shape degrades to no observation.
        """
        if isinstance(result, ReadFileResult):
            self.observe_read(
                result.file_path,
                result.revision,
                line_count=result.total_lines,
                sensitive_patterns=sensitive_patterns,
            )
        elif isinstance(result, EditResult):
            self.observe_edit(
                result.file, result.revision, sensitive_patterns=sensitive_patterns
            )
        elif isinstance(result, WriteFileResult):
            self.observe_write(
                result.file_path, result.revision, sensitive_patterns=sensitive_patterns
            )

    def observe_read(
        self,
        path: str,
        revision: str | None,
        *,
        line_count: int | None = None,
        sensitive_patterns: Sequence[str] = (),
    ) -> None:
        """Record a read result: path, read-time revision, current outline.

        ``revision`` is the full-read revision the read path vouched, or
        ``None`` when the read cannot vouch one (a client-mediated port read
        or a bounded window read) — a non-vouching observation that enrolls
        the path but never triggers a block or a publication.
        """
        self._observe(
            path,
            _read_revision(revision, line_count=line_count),
            sensitive_patterns=sensitive_patterns,
        )

    def observe_edit(
        self,
        path: str,
        revision: FileRevision | None,
        *,
        sensitive_patterns: Sequence[str] = (),
    ) -> None:
        """Record an edit result: path and post-edit revision."""
        self._observe(path, revision, sensitive_patterns=sensitive_patterns)

    def observe_write(
        self,
        path: str,
        revision: FileRevision | None,
        *,
        sensitive_patterns: Sequence[str] = (),
    ) -> None:
        """Record a write result: path and post-write revision.

        ``revision`` is the full content revision of the written file, or
        ``None`` for a client-mediated write (the tool I/O port performed it,
        so the host cannot attest the file's bytes — the same degradation as
        a client-mediated edit). The store/registry never receive a fabricated
        attested revision for a file whose bytes the host cannot attest; a
        ``None`` revision is a non-vouching observation, which never erases a
        vouched baseline.
        """
        self._observe(path, revision, sensitive_patterns=sensitive_patterns)

    def _observe(
        self,
        path: str,
        revision: FileRevision | None,
        *,
        sensitive_patterns: Sequence[str],
    ) -> None:
        path = os.path.normpath(path)
        if not path:
            return
        known = self.store.known_revision(path)
        changed = _revision_changed(known, revision)
        outline = (
            None if _is_sensitive(path, sensitive_patterns) else outline_file(path)
        )
        self.store.record_observation(path, revision=revision, outline=outline)
        if revision is not None and self.registry is not None:
            self.registry.publish(path, revision, sequence=self._next_sequence(path))
        if changed:
            self._pending.append(
                _PendingBlock(
                    path=path, revision=revision, outline=outline, external=False
                )
            )

    def _next_sequence(self, path: str) -> int:
        """Advance and return this session's publication sequence for ``path``.

        The per-path sequence orders this session's publications against the
        registry's stored ordering, so a delayed publication of an older
        observation is rejected as superseded instead of regressing tree
        truth. The counters are session-lifetime (a history clear keeps
        them), so publications after a clear are not falsely superseded by
        pre-clear registry ordering.
        """
        sequence = self._publish_sequences.get(path, 0) + 1
        self._publish_sequences[path] = sequence
        return sequence

    # -- flush (append points) ----------------------------------------------

    def flush_blocks(self) -> list[LLMMessage]:
        """Render pending blocks and adopt external bumps; return the messages.

        Own observations are rendered first (observation order), then one block
        per tracked path whose registry revision moved past this worker's
        baseline (an external bump): the bump is reported, a fresh outline is
        computed, and the store adopts the bumped revision so the next flush
        does not re-report it. Returns an empty list when nothing changed — an
        unchanged turn appends nothing.
        """
        blocks = list(self._pending)
        self._pending.clear()
        if self.registry is not None:
            for path in self.store.paths():
                bumped = self.registry.external_bump(
                    path, self.store.known_revision(path)
                )
                if bumped is None:
                    continue
                outline = None if _is_sensitive(path, ()) else outline_file(path)
                self.store.record_observation(path, revision=bumped, outline=outline)
                blocks.append(
                    _PendingBlock(
                        path=path, revision=bumped, outline=outline, external=True
                    )
                )
        return [
            LLMMessage(
                role=Role.user,
                content=render_file_state_block(
                    path=block.path,
                    revision=block.revision,
                    outline=block.outline,
                    external=block.external,
                ),
                injected=True,
            )
            for block in blocks
        ]

    # -- spawn handoff --------------------------------------------------------

    def handoff_message(self) -> LLMMessage | None:
        """The spawn handoff block for this working set, or ``None`` if empty."""
        content = render_file_state_handoff(self.store)
        if content is None:
            return None
        return LLMMessage(role=Role.user, content=content, injected=True)

    # -- compaction -----------------------------------------------------------

    def merge_replayed(self, replayed: FileStateStore) -> None:
        """Merge a transcript-replayed store into the live one (resume).

        Per path, the entry whose revision matches the registry (tree truth)
        wins; with no tree truth, a known envelope revision beats an unknown
        baseline; a ``None`` envelope revision never wipes a known baseline.
        """
        for path, entry in replayed.entries.items():
            current = self.store.entry(path)
            if current is None:
                self._adopt(path, entry)
                continue
            registry_revision = (
                self.registry.known_revision(path)
                if self.registry is not None
                else None
            )
            if registry_revision is None:
                if entry.revision is not None:
                    self._adopt(path, entry)
                continue
            entry_current = entry.revision is not None and entry.revision.matches(
                registry_revision
            )
            live_current = current.revision is not None and current.revision.matches(
                registry_revision
            )
            if entry_current and not live_current:
                self._adopt(path, entry)

    def _adopt(self, path: str, entry: FileStateEntry) -> None:
        self.store.record_observation(
            path, revision=entry.revision, outline=entry.outline
        )

    # -- lifecycle ------------------------------------------------------------

    def reset(self) -> None:
        """Drop the working set and pending blocks (an explicit history clear).

        The registry is dispatcher-level tree truth and is never cleared here;
        the per-path publication sequences survive too, so publications after
        a clear are not falsely superseded by pre-clear registry ordering.
        """
        self.store = FileStateStore()
        self._pending.clear()
