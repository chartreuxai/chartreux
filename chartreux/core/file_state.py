"""Working-set file state: the session-side view of the files a worker touches.

The store maps path -> {last-seen revision, symbol outline} so a worker can be
served bounded file state at spawn and on revision change instead of re-reading
working-set files it has already seen. It pairs the two standalone contracts of
the workspace-state work: content revisions
(:mod:`chartreux.core.file_revision`) and symbol outlines
(:mod:`chartreux.core.file_outline`).

Revision semantics: an entry's ``revision`` is the content revision of the most
recently *vouched* file state (a full read, or an edit/write result). Tool I/O
port reads and bounded window reads cannot vouch a revision (they report
``revision=None``); such a non-vouching observation never erases vouched
knowledge — it leaves the stored revision and outline intact, so the baseline
survives the dominant windowed-read pattern. An entry only carries
``revision=None`` ("revision unknown") while no observation has vouched one,
and the store never claims a revision an observation did not vouch for.
Staleness queries fail safe: an unknown stored revision (no vouched
observation yet) can never be judged stale; the state block reports
"revision unknown" instead.

Outline invariant: the stored outline always describes the stored revision's
state. An observation whose revision differs from the stored one drops the
stored outline unless the same observation supplies a replacement
(:meth:`FileStateStore.record_observation` records both atomically).

Compaction: the store serializes into the compaction envelope exactly like
``TodoState`` — :meth:`FileStateStore.to_envelope` rides in a
``PersistedToolResult`` output on a compaction-boundary message, and
:meth:`FileStateStore.replay` restores it structurally from a transcript.

Bounded count: at most :data:`MAX_TRACKED_FILES` entries are intended. The
store never evicts on its own — the work order requires eviction only at
compaction boundaries — so eviction is explicit (:meth:`evict_oldest`,
:meth:`trim_to_cap`) and the caller decides when. Insertion order is recency
order (re-recording a path refreshes it), so the oldest-first drop is a
single ``dict`` pop.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from chartreux.core.file_outline import FileOutline
from chartreux.core.file_revision import FileRevision

if TYPE_CHECKING:
    from chartreux.core.llm_models import LLMMessage

__all__ = ["MAX_TRACKED_FILES", "FileStateEntry", "FileStateStore"]

#: Upper bound on tracked working-set files. The store itself never evicts;
#: AM-D2 calls :meth:`FileStateStore.trim_to_cap` at compaction boundaries.
MAX_TRACKED_FILES = 64

_ENVELOPE_KEY = "file_state"


class FileStateEntry(BaseModel):
    """One tracked file: last-seen revision and symbol outline.

    ``revision`` is the content revision of the most recently vouched file
    state, or ``None`` while no observation has vouched one (tool I/O port
    reads and bounded window reads publish no revision). ``outline``
    describes the file state the entry's revision identifies; it is dropped
    whenever a revision transition is recorded without a replacement
    outline, and it survives non-vouching observations untouched.
    """

    path: str
    revision: FileRevision | None = None
    outline: FileOutline | None = None


class FileStateStore(BaseModel):
    """Bounded working-set state: path -> {last-seen revision, outline}."""

    entries: dict[str, FileStateEntry] = Field(default_factory=dict)

    def record_observation(
        self, path: str, *, revision: FileRevision | None, outline: FileOutline | None
    ) -> FileStateEntry:
        """Record one observed file state (a read or an edit result).

        ``revision`` is the observed content revision, or ``None`` when the
        observation cannot vouch one (tool I/O port reads, bounded window
        reads). ``outline`` is the symbol outline computed from the observed
        content, or ``None`` when the file is not outlined (non-Python,
        unparsable, oversized).

        The stored outline always describes the stored revision's state: an
        observation whose revision differs from the stored one drops the
        stored outline unless it supplies a replacement; re-observing the
        same revision keeps it.

        A non-vouching observation (``revision=None``) never erases vouched
        knowledge: on an existing entry it leaves the stored revision and the
        stored outline intact and only refreshes recency. Wiping the baseline
        on a windowed read of an unchanged file would make the next full read
        report a phantom revision change and would disable external-bump
        detection for the path. A fresh entry enrolls the path with
        ``revision=None`` ("revision unknown") and the supplied outline.
        """
        entry = self.entries.get(path)
        if entry is None:
            entry = FileStateEntry(path=path, revision=revision, outline=outline)
            self.entries[path] = entry
            return entry
        if revision is None:
            # Non-vouching observation (a port read or a bounded window): it
            # cannot identify the file state, so it must not erase previously
            # vouched knowledge — the stored revision and its outline survive
            # untouched (recency refresh only).
            self._refresh(path, entry)
            return entry
        if not _same_revision(entry.revision, revision):
            entry.outline = None
        entry.revision = revision
        if outline is not None:
            entry.outline = outline
        self._refresh(path, entry)
        return entry

    def record_revision(
        self, path: str, revision: FileRevision | None
    ) -> FileStateEntry:
        """Record an observed content revision (a read or an edit result).

        Convenience for :meth:`record_observation` without an outline: a
        revision transition drops the stored outline, because the post-edit
        or externally changed content no longer matches it. Passing
        ``None`` records a non-vouching observation, which never erases a
        vouched baseline (see :meth:`record_observation`).
        """
        return self.record_observation(path, revision=revision, outline=None)

    def record_outline(self, path: str, outline: FileOutline | None) -> FileStateEntry:
        """Record an observed symbol outline without touching the revision.

        The outline must have been computed from the content of the stored
        revision's observation (or of a no-revision observation when the
        stored revision is unknown).
        """
        entry = self.entries.get(path)
        if entry is None:
            entry = FileStateEntry(path=path, outline=outline)
            self.entries[path] = entry
            return entry
        entry.outline = outline
        self._refresh(path, entry)
        return entry

    def entry(self, path: str) -> FileStateEntry | None:
        """Return the tracked entry for ``path``, or ``None``."""
        return self.entries.get(path)

    def known_revision(self, path: str) -> FileRevision | None:
        """Return the last-known content revision for ``path``, or ``None``."""
        entry = self.entries.get(path)
        return entry.revision if entry is not None else None

    def outline_for(self, path: str) -> FileOutline | None:
        """Return the recorded symbol outline for ``path``, or ``None``."""
        entry = self.entries.get(path)
        return entry.outline if entry is not None else None

    def is_stale(self, path: str, revision: FileRevision | str | None) -> bool:
        """True when the known revision differs from ``revision``.

        Fail-safe: an unknown stored revision (no vouched observation yet)
        or a ``None`` probe can never be judged stale — the state block
        reports "revision unknown" instead.
        """
        entry = self.entries.get(path)
        if entry is None or entry.revision is None or revision is None:
            return False
        return not entry.revision.matches(revision)

    def paths(self) -> list[str]:
        """Return the tracked paths, least-recently-observed first."""
        return list(self.entries)

    def evict_oldest(self, count: int = 1) -> list[str]:
        """Drop the ``count`` least-recently-observed entries, oldest first.

        Returns the evicted paths (fewer when the store runs dry). This is
        the only eviction primitive; the store never calls it implicitly.
        """
        evicted: list[str] = []
        for _ in range(max(0, count)):
            if not self.entries:
                break
            path = next(iter(self.entries))
            self.entries.pop(path, None)
            evicted.append(path)
        return evicted

    def trim_to_cap(self) -> list[str]:
        """Evict oldest-first until at most ``MAX_TRACKED_FILES`` remain.

        AM-D2 calls this at compaction boundaries; the store itself never
        evicts implicitly, so a long turn cannot silently drop state. Returns
        the evicted paths.
        """
        return self.evict_oldest(len(self.entries) - MAX_TRACKED_FILES)

    def is_over_cap(self) -> bool:
        """True when more than ``MAX_TRACKED_FILES`` entries are tracked."""
        return len(self.entries) > MAX_TRACKED_FILES

    def to_envelope(self) -> dict[str, Any]:
        """Serialize for the compaction envelope (metadata only).

        Mirrors the ``todo_state`` envelope: the payload rides in a
        ``PersistedToolResult`` output on a compaction-boundary message.
        """
        return {_ENVELOPE_KEY: self.model_dump(mode="json")}

    @classmethod
    def from_envelope(cls, payload: Mapping[str, Any]) -> FileStateStore | None:
        """Rebuild from a compaction envelope payload.

        Returns ``None`` when the payload carries no state or a malformed
        one — a corrupt envelope degrades to no state, never an exception.
        """
        state = payload.get(_ENVELOPE_KEY)
        if not isinstance(state, dict):
            return None
        try:
            return cls.model_validate(state)
        except (TypeError, ValueError):
            return None

    @classmethod
    def replay(cls, messages: Sequence[LLMMessage]) -> FileStateStore:
        """Restore the working-set state from a transcript.

        Structural replay across compaction, mirroring ``TodoState.replay``:
        the last compaction-boundary envelope in the transcript wins.
        Observations recorded after that boundary are not in the transcript,
        so the caller merges live state on top.
        """
        store = cls()
        for message in messages:
            if message.context_boundary != "compaction":
                continue
            if message.tool_result is None:
                continue
            restored = cls.from_envelope(message.tool_result.output)
            if restored is not None:
                store = restored
        return store

    def __len__(self) -> int:
        return len(self.entries)

    def __contains__(self, path: object) -> bool:
        return path in self.entries

    def _refresh(self, path: str, entry: FileStateEntry) -> None:
        """Move ``path`` to the most-recently-observed end of the order."""
        self.entries.pop(path, None)
        self.entries[path] = entry


def _same_revision(stored: FileRevision | None, observed: FileRevision | None) -> bool:
    """True when both revisions identify the same (or both unknown) state."""
    if stored is None or observed is None:
        return stored is None and observed is None
    return stored.matches(observed)
