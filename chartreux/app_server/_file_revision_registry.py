"""Dispatcher-level file revision registry.

The revision source of truth across the workers of a session tree: the
last-known content revision per file path, shared by every worker (main
agent and subagents) so the next state block can report an *external* bump —
a file changed by another worker or outside the tree since this worker last
saw it.

Bumps come from two call sites, both funneling into :meth:`publish`:

- a worker's own edit result publishes its post-edit revision
  (``EditResult.revision``);
- a read that discovers the on-disk revision publishes the read-time
  revision — an external file change discovered through a read updates the
  registry the same way.

Concurrency model: the app server runs one event loop for the whole session
tree (children are coroutines in one process, admitted under the sessions'
generation fence and ``asyncio.Lock``), but tools also run blocking work
through ``asyncio.to_thread``. The registry is therefore guarded by a
``threading.Lock`` — a *leaf* lock with short, await-free, I/O-free critical
sections — which is correct under both the event loop and worker threads and
mirrors the ``threading.Lock`` convention in ``chartreux.app_server._runtime``.
As a leaf it acquires nothing while held, so it cannot participate in a
lock-ordering cycle with the sessions' asyncio locks.

Scope decision: entries are keyed by the normalized absolute path only
(``os.path.normpath``; no symlink resolution, no case folding). Repository
scoping was considered and rejected — revisions are content-addressed
(blake2b-128 over the bytes a read path consumed), and an absolute path
identifies exactly one file on the host, so a repository partition would
fragment the namespace without changing any truth.

Ordered publication: the loop observes tool results after awaited post-tool
finalization, so an older read result can be published after a newer
revision is already known (a delayed observation), and naively publishing it
would regress tree truth and send workers backwards "external bumps". Every
publication therefore carries a per-path sequence — numbered per session at
the observation seam — and a publication whose sequence is *behind* the
path's stored sequence is superseded: it is rejected (logged at DEBUG) instead
of overwriting, and it does not refresh recency. Sequences are session-local
ordering tokens: they reject a delayed publication whose session ordering is
behind the stored one. Content-time ordering is not observable at the seam,
so a delayed publisher whose counter has run ahead can still land; that
residual is documented here rather than hidden.

Fail-safe degradation: a path that was never published, or whose entry was
evicted past the bound, is simply *unknown* — staleness queries then report
"cannot judge" rather than a false bump.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from chartreux.core.file_revision import FileRevision

__all__ = ["MAX_REGISTRY_ENTRIES", "FileRevisionRegistry"]

logger = logging.getLogger(__name__)

#: Upper bound on tracked revisions; the least-recently-published entry is
#: evicted first.
MAX_REGISTRY_ENTRIES = 512


def _normalize_path(path: str) -> str:
    """Normalize a registry key: collapse ``.``/``//``/trailing-dot segments.

    No symlink resolution and no case folding: tools report absolute paths
    from the same host filesystem, so normalization is only about spelling.
    """
    return os.path.normpath(path)


class _PublishedRevision(NamedTuple):
    """One accepted publication: the revision and its ordering token.

    ``sequence`` is the per-path publication sequence of the last accepted
    publication for the path (0 while only unsequenced publications landed).
    """

    revision: FileRevision
    sequence: int


class FileRevisionRegistry:
    """Last-known content revision per file path, shared across a session tree."""

    def __init__(self, *, max_entries: int = MAX_REGISTRY_ENTRIES) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        self._max_entries = max_entries
        self._lock = threading.Lock()
        self._revisions: dict[str, _PublishedRevision] = {}

    def publish(
        self, path: str, revision: FileRevision, *, sequence: int | None = None
    ) -> None:
        """Publish the last-known revision for ``path``.

        Call sites: a worker's own edit result (the post-edit revision) and a
        read that discovers the on-disk revision (an external change).
        Re-publishing refreshes recency; beyond ``max_entries`` the
        least-recently-published entry is evicted, and an evicted path
        degrades to unknown for staleness queries — never a false bump.

        ``sequence`` is the publication's per-path ordering token (numbered
        at the observation seam). A publication whose sequence is behind the
        path's stored sequence is a superseded (delayed) publication: it is
        rejected — logged at DEBUG, never overwriting, never refreshing
        recency — instead of regressing tree truth. Equal sequences are
        accepted (last delivery wins), so a fresh worker's first publication
        of a path can land against the baseline its parent published.
        ``sequence=None`` carries no ordering claim and is always accepted;
        it does not advance the stored sequence.
        """
        key = _normalize_path(path)
        superseded: tuple[int, int] | None = None
        with self._lock:
            stored = self._revisions.get(key)
            if (
                sequence is not None
                and stored is not None
                and sequence < stored.sequence
            ):
                superseded = (sequence, stored.sequence)
            else:
                effective = (
                    sequence
                    if sequence is not None
                    else (stored.sequence if stored is not None else 0)
                )
                self._revisions.pop(key, None)
                self._revisions[key] = _PublishedRevision(revision, effective)
                while len(self._revisions) > self._max_entries:
                    self._revisions.pop(next(iter(self._revisions)))
        if superseded is not None:
            # Logged outside the critical section: no I/O while the leaf
            # lock is held.
            logger.debug(
                "Superseded file revision publication rejected path=%s "
                "sequence=%d stored_sequence=%d",
                key,
                superseded[0],
                superseded[1],
            )

    def known_revision(self, path: str) -> FileRevision | None:
        """Return the last-known revision for ``path``, or ``None``."""
        with self._lock:
            entry = self._revisions.get(_normalize_path(path))
            return entry.revision if entry is not None else None

    def external_bump(
        self, path: str, last_seen: FileRevision | str | None
    ) -> FileRevision | None:
        """Return the tree-wide revision when it moved past a worker's baseline.

        Returns the current last-known revision when another worker (or an
        external change) bumped ``path`` since ``last_seen`` — the external
        bump the next state block reports. Returns ``None`` when the file is
        unchanged, when it is unknown (never published or evicted), or when
        the worker has no baseline (``last_seen=None``): no false bumps.
        """
        current = self.known_revision(path)
        if current is None or last_seen is None or current.matches(last_seen):
            return None
        return current

    def snapshot(self) -> dict[str, FileRevision]:
        """Return a copy of path -> last-known revision (for state blocks)."""
        with self._lock:
            return {key: entry.revision for key, entry in self._revisions.items()}

    def paths(self) -> list[str]:
        """Return the published paths, least-recently-published first."""
        with self._lock:
            return list(self._revisions)

    def forget(self, path: str) -> None:
        """Drop the entry for ``path`` (e.g. the file was deleted)."""
        with self._lock:
            self._revisions.pop(_normalize_path(path), None)

    def clear(self) -> None:
        """Drop every entry."""
        with self._lock:
            self._revisions.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._revisions)

    def __contains__(self, path: object) -> bool:
        if not isinstance(path, str):
            return False
        with self._lock:
            return _normalize_path(path) in self._revisions
