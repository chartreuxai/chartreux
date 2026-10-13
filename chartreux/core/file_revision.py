"""Content revision contract for file reads.

A *content revision* is a stable, cheap identifier for a file state, computed
from the bytes a read path actually consumed — never from host-side metadata
(``stat``), and never from a second read that bypasses the tool's own read
path. Reads can be client-mediated (the tool I/O port serves the content, and
the host may not have the file at all), so the identifier must be derivable
from the content the tool serves, not from host files.

Identifier form: ``"blake2b-128:<32 hex chars>"`` over the file's raw bytes
(:func:`chartreux.utils.io.content_revision`). blake2b is stable across
processes and platforms, so revisions compare equal for equal content
regardless of where they were computed.

Scope rules:

- A revision with scope ``"full"`` identifies the complete file state. It is
  only produced when the read path consumed the entire file (a full scan that
  reached EOF, or a bounded window that reached EOF from line 1).
- A read that stopped early (a truncated window) cannot identify the file
  state, so it degrades gracefully: no revision is reported at all. A hash of
  a prefix is stable for that prefix but is not a file-state identifier, so it
  is never published as one.
- The tool I/O port (client-mediated reads) returns bounded windows of
  decoded lines and cannot provide a full-state revision; reads through it
  report ``revision=None``. This is a known degradation, not an error.

Traversal identity: an oversized read is served as one intentional traversal
of chunks. Each traversal has an id (:func:`new_traversal_id`) and records the
content revision observed when the traversal opened
(:class:`ReadTraversal`). Continuation calls echo the id; the tool compares
the read-time revision against the opening revision and restarts the
traversal when the file changed underneath it. Traversal records live in a
bounded per-tool registry (:class:`TraversalRegistry`) so state cannot grow
without limit.
"""

from __future__ import annotations

import uuid

from pydantic import BaseModel, Field

from chartreux.utils.io import ContentRevisionHasher, content_revision

__all__ = [
    "MAX_ACTIVE_TRAVERSALS",
    "REVISION_SCOPE_FULL",
    "ContentRevisionHasher",
    "FileRevision",
    "ReadTraversal",
    "TraversalRegistry",
    "content_revision",
    "full_revision",
    "new_traversal_id",
]

REVISION_SCOPE_FULL = "full"

#: Upper bound on simultaneously in-progress read traversals per tool.
MAX_ACTIVE_TRAVERSALS = 16


class FileRevision(BaseModel):
    """A content revision of a file state.

    ``scope`` is always ``"full"``: partial (window-only) views are never
    published as revisions, per the contract in the module docstring.
    ``size_bytes`` and ``line_count`` describe the hashed content.
    """

    revision: str
    scope: str = REVISION_SCOPE_FULL
    size_bytes: int
    line_count: int | None = None

    def matches(self, other: FileRevision | str | None) -> bool:
        """True when ``other`` identifies the same file state."""
        if other is None:
            return False
        identifier = other.revision if isinstance(other, FileRevision) else other
        return self.revision == identifier and self.scope == REVISION_SCOPE_FULL


def full_revision(data: bytes, *, line_count: int | None = None) -> FileRevision:
    """Build the full-scope revision for exact file content."""
    return FileRevision(
        revision=content_revision(data), size_bytes=len(data), line_count=line_count
    )


def new_traversal_id() -> str:
    """Return a fresh traversal identifier for a multi-chunk read."""
    return f"tr-{uuid.uuid4().hex}"


class ReadTraversal(BaseModel):
    """State of one in-progress read traversal (the continuation contract).

    ``opening_offset`` is the line the traversal started from;
    ``opening_revision`` is the content revision observed at the traversal's
    opening (``None`` when the read path cannot provide one, e.g. the tool
    I/O port — change detection is then unavailable for that traversal);
    ``next_offset`` is the offset the next continuation call should pass;
    ``in_progress_line``/``in_progress_chars`` track a line that is longer
    than one chunk and is being served in slices.
    """

    traversal_id: str
    file_path: str
    opening_offset: int
    opening_revision: str | None
    next_offset: int
    in_progress_line: int | None = None
    in_progress_chars: int = 0


class TraversalRegistry(BaseModel):
    """Bounded store of in-progress read traversals, keyed by traversal id.

    At most :data:`MAX_ACTIVE_TRAVERSALS` records are retained; the least
    recently used record is evicted first. A continuation whose record was
    evicted degrades to a fresh traversal from the requested offset.
    """

    records: dict[str, ReadTraversal] = Field(default_factory=dict)

    def open(self, record: ReadTraversal) -> None:
        """Register or replace ``record``, evicting the oldest records."""
        self.records.pop(record.traversal_id, None)
        self.records[record.traversal_id] = record
        while len(self.records) > MAX_ACTIVE_TRAVERSALS:
            self.records.pop(next(iter(self.records)))

    def get(self, traversal_id: str) -> ReadTraversal | None:
        """Return the record for ``traversal_id``, or ``None``."""
        record = self.records.get(traversal_id)
        if record is not None:
            # Refresh recency: reinsert at the end of the insertion order.
            self.records.pop(traversal_id)
            self.records[traversal_id] = record
        return record

    def discard(self, traversal_id: str) -> None:
        """Drop the record for ``traversal_id`` (e.g. traversal complete)."""
        self.records.pop(traversal_id, None)
