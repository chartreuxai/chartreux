from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from chartreux.core.file_revision import (
    MAX_ACTIVE_TRAVERSALS,
    REVISION_SCOPE_FULL,
    FileRevision,
    ReadTraversal,
    TraversalRegistry,
    content_revision,
    full_revision,
    new_traversal_id,
)
from chartreux.utils.io import (
    CONTENT_REVISION_ALGORITHM,
    ContentRevisionHasher,
    read_lines_hashed,
    read_lines_hashed_async,
)


class TestContentRevision:
    def test_same_content_same_revision(self) -> None:
        assert content_revision(b"hello\n") == content_revision(b"hello\n")

    def test_different_content_different_revision(self) -> None:
        assert content_revision(b"hello\n") != content_revision(b"hello\r\n")
        assert content_revision(b"a") != content_revision(b"b")

    def test_empty_content_has_a_revision(self) -> None:
        assert content_revision(b"") == content_revision(b"")

    def test_identifier_form(self) -> None:
        identifier = content_revision(b"data")
        prefix, _, digest = identifier.partition(":")
        assert prefix == CONTENT_REVISION_ALGORITHM
        assert len(digest) == 32
        int(digest, 16)  # valid hex

    def test_hasher_is_incremental_and_matches_one_shot(self) -> None:
        hasher = ContentRevisionHasher()
        hasher.update(b"cat")
        hasher.update(b"dog\n")
        assert hasher.identifier() == content_revision(b"catdog\n")

    def test_identifier_matches_independent_blake2b(self) -> None:
        # The identifier is a pure function of the bytes: an independent
        # blake2b computation pins the algorithm, digest size, and stability
        # across processes (blake2b is not seeded per process).
        expected = (
            "blake2b-128:" + hashlib.blake2b(b"stable", digest_size=16).hexdigest()
        )
        assert content_revision(b"stable") == expected


class TestFileRevision:
    def test_full_revision_metadata(self) -> None:
        revision = full_revision(b"a\nb\n", line_count=2)
        assert revision.revision == content_revision(b"a\nb\n")
        assert revision.scope == REVISION_SCOPE_FULL
        assert revision.size_bytes == 4
        assert revision.line_count == 2

    def test_matches_compares_identifiers(self) -> None:
        revision = full_revision(b"a\n")
        assert revision.matches(content_revision(b"a\n"))
        assert revision.matches(full_revision(b"a\n"))
        assert not revision.matches(content_revision(b"b\n"))
        assert not revision.matches(None)

    def test_partial_views_are_never_published(self) -> None:
        # The contract degrades to no revision for truncated reads; the model
        # has no "partial" scope to accidentally compare.
        assert REVISION_SCOPE_FULL == "full"
        assert (
            FileRevision(revision=content_revision(b"x"), size_bytes=1).scope == "full"
        )


class TestTraversalIdentity:
    def test_new_traversal_id_form_and_uniqueness(self) -> None:
        first = new_traversal_id()
        second = new_traversal_id()
        assert first.startswith("tr-")
        assert len(first) == len("tr-") + 32
        assert first != second

    def test_registry_open_get_discard(self) -> None:
        registry = TraversalRegistry()
        record = ReadTraversal(
            traversal_id="tr-1",
            file_path="/tmp/f.txt",
            opening_offset=1,
            opening_revision=content_revision(b"x"),
            next_offset=101,
        )
        registry.open(record)
        assert registry.get("tr-1") == record
        registry.discard("tr-1")
        assert registry.get("tr-1") is None

    def test_registry_is_bounded_and_evicts_oldest(self) -> None:
        registry = TraversalRegistry()
        for index in range(MAX_ACTIVE_TRAVERSALS + 4):
            registry.open(
                ReadTraversal(
                    traversal_id=f"tr-{index}",
                    file_path="/tmp/f.txt",
                    opening_offset=1,
                    opening_revision=None,
                    next_offset=2,
                )
            )
        assert len(registry.records) == MAX_ACTIVE_TRAVERSALS
        assert registry.get("tr-0") is None
        assert registry.get("tr-1") is None
        assert registry.get("tr-2") is None
        assert registry.get("tr-3") is None
        newest = MAX_ACTIVE_TRAVERSALS + 3
        assert registry.get(f"tr-{newest}") is not None

    def test_registry_get_refreshes_recency(self) -> None:
        registry = TraversalRegistry()
        for index in range(MAX_ACTIVE_TRAVERSALS):
            registry.open(
                ReadTraversal(
                    traversal_id=f"tr-{index}",
                    file_path="/tmp/f.txt",
                    opening_offset=1,
                    opening_revision=None,
                    next_offset=2,
                )
            )
        assert registry.get("tr-0") is not None  # refresh tr-0
        registry.open(
            ReadTraversal(
                traversal_id="tr-new",
                file_path="/tmp/f.txt",
                opening_offset=1,
                opening_revision=None,
                next_offset=2,
            )
        )
        assert registry.get("tr-0") is not None  # survived eviction
        assert registry.get("tr-1") is None  # evicted instead


class TestReadLinesHashed:
    def test_whole_file_window_carries_revision(self, tmp_path: Path) -> None:
        target = tmp_path / "f.txt"
        target.write_bytes(b"a\nb\nc\n")
        read = read_lines_hashed(target, limit=100, max_bytes=1024)
        assert read.lines == ["a", "b", "c"]
        assert read.total_lines == 3
        assert read.was_truncated is False
        assert read.revision == content_revision(b"a\nb\nc\n")
        assert read.first_line_truncated is False

    def test_truncated_window_has_no_revision(self, tmp_path: Path) -> None:
        target = tmp_path / "f.txt"
        target.write_bytes(b"".join(f"line {i}\n".encode() for i in range(100)))
        read = read_lines_hashed(target, limit=10, max_bytes=1024)
        assert len(read.lines) == 10
        assert read.total_lines is None
        assert read.was_truncated is True
        # A partial view is not a stable file-state identifier: no revision.
        assert read.revision is None

    def test_suffix_window_has_no_revision(self, tmp_path: Path) -> None:
        target = tmp_path / "f.txt"
        target.write_bytes(b"a\nb\nc\n")
        read = read_lines_hashed(target, start_line=2, limit=10, max_bytes=1024)
        assert read.lines == ["b", "c"]
        assert read.was_truncated is False
        assert read.total_lines == 3
        # The read consumed only a suffix of the file.
        assert read.revision is None

    def test_full_scan_reports_whole_file_from_any_offset(self, tmp_path: Path) -> None:
        target = tmp_path / "f.txt"
        target.write_bytes(b"".join(f"line {i}\n".encode() for i in range(50)))
        read = read_lines_hashed(
            target, start_line=40, limit=5, max_bytes=1024, full_scan=True
        )
        # Line 40 (1-indexed) is the label "line 39" in this 0-indexed file.
        assert read.lines == [f"line {i}" for i in range(39, 44)]
        assert read.total_lines == 50
        assert read.was_truncated is True
        # The scan consumed the whole file, so the revision is full-scope.
        assert read.revision == content_revision(target.read_bytes())

    def test_lines_are_never_cut_mid_line(self, tmp_path: Path) -> None:
        target = tmp_path / "f.txt"
        target.write_bytes(b"x" * 100 + b"\n" + b"y" * 100 + b"\n")
        read = read_lines_hashed(target, limit=10, max_bytes=150)
        # The second line would exceed the remaining budget: the window ends
        # before it instead of serving a silently cut line.
        assert read.lines == ["x" * 100]
        assert read.was_truncated is True
        assert read.total_lines is None

    def test_oversized_first_line_is_bounded_and_flagged(self, tmp_path: Path) -> None:
        target = tmp_path / "f.txt"
        target.write_bytes(b"x" * 5000 + b"\n")
        read = read_lines_hashed(target, limit=10, max_bytes=1024, max_line_bytes=1024)
        assert read.lines == ["x" * 1024]
        assert read.first_line_truncated is True
        assert read.was_truncated is True

    def test_empty_file(self, tmp_path: Path) -> None:
        target = tmp_path / "f.txt"
        target.write_bytes(b"")
        read = read_lines_hashed(target, limit=10, max_bytes=1024)
        assert read.lines == []
        assert read.total_lines == 0
        assert read.was_truncated is False
        assert read.revision == content_revision(b"")

    def test_crlf_is_normalized_in_lines(self, tmp_path: Path) -> None:
        target = tmp_path / "f.txt"
        target.write_bytes(b"a\r\nb\r\n")
        read = read_lines_hashed(target, limit=10, max_bytes=1024)
        assert read.lines == ["a", "b"]
        # The revision hashes the raw bytes as consumed.
        assert read.revision == content_revision(b"a\r\nb\r\n")

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(OSError):
            read_lines_hashed(tmp_path / "nope.txt", limit=10, max_bytes=1024)

    @pytest.mark.asyncio
    async def test_async_variant(self, tmp_path: Path) -> None:
        target = tmp_path / "f.txt"
        target.write_bytes(b"a\nb\n")
        read = await read_lines_hashed_async(target, limit=10, max_bytes=1024)
        assert read.lines == ["a", "b"]
        assert read.revision == content_revision(b"a\nb\n")
