from __future__ import annotations

import threading

import pytest

from chartreux.app_server._file_revision_registry import (
    MAX_REGISTRY_ENTRIES,
    FileRevisionRegistry,
)
from chartreux.core.file_revision import FileRevision, full_revision

PATH = "/repo/src/mod.py"


def _rev(text: bytes) -> FileRevision:
    return full_revision(text)


class TestPublishAndQuery:
    def test_publish_and_known_revision(self) -> None:
        registry = FileRevisionRegistry()
        revision = _rev(b"v1")
        registry.publish(PATH, revision)
        assert registry.known_revision(PATH) == revision
        assert PATH in registry
        assert len(registry) == 1
        assert registry.paths() == [PATH]
        assert registry.snapshot() == {PATH: revision}

    def test_unknown_path(self) -> None:
        registry = FileRevisionRegistry()
        assert registry.known_revision(PATH) is None
        assert PATH not in registry
        assert len(registry) == 0

    def test_snapshot_is_a_copy(self) -> None:
        registry = FileRevisionRegistry()
        revision = _rev(b"v1")
        registry.publish(PATH, revision)
        snapshot = registry.snapshot()
        snapshot.clear()
        assert registry.known_revision(PATH) == revision

    def test_forget_and_clear(self) -> None:
        registry = FileRevisionRegistry()
        registry.publish(PATH, _rev(b"v1"))
        registry.publish("/repo/other.py", _rev(b"v2"))
        registry.forget(PATH)
        assert registry.known_revision(PATH) is None
        assert len(registry) == 1
        registry.clear()
        assert len(registry) == 0

    def test_non_string_membership_is_false(self) -> None:
        registry = FileRevisionRegistry()
        assert 42 not in registry

    def test_max_entries_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            FileRevisionRegistry(max_entries=0)


class TestBumps:
    def test_own_edit_bump_is_not_reported_as_external(self) -> None:
        registry = FileRevisionRegistry()
        before = _rev(b"before")
        after = _rev(b"after")
        registry.publish(PATH, before)  # the worker's earlier read baseline
        registry.publish(PATH, after)  # the worker's own edit result
        # The worker's last-seen revision is its post-edit revision: no bump.
        assert registry.external_bump(PATH, after) is None
        assert registry.known_revision(PATH) == after

    def test_another_workers_bump_is_visible(self) -> None:
        registry = FileRevisionRegistry()
        worker_baseline = _rev(b"v1")
        other_edit = _rev(b"v2")
        registry.publish(PATH, worker_baseline)  # this worker's read
        registry.publish(PATH, other_edit)  # another worker's edit result
        assert registry.external_bump(PATH, worker_baseline) == other_edit
        # The probe may be a bare revision identifier.
        assert registry.external_bump(PATH, worker_baseline.revision) == other_edit

    def test_read_discovered_external_change_updates_the_registry(self) -> None:
        registry = FileRevisionRegistry()
        worker_baseline = _rev(b"v1")
        on_disk = _rev(b"v2")
        registry.publish(PATH, worker_baseline)
        # A read discovers the file changed outside the tree; the read-time
        # revision is published like any other observation.
        registry.publish(PATH, on_disk)
        assert registry.known_revision(PATH) == on_disk
        assert registry.external_bump(PATH, worker_baseline) == on_disk

    def test_no_baseline_no_bump(self) -> None:
        registry = FileRevisionRegistry()
        revision = _rev(b"v1")
        registry.publish(PATH, revision)
        assert registry.external_bump(PATH, None) is None
        assert registry.external_bump("/absent.py", revision) is None
        registry.forget(PATH)
        assert registry.external_bump(PATH, revision) is None


class TestPathKeys:
    def test_paths_are_normalized(self) -> None:
        registry = FileRevisionRegistry()
        revision = _rev(b"v1")
        registry.publish("/repo/./src//mod.py", revision)
        assert registry.known_revision("/repo/src/mod.py") == revision
        assert "/repo/src/mod.py" in registry
        assert len(registry) == 1

    def test_distinct_paths_are_isolated(self) -> None:
        registry = FileRevisionRegistry()
        revision = _rev(b"v1")
        registry.publish(PATH, revision)
        assert registry.known_revision("/repo/src/other.py") is None
        assert registry.external_bump("/repo/src/other.py", revision) is None


class TestBoundedEntries:
    def test_oldest_first_eviction(self) -> None:
        registry = FileRevisionRegistry(max_entries=4)
        for index in range(6):
            registry.publish(f"/repo/f{index}.py", _rev(b"v"))
        assert len(registry) == 4
        assert registry.paths() == [f"/repo/f{i}.py" for i in (2, 3, 4, 5)]
        assert registry.known_revision("/repo/f0.py") is None
        assert registry.known_revision("/repo/f5.py") is not None

    def test_republish_refreshes_recency(self) -> None:
        registry = FileRevisionRegistry(max_entries=3)
        for index in range(3):
            registry.publish(f"/repo/f{index}.py", _rev(b"a"))
        registry.publish("/repo/f0.py", _rev(b"b"))
        registry.publish("/repo/f3.py", _rev(b"c"))
        assert registry.paths() == ["/repo/f2.py", "/repo/f0.py", "/repo/f3.py"]

    def test_default_cap(self) -> None:
        registry = FileRevisionRegistry()
        for index in range(MAX_REGISTRY_ENTRIES + 10):
            registry.publish(f"/repo/f{index}.py", _rev(b"v"))
        assert len(registry) == MAX_REGISTRY_ENTRIES
        assert registry.known_revision("/repo/f0.py") is None
        assert (
            registry.known_revision(f"/repo/f{MAX_REGISTRY_ENTRIES + 9}.py") is not None
        )


class TestConcurrency:
    def test_last_publish_wins_across_threads(self) -> None:
        registry = FileRevisionRegistry()
        first = _rev(b"first")
        second = _rev(b"second")
        a_published = threading.Event()
        b_published = threading.Event()
        observed: list[FileRevision | None] = []

        def worker_a() -> None:
            registry.publish(PATH, first)
            a_published.set()
            assert b_published.wait(timeout=5)
            observed.append(registry.known_revision(PATH))

        def worker_b() -> None:
            assert a_published.wait(timeout=5)
            registry.publish(PATH, second)
            b_published.set()

        threads = [threading.Thread(target=worker_a), threading.Thread(target=worker_b)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        # The later bump is the last-known revision; the earlier one does
        # not resurrect — concurrent bumps never lose the latest.
        assert observed == [second]
        assert registry.known_revision(PATH) == second

    def test_concurrent_publish_storm_is_consistent(self) -> None:
        registry = FileRevisionRegistry()
        shared = "/repo/shared.py"
        revisions = [_rev(f"v{index}".encode()) for index in range(8)]
        barrier = threading.Barrier(8)
        observed: list[FileRevision] = []
        observed_lock = threading.Lock()

        def worker(index: int) -> None:
            revision = revisions[index]
            barrier.wait()
            for _ in range(25):
                registry.publish(shared, revision)
                registry.publish(f"/repo/file{index}.py", revision)
                current = registry.known_revision(shared)
                if current is not None:
                    with observed_lock:
                        observed.append(current)

        threads = [threading.Thread(target=worker, args=(index,)) for index in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        # One shared path plus eight distinct paths: nothing lost, nothing
        # duplicated, and every observation was a valid published revision.
        assert len(registry) == 9
        final = registry.known_revision(shared)
        assert final is not None
        assert any(final.matches(revision) for revision in revisions)
        assert observed
        for revision in observed:
            assert any(revision.matches(candidate) for candidate in revisions)


class TestOrderedPublication:
    """W5: sequenced publications; superseded (delayed) ones are rejected."""

    def test_superseded_publication_is_rejected(self) -> None:
        # Regression: publish B, then deliver older A — the registry stays B
        # and sends no backwards external bump.
        registry = FileRevisionRegistry()
        newer = _rev(b"v2")
        older = _rev(b"v1")
        registry.publish(PATH, newer, sequence=2)
        registry.publish(PATH, older, sequence=1)  # delayed, superseded
        assert registry.known_revision(PATH) == newer
        # No backwards bump: a worker baselined at the newer revision is
        # not told the file moved to the older one.
        assert registry.external_bump(PATH, newer) is None
        assert registry.external_bump(PATH, older) == newer

    def test_equal_sequence_is_accepted(self) -> None:
        # A fresh worker's first publication of a path (sequence 1) must land
        # against the baseline its parent published with the same sequence.
        registry = FileRevisionRegistry()
        parent = _rev(b"v1")
        child = _rev(b"v2")
        registry.publish(PATH, parent, sequence=1)
        registry.publish(PATH, child, sequence=1)
        assert registry.known_revision(PATH) == child

    def test_unsequenced_publication_is_always_accepted(self) -> None:
        registry = FileRevisionRegistry()
        registry.publish(PATH, _rev(b"v1"), sequence=5)
        # No ordering claim: always accepted ...
        registry.publish(PATH, _rev(b"v2"))
        assert registry.known_revision(PATH) == _rev(b"v2")
        # ... and it does not advance the stored sequence.
        registry.publish(PATH, _rev(b"v3"), sequence=4)
        assert registry.known_revision(PATH) == _rev(b"v2")

    def test_rejected_publication_does_not_refresh_recency(self) -> None:
        registry = FileRevisionRegistry(max_entries=2)
        registry.publish("/repo/a.py", _rev(b"a"))
        registry.publish(PATH, _rev(b"v2"), sequence=5)
        registry.publish("/repo/b.py", _rev(b"b"))
        assert registry.paths() == [PATH, "/repo/b.py"]  # a evicted
        # A superseded publication for PATH is rejected and must not refresh
        # its recency: PATH stays the oldest entry and is evicted next.
        registry.publish(PATH, _rev(b"v1"), sequence=1)
        registry.publish("/repo/c.py", _rev(b"c"))
        assert registry.paths() == ["/repo/b.py", "/repo/c.py"]

    def test_superseded_publication_of_an_evicted_path_is_accepted(self) -> None:
        # Fail-safe: an evicted path is unknown, so a late sequenced
        # publication for it is accepted (never a false rejection).
        registry = FileRevisionRegistry(max_entries=1)
        registry.publish(PATH, _rev(b"v2"), sequence=9)
        registry.publish("/repo/other.py", _rev(b"o"))
        assert registry.known_revision(PATH) is None
        registry.publish(PATH, _rev(b"v1"), sequence=1)
        assert registry.known_revision(PATH) == _rev(b"v1")

    def test_concurrent_storm_keeps_the_highest_sequence(self) -> None:
        # Regression: under a concurrent publication storm, last-write-wins
        # only for genuinely newer (higher-sequence) observations.
        registry = FileRevisionRegistry()
        revisions = [_rev(f"v{index}".encode()) for index in range(16)]
        barrier = threading.Barrier(8)

        def worker(index: int) -> None:
            barrier.wait()
            # Each thread delivers its two sequences in reverse order, so
            # the lower (older) one always arrives after a newer truth is
            # already stored.
            for sequence in range(index * 2 + 2, index * 2, -1):
                registry.publish(PATH, revisions[sequence - 1], sequence=sequence)

        threads = [threading.Thread(target=worker, args=(index,)) for index in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        # The highest sequence (16) wins regardless of delivery order.
        assert registry.known_revision(PATH) == revisions[-1]
        # A late, lower-sequence delivery does not resurrect an older truth.
        registry.publish(PATH, revisions[0], sequence=1)
        assert registry.known_revision(PATH) == revisions[-1]
