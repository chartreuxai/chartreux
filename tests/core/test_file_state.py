from __future__ import annotations

import json

from chartreux.core.file_outline import FileOutline, outline_source
from chartreux.core.file_revision import FileRevision, full_revision
from chartreux.core.file_state import MAX_TRACKED_FILES, FileStateEntry, FileStateStore
from chartreux.core.llm_models import LLMMessage, PersistedToolResult, Role

PATH = "/repo/src/mod.py"
OTHER = "/repo/src/other.py"

SOURCE_V1 = "def f():\n    return 1\n"
SOURCE_V2 = "def f():\n    return 2\n"


def _rev(text: bytes) -> FileRevision:
    return full_revision(text)


def _outline(text: str) -> FileOutline:
    outline = outline_source(text)
    assert outline is not None
    return outline


def _boundary_message(store: FileStateStore) -> LLMMessage:
    return LLMMessage(
        role=Role.user,
        content="summary",
        injected=True,
        context_boundary="compaction",
        tool_result=PersistedToolResult(output=store.to_envelope()),
    )


class TestRecordAndQuery:
    def test_record_observation_stores_revision_and_outline(self) -> None:
        store = FileStateStore()
        revision = _rev(b"v1")
        outline = _outline(SOURCE_V1)
        entry = store.record_observation(PATH, revision=revision, outline=outline)
        assert entry.path == PATH
        assert entry.revision == revision
        assert entry.outline == outline
        assert store.known_revision(PATH) == revision
        assert store.outline_for(PATH) == outline
        assert store.entry(PATH) is entry
        assert PATH in store
        assert len(store) == 1
        assert store.paths() == [PATH]

    def test_paths_are_ordered_by_observation(self) -> None:
        store = FileStateStore()
        store.record_revision(PATH, _rev(b"v1"))
        store.record_revision(OTHER, _rev(b"v2"))
        assert store.paths() == [PATH, OTHER]
        store.record_revision(PATH, _rev(b"v1"))
        assert store.paths() == [OTHER, PATH]

    def test_unknown_path_queries(self) -> None:
        store = FileStateStore()
        assert store.entry("/absent.py") is None
        assert store.known_revision("/absent.py") is None
        assert store.outline_for("/absent.py") is None
        assert "/absent.py" not in store

    def test_entry_model_defaults(self) -> None:
        entry = FileStateEntry(path=PATH)
        assert entry.revision is None
        assert entry.outline is None


class TestStaleness:
    def test_known_revision_vs_probe(self) -> None:
        store = FileStateStore()
        revision = _rev(b"v1")
        store.record_revision(PATH, revision)
        assert store.is_stale(PATH, revision) is False
        assert store.is_stale(PATH, revision.revision) is False
        assert store.is_stale(PATH, _rev(b"v2")) is True
        assert store.is_stale(PATH, _rev(b"v2").revision) is True

    def test_unknown_paths_and_probes_are_never_stale(self) -> None:
        store = FileStateStore()
        assert store.is_stale("/absent.py", _rev(b"v1")) is False
        store.record_revision(PATH, _rev(b"v1"))
        assert store.is_stale(PATH, None) is False


class TestRevisionTransitions:
    def test_reobserving_the_same_revision_keeps_the_outline(self) -> None:
        store = FileStateStore()
        outline = _outline(SOURCE_V1)
        store.record_observation(PATH, revision=_rev(b"v1"), outline=outline)
        store.record_observation(PATH, revision=_rev(b"v1"), outline=None)
        assert store.outline_for(PATH) == outline

    def test_revision_transition_drops_the_outline(self) -> None:
        store = FileStateStore()
        store.record_observation(
            PATH, revision=_rev(b"v1"), outline=_outline(SOURCE_V1)
        )
        store.record_observation(PATH, revision=_rev(b"v2"), outline=None)
        assert store.known_revision(PATH) == _rev(b"v2")
        assert store.outline_for(PATH) is None

    def test_revision_transition_with_replacement_outline(self) -> None:
        store = FileStateStore()
        store.record_observation(
            PATH, revision=_rev(b"v1"), outline=_outline(SOURCE_V1)
        )
        replacement = _outline(SOURCE_V2)
        store.record_observation(PATH, revision=_rev(b"v2"), outline=replacement)
        assert store.outline_for(PATH) == replacement

    def test_non_vouching_observation_keeps_vouched_revision_and_outline(self) -> None:
        # CORRECTED CONTRACT (review W1): an observation that cannot vouch a
        # revision (a tool I/O port read or a bounded window read) must NOT
        # erase previously vouched knowledge. The AM-D1 contract pinned the
        # opposite ("observing revision=None stores None"); that wiped the
        # baseline on the dominant windowed-read pattern, which caused a
        # phantom source='self' revision-change block on the next full read
        # of unchanged content and disabled external-bump detection for the
        # path.
        store = FileStateStore()
        revision = _rev(b"v1")
        outline = _outline(SOURCE_V1)
        store.record_observation(PATH, revision=revision, outline=outline)
        # A non-vouching observation (a port read or a bounded window):
        # the stored revision and its outline survive untouched.
        store.record_observation(PATH, revision=None, outline=None)
        assert store.known_revision(PATH) == revision
        assert store.outline_for(PATH) == outline
        # The baseline still vouches: staleness keeps judging across the
        # non-vouching observation.
        assert store.is_stale(PATH, revision) is False
        assert store.is_stale(PATH, _rev(b"v2")) is True

    def test_non_vouching_observation_only_refreshes_recency(self) -> None:
        store = FileStateStore()
        store.record_observation(
            PATH, revision=_rev(b"v1"), outline=_outline(SOURCE_V1)
        )
        store.record_observation(OTHER, revision=_rev(b"v2"), outline=None)
        assert store.paths() == [PATH, OTHER]
        store.record_observation(PATH, revision=None, outline=None)
        assert store.paths() == [OTHER, PATH]

    def test_unknown_to_known_transition_needs_a_replacement_outline(self) -> None:
        store = FileStateStore()
        store.record_observation(PATH, revision=None, outline=_outline(SOURCE_V1))
        store.record_observation(PATH, revision=_rev(b"v1"), outline=None)
        assert store.known_revision(PATH) == _rev(b"v1")
        assert store.outline_for(PATH) is None


class TestMissingRevisions:
    def test_port_read_records_a_path_without_a_revision(self) -> None:
        # A fresh entry enrolled by a non-vouching observation (a tool I/O
        # port read) carries revision=None: no observation has vouched one.
        store = FileStateStore()
        entry = store.record_revision(PATH, None)
        assert entry.revision is None
        assert PATH in store
        assert store.known_revision(PATH) is None
        assert store.is_stale(PATH, _rev(b"v1")) is False

    def test_record_revision_none_keeps_a_vouched_baseline(self) -> None:
        # CORRECTED CONTRACT (review W1): record_revision(path, None) is a
        # non-vouching observation; it must not wipe a vouched baseline.
        store = FileStateStore()
        store.record_revision(PATH, _rev(b"v1"))
        store.record_revision(PATH, None)
        assert store.known_revision(PATH) == _rev(b"v1")

    def test_port_read_can_carry_an_outline(self) -> None:
        store = FileStateStore()
        outline = _outline(SOURCE_V1)
        store.record_observation(PATH, revision=None, outline=outline)
        assert store.known_revision(PATH) is None
        assert store.outline_for(PATH) == outline
        # A further no-revision observation keeps the outline (a
        # non-vouching observation never disturbs the stored entry).
        store.record_observation(PATH, revision=None, outline=None)
        assert store.outline_for(PATH) == outline


class TestRecordOutline:
    def test_record_outline_creates_an_entry_without_a_revision(self) -> None:
        store = FileStateStore()
        outline = _outline(SOURCE_V1)
        entry = store.record_outline(PATH, outline)
        assert entry.revision is None
        assert entry.outline == outline
        assert store.known_revision(PATH) is None

    def test_record_outline_does_not_touch_the_revision(self) -> None:
        store = FileStateStore()
        revision = _rev(b"v1")
        store.record_revision(PATH, revision)
        store.record_outline(PATH, _outline(SOURCE_V1))
        assert store.known_revision(PATH) == revision
        store.record_outline(PATH, None)
        assert store.outline_for(PATH) is None


class TestSerialization:
    def _populated(self) -> FileStateStore:
        store = FileStateStore()
        store.record_observation(
            PATH, revision=_rev(b"v1"), outline=_outline(SOURCE_V1)
        )
        store.record_observation(OTHER, revision=_rev(b"v2"), outline=None)
        store.record_observation("/repo/notes.txt", revision=None, outline=None)
        return store

    def test_model_round_trip(self) -> None:
        store = self._populated()
        restored = FileStateStore.model_validate(store.model_dump(mode="json"))
        assert restored == store
        assert restored.paths() == store.paths()

    def test_envelope_round_trip(self) -> None:
        store = self._populated()
        envelope = store.to_envelope()
        assert set(envelope) == {"file_state"}
        # The envelope is metadata only and JSON-serializable.
        json.dumps(envelope)
        restored = FileStateStore.from_envelope(envelope)
        assert restored == store

    def test_from_envelope_degrades_on_bad_payloads(self) -> None:
        assert FileStateStore.from_envelope({}) is None
        assert FileStateStore.from_envelope({"other": {}}) is None
        assert FileStateStore.from_envelope({"file_state": "junk"}) is None
        assert FileStateStore.from_envelope({"file_state": {"entries": "junk"}}) is None
        assert FileStateStore.from_envelope({"file_state": {"entries": {}}}) is not None


class TestReplay:
    def test_replay_from_empty_transcript(self) -> None:
        assert FileStateStore.replay([]) == FileStateStore()

    def test_replay_restores_the_last_boundary_envelope(self) -> None:
        first = FileStateStore()
        first.record_revision(PATH, _rev(b"v1"))
        second = FileStateStore()
        second.record_revision(OTHER, _rev(b"v2"))
        messages = [
            LLMMessage(role=Role.user, content="hi"),
            _boundary_message(first),
            LLMMessage(role=Role.user, content="working"),
            _boundary_message(second),
        ]
        assert FileStateStore.replay(messages) == second

    def test_replay_skips_boundaries_without_state(self) -> None:
        store = FileStateStore()
        store.record_revision(PATH, _rev(b"v1"))
        messages = [
            _boundary_message(store),
            LLMMessage(role=Role.user, content="x", context_boundary="compaction"),
        ]
        assert FileStateStore.replay(messages) == store

    def test_replay_ignores_non_boundary_tool_results(self) -> None:
        store = FileStateStore()
        store.record_revision(PATH, _rev(b"v1"))
        stray = LLMMessage(
            role=Role.user,
            content="x",
            tool_result=PersistedToolResult(output=store.to_envelope()),
        )
        assert FileStateStore.replay([stray]) == FileStateStore()

    def test_replay_skips_malformed_envelopes(self) -> None:
        good = FileStateStore()
        good.record_revision(PATH, _rev(b"v1"))
        messages = [
            _boundary_message(good),
            LLMMessage(
                role=Role.user,
                content="x",
                injected=True,
                context_boundary="compaction",
                tool_result=PersistedToolResult(output={"file_state": "junk"}),
            ),
        ]
        assert FileStateStore.replay(messages) == good


class TestBoundedCount:
    def test_no_implicit_eviction_past_the_cap(self) -> None:
        store = FileStateStore()
        for index in range(MAX_TRACKED_FILES + 5):
            store.record_revision(f"/repo/f{index}.py", _rev(b"v"))
        assert len(store) == MAX_TRACKED_FILES + 5
        assert store.is_over_cap() is True
        # Every path is still tracked: eviction is explicit only.
        assert store.known_revision("/repo/f0.py") is not None
        assert store.known_revision(f"/repo/f{MAX_TRACKED_FILES + 4}.py") is not None

    def test_evict_oldest_drops_least_recently_observed(self) -> None:
        store = FileStateStore()
        for index in range(MAX_TRACKED_FILES + 5):
            store.record_revision(f"/repo/f{index}.py", _rev(b"v"))
        evicted = store.evict_oldest(2)
        assert evicted == ["/repo/f0.py", "/repo/f1.py"]
        assert len(store) == MAX_TRACKED_FILES + 3
        assert store.known_revision("/repo/f0.py") is None

    def test_trim_to_cap_evicts_oldest_first(self) -> None:
        store = FileStateStore()
        for index in range(MAX_TRACKED_FILES + 5):
            store.record_revision(f"/repo/f{index}.py", _rev(b"v"))
        evicted = store.trim_to_cap()
        assert evicted == [f"/repo/f{i}.py" for i in range(5)]
        assert len(store) == MAX_TRACKED_FILES
        assert store.is_over_cap() is False
        assert store.paths() == [
            f"/repo/f{i}.py" for i in range(5, MAX_TRACKED_FILES + 5)
        ]

    def test_trim_to_cap_is_a_no_op_under_the_cap(self) -> None:
        store = FileStateStore()
        store.record_revision(PATH, _rev(b"v1"))
        assert store.trim_to_cap() == []
        assert store.paths() == [PATH]

    def test_rerecording_refreshes_recency(self) -> None:
        store = FileStateStore()
        store.record_revision("/repo/a.py", _rev(b"1"))
        store.record_revision("/repo/b.py", _rev(b"2"))
        store.record_revision("/repo/c.py", _rev(b"3"))
        store.record_revision("/repo/a.py", _rev(b"1"))
        assert store.evict_oldest(1) == ["/repo/b.py"]

    def test_evict_oldest_runs_dry(self) -> None:
        store = FileStateStore()
        store.record_revision(PATH, _rev(b"v1"))
        assert store.evict_oldest(0) == []
        assert store.evict_oldest(10) == [PATH]
        assert len(store) == 0
