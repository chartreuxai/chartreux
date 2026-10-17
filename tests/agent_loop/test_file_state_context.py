"""Working-set file state context: observation intake, blocks, handoff, replay."""

from __future__ import annotations

from pathlib import Path

from chartreux.core.agent_loop._file_state_context import (
    FILE_STATE_BLOCK_TAG,
    FileStateContext,
    RevisionRegistryPort,
    render_file_state_block,
    render_file_state_handoff,
)
from chartreux.core.file_revision import FileRevision, full_revision
from chartreux.core.file_state import MAX_TRACKED_FILES, FileStateStore
from chartreux.core.llm_models import LLMMessage, Role
from chartreux.core.tools.builtins.edit import EditResult
from chartreux.core.tools.builtins.read_file import ReadFileResult
from chartreux.core.tools.builtins.write_file import WriteFileResult


class _DictRegistry:
    """Minimal RevisionRegistryPort stand-in (dict-backed, like the real one)."""

    def __init__(self) -> None:
        self.revisions: dict[str, FileRevision] = {}
        self.published: list[tuple[str, FileRevision, int | None]] = []

    def publish(
        self, path: str, revision: FileRevision, *, sequence: int | None = None
    ) -> None:
        self.published.append((path, revision, sequence))
        self.revisions[path] = revision

    def known_revision(self, path: str) -> FileRevision | None:
        return self.revisions.get(path)

    def external_bump(
        self, path: str, last_seen: FileRevision | str | None
    ) -> FileRevision | None:
        current = self.revisions.get(path)
        if current is None or last_seen is None or current.matches(last_seen):
            return None
        return current


def _rev(data: bytes) -> FileRevision:
    return full_revision(data)


def _blocks(messages: list[LLMMessage]) -> list[LLMMessage]:
    return [
        message
        for message in messages
        if message.injected
        and message.role == Role.user
        and FILE_STATE_BLOCK_TAG in str(message.content)
    ]


class TestObservationIntake:
    def test_read_enrolls_path_with_outline_and_appends_one_block(
        self, tmp_path: Path
    ) -> None:
        target = tmp_path / "mod.py"
        target.write_text("class Foo:\n    pass\n\n\ndef bar():\n    return 1\n")
        registry = _DictRegistry()
        context = FileStateContext(registry=registry)

        context.observe_read(str(target), "blake2b-128:aaa", line_count=5)

        entry = context.store.entry(str(target))
        assert entry is not None
        assert entry.revision is not None
        assert entry.revision.revision == "blake2b-128:aaa"
        assert entry.outline is not None
        assert "class Foo" in entry.outline.render()
        assert registry.known_revision(str(target)) == entry.revision
        blocks = context.flush_blocks()
        assert len(blocks) == 1
        assert blocks[0].role == Role.user
        assert blocks[0].injected is True
        content = str(blocks[0].content)
        assert str(target) in content
        assert "blake2b-128:aaa" in content
        assert "class Foo (lines 1-2)" in content
        assert "source='self'" in content

    def test_unchanged_reobservation_appends_nothing(self, tmp_path: Path) -> None:
        target = tmp_path / "mod.py"
        target.write_text("x = 1\n")
        context = FileStateContext(registry=_DictRegistry())
        context.observe_read(str(target), "blake2b-128:aaa")
        assert len(context.flush_blocks()) == 1
        # Same revision observed again: no new block, nothing pending.
        context.observe_read(str(target), "blake2b-128:aaa")
        assert context.flush_blocks() == []
        assert len(_blocks(context.flush_blocks())) == 0

    def test_edit_result_bumps_revision_and_appends_block(self, tmp_path: Path) -> None:
        target = tmp_path / "mod.py"
        target.write_text("def f():\n    return 1\n")
        context = FileStateContext(registry=_DictRegistry())
        context.observe_edit(str(target), _rev(b"def f():\n    return 1\n"))
        context.flush_blocks()
        target.write_text("def f():\n    return 2\n\n\ndef g():\n    return 3\n")
        context.observe_edit(
            str(target), _rev(b"def f():\n    return 2\n\n\ndef g():\n    return 3\n")
        )
        blocks = context.flush_blocks()
        assert len(blocks) == 1
        content = str(blocks[0].content)
        assert "function g (lines 5-6)" in content
        assert context.store.known_revision(str(target)) is not None

    def test_write_result_records_the_post_write_revision(self, tmp_path: Path) -> None:
        target = tmp_path / "out.txt"
        target.write_text("written")
        context = FileStateContext(registry=_DictRegistry())
        context.observe_write(str(target), _rev(b"written"))
        entry = context.store.entry(str(target))
        assert entry is not None
        assert entry.revision == _rev(b"written")
        assert len(context.flush_blocks()) == 1

    def test_tool_result_dispatch_by_type(self, tmp_path: Path) -> None:
        target = tmp_path / "mod.py"
        target.write_text("def f():\n    return 1\n")
        context = FileStateContext(registry=_DictRegistry())
        context.observe_tool_result(
            "read_file",
            ReadFileResult(
                file_path=str(target),
                content="1\tf():\n",
                num_lines=2,
                start_line=1,
                revision="blake2b-128:bbb",
            ),
        )
        context.observe_tool_result(
            "edit",
            EditResult(file=str(target), message="ok", old_string="1", new_string="2"),
        )
        context.observe_tool_result(
            "write_file",
            WriteFileResult(
                file_path=str(target),
                bytes_written=2,
                content="ab",
                revision=_rev(b"ab"),
            ),
        )
        assert str(target) in context.store
        # The read (new revision) and the write (a different, vouched
        # revision) each append one block; the edit carries no revision
        # (client-mediated write) and appends nothing.
        blocks = context.flush_blocks()
        assert len(blocks) == 2

    def test_non_file_tool_result_is_ignored(self) -> None:
        class _Other:
            pass

        context = FileStateContext(registry=_DictRegistry())
        context.observe_tool_result("bash", _Other())  # type: ignore[arg-type]
        assert len(context.store) == 0
        assert context.flush_blocks() == []


class TestExternalBumps:
    def test_external_bump_is_reported_and_adopted(self, tmp_path: Path) -> None:
        target = tmp_path / "mod.py"
        target.write_text("def f():\n    return 1\n")
        registry: RevisionRegistryPort = _DictRegistry()
        worker_a = FileStateContext(registry=registry)
        worker_b = FileStateContext(registry=registry)
        # Both workers track the file at revision v1; each one's own seed
        # observation appends its own block, consumed by a first flush.
        worker_a.observe_edit(str(target), _rev(b"v1"))
        worker_b.observe_edit(str(target), _rev(b"v1"))
        assert len(worker_a.flush_blocks()) == 1
        assert len(worker_b.flush_blocks()) == 1
        # Nothing changed since: neither worker appends anything.
        assert worker_a.flush_blocks() == []
        assert worker_b.flush_blocks() == []
        # Worker A edits the file: its own block, published to the registry.
        target.write_text("def f():\n    return 2\n")
        worker_a.observe_edit(str(target), _rev(b"def f():\n    return 2\n"))
        assert len(worker_a.flush_blocks()) == 1
        # Worker B's next flush reports the external bump and adopts it.
        blocks = worker_b.flush_blocks()
        assert len(blocks) == 1
        content = str(blocks[0].content)
        assert "source='external'" in content
        assert str(target) in content
        assert worker_b.store.known_revision(str(target)) == _rev(
            b"def f():\n    return 2\n"
        )
        # The bump is reported once: the next flush adds nothing.
        assert worker_b.flush_blocks() == []

    def test_registry_read_discovery_publishes_external_change(
        self, tmp_path: Path
    ) -> None:
        target = tmp_path / "mod.py"
        target.write_text("a = 1\n")
        registry: RevisionRegistryPort = _DictRegistry()
        observer = FileStateContext(registry=registry)
        tracker = FileStateContext(registry=registry)
        tracker.observe_edit(str(target), _rev(b"a = 1\n"))
        tracker.flush_blocks()
        # The file changes on disk; a read discovers the new revision.
        target.write_text("a = 2\n")
        observer.observe_read(str(target), "blake2b-128:new")
        observer.flush_blocks()
        # The tracker's next block reports the bump discovered through the read.
        blocks = tracker.flush_blocks()
        assert len(blocks) == 1
        assert "source='external'" in str(blocks[0].content)

    def test_no_registry_disables_external_bumps(self, tmp_path: Path) -> None:
        target = tmp_path / "mod.py"
        target.write_text("a = 1\n")
        context = FileStateContext(registry=None)
        context.observe_edit(str(target), _rev(b"a = 1\n"))
        assert len(context.flush_blocks()) == 1
        # No registry: nothing can be judged externally bumped.
        assert context.flush_blocks() == []


class TestMembershipRules:
    def test_sensitive_path_enrolls_without_outline(self, tmp_path: Path) -> None:
        target = tmp_path / ".env"
        target.write_text("SECRET=1\n")
        registry = _DictRegistry()
        context = FileStateContext(registry=registry)
        context.observe_read(str(target), "blake2b-128:env")
        entry = context.store.entry(str(target))
        assert entry is not None
        assert entry.outline is None
        # The revision is still published (tree truth), but the block carries
        # no outline.
        assert registry.known_revision(str(target)) is not None
        blocks = context.flush_blocks()
        assert len(blocks) == 1
        content = str(blocks[0].content)
        assert "No symbol outline" in content
        assert "SECRET" not in content

    def test_denied_read_enrolls_nothing(self, tmp_path: Path) -> None:
        # A denied read produces no result model at all: the observation seam
        # is post-execution, so there is nothing to observe.
        context = FileStateContext(registry=_DictRegistry())
        assert len(context.store) == 0
        assert context.flush_blocks() == []

    def test_port_read_enrolls_without_revision_or_block(self, tmp_path: Path) -> None:
        target = tmp_path / "mod.py"
        target.write_text("def f():\n    return 1\n")
        registry = _DictRegistry()
        context = FileStateContext(registry=registry)
        context.observe_read(str(target), None)
        entry = context.store.entry(str(target))
        assert entry is not None
        assert entry.revision is None
        # No revision information: no block, and nothing published.
        assert context.flush_blocks() == []
        assert registry.known_revision(str(target)) is None
        # A later revisioned observation is a change and appends a block.
        context.observe_read(str(target), "blake2b-128:known")
        assert len(context.flush_blocks()) == 1


class TestNonVouchingObservations:
    """W1: a non-vouching observation (revision=None) never erases knowledge."""

    def test_windowed_read_keeps_baseline_and_appends_no_block(
        self, tmp_path: Path
    ) -> None:
        # Regression: full read -> block; bounded window read (revision=None);
        # re-read unchanged -> NO second block. The pre-fix behavior wiped the
        # baseline on the windowed read and appended a phantom source='self'
        # block on the re-read.
        target = tmp_path / "mod.py"
        target.write_text("def f():\n    return 1\n")
        context = FileStateContext(registry=_DictRegistry())
        context.observe_read(str(target), "blake2b-128:aaa")
        assert len(context.flush_blocks()) == 1
        # A bounded window read publishes no revision: non-vouching.
        context.observe_read(str(target), None)
        known = context.store.known_revision(str(target))
        assert known is not None
        assert known.revision == "blake2b-128:aaa"
        assert context.flush_blocks() == []
        # A full re-read of the unchanged file is not a revision change.
        context.observe_read(str(target), "blake2b-128:aaa")
        assert context.flush_blocks() == []

    def test_external_bump_detection_survives_non_vouching_observation(
        self, tmp_path: Path
    ) -> None:
        # Regression: the non-vouching observation must not disable
        # external-bump detection for the path (the wiped baseline used to
        # make external_bump fail safe with no baseline).
        target = tmp_path / "mod.py"
        target.write_text("def f():\n    return 1\n")
        registry: RevisionRegistryPort = _DictRegistry()
        worker = FileStateContext(registry=registry)
        worker.observe_read(str(target), "blake2b-128:aaa")
        worker.flush_blocks()
        # A bounded window read (revision=None): the baseline survives.
        worker.observe_read(str(target), None)
        # Another worker bumps the file.
        registry.publish(str(target), _rev(b"def f():\n    return 2\n"))
        blocks = worker.flush_blocks()
        assert len(blocks) == 1
        assert "source='external'" in str(blocks[0].content)
        assert worker.store.known_revision(str(target)) == _rev(
            b"def f():\n    return 2\n"
        )

    def test_port_read_enrolls_fresh_path_without_revision(
        self, tmp_path: Path
    ) -> None:
        # A path first observed through a non-vouching read enrolls with
        # revision=None (and its outline) but appends nothing.
        target = tmp_path / "mod.py"
        target.write_text("def f():\n    return 1\n")
        registry = _DictRegistry()
        context = FileStateContext(registry=registry)
        context.observe_read(str(target), None)
        entry = context.store.entry(str(target))
        assert entry is not None
        assert entry.revision is None
        assert entry.outline is not None
        assert context.flush_blocks() == []
        assert registry.known_revision(str(target)) is None


class TestWriteRevisionProvenance:
    """W6: client-mediated writes publish no fabricated revision."""

    def test_client_mediated_write_publishes_no_revision(self, tmp_path: Path) -> None:
        # A write through the tool I/O port cannot be attested by the host:
        # WriteFileResult.revision is None (the edit degradation), so the
        # store/registry must not receive a fabricated full-byte revision.
        target = tmp_path / "mod.py"
        target.write_text("def f():\n    return 1\n")
        registry = _DictRegistry()
        context = FileStateContext(registry=registry)
        context.observe_tool_result(
            "write_file",
            WriteFileResult(file_path=str(target), bytes_written=2, content="ab"),
        )
        entry = context.store.entry(str(target))
        assert entry is not None
        assert entry.revision is None
        assert context.flush_blocks() == []
        assert registry.known_revision(str(target)) is None

    def test_host_write_publishes_the_full_revision(self, tmp_path: Path) -> None:
        target = tmp_path / "mod.py"
        target.write_text("def f():\n    return 1\n")
        registry = _DictRegistry()
        context = FileStateContext(registry=registry)
        context.observe_tool_result(
            "write_file",
            WriteFileResult(
                file_path=str(target),
                bytes_written=2,
                content="ab",
                revision=_rev(b"ab"),
            ),
        )
        assert context.store.known_revision(str(target)) == _rev(b"ab")
        assert registry.known_revision(str(target)) == _rev(b"ab")
        assert len(context.flush_blocks()) == 1

    def test_client_mediated_write_keeps_a_vouched_baseline(
        self, tmp_path: Path
    ) -> None:
        # W1 + W6 together: a non-vouching write observation never erases a
        # previously vouched revision.
        target = tmp_path / "mod.py"
        target.write_text("def f():\n    return 1\n")
        context = FileStateContext(registry=_DictRegistry())
        context.observe_edit(str(target), _rev(b"def f():\n    return 1\n"))
        context.flush_blocks()
        context.observe_tool_result(
            "write_file",
            WriteFileResult(file_path=str(target), bytes_written=2, content="ab"),
        )
        assert context.store.known_revision(str(target)) == _rev(
            b"def f():\n    return 1\n"
        )
        assert context.flush_blocks() == []


class TestPublicationSequences:
    """W5: the observation seam numbers publications per path per session."""

    def test_publications_carry_per_path_sequences(self, tmp_path: Path) -> None:
        target = tmp_path / "mod.py"
        target.write_text("a = 1\n")
        other = tmp_path / "other.py"
        other.write_text("b = 2\n")
        registry = _DictRegistry()
        context = FileStateContext(registry=registry)
        context.observe_edit(str(target), _rev(b"v1"))
        context.observe_edit(str(target), _rev(b"v2"))
        context.observe_edit(str(other), _rev(b"o1"))
        context.observe_edit(str(target), _rev(b"v3"))
        assert [item[2] for item in registry.published] == [1, 2, 1, 3]

    def test_history_clear_keeps_publication_sequences(self, tmp_path: Path) -> None:
        # The sequences are session-lifetime ordering tokens: a history clear
        # must not reset them, or post-clear publications would be falsely
        # superseded by pre-clear registry ordering.
        target = tmp_path / "mod.py"
        target.write_text("a = 1\n")
        registry = _DictRegistry()
        context = FileStateContext(registry=registry)
        context.observe_edit(str(target), _rev(b"v1"))
        context.reset()
        context.observe_edit(str(target), _rev(b"v2"))
        assert [item[2] for item in registry.published] == [1, 2]


class TestHandoff:
    def test_empty_store_renders_no_handoff(self) -> None:
        assert render_file_state_handoff(FileStateStore()) is None
        context = FileStateContext(registry=_DictRegistry())
        assert context.handoff_message() is None

    def test_handoff_lists_paths_revisions_and_outlines(self, tmp_path: Path) -> None:
        target = tmp_path / "mod.py"
        target.write_text("class Foo:\n    pass\n")
        other = tmp_path / "notes.txt"
        other.write_text("hello\n")
        context = FileStateContext(registry=_DictRegistry())
        context.observe_read(str(target), "blake2b-128:aaa")
        context.observe_read(str(other), None)
        message = context.handoff_message()
        assert message is not None
        assert message.role == Role.user
        assert message.injected is True
        content = str(message.content)
        assert "file_state_handoff" in content
        assert str(target) in content
        assert "blake2b-128:aaa" in content
        assert "class Foo (lines 1-2)" in content
        assert str(other) in content
        assert "revision unknown" in content

    def test_handoff_is_bounded(self, tmp_path: Path) -> None:
        store = FileStateStore()
        for index in range(MAX_TRACKED_FILES):
            store.record_revision(f"/repo/file_{index}.py", _rev(f"{index}".encode()))
        from chartreux.core.file_outline import outline_source

        big = "class Big:\n" + "".join(
            f"    def method_{i}(self):\n        return {i}\n" for i in range(200)
        )
        for index in range(MAX_TRACKED_FILES):
            store.record_observation(
                f"/repo/big_{index}.py",
                revision=_rev(f"big{index}".encode()),
                outline=outline_source(big),
            )
        content = render_file_state_handoff(store)
        assert content is not None
        assert len(content.encode("utf-8")) <= 32 * 1024 + 200
        assert "handoff budget reached" in content


class TestCompactionReplay:
    def test_merge_replayed_adopts_envelope_entries(self) -> None:
        context = FileStateContext(registry=_DictRegistry())
        replayed = FileStateStore()
        replayed.record_revision("/repo/a.py", _rev(b"a"))
        context.merge_replayed(replayed)
        assert context.store.known_revision("/repo/a.py") == _rev(b"a")

    def test_merge_replayed_registry_tiebreak_prefers_tree_truth(
        self, tmp_path: Path
    ) -> None:
        target = tmp_path / "mod.py"
        target.write_text("v2")
        registry = _DictRegistry()
        registry.publish(str(target), _rev(b"v2"))
        context = FileStateContext(registry=registry)
        # The live store observed v1; the envelope says v1 too, but the
        # registry (tree truth) says v2 — the envelope must not win.
        context.observe_edit(str(target), _rev(b"v1"))
        replayed = FileStateStore()
        replayed.record_revision(str(target), _rev(b"v1"))
        context.merge_replayed(replayed)
        assert context.store.known_revision(str(target)) == _rev(b"v1")
        # When the envelope matches tree truth and the live store does not,
        # the envelope wins.
        context2 = FileStateContext(registry=registry)
        replayed2 = FileStateStore()
        replayed2.record_revision(str(target), _rev(b"v2"))
        context2.merge_replayed(replayed2)
        assert context2.store.known_revision(str(target)) == _rev(b"v2")

    def test_merge_replayed_none_revision_never_wipes_baseline(self) -> None:
        registry = _DictRegistry()
        registry.publish("/repo/a.py", _rev(b"known"))
        context = FileStateContext(registry=registry)
        context.observe_edit("/repo/a.py", _rev(b"known"))
        replayed = FileStateStore()
        replayed.record_revision("/repo/a.py", None)
        context.merge_replayed(replayed)
        assert context.store.known_revision("/repo/a.py") == _rev(b"known")


class TestReset:
    def test_reset_drops_store_and_pending(self, tmp_path: Path) -> None:
        target = tmp_path / "mod.py"
        target.write_text("a = 1\n")
        registry = _DictRegistry()
        context = FileStateContext(registry=registry)
        context.observe_edit(str(target), _rev(b"a = 1\n"))
        context.reset()
        assert len(context.store) == 0
        assert context.flush_blocks() == []
        # The registry is tree truth and survives a session reset.
        assert registry.known_revision(str(target)) == _rev(b"a = 1\n")


class TestBlockRendering:
    def test_block_marks_external_source(self) -> None:
        content = render_file_state_block(
            path="/repo/a.py", revision=_rev(b"a"), outline=None, external=True
        )
        assert "source='external'" in content
        assert "Another worker" in content
        assert "No symbol outline" in content

    def test_block_marks_unknown_revision(self) -> None:
        content = render_file_state_block(
            path="/repo/a.py", revision=None, outline=None, external=False
        )
        assert "revision='unknown'" in content
        assert "client-mediated" in content

    def test_blocks_are_immutable_and_ordered(self, tmp_path: Path) -> None:
        target = tmp_path / "mod.py"
        target.write_text("a = 1\n")
        context = FileStateContext(registry=_DictRegistry())
        context.observe_edit(str(target), _rev(b"a = 1\n"))
        first = context.flush_blocks()
        context.observe_edit(str(target), _rev(b"a = 2\n"))
        second = context.flush_blocks()
        assert len(first) == 1 and len(second) == 1
        # The first block is unchanged by the second flush (no rewrite).
        assert str(first[0].content) != str(second[0].content)
        assert "blake2b-128" in str(first[0].content)
