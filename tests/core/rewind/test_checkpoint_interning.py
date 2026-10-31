"""P1a checkpoint content interning: identical immutable FileState content
shares one object across turns; the pool dies on reset/rewind; turn marks and
carried-path membership are preserved exactly (only identical content shares).
"""

from __future__ import annotations

from chartreux.core.checkpoints._events import _Edit, _TurnMark
from chartreux.core.checkpoints.checkpointer import Checkpointer
from chartreux.core.checkpoints.file_store import FileStore
from chartreux.core.checkpoints.models import FileState
from chartreux.core.checkpoints.recorder import CheckpointRecorder, FileSnapshot
from chartreux.core.llm_models import LLMMessage, Role
from chartreux.core.message_list import MessageList
from tests.stubs.fake_filesystem import FakeFilesystem

# -- Helpers -------------------------------------------------------------------


class _DiskLikeFilesystem(FakeFilesystem):
    """Returns a fresh bytes object per read, like a real disk read does, so the
    retention test cannot pass by the stub aliasing its backing bytes.
    """

    def read_bytes(self, path: str) -> bytes | None:
        data = super().read_bytes(path)
        return None if data is None else bytes(data)


def _messages() -> MessageList:
    return MessageList([LLMMessage(role=Role.system, content="system")])


def _grow_messages(messages: MessageList, content: str) -> None:
    """Advance the transcript one turn: the recorder's turn id is len(messages)."""
    messages.append(LLMMessage(role=Role.user, content=content))
    messages.append(LLMMessage(role=Role.assistant, content=f"reply to {content}"))


class _Shells:
    def __init__(self, files: dict[str, bytes]) -> None:
        self.fs = _DiskLikeFilesystem(files)
        self.messages = _messages()
        self.checkpointer = Checkpointer()
        self.recorder = CheckpointRecorder(
            self.checkpointer, self.messages, files=FileStore(self.fs)
        )

    def snapshot(self, path: str) -> FileSnapshot:
        return FileSnapshot(path=path, state=FileState(self.fs.files.get(path)))


def _retained_state_bytes(cp: Checkpointer) -> int:
    """Content bytes held by distinct FileState objects reachable from the log —
    the retained-bytes metric the flat-retention outcome is stated over.
    """
    seen: set[int] = set()
    total = 0

    def visit(state: FileState) -> None:
        nonlocal total
        if id(state) in seen:
            return
        seen.add(id(state))
        if state.data is not None:
            total += len(state.data)

    for event in cp._events:
        if isinstance(event, _TurnMark):
            for state in event.pre.values():
                visit(state)
        elif isinstance(event, _Edit):
            visit(event.before)
            visit(event.after)
    return total


def _record_turn(cp: Checkpointer, turn_id: int, pre: str, post: str) -> None:
    cp.begin_turn(turn_id)
    cp.record_pre_edit("f", FileState.from_text(pre))
    cp.record_post_edit("f", FileState.from_text(post))
    cp.seal_turn()


# -- (a) Identity sharing across turns -----------------------------------------


class TestIdentitySharing:
    def test_identical_content_shares_one_state_across_turns(self) -> None:
        sh = _Shells({"a.txt": b"same\n"})

        for turn in range(1, 4):
            _grow_messages(sh.messages, f"turn {turn} prompt")
            sh.recorder.create_checkpoint()
            # A fresh FileState object per turn, as a real snapshot would be.
            sh.recorder.add_snapshot(sh.snapshot("a.txt"))
            sh.recorder.seal_turn()

        turns = sh.checkpointer.view().turns
        # Distinct growing turn ids: interning must not collapse turn marks.
        assert [turn_id for turn_id, _pre in turns] == [3, 5, 7]
        pres = [pre["a.txt"] for _turn_id, pre in turns]
        assert all(pre is pres[0] for pre in pres)
        # Structure preserved: every turn mark still carries the path exactly once.
        assert all(list(pre) == ["a.txt"] for _turn_id, pre in turns)

    def test_distinct_content_does_not_share(self) -> None:
        sh = _Shells({"a.txt": b"same\n"})

        sh.recorder.create_checkpoint()
        sh.recorder.add_snapshot(sh.snapshot("a.txt"))
        sh.fs.files["a.txt"] = b"changed\n"
        sh.recorder.seal_turn()
        sh.recorder.create_checkpoint()
        sh.recorder.add_snapshot(sh.snapshot("a.txt"))
        sh.recorder.seal_turn()

        turns = sh.checkpointer.view().turns
        assert turns[0][1]["a.txt"] is not turns[1][1]["a.txt"]
        assert turns[1][1]["a.txt"].data == b"changed\n"


# -- (b) Pool empties on both clear paths ---------------------------------------


class TestPoolLifetime:
    def test_pool_empties_on_clear(self) -> None:
        cp = Checkpointer()
        _record_turn(cp, 1, "a\n", "A\n")

        assert cp.interned_state_count > 0
        cp.clear()
        assert cp.interned_state_count == 0

    def test_pool_empties_on_rewind(self) -> None:
        cp = Checkpointer()
        _record_turn(cp, 1, "a\n", "A\n")
        _record_turn(cp, 2, "A\n", "B\n")

        assert cp.interned_state_count > 0
        cp.drop_turns_from(2)
        assert cp.interned_state_count == 0
        # Remaining structure is intact: truncation at turn 2 leaves turn 1's
        # post state as the projection.
        assert cp.view().content("f") == FileState.from_text("A\n")


# -- (c) Retention flat across many turns ---------------------------------------


class TestRetentionFlat:
    def test_retained_bytes_flat_across_1000_turns_for_unchanged_files(self) -> None:
        files = {name: bytes([65 + i]) * 1024 for i, name in enumerate("abc")}
        sh = _Shells(files)
        # Denominator: one retained copy per tracked file.
        floor = sum(len(data) for data in files.values())

        samples: dict[int, int] = {}
        for turn in range(1, 1001):
            sh.recorder.create_checkpoint()
            for name in files:
                sh.recorder.add_snapshot(sh.snapshot(name))
            sh.recorder.seal_turn()
            if turn in (100, 500, 1000):
                samples[turn] = _retained_state_bytes(sh.checkpointer)
            _grow_messages(sh.messages, f"turn {turn} reply")

        assert samples[100] >= floor
        # Flat: later samples stay approximately at the first sample's level
        # (interned content is shared, not re-retained per turn).
        for turn in (500, 1000):
            assert samples[turn] <= samples[100] * 1.1
