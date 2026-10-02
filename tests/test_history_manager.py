from __future__ import annotations

import json
from pathlib import Path
import select
import subprocess
import sys

import pytest

from chartreux.cli.history_manager import HistoryManager


def submit(manager: HistoryManager, text: str) -> None:
    manager.add(text)
    manager.reset_navigation()
    manager.persist(text)


def test_history_manager_normalizes_loaded_entries_like_numbers_to_strings(
    tmp_path: Path,
) -> None:
    # ideally, we would not use real I/O; but this test is a quick bugfix, thus it
    # does not intend to refactor the HistoryManager
    history_file = tmp_path / "history.jsonl"
    history_entries = ["hello", 123]
    history_file.write_text(
        "\n".join(json.dumps(entry) for entry in history_entries) + "\n",
        encoding="utf-8",
    )
    manager = HistoryManager(history_file)

    result = manager.get_previous(current_input="")

    assert result == "123"


def test_history_manager_retains_a_fixed_number_of_entries(tmp_path: Path) -> None:
    history_file = tmp_path / "history.jsonl"
    manager = HistoryManager(history_file, max_entries=3)

    submit(manager, "first")
    submit(manager, "second")
    submit(manager, "third")
    submit(manager, "fourth")

    reloaded = HistoryManager(history_file)

    assert reloaded.get_previous(current_input="") == "fourth"
    assert reloaded.get_previous(current_input="") == "third"
    assert reloaded.get_previous(current_input="") == "second"
    # "first" is not proposed as we defined number of entries to 3
    assert reloaded.get_previous(current_input="") is None


def test_history_manager_filters_invalid_and_duplicated_entries(tmp_path: Path) -> None:
    history_file = tmp_path / "history.jsonl"
    manager = HistoryManager(history_file, max_entries=5)
    submit(manager, "")  # empty
    submit(manager, "   ")  # is trimmed
    submit(manager, "first")
    submit(manager, "second")
    submit(manager, "second")  # duplicate
    submit(manager, "third")

    reloaded = HistoryManager(history_file)

    assert reloaded.get_previous(current_input="") == "third"
    assert reloaded.get_previous(current_input="") == "second"
    assert reloaded.get_previous(current_input="") == "first"
    assert reloaded.get_previous(current_input="") is None
    assert reloaded.get_previous(current_input="") is None


def test_history_manager_stores_slash_prefixed_entries(tmp_path: Path) -> None:
    history_file = tmp_path / "history.jsonl"
    manager = HistoryManager(history_file, max_entries=5)
    submit(manager, "first")
    submit(manager, "/tool_call arg1 arg2")

    reloaded = HistoryManager(history_file)

    assert reloaded.get_previous(current_input="") == "/tool_call arg1 arg2"
    assert reloaded.get_previous(current_input="") == "first"
    assert reloaded.get_previous(current_input="") is None


def test_history_manager_keeps_entries_when_reload_read_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    history_file = tmp_path / "history.jsonl"
    manager = HistoryManager(history_file)
    submit(manager, "first")

    def raise_os_error(*args: object, **kwargs: object) -> None:
        raise PermissionError("history is unreadable")

    before = history_file.read_bytes()
    # Simulate a newer session so a stale in-memory fallback would lose data.
    history_file.write_text(before.decode() + json.dumps("other session") + "\n")
    before = history_file.read_bytes()
    with monkeypatch.context() as patch:
        patch.setattr("chartreux.cli.history_manager.read_safe", raise_os_error)
        submit(manager, "second")
        assert history_file.read_bytes() == before
        assert manager._pending_entries == ["second"]

    assert manager.get_previous(current_input="") == "second"
    assert manager.get_previous(current_input="") == "first"
    manager.persist("second")
    assert manager._pending_entries == []
    assert [json.loads(line) for line in history_file.read_text().splitlines()] == [
        "first",
        "other session",
        "second",
    ]


def test_history_manager_merges_other_sessions_entries_on_persist(
    tmp_path: Path,
) -> None:
    history_file = tmp_path / "history.jsonl"
    manager = HistoryManager(history_file, max_entries=2)
    submit(manager, "a")
    submit(manager, "b")

    other = HistoryManager(history_file, max_entries=10)
    submit(other, "c")
    submit(other, "d")
    submit(other, "e")

    assert manager.get_previous(current_input="") == "b"
    submit(manager, "e")

    assert manager.get_previous(current_input="") == "e"
    assert manager.get_previous(current_input="") == "d"
    assert manager.get_previous(current_input="") is None


def test_history_manager_persists_pending_entries_in_submission_order(
    tmp_path: Path,
) -> None:
    history_file = tmp_path / "history.jsonl"
    manager = HistoryManager(history_file)

    manager.add("first")
    manager.reset_navigation()
    manager.add("second")
    manager.reset_navigation()
    manager.persist("second")
    manager.persist("first")

    reloaded = HistoryManager(history_file)

    assert reloaded.get_previous(current_input="") == "second"
    assert reloaded.get_previous(current_input="") == "first"
    assert reloaded.get_previous(current_input="") is None


def test_history_manager_keeps_pending_entries_when_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    history_file = tmp_path / "history.jsonl"
    manager = HistoryManager(history_file)
    original_write_entries = manager._write_entries
    calls = 0

    def fail_once(entries: list[str]) -> bool:
        nonlocal calls
        calls += 1
        if calls == 1:
            return False
        return original_write_entries(entries)

    monkeypatch.setattr(manager, "_write_entries", fail_once)

    manager.add("first")
    manager.persist("first")
    manager.persist("first")

    reloaded = HistoryManager(history_file)

    assert reloaded.get_previous(current_input="") == "first"
    assert reloaded.get_previous(current_input="") is None


def test_history_manager_clamps_navigation_after_entries_shrink(tmp_path: Path) -> None:
    history_file = tmp_path / "history.jsonl"
    manager = HistoryManager(history_file)

    manager.add("first")
    manager.add("second")
    manager.add("third")

    assert manager.get_previous(current_input="") == "third"
    manager._entries = ["only"]

    assert manager.get_previous(current_input="") == "only"
    assert manager.get_previous(current_input="") is None


def test_history_manager_allows_navigation_round_trip(tmp_path: Path) -> None:
    history_file = tmp_path / "history.jsonl"
    manager = HistoryManager(history_file)

    manager.add("alpha")
    manager.add("beta")

    assert manager.get_previous(current_input="typed") == "beta"
    assert manager.get_previous(current_input="typed") == "alpha"
    assert manager.get_next() == "beta"
    assert manager.get_next() == "typed"
    assert manager.get_next() is None


def test_history_manager_preserves_original_draft_during_navigation(
    tmp_path: Path,
) -> None:
    history_file = tmp_path / "history.jsonl"
    manager = HistoryManager(history_file)

    manager.add("foo")
    manager.add("bar")
    manager.add("fizz")

    assert manager.get_previous(current_input="draft") == "fizz"
    assert manager.get_previous(current_input="overwritten draft") == "bar"
    assert manager.get_next() == "fizz"
    assert manager.get_next() == "draft"


# Child interpreters import production code only, never pytest's test modules.
_HISTORY_CHILD_SCRIPT = """
import fcntl
from pathlib import Path
import sys

from chartreux.cli.history_manager import HistoryManager

mode, filename, prefix, capacity = sys.argv[1:]
history_file = Path(filename)

if mode == "hold":
    with history_file.with_name(history_file.name + ".lock").open("a") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        print("ready", flush=True)
        assert sys.stdin.buffer.read(1) == b"x"
else:
    manager = HistoryManager(history_file, max_entries=int(capacity))
    original_read = manager._read_entries
    original_flock = fcntl.flock

    def controlled_read():
        entries = original_read()
        print("read", flush=True)
        assert sys.stdin.buffer.read(1) == b"x"
        return entries

    def observed_flock(fd, operation):
        if operation == fcntl.LOCK_EX:
            print("attempting", flush=True)
        original_flock(fd, operation)

    manager._read_entries = controlled_read
    fcntl.flock = observed_flock
    for index in range(4):
        manager.add(f"{prefix}-{index}")
    manager.persist(prefix)
"""


def _start_history_child(
    history_file: Path, prefix: str, max_entries: int, *, mode: str = "write"
) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [
            sys.executable,
            "-c",
            _HISTORY_CHILD_SCRIPT,
            mode,
            str(history_file),
            prefix,
            str(max_entries),
        ],
        cwd=Path(__file__).resolve().parents[1],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        # Unbuffered reads keep select() and readline() in agreement.
        bufsize=0,
    )


def _wait_for_child_signal(process: subprocess.Popen[bytes], signal: bytes) -> None:
    assert process.stdout is not None
    assert select.select([process.stdout], [], [], 10)[0], "child handshake timed out"
    assert process.stdout.readline() == signal + b"\n"


def _release_history_child(process: subprocess.Popen[bytes]) -> None:
    assert process.stdin is not None
    process.stdin.write(b"x")
    process.stdin.flush()


def _assert_child_succeeded(process: subprocess.Popen[bytes]) -> None:
    _, stderr = process.communicate(timeout=10)
    assert process.returncode == 0, stderr.decode()


@pytest.mark.parametrize("max_entries", [100, 3])
def test_history_manager_serializes_process_read_merge_replace(
    tmp_path: Path, max_entries: int
) -> None:
    history_file = tmp_path / "history.jsonl"
    processes = []
    first = _start_history_child(history_file, "first", max_entries)
    processes.append(first)
    try:
        _wait_for_child_signal(first, b"attempting")
        _wait_for_child_signal(first, b"read")
        second = _start_history_child(history_file, "second", max_entries)
        processes.append(second)
        _release_history_child(second)
        _wait_for_child_signal(second, b"attempting")
        # The second process cannot read the stale snapshot held by the first.
        assert not select.select([second.stdout], [], [], 0.2)[0]
        _release_history_child(first)
        _assert_child_succeeded(first)
        _assert_child_succeeded(second)
        expected = [f"{prefix}-{i}" for prefix in ("first", "second") for i in range(4)]
        assert [json.loads(line) for line in history_file.read_text().splitlines()] == (
            expected[-max_entries:]
        )
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=10)


def test_history_manager_initializes_absent_history(tmp_path: Path) -> None:
    history_file = tmp_path / "new-directory" / "history.jsonl"
    manager = HistoryManager(history_file)
    submit(manager, "first")
    assert history_file.read_text() == '"first"\n'
    assert manager._pending_entries == []


def test_history_manager_lock_released_when_holder_is_killed(tmp_path: Path) -> None:
    history_file = tmp_path / "history.jsonl"
    processes = []
    holder = _start_history_child(history_file, "", 100, mode="hold")
    processes.append(holder)
    try:
        _wait_for_child_signal(holder, b"ready")
        writer = _start_history_child(history_file, "writer", 100)
        processes.append(writer)
        _release_history_child(writer)
        _wait_for_child_signal(writer, b"attempting")
        assert not select.select([writer.stdout], [], [], 0.2)[0]
        holder.kill()
        holder.communicate(timeout=10)
        _assert_child_succeeded(writer)
        assert len(history_file.read_text().splitlines()) == 4
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=10)
