from __future__ import annotations

from datetime import timedelta
import multiprocessing
from multiprocessing.process import BaseProcess
from pathlib import Path
from queue import Empty
from typing import Any

from git import Repo
import pytest

from chartreux.core.git.worktree import (
    SNAPSHOT_REF_PREFIX,
    ManagedWorktree,
    PreparedWorktree,
    WorktreeError,
    WorktreeReleaseOutcome,
    WorktreeRepository,
    abort_transfer,
    begin_transfer,
    remove_if_unheld,
)
from chartreux.core.git.worktree.record import (
    AcquireResult,
    HolderKind,
    WorktreeClaim,
    WorktreeRecord,
    acquire_holder,
    inspect_holders,
    managed_bucket_name,
    release_holder,
)

_TIMEOUT = 5


def _init_repo(root: Path) -> Repo:
    repo = Repo.init(root, initial_branch="main")
    repo.config_writer().set_value("user", "name", "Tester").release()
    repo.config_writer().set_value("user", "email", "t@example.com").release()
    (root / "file.txt").write_text("hello\n")
    repo.index.add(["file.txt"])
    repo.index.commit("initial")
    return repo


def _prepare(base: Path, name: str) -> Any:
    with WorktreeRepository.open(base) as repository:
        prepared = repository.prepare(name)
    assert prepared.pending_token is not None
    release_holder(prepared.pending_token)
    return prepared


def _claim(base: Path, name: str) -> WorktreeClaim:
    with WorktreeRepository.open(base) as repository:
        paths = repository._paths
        return WorktreeClaim(
            bucket=managed_bucket_name(paths.repo_root, paths.common_git_dir), name=name
        )


def _join(process: BaseProcess) -> None:
    process.join(_TIMEOUT)
    assert not process.is_alive(), "worker did not finish"
    assert process.exitcode == 0


def _result(queue: Any) -> Any:
    try:
        return queue.get(timeout=_TIMEOUT)
    except Empty as exc:  # pragma: no cover - assertion makes failures useful
        raise AssertionError("worker did not report a result") from exc


def _hold_attachment(cwd: Path, acquired: Any, release: Any, result: Any) -> None:
    held = WorktreeRepository.acquire_for_attachment(cwd)
    result.put(held.outcome.value)
    acquired.set()
    release.wait(_TIMEOUT)
    if held.token is not None:
        release_holder(held.token)


def _remove(cwd: Path, start: Any, result: Any) -> None:
    start.wait(_TIMEOUT)
    result.put(remove_if_unheld(cwd).outcome.value)


def _starting_creation(
    base: Path, name: str, ready: Any, release: Any, result: Any
) -> None:
    claim = _claim(base, name)
    target = WorktreeRepository.open(base)
    with target as repository:
        root = repository.worktree_root / name
        root.mkdir(parents=True)
        record = WorktreeRecord.new(
            name=name, branch=name, repo_root=repository.root, branch_created=True
        )
        record = record.model_copy(
            update={"claimed_at": record.claimed_at - timedelta(minutes=30)}
        )
    claim.write(record)
    acquired = acquire_holder(
        claim,
        "creating",
        kind=HolderKind.STARTING,
        expected_generation=record.claimed_at.isoformat(),
    )
    result.put(acquired.outcome.value)
    ready.set()
    release.wait(_TIMEOUT)
    if acquired.token is not None:
        release_holder(acquired.token)


def _acquire_starting(
    claim: WorktreeClaim, generation: str, ready: Any, release: Any, result: Any
) -> None:
    acquired = acquire_holder(
        claim, "starting", kind=HolderKind.STARTING, expected_generation=generation
    )
    result.put(acquired.outcome.value)
    ready.set()
    release.wait(_TIMEOUT)
    if acquired.token is not None:
        release_holder(acquired.token)


def _crash_while_holding(claim: WorktreeClaim, generation: str, ready: Any) -> None:
    acquired = acquire_holder(
        claim,
        "crashed-session",
        kind=HolderKind.SESSION,
        expected_generation=generation,
    )
    assert acquired.outcome is AcquireResult.SUCCESS
    ready.set()
    # Deliberately do not release: process exit drops flock, leaving a stale file.


def test_creation_starting_holder_blocks_a_concurrent_sweep(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    context = multiprocessing.get_context("spawn")
    ready, release = context.Event(), context.Event()
    results = context.Queue()
    process = context.Process(
        target=_starting_creation, args=(tmp_path, "creating", ready, release, results)
    )
    process.start()
    assert ready.wait(_TIMEOUT)
    assert _result(results) == "success"

    # The explicit barrier places sweep exactly while the creator owns STARTING.
    WorktreeRepository.sweep_claims(tmp_path, in_use=[])
    claim = _claim(tmp_path, "creating")
    assert claim.read() is not None
    assert inspect_holders(claim).starting is True

    release.set()
    _join(process)


def test_attachment_blocks_concurrent_removal(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    worktree = _prepare(tmp_path, "attached")
    context = multiprocessing.get_context("spawn")
    acquired, release = context.Event(), context.Event()
    results = context.Queue()
    process = context.Process(
        target=_hold_attachment, args=(worktree.root, acquired, release, results)
    )
    process.start()
    assert acquired.wait(_TIMEOUT)
    assert _result(results) == "success"

    assert remove_if_unheld(worktree.root).outcome is WorktreeReleaseOutcome.KEPT_IN_USE
    assert worktree.root.exists()
    release.set()
    _join(process)


def test_two_processes_cannot_both_remove_a_worktree(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    worktree = _prepare(tmp_path, "pruned")
    context = multiprocessing.get_context("spawn")
    start, results = context.Event(), context.Queue()
    workers = [
        context.Process(target=_remove, args=(worktree.root, start, results))
        for _ in range(2)
    ]
    for worker in workers:
        worker.start()
    start.set()
    outcomes = {_result(results), _result(results)}
    for worker in workers:
        _join(worker)

    assert outcomes == {"removed", "kept_unmanaged"}
    assert not worktree.root.exists()


def test_same_process_session_holders_are_reference_counted(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    worktree = _prepare(tmp_path, "counted")
    claim = _claim(tmp_path, worktree.name)
    record = claim.read()
    assert record is not None
    generation = record.claimed_at.isoformat()
    first = acquire_holder(
        claim, "session", kind=HolderKind.SESSION, expected_generation=generation
    )
    second = acquire_holder(
        claim, "session", kind=HolderKind.SESSION, expected_generation=generation
    )
    assert first.token is not None and second.token is not None

    release_holder(first.token)
    assert inspect_holders(claim).holders == {"session": 1}
    release_holder(second.token)
    assert inspect_holders(claim).holders == {}


def test_duplicate_starting_is_rejected_in_one_process(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    worktree = _prepare(tmp_path, "starting")
    claim = _claim(tmp_path, worktree.name)
    record = claim.read()
    assert record is not None
    first = acquire_holder(
        claim,
        "one",
        kind=HolderKind.STARTING,
        expected_generation=record.claimed_at.isoformat(),
    )
    second = acquire_holder(
        claim,
        "two",
        kind=HolderKind.STARTING,
        expected_generation=record.claimed_at.isoformat(),
    )
    assert first.outcome is AcquireResult.SUCCESS
    assert second.outcome is AcquireResult.CONFLICT
    assert first.token is not None
    release_holder(first.token)


def test_duplicate_starting_is_rejected_across_processes(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    worktree = _prepare(tmp_path, "cross-starting")
    claim = _claim(tmp_path, worktree.name)
    record = claim.read()
    assert record is not None
    context = multiprocessing.get_context("spawn")
    ready, release = context.Event(), context.Event()
    results = context.Queue()
    process = context.Process(
        target=_acquire_starting,
        args=(claim, record.claimed_at.isoformat(), ready, release, results),
    )
    process.start()
    assert ready.wait(_TIMEOUT)
    assert _result(results) == "success"

    conflict = acquire_holder(
        claim,
        "parent",
        kind=HolderKind.STARTING,
        expected_generation=record.claimed_at.isoformat(),
    )
    assert conflict.outcome is AcquireResult.CONFLICT
    release.set()
    _join(process)


def test_stale_holder_file_is_reclaimed_after_process_crash(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    worktree = _prepare(tmp_path, "crashed")
    claim = _claim(tmp_path, worktree.name)
    record = claim.read()
    assert record is not None
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    process = context.Process(
        target=_crash_while_holding, args=(claim, record.claimed_at.isoformat(), ready)
    )
    process.start()
    assert ready.wait(_TIMEOUT)
    _join(process)
    crashed_marker = claim.directory / "holders" / "crashed-session"
    assert crashed_marker.exists()

    snapshot = inspect_holders(claim)
    assert snapshot.outcome is AcquireResult.SUCCESS
    assert "crashed-session" not in snapshot.holders
    assert not crashed_marker.exists()


def test_transfer_abort_releases_target_and_keeps_source(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    source, target = _prepare(tmp_path, "source"), _prepare(tmp_path, "target")
    source_managed = ManagedWorktree.at(source.root)
    target_managed = ManagedWorktree.at(target.root)
    assert source_managed is not None and target_managed is not None
    source_token = source_managed.hold("session").token
    assert source_token is not None

    attempt = begin_transfer(source_token, target.root)
    abort_transfer(attempt)
    assert source_managed.holders() == {"session"}
    assert target_managed.holders() == set()
    release_holder(source_token)


def test_cli_hold_allows_attachment_but_prevents_prompt_cleanup(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    worktree = _prepare(tmp_path, "cli-prompt")
    claim = _claim(tmp_path, worktree.name)
    record = claim.read()
    assert record is not None
    cli = acquire_holder(
        claim,
        "cli",
        kind=HolderKind.CLI,
        expected_generation=record.claimed_at.isoformat(),
    )
    assert cli.token is not None
    context = multiprocessing.get_context("spawn")
    acquired, release = context.Event(), context.Event()
    results = context.Queue()
    process = context.Process(
        target=_hold_attachment, args=(worktree.root, acquired, release, results)
    )
    process.start()
    try:
        # The child represents the app-server session reaching the CLI prompt.
        assert acquired.wait(_TIMEOUT)
        assert _result(results) == "success"
        assert (
            remove_if_unheld(worktree.root, retiring_token=cli.token).outcome
            is WorktreeReleaseOutcome.KEPT_IN_USE
        )
        assert worktree.root.exists()
    finally:
        release.set()
        _join(process)


def test_incomplete_claim_with_commits_is_not_reaped(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    worktree = _prepare(tmp_path, "recover")
    checkout = Repo(worktree.root)
    (worktree.root / "file.txt").write_text("valuable\n")
    checkout.index.add(["file.txt"])
    checkout.index.commit("valuable commit")
    claim = _claim(tmp_path, worktree.name)
    record = claim.read()
    assert record is not None
    claim.write(
        record.model_copy(
            update={
                "base_commit": None,
                "claimed_at": record.claimed_at - timedelta(minutes=30),
            }
        )
    )

    WorktreeRepository.sweep_claims(tmp_path, in_use=[])
    assert worktree.root.exists()
    assert Repo(worktree.root).head.commit.message.strip() == "valuable commit"
    assert repo.git.worktree("list")


def test_failed_removal_after_snapshot_leaves_checkout_recoverable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _init_repo(tmp_path)
    worktree = _prepare(tmp_path, "snapshot-failure")
    (worktree.root / "valuable.txt").write_text("valuable\n")
    managed = ManagedWorktree.at(worktree.root)
    assert managed is not None

    def fail_remove(self: PreparedWorktree, *, delete_branch: bool = True) -> None:
        raise WorktreeError("simulated deletion failure")

    monkeypatch.setattr(PreparedWorktree, "remove", fail_remove)
    with pytest.raises(WorktreeError, match="simulated deletion failure"):
        managed.remove_if_unheld()

    assert worktree.root.exists()
    assert (worktree.root / "valuable.txt").read_text() == "valuable\n"
    assert repo.commit(f"{SNAPSHOT_REF_PREFIX}/{worktree.name}")
    assert _claim(tmp_path, worktree.name).read() is not None
