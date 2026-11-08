"""WP1: the measurement path guard and the C1/C2 save-path observers.

The guard is exercised through its pure entry point (an explicit environment
mapping plus fabricated scratchpad/repository roots), so every negative is
hermetic; the observer contract drives a real ``SessionLogger`` save path and
asserts that observing it changes neither ordering, cancellation, exception
propagation nor thread identity.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import sys
import threading
from unittest.mock import MagicMock

import pytest

from chartreux.core.config import ChartreuxConfigSchema, SessionLoggingConfig
from chartreux.core.llm_models import LLMMessage, Role
from chartreux.core.session import session_logger as session_logger_module
from chartreux.core.session.session_logger import SessionLogger
from chartreux.core.session_types import AgentStats
from chartreux.core.tools.manager import ToolManager
from tests.conftest import build_test_vibe_config
from tests.perf._runtime_metrics import (
    REPORT_ENV,
    RUN_DIR_ENV,
    PerfPathError,
    observe_lock_held_await,
    observe_save_scan,
    observe_worker,
    repository_root,
    resolve_perf_paths,
)

pytestmark = [
    pytest.mark.perf,
    pytest.mark.skipif(
        sys.platform != "linux",
        reason="runtime measurement primitives assume Linux thread and /proc semantics",
    ),
]

_SCAN_ATTRIBUTE = "_check_transcript_persistence"
_SAVE_COUNT = 3


def _fabricated_roots(tmp_path: Path) -> tuple[Path, Path]:
    scratchpad = tmp_path / "chartreux-scratchpad-fabricated"
    repo = tmp_path / "repo"
    scratchpad.mkdir(parents=True)
    repo.mkdir(parents=True)
    return scratchpad, repo


def _environ(run_dir: Path | str | None, report: str | None = None) -> dict[str, str]:
    environ: dict[str, str] = {}
    if run_dir is not None:
        environ[RUN_DIR_ENV] = str(run_dir)
    if report is not None:
        environ[REPORT_ENV] = report
    return environ


def _workspace(scratchpad: Path) -> Path:
    run_dir = scratchpad / "run"
    run_dir.mkdir()
    return run_dir


def _is_at_or_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _assert_run_dir_boundary(paths: Path, repo: Path, scratchpad: Path) -> None:
    assert paths.is_absolute()
    assert paths.is_dir()
    assert scratchpad in paths.parents
    assert not _is_at_or_within(paths, repo)
    assert not _is_at_or_within(repo, paths)


async def _save_repeatedly(
    logger: SessionLogger,
    messages: list[LLMMessage],
    stats: AgentStats,
    config: ChartreuxConfigSchema,
    tool_manager: ToolManager,
    save_count: int,
) -> None:
    for _ in range(save_count):
        await logger.save_interaction(
            messages, stats=stats, config=config, tool_manager=tool_manager
        )


def _assert_guard_positives(tmp_path: Path) -> None:
    if RUN_DIR_ENV in os.environ:
        live = resolve_perf_paths()
        assert live.run_dir is not None
        assert live.scratchpad_root is not None
        _assert_run_dir_boundary(live.run_dir, live.repo_root, live.scratchpad_root)
        if REPORT_ENV not in os.environ:
            assert live.report is None
    else:
        scratchpad, repo = _fabricated_roots(tmp_path / "live-guard")
        run_dir = _workspace(scratchpad)
        fabricated = resolve_perf_paths(
            _environ(run_dir), scratchpad_root=scratchpad, repo_root=repo
        )
        assert fabricated.run_dir is not None
        _assert_run_dir_boundary(fabricated.run_dir, repo, scratchpad)
        assert fabricated.report is None

    scratchpad, repo = _fabricated_roots(tmp_path / "report-guard")
    run_dir = _workspace(scratchpad)
    report = run_dir / "report.jsonl"
    accepted = resolve_perf_paths(
        _environ(run_dir, str(report)), scratchpad_root=scratchpad, repo_root=repo
    )
    assert accepted.run_dir == run_dir
    assert accepted.report == report
    assert accepted.scratchpad_root == scratchpad
    assert accepted.repo_root == repo

    bare = resolve_perf_paths({}, scratchpad_root=scratchpad, repo_root=repo)
    assert bare.run_dir is None
    assert bare.report is None


def _assert_save_observers(tmp_path: Path) -> None:
    config = build_test_vibe_config()
    stats = AgentStats()
    tool_manager = MagicMock(spec=ToolManager)
    tool_manager.available_tools = {}
    tool_manager.available_tool_specs.return_value = []
    session_config = SessionLoggingConfig(
        save_dir=str(tmp_path / "sessions"), session_prefix="wp1", enabled=True
    )
    logger = SessionLogger(session_config, "wp1-observer")
    messages = [
        LLMMessage(role=Role.user, content="first turn"),
        LLMMessage(role=Role.assistant, content="first reply"),
    ]
    loop_thread = threading.get_ident()

    with observe_save_scan(logger, session_logger_module) as scan:
        asyncio.run(
            _save_repeatedly(logger, messages, stats, config, tool_manager, _SAVE_COUNT)
        )

    assert scan.worker.target == _SCAN_ATTRIBUTE
    worker_samples = scan.worker.samples
    await_samples = scan.lock_await.for_target(_SCAN_ATTRIBUTE)
    assert len(worker_samples) == _SAVE_COUNT
    assert len(await_samples) == _SAVE_COUNT
    for worker_sample, await_sample in zip(worker_samples, await_samples, strict=True):
        assert worker_sample.target == _SCAN_ATTRIBUTE
        assert worker_sample.outcome == "returned"
        assert worker_sample.duration_ms >= 0.0
        assert worker_sample.thread_ident != loop_thread
        assert await_sample.outcome == "returned"
        assert await_sample.thread_ident == loop_thread
        assert await_sample.started_ns <= worker_sample.started_ns
        assert await_sample.finished_ns >= worker_sample.finished_ns
    assert len(scan.worker_ms()) == _SAVE_COUNT
    assert len(scan.lock_await_ms()) == _SAVE_COUNT

    assert logger.session_dir is not None
    transcript = logger.messages_filepath
    assert transcript.exists()
    assert transcript.stat().st_size > 0
    assert _SCAN_ATTRIBUTE not in vars(logger)
    assert session_logger_module.asyncio is asyncio


class _AwaitProbe:
    """Minimal stand-in for a module that awaits ``asyncio.to_thread``."""

    def __init__(self) -> None:
        self.asyncio = asyncio


class _WorkerProbe:
    def scan(self) -> None:
        raise ValueError("scan exploded")


def _assert_observer_primitives() -> None:
    probe = _AwaitProbe()
    entered = threading.Event()
    release = threading.Event()
    loop_thread = threading.get_ident()

    def blocking_scan() -> str:
        entered.set()
        release.wait(5.0)
        return "released"

    def exploding_scan() -> None:
        raise RuntimeError("scan failed")

    async def blocking_call() -> str:
        return await probe.asyncio.to_thread(blocking_scan)

    async def scenario() -> None:
        task = asyncio.create_task(blocking_call())
        for _ in range(2_000):
            if entered.is_set():
                break
            await asyncio.sleep(0.001)
        assert entered.is_set()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        release.set()
        with pytest.raises(RuntimeError):
            await probe.asyncio.to_thread(exploding_scan)

    with observe_lock_held_await(probe) as lock_await:
        asyncio.run(scenario())

    assert probe.asyncio is asyncio
    cancelled, raised = lock_await.samples
    assert cancelled.target == "blocking_scan"
    assert cancelled.outcome == "cancelled"
    assert cancelled.thread_ident == loop_thread
    assert raised.target == "exploding_scan"
    assert raised.outcome == "raised"
    assert raised.error_type == "RuntimeError"

    worker_probe = _WorkerProbe()
    with observe_worker(worker_probe, "scan") as worker_observer:
        with pytest.raises(ValueError):
            worker_probe.scan()
    assert len(worker_observer.samples) == 1
    assert worker_observer.samples[0].outcome == "raised"
    assert worker_observer.samples[0].error_type == "ValueError"
    assert "scan" not in vars(worker_probe)


@pytest.mark.timeout(8)
def test_runtime_observer_contract(tmp_path: Path) -> None:
    """The path guard and the C1/C2 observers hold their contracts."""
    _assert_guard_positives(tmp_path)
    _assert_save_observers(tmp_path)
    _assert_observer_primitives()


@pytest.mark.timeout(4)
def test_guard_rejects_relative_report(tmp_path: Path) -> None:
    """A relative report path is rejected even with a valid run directory."""
    scratchpad, repo = _fabricated_roots(tmp_path)
    run_dir = _workspace(scratchpad)
    with pytest.raises(PerfPathError, match="relative_report"):
        resolve_perf_paths(
            _environ(run_dir, "report.jsonl"),
            scratchpad_root=scratchpad,
            repo_root=repo,
        )


@pytest.mark.timeout(4)
def test_guard_rejects_report_equal_run_dir(tmp_path: Path) -> None:
    """The report may not be the run directory itself."""
    scratchpad, repo = _fabricated_roots(tmp_path)
    run_dir = _workspace(scratchpad)
    with pytest.raises(PerfPathError, match="report_equal_run_dir"):
        resolve_perf_paths(
            _environ(run_dir, str(run_dir)), scratchpad_root=scratchpad, repo_root=repo
        )


@pytest.mark.timeout(4)
def test_guard_rejects_run_dir_equal_or_under_repo_root(tmp_path: Path) -> None:
    """A run directory at or under the repository root is rejected."""
    scratchpad, repo = _fabricated_roots(tmp_path)
    with pytest.raises(PerfPathError, match="run_dir_under_repo_root"):
        resolve_perf_paths(_environ(repo), scratchpad_root=scratchpad, repo_root=repo)
    nested = repo / "tests" / "perf" / "run"
    nested.mkdir(parents=True)
    with pytest.raises(PerfPathError, match="run_dir_under_repo_root"):
        resolve_perf_paths(_environ(nested), scratchpad_root=scratchpad, repo_root=repo)
    with pytest.raises(PerfPathError, match="run_dir_under_repo_root"):
        resolve_perf_paths(_environ(repository_root()))


@pytest.mark.timeout(4)
def test_guard_rejects_run_dir_ancestor_of_repo_root(tmp_path: Path) -> None:
    """A run directory that contains the repository root is rejected."""
    scratchpad, repo = _fabricated_roots(tmp_path)
    with pytest.raises(PerfPathError, match="run_dir_ancestor_of_repo_root"):
        resolve_perf_paths(
            _environ(scratchpad.parent), scratchpad_root=scratchpad, repo_root=repo
        )
    with pytest.raises(PerfPathError, match="run_dir_ancestor_of_repo_root"):
        resolve_perf_paths(_environ(repository_root().parent))


@pytest.mark.timeout(4)
def test_guard_rejects_report_under_repo_root(tmp_path: Path) -> None:
    """A report path inside the repository tree is rejected."""
    scratchpad, repo = _fabricated_roots(tmp_path)
    run_dir = _workspace(scratchpad)
    with pytest.raises(PerfPathError, match="report_under_repo_root"):
        resolve_perf_paths(
            _environ(run_dir, str(repo / "report.jsonl")),
            scratchpad_root=scratchpad,
            repo_root=repo,
        )


@pytest.mark.timeout(4)
def test_guard_rejects_symlink_traversal_escape(tmp_path: Path) -> None:
    """Symlink and traversal escapes out of the scratchpad are rejected."""
    scratchpad, repo = _fabricated_roots(tmp_path)
    escape = scratchpad / "escape"
    escape.symlink_to(repo, target_is_directory=True)
    with pytest.raises(PerfPathError, match="symlink_traversal_escape"):
        resolve_perf_paths(
            _environ(escape / "run"), scratchpad_root=scratchpad, repo_root=repo
        )
    traversing = f"{scratchpad}/run/../run"
    with pytest.raises(PerfPathError, match="run_dir_not_canonical"):
        resolve_perf_paths(
            _environ(traversing), scratchpad_root=scratchpad, repo_root=repo
        )


@pytest.mark.timeout(4)
def test_guard_rejects_report_with_unset_run_dir(tmp_path: Path) -> None:
    """A configured report path without a run directory is rejected."""
    report = tmp_path / "report.jsonl"
    with pytest.raises(PerfPathError, match="report_without_run_dir"):
        resolve_perf_paths(
            {REPORT_ENV: str(report)},
            scratchpad_root=tmp_path,
            repo_root=tmp_path / "repo",
        )
