"""Shared runtime-measurement primitives for the stage-2 performance packages.

Nothing here changes production behaviour. The path guard keeps every
measurement artifact outside the repository tree, and the C1/C2 observers wrap
the landed P5 save path from the test side: the worker wrapper times the
synchronous scan, the await observer times the lock-held ``to_thread`` call
including its executor dispatch, and both leave ordering, cancellation and
thread identity exactly as production has them.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
import functools
import os
from pathlib import Path
import threading
from time import perf_counter_ns
from types import ModuleType
from typing import Any

RUN_DIR_ENV = "CHARTREUX_PERF_RUN_DIR"
REPORT_ENV = "CHARTREUX_PERF_REPORT"
SCRATCHPAD_ENV = "CHARTREUX_PERF_SCRATCHPAD"
SCRATCHPAD_DIR_PREFIX = "chartreux-scratchpad-"

_NANOSECONDS_PER_MILLISECOND = 1_000_000
_MISSING: Any = object()


class PerfPathError(RuntimeError):
    """A measurement path escapes the worker output boundary."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"perf path rejected ({reason}): {detail}")
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True, slots=True)
class PerfPaths:
    """The resolved, boundary-checked measurement paths of one run."""

    run_dir: Path | None
    report: Path | None
    scratchpad_root: Path | None
    repo_root: Path


def repository_root() -> Path:
    """Return the repository root that contains this module."""
    return Path(__file__).resolve().parents[2]


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _is_at_or_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _is_strictly_within(path: Path, root: Path) -> bool:
    return root in path.parents


def _checked_path(path_text: str, label: str) -> Path:
    """Return the canonical form of one configured path, or raise on an escape."""
    candidate = Path(path_text)
    if not candidate.is_absolute():
        raise PerfPathError(
            f"relative_{label}", f"{label} must be absolute: {path_text}"
        )
    if Path(os.path.abspath(path_text)) != candidate:
        raise PerfPathError(
            f"{label}_not_canonical",
            f"{label} carries undeclared traversal: {path_text}",
        )
    resolved = candidate.resolve()
    if resolved != candidate:
        raise PerfPathError(
            f"{label}_symlink_traversal_escape",
            f"{label} traverses a symlink: {path_text} resolves to {resolved}",
        )
    return resolved


def _discover_scratchpad_root(path: Path) -> Path | None:
    for parent in path.parents:
        if parent.name.startswith(SCRATCHPAD_DIR_PREFIX):
            return parent
    return None


def resolve_perf_paths(
    environ: Mapping[str, str] | None = None,
    *,
    scratchpad_root: Path | None = None,
    repo_root: Path | None = None,
) -> PerfPaths:
    """Resolve and validate the RUN_DIR/report pair for one measurement.

    ``RUN_DIR`` must be canonical, an existing directory, a strict descendant
    of the session scratchpad, and disjoint from the repository root in both
    directions. ``CHARTREUX_PERF_REPORT`` is optional; when set it must be an
    absolute canonical path strictly inside ``RUN_DIR`` and outside the
    repository tree.
    """
    source = os.environ if environ is None else environ
    root = (repository_root() if repo_root is None else repo_root).resolve()
    raw_run_dir = _clean(source.get(RUN_DIR_ENV))
    raw_report = _clean(source.get(REPORT_ENV))

    if raw_report is not None and raw_run_dir is None:
        raise PerfPathError(
            "report_without_run_dir", f"{REPORT_ENV} is set while {RUN_DIR_ENV} is not"
        )
    if raw_report is not None and not Path(raw_report).is_absolute():
        raise PerfPathError(
            "relative_report", f"{REPORT_ENV} must be absolute: {raw_report}"
        )

    if raw_run_dir is None:
        if scratchpad_root is not None:
            scratchpad_root = scratchpad_root.resolve()
        return PerfPaths(None, None, scratchpad_root, root)

    run_dir = _checked_path(raw_run_dir, "run_dir")
    if _is_at_or_within(run_dir, root):
        raise PerfPathError(
            "run_dir_under_repo_root", f"{run_dir} is at or under {root}"
        )
    if run_dir in root.parents:
        raise PerfPathError(
            "run_dir_ancestor_of_repo_root", f"{run_dir} is an ancestor of {root}"
        )

    scratchpad = scratchpad_root.resolve() if scratchpad_root is not None else None
    if scratchpad is None:
        declared = _clean(source.get(SCRATCHPAD_ENV))
        if declared is not None:
            scratchpad = Path(declared).resolve()
    if scratchpad is None:
        scratchpad = _discover_scratchpad_root(run_dir)
    if scratchpad is None or not _is_strictly_within(run_dir, scratchpad):
        raise PerfPathError(
            "run_dir_outside_scratchpad",
            f"{run_dir} is not a strict descendant of the session scratchpad"
            f" ({scratchpad})",
        )
    if not run_dir.is_dir():
        raise PerfPathError(
            "run_dir_missing", f"{run_dir} is not an existing directory"
        )

    report: Path | None = None
    if raw_report is not None:
        report = _checked_path(raw_report, "report")
        if _is_at_or_within(report, root):
            raise PerfPathError(
                "report_under_repo_root", f"{report} is at or under {root}"
            )
        if report in root.parents:
            raise PerfPathError(
                "report_ancestor_of_repo_root", f"{report} is an ancestor of {root}"
            )
        if report == run_dir:
            raise PerfPathError(
                "report_equal_run_dir", f"{report} is the run directory itself"
            )
        if not _is_strictly_within(report, run_dir):
            raise PerfPathError(
                "report_outside_run_dir",
                f"{report} is not a strict descendant of {run_dir}",
            )
        if report.is_dir():
            raise PerfPathError("report_is_directory", f"{report} is a directory")

    return PerfPaths(
        run_dir=run_dir, report=report, scratchpad_root=scratchpad, repo_root=root
    )


def current_perf_paths() -> PerfPaths:
    """Resolve the measurement paths from the live environment."""
    return resolve_perf_paths()


@dataclass(frozen=True, slots=True)
class WorkerSample:
    """One measurement of a synchronous worker's own execution wall."""

    target: str
    duration_ms: float
    started_ns: int
    finished_ns: int
    thread_ident: int
    thread_name: str
    outcome: str
    error_type: str | None


@dataclass(frozen=True, slots=True)
class AwaitSample:
    """One measurement of an awaited ``asyncio.to_thread`` call.

    The span opens when the awaiting coroutine enters the call (so it includes
    the executor dispatch) and closes when that await returns, raises or is
    cancelled.
    """

    target: str
    duration_ms: float
    started_ns: int
    finished_ns: int
    thread_ident: int
    outcome: str
    error_type: str | None


@dataclass(slots=True)
class WorkerObserver:
    """Samples collected by :func:`observe_worker`."""

    target: str
    samples: list[WorkerSample] = field(default_factory=list)

    def durations_ms(self) -> list[float]:
        return [sample.duration_ms for sample in self.samples]


@dataclass(slots=True)
class LockAwaitObserver:
    """Samples collected by :func:`observe_lock_held_await`."""

    samples: list[AwaitSample] = field(default_factory=list)

    def for_target(self, target: str) -> list[AwaitSample]:
        return [sample for sample in self.samples if sample.target == target]

    def durations_ms(self, target: str | None = None) -> list[float]:
        selected = self.samples if target is None else self.for_target(target)
        return [sample.duration_ms for sample in selected]


@contextmanager
def observe_worker(
    owner: object, attribute: str, *, target: str | None = None
) -> Iterator[WorkerObserver]:
    """Time one synchronous worker attribute of ``owner`` from the test side.

    The wrapper delegates unchanged and re-raises whatever the worker raises,
    so the observed call keeps its result, its exception and its thread. It is
    installed on the instance, never on the class, and removed on exit.
    """
    original = getattr(owner, attribute)
    observer = WorkerObserver(target=target or getattr(original, "__name__", attribute))
    instance_attributes = getattr(owner, "__dict__", {})
    had_own_attribute = (
        isinstance(instance_attributes, dict) and attribute in instance_attributes
    )

    @functools.wraps(original)
    def observed(*args: Any, **kwargs: Any) -> Any:
        current = threading.current_thread()
        started = perf_counter_ns()
        outcome = "returned"
        error_type: str | None = None
        try:
            return original(*args, **kwargs)
        except BaseException as error:
            outcome = "raised"
            error_type = type(error).__name__
            raise
        finally:
            finished = perf_counter_ns()
            observer.samples.append(
                WorkerSample(
                    target=observer.target,
                    duration_ms=(finished - started) / _NANOSECONDS_PER_MILLISECOND,
                    started_ns=started,
                    finished_ns=finished,
                    thread_ident=current.ident or 0,
                    thread_name=current.name,
                    outcome=outcome,
                    error_type=error_type,
                )
            )

    setattr(owner, attribute, observed)
    try:
        yield observer
    finally:
        if had_own_attribute:
            setattr(owner, attribute, original)
        else:
            try:
                delattr(owner, attribute)
            except AttributeError:
                pass


class _AsyncioProxy:
    """Stands in for the ``asyncio`` module inside one instrumented module."""

    __slots__ = ("_observer", "_real_module", "_real_to_thread")

    def __init__(
        self,
        real_module: ModuleType,
        real_to_thread: Callable[..., Any],
        observer: LockAwaitObserver,
    ) -> None:
        self._real_module = real_module
        self._real_to_thread = real_to_thread
        self._observer = observer

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real_module, name)

    def to_thread(
        self, func: Callable[..., Any], /, *args: Any, **kwargs: Any
    ) -> Awaitable[Any]:
        return self._observed_call(func, args, kwargs)

    async def _observed_call(
        self, func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:
        target = getattr(func, "__name__", type(func).__name__)
        current = threading.current_thread()
        started = perf_counter_ns()
        outcome = "returned"
        error_type: str | None = None
        try:
            return await self._real_to_thread(func, *args, **kwargs)
        except asyncio.CancelledError:
            outcome = "cancelled"
            raise
        except BaseException as error:
            outcome = "raised"
            error_type = type(error).__name__
            raise
        finally:
            finished = perf_counter_ns()
            self._observer.samples.append(
                AwaitSample(
                    target=target,
                    duration_ms=(finished - started) / _NANOSECONDS_PER_MILLISECOND,
                    started_ns=started,
                    finished_ns=finished,
                    thread_ident=current.ident or 0,
                    outcome=outcome,
                    error_type=error_type,
                )
            )


@contextmanager
def observe_lock_held_await(
    module: object, *, attribute: str = "asyncio"
) -> Iterator[LockAwaitObserver]:
    """Time the awaited ``to_thread`` calls one module makes, C2-style.

    Only the observed module's ``asyncio`` reference is replaced (by a proxy
    that delegates every other attribute and the real ``to_thread`` itself);
    the process-wide ``asyncio`` module is untouched, and everything is
    restored on exit.
    """
    real_module = getattr(module, attribute, None)
    real_to_thread = getattr(real_module, "to_thread", None)
    if not isinstance(real_module, ModuleType) or not callable(real_to_thread):
        raise TypeError(f"{attribute}.to_thread is not callable on {module!r}")
    observer = LockAwaitObserver()
    proxy = _AsyncioProxy(real_module, real_to_thread, observer)
    previous = getattr(module, "__dict__", {}).get(attribute, _MISSING)
    setattr(module, attribute, proxy)
    try:
        yield observer
    finally:
        if previous is _MISSING:
            delattr(module, attribute)
        else:
            setattr(module, attribute, previous)


@dataclass(slots=True)
class SaveScanObserver:
    """The C1 worker samples and C2 lock-held await samples of one save cell."""

    worker: WorkerObserver
    lock_await: LockAwaitObserver

    def worker_ms(self) -> list[float]:
        return self.worker.durations_ms()

    def lock_await_ms(self) -> list[float]:
        return self.lock_await.durations_ms(self.worker.target)


@contextmanager
def observe_save_scan(
    logger: object,
    module: ModuleType,
    *,
    attribute: str = "_check_transcript_persistence",
) -> Iterator[SaveScanObserver]:
    """Install both save-cell observers (C1 worker wall, C2 lock-held await)."""
    with (
        observe_worker(logger, attribute) as worker,
        observe_lock_held_await(module) as lock_await,
    ):
        yield SaveScanObserver(worker=worker, lock_await=lock_await)
