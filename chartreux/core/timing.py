"""Task-local elapsed clocks for static, completed operation timings."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import time

_clock: ContextVar[Callable[[], float] | None] = ContextVar(
    "elapsed_clock", default=None
)


def elapsed_time() -> float:
    clock = _clock.get()
    return clock() if clock is not None else time.perf_counter()


@contextmanager
def use_elapsed_clock(clock: Callable[[], float]) -> Iterator[None]:
    """Inject a clock without changing the process-wide performance counter."""
    token = _clock.set(clock)
    try:
        yield
    finally:
        _clock.reset(token)


@dataclass(frozen=True, slots=True)
class CompletedTurnTiming:
    """The retained prose owner and frozen whole-invocation seconds."""

    message_id: str
    duration: float


@dataclass(slots=True)
class InvocationTiming:
    """One invocation's frozen seconds; unstarted is distinct from measured zero."""

    started_at: float | None = None
    duration: float | None = None

    def start(self) -> None:
        self.started_at = elapsed_time()

    def finish(self) -> float | None:
        if self.started_at is not None and self.duration is None:
            self.duration = max(0.0, elapsed_time() - self.started_at)
        return self.duration

    @property
    def duration_ms(self) -> float | None:
        return self.duration * 1000.0 if self.duration is not None else None
