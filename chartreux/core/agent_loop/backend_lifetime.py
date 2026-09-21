from __future__ import annotations

import asyncio
from collections.abc import Generator
import contextlib

from chartreux.core.llm.types import BackendLike

# Hold strong references to background backend-close tasks so CPython doesn't
# GC them mid-execution (the loop keeps only weak refs to tasks). Discarded on
# completion. Matches the retention pattern used elsewhere in the codebase.
_pending_close_tasks: set[asyncio.Task[None]] = set()


def _close_backend_in_background(backend: BackendLike) -> asyncio.Task[None]:
    """Close a backend without blocking the caller."""

    async def _close() -> None:
        with contextlib.suppress(Exception):
            await backend.__aexit__(None, None, None)

    task = asyncio.create_task(_close())
    _pending_close_tasks.add(task)

    def _on_done(done: asyncio.Task[None]) -> None:
        _pending_close_tasks.discard(done)
        if not done.cancelled():
            done.exception()  # mark retrieved, suppress "never retrieved" warning

    task.add_done_callback(_on_done)
    return task


class BackendPublish:
    """A reversible active-backend publication awaiting final retirement."""

    def __init__(
        self, owner: BackendLifetime, previous: BackendLike, replacement: BackendLike
    ) -> None:
        self._owner = owner
        self.previous = previous
        self.replacement = replacement
        self._done = previous is replacement
        if not self._done:
            owner._publications.add(self)

    def rollback(self) -> bool:
        """Undo this publication when current; retire it when superseded.

        Returns whether the prior backend was restored. A newer publication always
        wins, so completing a superseded publication only retires its older backend.
        """
        if self._done:
            return False
        restored = self._owner._active is self.replacement
        if restored:
            self._owner._active = self.previous
            self._owner.retire(self.replacement, whole_turn_active=False)
        else:
            self._owner.retire(self.previous, whole_turn_active=False)
        self._complete()
        return restored

    def finalize(self, *, whole_turn_active: bool) -> None:
        if self._done:
            return
        self._owner.retire(self.previous, whole_turn_active=whole_turn_active)
        self._complete()

    def _complete(self) -> None:
        self._done = True
        self._owner._publications.discard(self)


class BackendLifetime:
    """Own the active backend and close retired backends when they are safe."""

    def __init__(self, initial: BackendLike) -> None:
        self._active = initial
        self._retired: list[BackendLike] = []
        self._borrow_counts: dict[int, int] = {}
        self._close_tasks: set[asyncio.Task[None]] = set()
        self._publications: set[BackendPublish] = set()
        self._closed = False

    @property
    def active(self) -> BackendLike:
        """Read-only compatibility access; no ownership transfer."""
        return self._active

    @contextlib.contextmanager
    def borrow(self) -> Generator[BackendLike, None, None]:
        """Pin the active backend for an operation without transferring ownership."""
        if self._closed:
            raise RuntimeError("BackendLifetime is closed")
        backend = self._active
        key = id(backend)
        self._borrow_counts[key] = self._borrow_counts.get(key, 0) + 1
        try:
            yield backend
        finally:
            remaining = self._borrow_counts[key] - 1
            if remaining:
                self._borrow_counts[key] = remaining
            else:
                del self._borrow_counts[key]

    def publish_reversible(self, replacement: BackendLike) -> BackendPublish:
        """Publish without retiring the prior backend until finalize()."""
        if self._closed:
            raise RuntimeError("BackendLifetime is closed")
        previous = self._active
        self._active = replacement
        return BackendPublish(self, previous, replacement)

    def replace(self, replacement: BackendLike, *, whole_turn_active: bool) -> None:
        """Publish *replacement* and retire the previously active backend."""
        if replacement is self._active:
            return
        previous = self._active
        self._active = replacement
        self.retire(previous, whole_turn_active=whole_turn_active)

    def retire(self, backend: BackendLike, *, whole_turn_active: bool) -> None:
        """Retire *backend*, closing it only once no turn or borrow protects it."""
        if self._closed:
            return
        if any(backend is retired for retired in self._retired):
            return
        self._retired.append(backend)
        if not whole_turn_active and self._borrow_counts.get(id(backend), 0) == 0:
            self._schedule_eligible_closes()

    def drain(self, *, whole_turn_active: bool) -> None:
        """Schedule deferred closes when the owner has established turn idleness."""
        if self._closed:
            return
        if whole_turn_active:
            raise RuntimeError("Cannot drain backend closes while a turn is active")
        self._schedule_eligible_closes()

    def _schedule_eligible_closes(self) -> None:
        pending = self._retired
        self._retired = []
        for backend in pending:
            if self._borrow_counts.get(id(backend), 0):
                self._retired.append(backend)
                continue
            task = _close_backend_in_background(backend)
            self._close_tasks.add(task)
            task.add_done_callback(self._close_tasks.discard)

    async def aclose(self) -> None:
        """Reject new borrows, close owned backends, and join close tasks."""
        if self._closed:
            return
        self._closed = True
        backends = [self._active, *self._retired]
        for publication in self._publications:
            backends.extend((publication.previous, publication.replacement))
        self._publications.clear()
        self._retired = []
        unique_backends = list({id(backend): backend for backend in backends}.values())
        # Backends already handed to close tasks remain owned by those tasks. Join
        # them instead of issuing a second close concurrently.
        await asyncio.gather(
            *(backend.__aexit__(None, None, None) for backend in unique_backends),
            *self._close_tasks,
            return_exceptions=True,
        )
