from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine, Iterable
from typing import TYPE_CHECKING, Any

from chartreux.observability.logging import logger

if TYPE_CHECKING:
    from chartreux.app_server.protocol import UsageUpdatedParams
    from chartreux.app_server.session import AppServerSession

# Poll interval for the spend-segment reconciliation loop. app.py re-exports
# this name; the poll loop reads the value through an injected getter that
# resolves the app module global at call time, so tests patching
# chartreux.cli.textual_ui.app._USAGE_POLL_INTERVAL_SECONDS keep working.
_USAGE_POLL_INTERVAL_SECONDS = 60

# Status-line segments that make the periodic usage read worthwhile.
_SPEND_POLL_SEGMENTS = frozenset({"spend-today", "spend-week", "spend-month"})


class UsageController:
    """Owns the usage resource subscription and poll tasks for one app.

    Plain object (no mixins): the app injects the host-dependent reads as
    callables, so this leaf never imports the app. The presentation reaction
    stays on the app behind the ``on_updated`` port.
    """

    def __init__(
        self,
        *,
        current_binding: Callable[[], tuple[AppServerSession, str] | None],
        is_binding_current: Callable[[tuple[AppServerSession, str]], bool],
        status_line_segments: Callable[[], Iterable[str]],
        on_updated: Callable[[UsageUpdatedParams], None],
        poll_interval: Callable[[], float],
    ) -> None:
        self._current_binding = current_binding
        self._is_binding_current = is_binding_current
        self._status_line_segments = status_line_segments
        self._on_updated = on_updated
        self._poll_interval = poll_interval
        self._binding: tuple[AppServerSession, str] | None = None
        self._unsubscribe: Callable[[], None] | None = None
        self._initial_task: asyncio.Task[None] | None = None
        self._poll_task: asyncio.Task[None] | None = None
        self._tasks: set[asyncio.Task[None]] = set()

    @property
    def poll_task(self) -> asyncio.Task[None] | None:
        """The reconciliation poll task, if one is running."""
        return self._poll_task

    @property
    def tasks(self) -> set[asyncio.Task[None]]:
        """Live usage tasks (initial read + poll), for shutdown draining."""
        return self._tasks

    def sync(self) -> None:
        """Reconcile the subscription and poll task with the current session."""
        binding = self._current_binding()
        if binding is None:
            return
        session = binding[0]
        if binding != self._binding:
            self.stop()
            self._binding = binding
            usage = session.resources.usage

            def updated(snapshot: UsageUpdatedParams) -> None:
                if self._binding == binding and self._is_binding_current(binding):
                    self._on_updated(snapshot)

            self._unsubscribe = usage.subscribe(updated)
            if usage.current is not None:
                updated(usage.current)
            # No usage RPC is awaited on the mount/readiness path.
            self._initial_task = self._track_usage_task(self._read_usage(session))
        enabled = any(
            segment in _SPEND_POLL_SEGMENTS for segment in self._status_line_segments()
        )
        if enabled and self._poll_task is None:
            self._poll_task = self._track_usage_task(self._poll_usage(session))
        elif not enabled and self._poll_task is not None:
            self._poll_task.cancel()
            self._poll_task = None

    def stop(self) -> None:
        """Tear down the subscription and cancel the tracked tasks."""
        self._binding = None
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None
        for task in (self._initial_task, self._poll_task):
            if task is not None:
                task.cancel()
        self._initial_task = None
        self._poll_task = None

    def _track_usage_task(
        self, coroutine: Coroutine[Any, Any, None]
    ) -> asyncio.Task[None]:
        task = asyncio.create_task(coroutine)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def _read_usage(self, session: AppServerSession) -> None:
        try:
            # The resource publishes reconciled global summaries to subscribers.
            await session.resources.usage.read()
        except Exception as exc:
            logger.debug("Usage refresh failed", exc_info=exc)

    async def _poll_usage(self, session: AppServerSession) -> None:
        while True:
            await asyncio.sleep(self._poll_interval())
            await self._read_usage(session)
