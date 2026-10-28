"""C5b — the first-semantic-progress deadline for resumed subagent runs.

A resumed subagent run (a WP2 resume admission — the only explicitly marked
resumed context) can stall indefinitely on its first model call: the provider
accepts the request and never streams a byte, or streams only keep-alive
frames. The advisory recovery deadlines are consulted only BETWEEN attempts,
so such an in-flight stall never trips them. This module owns the ONE hard
in-flight kill in the recovery policy: a run-scoped, run-armed,
first-semantic-progress deadline — the explicit exception to the
advisory-deadlines policy documented on :class:`RecoveryConfig`.

Lifecycle:

- The app-server resume admission arms a :class:`ResumeProgressWatch` on the
  successor run's loop (``prepare_background`` with a resume admission);
  every other admission clears any stale watch, so ordinary runs are
  structurally unaffected.
- The watch arms at the run's first conversation model call (the gateway's
  logical-call entry) and disarms at the OBSERVATION of the first semantic
  delta — a content, reasoning, or tool-call delta. A tool-call delta
  disarms at emission: a resumed run whose first tool call then executes for
  longer than the deadline is NOT killed. Heartbeat and usage-only chunks
  never disarm.
- On expiry the watch cancels the in-flight call task; the gateway translates
  that cancellation into :class:`ResumeProgressDeadlineError` — the C5a
  producer contract (``error_code = resume_progress_deadline``) — and the
  call's cleanup (settlement, rollback) joins before the typed failure
  propagates. An expiry between calls (no in-flight task) fails the run's
  next conversation model call immediately.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from chartreux.core.agent_loop.errors import AgentLoopLLMResponseError
from chartreux.core.llm_models import LLMMessage
from chartreux.core.recovery_models import MeasuredTimeoutReason
from chartreux.observability.logging import logger

if TYPE_CHECKING:
    from chartreux.core.agent_loop import AgentLoop

__all__ = [
    "ResumeProgressDeadlineError",
    "ResumeProgressWatch",
    "arm_resume_progress_deadline",
    "clear_resume_progress_deadline",
    "semantic_progress",
]


class ResumeProgressDeadlineError(AgentLoopLLMResponseError):
    """The harness-measured first-semantic-progress deadline expiry (C5b).

    The C5a producer contract: the raised deadline failure carries the
    measured timeout reason value as its ``error_code``, which the WP0
    observation carrier extracts from the exception chain — classification,
    evidence encoding, and the controller's automatic-continuation
    prohibition all key off that single contract. ``non_retryable`` keeps
    the gateway's error mapper from re-wrapping the typed failure and keeps
    it off the failover ladder (a harness-measured local stop, never a
    provider fault).
    """

    error_code = MeasuredTimeoutReason.RESUME_PROGRESS_DEADLINE.value
    non_retryable = True

    def __init__(self, deadline_seconds: float) -> None:
        self.deadline_seconds = deadline_seconds
        super().__init__(
            "no semantic progress within the resume progress deadline "
            f"({deadline_seconds:g}s)"
        )


def semantic_progress(message: LLMMessage) -> bool:
    """The semantic-progress predicate (C5b).

    Content, reasoning, or tool-call deltas are semantic progress; heartbeat
    and usage-only chunks are not. Mirrors the failover path's semantic-delta
    predicate so the deadline and the failover acceleration agree on what
    counts as progress.
    """
    return bool(message.content or message.reasoning_content or message.tool_calls)


class ResumeProgressWatch:
    """One resumed run's first-semantic-progress deadline (C5b).

    One-shot and run-scoped: armed at the first conversation model call,
    terminal at the first observed semantic delta (disarmed) or at expiry
    (expired) — whichever comes first. Expiry cancels the registered
    in-flight call task; the gateway translates that cancellation into the
    typed deadline failure after its cleanup joined.
    """

    def __init__(self, deadline_seconds: float) -> None:
        self._deadline_seconds = deadline_seconds
        self._armed = False
        self._disarmed = False
        self._expired = False
        self._handle: asyncio.TimerHandle | None = None
        self._call_task: asyncio.Task[None] | None = None

    @property
    def disarmed(self) -> bool:
        """Whether a semantic delta was already observed (the run progressed)."""
        return self._disarmed

    @property
    def expired(self) -> bool:
        """Whether the deadline expired (the run made no semantic progress)."""
        return self._expired

    def enter_call(self) -> None:
        """Register the in-flight call task; arm at the first model call.

        Raises the typed deadline failure when the deadline already expired
        between calls — a run with no observed semantic progress gets no
        further conversation model calls.
        """
        if self._disarmed:
            return
        if self._expired:
            raise ResumeProgressDeadlineError(self._deadline_seconds)
        self._call_task = asyncio.current_task()
        if not self._armed:
            self._armed = True
            self._handle = asyncio.get_running_loop().call_later(
                self._deadline_seconds, self._on_deadline
            )

    def exit_call(self) -> None:
        """Unregister the in-flight call task (cleanup may await after this)."""
        self._call_task = None

    def observe(self, message: LLMMessage) -> None:
        """The disarm boundary: the OBSERVATION of the first semantic delta."""
        if self._disarmed or self._expired:
            return
        if semantic_progress(message):
            self.disarm()

    def disarm(self) -> None:
        """Terminal disarm: cancel the pending deadline, stop watching."""
        if self._handle is not None:
            self._handle.cancel()
            self._handle = None
        self._disarmed = True

    def deadline_failure(self) -> ResumeProgressDeadlineError:
        """The typed C5a-contract failure for this watch's expiry (C5b)."""
        return ResumeProgressDeadlineError(self._deadline_seconds)

    def _on_deadline(self) -> None:
        self._expired = True
        logger.warning(
            "Resume progress deadline (%gs) expired without semantic progress; "
            "killing the in-flight model call",
            self._deadline_seconds,
        )
        task, self._call_task = self._call_task, None
        if task is not None and not task.done():
            task.cancel()


def arm_resume_progress_deadline(loop: AgentLoop) -> None:
    """Arm the C5b deadline on an explicitly marked resumed run's loop.

    Called by the app-server's resume admission (``prepare_background`` with
    a WP2 resume admission) — the only explicitly marked resumed context. A
    fresh watch replaces any stale one from a previous run of this loop.
    """
    seconds = loop.config.recovery.subagents.resume_progress_deadline_seconds
    loop._completion_coordinator._resume_progress_watch = ResumeProgressWatch(seconds)


def clear_resume_progress_deadline(loop: AgentLoop) -> None:
    """Clear the loop's resume-progress watch (C5b run-end hygiene).

    Every non-resume admission and every run-end path calls this, so an
    ordinary run never inherits — and is never killed by — a previous
    resumed run's watch.
    """
    loop._completion_coordinator._resume_progress_watch = None
