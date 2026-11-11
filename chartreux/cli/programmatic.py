from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import aclosing, suppress
from dataclasses import dataclass, field
from enum import StrEnum, auto
import json
import sys
import time
from typing import TYPE_CHECKING, Any, TextIO

from pydantic import BaseModel

from chartreux.app_server._subagent_results import (
    AgentResultExpiredError,
    RunStopReason,
)
from chartreux.app_server.events import (
    AppServerEvent,
    CallbackRequested,
    HistoryEntryAdded,
    HistoryEntryUpdated,
    TurnCompleted,
)
from chartreux.app_server.local import (
    ContinueSessionIntent,
    LocalHarness,
    LocalHarnessOptions,
    ResumeSessionIntent,
)
from chartreux.app_server.models import (
    PublicEntryGenerationStatus,
    PublicHistoryEntry,
    PublicMessageEntry,
    PublicTurnStatus,
    PublicTurnStopReason,
)
from chartreux.app_server.session import AppServerSession
from chartreux.observability.logging import logger

if TYPE_CHECKING:
    from chartreux.app_server._sessions import SessionRuntimeRegistry


class OutputFormat(StrEnum):
    TEXT = auto()
    JSON = auto()
    STREAMING = auto()


class ProgrammaticLimitError(RuntimeError):
    pass


class ProgrammaticPersistenceError(RuntimeError):
    """The session could not be durably persisted before the final response."""


class ProgrammaticOutput:
    def __init__(
        self, output_format: OutputFormat, stream: TextIO | None = None
    ) -> None:
        self._format = output_format
        self._stream = stream or sys.stdout
        self._emitted: set[str] = set()
        self._terminal_notice: str | None = None

    def set_terminal_notice(self, notice: str | None) -> None:
        """Attach the continuation's terminal notice to the final outcome."""
        self._terminal_notice = notice

    @property
    def terminal_notice(self) -> str | None:
        return self._terminal_notice

    def start(self, history: list[PublicHistoryEntry]) -> None:
        if self._format is not OutputFormat.STREAMING:
            return
        for entry in history:
            self._emit_completed(entry)

    def consume(self, event: AppServerEvent) -> None:
        if self._format is not OutputFormat.STREAMING:
            return
        match event:
            case HistoryEntryAdded(entry=entry) | HistoryEntryUpdated(entry=entry):
                self._emit_completed(entry)
            case _:
                pass

    def finalize(self, history: list[PublicHistoryEntry]) -> str | None:
        if self._format is OutputFormat.STREAMING:
            if self._terminal_notice is not None:
                self._write_notice(self._terminal_notice)
            return None
        if self._format is OutputFormat.JSON:
            history_json: list[Any] = [
                entry.model_dump(mode="json", by_alias=True) for entry in history
            ]
            if self._terminal_notice is not None:
                history_json.append({
                    "type": "terminal_notice",
                    "text": self._terminal_notice,
                })
            json.dump(history_json, self._stream, indent=2, ensure_ascii=False)
            self._stream.write("\n")
            self._stream.flush()
            return None
        text = _last_assistant_text(history)
        if self._terminal_notice is not None:
            text = (
                f"{text}\n\n{self._terminal_notice}"
                if text is not None
                else self._terminal_notice
            )
        return text

    def _emit_completed(self, entry: PublicHistoryEntry) -> None:
        if (
            entry.id in self._emitted
            or entry.generation_status is not PublicEntryGenerationStatus.COMPLETED
        ):
            return
        self._emitted.add(entry.id)
        self._write_json(entry)

    def _write_json(self, value: BaseModel) -> None:
        json.dump(
            value.model_dump(mode="json", by_alias=True),
            self._stream,
            ensure_ascii=False,
        )
        self._stream.write("\n")
        self._stream.flush()

    def _write_notice(self, notice: str) -> None:
        json.dump(
            {"type": "terminal_notice", "text": notice},
            self._stream,
            ensure_ascii=False,
        )
        self._stream.write("\n")
        self._stream.flush()

    def _print(self, text: str) -> None:
        print(text, file=self._stream)


# P2c: the harness-owned continuation lifecycle (design P2a, programmatic
# mode). While owned dependency runs are active after the single act()
# stream, the programmatic session stays open; the bounds are the child run
# budgets plus this wall-clock cap.
CONTINUATION_CAP_SECONDS = 3600.0
_CONTINUATION_POLL_SECONDS = 0.1
_CANCEL_GRACE_SECONDS = 2.0


class ContinuationStopReason(StrEnum):
    """Why the harness-owned continuation stopped (design P2a bounds)."""

    QUIESCENT = "quiescent"
    WALL_CLOCK_CAP = "wall-clock-cap"
    LIMIT_STOP = "limit-stop"
    SESSION_CLOSED = "session-closed"


@dataclass
class SurfacedResult:
    """One force-surfaced result at the cap / mid-continuation STOP."""

    agent_id: str
    run_id: str
    disposition: str  # "full" (retained body) or "TERMINAL-UNKNOWN"
    body: str | None = None
    retrieval_state: str | None = None


@dataclass
class ContinuationOutcome:
    """The terminal outcome of the harness-owned continuation lifecycle."""

    stop_reason: ContinuationStopReason = ContinuationStopReason.QUIESCENT
    surfaced: list[SurfacedResult] = field(default_factory=list)
    # (agent_id, run_id, stop reason) named separately from the surfaced bodies.
    cancelled: list[tuple[str, str, str]] = field(default_factory=list)
    recovery_notice: str | None = None


async def _persist_session_before_response(
    session: AppServerSession, registry: SessionRuntimeRegistry | None
) -> None:
    """Save-before-response (WP3 [3D]): the final response is emitted only
    after the session transcript is durably persisted.

    Forces one save on the root writer and fails closed when the save did not
    persist (a recorded fail-closed save diagnostic, or logging enabled but
    nothing persisted): the response is never emitted over an unsaved session.
    """
    if registry is None:
        registry = _resolve_session_registry(session)
    root = getattr(registry, "_root", None) if registry is not None else None
    loop = getattr(root, "agent_loop", None)
    if loop is None:
        return
    session_logger = getattr(loop, "session_logger", None)
    if session_logger is None or not session_logger.enabled:
        return
    await loop._save_messages()
    diagnostic = getattr(session_logger, "last_save_diagnostic", None)
    if diagnostic is not None or not session_logger.persisted:
        raise ProgrammaticPersistenceError(
            f"Session {session.session_id} could not be durably persisted "
            "before the final response was emitted"
        )


def _resolve_session_registry(
    session: AppServerSession,
) -> SessionRuntimeRegistry | None:
    """The owned run registry of the in-process harness (None off that path).

    The programmatic harness serves the app server in-process, so the client's
    run peer is the harness server itself; its root backend owns the registry.
    """
    from chartreux.app_server._sessions import SessionRuntimeRegistry

    connection = getattr(session, "_connection", None)
    client = getattr(connection, "current", None)
    run_peer = getattr(client, "_run_peer", None)
    harness_server = getattr(run_peer, "__self__", None)
    server = getattr(harness_server, "_server", None)
    root = getattr(server, "_root", None)
    registry = getattr(root, "children", None)
    if isinstance(registry, SessionRuntimeRegistry):
        return registry
    return None


def _detect_restart_orphans(
    session: AppServerSession, registry: SessionRuntimeRegistry
) -> str | None:
    """Transcript-derived orphan detection (design P2a restart reconciliation).

    The run registry is never persisted, so after a restart every background
    run the transcript launched without a live registry record is an orphan:
    it is force-marked TERMINAL-UNKNOWN in the recovery notice, and results
    completed before the crash are unrecoverable (and said so).
    """
    root = getattr(registry, "_root", None)
    loop = getattr(root, "agent_loop", None)
    messages = getattr(loop, "messages", None)
    if messages is None:
        return None
    session_logger = getattr(loop, "session_logger", None)
    info = None
    if session_logger is not None:
        get_info = getattr(session_logger, "_get_session_info", None)
        if get_info is not None:
            info = get_info()
    links = (
        {link.tool_call_id: link.session_id for link in info[1].child_sessions}
        if info is not None
        else {}
    )
    live_children = getattr(registry, "_children", {})
    orphans: list[str] = []
    for message in messages:
        for tool_call in getattr(message, "tool_calls", None) or []:
            function = getattr(tool_call, "function", None)
            if getattr(function, "name", None) != "task":
                continue
            try:
                arguments = json.loads(getattr(function, "arguments", None) or "{}")
            except (TypeError, ValueError):
                continue
            if not arguments.get("background"):
                continue
            child_session_id = links.get(tool_call.id)
            if child_session_id is not None and child_session_id in live_children:
                continue  # a live owned dependency run, not an orphan
            orphans.append(tool_call.id)
    if not orphans:
        return None
    lines = [
        "Recovery notice: the session restarted with orphaned background runs "
        "(the run registry is never persisted). Orphans are force-marked "
        "TERMINAL-UNKNOWN:"
    ]
    for tool_call_id in orphans:
        lines.append(
            f"- task tool call {tool_call_id}: TERMINAL-UNKNOWN "
            "(no live run record after the restart)"
        )
    lines.append("Results completed before the crash are unrecoverable.")
    return "\n".join(lines)


_PUMP_CLOSED = object()


async def _pump_events(events: Any, queue: asyncio.Queue[Any]) -> None:
    """Forward the continuation event subscription into a cancellable queue.

    ``asyncio.wait_for`` on a raw ``__anext__`` would cancel the async
    generator itself on timeout; the pump keeps the subscription alive across
    poll slices.
    """
    try:
        async for event in events:
            await queue.put(event)
    finally:
        await queue.put(_PUMP_CLOSED)


async def _cancel_active_dependencies(
    registry: SessionRuntimeRegistry,
    requester_session_id: str,
    outcome: ContinuationOutcome,
    clock: Callable[[], float],
) -> None:
    """At the cap / mid-continuation STOP: cancel every active dependency run
    via the registry cancel path, so no child runs on under a closed session.
    """
    try:
        snapshot = await registry.quiescence_snapshot()
    except Exception as exc:
        logger.warning("P2c continuation could not read the run registry", exc_info=exc)
        return
    for agent_id, run_id in snapshot.active_dependency_runs:
        try:
            result = await registry.cancel_run(
                agent_id,
                run_id,
                reason=RunStopReason.ORCHESTRATOR_CANCELLED,
                requester_session_id=requester_session_id,
            )
        except Exception as exc:
            logger.warning(
                "P2c continuation failed to cancel run %s/%s",
                agent_id,
                run_id,
                exc_info=exc,
            )
            continue
        outcome.cancelled.append((
            agent_id,
            run_id,
            str(result.stop_reason or RunStopReason.ORCHESTRATOR_CANCELLED),
        ))
    # Grace window for the cancellation to publish its terminal state before
    # the unfetched results are force-surfaced.
    deadline = clock() + _CANCEL_GRACE_SECONDS
    while clock() < deadline:
        try:
            snapshot = await registry.quiescence_snapshot()
        except Exception:
            return
        if not snapshot.active_dependency_runs:
            return
        await asyncio.sleep(0.01)


async def _force_surface_unfetched(
    registry: SessionRuntimeRegistry, outcome: ContinuationOutcome
) -> None:
    """Force-surface EVERY completed-but-unfetched result in the terminal
    outcome before finalization: the full body if the result is retained, the
    disposition TERMINAL-UNKNOWN with its retrieval state if it is evicted.
    """
    try:
        async with registry._registry_lock:
            keys = sorted(
                key
                for key in registry._result_store
                if key not in registry._collected_results
            )
    except Exception as exc:
        logger.warning("P2c continuation could not read the result store", exc_info=exc)
        return
    for agent_id, run_id in keys:
        try:
            result = await registry.get_agent_result(agent_id, run_id)
        except AgentResultExpiredError:
            outcome.surfaced.append(
                SurfacedResult(
                    agent_id, run_id, "TERMINAL-UNKNOWN", retrieval_state="evicted"
                )
            )
            continue
        except Exception as exc:
            logger.warning(
                "P2c force-surface failed for run %s/%s", agent_id, run_id, exc_info=exc
            )
            outcome.surfaced.append(
                SurfacedResult(
                    agent_id, run_id, "TERMINAL-UNKNOWN", retrieval_state="unavailable"
                )
            )
            continue
        if result is None:
            outcome.surfaced.append(
                SurfacedResult(
                    agent_id, run_id, "TERMINAL-UNKNOWN", retrieval_state="running"
                )
            )
            continue
        outcome.surfaced.append(
            SurfacedResult(agent_id, run_id, "full", body=result.response)
        )


def _render_terminal_notice(outcome: ContinuationOutcome) -> str | None:
    """Render the continuation's terminal notice for the session outcome."""
    lines: list[str] = []
    if outcome.stop_reason is not ContinuationStopReason.QUIESCENT:
        lines.append(
            f"[continuation] stopped at {outcome.stop_reason.value}; no orphaned "
            "children run under the closed session."
        )
    if outcome.surfaced:
        lines.append("Force-surfaced completed-but-unfetched results:")
        for item in outcome.surfaced:
            if item.disposition == "full":
                lines.append(f"- agent {item.agent_id} run {item.run_id}: {item.body}")
            else:
                lines.append(
                    f"- agent {item.agent_id} run {item.run_id}: TERMINAL-UNKNOWN "
                    f"(retrieval state: {item.retrieval_state})"
                )
    if outcome.cancelled:
        lines.append("Cancelled dependency runs:")
        for agent_id, run_id, reason in outcome.cancelled:
            lines.append(f"- agent {agent_id} run {run_id}: {reason}")
    if outcome.recovery_notice:
        lines.append(outcome.recovery_notice)
    return "\n".join(lines) if lines else None


async def run_continuation(
    session: AppServerSession,
    output: ProgrammaticOutput,
    *,
    registry: SessionRuntimeRegistry | None = None,
    cap_seconds: float = CONTINUATION_CAP_SECONDS,
    clock: Callable[[], float] = time.monotonic,
) -> ContinuationOutcome:
    """The harness-owned continuation lifecycle (design P2a, programmatic mode).

    While OWNED DEPENDENCY RUNS are active — task-launched runs with
    non-terminal status in the owned registry; released/cancelled/terminal
    runs are not dependencies; suspended pre-resume runs are terminal for this
    purpose and resume-successors are dependencies; retained idle agents do
    not block — the session stays open and consumes notification-driven turns
    through the named continuation event subscription (``session.events()``).
    The reconciliation sweep executes in this lifecycle across turn
    boundaries; its site is pinned in the registry, not inside a single act()
    stream. Termination happens only at the atomic quiescence check: the final
    check + termination declaration run while HOLDING the terminal-state
    registry acquisition. At the wall-clock cap or a mid-continuation STOP
    (token/price), active dependency runs are cancelled via the registry
    cancel path and every completed-but-unfetched result is force-surfaced in
    the terminal outcome before finalization.
    """
    if registry is None:
        registry = _resolve_session_registry(session)
    outcome = ContinuationOutcome()
    if registry is None:
        # Off the in-process harness path there is no owned registry to
        # continue on; the single act() stream remains the whole session.
        return outcome
    deadline = clock() + cap_seconds
    events = session.events()
    event_queue: asyncio.Queue[Any] = asyncio.Queue()
    pump = asyncio.create_task(_pump_events(events, event_queue))
    try:
        while True:
            final = await registry.check_quiescence_and_declare_termination()
            # The registry quiescence is necessary but not sufficient: a
            # notification-driven parent turn still draining must finish
            # before the outcome is finalized (completion during final drain).
            parent_busy = any(
                turn.status is PublicTurnStatus.IN_PROGRESS
                for turn in session.state.turns or []
            )
            if final.quiescent and not parent_busy:
                outcome.stop_reason = ContinuationStopReason.QUIESCENT
                break
            last_turn = next(reversed(session.state.turns or []), None)
            if (
                last_turn is not None
                and last_turn.stop_reason is PublicTurnStopReason.LIMIT
            ):
                outcome.stop_reason = ContinuationStopReason.LIMIT_STOP
                break
            remaining = deadline - clock()
            if remaining <= 0:
                outcome.stop_reason = ContinuationStopReason.WALL_CLOCK_CAP
                break
            try:
                event = await asyncio.wait_for(
                    event_queue.get(),
                    timeout=min(remaining, _CONTINUATION_POLL_SECONDS),
                )
            except TimeoutError:
                continue
            if event is _PUMP_CLOSED:
                outcome.stop_reason = ContinuationStopReason.SESSION_CLOSED
                break
            output.consume(event)
            if isinstance(event, CallbackRequested):
                # Approval callbacks during the continuation auto-deny: a
                # programmatic session has no user to answer them.
                await session.deny_callback(event.callback)
            elif (
                isinstance(event, TurnCompleted)
                and event.turn.stop_reason is PublicTurnStopReason.LIMIT
            ):
                outcome.stop_reason = ContinuationStopReason.LIMIT_STOP
                break
    finally:
        pump.cancel()
        with suppress(asyncio.CancelledError):
            await pump
        await events.aclose()
    if outcome.stop_reason is not ContinuationStopReason.QUIESCENT:
        await _cancel_active_dependencies(registry, session.session_id, outcome, clock)
        await _force_surface_unfetched(registry, outcome)
    return outcome


def run_programmatic(
    *,
    harness_options: LocalHarnessOptions,
    prompt: str,
    output_format: OutputFormat = OutputFormat.TEXT,
    continuation_cap_seconds: float = CONTINUATION_CAP_SECONDS,
) -> str | None:
    logger.info("USER: %s", prompt)
    output = ProgrammaticOutput(output_format)

    async def run() -> str | None:
        harness = LocalHarness(harness_options)
        session: AppServerSession | None = None
        try:
            session = await harness.start()
            await session.resources.runtime.wait_until_ready()
            await _warn_if_workspace_untrusted(session)
            output.start(session.history)
            async with aclosing(session.act(prompt)) as events:
                async for event in events:
                    output.consume(event)
                    if isinstance(event, CallbackRequested):
                        await session.deny_callback(event.callback)

            # P2c: the harness-owned continuation lifecycle — the session
            # stays open while owned dependency runs are active and consumes
            # notification-driven turns across the act() boundary; termination
            # only at the atomic quiescence check.
            registry = _resolve_session_registry(session)
            # Restart reconciliation runs BEFORE the continuation: the
            # transcript-derived orphan detection must see this session's live
            # children (only a restart leaves transcript-launched runs without
            # a live registry record).
            restart = isinstance(
                harness_options.session, (ResumeSessionIntent, ContinueSessionIntent)
            )
            recovery_notice = (
                _detect_restart_orphans(session, registry)
                if restart and registry is not None
                else None
            )
            outcome = await run_continuation(
                session, output, registry=registry, cap_seconds=continuation_cap_seconds
            )
            if recovery_notice is not None:
                outcome.recovery_notice = recovery_notice
            output.set_terminal_notice(_render_terminal_notice(outcome))

            # WP3 [3D] save-before-response: persist the session durably
            # before the final response (or the limit error) is emitted.
            await _persist_session_before_response(session, registry)

            turn = next(reversed(session.state.turns or []), None)
            # A mid-continuation STOP is recorded on the outcome itself: a
            # notification-driven turn started after the LIMIT turn must not
            # mask the limit from the final check.
            limit_reached = (
                outcome.stop_reason is ContinuationStopReason.LIMIT_STOP
                or (turn is not None and turn.stop_reason is PublicTurnStopReason.LIMIT)
            )
            if limit_reached:
                message = _last_assistant_text(session.history) or (
                    "The configured conversation limit was reached"
                )
                if output.terminal_notice is not None:
                    message = f"{message}\n\n{output.terminal_notice}"
                raise ProgrammaticLimitError(message)
            return output.finalize(session.history)
        finally:
            try:
                if session is not None:
                    await session.close()
            finally:
                await harness.close()

    return asyncio.run(run())


async def _warn_if_workspace_untrusted(session: AppServerSession) -> None:
    trust = await session.resources.workspace.trust_status()
    details = trust.details
    if trust.status != "untrusted" or details is None:
        return
    detected_files = list(
        dict.fromkeys([*details.detected_files, *details.repo_detected_files])
    )
    if not detected_files:
        return
    print(
        f"Warning: {details.cwd} is not trusted; project configuration "
        f"({', '.join(detected_files)}) will be ignored. Re-run with --trust "
        "to trust this folder temporarily.",
        file=sys.stderr,
    )


def _last_assistant_text(history: list[PublicHistoryEntry]) -> str | None:
    return next(
        (
            entry.text
            for entry in reversed(history)
            if isinstance(entry, PublicMessageEntry)
            and entry.role == "assistant"
            and entry.text
        ),
        None,
    )
