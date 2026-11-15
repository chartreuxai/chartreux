"""F5: the psb event-stream cursor contract — drains, adopts and the gap.

The B-gap incident (agent-53 RCA, repro 3/3): a "drained" notification
(``ClientProjection.consume`` returns None while still advancing the
projection cursor, events.py:487) made ``publish_notification`` return
False, so the coordinator routed the numbered event around the psb queue
(server.py:1574-1580) while the per-session watermark had already moved;
the NEXT queued publication then raised the fatal
``SessionBackendError: The session event stream has a gap`` and tore down
the live session. The v0.7.9 amplifier was the history-less
``runtime/read`` adopt (``include_history=False``), which rebuilt the
subscribed projection's entry index from nothing so every later
``history/entryUpdated`` for a pre-existing entry drained.

The repair pins here:
- a history-less adopt keeps the subscribed projection's entry index alive
  (the incident shape flows with no gap);
- a genuine drain (an entry no projection could know) is carried by the
  stream as a cursor-only advance PLUS a forwarded envelope, publish
  True: the drained notification still reaches the wire client
  (F9/W2 -- the F5 echo alone swallowed it), and the truncated-history
  adopt that rebuilt the incident the same way is covered too;
- a publication the projection did NOT advance (a stale duplicate) keeps
  the False direct-route contract;
- the gap raise stays typed and fatal, and its message discloses the
  expected and received event ids the earlier diagnosis lost;
- the agent liveness poller's diagnostics emit through the shared
  observability ("vibe") logger, the one the rotating file handler rides.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import cast

import pytest

from chartreux.app_server._session_backend_port import (
    SessionBackendError,
    SessionBackendEvent,
)
from chartreux.app_server._turn_queue import TurnQueue
from chartreux.app_server.models import (
    IdleSessionStatus,
    PublicEntryGenerationStatus,
    PublicMessageEntry,
    PublicSession,
    PublicSessionState,
    TextContentBlock,
)
from chartreux.app_server.process_session_backend import ProcessSessionBackend
from chartreux.app_server.protocol import (
    AgentsInventoryReadResponse,
    HistoryEntryAddedParams,
    HistoryEntryUpdatedParams,
    JsonPatchOperation,
    ProtocolErrorCode,
    SessionUpdatedParams,
)
from chartreux.app_server.root_agent_process import RootAgentProcess

SESSION_ID = "sess-bgap-0001"
ENTRY_ID = "entry-bgap-0001"


def _now_ms() -> int:
    return int(time.time() * 1000)


class _FakeServices:
    """The coordinator's SessionBackendServices seam the psb touches."""

    def __init__(self) -> None:
        self.watermarks: dict[str, int] = {}

    def event_watermark(self, session_id: str) -> int:
        return self.watermarks.get(session_id, 0)


class _FakeProcess:
    """RootAgentProcess duck-type: only is_dead's read of ``death``."""

    death = None


def _make_state(*, history: list | None) -> PublicSessionState:
    return PublicSessionState(
        event_id=0,
        session=PublicSession(
            id=SESSION_ID,
            status=IdleSessionStatus(),
            created_at=_now_ms(),
            updated_at=_now_ms(),
        ),
        history=history,
    )


def _make_entry(entry_id: str = ENTRY_ID) -> PublicMessageEntry:
    return PublicMessageEntry(
        id=entry_id,
        session_id=SESSION_ID,
        created_at=_now_ms(),
        updated_at=_now_ms(),
        generation_status=PublicEntryGenerationStatus.IN_PROGRESS,
        role="assistant",
        content=[TextContentBlock(text="streaming")],
    )


def _make_backend() -> tuple[ProcessSessionBackend, _FakeServices]:
    # Bypass __init__ (it needs a live RootAgentProcess): the publication
    # and stream methods under test only need these attributes.
    backend = object.__new__(ProcessSessionBackend)
    services = _FakeServices()
    backend._process = cast(RootAgentProcess, _FakeProcess())
    backend._services = services
    backend._events = asyncio.Queue()
    backend._events_idle = asyncio.Event()
    backend._events_closed = asyncio.Event()
    backend._events_lock = asyncio.Lock()
    backend._events_subscribed = False
    backend._projection = None
    backend._retired = False
    backend._shell_retained = {}
    backend._turn_queue = TurnQueue()
    return backend, services


def _entry_added(event_id: int) -> HistoryEntryAddedParams:
    return HistoryEntryAddedParams(
        event_id=event_id,
        session_id=SESSION_ID,
        entry=_make_entry(),
        emitted_at=_now_ms(),
    )


def _entry_updated(
    event_id: int, *, entry_id: str = ENTRY_ID
) -> HistoryEntryUpdatedParams:
    return HistoryEntryUpdatedParams(
        event_id=event_id,
        session_id=SESSION_ID,
        entry_id=entry_id,
        patch=[JsonPatchOperation(op="replace", path="/updatedAt", value=_now_ms())],
        emitted_at=_now_ms(),
    )


def _session_updated(event_id: int) -> SessionUpdatedParams:
    return SessionUpdatedParams(
        event_id=event_id,
        session_id=SESSION_ID,
        patch=[JsonPatchOperation(op="replace", path="/preview", value=f"p{event_id}")],
        emitted_at=_now_ms(),
    )


class _Collector:
    def __init__(self) -> None:
        self.yielded: list[SessionBackendEvent] = []
        self.error: BaseException | None = None
        self.finished = asyncio.Event()
        self.task: asyncio.Task[None] | None = None

    async def run(self, stream) -> None:
        try:
            async for envelope in stream:
                self.yielded.append(envelope)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:  # recording the failure is the test
            self.error = exc
        finally:
            self.finished.set()


async def _start_stream(backend: ProcessSessionBackend) -> _Collector:
    backend._events_subscribed = True
    collector = _Collector()
    stream = backend._event_stream(session_id=SESSION_ID, after_event_id=0)
    collector.task = asyncio.create_task(collector.run(stream))
    # Let the stream coroutine enter its queue wait.
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    return collector


async def _settle(collector: _Collector, *, ticks: int = 3) -> None:
    for _ in range(ticks):
        await asyncio.sleep(0)
    assert collector.task is not None
    collector.task.cancel()
    await asyncio.gather(collector.task, return_exceptions=True)


class TestHistoryLessAdoption:
    def test_history_less_read_adopt_preserves_the_entry_index(self) -> None:
        backend, _services = _make_backend()
        backend._adopt_state(_make_state(history=[_make_entry()]))
        backend._adopt_state(_make_state(history=None))
        assert backend._projection is not None
        assert [entry.id for entry in backend._projection.history] == [ENTRY_ID]

    @pytest.mark.asyncio
    async def test_history_less_adopt_then_entry_update_flows_without_a_gap(
        self,
    ) -> None:
        # The incident shape, repaired: the history-less runtime/read adopt no
        # longer empties the entry index, so the pre-existing entry's update
        # consumes cleanly and the stream never sees a cursor discontinuity.
        backend, services = _make_backend()

        # Subscribe-time adopt: projection at watermark 0.
        backend._adopt_state(_make_state(history=[]))
        collector = await _start_stream(backend)

        # Event 1: the turn stream adds an entry.
        services.watermarks[SESSION_ID] = 1
        assert await backend.publish_notification("history/entryAdded", _entry_added(1))

        # The v0.7.9 runtime/read adopt: a history-less snapshot of the same
        # session, mid-subscription.
        backend._adopt_state(_make_state(history=None))

        # Event 2: the update for the pre-existing entry must NOT drain.
        services.watermarks[SESSION_ID] = 2
        assert await backend.publish_notification(
            "history/entryUpdated", _entry_updated(2)
        )

        # Event 3: the next publication queues behind it with no gap.
        services.watermarks[SESSION_ID] = 3
        assert await backend.publish_notification(
            "session/updated", _session_updated(3)
        )

        await _settle(collector)

        assert collector.error is None
        assert len(collector.yielded) == 3


class TestDrainedPublicationContract:
    @pytest.mark.asyncio
    async def test_drained_event_advances_the_cursor_and_reaches_the_subscriber(
        self,
    ) -> None:
        # A drain no projection can cure (the entry never existed) still may
        # not desynchronize the stream NOR vanish from the wire: publish
        # carries the consumed advance as a cursor-only echo plus a
        # forwarded envelope the stream yields unnumbered, so the client
        # receives the drained update and the next publication queues
        # cleanly behind it. (F9/W2: the F5 echo alone swallowed the
        # update -- pre-F5 the direct route had delivered it.)
        backend, services = _make_backend()

        backend._adopt_state(_make_state(history=[]))
        collector = await _start_stream(backend)

        services.watermarks[SESSION_ID] = 1
        assert await backend.publish_notification(
            "session/updated", _session_updated(1)
        )

        drained = _entry_updated(2, entry_id="entry-never-added")
        services.watermarks[SESSION_ID] = 2
        assert await backend.publish_notification("history/entryUpdated", drained)

        services.watermarks[SESSION_ID] = 3
        assert await backend.publish_notification(
            "session/updated", _session_updated(3)
        )

        await _settle(collector)

        assert collector.error is None
        # All three publications reached the subscriber: ids 1 and 3 as
        # numbered events, the drained update as the forwarded envelope
        # riding between them.
        assert len(collector.yielded) == 3
        forwarded = collector.yielded[1]
        assert forwarded.method == "history/entryUpdated"
        assert forwarded.params is drained
        assert forwarded.event_id is None

    @pytest.mark.asyncio
    async def test_truncated_history_adopt_keeps_delivering_dropped_entry_updates(
        self,
    ) -> None:
        # F9/W2 residue: SessionReadParams.history defaults to
        # PageRequest(limit=200), and a carrying-history adopt replaces the
        # psb index wholesale -- so an update to an entry the returned page
        # truncated away drains. The client can hold that entry via
        # Load-More backfill, and the forwarded envelope keeps its copy
        # from going stale while the stream stays cursor-consistent.
        backend, services = _make_backend()

        backend._adopt_state(_make_state(history=[_make_entry()]))
        collector = await _start_stream(backend)

        # A carrying-history read whose page truncated the entry away: the
        # snapshot carries its own history, so it still replaces the
        # projection's index wholesale.
        backend._adopt_state(
            _make_state(history=[_make_entry(entry_id="entry-in-the-page")])
        )

        drained = _entry_updated(1)
        services.watermarks[SESSION_ID] = 1
        assert await backend.publish_notification("history/entryUpdated", drained)

        services.watermarks[SESSION_ID] = 2
        assert await backend.publish_notification(
            "session/updated", _session_updated(2)
        )

        await _settle(collector)

        assert collector.error is None
        # The truncate-triggered drain did not swallow the update: the
        # forwarded envelope delivered it, and event 2 queued behind the
        # advance with no gap.
        assert len(collector.yielded) == 2
        forwarded = collector.yielded[0]
        assert forwarded.method == "history/entryUpdated"
        assert forwarded.params is drained
        assert forwarded.event_id is None
        assert collector.yielded[1].event_id == 2

    @pytest.mark.asyncio
    async def test_stale_publication_keeps_the_direct_route(self) -> None:
        # A publication the projection did NOT advance (a duplicate event id
        # at or behind the cursor) returns False: the coordinator's direct
        # route keeps it, and the stream dedupes it by event id.
        backend, services = _make_backend()

        backend._adopt_state(_make_state(history=[]))
        collector = await _start_stream(backend)

        services.watermarks[SESSION_ID] = 1
        assert await backend.publish_notification(
            "session/updated", _session_updated(1)
        )
        services.watermarks[SESSION_ID] = 1
        assert not await backend.publish_notification(
            "session/updated", _session_updated(1)
        )

        await _settle(collector)

        assert collector.error is None
        assert len(collector.yielded) == 1

    @pytest.mark.asyncio
    async def test_gap_raise_stays_typed_and_discloses_the_event_ids(self) -> None:
        # The gap raise remains fatal and typed (STALE_CURSOR); the message
        # now carries the expected/received ids the rotating formatter would
        # otherwise drop with the data dict.
        backend, services = _make_backend()

        backend._adopt_state(_make_state(history=[]))
        collector = await _start_stream(backend)

        services.watermarks[SESSION_ID] = 1
        assert await backend.publish_notification(
            "session/updated", _session_updated(1)
        )

        # A queue item whose event id skips 2: the stream gate must raise.
        backend._events.put_nowait(
            SessionBackendEvent(
                event=None,  # type: ignore[arg-type] - the gate reads only ids
                session_id=SESSION_ID,
                event_id=3,
            )
        )

        assert collector.task is not None
        await asyncio.wait_for(collector.finished.wait(), timeout=5)

        assert collector.error is not None
        assert isinstance(collector.error, SessionBackendError)
        assert collector.error.code is ProtocolErrorCode.STALE_CURSOR
        assert "The session event stream has a gap" in str(collector.error)
        assert "expected event 2" in str(collector.error)
        assert "received event 3" in str(collector.error)
        data = getattr(collector.error, "data", None) or {}
        assert data.get("expectedEventId") == 2, data
        assert data.get("receivedEventId") == 3, data
        # Only event 1 reached the subscriber before the raise.
        assert len(collector.yielded) == 1


class _CollectingHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


class TestLivenessDiagnosticsLogger:
    def test_module_logger_is_the_shared_observability_logger(self) -> None:
        from chartreux.cli.textual_ui import agent_liveness
        from chartreux.observability import logging as observability_logging

        assert agent_liveness.logger is observability_logging.logger
        assert agent_liveness.logger.name == "vibe"

    @pytest.mark.asyncio
    async def test_reconciliation_failure_reaches_the_shared_logger(self) -> None:
        # The crash-window anomaly: the poller's bounded diagnostics and its
        # on-accept failure log must land on the "vibe" logger the rotating
        # file handler attaches to, not an orphan chartreux.* logger.
        from chartreux.cli.textual_ui.agent_liveness import AgentLivenessPoller

        issued = 0

        async def read() -> AgentsInventoryReadResponse:
            nonlocal issued
            issued += 1
            if issued == 1:
                return AgentsInventoryReadResponse(
                    session_id="session-1", generation=1, revision=1, agents=[]
                )
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        def failing_accept(snapshot: object) -> None:
            raise ValueError("reconciliation seam")

        poller = AgentLivenessPoller(
            read=read,
            current_session_id=lambda: "session-1",
            is_active=lambda agent: agent.availability == "working",
            on_accept=failing_accept,
            poll_interval=0.05,
        )
        handler = _CollectingHandler()
        shared = logging.getLogger("vibe")
        shared.addHandler(handler)
        poller.attach()
        try:

            async def _until_logged() -> None:
                while not handler.records:
                    await asyncio.sleep(0)

            await asyncio.wait_for(_until_logged(), timeout=5)
        finally:
            poller.detach()
            shared.removeHandler(handler)

        assert any(
            record.name == "vibe"
            and record.levelno == logging.ERROR
            and "Liveness snapshot reconciliation failed" in record.getMessage()
            for record in handler.records
        ), handler.records
