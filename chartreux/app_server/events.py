from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum, auto
import json
import math
import time
from typing import Any

from chartreux.app_server._model import validate_wire
from chartreux.app_server._patch import (
    apply_json_patch,
    is_completed_effect_replay,
    is_timing_only_patch,
    make_json_patch,
)
from chartreux.app_server.models import (
    JsonPatchOperation,
    PublicCallbackEntry,
    PublicEntryGenerationStatus,
    PublicHistoryEntry,
    PublicQueuedTurn,
    PublicSession,
    PublicSessionState,
    PublicTurn,
    PublicTurnQueue,
    PublicTurnStatus,
    validate_history_entry,
)
from chartreux.app_server.protocol import (
    AgentEvictionModel,
    AgentSummaryModel,
    AgentsUpdateParams,
    ChildCallbackClosedParams,
    HistoryEntryAddedParams,
    HistoryEntryUpdatedParams,
    JsonValue,
    MCPAuthRequiredParams,
    Notification,
    ServerErrorParams,
    ServerWarningParams,
    SessionCompactedParams,
    SessionSnapshotParams,
    SessionUpdatedParams,
    StatsUpdatedParams,
    TurnCompletedParams,
    TurnQueueUpdatedParams,
    TurnRetryingParams,
    TurnStartedParams,
    TurnUpdatedParams,
)
from chartreux.app_server.relay import FrameClass, FrameSchemaError, classify_frame


@dataclass(frozen=True, slots=True)
class HistoryEntryAdded:
    entry: PublicHistoryEntry


@dataclass(frozen=True, slots=True)
class HistoryEntryUpdated:
    previous: PublicHistoryEntry
    entry: PublicHistoryEntry
    patch: list[JsonPatchOperation]


@dataclass(frozen=True, slots=True)
class SessionSnapshot:
    state: PublicSessionState


@dataclass(frozen=True, slots=True)
class SessionCompacted:
    params: SessionCompactedParams


@dataclass(frozen=True, slots=True)
class SessionUpdated:
    previous: PublicSession
    session: PublicSession
    patch: list[JsonPatchOperation]


@dataclass(frozen=True, slots=True)
class TurnStarted:
    turn: PublicTurn


@dataclass(frozen=True, slots=True)
class TurnUpdated:
    turn: PublicTurn


@dataclass(frozen=True, slots=True)
class TurnCompleted:
    turn: PublicTurn


@dataclass(frozen=True, slots=True)
class AgentsUpdate:
    agents: list[AgentSummaryModel]
    evictions: list[AgentEvictionModel] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class TurnQueueUpdated:
    queue: PublicTurnQueue


@dataclass(frozen=True, slots=True)
class StatsUpdated:
    params: StatsUpdatedParams


@dataclass(frozen=True, slots=True)
class CallbackRequested:
    callback: PublicCallbackEntry


@dataclass(frozen=True, slots=True)
class TurnRetrying:
    params: TurnRetryingParams


@dataclass(frozen=True, slots=True)
class ServerWarning:
    params: ServerWarningParams


@dataclass(frozen=True, slots=True)
class ServerError:
    params: ServerErrorParams


@dataclass(frozen=True, slots=True)
class MCPAuthorizationRequiredEvent:
    params: MCPAuthRequiredParams


@dataclass(frozen=True, slots=True)
class ChildCallbackClosed:
    """A child-owned callback became terminal (never part of root history)."""

    params: ChildCallbackClosedParams


type AppServerEvent = (
    HistoryEntryAdded
    | HistoryEntryUpdated
    | SessionSnapshot
    | SessionCompacted
    | SessionUpdated
    | TurnStarted
    | TurnUpdated
    | TurnCompleted
    | AgentsUpdate
    | TurnQueueUpdated
    | StatsUpdated
    | CallbackRequested
    | TurnRetrying
    | ServerWarning
    | ServerError
    | MCPAuthorizationRequiredEvent
    | ChildCallbackClosed
)

_STREAMING_TEXT_PATHS = {"/content/0/text", "/text", "/state/outputText"}
# Keep retired IDs for as long as the server keeps retired enqueue retries.
_RETIRED_QUEUE_ITEM_HISTORY_SIZE = 256


def _merge_snapshot_suffix[ItemT](
    previous: list[ItemT] | None,
    current: list[ItemT] | None,
    *,
    item_id: Callable[[ItemT], str],
) -> tuple[list[ItemT] | None, bool]:
    if current is None:
        return previous, previous is not None
    if not previous or not current:
        return current, False

    first_current_id = item_id(current[0])
    overlap_start = next(
        (
            index
            for index, item in enumerate(previous)
            if item_id(item) == first_current_id
        ),
        None,
    )
    if overlap_start is None:
        return current, False

    overlap = previous[overlap_start:]
    if len(overlap) > len(current):
        return current, False
    if [item_id(item) for item in overlap] != [
        item_id(item) for item in current[: len(overlap)]
    ]:
        return current, False
    return [*previous[:overlap_start], *current], overlap_start > 0


def reconcile_snapshot(
    previous: PublicSessionState, current: PublicSessionState
) -> list[AppServerEvent]:
    events: list[AppServerEvent] = [SessionSnapshot(current)]
    if previous.session != current.session:
        events.append(
            SessionUpdated(
                previous=previous.session,
                session=current.session,
                patch=make_json_patch(
                    previous.session.model_dump(mode="json", by_alias=True),
                    current.session.model_dump(mode="json", by_alias=True),
                ),
            )
        )

    previous_entries = {entry.id: entry for entry in previous.history or []}
    for entry in current.history or []:
        prior = previous_entries.get(entry.id)
        if prior is None:
            events.append(HistoryEntryAdded(entry))
            continue
        if prior == entry:
            continue
        events.append(
            HistoryEntryUpdated(
                previous=prior,
                entry=entry,
                patch=make_json_patch(
                    prior.model_dump(mode="json", by_alias=True),
                    entry.model_dump(mode="json", by_alias=True),
                    append_paths=_STREAMING_TEXT_PATHS,
                ),
            )
        )

    previous_turns = {turn.id: turn for turn in previous.turns or []}
    for current_turn in current.turns or []:
        previous_turn = previous_turns.get(current_turn.id)
        if current_turn == previous_turn:
            continue
        if current_turn.status is PublicTurnStatus.IN_PROGRESS:
            events.append(TurnStarted(current_turn))
        else:
            events.append(TurnCompleted(current_turn))
    if previous.turn_queue != current.turn_queue:
        events.append(TurnQueueUpdated(current.turn_queue))
    return events


type _KnownEventParams = (
    HistoryEntryAddedParams
    | HistoryEntryUpdatedParams
    | SessionSnapshotParams
    | SessionCompactedParams
    | SessionUpdatedParams
    | StatsUpdatedParams
    | TurnCompletedParams
    | AgentsUpdateParams
    | TurnQueueUpdatedParams
    | TurnStartedParams
    | TurnUpdatedParams
)


class EventSequenceError(RuntimeError):
    pass


class UnknownNotificationError(RuntimeError):
    pass


_PROJECTION_EVENT_PARAMS: dict[str, type[_KnownEventParams]] = {
    "session/snapshot": SessionSnapshotParams,
    "session/compacted": SessionCompactedParams,
    "session/updated": SessionUpdatedParams,
    "history/entryAdded": HistoryEntryAddedParams,
    "history/entryUpdated": HistoryEntryUpdatedParams,
    "turn/started": TurnStartedParams,
    "turn/updated": TurnUpdatedParams,
    "turn/completed": TurnCompletedParams,
    "agents/update": AgentsUpdateParams,
    "turn_queue_updated": TurnQueueUpdatedParams,
    "session/statsUpdated": StatsUpdatedParams,
}


def projection_notification_methods() -> frozenset[str]:
    return frozenset(_PROJECTION_EVENT_PARAMS)


def _parse_server_warning(notification: Notification) -> ServerWarning:
    return ServerWarning(validate_wire(ServerWarningParams, notification.params))


def _parse_server_error(notification: Notification) -> ServerError:
    return ServerError(validate_wire(ServerErrorParams, notification.params))


def _parse_turn_retrying(notification: Notification) -> TurnRetrying:
    return TurnRetrying(validate_wire(TurnRetryingParams, notification.params))


def _parse_mcp_auth_required(
    notification: Notification,
) -> MCPAuthorizationRequiredEvent:
    return MCPAuthorizationRequiredEvent(
        validate_wire(MCPAuthRequiredParams, notification.params)
    )


def _parse_child_callback_closed(notification: Notification) -> ChildCallbackClosed:
    return ChildCallbackClosed(
        validate_wire(ChildCallbackClosedParams, notification.params)
    )


_SERVER_EVENT_DISPATCH: dict[
    str,
    Callable[
        [Notification],
        ServerWarning
        | ServerError
        | TurnRetrying
        | MCPAuthorizationRequiredEvent
        | ChildCallbackClosed,
    ],
] = {
    "warning": _parse_server_warning,
    "error": _parse_server_error,
    "turn/retrying": _parse_turn_retrying,
    "mcp_catalog/authRequired": _parse_mcp_auth_required,
    "callback/childClosed": _parse_child_callback_closed,
}


def server_event_notification_methods() -> frozenset[str]:
    return frozenset(_SERVER_EVENT_DISPATCH)


def parse_server_event(
    notification: Notification,
) -> (
    ServerWarning
    | ServerError
    | TurnRetrying
    | MCPAuthorizationRequiredEvent
    | ChildCallbackClosed
    | None
):
    parser = _SERVER_EVENT_DISPATCH.get(notification.method)
    if parser is None:
        return None
    return parser(notification)


class ClientProjection:
    def __init__(self, state: PublicSessionState) -> None:
        self.state = state
        self._entries = {entry.id: entry for entry in state.history or []}
        self._entry_positions: dict[str, int] = {}
        self._reindex_entries()
        self._last_event_id = state.event_id
        self._generation = 0
        self._retired_queue_item_ids: OrderedDict[str, None] = OrderedDict()
        self._remember_started_queue_items(state)

    @property
    def history(self) -> list[PublicHistoryEntry]:
        return self.state.history or []

    @property
    def last_event_id(self) -> int:
        return self._last_event_id

    @property
    def generation(self) -> int:
        """How many times the whole state has been swapped out for another.

        Event IDs count what the server has published, so a state a request
        answered with carries the same watermark as the state it replaced.
        This counts the replacements themselves, which is what tells a caller
        holding a state that it is reading the conversation it started on.
        """
        return self._generation

    @property
    def history_before_cursor(self) -> str | None:
        return self.state.history_before_cursor

    def replace_state(self, state: PublicSessionState) -> None:
        self._replace_state(state)

    def prepend_history_page(self, entries: list[PublicHistoryEntry]) -> None:
        page_entries: list[PublicHistoryEntry] = []
        page_ids: set[str] = set()
        for entry in entries:
            if entry.id in page_ids:
                raise ValueError(f"Duplicate paged history entry: {entry.id}")
            page_ids.add(entry.id)
            if existing := self._entries.get(entry.id):
                if existing != entry:
                    raise ValueError(f"Conflicting paged history entry: {entry.id}")
                continue
            self._entries[entry.id] = entry
            page_entries.append(entry)
        if self.state.history is None:
            self.state.history = []
        self.state.history[:0] = page_entries
        self._reindex_entries()

    def ensure_callback(self, callback: PublicCallbackEntry) -> bool:
        if callback.id in self._entries:
            return False
        self._add_entry(callback)
        return True

    def adopt_turn_queue(
        self, queue: PublicTurnQueue, *, after_event_id: int
    ) -> PublicTurnQueue:
        """Adopt a queue read unless a newer event was already reduced."""
        if self._last_event_id > after_event_id:
            return self.state.turn_queue
        self._replace_turn_queue(queue)
        return queue

    def track_queued_turn(
        self, queued_turn: PublicQueuedTurn, *, session_id: str
    ) -> None:
        """Reconcile an enqueue result unless that queue item already retired."""
        if session_id != self.state.session.id:
            return
        if any(item.id == queued_turn.id for item in self.state.turn_queue.items):
            return
        if queued_turn.id in self._retired_queue_item_ids:
            return
        if any(turn.queue_item_id == queued_turn.id for turn in self.state.turns or []):
            self._remember_retired_queue_item(queued_turn.id)
            return
        self.state.turn_queue.items.append(queued_turn)

    def replace_queued_turn(self, queued_turn: PublicQueuedTurn) -> None:
        """Optimistically update an existing queued item without resurrecting it."""
        for index, item in enumerate(self.state.turn_queue.items):
            if item.id == queued_turn.id:
                self.state.turn_queue.items[index] = queued_turn
                return

    def begin_turn(self, turn: PublicTurn) -> None:
        if turn.session_id != self.state.session.id:
            raise EventSequenceError(
                f"Turn belongs to session {turn.session_id!r}, "
                f"expected {self.state.session.id!r}"
            )
        self._replace_turn(turn)

    def consume(
        self, notification: Notification, *, dedupe_history_additions: bool = True
    ) -> AppServerEvent | None:
        params = _parse_event_params(notification)
        if isinstance(params, SessionCompactedParams):
            self._validate_handoff(params)
            self._replace_state(params.state)
            return SessionCompacted(params)
        if isinstance(params, SessionSnapshotParams):
            self._validate_snapshot(params)
        event_id = self._next_event_id(params.event_id, params.session_id)
        if event_id is None:
            return None
        event = self._apply_event(params)
        if (
            event is None
            and not dedupe_history_additions
            and isinstance(params, HistoryEntryAddedParams)
        ):
            # Backend streams must forward every newly numbered addition even
            # when their own projection already knows it. The consumer dedupes
            # the addition while advancing its sequence watermark.
            event = HistoryEntryAdded(self._entries[params.entry.id])
        self._last_event_id = event_id
        return event

    def _apply_event(self, params: _KnownEventParams) -> AppServerEvent | None:
        event: AppServerEvent | None
        match params:
            case SessionSnapshotParams():
                self._apply_snapshot(params.state)
                event = SessionSnapshot(self.state)
            case SessionUpdatedParams():
                event = self._update_session(params)
            case HistoryEntryAddedParams():
                event = (
                    HistoryEntryAdded(params.entry)
                    if self._add_entry(params.entry)
                    else None
                )
            case HistoryEntryUpdatedParams():
                event = self._update_entry(params)
            case TurnStartedParams():
                self._replace_turn(params.turn)
                event = TurnStarted(params.turn)
            case TurnUpdatedParams():
                self._replace_turn(params.turn)
                event = TurnUpdated(params.turn)
            case TurnCompletedParams():
                self._replace_turn(params.turn)
                event = TurnCompleted(params.turn)
            case AgentsUpdateParams():
                event = AgentsUpdate(params.agents, params.evictions)
            case TurnQueueUpdatedParams():
                self._replace_turn_queue(params.queue)
                event = TurnQueueUpdated(params.queue)
            case StatsUpdatedParams():
                self.state.session.token_usage = params.stats.token_usage
                event = StatsUpdated(params)
            case SessionCompactedParams():
                raise AssertionError("Session handoffs are reduced before events")
        return event

    def _next_event_id(self, event_id: object, session_id: object) -> int | None:
        if not isinstance(event_id, int) or isinstance(event_id, bool) or event_id <= 0:
            raise EventSequenceError("App-server event IDs must be positive integers")
        if session_id != self.state.session.id:
            raise EventSequenceError(
                f"App-server event belongs to session {session_id!r}, "
                f"expected {self.state.session.id!r}"
            )
        if event_id <= self._last_event_id:
            return None
        expected = self._last_event_id + 1
        if event_id != expected:
            raise EventSequenceError(
                f"App-server event gap: expected {expected}, received {event_id}"
            )
        return event_id

    def _validate_snapshot(self, params: SessionSnapshotParams) -> None:
        if params.state.session.id != params.session_id:
            raise EventSequenceError(
                "App-server snapshot session does not match its notification"
            )
        if params.state.event_id != params.event_id:
            raise EventSequenceError(
                "App-server snapshot watermark does not match its notification"
            )

    def _validate_handoff(self, params: SessionCompactedParams) -> None:
        if params.old_session_id != self.state.session.id:
            raise EventSequenceError(
                "App-server handoff does not replace the active session"
            )
        if params.session_id == params.old_session_id:
            raise EventSequenceError("App-server handoff did not change session")
        if params.state.session.id != params.session_id:
            raise EventSequenceError(
                "App-server handoff state does not match its notification"
            )
        if params.state.event_id != params.event_id:
            raise EventSequenceError(
                "App-server handoff watermark does not match its notification"
            )
        if params.event_id <= 0:
            raise EventSequenceError("App-server handoff watermark must be positive")

    def _replace_state(self, state: PublicSessionState) -> None:
        if state.session.id == self.state.session.id:
            self._remember_removed_queue_items(self.state.turn_queue, state.turn_queue)
        else:
            self._retired_queue_item_ids.clear()
        self.state = state
        self._entries = {entry.id: entry for entry in state.history or []}
        self._reindex_entries()
        self._last_event_id = state.event_id
        self._generation += 1
        self._remember_started_queue_items(state)

    def _apply_snapshot(self, state: PublicSessionState) -> None:
        previous = self.state
        if state.session.id != previous.session.id:
            self._replace_state(state)
            return

        # History and turns are append-only within one session. Live snapshots
        # carry only their latest page, so retain a contiguous page already loaded
        # before that suffix while taking updated entries from the snapshot.
        history, preserved_history_prefix = _merge_snapshot_suffix(
            previous.history, state.history, item_id=lambda entry: entry.id
        )
        turns, _ = _merge_snapshot_suffix(
            previous.turns, state.turns, item_id=lambda turn: turn.id
        )
        self._replace_state(
            state.model_copy(
                update={
                    "history": history,
                    "history_before_cursor": (
                        previous.history_before_cursor
                        if preserved_history_prefix
                        else state.history_before_cursor
                    ),
                    "turns": turns,
                }
            )
        )

    def _add_entry(self, entry: PublicHistoryEntry) -> bool:
        if entry.id in self._entries:
            # A pending notification can be retried with a new event ID after
            # delivery or snapshot recovery. Keep the already-known entry,
            # which may have advanced beyond the original addition.
            return False
        self._entries[entry.id] = entry
        if self.state.history is None:
            self.state.history = []
        self.state.history.append(entry)
        self._entry_positions[entry.id] = len(self.state.history) - 1
        if (
            isinstance(entry, PublicCallbackEntry)
            and entry.state.status == "open"
            and all(
                callback.callback_id != entry.callback_id
                for callback in self.state.active_callbacks
            )
        ):
            self.state.active_callbacks.append(entry)
        return True

    def _update_entry(
        self, params: HistoryEntryUpdatedParams
    ) -> HistoryEntryUpdated | None:
        previous = self._entries.get(params.entry_id)
        if previous is None:
            return None
        raw = previous.model_dump(mode="json", by_alias=True)
        if is_completed_effect_replay(previous, params.patch, raw=raw):
            # Ambiguous delivery and snapshot recovery can already contain this
            # completion. Drain the retry without changing the frozen entry.
            return None
        if (
            previous.generation_status is PublicEntryGenerationStatus.COMPLETED
            and not is_timing_only_patch(previous, params.patch, raw=raw)
        ):
            raise ValueError(
                f"Completed public history entry is frozen: {params.entry_id}"
            )
        entry = validate_history_entry(apply_json_patch(raw, params.patch))
        self._replace_entry(entry)
        if isinstance(entry, PublicCallbackEntry):
            self.state.active_callbacks = [
                callback
                for callback in self.state.active_callbacks
                if callback.callback_id != entry.callback_id
            ]
            if entry.state.status == "open":
                self.state.active_callbacks.append(entry)
        return HistoryEntryUpdated(previous=previous, entry=entry, patch=params.patch)

    def _reindex_entries(self) -> None:
        self._entry_positions = {
            entry.id: position
            for position, entry in enumerate(self.state.history or [])
        }

    def _entry_position(self, entry_id: str) -> int | None:
        """Locate an entry through the maintained id-to-position index."""
        history = self.state.history
        if history is None:
            return None
        index = self._entry_positions.get(entry_id)
        if index is not None and index < len(history):
            if history[index].id == entry_id:
                return index
        # State is public; a foreign rewrite of its history invalidates positions.
        self._reindex_entries()
        return self._entry_positions.get(entry_id)

    def _replace_entry(self, entry: PublicHistoryEntry) -> None:
        self._entries[entry.id] = entry
        history = self.state.history
        if history is None:
            return
        index = self._entry_position(entry.id)
        if index is None:
            return
        history[index] = entry

    def _replace_turn(self, turn: PublicTurn) -> None:
        if turn.queue_item_id is not None:
            self._remember_retired_queue_item(turn.queue_item_id)
        turns = self.state.turns
        if turns is None:
            self.state.turns = [turn]
            return
        for index, existing in enumerate(turns):
            if existing.id == turn.id:
                turns[index] = turn
                return
        turns.append(turn)

    def _replace_turn_queue(self, queue: PublicTurnQueue) -> None:
        self._remember_removed_queue_items(self.state.turn_queue, queue)
        self.state.turn_queue = queue

    def _remember_removed_queue_items(
        self, previous: PublicTurnQueue, current: PublicTurnQueue
    ) -> None:
        current_ids = {item.id for item in current.items}
        for item in previous.items:
            if item.id not in current_ids:
                self._remember_retired_queue_item(item.id)

    def _remember_started_queue_items(self, state: PublicSessionState) -> None:
        for turn in state.turns or []:
            if turn.queue_item_id is not None:
                self._remember_retired_queue_item(turn.queue_item_id)

    def _remember_retired_queue_item(self, queue_item_id: str) -> None:
        self._retired_queue_item_ids[queue_item_id] = None
        self._retired_queue_item_ids.move_to_end(queue_item_id)
        while len(self._retired_queue_item_ids) > _RETIRED_QUEUE_ITEM_HISTORY_SIZE:
            self._retired_queue_item_ids.popitem(last=False)

    def _update_session(self, params: SessionUpdatedParams) -> SessionUpdated:
        previous = self.state.session
        raw = previous.model_dump(mode="json", by_alias=True)
        patched = apply_json_patch(raw, params.patch)
        session = PublicSession.model_validate(patched)
        self.state.session = session
        return SessionUpdated(previous=previous, session=session, patch=params.patch)


def _normalize_legacy_agents_update_params(
    params: dict[str, JsonValue],
) -> dict[str, JsonValue]:
    """Accept pre-0.5.3 ``agents/update`` payloads that predate the name field.

    The server always populates ``name``; older servers omit it. Client-side
    parsing normalizes a missing ``name`` to the empty string BEFORE wire
    validation so legacy payloads stay acceptable (design §5).
    """
    agents = params.get("agents")
    if not isinstance(agents, list):
        return params
    normalized: list[JsonValue] = []
    changed = False
    for agent in agents:
        if isinstance(agent, dict) and "name" not in agent:
            agent = {**agent, "name": ""}
            changed = True
        normalized.append(agent)
    if not changed:
        return params
    return {**params, "agents": normalized}


def _parse_event_params(notification: Notification) -> _KnownEventParams:
    params_type = _PROJECTION_EVENT_PARAMS.get(notification.method)
    if params_type is None:
        raise UnknownNotificationError(
            f"Unknown app-server notification: {notification.method}"
        )
    params = notification.params
    if params_type is AgentsUpdateParams:
        params = _normalize_legacy_agents_update_params(params)
    return validate_wire(params_type, params)


# --- Presentation consumer (the TUI side of the relay) ----------------------
#
# Every frame on an incarnation-enabled relay stream carries the hop-stamped
# ``relay_incarnation`` generation; ``relay/retire`` retires one generation and
# the next generation's frames announce its replacement. The consumer tracks
# the announced incarnation per stream, installs each incarnation's paged
# snapshot atomically at its boundary-bearing final page, and keeps its
# registration timers generation-scoped: a retired incarnation's timers are
# disarmed at retirement and can never fire into the replacement.

# Numeric bounds from the v0.7.7 plan's table. They are validation candidates,
# not constants: a consumer may be built with tighter limits per deployment.
SNAPSHOT_PAGE_MAX_BYTES = 64 * 1024
SNAPSHOT_MAX_PAGES = 256
SNAPSHOT_MAX_BYTES = 16 * 1024 * 1024
REGISTRATION_AWAIT_TIMEOUT = 30.0
REGISTRATION_TO_INSTALL_TIMEOUT = 8.0
REGISTRATION_MAX_ATTEMPTS = 3
REGISTRATION_BACKOFF_SECONDS = (0.5, 1.0)

# Relay wire methods, mirroring the closed schema owned by
# ``chartreux.app_server.relay`` (the hop/producer side of the same contract).
RELAY_INCARNATION_FIELD = "relay_incarnation"
RETIRE_METHOD = "relay/retire"
TERMINAL_METHOD = "relay/terminal"
SNAPSHOT_PAGE_METHOD = "relay/snapshotPage"

DEFAULT_PRESENTATION_STREAM_ID = "session"

# A presentation-state listener observes one stream's state after every
# frame-driven transition and every timer firing; the TUI uses it to surface
# stale/refreshing, atomic installs, and persistent typed failures.
type PresentationStateListener = Callable[[str, "PresentationStreamState"], None]


class PresentationProtocolError(RuntimeError):
    """A relay frame is outside the consumer's closed presentation schema.

    Raised for a frame of an incarnation the registration protocol never
    announced and for malformed relay control frames; the affected stream is
    terminally failed typed.
    """


class SnapshotAttemptInvalidated(RuntimeError):
    """One snapshot attempt failed; its generation is retired before retry."""


def _is_control_frame(message: Mapping[str, Any]) -> bool:
    """Whether one relay frame is control-class rather than a presentation delta.

    The relay's closed schema owns the classification; a frame outside that
    schema is not a control frame and stays an ordinary presentation frame.
    """
    try:
        return classify_frame(message) is FrameClass.CONTROL
    except FrameSchemaError:
        return False


@dataclass(frozen=True, slots=True)
class SnapshotPage:
    """One bounded page of one incarnation's initial snapshot."""

    page_index: int
    page_count: int
    payload: str
    sequence_boundary: int | None
    encoded_size: int


def parse_snapshot_page(params: object) -> SnapshotPage:
    """Validate one snapshot-page frame's params, or fail typed."""
    if not isinstance(params, Mapping):
        raise PresentationProtocolError("snapshot page params must be an object")
    page_index = params.get("pageIndex")
    page_count = params.get("pageCount")
    payload = params.get("payload")
    boundary = params.get("sequenceBoundary")
    if type(page_index) is not int or page_index < 0:
        raise PresentationProtocolError(
            "snapshot pageIndex must be a non-negative integer"
        )
    if type(page_count) is not int or page_count < 1:
        raise PresentationProtocolError("snapshot pageCount must be a positive integer")
    if not isinstance(payload, str):
        raise PresentationProtocolError("snapshot payload must be a string")
    if boundary is not None and (type(boundary) is not int or boundary < 0):
        raise PresentationProtocolError(
            "sequenceBoundary must be a non-negative integer or null"
        )
    encoded_size = len(json.dumps(params, separators=(",", ":")).encode())
    return SnapshotPage(page_index, page_count, payload, boundary, encoded_size)


@dataclass(frozen=True, slots=True)
class SnapshotInstall:
    """One atomically installed snapshot: every page of one generation."""

    generation: int
    sequence_boundary: int
    pages: tuple[str, ...]


class SnapshotAssembler:
    """Assemble one generation's paged snapshot; install only at the final page.

    All-or-nothing per snapshot generation: pages accumulate until the
    boundary-bearing final page arrives, and only then does the complete
    snapshot install as one unit — no delta may apply before that. An
    oversized page, a page-count or total-byte overflow, a duplicate or
    out-of-order page, or a final page without its sequence boundary
    invalidates the whole attempt; nothing partial is ever installed. An
    empty snapshot is a single final page carrying the boundary.
    """

    def __init__(
        self,
        generation: int,
        *,
        page_max_bytes: int = SNAPSHOT_PAGE_MAX_BYTES,
        max_pages: int = SNAPSHOT_MAX_PAGES,
        max_bytes: int = SNAPSHOT_MAX_BYTES,
    ) -> None:
        if type(generation) is not int or generation < 0:
            raise ValueError("generation must be a non-negative integer")
        if page_max_bytes < 1 or max_pages < 1 or max_bytes < 1:
            raise ValueError("snapshot bounds must be positive")
        self.generation = generation
        self._page_max_bytes = page_max_bytes
        self._max_pages = max_pages
        self._max_bytes = max_bytes
        self._pages: list[str] = []
        self._bytes = 0
        self._invalidated: str | None = None

    @property
    def invalidated(self) -> str | None:
        """Why this attempt was invalidated, or ``None`` while it may proceed."""
        return self._invalidated

    @property
    def accepted_pages(self) -> int:
        """Pages accepted so far, in order."""
        return len(self._pages)

    def _invalidate(self, reason: str) -> None:
        if self._invalidated is None:
            self._invalidated = reason

    def accept(self, page: SnapshotPage) -> SnapshotInstall | None:
        """Accept one page; return the installed snapshot at the final page.

        Raises :class:`SnapshotAttemptInvalidated` when the page violates a
        bound or the ordering, invalidating the whole attempt.
        """
        if self._invalidated is not None:
            raise SnapshotAttemptInvalidated(self._invalidated)
        if page.page_count > self._max_pages:
            self._invalidate("snapshot page count exceeds the per-snapshot page bound")
            raise SnapshotAttemptInvalidated(self._invalidated)
        if page.page_index >= page.page_count:
            self._invalidate("snapshot page index outside its declared page count")
            raise SnapshotAttemptInvalidated(self._invalidated)
        if page.encoded_size > self._page_max_bytes:
            self._invalidate("snapshot page exceeds the encoded page bound")
            raise SnapshotAttemptInvalidated(self._invalidated)
        if self._bytes + page.encoded_size > self._max_bytes:
            self._invalidate("snapshot exceeds the total encoded byte bound")
            raise SnapshotAttemptInvalidated(self._invalidated)
        if page.page_index != len(self._pages):
            self._invalidate("snapshot page disorder or duplicate")
            raise SnapshotAttemptInvalidated(self._invalidated)
        self._pages.append(page.payload)
        self._bytes += page.encoded_size
        if page.page_index == page.page_count - 1:
            if page.sequence_boundary is None:
                self._invalidate("final snapshot page carries no sequence boundary")
                raise SnapshotAttemptInvalidated(self._invalidated)
            return SnapshotInstall(
                generation=self.generation,
                sequence_boundary=page.sequence_boundary,
                pages=tuple(self._pages),
            )
        return None


class TimerKind(StrEnum):
    """The generation-scoped deadlines the consumer arms."""

    REGISTRATION_AWAIT = auto()
    REGISTRATION_TO_INSTALL = auto()
    BACKOFF = auto()


@dataclass(frozen=True, slots=True)
class Timer:
    """One armed deadline, scoped to the generation it was armed for."""

    kind: TimerKind
    generation: int
    deadline: float
    attempt: int


class StreamPhase(StrEnum):
    """Where one stream's current incarnation stands."""

    INSTALLING = auto()
    INSTALLED = auto()
    AWAITING = auto()
    TERMINAL = auto()


@dataclass(frozen=True, slots=True)
class StreamFailure:
    """The persistent typed failure that ended a stream."""

    stream_id: str
    reason: str


def _validated_backoff_schedule(backoff_schedule: Sequence[float]) -> tuple[float, ...]:
    if not backoff_schedule or not all(
        math.isfinite(value) and value >= 0 for value in backoff_schedule
    ):
        raise ValueError("backoff schedule must be non-empty with non-negative entries")
    return tuple(backoff_schedule)


def _validated_registration_bounds(
    registration_await_timeout: float,
    registration_to_install_timeout: float,
    max_attempts: int,
    backoff_schedule: Sequence[float],
) -> tuple[float, ...]:
    if not all(
        math.isfinite(value) and value > 0
        for value in (registration_await_timeout, registration_to_install_timeout)
    ):
        raise ValueError("registration timeouts must be finite and positive")
    if type(max_attempts) is not int or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer")
    return _validated_backoff_schedule(backoff_schedule)


class PresentationStreamState:
    """Consumer-side state of one relay presentation stream.

    Tracks the announced incarnation (generation), installs paged snapshots
    atomically per generation, and arms generation-scoped timers: retiring a
    generation disarms its timers, so they can never fire into the
    replacement. Retirement, deadlines, and completion race by first-winner:
    exactly one terminal disposition is delivered.
    """

    def __init__(
        self,
        stream_id: str,
        *,
        registration_await_timeout: float = REGISTRATION_AWAIT_TIMEOUT,
        registration_to_install_timeout: float = REGISTRATION_TO_INSTALL_TIMEOUT,
        max_attempts: int = REGISTRATION_MAX_ATTEMPTS,
        backoff_schedule: Sequence[float] = REGISTRATION_BACKOFF_SECONDS,
        page_max_bytes: int = SNAPSHOT_PAGE_MAX_BYTES,
        snapshot_max_pages: int = SNAPSHOT_MAX_PAGES,
        snapshot_max_bytes: int = SNAPSHOT_MAX_BYTES,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not stream_id:
            raise ValueError("stream_id must be non-empty")
        self.stream_id = stream_id
        self.registration_await_timeout = registration_await_timeout
        self.registration_to_install_timeout = registration_to_install_timeout
        self.max_attempts = max_attempts
        self.backoff_schedule = _validated_registration_bounds(
            registration_await_timeout,
            registration_to_install_timeout,
            max_attempts,
            backoff_schedule,
        )
        self._page_max_bytes = page_max_bytes
        self._snapshot_max_pages = snapshot_max_pages
        self._snapshot_max_bytes = snapshot_max_bytes
        self._clock = clock
        # The stream's initial incarnation is announced by the stream itself;
        # its snapshot is the first presentation frames of generation 0.
        self._current: int | None = 0
        self._retired: set[int] = set()
        self._awaited: int | None = None
        self._phase = StreamPhase.INSTALLING
        self._timers: dict[TimerKind, Timer] = {}
        self._assembler: SnapshotAssembler | None = self._new_assembler(0)
        self._attempts = 0
        self._terminal: StreamFailure | None = None
        self._last_install: SnapshotInstall | None = None
        self._arm(
            TimerKind.REGISTRATION_TO_INSTALL,
            0,
            self.registration_to_install_timeout,
            0,
        )

    @property
    def current(self) -> int | None:
        """The announced incarnation, or ``None`` while awaiting one."""
        return self._current

    @property
    def phase(self) -> StreamPhase:
        return self._phase

    @property
    def terminal(self) -> StreamFailure | None:
        """The persistent typed failure that ended the stream, if any."""
        return self._terminal

    @property
    def attempts(self) -> int:
        """Failed registration attempts since the last successful install."""
        return self._attempts

    @property
    def last_install(self) -> SnapshotInstall | None:
        """The most recently installed snapshot."""
        return self._last_install

    @property
    def timers(self) -> tuple[Timer, ...]:
        """The currently armed timers."""
        return tuple(self._timers.values())

    def receive_frame(  # noqa: PLR0911 - one return per protocol disposition
        self, generation: int | None, message: Mapping[str, Any]
    ) -> bool:
        """Accept one inbound relay frame; report whether the consumer ate it.

        Returns ``False`` for a current-generation presentation frame that
        must pass through to the ordinary projection, ``True`` for relay
        control frames and stale frames the consumer discarded. Raises
        :class:`PresentationProtocolError` for an incarnation the
        registration protocol never announced.
        """
        if type(generation) is not int or generation < 0:
            raise PresentationProtocolError(
                "frame must carry a non-negative incarnation generation"
            )
        if self._terminal is not None:
            return True
        if generation in self._retired:
            return True
        method = message.get("method")
        if method == TERMINAL_METHOD:
            # Teardown for the announced incarnation ends the stream; teardown
            # for a retired or unannounced incarnation is stale, not fatal.
            if generation == self._effective_generation():
                params = message.get("params")
                reason = params.get("reason") if isinstance(params, Mapping) else None
                detail = (
                    reason if isinstance(reason, str) and reason else "terminal notice"
                )
                self._terminate(f"relay terminal notice: {detail}")
            return True
        if method == RETIRE_METHOD:
            if generation == self._current:
                self.retire(generation)
            else:
                raise PresentationProtocolError(
                    f"retire notice for unannounced incarnation {generation}"
                )
            return True
        if generation != self._current:
            replacement_of = (
                self._current if self._current is not None else self._awaited
            )
            if replacement_of is not None and generation == replacement_of + 1:
                self._accept_announcement(generation)
            else:
                raise PresentationProtocolError(
                    f"frame for unannounced incarnation {generation}"
                )
        if method == SNAPSHOT_PAGE_METHOD:
            return self._accept_page(generation, message)
        if self._phase is not StreamPhase.INSTALLED:
            if _is_control_frame(message):
                # Approvals, settlements, results, and revalidation ride the
                # reserved control lane: they stay available while this
                # incarnation's snapshot is still installing, and they are
                # never presentation deltas.
                return False
            raise PresentationProtocolError(
                "presentation delta before the incarnation's snapshot installed"
            )
        return False

    def retire(self, generation: int) -> bool:
        """Retire one announced generation; first-winner, exactly once.

        Disarms the retired generation's timers so they can never fire into
        its replacement, and arms the registration-await deadline that
        governs the wait for the replacement announcement.
        """
        if self._terminal is not None or generation != self._current:
            return False
        self._retired.add(generation)
        self._current = None
        self._awaited = generation
        self._assembler = None
        self._phase = StreamPhase.AWAITING
        self._disarm_generation(generation)
        self._arm(
            TimerKind.REGISTRATION_AWAIT,
            generation,
            self.registration_await_timeout,
            self._attempts,
        )
        return True

    def fire_due(self) -> list[Timer]:
        """Fire due timers still in scope, then advance the state machine.

        A retired incarnation's timers are disarmed at retirement, and an
        armed timer fires only while its generation is still the one its kind
        governs — so a stale timer can never fire into the replacement.
        """
        if self._terminal is not None:
            return []
        now = self._clock()
        fired: list[Timer] = []
        for kind in list(self._timers):
            timer = self._timers[kind]
            if now < timer.deadline:
                continue
            if not self._in_scope(timer):
                self._disarm(kind)
                continue
            self._disarm(kind)
            fired.append(timer)
        for timer in fired:
            self._on_timer(timer)
        return fired

    def _effective_generation(self) -> int | None:
        """The generation frames may validly carry right now.

        While a replacement is awaited, the retired generation's successor is
        the effective one, so a terminal notice for it still terminates.
        """
        if self._current is not None:
            return self._current
        if self._awaited is not None:
            return self._awaited + 1
        return None

    def _in_scope(self, timer: Timer) -> bool:
        # An install deadline is in scope only while its generation is still
        # the announced one; the await/backoff deadlines are scoped to the
        # retired generation whose replacement they govern. Retirement and
        # announcement acceptance disarm the old generation's timers outright,
        # so a retired incarnation's timers can never fire into the new one.
        if timer.kind is TimerKind.REGISTRATION_TO_INSTALL:
            return timer.generation == self._current
        return timer.generation == self._awaited

    def _new_assembler(self, generation: int) -> SnapshotAssembler:
        return SnapshotAssembler(
            generation,
            page_max_bytes=self._page_max_bytes,
            max_pages=self._snapshot_max_pages,
            max_bytes=self._snapshot_max_bytes,
        )

    def _arm(
        self, kind: TimerKind, generation: int, timeout: float, attempt: int
    ) -> Timer:
        timer = Timer(kind, generation, self._clock() + timeout, attempt)
        self._timers[kind] = timer
        return timer

    def _disarm(self, kind: TimerKind) -> None:
        self._timers.pop(kind, None)

    def _disarm_generation(self, generation: int) -> None:
        for kind in list(self._timers):
            if self._timers[kind].generation == generation:
                self._disarm(kind)

    def _accept_announcement(self, generation: int) -> None:
        """Accept a replacement announcement; first-winner retires the old.

        Acceptance retires the predecessor's await deadline and arms exactly
        one registration-to-install deadline scoped to the announced
        generation.
        """
        old = self._current if self._current is not None else self._awaited
        if old is not None:
            self._retired.add(old)
            self._disarm_generation(old)
        self._current = generation
        self._awaited = None
        self._phase = StreamPhase.INSTALLING
        self._assembler = self._new_assembler(generation)
        self._disarm(TimerKind.REGISTRATION_AWAIT)
        self._arm(
            TimerKind.REGISTRATION_TO_INSTALL,
            generation,
            self.registration_to_install_timeout,
            self._attempts,
        )

    def _accept_page(self, generation: int, message: Mapping[str, Any]) -> bool:
        assembler = self._assembler
        if assembler is None or assembler.generation != generation:
            raise PresentationProtocolError(
                f"snapshot page for incarnation {generation} with no pending snapshot"
            )
        page = parse_snapshot_page(message.get("params"))
        try:
            install = assembler.accept(page)
        except SnapshotAttemptInvalidated as exc:
            self._fail_attempt(str(exc))
            return True
        if install is not None:
            self._install(install)
        return True

    def _install(self, install: SnapshotInstall) -> None:
        """Install the completed snapshot atomically; one terminal transition."""
        self._last_install = install
        self._phase = StreamPhase.INSTALLED
        self._assembler = None
        self._disarm(TimerKind.REGISTRATION_TO_INSTALL)
        self._attempts = 0

    def _fail_attempt(self, reason: str) -> None:
        """Retire the failed attempt before any fresh incarnation is tried."""
        generation = self._current if self._current is not None else self._awaited
        if self._current is not None:
            self._retired.add(self._current)
            self._disarm_generation(self._current)
            self._current = None
        self._awaited = generation
        self._assembler = None
        self._attempts += 1
        if self._attempts >= self.max_attempts:
            self._terminate(f"registration attempts exhausted: {reason}")
            return
        self._phase = StreamPhase.AWAITING
        backoff = self.backoff_schedule[
            min(self._attempts - 1, len(self.backoff_schedule) - 1)
        ]
        self._arm(
            TimerKind.BACKOFF,
            generation if generation is not None else 0,
            backoff,
            self._attempts,
        )

    def _on_timer(self, timer: Timer) -> None:
        if self._terminal is not None:
            return
        match timer.kind:
            case TimerKind.REGISTRATION_AWAIT:
                self._terminate(
                    f"registration await expired for incarnation {timer.generation}"
                )
            case TimerKind.REGISTRATION_TO_INSTALL:
                self._fail_attempt(
                    "registration-to-install deadline expired for incarnation "
                    f"{timer.generation}"
                )
            case TimerKind.BACKOFF:
                self._arm(
                    TimerKind.REGISTRATION_AWAIT,
                    timer.generation,
                    self.registration_await_timeout,
                    timer.attempt,
                )

    def _terminate(self, reason: str) -> None:
        if self._terminal is not None:
            return
        self._terminal = StreamFailure(self.stream_id, reason)
        self._phase = StreamPhase.TERMINAL
        self._timers.clear()
        self._assembler = None


class PresentationConsumer:
    """The TUI-side consumer of the relay's per-stream presentation protocol.

    Tracks the announced incarnation per stream, installs snapshot pages
    atomically per generation, and keeps registration timers
    generation-scoped. All deadlines derive from the injected monotonic
    clock; tests pass a fake clock.
    """

    def __init__(
        self,
        *,
        registration_await_timeout: float = REGISTRATION_AWAIT_TIMEOUT,
        registration_to_install_timeout: float = REGISTRATION_TO_INSTALL_TIMEOUT,
        max_attempts: int = REGISTRATION_MAX_ATTEMPTS,
        backoff_schedule: Sequence[float] = REGISTRATION_BACKOFF_SECONDS,
        page_max_bytes: int = SNAPSHOT_PAGE_MAX_BYTES,
        snapshot_max_pages: int = SNAPSHOT_MAX_PAGES,
        snapshot_max_bytes: int = SNAPSHOT_MAX_BYTES,
        clock: Callable[[], float] = time.monotonic,
        listener: PresentationStateListener | None = None,
    ) -> None:
        self._clock = clock
        self._registration_await_timeout = registration_await_timeout
        self._registration_to_install_timeout = registration_to_install_timeout
        self._max_attempts = max_attempts
        self._backoff_schedule = _validated_registration_bounds(
            registration_await_timeout,
            registration_to_install_timeout,
            max_attempts,
            backoff_schedule,
        )
        self._page_max_bytes = page_max_bytes
        self._snapshot_max_pages = snapshot_max_pages
        self._snapshot_max_bytes = snapshot_max_bytes
        self._streams: dict[str, PresentationStreamState] = {}
        self._listener = listener

    def set_listener(self, listener: PresentationStateListener | None) -> None:
        """Attach (or detach, with ``None``) the stream-state observer."""
        self._listener = listener

    def _notify(self, stream_id: str, stream_state: PresentationStreamState) -> None:
        listener = self._listener
        if listener is not None:
            listener(stream_id, stream_state)

    def stream(self, stream_id: str) -> PresentationStreamState:
        """Return the stream's state, creating it on first use."""
        existing = self._streams.get(stream_id)
        if existing is None:
            existing = PresentationStreamState(
                stream_id,
                registration_await_timeout=self._registration_await_timeout,
                registration_to_install_timeout=self._registration_to_install_timeout,
                max_attempts=self._max_attempts,
                backoff_schedule=self._backoff_schedule,
                page_max_bytes=self._page_max_bytes,
                snapshot_max_pages=self._snapshot_max_pages,
                snapshot_max_bytes=self._snapshot_max_bytes,
                clock=self._clock,
            )
            self._streams[stream_id] = existing
        return existing

    def receive_frame(
        self, stream_id: str, generation: int | None, message: Mapping[str, Any]
    ) -> bool:
        """Route one inbound relay frame to its stream; report consumption."""
        stream_state = self.stream(stream_id)
        consumed = stream_state.receive_frame(generation, message)
        self._notify(stream_id, stream_state)
        return consumed

    def fire_due(self, stream_id: str | None = None) -> list[Timer]:
        """Fire due timers, for one stream or all of them."""
        if stream_id is not None:
            stream_state = self.stream(stream_id)
            fired = stream_state.fire_due()
            self._notify(stream_id, stream_state)
            return fired
        fired: list[Timer] = []
        for sid, stream_state in list(self._streams.items()):
            fired.extend(stream_state.fire_due())
            self._notify(sid, stream_state)
        return fired
