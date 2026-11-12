"""Wire codec for the agent-process turn protocol.

The root agent runs in its own process behind the agent-side execution port.
This module is the closed schema between that process and its host: turn
requests travel one way, turn events, retry notices, terminal outcomes, and
transcript pages travel the other. Only declared frame payloads are accepted;
an event or request the schema does not name fails typed instead of being
guessed at.

Two core event models carry process-local values that cannot cross a pipe:
``ToolCallEvent`` and ``ToolResultEvent`` hold the tool's Python class, and
tool arguments and results are typed models whose classes live with the tool.
The wire carries the tool name and the JSON form of those values, which is
exactly what the coordinator-side projection reads.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, JsonValue, TypeAdapter, ValidationError

from chartreux.app_server.agent_admission import OperationMarker
from chartreux.app_server.agent_snapshot import (
    AgentOpenReply,
    AgentSessionSnapshot,
    RootOpenAccounting,
)
from chartreux.app_server.models import (
    ContentBlock,
    MentionStats,
    PublicSessionState,
    SessionLogSummary,
)
from chartreux.app_server.protocol import (
    AgentTranscriptGetResponse,
    ConfigWriteParams,
    ConfigWriteResponse,
    ContextInjectParams,
    RuntimeSnapshot,
    SessionPlanHandoffParams,
    TurnSteerParams,
)
from chartreux.app_server.transport import TRANSPORT_ENV_VAR
from chartreux.core.agent_loop._execution_port import ExecutionTurnRequest
from chartreux.core.agent_loop._loop_types import AgentTurnOptions
from chartreux.core.agent_loop._request_broker import UserInputSettlement
from chartreux.core.events import (
    AssistantEvent,
    BackgroundJobsChangedEvent,
    BackgroundWorkEvent,
    BaseEvent,
    CompactEndEvent,
    CompactStartEvent,
    ReasoningEvent,
    RouteTableAppliedEvent,
    SessionTitleUpdatedEvent,
    TokenUsageUpdatedEvent,
    ToolCallEvent,
    ToolResultEvent,
    ToolStreamEvent,
    ToolWaitStateChangedEvent,
    UserInputRequestEvent,
    UserMessageEvent,
    WaitingForInputEvent,
)
from chartreux.core.hooks.models import (
    HookEndEvent,
    HookRunEndEvent,
    HookRunStartEvent,
    HookStartEvent,
)
from chartreux.core.llm_models import ImageAttachment, ManualShellContext
from chartreux.core.model_catalog.availability import (
    AvailabilitySnapshot,
    DeploymentExclusion,
    ExclusionReason,
)
from chartreux.core.session_types import CommittedModelIdentity
from chartreux.core.subagents import TaskArgs, TaskResult
from chartreux.core.tools.builtins.ask_user_question import AskUserQuestionArgs
from chartreux.core.tools.io_port import ShellCommandRequest, ShellCommandResult
from chartreux.core.usage import (
    CoverageWarning,
    UsageRecord,
    UsageWriteDisposition,
    UsageWriteResult,
)
from chartreux.core.utils.retry import RetryCategory, RetryReason
from chartreux.questions import (
    UnansweredQuestions,
    UnattendedExpiryPolicy,
    UserQuestionRequest,
    UserQuestionResult,
)
from chartreux.user_content import UserDisplayContent, UserResource
from chartreux.utils.io import BoundedReadResult, ReadSafeResult
from chartreux.utils.tool_presentation import (
    ToolCallPresentation,
    ToolResultPresentation,
)

AGENT_OPEN_METHOD = "agent/open"
SUBAGENT_OPEN_METHOD = "subagent/open"
AGENT_TURN_START_METHOD = "agent/turn/start"
AGENT_TURN_EVENT_METHOD = "agent/turn/event"
AGENT_TURN_RETRYING_METHOD = "agent/turn/retrying"
AGENT_TURN_ENDED_METHOD = "agent/turn/ended"
AGENT_TURN_RESOLVE_METHOD = "agent/turn/resolve"
AGENT_TURN_REJECT_METHOD = "agent/turn/reject"
AGENT_TURN_RELAY_METHOD = "agent/turn/relay"
AGENT_TURN_ANSWER_METHOD = "agent/turn/answer"
AGENT_TURN_REFUSE_METHOD = "agent/turn/refuse"
AGENT_TURN_KEEP_WAITING_METHOD = "agent/turn/keepWaiting"
AGENT_TURN_CANCEL_METHOD = "agent/turn/cancel"
# The request half of the terminal-settlement pair. The request record lives in
# the process that owns the loop, so that process is the one that settles it --
# but the terminal projection is the coordinator's, and it must land BEFORE the
# waiter unblocks (a reconnect, or a client answering a dead request, must see a
# closed entry, never a ghost open callback). A request/reply frame keeps that
# publication-before-unblock ordering across the pipe: the agent asks the
# coordinator to publish, and settles its waiter on the acknowledgement.
AGENT_TURN_CALLBACK_METHOD = "agent/turn/callback"
AGENT_TRANSCRIPT_PAGE_METHOD = "agent/transcript/page"
# Turn control: the interrupt notification cancels the turn it names, steering
# arbitrates one delivery against a turn in flight, an injection lands between
# turns, and the agent's out-of-band events ride turn-less notices back.
AGENT_TURN_INTERRUPT_METHOD = "agent/turn/interrupt"
AGENT_TURN_STEER_METHOD = "agent/turn/steer"
AGENT_INJECT_METHOD = "agent/inject"
AGENT_NOTICE_METHOD = "agent/notice"

# The manual `!` command's effect runs where the loop lives. The coordinator
# queues the shell input beside queued prompts and asks this process for the
# command's record at idle, so an interrupt can still drop a command that never
# ran; the reply carries the one redacted record the in-process route builds.
AGENT_SHELL_RUN_METHOD = "agent/shell/run"
AGENT_SHELL_INTERRUPT_METHOD = "agent/shell/interrupt"

# Op 4: the snapshot family. A snapshot read answers with the public session
# envelope, the recovery gate's computed decision view, the session-log
# projection and the long-operation marker records, all built agent-side at
# reply-write time. Rewind preview, rewind and history clear are siblings of
# the family: over the process wire they answer typed and name the in_process
# escape instead of pretending a carrier exists.
AGENT_SESSION_SNAPSHOT_METHOD = "agent/session/snapshot"
AGENT_SESSION_REWIND_READ_METHOD = "agent/session/rewind/read"
AGENT_SESSION_REWIND_METHOD = "agent/session/rewind"
AGENT_SESSION_HISTORY_CLEAR_METHOD = "agent/session/history/clear"

# Op 10: the session-identity family. A replace rebinds the session the agent
# owns to a stored one, a rename applies one manual title, and the operations
# this release does not carry -- fork, relocate and the rewind pair -- answer
# typed and name the in_process escape instead of pretending a carrier exists.
AGENT_SESSION_REPLACE_METHOD = "agent/session/replace"
AGENT_SESSION_RENAME_METHOD = "agent/session/rename"
AGENT_SESSION_FORK_METHOD = "agent/session/fork"
AGENT_SESSION_RELOCATE_METHOD = "agent/session/relocate"

AGENT_DEFERRED_SESSION_METHODS = frozenset({
    AGENT_SESSION_REWIND_READ_METHOD,
    AGENT_SESSION_REWIND_METHOD,
    AGENT_SESSION_HISTORY_CLEAR_METHOD,
    AGENT_SESSION_FORK_METHOD,
    AGENT_SESSION_RELOCATE_METHOD,
})

# Op 10 mode group: one reconfigure request carries ONE invocation mode. The
# short modes -- a settings apply-and-ack and the /clear history reset -- answer
# once their steps land and are refused typed while a long operation holds the
# agent's fence. The plan handoff and the config write run as long operations:
# the handoff's receipt carries the fence outcome and what it adopted, and the
# config write's receipt carries the SHIPPED transaction result verbatim.
AGENT_SESSION_RECONFIGURE_METHOD = "agent/session/reconfigure"

# The session resources the agent's own config and loop answer: the stored
# session list, one config reload transaction, the durable plan record, and the
# policy read. Each is computed where the state lives, so the coordinator never
# reconstructs a session view from a copy it does not own.
AGENT_SESSION_LIST_METHOD = "agent/session/list"
AGENT_SESSION_CONFIG_RELOAD_METHOD = "agent/session/config/reload"
AGENT_SESSION_PLAN_RECORD_READ_METHOD = "agent/session/plan/record/read"
AGENT_SESSION_PLAN_RECORD_WRITE_METHOD = "agent/session/plan/record/write"
AGENT_SESSION_POLICY_READ_METHOD = "agent/session/policy/read"
# The authorized-roots view and its grant save: both are composed from the
# agent's own orchestrator and layers, so the process that holds them answers.
AGENT_SESSION_ROOTS_READ_METHOD = "agent/session/roots/read"
AGENT_SESSION_ROOTS_SAVE_METHOD = "agent/session/roots/save"

RECONFIGURE_MODE_SETTINGS = "settings-update"
RECONFIGURE_MODE_HISTORY_CLEAR = "history-clear"
RECONFIGURE_MODE_PLAN_HANDOFF = "plan-handoff"
RECONFIGURE_MODE_CONFIG_WRITE = "config-write"
RECONFIGURE_MODES = frozenset({
    RECONFIGURE_MODE_SETTINGS,
    RECONFIGURE_MODE_HISTORY_CLEAR,
    RECONFIGURE_MODE_PLAN_HANDOFF,
    RECONFIGURE_MODE_CONFIG_WRITE,
})
RECONFIGURE_SHORT_MODES = frozenset({
    RECONFIGURE_MODE_SETTINGS,
    RECONFIGURE_MODE_HISTORY_CLEAR,
})
RECONFIGURE_LONG_MODES = frozenset({
    RECONFIGURE_MODE_PLAN_HANDOFF,
    RECONFIGURE_MODE_CONFIG_WRITE,
})

# The reply classes the agent mints. A clear that completed its steps and then
# could not finish a follow-up stage is its own class: the history IS cleared,
# so reporting a plain failure would tell the user to retry a destructive step
# that already ran.
RECONFIGURE_OUTCOME_APPLIED = "applied"
RECONFIGURE_OUTCOME_CLEARED = "cleared"
RECONFIGURE_OUTCOME_CLEARED_WITH_FOLLOWUP_FAILURE = "cleared-with-followup-failure"
RECONFIGURE_OUTCOME_ACCEPTED = "accepted"
RECONFIGURE_OUTCOMES = frozenset({
    RECONFIGURE_OUTCOME_APPLIED,
    RECONFIGURE_OUTCOME_CLEARED,
    RECONFIGURE_OUTCOME_CLEARED_WITH_FOLLOWUP_FAILURE,
    RECONFIGURE_OUTCOME_ACCEPTED,
})

# The steps a history clear names in its receipt, in the order they run.
CLEAR_STAGE_HISTORY = "history-cleared"
CLEAR_STAGE_MODEL_OVERRIDE = "model-override-cleared"
CLEAR_STAGE_ALREADY_CLEAR = "already-clear"

RECONFIGURE_CONFLICT_ERROR = "AgentReconfigureConflictError"
RECONFIGURE_REFUSED_ERROR = "AgentReconfigureRefusedError"
RECONFIGURE_FAILED_ERROR = "AgentReconfigureFailedError"

AGENT_SESSION_NOT_FOUND_ERROR = "AgentSessionNotFoundError"
"""The wire error code a replace answers when the session it names is unknown."""

# Op 5: manual compaction. The request is a long operation -- the coordinator's
# absolute deadline rides its params and its reply carries a publication
# receipt -- and the id-less cancel notification names the operation it
# cancels. A cancel has no reply of its own: the operation's settlement is the
# only assertable outcome.
AGENT_COMPACT_METHOD = "agent/compact"
AGENT_OP_CANCEL_METHOD = "agent/op/cancel"

# The long-operation admission announcement: an id-less notification with its
# own method string. It must not ride the request's reply id, because a peer
# settles one future per request id and drops later same-id frames; only the
# operation's final completion settles the request id.
AGENT_OP_ADMISSION_METHOD = "agent/op/admission"

# The op-9 host-service channel: the agent-initiated requests. The coordinator
# owns the two host REQUEST ports a turn needs — the subagent runner and the
# client tool I/O port — so the agent asks it for them over the same pipe
# instead of receiving them across the process boundary.
SUBAGENT_RUN_METHOD = "subagent/run"
HOST_TOOL_IO_METHOD = "host/tool/io"
# The coordinator's answers to a host subagent run: id-less notifications that
# stream the run's frames and close it exactly once.
HOST_SUBAGENT_EVENT_METHOD = "host/subagent/event"
HOST_SUBAGENT_ENDED_METHOD = "host/subagent/ended"

# The cross-process policy-tree phases (WP5 [5D]): one node serves these
# frames, and the coordinator-side transaction drives them in order.
POLICY_REVISION_METHOD = "policy/revision"
POLICY_RESERVE_METHOD = "policy/reserve"
POLICY_PREPARE_METHOD = "policy/prepare"
POLICY_COMMIT_METHOD = "policy/commit"
POLICY_RETIRE_METHOD = "policy/retire"
POLICY_ABORT_METHOD = "policy/abort"
POLICY_METHODS = frozenset({
    POLICY_REVISION_METHOD,
    POLICY_RESERVE_METHOD,
    POLICY_PREPARE_METHOD,
    POLICY_COMMIT_METHOD,
    POLICY_RETIRE_METHOD,
    POLICY_ABORT_METHOD,
})

# Item 1 child seam: the publication teardown. The aggregate child publication
# transaction answers ONE reply; when that reply is lost the coordinator still
# holds the attempt id it minted, but not the outcome, so it asks the agent that
# owns the children to inspect the attempt's durable records, cancel the
# prepared child and retract what it persisted. The reply names what the
# teardown ESTABLISHED, never a claimed success: an attempt that cannot be
# settled inside the bounded rollback wait answers inconclusive.
AGENT_CHILD_TEARDOWN_METHOD = "agent/child/teardown"
CHILD_TEARDOWN_REAPED = "reaped"
CHILD_TEARDOWN_TOMBSTONED = "tombstoned"
CHILD_TEARDOWN_INCONCLUSIVE = "inconclusive"
CHILD_TEARDOWN_ATTEMPT_UNKNOWN = "attempt-unknown"
CHILD_TEARDOWN_OUTCOMES = frozenset({
    CHILD_TEARDOWN_REAPED,
    CHILD_TEARDOWN_TOMBSTONED,
    CHILD_TEARDOWN_INCONCLUSIVE,
    CHILD_TEARDOWN_ATTEMPT_UNKNOWN,
})

# Origin namespaces. Each peer mints its request ids from its own namespace, so
# a coordinator-origin id can never collide with an agent-origin one once the
# agent starts originating requests of its own (item 1 op 9).
COORDINATOR_ORIGIN = "coordinator"
AGENT_ORIGIN = "agent"

_COORDINATOR_ID_PREFIX = f"{COORDINATOR_ORIGIN}:"
_AGENT_ID_PREFIX = f"{AGENT_ORIGIN}:"

# The closed per-direction method tables. Dispatch keys on method presence plus
# membership here, never on bare id presence: an id-bearing frame naming an
# unlisted method is a request the receiver answers with a typed compat
# rejection, and an id-less frame naming one is a notification it records as a
# diagnostic while the session continues.
AGENT_RECEIVING_REQUESTS = (
    frozenset({
        AGENT_OPEN_METHOD,
        AGENT_TRANSCRIPT_PAGE_METHOD,
        AGENT_TURN_ANSWER_METHOD,
        AGENT_TURN_CANCEL_METHOD,
        AGENT_TURN_KEEP_WAITING_METHOD,
        AGENT_TURN_REFUSE_METHOD,
        # Turn control: steering arbitrates one delivery against a turn in
        # flight, and an injection lands between turns.
        AGENT_TURN_STEER_METHOD,
        AGENT_INJECT_METHOD,
        # The manual `!` command's effect and its interrupt: both answer on
        # their own request ids, so a command that cannot run is a typed
        # refusal rather than an unknown method.
        AGENT_SHELL_RUN_METHOD,
        AGENT_SHELL_INTERRUPT_METHOD,
        # Op 4: the snapshot family. Rewind and clear are listed because their
        # refusals are typed answers on their own request ids, not unknowns.
        AGENT_SESSION_SNAPSHOT_METHOD,
        AGENT_SESSION_REWIND_READ_METHOD,
        AGENT_SESSION_REWIND_METHOD,
        AGENT_SESSION_HISTORY_CLEAR_METHOD,
        # Op 5: compaction answers on its own request id with a settlement
        # receipt, so a cancelled or failed run is still a typed answer.
        AGENT_COMPACT_METHOD,
        # Op 10: the identity family. A replace and a rename are carried; the
        # deferred siblings are listed because their refusals are typed answers
        # on their own request ids, not unknowns.
        AGENT_SESSION_REPLACE_METHOD,
        AGENT_SESSION_RENAME_METHOD,
        AGENT_SESSION_FORK_METHOD,
        AGENT_SESSION_RELOCATE_METHOD,
        # Op 10 mode group: settings, history clear and the plan handoff answer
        # on their own request ids, so a refused or aborted mode is still a
        # typed answer rather than an unknown method.
        AGENT_SESSION_RECONFIGURE_METHOD,
        # The session resources answered from the agent's own config and loop:
        # the stored session list, one config reload, the durable plan record
        # and the policy read.
        AGENT_SESSION_LIST_METHOD,
        AGENT_SESSION_CONFIG_RELOAD_METHOD,
        AGENT_SESSION_PLAN_RECORD_READ_METHOD,
        AGENT_SESSION_PLAN_RECORD_WRITE_METHOD,
        AGENT_SESSION_POLICY_READ_METHOD,
        AGENT_SESSION_ROOTS_READ_METHOD,
        AGENT_SESSION_ROOTS_SAVE_METHOD,
        # Served by the child agent process; the root refuses it typed.
        SUBAGENT_OPEN_METHOD,
        # Item 1 child seam: the publication teardown answers on its own
        # request id with the class it established, including the typed
        # refusal of an attempt this end never owned.
        AGENT_CHILD_TEARDOWN_METHOD,
    })
    | POLICY_METHODS
)

AGENT_RECEIVING_NOTIFICATIONS = frozenset({
    AGENT_TURN_INTERRUPT_METHOD,
    # Op 5: an id-less, idempotent cancel scoped to the compaction it names.
    AGENT_OP_CANCEL_METHOD,
    AGENT_TURN_REJECT_METHOD,
    AGENT_TURN_RELAY_METHOD,
    AGENT_TURN_RESOLVE_METHOD,
    AGENT_TURN_START_METHOD,
    # The coordinator's streamed answers to an agent-origin host subagent run.
    HOST_SUBAGENT_EVENT_METHOD,
    HOST_SUBAGENT_ENDED_METHOD,
})

COORDINATOR_RECEIVING_REQUESTS: frozenset[str] = frozenset({
    # The op-9 host-service channel: the agent asks the coordinator for the
    # two host REQUEST ports it serves. Both are served from the ports the
    # coordinator offered with the turn in flight; outside a turn there is
    # nothing to serve and the request is answered typed.
    HOST_TOOL_IO_METHOD,
    SUBAGENT_RUN_METHOD,
    # The terminal-settlement pair: the agent settled a user-input request in
    # the process that owns its record and asks this end to publish the
    # terminal projection before its waiter unblocks. The reply carries
    # whether the projection was published, never a claimed settlement.
    AGENT_TURN_CALLBACK_METHOD,
})
"""Agent-origin requests the coordinator serves."""

COORDINATOR_RECEIVING_NOTIFICATIONS = frozenset({
    AGENT_NOTICE_METHOD,
    AGENT_OP_ADMISSION_METHOD,
    AGENT_TURN_ENDED_METHOD,
    AGENT_TURN_EVENT_METHOD,
    AGENT_TURN_RETRYING_METHOD,
})

TURN_OUTCOME_COMPLETED = "completed"
TURN_OUTCOME_FAILED = "failed"
TURN_OUTCOME_INTERRUPTED = "interrupted"

# The three publication classes a compaction receipt carries. Only the agent
# mints them: it is the peer that knows whether a boundary was published and
# whether the state that follows it was settled. A coordinator timeout is an
# observation of its own and never rides these classes.
COMPACT_SETTLEMENT_PUBLISHED_SETTLED = "published-and-settled"
COMPACT_SETTLEMENT_PUBLISHED_UNSETTLED = "published-but-unsettled"
COMPACT_SETTLEMENT_CANCELLED_UNPUBLISHED = "cancelled-unpublished"
COMPACT_AGENT_SETTLEMENTS = frozenset({
    COMPACT_SETTLEMENT_PUBLISHED_SETTLED,
    COMPACT_SETTLEMENT_PUBLISHED_UNSETTLED,
    COMPACT_SETTLEMENT_CANCELLED_UNPUBLISHED,
})


def coordinator_request_id(sequence: int) -> str:
    """Mint the coordinator-origin request id of one wire request."""
    return f"{_COORDINATOR_ID_PREFIX}{sequence}"


def agent_request_id(sequence: int) -> str:
    """Mint the agent-origin request id of one agent-initiated request."""
    return f"{_AGENT_ID_PREFIX}{sequence}"


def request_id_origin(request_id: object) -> str | None:
    """The origin namespace of one request id, or ``None`` for a foreign id."""
    if not isinstance(request_id, str):
        return None
    if request_id.startswith(_COORDINATOR_ID_PREFIX):
        return COORDINATOR_ORIGIN
    if request_id.startswith(_AGENT_ID_PREFIX):
        return AGENT_ORIGIN
    return None


def deferred_session_message(method: str) -> str:
    """The typed refusal of one session method this release does not carry.

    Both peers use this one string: the agent refuses a frame it receives and
    the coordinator refuses the operation before it would send one, so the user
    reads the same next step either way -- run the session on the in-process
    transport.
    """
    return (
        f"{method} is not carried over the process wire in this release: run "
        f"the session with the in_process transport ({TRANSPORT_ENV_VAR}=in_process)"
    )


@dataclass(frozen=True, slots=True)
class WireDiagnostic:
    """One wire-compat observation a receiver records instead of failing.

    Notifications carry no reply correlation, so a frame a receiver cannot
    honor is a visible warning that leaves the session running rather than a
    session failure. The record is kept on the receiving endpoint and written to
    the log, so a lost notification is never silent.
    """

    direction: str
    reason: str
    method: object = None
    request_id: object = None

    def message(self) -> str:
        """The one user-facing warning line this diagnostic renders as."""
        correlation = (
            f" (request id {self.request_id!r})" if self.request_id is not None else ""
        )
        return (
            f"Wire compatibility ({self.direction}-origin frame): {self.reason}"
            f"{correlation}; the session continues"
        )


class AgentWireError(RuntimeError):
    """Base class for the agent-process wire schema failures."""


class UnsupportedAgentEventError(AgentWireError):
    """The agent produced an event the process wire schema does not name."""

    def __init__(self, event_name: str) -> None:
        super().__init__(f"event {event_name!r} is not part of the agent wire schema")
        self.event_name = event_name


class UnsupportedAgentRequestError(AgentWireError):
    """A turn request carries a value that has no process representation."""


class UnsupportedUserInputResultError(AgentWireError):
    """A user-input result class is not part of the agent wire schema."""


class RemoteUserInputRejectedError(AgentWireError):
    """The host rejected an open user-input request across the process wire."""


class WireMethodNotSupportedError(AgentWireError):
    """A frame names a method outside its direction's closed table.

    An id-bearing request is answered with this typed rejection naming the
    method, so the peer settles instead of hanging; an id-less notification
    cannot be answered, so the receiver records a diagnostic instead.
    """

    def __init__(self, method: object, *, direction: str) -> None:
        super().__init__(
            f"{direction}-origin frame names method {method!r}, which is not part "
            "of this direction's closed method table"
        )
        self.method = method
        self.direction = direction


class RemoteWireRequestError(AgentWireError):
    """The peer answered an agent-origin request with a typed wire failure."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class WireTransportClosedError(AgentWireError):
    """The peer pipe closed: every request this end awaits is settled typed."""


class AgentSessionNotFoundError(AgentWireError):
    """A request names a stored session this agent cannot find.

    The class name is the wire error code the coordinator maps back to the
    typed not-found refusal, so a lookup never surfaces as a generic failure.
    """


class AgentReconfigureConflictError(AgentWireError):
    """A reconfigure mode is refused before any step runs: it conflicts.

    The refusal classes mirror the in-process one a reconfigure answers with: a
    conflicting state (an active turn, live background work, an episode that
    cannot accept) is a CONFLICT, a malformed request or consent body is a
    param refusal, and a step that failed after the operation started is an
    internal failure.
    """


class AgentReconfigureRefusedError(AgentWireError):
    """A reconfigure request is refused typed: malformed or inadmissible."""


class AgentReconfigureFailedError(AgentWireError):
    """A reconfigure step failed after its operation had started."""


class AgentShellRefusedError(AgentWireError):
    """A manual ``!`` command the agent refuses before it would execute.

    The class name is the wire error code the coordinator maps back to the
    in-process param refusal, so a working directory that does not exist reads
    as a plain error on either route.
    """


_JSON_EVENT_TYPES: tuple[type[BaseEvent], ...] = (
    AssistantEvent,
    BackgroundJobsChangedEvent,
    BackgroundWorkEvent,
    CompactEndEvent,
    CompactStartEvent,
    HookEndEvent,
    HookRunEndEvent,
    HookRunStartEvent,
    HookStartEvent,
    ReasoningEvent,
    RouteTableAppliedEvent,
    SessionTitleUpdatedEvent,
    TokenUsageUpdatedEvent,
    ToolStreamEvent,
    ToolWaitStateChangedEvent,
    UserMessageEvent,
    WaitingForInputEvent,
)
_EVENT_TYPES: dict[str, type[BaseEvent]] = {
    event_type.__name__: event_type for event_type in _JSON_EVENT_TYPES
}

_USER_INPUT_ARGS_TYPES: dict[str, type[BaseModel]] = {
    UserQuestionRequest.__name__: UserQuestionRequest,
    # The model-facing question tool's own request: it adds the per-request
    # unattended-timeout override, which the host relays verbatim, so the
    # shipped args type is the one that crosses the pipe.
    AskUserQuestionArgs.__name__: AskUserQuestionArgs,
}
_USER_INPUT_RESULT_TYPES: dict[str, type[BaseModel]] = {
    result_type.__name__: result_type
    for result_type in (UserQuestionResult, UnansweredQuestions)
}

_IMAGES_ADAPTER: TypeAdapter[list[ImageAttachment]] = TypeAdapter(list[ImageAttachment])
_RESOURCES_ADAPTER: TypeAdapter[list[UserResource]] = TypeAdapter(list[UserResource])


def encode_event(event: BaseEvent) -> dict[str, Any]:
    """Project one core event into its JSON-safe wire payload."""
    match event:
        case ToolCallEvent():
            return {
                "type": "ToolCallEvent",
                "tool_call_id": event.tool_call_id,
                "tool_name": event.tool_name,
                "tool_call_index": event.tool_call_index,
                "args": _encode_value(event.args),
                "presentation": _encode_value(event.presentation),
            }
        case ToolResultEvent():
            payload = event.model_dump(mode="json", exclude={"tool_class", "result"})
            payload["type"] = "ToolResultEvent"
            payload["result"] = _encode_value(event.result)
            return payload
        case UserInputRequestEvent():
            return {
                "type": "UserInputRequestEvent",
                "request_id": event.request_id,
                "tool_call_id": event.tool_call_id,
                "origin": event.origin.value,
                "args_type": type(event.args).__name__,
                "args": _encode_value(event.args),
                "unattended_expiry": (
                    event.unattended_expiry.model_dump(mode="json")
                    if event.unattended_expiry is not None
                    else None
                ),
            }
    event_type = _EVENT_TYPES.get(type(event).__name__)
    if event_type is None:
        raise UnsupportedAgentEventError(type(event).__name__)
    return {"type": event_type.__name__, "data": event.model_dump(mode="json")}


def decode_event(payload: Mapping[str, Any]) -> BaseEvent:
    """Reconstruct one core event from its wire payload."""
    kind = payload.get("type")
    if not isinstance(kind, str):
        raise UnsupportedAgentEventError(repr(kind))
    if kind == "ToolCallEvent":
        return ToolCallEvent.model_construct(
            tool_call_id=payload["tool_call_id"],
            tool_name=payload["tool_name"],
            tool_class=None,
            tool_call_index=payload.get("tool_call_index"),
            args=payload.get("args"),
            presentation=_decode_presentation(
                ToolCallPresentation, payload.get("presentation")
            ),
        )
    if kind == "ToolResultEvent":
        fields = dict(payload)
        fields.pop("type", None)
        fields["tool_class"] = None
        fields["presentation"] = _decode_presentation(
            ToolResultPresentation, fields.get("presentation")
        )
        return ToolResultEvent.model_construct(**fields)
    if kind == "UserInputRequestEvent":
        args_type = payload.get("args_type")
        args_model = _USER_INPUT_ARGS_TYPES.get(str(args_type))
        if args_model is None:
            raise UnsupportedAgentRequestError(
                f"user-input args {args_type!r} are not part of the wire schema"
            )
        expiry_payload = payload.get("unattended_expiry")
        return UserInputRequestEvent(
            request_id=payload["request_id"],
            tool_call_id=payload["tool_call_id"],
            origin=payload["origin"],
            args=args_model.model_validate(payload["args"]),
            unattended_expiry=(
                UnattendedExpiryPolicy.model_validate(expiry_payload)
                if expiry_payload is not None
                else None
            ),
        )
    event_type = _EVENT_TYPES.get(kind)
    if event_type is None:
        raise UnsupportedAgentEventError(kind)
    return event_type.model_validate(payload.get("data"))


def encode_turn_request(request: ExecutionTurnRequest) -> dict[str, Any]:
    """Encode the serializable part of a turn request."""
    options = request.turn_options or AgentTurnOptions()
    return {
        "message": request.message,
        "client_message_id": request.client_message_id,
        "auto_title": request.auto_title,
        "images": [image.model_dump(mode="json") for image in request.images or ()],
        "user_display_content": _encode_value(request.user_display_content),
        "input_text": request.input_text,
        "resources": [_encode_value(resource) for resource in request.resources or ()],
        "turn_options": {
            "turn_id": options.turn_id,
            "injected": options.injected,
            "user_initiated_retry": options.user_initiated_retry,
        },
    }


def decode_turn_request(params: Mapping[str, Any]) -> ExecutionTurnRequest:
    """Rebuild a turn request, refusing values the wire cannot carry."""
    if params.get("subagent_runner") is not None or params.get("tool_io") is not None:
        raise UnsupportedAgentRequestError(
            "host-side subagent and tool I/O ports cannot cross the process wire"
        )
    options = params.get("turn_options") or {}
    display_content = params.get("user_display_content")
    return ExecutionTurnRequest(
        message=params.get("message"),
        client_message_id=params.get("client_message_id"),
        auto_title=params.get("auto_title"),
        images=_IMAGES_ADAPTER.validate_python(params.get("images") or []),
        user_display_content=(
            UserDisplayContent.model_validate(display_content)
            if display_content is not None
            else None
        ),
        input_text=params.get("input_text"),
        resources=_RESOURCES_ADAPTER.validate_python(params.get("resources") or []),
        turn_options=AgentTurnOptions(
            turn_id=options.get("turn_id"),
            injected=bool(options.get("injected")),
            user_initiated_retry=bool(options.get("user_initiated_retry")),
        ),
    )


def encode_retry_reason(reason: RetryReason) -> dict[str, str]:
    return {"category": reason.category.value, "detail": reason.detail}


def decode_retry_reason(payload: Mapping[str, Any]) -> RetryReason:
    return RetryReason(
        category=RetryCategory(payload["category"]), detail=str(payload["detail"])
    )


def encode_user_input_result(result: BaseModel) -> dict[str, Any]:
    type_name = type(result).__name__
    if type_name not in _USER_INPUT_RESULT_TYPES:
        raise UnsupportedUserInputResultError(
            f"user-input result {type_name!r} is not part of the wire schema"
        )
    return {"type": type_name, "data": result.model_dump(mode="json")}


def decode_user_input_result(payload: Mapping[str, Any]) -> BaseModel:
    result_type = _USER_INPUT_RESULT_TYPES.get(str(payload.get("type")))
    if result_type is None:
        raise UnsupportedUserInputResultError(
            f"user-input result {payload.get('type')!r} is not part of the wire schema"
        )
    return result_type.model_validate(payload.get("data"))


USER_INPUT_SETTLEMENT_KINDS = frozenset({
    "answered",
    "rejected",
    "expired",
    "cancelled",
})


def encode_user_input_settlement(
    request_id: str, settlement: UserInputSettlement
) -> dict[str, Any]:
    """Project one terminal user-input disposition into its wire payload."""
    payload: dict[str, Any] = {
        "request_id": request_id,
        "kind": settlement.kind,
        "message": settlement.message,
        "was_open": settlement.was_open,
    }
    if settlement.result is not None:
        payload["result"] = encode_user_input_result(settlement.result)
    return payload


def decode_user_input_settlement(
    payload: Mapping[str, Any],
) -> tuple[str, UserInputSettlement]:
    """Reconstruct one terminal disposition, or fail typed on an unknown kind."""
    kind = payload.get("kind")
    if kind not in USER_INPUT_SETTLEMENT_KINDS:
        raise UnsupportedUserInputResultError(
            f"user-input settlement {kind!r} is not part of the wire schema"
        )
    result = payload.get("result")
    message = payload.get("message")
    return str(payload.get("request_id")), UserInputSettlement(
        kind=cast("Any", kind),
        result=(
            decode_user_input_result(result) if isinstance(result, Mapping) else None
        ),
        message=message if isinstance(message, str) else None,
        was_open=bool(payload.get("was_open", True)),
    )


def encode_transcript_page(response: AgentTranscriptGetResponse) -> dict[str, Any]:
    return response.model_dump(mode="json")


def decode_transcript_page(payload: Mapping[str, Any]) -> AgentTranscriptGetResponse:
    return AgentTranscriptGetResponse.model_validate(payload)


# --- op-9 host-service channel: codecs ---------------------------------------

TOOL_IO_CAPABILITY_FIELDS = ("supports_read", "supports_write", "supports_terminal")

TOOL_IO_OPS = frozenset({"read_lines", "read_text", "write_text", "run_shell"})


@dataclass(frozen=True, slots=True)
class ToolIOCapabilities:
    """The static capability flags the host's tool I/O port advertises."""

    supports_read: bool = False
    supports_write: bool = False
    supports_terminal: bool = False


@dataclass(frozen=True, slots=True)
class HostServiceOffers:
    """The host ports a turn offer tells the agent it may call back for."""

    subagent_runner: bool
    tool_io: ToolIOCapabilities | None


def encode_host_service_offers(
    *, subagent_runner: object | None, tool_io: object | None
) -> dict[str, Any]:
    """Announce the host ports a turn offer makes reachable.

    The capability flags are static facts about the host's port, so they ride
    the turn offer: the agent-side proxy needs them synchronously
    (``supports_read`` and friends are plain properties), and asking per call
    would make a capability probe a network round trip.
    """
    capabilities = None
    if tool_io is not None:
        capabilities = {
            field: bool(getattr(tool_io, field, False))
            for field in TOOL_IO_CAPABILITY_FIELDS
        }
    return {"subagent_runner": subagent_runner is not None, "tool_io": capabilities}


def decode_host_service_offers(params: Mapping[str, Any]) -> HostServiceOffers:
    """Read the host-port offer of one turn request; absent means none."""
    offers = _mapping(params.get("host_services"))
    raw = offers.get("tool_io")
    capabilities = None
    if isinstance(raw, Mapping):
        capabilities = ToolIOCapabilities(**{
            field: bool(raw.get(field)) for field in TOOL_IO_CAPABILITY_FIELDS
        })
    return HostServiceOffers(
        subagent_runner=bool(offers.get("subagent_runner")), tool_io=capabilities
    )


_HOST_CONTEXT_FIELDS = (
    "tool_call_id",
    "session_id",
    "session_dir",
    "scratchpad_dir",
    "is_subagent",
)


def encode_host_context(context: object) -> dict[str, Any]:
    """The portable ``InvokeContext`` fields a host subagent run needs.

    The context itself never crosses the pipe — its collaborators are
    process-local. Only the identity and location fields the coordinator's
    runner reads travel; the coordinator rebuilds the context around its own
    collaborators.
    """
    payload: dict[str, Any] = {}
    for field_name in _HOST_CONTEXT_FIELDS:
        value = getattr(context, field_name, None)
        if isinstance(value, Path):
            value = str(value)
        if value is not None:
            payload[field_name] = value
    payload["is_subagent"] = bool(payload.get("is_subagent"))
    return payload


def decode_host_context(payload: Mapping[str, Any]) -> dict[str, Any]:
    """The normalized portable context fields of one host subagent request."""
    return {
        "tool_call_id": str(payload.get("tool_call_id") or ""),
        "session_id": _optional_str(payload.get("session_id")),
        "session_dir": _optional_str(payload.get("session_dir")),
        "scratchpad_dir": _optional_str(payload.get("scratchpad_dir")),
        "is_subagent": bool(payload.get("is_subagent")),
    }


def encode_subagent_run_request(
    run_id: str, args: TaskArgs, context: object
) -> dict[str, Any]:
    """Encode one agent-origin host subagent run request."""
    return {
        "run_id": run_id,
        "args": args.model_dump(mode="json"),
        "context": encode_host_context(context),
    }


def decode_subagent_run_request(
    params: Mapping[str, Any],
) -> tuple[str, TaskArgs, dict[str, Any]]:
    """Rebuild one host subagent run request, refusing a malformed frame."""
    run_id = params.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise UnsupportedAgentRequestError(
            "a host subagent run request carries no run id"
        )
    args_payload = params.get("args")
    if not isinstance(args_payload, Mapping):
        raise UnsupportedAgentRequestError(
            "a host subagent run request carries no task arguments"
        )
    return (
        run_id,
        TaskArgs.model_validate(dict(args_payload)),
        decode_host_context(_mapping(params.get("context"))),
    )


def encode_host_run_frame(frame: ToolStreamEvent | TaskResult) -> dict[str, Any]:
    """Encode one frame of a host subagent run: a tool event or the result."""
    if isinstance(frame, TaskResult):
        return {"type": "TaskResult", "data": frame.model_dump(mode="json")}
    return encode_event(frame)


def decode_host_run_frame(payload: Mapping[str, Any]) -> ToolStreamEvent | TaskResult:
    """Rebuild one frame of a host subagent run.

    A host run streams tool events and one terminal result; any other event
    shape is outside this frame's schema and fails typed.
    """
    if payload.get("type") == "TaskResult":
        return TaskResult.model_validate(payload.get("data"))
    event = decode_event(payload)
    if not isinstance(event, ToolStreamEvent):
        raise UnsupportedAgentEventError(type(event).__name__)
    return event


def encode_host_run_end(
    *, result: TaskResult | None = None, error: str | None = None
) -> dict[str, Any]:
    """Encode the terminal host subagent run frame."""
    return {
        "result": result.model_dump(mode="json") if result is not None else None,
        "error": error,
    }


def decode_host_run_end(
    payload: Mapping[str, Any],
) -> tuple[TaskResult | None, str | None]:
    """Rebuild the terminal host subagent run frame."""
    raw_result = payload.get("result")
    result = (
        TaskResult.model_validate(raw_result)
        if isinstance(raw_result, Mapping)
        else None
    )
    error = payload.get("error")
    return result, error if isinstance(error, str) and error else None


def encode_tool_io_request(
    op: str, params: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Encode one agent-origin tool I/O request."""
    if op not in TOOL_IO_OPS:
        raise UnsupportedAgentRequestError(
            f"tool I/O op {op!r} is not part of the wire schema"
        )
    return {"op": op, **dict(params or {})}


def decode_tool_io_request(params: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """Rebuild one tool I/O request, refusing an unlisted operation."""
    op = params.get("op")
    if not isinstance(op, str) or op not in TOOL_IO_OPS:
        raise UnsupportedAgentRequestError(
            f"tool I/O op {op!r} is not part of the wire schema"
        )
    return op, {key: value for key, value in params.items() if key != "op"}


def encode_tool_io_result(op: str, value: object) -> dict[str, Any]:
    """Encode one tool I/O reply payload."""
    if op == "read_lines" and isinstance(value, BoundedReadResult):
        return {
            "result": {
                "lines": list(value.lines),
                "total_lines": value.total_lines,
                "was_truncated": value.was_truncated,
            }
        }
    if op == "read_text" and isinstance(value, ReadSafeResult):
        return {
            "result": {
                "text": value.text,
                "encoding": value.encoding,
                "newline": value.newline,
            }
        }
    if op == "run_shell" and isinstance(value, ShellCommandResult):
        return {
            "result": {
                "stdout": value.stdout,
                "stderr": value.stderr,
                "returncode": value.returncode,
            }
        }
    return {"result": None}


def decode_tool_io_result(op: str, payload: Mapping[str, Any]) -> object:
    """Rebuild one tool I/O reply value from its wire payload."""
    raw = _mapping(payload.get("result"))
    if op == "read_lines":
        total_lines = raw.get("total_lines")
        return BoundedReadResult(
            lines=[str(line) for line in raw.get("lines") or []],
            total_lines=total_lines if isinstance(total_lines, int) else None,
            was_truncated=bool(raw.get("was_truncated")),
        )
    if op == "read_text":
        return ReadSafeResult(
            text=str(raw.get("text") or ""),
            encoding=str(raw.get("encoding") or "utf-8"),
            newline=str(raw.get("newline") or "\n"),
        )
    if op == "run_shell":
        returncode = raw.get("returncode")
        return ShellCommandResult(
            stdout=str(raw.get("stdout") or ""),
            stderr=str(raw.get("stderr") or ""),
            returncode=returncode if isinstance(returncode, int) else -1,
        )
    return None


def encode_shell_command_request(request: ShellCommandRequest) -> dict[str, Any]:
    """Encode one shell command request for the host's tool I/O port."""
    return {
        "session_id": request.session_id,
        "tool_call_id": request.tool_call_id,
        "command": request.command,
        "cwd": str(request.cwd),
        "timeout": request.timeout,
        "max_output_bytes": request.max_output_bytes,
        "args": list(request.args) if request.args is not None else None,
        "env": dict(request.env) if request.env is not None else None,
    }


def decode_shell_command_request(payload: Mapping[str, Any]) -> ShellCommandRequest:
    """Rebuild one shell command request from its wire payload."""
    cwd = payload.get("cwd")
    args = payload.get("args")
    env = payload.get("env")
    return ShellCommandRequest(
        session_id=str(payload.get("session_id") or ""),
        tool_call_id=str(payload.get("tool_call_id") or ""),
        command=str(payload.get("command") or ""),
        cwd=Path(str(cwd)) if isinstance(cwd, str) and cwd else Path("."),
        timeout=float(payload.get("timeout") or 0.0),
        max_output_bytes=int(payload.get("max_output_bytes") or 0),
        args=[str(item) for item in args] if isinstance(args, list) else None,
        env={str(key): str(value) for key, value in env.items()}
        if isinstance(env, Mapping)
        else None,
    )


# --- turn control: interrupt, steering, injection and notices ----------------

_MANUAL_SHELL_STATUSES = frozenset({
    "ok",
    "refused",
    "failed",
    "timed_out",
    "interrupted",
})

_CONTENT_BLOCKS_ADAPTER: TypeAdapter[list[ContentBlock]] = TypeAdapter(
    list[ContentBlock]
)

# The loop's out-of-band event classes: the only ones a turn-less notice frame
# carries. Everything else belongs to a turn's own event stream.
_NOTICE_EVENT_NAMES = frozenset({
    "BackgroundJobsChangedEvent",
    "BackgroundWorkEvent",
    "RouteTableAppliedEvent",
    "SessionTitleUpdatedEvent",
})


@dataclass(frozen=True, slots=True)
class SteerSettlement:
    """One steering delivery's settlement: whether it committed, and its events."""

    committed: bool
    events: int = 0


def encode_manual_shell_context(context: ManualShellContext) -> dict[str, Any]:
    """Encode the manual-shell record that rides an injection.

    The record is message metadata, never turn input: its text is the ordinary
    user message, and this payload is what the timeline rebuild reads.
    """
    return context.model_dump(mode="json")


def decode_manual_shell_context(payload: Mapping[str, Any]) -> ManualShellContext:
    """Rebuild a manual-shell record, refusing an outcome class it cannot name.

    ``exit_code`` is optional: a run that timed out, was interrupted, or never
    reached an exit has no code to report, and the status carries its outcome.
    """
    status = payload.get("status")
    if status is not None and status not in _MANUAL_SHELL_STATUSES:
        raise UnsupportedAgentRequestError(
            f"manual-shell status {status!r} is not part of the wire schema"
        )
    exit_code = payload.get("exit_code")
    if exit_code is not None and not isinstance(exit_code, int):
        raise UnsupportedAgentRequestError(
            "a manual-shell exit code is an integer or absent"
        )
    try:
        return ManualShellContext.model_validate(dict(payload))
    except ValidationError as exc:
        raise UnsupportedAgentRequestError(
            f"a manual-shell record is malformed: {exc}"
        ) from exc


@dataclass(frozen=True, slots=True)
class ShellRunRequest:
    """One queued manual ``!`` command the agent executes for its record.

    The command, where the user pointed it, and its own deadline cross the
    pipe; the effect, the redaction and the capped record all happen in the
    process that owns the loop, so both topologies produce the same record.
    """

    operation_id: str
    command: str
    cwd: str | None = None
    timeout_seconds: float = 30.0


def encode_shell_run_request(request: ShellRunRequest) -> dict[str, Any]:
    """Encode one manual command the coordinator wants executed."""
    return {
        "operation_id": request.operation_id,
        "command": request.command,
        "cwd": request.cwd,
        "timeout_seconds": request.timeout_seconds,
    }


def decode_shell_run_request(payload: Mapping[str, Any]) -> ShellRunRequest:
    """Rebuild one manual command, refusing a frame without its identity."""
    operation_id = payload.get("operation_id")
    if not isinstance(operation_id, str) or not operation_id:
        raise UnsupportedAgentRequestError("a shell run request names no operation")
    command = payload.get("command")
    if not isinstance(command, str):
        raise UnsupportedAgentRequestError("a shell run request carries no command")
    timeout = payload.get("timeout_seconds")
    return ShellRunRequest(
        operation_id=operation_id,
        command=command,
        cwd=_optional_str(payload.get("cwd")),
        timeout_seconds=(
            float(timeout)
            if isinstance(timeout, int | float) and not isinstance(timeout, bool)
            else 30.0
        ),
    )


def encode_shell_run_reply(text: str, context: ManualShellContext) -> dict[str, Any]:
    """Encode one finished command's record: its turn text and its metadata."""
    return {"text": text, "context": encode_manual_shell_context(context)}


def decode_shell_run_reply(
    payload: Mapping[str, Any],
) -> tuple[str, ManualShellContext]:
    """Rebuild one finished command's record, refusing a text-less reply."""
    text = payload.get("text")
    if not isinstance(text, str):
        raise UnsupportedAgentRequestError("a shell run reply carries no record text")
    return text, decode_manual_shell_context(_mapping(payload.get("context")))


def encode_shell_interrupt(operation_id: str) -> dict[str, Any]:
    """Encode one manual command's interrupt: the operation it names."""
    return {"operation_id": operation_id}


def decode_shell_interrupt(payload: Mapping[str, Any]) -> str:
    """Rebuild one interrupt's operation id, refusing a frame without one."""
    operation_id = payload.get("operation_id")
    if not isinstance(operation_id, str) or not operation_id:
        raise UnsupportedAgentRequestError("a shell interrupt names no operation")
    return operation_id


def encode_shell_interrupt_reply(interrupted: bool) -> dict[str, Any]:
    """Encode whether the named command was still running to be stopped."""
    return {"interrupted": interrupted}


def decode_shell_interrupt_reply(payload: Mapping[str, Any]) -> bool:
    """Rebuild one interrupt's outcome: a command it did not find ran nothing."""
    return bool(payload.get("interrupted"))


def encode_steer_request(params: TurnSteerParams) -> dict[str, Any]:
    """Encode one steering request: its delivery, its target, its replay key."""
    return {
        "session_id": params.session_id,
        "expected_turn_id": params.expected_turn_id,
        "message": [_encode_value(block) for block in params.message],
        "require_waiting_only": params.require_waiting_only,
        "idempotency_key": params.idempotency_key,
        "client_user_message_id": params.client_user_message_id,
        "inject_invoked_skill": params.inject_invoked_skill,
        "mention_stats": _encode_optional(params.mention_stats),
    }


def decode_steer_request(payload: Mapping[str, Any]) -> TurnSteerParams:
    """Rebuild one steering request, refusing a frame without its target turn."""
    expected_turn_id = payload.get("expected_turn_id")
    if not isinstance(expected_turn_id, str) or not expected_turn_id:
        raise UnsupportedAgentRequestError("a steering request names no turn")
    session_id = payload.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise UnsupportedAgentRequestError("a steering request names no session")
    stats = payload.get("mention_stats")
    return TurnSteerParams(
        session_id=session_id,
        expected_turn_id=expected_turn_id,
        message=_CONTENT_BLOCKS_ADAPTER.validate_python(payload.get("message") or []),
        require_waiting_only=bool(payload.get("require_waiting_only")),
        idempotency_key=_optional_str(payload.get("idempotency_key")),
        client_user_message_id=_optional_str(payload.get("client_user_message_id")),
        inject_invoked_skill=bool(payload.get("inject_invoked_skill", True)),
        mention_stats=(
            MentionStats.model_validate(stats) if isinstance(stats, Mapping) else None
        ),
    )


def encode_steer_settlement(settlement: SteerSettlement) -> dict[str, Any]:
    """Encode the agent-side settlement receipt of one steering delivery."""
    return {
        "accepted": True,
        "committed": settlement.committed,
        "events": settlement.events,
    }


def decode_steer_settlement(payload: Mapping[str, Any]) -> SteerSettlement:
    """Rebuild one steering settlement receipt."""
    events = payload.get("events")
    return SteerSettlement(
        committed=bool(payload.get("committed")),
        events=events if isinstance(events, int) else 0,
    )


def encode_inject_request(
    params: ContextInjectParams, manual_shell: ManualShellContext | None
) -> dict[str, Any]:
    """Encode one injection request, with the manual-shell record it carries."""
    return {
        "session_id": params.session_id,
        "input": [_encode_value(block) for block in params.input],
        "as_message": params.as_message,
        "inject_invoked_skill": params.inject_invoked_skill,
        "client_user_message_id": params.client_user_message_id,
        "mention_stats": _encode_optional(params.mention_stats),
        "manual_shell": (
            encode_manual_shell_context(manual_shell)
            if manual_shell is not None
            else None
        ),
    }


def decode_inject_request(
    payload: Mapping[str, Any],
) -> tuple[ContextInjectParams, ManualShellContext | None]:
    """Rebuild one injection request and the manual-shell record beside it."""
    input_blocks = payload.get("input")
    if not isinstance(input_blocks, list) or not input_blocks:
        raise UnsupportedAgentRequestError("an inject frame carries no input")
    session_id = payload.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise UnsupportedAgentRequestError("an inject frame names no session")
    manual_shell = payload.get("manual_shell")
    stats = payload.get("mention_stats")
    return (
        ContextInjectParams(
            session_id=session_id,
            input=_CONTENT_BLOCKS_ADAPTER.validate_python(input_blocks),
            as_message=bool(payload.get("as_message")),
            inject_invoked_skill=bool(payload.get("inject_invoked_skill")),
            client_user_message_id=_optional_str(payload.get("client_user_message_id")),
            mention_stats=(
                MentionStats.model_validate(stats)
                if isinstance(stats, Mapping)
                else None
            ),
        ),
        (
            decode_manual_shell_context(manual_shell)
            if isinstance(manual_shell, Mapping)
            else None
        ),
    )


def encode_inject_reply(events: Sequence[BaseEvent]) -> dict[str, Any]:
    """Encode the injected events an injection reply carries."""
    return {"events": [encode_event(event) for event in events]}


def decode_inject_reply(payload: Mapping[str, Any]) -> list[BaseEvent]:
    """Rebuild the injected events of one injection reply."""
    events = payload.get("events")
    if not isinstance(events, list):
        return []
    return [decode_event(_mapping(event)) for event in events]


def encode_notice(event: BaseEvent) -> dict[str, Any]:
    """Encode one turn-less notice: an out-of-band event class only."""
    name = type(event).__name__
    if name not in _NOTICE_EVENT_NAMES:
        raise UnsupportedAgentEventError(name)
    return encode_event(event)


def decode_notice(payload: Mapping[str, Any]) -> BaseEvent:
    """Rebuild one turn-less notice, refusing an event class it cannot carry."""
    event = decode_event(payload)
    name = type(event).__name__
    if name not in _NOTICE_EVENT_NAMES:
        raise UnsupportedAgentEventError(name)
    return event


@dataclass(frozen=True, slots=True)
class AgentOpenRequest:
    """The decoded params of one ``agent/open`` request.

    ``session_id`` and ``continue_latest`` are the op-6 extension: an open that
    names a stored session (or asks for the latest one) is a long-budget
    operation, so it also carries the coordinator's absolute monotonic
    deadline. A fresh open carries neither and keeps the short default.
    """

    agent_config: dict[str, Any]
    client_info: dict[str, Any]
    client_capabilities: dict[str, Any]
    session_id: str | None = None
    continue_latest: bool = False
    deadline: float | None = None

    @property
    def resumes(self) -> bool:
        """Whether this open resumes a stored session instead of opening fresh."""
        return self.session_id is not None or self.continue_latest


def encode_open_request(
    *,
    session_options: BaseModel,
    client_info: BaseModel,
    client_capabilities: BaseModel | None = None,
    session_id: str | None = None,
    continue_latest: bool = False,
    deadline: float | None = None,
) -> dict[str, Any]:
    """Encode the params of one open request."""
    params: dict[str, Any] = {
        "agent_config": session_options.model_dump(mode="json", by_alias=True),
        "client_info": client_info.model_dump(mode="json", by_alias=True),
        "client_capabilities": (
            client_capabilities.model_dump(mode="json", by_alias=True)
            if client_capabilities is not None
            else {}
        ),
    }
    if session_id is not None:
        params["session_id"] = session_id
    if continue_latest:
        params["continue_latest"] = True
    if deadline is not None:
        params["deadline"] = deadline
    return params


def decode_open_request(params: Mapping[str, Any]) -> AgentOpenRequest:
    """Decode one open request; a malformed extension fails typed."""
    session_id = params.get("session_id")
    if session_id is not None and not isinstance(session_id, str):
        raise UnsupportedAgentRequestError("the open request names no session id")
    continue_latest = params.get("continue_latest", False)
    if not isinstance(continue_latest, bool):
        raise UnsupportedAgentRequestError(
            "the open request's continue_latest is not a boolean"
        )
    deadline = params.get("deadline")
    if deadline is not None and not isinstance(deadline, int | float):
        raise UnsupportedAgentRequestError(
            "the open request carries no absolute deadline"
        )
    return AgentOpenRequest(
        agent_config=dict(_mapping(params.get("agent_config"))),
        client_info=dict(_mapping(params.get("client_info"))),
        client_capabilities=dict(_mapping(params.get("client_capabilities"))),
        session_id=session_id,
        continue_latest=continue_latest,
        deadline=float(deadline) if deadline is not None else None,
    )


def encode_open_reply(reply: AgentOpenReply) -> dict[str, Any]:
    """Encode the session identity, directories and accounting one open adopted."""
    return {
        "session_id": reply.session_id,
        "session_dir": reply.session_dir,
        "jobs_dir": reply.jobs_dir,
        "accounting": encode_open_accounting(reply.accounting),
    }


def decode_open_reply(payload: Mapping[str, Any]) -> AgentOpenReply:
    """Rebuild one open reply; a malformed reply fails typed."""
    session_id = _require_str(payload.get("session_id"), "session id")
    return AgentOpenReply(
        session_id=session_id,
        session_dir=_optional_str(payload.get("session_dir")),
        jobs_dir=_optional_str(payload.get("jobs_dir")),
        accounting=decode_open_accounting(_mapping(payload.get("accounting"))),
    )


def encode_open_accounting(accounting: RootOpenAccounting) -> dict[str, Any]:
    """Encode the accounting identity and replay records of one open."""
    return {
        "root_session_id": accounting.root_session_id,
        "project_key": accounting.project_key,
        "adoption": accounting.adoption,
        "settlements": [
            {
                "record": record.model_dump(mode="json"),
                "result": {
                    "disposition": str(result.disposition),
                    "warning": (
                        result.warning.model_dump(mode="json")
                        if result.warning is not None
                        else None
                    ),
                },
            }
            for record, result in accounting.settlements
        ],
        "warnings": [
            warning.model_dump(mode="json") for warning in accounting.warnings
        ],
    }


def decode_open_accounting(payload: Mapping[str, Any]) -> RootOpenAccounting:
    """Rebuild the accounting records one open reply carries."""
    return RootOpenAccounting(
        root_session_id=_optional_str(payload.get("root_session_id")),
        project_key=_optional_str(payload.get("project_key")),
        adoption=_optional_str(payload.get("adoption")) or "none",
        settlements=tuple(
            (
                UsageRecord.model_validate(_mapping(entry.get("record"))),
                _decode_usage_write_result(_mapping(entry.get("result"))),
            )
            for entry in _sequence(payload.get("settlements"))
        ),
        warnings=tuple(
            CoverageWarning.model_validate(warning)
            for warning in _sequence(payload.get("warnings"))
        ),
    )


def encode_operation_marker(marker: OperationMarker) -> dict[str, Any]:
    """Encode one long-operation marker record for a snapshot or admission frame."""
    return {
        "request_id": marker.request_id,
        "op_id": marker.op_id,
        "method": marker.method,
        "state": marker.state,
        "deadline": marker.deadline,
        "outcome": marker.outcome,
    }


def decode_operation_marker(payload: Mapping[str, Any]) -> OperationMarker:
    """Rebuild one long-operation marker record."""
    return OperationMarker(
        request_id=_require_str(payload.get("request_id"), "marker request id"),
        op_id=_require_str(payload.get("op_id"), "marker op id"),
        method=_require_str(payload.get("method"), "marker method"),
        state=_require_str(payload.get("state"), "marker state"),
        deadline=_require_float(payload.get("deadline"), "marker deadline"),
        outcome=_optional_str(payload.get("outcome")),
    )


def encode_availability(snapshot: AvailabilitySnapshot) -> dict[str, Any]:
    """Encode the non-mutating eligibility decision a snapshot carries."""
    return {
        "available": snapshot.available,
        "permanent": snapshot.permanent,
        "endpoint_blocked": snapshot.endpoint_blocked,
        "probe_in_flight": snapshot.probe_in_flight,
        "next_check_delay": snapshot.next_check_delay,
        "exclusions": [
            {
                "base_model": exclusion.base_model,
                "provider": exclusion.provider,
                "wire_name": exclusion.wire_name,
                "reason": str(exclusion.reason),
                "detail": exclusion.detail,
            }
            for exclusion in snapshot.exclusions
        ],
    }


def decode_availability(payload: Mapping[str, Any]) -> AvailabilitySnapshot:
    """Rebuild the eligibility decision view of one snapshot payload."""
    return AvailabilitySnapshot(
        available=bool(payload.get("available")),
        permanent=bool(payload.get("permanent")),
        endpoint_blocked=bool(payload.get("endpoint_blocked")),
        probe_in_flight=bool(payload.get("probe_in_flight")),
        next_check_delay=_require_float(
            payload.get("next_check_delay"), "availability recheck delay"
        ),
        exclusions=tuple(
            DeploymentExclusion(
                base_model=_require_str(entry.get("base_model"), "excluded base model"),
                provider=_require_str(entry.get("provider"), "excluded provider"),
                wire_name=_require_str(entry.get("wire_name"), "excluded wire name"),
                reason=ExclusionReason(_require_str(entry.get("reason"), "exclusion")),
                detail=_optional_str(entry.get("detail")),
            )
            for entry in _sequence(payload.get("exclusions"))
            if isinstance(entry, Mapping)
        ),
    )


def encode_session_snapshot(snapshot: AgentSessionSnapshot) -> dict[str, Any]:
    """Encode one reply-write-time session snapshot payload."""
    return {
        "constructed_at": snapshot.constructed_at,
        "incarnation_id": snapshot.incarnation_id,
        "state": snapshot.state.model_dump(mode="json"),
        "availability": encode_availability(snapshot.availability),
        "committed_model": _encode_optional(snapshot.committed_model),
        "markers": [encode_operation_marker(marker) for marker in snapshot.markers],
        "log": snapshot.log.model_dump(mode="json"),
        "runtime": snapshot.runtime.model_dump(mode="json"),
        "session_dir": snapshot.session_dir,
        "jobs_dir": snapshot.jobs_dir,
    }


def decode_session_snapshot(payload: Mapping[str, Any]) -> AgentSessionSnapshot:
    """Rebuild one session snapshot payload; a malformed payload fails typed."""
    return AgentSessionSnapshot(
        constructed_at=_require_float(
            payload.get("constructed_at"), "snapshot construction time"
        ),
        incarnation_id=_require_str(payload.get("incarnation_id"), "incarnation id"),
        state=PublicSessionState.model_validate(_mapping(payload.get("state"))),
        availability=decode_availability(_mapping(payload.get("availability"))),
        committed_model=(
            CommittedModelIdentity.model_validate(payload["committed_model"])
            if payload.get("committed_model") is not None
            else None
        ),
        markers=tuple(
            decode_operation_marker(entry)
            for entry in _sequence(payload.get("markers"))
            if isinstance(entry, Mapping)
        ),
        log=SessionLogSummary.model_validate(_mapping(payload.get("log"))),
        runtime=RuntimeSnapshot.model_validate(_mapping(payload.get("runtime"))),
        session_dir=_optional_str(payload.get("session_dir")),
        jobs_dir=_optional_str(payload.get("jobs_dir")),
    )


@dataclass(frozen=True, slots=True)
class CompactRequest:
    """The decoded params of one ``agent/compact`` request.

    Compaction always runs as a long operation, so the request carries the
    coordinator's absolute monotonic deadline; a request without one is
    malformed rather than a short op.
    """

    extra_instructions: str = ""
    deadline: float | None = None


@dataclass(frozen=True, slots=True)
class CompactSettlementReply:
    """One compaction's reply: the settlement receipt and the handoff state.

    ``settlement`` is one of the three agent-minted publication classes. A
    published compaction carries its summary and the state that follows it; an
    unsettled one still carries whatever reconciliation state could be read,
    and ``detail`` says what went wrong. ``session_id`` and ``session_dir`` are
    the identity the handoff leaves behind -- compaction keeps the session, so
    they are the session's own, published for the coordinator's accounting.
    """

    settlement: str
    session_id: str | None = None
    session_dir: str | None = None
    summary: str | None = None
    state: PublicSessionState | None = None
    session_log: SessionLogSummary | None = None
    detail: str = ""

    @property
    def published(self) -> bool:
        """Whether this compaction published a boundary."""
        return self.settlement != COMPACT_SETTLEMENT_CANCELLED_UNPUBLISHED

    @property
    def settled(self) -> bool:
        """Whether the published state was also settled (published iff so)."""
        return self.settlement == COMPACT_SETTLEMENT_PUBLISHED_SETTLED


def encode_compact_request(
    extra_instructions: str, *, deadline: float
) -> dict[str, Any]:
    """Encode the params of one compaction request."""
    return {"extra_instructions": extra_instructions, "deadline": deadline}


def decode_compact_request(params: Mapping[str, Any]) -> CompactRequest:
    """Decode one compaction request; a malformed request fails typed."""
    instructions = params.get("extra_instructions", "")
    if not isinstance(instructions, str):
        raise UnsupportedAgentRequestError(
            "the compaction request's instructions are not text"
        )
    deadline = params.get("deadline")
    if deadline is not None and (
        not isinstance(deadline, int | float) or isinstance(deadline, bool)
    ):
        raise UnsupportedAgentRequestError(
            "the compaction request carries no absolute deadline"
        )
    return CompactRequest(
        extra_instructions=instructions,
        deadline=float(deadline) if deadline is not None else None,
    )


def encode_compact_reply(reply: CompactSettlementReply) -> dict[str, Any]:
    """Encode one compaction reply: its receipt plus the identity handoff."""
    return {
        "settlement": reply.settlement,
        "session_id": reply.session_id,
        "session_dir": reply.session_dir,
        "summary": reply.summary,
        "state": (
            reply.state.model_dump(mode="json") if reply.state is not None else None
        ),
        "session_log": (
            reply.session_log.model_dump(mode="json")
            if reply.session_log is not None
            else None
        ),
        "detail": reply.detail,
    }


def decode_compact_reply(payload: Mapping[str, Any]) -> CompactSettlementReply:
    """Rebuild one compaction reply, refusing a class the agent never mints."""
    settlement = _require_str(payload.get("settlement"), "compaction settlement")
    if settlement not in COMPACT_AGENT_SETTLEMENTS:
        raise UnsupportedAgentRequestError(
            f"compaction settlement {settlement!r} is not an agent-minted class"
        )
    state = payload.get("state")
    log = payload.get("session_log")
    return CompactSettlementReply(
        settlement=settlement,
        session_id=_optional_str(payload.get("session_id")),
        session_dir=_optional_str(payload.get("session_dir")),
        summary=_optional_str(payload.get("summary")),
        state=(
            PublicSessionState.model_validate(state)
            if isinstance(state, Mapping)
            else None
        ),
        session_log=(
            SessionLogSummary.model_validate(log) if isinstance(log, Mapping) else None
        ),
        detail=str(payload.get("detail") or ""),
    )


def encode_op_cancel(op_id: str) -> dict[str, Any]:
    """Encode the params of one id-less long-operation cancel."""
    return {"op_id": op_id}


@dataclass(frozen=True, slots=True)
class ReplaceRequest:
    """The decoded params of one ``agent/session/replace`` request.

    A replace rebinds the session the agent owns to a stored one, which loads a
    transcript and builds an MCP registry, so it is always a long operation: a
    request without the coordinator's absolute deadline is malformed rather
    than a short op.
    """

    session_id: str | None = None
    continue_latest: bool = False
    deadline: float | None = None


@dataclass(frozen=True, slots=True)
class ReplaceReply:
    """One identity replacement's reply: the fresh snapshot and its accounting.

    ``accounting`` is absent on the re-check path: a lost reply is settled from
    the operation's own seeded marker, and the snapshot the re-check read is the
    receipt.
    """

    snapshot: AgentSessionSnapshot
    accounting: RootOpenAccounting | None = None


@dataclass(frozen=True, slots=True)
class RenameReply:
    """One manual title application, as the agent process persisted it."""

    title: str
    updated_at: str | None = None


def encode_replace_request(
    *, session_id: str | None, continue_latest: bool, deadline: float
) -> dict[str, Any]:
    """Encode the params of one session replacement."""
    params: dict[str, Any] = {"deadline": deadline}
    if session_id is not None:
        params["session_id"] = session_id
    if continue_latest:
        params["continue_latest"] = True
    return params


def decode_replace_request(params: Mapping[str, Any]) -> ReplaceRequest:
    """Decode one replacement request; a malformed one fails typed."""
    session_id = params.get("session_id")
    if session_id is not None and (not isinstance(session_id, str) or not session_id):
        raise UnsupportedAgentRequestError("the replace request names no session id")
    continue_latest = params.get("continue_latest", False)
    if not isinstance(continue_latest, bool):
        raise UnsupportedAgentRequestError(
            "the replace request's continue_latest is not a boolean"
        )
    if session_id is None and not continue_latest:
        raise UnsupportedAgentRequestError(
            "the replace request names neither a session id nor continue_latest"
        )
    deadline = params.get("deadline")
    if deadline is not None and (
        not isinstance(deadline, int | float) or isinstance(deadline, bool)
    ):
        raise UnsupportedAgentRequestError(
            "the replace request carries no absolute deadline"
        )
    return ReplaceRequest(
        session_id=session_id,
        continue_latest=continue_latest,
        deadline=float(deadline) if deadline is not None else None,
    )


def encode_replace_reply(reply: ReplaceReply) -> dict[str, Any]:
    """Encode one replacement reply: the fresh identity and what it adopted."""
    return {
        "snapshot": encode_session_snapshot(reply.snapshot),
        "accounting": (
            encode_open_accounting(reply.accounting)
            if reply.accounting is not None
            else None
        ),
    }


def decode_replace_reply(payload: Mapping[str, Any]) -> ReplaceReply:
    """Rebuild one replacement reply; a malformed one fails typed."""
    accounting = payload.get("accounting")
    return ReplaceReply(
        snapshot=decode_session_snapshot(_mapping(payload.get("snapshot"))),
        accounting=(
            decode_open_accounting(accounting)
            if isinstance(accounting, Mapping)
            else None
        ),
    )


def encode_rename_request(title: str) -> dict[str, Any]:
    """Encode the params of one manual rename."""
    return {"title": title}


def decode_rename_request(params: Mapping[str, Any]) -> str:
    """Decode one rename request; a request without a title fails typed."""
    title = params.get("title")
    if not isinstance(title, str):
        raise UnsupportedAgentRequestError("the rename request carries no title")
    return title


def encode_rename_reply(reply: RenameReply) -> dict[str, Any]:
    """Encode one rename reply."""
    return {"title": reply.title, "updated_at": reply.updated_at}


def decode_rename_reply(payload: Mapping[str, Any]) -> RenameReply:
    """Rebuild one rename reply; a reply without a title fails typed."""
    return RenameReply(
        title=_require_str(payload.get("title"), "session title"),
        updated_at=_optional_str(payload.get("updated_at")),
    )


@dataclass(frozen=True, slots=True)
class ReconfigureRequest:
    """The decoded params of one ``agent/session/reconfigure`` request.

    ``mode`` decides which members are meaningful: a settings-update carries the
    bounds to apply, a history clear carries nothing but the session it clears,
    a plan handoff carries the consent snapshot exactly as the in-process
    request does, and a config write carries the config-write candidate. Its
    ``reload_runtime`` flag is dropped on this path: the mode IS the reload leg,
    triggered by its own request rather than by reading a flag. ``deadline``
    rides the long modes only -- a short mode has no absolute deadline to honour.
    """

    mode: str
    session_id: str
    deadline: float | None = None
    max_turns: int | None = None
    max_tokens: int | None = None
    handoff: SessionPlanHandoffParams | None = None
    config_write: ConfigWriteParams | None = None


@dataclass(frozen=True, slots=True)
class ReconfigureReply:
    """One reconfigure mode's answer.

    ``outcome`` is the class the agent minted for the mode. ``stage`` names the
    steps this call completed, in order, so a retry after a partial failure can
    see exactly what landed and what it finished itself. ``followup`` names the
    outcome of the post-response step that ran inside the operation boundary --
    over the wire there is no deferred-action carrier, so it executes before
    the reply and its failure rides the receipt. ``snapshot`` is the state the
    reply leaves behind, built at reply-write time like every other snapshot.

    A config-write reply's ``config_write`` is the SHIPPED transaction result
    carried wholesale, and its ``outcome`` names that result's own runtime
    application dimension: the persistence x application composition is never
    collapsed into a smaller receipt.
    """

    mode: str
    outcome: str
    stage: str = ""
    detail: str = ""
    followup: str = ""
    max_turns: int | None = None
    max_tokens: int | None = None
    snapshot: AgentSessionSnapshot | None = None
    accounting: RootOpenAccounting | None = None
    episode_id: str | None = None
    attempt_generation: int | None = None
    config_write: ConfigWriteResponse | None = None


def encode_reconfigure_request(
    *,
    mode: str,
    session_id: str | None = None,
    deadline: float | None = None,
    max_turns: int | None = None,
    max_tokens: int | None = None,
    handoff: SessionPlanHandoffParams | None = None,
    config_write: ConfigWriteParams | None = None,
) -> dict[str, Any]:
    """Encode the params of one reconfigure request.

    A plan handoff carries its own session identity inside the consent snapshot,
    so the request carries no second copy of it. A config-write candidate
    crosses without its ``reload_runtime`` flag: this mode is the reload leg.
    """
    params: dict[str, Any] = {"mode": mode}
    if session_id is not None:
        params["session_id"] = session_id
    if deadline is not None:
        params["deadline"] = deadline
    if max_turns is not None:
        params["max_turns"] = max_turns
    if max_tokens is not None:
        params["max_tokens"] = max_tokens
    if handoff is not None:
        params["handoff"] = handoff.model_dump(mode="json")
    if config_write is not None:
        candidate = config_write.model_dump(mode="json", by_alias=True)
        candidate.pop("reload_runtime", None)
        params["config_write"] = candidate
    return params


def decode_reconfigure_request(params: Mapping[str, Any]) -> ReconfigureRequest:
    """Decode one reconfigure request; a malformed one fails typed."""
    mode = params.get("mode")
    if mode not in RECONFIGURE_MODES:
        raise UnsupportedAgentRequestError(
            f"the reconfigure request names no mode this wire carries: {mode!r}"
        )
    deadline = params.get("deadline")
    if deadline is not None and (
        not isinstance(deadline, int | float) or isinstance(deadline, bool)
    ):
        raise UnsupportedAgentRequestError(
            "the reconfigure request carries no absolute deadline"
        )
    if mode == RECONFIGURE_MODE_PLAN_HANDOFF:
        # The consent snapshot is the request: its session id, the executor
        # selection, and the verifiable document body travel together.
        raw = params.get("handoff")
        if not isinstance(raw, Mapping):
            raise UnsupportedAgentRequestError(
                "the plan-handoff reconfigure carries no consent snapshot"
            )
        try:
            handoff = SessionPlanHandoffParams.model_validate(raw)
        except ValidationError as exc:
            raise UnsupportedAgentRequestError(
                "the plan-handoff reconfigure carries an unusable consent "
                f"snapshot: {exc}"
            ) from exc
        if deadline is None:
            raise UnsupportedAgentRequestError(
                "a plan handoff must carry the coordinator's absolute deadline"
            )
        return ReconfigureRequest(
            mode=mode,
            session_id=handoff.session_id,
            deadline=float(deadline),
            handoff=handoff,
        )
    if mode == RECONFIGURE_MODE_CONFIG_WRITE:
        # The candidate IS the request: the ops, their target and the revision
        # they were read against travel together, and the deadline is what
        # bounds the transaction agent-side. ``reload_runtime`` is not read.
        raw = params.get("config_write")
        if not isinstance(raw, Mapping):
            raise UnsupportedAgentRequestError(
                "the config-write reconfigure carries no config-write candidate"
            )
        try:
            candidate = ConfigWriteParams.model_validate(raw)
        except ValidationError as exc:
            raise UnsupportedAgentRequestError(
                f"the config-write reconfigure carries an unusable candidate: {exc}"
            ) from exc
        if deadline is None:
            raise UnsupportedAgentRequestError(
                "a config-write reconfigure must carry the coordinator's "
                "absolute deadline"
            )
        if candidate.session_id != _require_str(
            params.get("session_id"), "reconfigure session id"
        ):
            raise UnsupportedAgentRequestError(
                "the config-write candidate names a different session than the "
                "reconfigure request"
            )
        return ReconfigureRequest(
            mode=mode,
            session_id=candidate.session_id,
            deadline=float(deadline),
            config_write=candidate,
        )
    session_id = _require_str(params.get("session_id"), "reconfigure session id")
    max_turns = _optional_int(params.get("max_turns"), "max turns")
    max_tokens = _optional_int(params.get("max_tokens"), "max tokens")
    if mode == RECONFIGURE_MODE_SETTINGS and max_turns is None and max_tokens is None:
        raise UnsupportedAgentRequestError(
            "the settings-update reconfigure carries no update"
        )
    return ReconfigureRequest(
        mode=mode,
        session_id=session_id,
        deadline=float(deadline) if deadline is not None else None,
        max_turns=max_turns,
        max_tokens=max_tokens,
    )


def encode_reconfigure_reply(reply: ReconfigureReply) -> dict[str, Any]:
    """Encode one reconfigure reply: its class, its stages and its state."""
    return {
        "mode": reply.mode,
        "outcome": reply.outcome,
        "stage": reply.stage,
        "detail": reply.detail,
        "followup": reply.followup,
        "max_turns": reply.max_turns,
        "max_tokens": reply.max_tokens,
        "snapshot": (
            encode_session_snapshot(reply.snapshot)
            if reply.snapshot is not None
            else None
        ),
        "accounting": (
            encode_open_accounting(reply.accounting)
            if reply.accounting is not None
            else None
        ),
        "episode_id": reply.episode_id,
        "attempt_generation": reply.attempt_generation,
        "config_write": (
            reply.config_write.model_dump(mode="json", by_alias=True)
            if reply.config_write is not None
            else None
        ),
    }


def decode_reconfigure_reply(payload: Mapping[str, Any]) -> ReconfigureReply:
    """Rebuild one reconfigure reply, refusing a class the agent never mints.

    A config-write reply is judged by the SHIPPED dimensions it carries: it must
    carry the transaction result, and its outcome must name that result's own
    runtime application dimension rather than one of the reconfigure classes.
    """
    mode = _require_str(payload.get("mode"), "reconfigure mode")
    if mode not in RECONFIGURE_MODES:
        raise UnsupportedAgentRequestError(
            f"the reply names no reconfigure mode this wire carries: {mode!r}"
        )
    outcome = _require_str(payload.get("outcome"), "reconfigure outcome")
    config_write = _decode_config_write_result(payload)
    if mode == RECONFIGURE_MODE_CONFIG_WRITE:
        if config_write is None:
            raise UnsupportedAgentRequestError(
                "the config-write reply carries no transaction result"
            )
        if outcome != config_write.application:
            raise UnsupportedAgentRequestError(
                f"config-write outcome {outcome!r} is not the transaction's "
                f"application result {config_write.application!r}"
            )
    elif outcome not in RECONFIGURE_OUTCOMES:
        raise UnsupportedAgentRequestError(
            f"reconfigure outcome {outcome!r} is not an agent-minted class"
        )
    snapshot = payload.get("snapshot")
    accounting = payload.get("accounting")
    return ReconfigureReply(
        mode=mode,
        outcome=outcome,
        stage=str(payload.get("stage") or ""),
        detail=str(payload.get("detail") or ""),
        followup=str(payload.get("followup") or ""),
        max_turns=_optional_int(payload.get("max_turns"), "max turns"),
        max_tokens=_optional_int(payload.get("max_tokens"), "max tokens"),
        snapshot=(
            decode_session_snapshot(_mapping(snapshot))
            if isinstance(snapshot, Mapping)
            else None
        ),
        accounting=(
            decode_open_accounting(accounting)
            if isinstance(accounting, Mapping)
            else None
        ),
        episode_id=_optional_str(payload.get("episode_id")),
        attempt_generation=_optional_int(
            payload.get("attempt_generation"), "attempt generation"
        ),
        config_write=config_write,
    )


def _decode_config_write_result(
    payload: Mapping[str, Any],
) -> ConfigWriteResponse | None:
    """The shipped transaction result one reply carries, or ``None``."""
    raw = payload.get("config_write")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise UnsupportedAgentRequestError(
            "the config-write reply carries no transaction result"
        )
    try:
        return ConfigWriteResponse.model_validate(raw)
    except ValidationError as exc:
        raise UnsupportedAgentRequestError(
            f"the config-write reply carries an unusable transaction result: {exc}"
        ) from exc


def _decode_usage_write_result(payload: Mapping[str, Any]) -> UsageWriteResult:
    warning = payload.get("warning")
    return UsageWriteResult(
        disposition=UsageWriteDisposition(
            _require_str(payload.get("disposition"), "usage write disposition")
        ),
        warning=(
            CoverageWarning.model_validate(warning)
            if isinstance(warning, Mapping)
            else None
        ),
    )


def _sequence(value: object) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _require_str(value: object, what: str) -> str:
    if not isinstance(value, str) or not value:
        raise UnsupportedAgentRequestError(f"the frame carries no {what}")
    return value


def _require_float(value: object, what: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise UnsupportedAgentRequestError(f"the frame carries no {what}")
    return float(value)


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _optional_int(value: object, what: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise UnsupportedAgentRequestError(f"the frame carries no {what}")
    return value


def _encode_optional(value: BaseModel | None) -> JsonValue:
    return _encode_value(value) if value is not None else None


def _encode_value(value: BaseModel | JsonValue) -> JsonValue:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", by_alias=True)
    return value


def _decode_presentation[PresentationT: BaseModel](
    presentation_type: type[PresentationT], payload: object
) -> PresentationT | None:
    if payload is None:
        return None
    return presentation_type.model_validate(payload)
