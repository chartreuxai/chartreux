"""Coordinator-side handle for a root agent process.

The handle spawns the root agent through the WP1 bootstrap handshake and
serves the agent-side :class:`TurnExecutionPort` across the process boundary:
one turn at a time in the agent's production order, user-input answers routed
back by request id, retry notices forwarded to the host's retry sink, and
live transcript pages routed through the session registry manifest exactly as
a child agent's are.

Selection is experimental. The multi-process route is opt-in by the
experimental transport selector, so a run that selects ``in_process`` — or
selects nothing at all — never reaches this module and keeps one loop in one
process.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping, Sequence
from contextlib import aclosing, suppress
from dataclasses import dataclass, replace
from enum import StrEnum
import itertools
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, NoReturn, cast
from uuid import uuid4

from pydantic import BaseModel

from chartreux.app_server.agent_admission import OP_OUTCOME_COMPLETED
from chartreux.app_server.agent_snapshot import (
    MARKER_ABSENT,
    MARKER_ACTIVE,
    MARKER_TERMINAL,
    AgentOpenReply,
    AgentSessionSnapshot,
    MarkerVerdict,
    RootOpenAccounting,
    marker_verdict,
)
from chartreux.app_server.agent_wire import (
    AGENT_CHILD_TEARDOWN_METHOD,
    AGENT_COMPACT_METHOD,
    AGENT_INJECT_METHOD,
    AGENT_NOTICE_METHOD,
    AGENT_OP_ADMISSION_METHOD,
    AGENT_OP_CANCEL_METHOD,
    AGENT_OPEN_METHOD,
    AGENT_ORIGIN,
    AGENT_SESSION_CONFIG_RELOAD_METHOD,
    AGENT_SESSION_LIST_METHOD,
    AGENT_SESSION_NOT_FOUND_ERROR,
    AGENT_SESSION_PLAN_RECORD_READ_METHOD,
    AGENT_SESSION_PLAN_RECORD_WRITE_METHOD,
    AGENT_SESSION_POLICY_READ_METHOD,
    AGENT_SESSION_RECONFIGURE_METHOD,
    AGENT_SESSION_RENAME_METHOD,
    AGENT_SESSION_REPLACE_METHOD,
    AGENT_SESSION_ROOTS_READ_METHOD,
    AGENT_SESSION_ROOTS_SAVE_METHOD,
    AGENT_SESSION_SNAPSHOT_METHOD,
    AGENT_SHELL_INTERRUPT_METHOD,
    AGENT_SHELL_RUN_METHOD,
    AGENT_TRANSCRIPT_PAGE_METHOD,
    AGENT_TURN_ANSWER_METHOD,
    AGENT_TURN_CALLBACK_METHOD,
    AGENT_TURN_CANCEL_METHOD,
    AGENT_TURN_ENDED_METHOD,
    AGENT_TURN_EVENT_METHOD,
    AGENT_TURN_INTERRUPT_METHOD,
    AGENT_TURN_KEEP_WAITING_METHOD,
    AGENT_TURN_REFUSE_METHOD,
    AGENT_TURN_REJECT_METHOD,
    AGENT_TURN_RELAY_METHOD,
    AGENT_TURN_RESOLVE_METHOD,
    AGENT_TURN_RETRYING_METHOD,
    AGENT_TURN_START_METHOD,
    AGENT_TURN_STEER_METHOD,
    COMPACT_AGENT_SETTLEMENTS,
    COMPACT_SETTLEMENT_CANCELLED_UNPUBLISHED,
    COMPACT_SETTLEMENT_PUBLISHED_SETTLED,
    COMPACT_SETTLEMENT_PUBLISHED_UNSETTLED,
    COORDINATOR_RECEIVING_NOTIFICATIONS,
    COORDINATOR_RECEIVING_REQUESTS,
    HOST_SUBAGENT_ENDED_METHOD,
    HOST_SUBAGENT_EVENT_METHOD,
    HOST_TOOL_IO_METHOD,
    RECONFIGURE_CONFLICT_ERROR,
    RECONFIGURE_MODE_CONFIG_WRITE,
    RECONFIGURE_MODE_HISTORY_CLEAR,
    RECONFIGURE_MODE_PLAN_HANDOFF,
    RECONFIGURE_MODE_SETTINGS,
    RECONFIGURE_OUTCOME_ACCEPTED,
    TURN_OUTCOME_COMPLETED,
    TURN_OUTCOME_FAILED,
    TURN_OUTCOME_INTERRUPTED,
    AgentWireError,
    CompactSettlementReply,
    ReconfigureReply,
    RenameReply,
    ReplaceReply,
    ShellRunRequest,
    SteerSettlement,
    WireDiagnostic,
    WireMethodNotSupportedError,
    coordinator_request_id,
    decode_compact_reply,
    decode_event,
    decode_inject_reply,
    decode_notice,
    decode_open_reply,
    decode_operation_marker,
    decode_reconfigure_reply,
    decode_rename_reply,
    decode_replace_reply,
    decode_retry_reason,
    decode_session_snapshot,
    decode_shell_command_request,
    decode_shell_interrupt_reply,
    decode_shell_run_reply,
    decode_steer_settlement,
    decode_subagent_run_request,
    decode_tool_io_request,
    decode_transcript_page,
    decode_user_input_settlement,
    deferred_session_message,
    encode_compact_request,
    encode_event,
    encode_host_run_end,
    encode_host_run_frame,
    encode_host_service_offers,
    encode_inject_request,
    encode_op_cancel,
    encode_open_request,
    encode_reconfigure_request,
    encode_rename_request,
    encode_replace_request,
    encode_shell_interrupt,
    encode_shell_run_request,
    encode_steer_request,
    encode_tool_io_result,
    encode_turn_request,
    encode_user_input_result,
    request_id_origin,
)
from chartreux.app_server.models import SessionLogSummary
from chartreux.app_server.process_bootstrap import (
    DEFAULT_SPAWN_READY_TIMEOUT,
    ProcessTreeBudget,
    SpawnedProcess,
    spawn_process,
)
from chartreux.app_server.protocol import (
    AgentTranscriptGetResponse,
    ConfigMutationResponse,
    ConfigReloadParams,
    ConfigWriteParams,
    ContextInjectParams,
    PolicyReadParams,
    PolicyReadResponse,
    RootsReadParams,
    RootsReadResponse,
    RootsSaveParams,
    RootsSaveResponse,
    SessionListParams,
    SessionListResponse,
    SessionPlanHandoffParams,
    SessionPlanRecord,
    SessionPlanRecordWriteParams,
    SessionPlanRecordWriteResponse,
    TurnSteerParams,
)
from chartreux.app_server.registry_manifest import AgentRoute, RegistryManifest
from chartreux.app_server.root_agent import ROOT_AGENT_MODULE
from chartreux.app_server.subagent_ports import (
    ChildTeardownReport,
    ChildTeardownRequest,
)
from chartreux.app_server.transport import (
    MULTI_PROCESS_TRANSPORT_SELECTORS,
    TRANSPORT_ENV_VAR,
)
from chartreux.core.agent_loop._execution_port import (
    ExecutionTurnRequest,
    TurnOrderingConflict,
)
from chartreux.core.agent_loop._loop_types import AgentTurnOptions
from chartreux.core.agent_loop._request_broker import (
    KeepWaitingOutcome,
    UserInputSettlement,
    UserInputSettleOutcome,
)
from chartreux.core.events import BaseEvent, UserInputRequestEvent
from chartreux.core.llm_models import ManualShellContext
from chartreux.core.subagents import SubagentRunnerPort, TaskResult
from chartreux.core.tools.base import InvokeContext
from chartreux.core.tools.io_port import ToolIOPort
from chartreux.core.utils.retry import RetryObserver
from chartreux.core.utils.shell import publish_shell_executor
from chartreux.observability.logging import logger

DEFAULT_REQUEST_TIMEOUT = 30.0

SHELL_REQUEST_MARGIN = 30.0
"""The settling margin a manual ``!`` request adds over the command's deadline."""

DEFAULT_ROOT_HEARTBEAT_INTERVAL = 2.0
"""The root pipe's liveness heartbeat period (plan.md numeric bounds, WP7 [7C])."""

IDLE_WEDGE_TIMEOUT_STREAK = 2
"""Consecutive op timeouts, nothing answered between, that are an idle wedge.

One expired deadline with a live root is a single wedged operation; the same
thing repeating is the pinned idle-wedge residual, and the diagnostic says so
rather than reporting another slow op.
"""

RESUME_OPEN_DEADLINE = 120.0
"""The agent-side absolute deadline of an open that resumes a stored session.

Transcript load and MCP registry construction are the long part of a resume;
a fresh open keeps the short default instead.
"""

RESUME_OPEN_REPLY_GRACE = 30.0
"""How long past the deadline the coordinator waits for the abort's reply.

The agent aborts the operation at the deadline and answers its request id
typed; this grace covers that settlement and the pipe, so the coordinator reads
a terminal answer instead of timing out first.
"""

COMPACT_DEADLINE = 900.0
"""The agent-side absolute deadline of one manual compaction.

It is sized to the in-process budget (the 720s manual-compaction timeout the
handler applies) plus margin: the agent aborts the operation at this same
absolute deadline, so a wedged-but-alive run cannot outlive it.
"""

COMPACT_REPLY_GRACE = 30.0
"""How long past the compaction deadline the coordinator waits for its reply.

Compaction answers its request id with a settlement receipt even when it is
aborted or cancelled, so this grace is what keeps an honest answer ahead of the
coordinator's own timeout. Only a lost reply reaches the one-shot re-check.
"""

COMPACT_OBSERVATION_IN_FLIGHT = "in-flight"
"""Coordinator-minted: admitted, unsettled at the re-check, still to resolve."""

COMPACT_OBSERVATION_SETTLEMENT_UNKNOWN = "settlement-unknown"
"""Coordinator-minted: the re-check could not judge, so nothing is claimed."""

COMPACT_OBSERVATION_SETTLEMENT_VOIDED = "settlement-voided"
"""Coordinator-minted: session failure ended the incarnation a pending op held."""

REPLACE_DEADLINE = 300.0
"""The agent-side absolute deadline of one identity replacement.

It is sized to the transcript load and MCP registry construction the rebind
performs agent-side: the agent aborts the operation at this same absolute
deadline, so a wedged-but-alive rebind cannot outlive it.
"""

REPLACE_REPLY_GRACE = 30.0
"""How long past the replacement deadline the coordinator waits for its reply.

Only a lost reply reaches the one-shot marker re-check; this grace is what lets
an honest abort answer land first.
"""

COMPACT_PUBLISHED_SETTLEMENTS = frozenset({
    COMPACT_SETTLEMENT_PUBLISHED_SETTLED,
    COMPACT_SETTLEMENT_PUBLISHED_UNSETTLED,
})
"""The agent-minted classes that claim a published boundary."""

PLAN_HANDOFF_DEADLINE = 300.0
"""The agent-side absolute deadline of one plan handoff.

It is sized to the full consent transaction the agent runs -- the episode
fence, the durable retirement and marker, the transcript reset, the selection
install and the injection -- and the agent aborts the operation at this same
absolute deadline, so a wedged-but-alive handoff cannot outlive it.
"""

PLAN_HANDOFF_REPLY_GRACE = 30.0
"""How long past the handoff deadline the coordinator waits for its reply.

Only a lost reply reaches the one-shot marker re-check; this grace is what lets
an honest abort answer land first.
"""

CONFIG_WRITE_DEADLINE = 60.0
"""The agent-side absolute deadline of one config-write reconfigure (mode d).

It bounds the prepare/commit-reload legs the agent runs against the candidate
backend; the agent aborts the transaction at this same absolute deadline, so a
wedged-but-alive write cannot outlive it.
"""

CONFIG_WRITE_REPLY_GRACE = 15.0
"""How long past the config-write deadline the coordinator waits for its reply.

Only a lost reply reaches the one-shot marker re-check; this grace is what lets
an honest abort answer land first.
"""


_DEFAULT_CONNECTION_ID = "root-agent"


class RootAgentProcessError(RuntimeError):
    """Base class for failures of the root agent process handle."""


class RootFailureSignature(StrEnum):
    """The four observed failure signatures the diagnostic text distinguishes.

    A user reading a session failure must be able to tell the cases apart, so
    every root failure names the signature this end observed rather than
    leaving four different causes under one sentence.
    """

    #: The liveness watcher saw the process exit inside the poll interval.
    DEATH_RECEIPT = "death-receipt"
    #: The pipe closed while the process was still running.
    PIPE_EOF = "pipe-eof-while-alive"
    #: An in-flight op passed its deadline with a live root.
    OP_DEADLINE = "op-deadline-timeout"
    #: Timeouts repeated with a live watcher and no op in flight.
    IDLE_WEDGE = "repeated-timeout-idle-wedge"


@dataclass(frozen=True, slots=True)
class RootDeath:
    """One observed death of the root agent process, typed."""

    reason: str
    exit_code: int | None = None
    signature: RootFailureSignature = RootFailureSignature.DEATH_RECEIPT

    @property
    def diagnostic(self) -> str:
        """The failure text, naming the signature that was observed."""
        return f"{self.signature.value}: {self.reason}"


class RootAgentExitedError(RootAgentProcessError):
    """The root agent process died: its pipe closed or its process is gone.

    Root death is a typed lifecycle observation, never a transport error: the
    handle stops serving requests and turns, and the coordinator disposes of
    the child tree before anything is respawned or resumed.
    """

    def __init__(
        self,
        reason: str,
        *,
        exit_code: int | None = None,
        signature: RootFailureSignature = RootFailureSignature.DEATH_RECEIPT,
    ) -> None:
        super().__init__(f"{signature.value}: {reason}")
        self.reason = reason
        self.exit_code = exit_code
        self.signature = signature


class RootAgentOpTimeoutError(RootAgentProcessError, TimeoutError):
    """A request passed its deadline while the root was still serving.

    A single wedged op is the op-deadline signature; the same thing repeating
    with a live watcher and no op in flight is the idle-wedge residual, which
    is never a claimed abort. The error stays a ``TimeoutError`` so the
    long-operation settlement paths catch it exactly as they caught the bare
    timeout before it learned to name its signature.
    """

    def __init__(
        self,
        method: str,
        timeout: float,
        *,
        signature: RootFailureSignature,
        occurrences: int,
    ) -> None:
        super().__init__(
            f"{signature.value}: {method} passed its {timeout:g}s deadline "
            f"with the root still serving ({occurrences} consecutive)"
        )
        self.method = method
        self.timeout = timeout
        self.signature = signature
        self.occurrences = occurrences


class RootAgentSelectionError(RootAgentProcessError):
    """The run did not select the experimental multi-process transport."""


class RootAgentRequestError(RootAgentProcessError):
    """The agent answered a request with a typed wire failure."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class AgentProcessTurnFailed(RootAgentProcessError):
    """The agent's turn ended in failure after its events were delivered."""


class UnsupportedProcessTurnRequest(RootAgentProcessError):
    """The turn request carries a host-side port that has no wire form."""


class RootAgentReplaceConflict(RootAgentProcessError):
    """A replacement is refused before anything is sent, typed.

    Raised for the coordinator's own unresolved state (a pending settlement or
    a prior replacement this end has not resolved) and for a lost reply whose
    marker re-check proved nothing: a replacement is never resent on an
    ambiguous outcome.
    """


class RootAgentSessionNotFound(RootAgentProcessError):
    """The stored session a replacement names does not exist."""


class RootAgentReconfigureConflict(RootAgentProcessError):
    """A reconfigure mode is refused before a frame is sent, typed.

    Raised for the coordinator's own unresolved state (a settlement or a long
    operation still in flight) and for a handoff whose lost reply proved
    nothing: a reconfigure is never resent on an ambiguous outcome.
    """


class RootAgentDeferredOperation(RootAgentProcessError):
    """The session operation this release does not carry over the wire.

    The refusal is minted coordinator-side, before a frame would be sent, and
    names the same in_process escape the agent would name.
    """


@dataclass(frozen=True, slots=True)
class HostServicePorts:
    """The coordinator's host ports for the turn in flight.

    The op-9 channel serves exactly the two REQUEST ports a root turn needs:
    the subagent runner (streamed back as notifications) and the client tool
    I/O port (one request, one reply). An empty pair means no turn is in
    flight, so an agent-origin host request has nothing to serve.
    """

    subagent_runner: SubagentRunnerPort | None = None
    tool_io: ToolIOPort | None = None


@dataclass(frozen=True, slots=True)
class AdmittedOperation:
    """The admission one long operation announced, as the coordinator holds it.

    The announcement is an id-less notification carrying the coordinator's own
    request id, so attribution is direct: the coordinator never has to infer
    which request an op id belongs to.
    """

    request_id: str
    op_id: str
    method: str
    deadline: float

    @property
    def incarnation_id(self) -> str:
        """The agent incarnation whose marker table minted this op id."""
        return self.op_id.partition(":")[0]


type _Frame = tuple[str, dict[str, Any]]


@dataclass(frozen=True, slots=True)
class CompactSettlementRecord:
    """What the coordinator holds about one compaction's settlement.

    ``settlement`` is either one of the three agent-minted publication classes
    (delivered by the reply, or read from the op's TERMINAL marker record) or a
    coordinator-minted observation -- a timeout or a failed re-check is the
    coordinator's own observation, never something the agent could mint. While
    ``pending`` the operation may still settle, so every later snapshot payload
    re-checks the marker and resolves it; ``cancel_pending`` is the cancel a
    cancel notification left in flight until that settlement lands.
    """

    request_id: str
    settlement: str
    detail: str = ""
    op_id: str | None = None
    outcome: str | None = None
    reply: CompactSettlementReply | None = None
    pending: bool = False
    cancel_pending: bool = False

    @property
    def published(self) -> bool:
        """Whether the recorded settlement claims a published boundary."""
        return self.settlement in COMPACT_PUBLISHED_SETTLEMENTS

    @property
    def resolved(self) -> bool:
        """Whether this settlement is terminal and needs no further re-check."""
        return not self.pending


class RootAgentProcess:  # noqa: PLR0904 - the coordinator-side port surface
    """A spawned root agent serving the agent-side execution port."""

    def __init__(  # noqa: PLR0913, PLR0915 - one deadline pair per long operation
        self,
        spawned: SpawnedProcess,
        *,
        connection_id: str = _DEFAULT_CONNECTION_ID,
        incarnation_id: str | None = None,
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
        heartbeat_interval: float = DEFAULT_ROOT_HEARTBEAT_INTERVAL,
        compact_deadline: float = COMPACT_DEADLINE,
        compact_reply_grace: float = COMPACT_REPLY_GRACE,
        replace_deadline: float = REPLACE_DEADLINE,
        replace_reply_grace: float = REPLACE_REPLY_GRACE,
        handoff_deadline: float = PLAN_HANDOFF_DEADLINE,
        handoff_reply_grace: float = PLAN_HANDOFF_REPLY_GRACE,
        config_write_deadline: float = CONFIG_WRITE_DEADLINE,
        config_write_reply_grace: float = CONFIG_WRITE_REPLY_GRACE,
    ) -> None:
        if request_timeout <= 0:
            raise ValueError("request_timeout must be positive")
        if heartbeat_interval <= 0:
            raise ValueError("heartbeat_interval must be positive")
        if compact_deadline <= 0:
            raise ValueError("compact_deadline must be positive")
        if replace_deadline <= 0:
            raise ValueError("replace_deadline must be positive")
        if handoff_deadline <= 0:
            raise ValueError("handoff_deadline must be positive")
        if config_write_deadline <= 0:
            raise ValueError("config_write_deadline must be positive")
        self._spawned = spawned
        self._transport = spawned.transport
        self.connection_id = connection_id
        self.incarnation_id = incarnation_id or str(uuid4())
        self.session_id: str | None = None
        self.session_dir: str | None = None
        self.jobs_dir: str | None = None
        self.open_accounting: RootOpenAccounting | None = None
        self._request_timeout = request_timeout
        self._compact_deadline = compact_deadline
        self._compact_reply_grace = max(compact_reply_grace, 0.0)
        self._replace_deadline = replace_deadline
        self._replace_reply_grace = max(replace_reply_grace, 0.0)
        self._handoff_deadline = handoff_deadline
        self._handoff_reply_grace = max(handoff_reply_grace, 0.0)
        self._config_write_deadline = config_write_deadline
        self._config_write_reply_grace = max(config_write_reply_grace, 0.0)
        self._replace_request: str | None = None
        self._request_ids = itertools.count(1)
        self._last_request_id: str | None = None
        # The long operations this end is waiting on: a short reconfigure mode
        # is refused while one is in flight, because the agent's fence would
        # refuse it there anyway.
        self._long_inflight: dict[str, str] = {}
        self._pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._operations: dict[str, AdmittedOperation] = {}
        self._operation_deadlines: dict[str, float] = {}
        self._compactions: dict[str, CompactSettlementRecord] = {}
        self._voided_compactions: dict[str, CompactSettlementRecord] = {}
        self._turns: dict[str, asyncio.Queue[_Frame]] = {}
        self._retry_sinks: dict[str, RetryObserver | None] = {}
        self._notifications: set[asyncio.Task[None]] = set()
        self._host_ports = HostServicePorts()
        self._host_tasks: set[asyncio.Task[None]] = set()
        self._turn_terminals: dict[str, dict[str, Any]] = {}
        self._notices: asyncio.Queue[BaseEvent | None] = asyncio.Queue()
        # The terminal user-input dispositions the agent asks this end to
        # publish, served from the sink the live backend binds.
        self._callback_settlement_sink: (
            Callable[[str, UserInputSettlement], Awaitable[bool]] | None
        ) = None
        self._routes: list[tuple[RegistryManifest, str]] = []
        self._active_turn_id: str | None = None
        self._closed = False
        self._death: RootDeath | None = None
        self._death_event = asyncio.Event()
        self._diagnostics: list[WireDiagnostic] = []
        # Consecutive op timeouts with no reply in between: the second one is
        # the idle-wedge signature rather than a single wedged operation.
        self._consecutive_timeouts = 0
        self._heartbeat_interval = heartbeat_interval
        self._pump: asyncio.Task[None] = asyncio.create_task(self._pump_messages())
        # WP7 [7C]: the liveness heartbeat. Pipe EOF is the usual death signal,
        # but a grandchild holding the write end open keeps the pipe silent
        # after the root is gone; the periodic process check catches that case.
        self._heartbeat: asyncio.Task[None] = asyncio.create_task(
            self._watch_liveness()
        )

    @property
    def process(self) -> subprocess.Popen[bytes]:
        return self._spawned.process

    @property
    def alive(self) -> bool:
        """Whether the root process is running and has not been declared dead."""
        return (
            not self._closed
            and self._death is None
            and self._spawned.process.poll() is None
        )

    @property
    def death(self) -> RootDeath | None:
        """The observed root death, or ``None`` while the root is alive."""
        return self._death

    @property
    def diagnostics(self) -> tuple[WireDiagnostic, ...]:
        """The wire-compat diagnostics this end recorded, oldest first."""
        return tuple(self._diagnostics)

    async def wait_for_death(self) -> RootDeath:
        """Wait until root death is declared and return the typed record."""
        await self._death_event.wait()
        death = self._death
        if death is None:
            raise RootAgentProcessError("the root agent process is not dead")
        return death

    def turn_terminal(self, turn_id: str) -> Mapping[str, Any] | None:
        """The terminal frame one turn ended on, when this end saw it.

        A child run's terminal frame is the run's whole completion evidence:
        the outcome, the error, and the typed failure block the child derived
        where its evidence lived.
        """
        return self._turn_terminals.get(turn_id)

    @property
    def last_request_id(self) -> str | None:
        """The id of the most recent coordinator-origin request, or ``None``."""
        return self._last_request_id

    def admitted_operation(self, request_id: str) -> AdmittedOperation | None:
        """The admission one request announced, when this end learned it."""
        return self._operations.get(request_id)

    def marker_verdict(
        self, snapshot: AgentSessionSnapshot, request_id: str
    ) -> MarkerVerdict:
        """Judge one long-operation request against one consumed snapshot.

        Correlation is direct by the coordinator's own request id (S9-C1): the
        marker record carries it, so the verdict needs no attribution logic.
        """
        operation = self._operations.get(request_id)
        deadline = self._operation_deadlines.get(request_id)
        return marker_verdict(
            snapshot,
            request_id=request_id,
            deadline=deadline if deadline is not None else snapshot.constructed_at,
            incarnation_id=(
                operation.incarnation_id if operation is not None else None
            ),
        )

    def compaction_settlement(self, request_id: str) -> CompactSettlementRecord | None:
        """The settlement this end holds for one compaction request, if any."""
        return self._compactions.get(request_id)

    def pending_compactions(self) -> tuple[CompactSettlementRecord, ...]:
        """Every compaction whose settlement is still open."""
        return tuple(record for record in self._compactions.values() if record.pending)

    def voided_compactions(self) -> tuple[CompactSettlementRecord, ...]:
        """Every settlement a session failure voided instead of resolving."""
        return tuple(self._voided_compactions.values())

    async def compact(self, extra_instructions: str = "") -> CompactSettlementRecord:
        """Run one manual compaction and answer with its settlement.

        The request is a long operation: it carries the coordinator's absolute
        monotonic deadline at the agent's own deadline, and this end waits one
        reply grace past it so the agent's abort -- which settles the record and
        answers the request id -- lands before any timeout. A compaction is
        never retried: a lost reply becomes ONE marker re-check against a
        snapshot, and the record carries what that re-check or the receipt
        proved.
        """
        deadline = time.monotonic() + self._compact_deadline
        params = encode_compact_request(extra_instructions, deadline=deadline)
        request_id = self._begin_request(params, long_method=AGENT_COMPACT_METHOD)
        try:
            payload = await self._await_request(
                request_id,
                AGENT_COMPACT_METHOD,
                params,
                timeout=self._compact_deadline + self._compact_reply_grace,
            )
        except TimeoutError:
            return await self._settle_compaction_timeout(request_id)
        reply = decode_compact_reply(payload)
        record = CompactSettlementRecord(
            request_id=request_id,
            settlement=reply.settlement,
            detail=reply.detail,
            reply=reply,
        )
        self._compactions[request_id] = record
        return record

    async def cancel_operation(self, op_id: str) -> None:
        """Send the idempotent cancel for one long operation.

        The notification carries no reply: the operation's own settlement is
        the assertable outcome. While that settlement is open the record is
        marked cancel-pending, which is the state a client surfaces until the
        next snapshot resolves it.
        """
        await self._notify(AGENT_OP_CANCEL_METHOD, encode_op_cancel(op_id))
        request_id = next(
            (
                candidate
                for candidate, operation in self._operations.items()
                if operation.op_id == op_id
            ),
            None,
        )
        record = self._compactions.get(request_id) if request_id is not None else None
        if record is None or not record.pending:
            return
        self._compactions[record.request_id] = replace(
            record,
            cancel_pending=True,
            detail=record.detail
            or "a cancel is in flight; the operation settles on its next await point",
        )

    async def _settle_compaction_timeout(
        self, request_id: str
    ) -> CompactSettlementRecord:
        """Settle one timed-out compaction with the ONE marker re-check.

        The op-4 marker set is authoritative and correlated by the request id
        (S9-C1): a TERMINAL record settles to its recorded class, an ACTIVE one
        settles provisionally in-flight, and absence from a payload constructed
        after the deadline proves never-admitted. The one-shot rule is what
        makes this a settlement rather than a retry: nothing is resent.
        """
        if self._operations.get(request_id) is None:
            # Transport class: no admission notification ever arrived, so this
            # end never held an op id. The re-check still runs -- absence from a
            # post-deadline payload is the proof -- and the diagnostic records
            # which signature the user is looking at.
            self._record_diagnostic(
                "a compaction request timed out with no admission notification; "
                "the operation is unattributed (transport class)",
                request_id,
                method=AGENT_COMPACT_METHOD,
            )
        try:
            snapshot = await self.snapshot(include_history=False)
        except BaseException as exc:
            if isinstance(exc, asyncio.CancelledError):
                raise
            record = CompactSettlementRecord(
                request_id=request_id,
                settlement=COMPACT_OBSERVATION_SETTLEMENT_UNKNOWN,
                detail=(
                    "the one-shot marker re-check failed "
                    f"({type(exc).__name__}: {exc}); a later snapshot clears it"
                ),
                pending=True,
            )
            self._compactions[request_id] = record
            return record
        record = self._settlement_from_verdict(
            request_id, self.marker_verdict(snapshot, request_id)
        )
        self._compactions[request_id] = record
        return record

    def _settlement_from_verdict(
        self, request_id: str, verdict: MarkerVerdict
    ) -> CompactSettlementRecord:
        """Turn one marker verdict into the settlement the coordinator holds."""
        if verdict.state == MARKER_TERMINAL:
            settlement = self._compaction_class(verdict.outcome)
            return CompactSettlementRecord(
                request_id=request_id,
                settlement=settlement,
                detail=(
                    "the operation's TERMINAL marker records the settlement"
                    if settlement != COMPACT_OBSERVATION_SETTLEMENT_UNKNOWN
                    else "the marker's terminal outcome is not a compaction class"
                ),
                op_id=verdict.op_id,
                outcome=verdict.outcome,
                pending=settlement == COMPACT_OBSERVATION_SETTLEMENT_UNKNOWN,
            )
        if verdict.state == MARKER_ABSENT:
            return CompactSettlementRecord(
                request_id=request_id,
                settlement=COMPACT_SETTLEMENT_CANCELLED_UNPUBLISHED,
                detail=(
                    "no marker record in a payload constructed after the "
                    "deadline: the operation was never admitted"
                ),
            )
        if verdict.state == MARKER_ACTIVE:
            return CompactSettlementRecord(
                request_id=request_id,
                settlement=COMPACT_OBSERVATION_IN_FLIGHT,
                detail=(
                    "admitted and still in flight at the re-check; the next "
                    "snapshot carrying its TERMINAL marker resolves it"
                ),
                op_id=verdict.op_id,
                pending=True,
            )
        return CompactSettlementRecord(
            request_id=request_id,
            settlement=COMPACT_OBSERVATION_SETTLEMENT_UNKNOWN,
            detail=verdict.detail or "the marker re-check was inconclusive",
            pending=True,
        )

    @staticmethod
    def _compaction_class(outcome: str | None) -> str:
        """The settlement class one TERMINAL outcome names, or unknown."""
        if outcome in COMPACT_AGENT_SETTLEMENTS:
            return str(outcome)
        return COMPACT_OBSERVATION_SETTLEMENT_UNKNOWN

    def _recheck_pending_compactions(self, snapshot: AgentSessionSnapshot) -> None:
        """Re-check every open settlement against one consumed snapshot payload.

        The record is authoritative and is created at admission, so a later
        payload resolves a provisional in-flight observation to its terminal
        class without any new request. An unresolved verdict leaves the record
        exactly as it was: a stale or still-active payload proves nothing new.
        """
        for request_id, record in list(self._compactions.items()):
            if not record.pending:
                continue
            resolved = self._settlement_from_verdict(
                request_id, self.marker_verdict(snapshot, request_id)
            )
            if resolved.pending:
                continue
            # A settlement ends the cancel's pending window: the operation it
            # named has settled, so nothing is still in flight to surface.
            self._compactions[request_id] = replace(
                resolved, reply=record.reply, cancel_pending=False
            )

    def _void_pending_compactions(self, reason: str) -> None:
        """Void open settlements on session failure: their table died with it."""
        for request_id, record in list(self._compactions.items()):
            if not record.pending:
                continue
            voided = replace(
                record,
                settlement=COMPACT_OBSERVATION_SETTLEMENT_VOIDED,
                detail=f"{record.detail} ({reason})" if record.detail else reason,
                pending=False,
            )
            self._compactions[request_id] = voided
            self._voided_compactions[request_id] = voided

    async def open(
        self,
        *,
        session_options: BaseModel,
        client_info: BaseModel,
        client_capabilities: BaseModel | None = None,
        session_id: str | None = None,
        continue_latest: bool = False,
    ) -> AgentOpenReply:
        """Open the session in the agent process and adopt its open reply.

        An open that resumes a stored session is a long-budget operation: it
        carries the coordinator's absolute monotonic deadline, and the agent
        aborts it at that deadline and answers the request id typed. A fresh
        open keeps the short default.
        """
        params = encode_open_request(
            session_options=session_options,
            client_info=client_info,
            client_capabilities=client_capabilities,
            session_id=session_id,
            continue_latest=continue_latest,
        )
        timeout: float | None = None
        if session_id is not None or continue_latest:
            deadline = time.monotonic() + RESUME_OPEN_DEADLINE
            params["deadline"] = deadline
            timeout = RESUME_OPEN_DEADLINE + RESUME_OPEN_REPLY_GRACE
        result = await self._request(AGENT_OPEN_METHOD, params, timeout=timeout)
        reply = decode_open_reply(result)
        self.session_id = reply.session_id
        self.session_dir = reply.session_dir
        self.jobs_dir = reply.jobs_dir
        self.open_accounting = reply.accounting
        return reply

    async def snapshot(
        self, *, history_limit: int = 200, include_history: bool = True
    ) -> AgentSessionSnapshot:
        """Read one session snapshot the agent builds at reply-write time.

        Every consumed payload re-checks the open compaction settlements: the
        marker set it carries is authoritative, so a provisional in-flight
        record resolves here without a request of its own.
        """
        payload = await self._request(
            AGENT_SESSION_SNAPSHOT_METHOD,
            {"history_limit": history_limit, "include_history": include_history},
        )
        snapshot = decode_session_snapshot(payload)
        self._recheck_pending_compactions(snapshot)
        return snapshot

    async def read_session_log(self) -> SessionLogSummary:
        """Read the session log projection from one snapshot payload."""
        return (await self.snapshot(include_history=False)).log

    def register_live_route(self, manifest: RegistryManifest, agent_id: str) -> None:
        """Route live pages for this agent through the registry manifest."""
        route = AgentRoute(
            connection_id=self.connection_id, incarnation_id=self.incarnation_id
        )
        manifest.register_agent(agent_id, route, self._page_reader)
        self._routes.append((manifest, agent_id))

    async def read_page(
        self, agent_id: str, *, before: str | None = None, limit: int = 50
    ) -> AgentTranscriptGetResponse:
        """Read one live page of the root agent's transcript."""
        payload = await self._request(
            AGENT_TRANSCRIPT_PAGE_METHOD,
            {"agent_id": agent_id, "before": before, "limit": limit},
        )
        return decode_transcript_page(payload)

    async def run_turn(
        self, request: ExecutionTurnRequest
    ) -> AsyncGenerator[BaseEvent, None]:
        """Run one turn in the agent process, yielding its events in order."""
        if self._closed:
            raise RootAgentProcessError("the root agent process is closed")
        if self._active_turn_id is not None:
            raise TurnOrderingConflict(self._active_turn_id)
        turn_id = request.turn_options.turn_id if request.turn_options else None
        turn_id = turn_id or str(uuid4())
        channel: asyncio.Queue[_Frame] = asyncio.Queue()
        self._turns[turn_id] = channel
        self._retry_sinks[turn_id] = (
            request.turn_options.retry_sink if request.turn_options else None
        )
        self._active_turn_id = turn_id
        # The host port proxies ride THIS turn: an agent-origin host request is
        # served from the ports the coordinator offered here, and an in-flight
        # host call at the turn boundary follows the turn's cancellation rules.
        self._host_ports = HostServicePorts(
            subagent_runner=request.subagent_runner, tool_io=request.tool_io
        )
        try:
            await self._transport.send({
                "jsonrpc": "2.0",
                "method": AGENT_TURN_START_METHOD,
                "params": {
                    "turn_id": turn_id,
                    "host_services": encode_host_service_offers(
                        subagent_runner=request.subagent_runner, tool_io=request.tool_io
                    ),
                    **encode_turn_request(
                        replace(
                            request,
                            subagent_runner=None,
                            tool_io=None,
                            turn_options=replace(
                                request.turn_options or AgentTurnOptions(),
                                turn_id=turn_id,
                                retry_sink=None,
                            ),
                        )
                    ),
                },
            })
            while True:
                kind, payload = await channel.get()
                if kind == "event":
                    yield decode_event(_mapping(payload.get("event")))
                elif kind == "retrying":
                    sink = self._retry_sinks[turn_id]
                    if sink is not None:
                        await sink(decode_retry_reason(_mapping(payload.get("reason"))))
                else:
                    await self._finish_turn(payload)
                    return
        finally:
            self._host_ports = HostServicePorts()
            self._cancel_host_tasks()
            self._turns.pop(turn_id, None)
            self._retry_sinks.pop(turn_id, None)
            self._active_turn_id = None

    def resolve_user_input(self, request_id: str, result: BaseModel) -> None:
        """Answer an open user-input request in the agent process."""
        self._schedule_notification(
            AGENT_TURN_RESOLVE_METHOD,
            {"request_id": request_id, "result": encode_user_input_result(result)},
        )

    def reject_user_input(self, request_id: str, error: BaseException) -> None:
        """Fail an open user-input request in the agent process."""
        self._schedule_notification(
            AGENT_TURN_REJECT_METHOD, {"request_id": request_id, "message": str(error)}
        )

    def relay_user_input_request(self, event: UserInputRequestEvent) -> None:
        """Relay a host-injected request; its record is adopted agent-side."""
        self._schedule_notification(
            AGENT_TURN_RELAY_METHOD, {"event": encode_event(event)}
        )

    async def settle_user_input_answer(
        self, request_id: str, result: BaseModel
    ) -> UserInputSettleOutcome:
        """Relay an answer; the agent process arbitrates and settles."""
        payload = await self._request(
            AGENT_TURN_ANSWER_METHOD,
            {"request_id": request_id, "result": encode_user_input_result(result)},
        )
        return cast("UserInputSettleOutcome", payload.get("outcome"))

    async def settle_user_input_rejection(
        self, request_id: str, error: BaseException
    ) -> UserInputSettleOutcome:
        """Relay a rejection; the agent process arbitrates and settles."""
        payload = await self._request(
            AGENT_TURN_REFUSE_METHOD, {"request_id": request_id, "message": str(error)}
        )
        return cast("UserInputSettleOutcome", payload.get("outcome"))

    async def keep_waiting_user_input(self, request_id: str) -> KeepWaitingOutcome:
        """Relay a keep-waiting request; the deadline decision is agent-side."""
        payload = await self._request(
            AGENT_TURN_KEEP_WAITING_METHOD, {"request_id": request_id}
        )
        return cast("KeepWaitingOutcome", payload.get("outcome"))

    async def cancel_user_input(self, request_id: str, error: BaseException) -> None:
        """Relay a cancellation; the request is closed agent-side."""
        await self._request(
            AGENT_TURN_CANCEL_METHOD, {"request_id": request_id, "message": str(error)}
        )

    async def interrupt_turn(self, turn_id: str) -> None:
        """Interrupt the turn running in the agent process.

        The frame is a notification: there is no reply to settle, and the turn's
        own terminal frame still reports its interrupted outcome.
        """
        await self._notify(AGENT_TURN_INTERRUPT_METHOD, {"turn_id": turn_id})

    async def steer(self, params: TurnSteerParams) -> SteerSettlement:
        """Deliver one steering request; the agent arbitrates and settles it."""
        payload = await self._request(
            AGENT_TURN_STEER_METHOD, encode_steer_request(params)
        )
        return decode_steer_settlement(payload)

    async def inject(
        self,
        params: ContextInjectParams,
        *,
        manual_shell: ManualShellContext | None = None,
    ) -> list[BaseEvent]:
        """Inject one delivery between turns; the reply carries its events."""
        payload = await self._request(
            AGENT_INJECT_METHOD, encode_inject_request(params, manual_shell)
        )
        return decode_inject_reply(payload)

    async def run_shell(
        self, request: ShellRunRequest
    ) -> tuple[str, ManualShellContext]:
        """Run one queued manual ``!`` command; the reply is its record.

        The command's own deadline bounds the effect, so this end waits that
        deadline plus a settling margin: a user-bounded command must be
        answered by its own timeout, never by a request timeout that fires
        first and reports an execution that was still being captured.
        """
        payload = await self._request(
            AGENT_SHELL_RUN_METHOD,
            encode_shell_run_request(request),
            timeout=request.timeout_seconds + SHELL_REQUEST_MARGIN,
        )
        return decode_shell_run_reply(payload)

    async def interrupt_shell(self, operation_id: str) -> bool:
        """Stop one running manual command; an unknown one runs nothing."""
        payload = await self._request(
            AGENT_SHELL_INTERRUPT_METHOD, encode_shell_interrupt(operation_id)
        )
        return decode_shell_interrupt_reply(payload)

    async def teardown_child_attempt(
        self, request: ChildTeardownRequest
    ) -> ChildTeardownReport:
        """Settle one child publication attempt whose outcome was lost.

        The coordinator mints the attempt id, so it still holds the identity
        when the aggregate reply never arrives: this asks the agent that owns
        the children to settle the attempt conservatively. The reply names the
        class the owner established and is never a claimed rollback.
        """
        payload = await self._request(AGENT_CHILD_TEARDOWN_METHOD, request.to_wire())
        return ChildTeardownReport.from_wire(payload)

    async def replace(
        self, *, session_id: str | None = None, continue_latest: bool = False
    ) -> ReplaceReply:
        """Rebind the agent's session to a stored one.

        The old cohort is the coordinator's to dispose of and is reaped BEFORE
        this call, so this end owns only what the agent can see: a replacement
        is refused typed before a frame is sent while this end holds an
        unresolved settlement or a prior replacement, it carries an absolute
        deadline, and a lost reply is settled by ONE marker re-check -- the
        operation's own TERMINAL record seeds the incarnation its rebind
        created, so the re-check always finds its evidence.
        """
        if self._closed:
            raise RootAgentProcessError("the root agent process is closed")
        if self._death is not None:
            raise RootAgentExitedError(
                self._death.reason, exit_code=self._death.exit_code
            )
        self._require_replace_admissible()
        deadline = time.monotonic() + self._replace_deadline
        params = encode_replace_request(
            session_id=session_id, continue_latest=continue_latest, deadline=deadline
        )
        request_id = self._begin_request(
            params, long_method=AGENT_SESSION_REPLACE_METHOD
        )
        self._replace_request = request_id
        try:
            reply = await self._replace_reply(request_id, params)
        finally:
            if self._replace_request == request_id:
                self._replace_request = None
        self._adopt_replacement(reply)
        return reply

    async def _replace_reply(
        self, request_id: str, params: dict[str, Any]
    ) -> ReplaceReply:
        """Await one replacement's reply, or settle its loss with the re-check."""
        try:
            payload = await self._await_request(
                request_id,
                AGENT_SESSION_REPLACE_METHOD,
                params,
                timeout=self._replace_deadline + self._replace_reply_grace,
            )
        except TimeoutError:
            return await self._settle_replace_timeout(request_id)
        except RootAgentRequestError as exc:
            if exc.code == AGENT_SESSION_NOT_FOUND_ERROR:
                raise RootAgentSessionNotFound(exc.message) from exc
            raise
        return decode_replace_reply(payload)

    def _require_replace_admissible(self) -> None:
        """Refuse a replacement before a frame is sent, typed.

        Both checks are the coordinator's own state: an unresolved settlement
        holds an op id whose record the rebind clears, and a prior unresolved
        replacement would race this one's seed evidence with its own.
        """
        pending = self.pending_compactions()
        if pending:
            raise RootAgentReplaceConflict(
                "Cannot replace the session while "
                f"{pending[0].op_id or pending[0].request_id} is still unsettled. "
                "Read the session state and retry once it settles. Nothing was "
                "replaced."
            )
        if self._replace_request is not None:
            raise RootAgentReplaceConflict(
                "Cannot replace the session while the previous replacement is "
                "still unresolved. Read the session state, then retry. Nothing "
                "was replaced."
            )

    async def _settle_replace_timeout(self, request_id: str) -> ReplaceReply:
        """Settle one timed-out replacement with the ONE marker re-check.

        The operation's own record survives the rebind it belongs to, so a
        payload constructed after the deadline carries it: a completed record
        proves the rebind landed and that payload is the fresh identity.
        Anything else claims nothing -- a replacement is never resent.
        """
        try:
            snapshot = await self.snapshot(include_history=False)
        except BaseException as exc:
            if isinstance(exc, asyncio.CancelledError):
                raise
            raise RootAgentReplaceConflict(
                "the replacement's one-shot marker re-check failed "
                f"({type(exc).__name__}: {exc}), so nothing is claimed and "
                "nothing is resent; read the session state before replacing again"
            ) from exc
        verdict = self.marker_verdict(snapshot, request_id)
        if verdict.state == MARKER_TERMINAL and verdict.outcome == OP_OUTCOME_COMPLETED:
            return ReplaceReply(snapshot=snapshot)
        outcome = f" ({verdict.outcome})" if verdict.outcome else ""
        raise RootAgentReplaceConflict(
            "the replacement's reply was lost and its marker re-check answered "
            f"{verdict.state}{outcome}: nothing is claimed and nothing is "
            "resent; read the session state before replacing again"
        )

    def _adopt_replacement(self, reply: ReplaceReply) -> None:
        """Adopt the identity one replacement left behind."""
        self.session_id = reply.snapshot.state.session.id
        self.session_dir = reply.snapshot.session_dir
        self.jobs_dir = reply.snapshot.jobs_dir
        if reply.accounting is not None:
            self.open_accounting = reply.accounting
        # The rebind cleared the marker table, so nothing that was pending can
        # resolve against it any more: those settlements are voided and
        # enumerated instead of waiting for a payload that cannot carry them.
        self._void_pending_compactions(
            "the identity replacement cleared the marker table"
        )

    async def rename(self, title: str) -> RenameReply:
        """Apply one manual title where the session metadata lives.

        The title is session-scoped state the agent process owns, so it is
        applied there and the reply carries what actually persisted.
        """
        payload = await self._request(
            AGENT_SESSION_RENAME_METHOD, encode_rename_request(title)
        )
        return decode_rename_reply(payload)

    async def update_settings(
        self, *, max_turns: int | None = None, max_tokens: int | None = None
    ) -> ReconfigureReply:
        """Apply the session bounds where the loop lives: apply and acknowledge.

        The mode is short: it sets the bounds and answers, with no history reset
        and no plan readback. It is refused typed while a long operation is in
        flight, because the agent's one-long-op fence would refuse it there.
        """
        self._require_short_mode_admissible("update the session settings")
        params = encode_reconfigure_request(
            mode=RECONFIGURE_MODE_SETTINGS,
            session_id=self._require_session_id(),
            max_turns=max_turns,
            max_tokens=max_tokens,
        )
        return await self._reconfigure_request(params)

    async def clear_history(self) -> ReconfigureReply:
        """Clear the history in the agent process.

        Everything the coordinator owns stays coordinator-side and runs BEFORE
        this call: the /clear plan-episode settlement and the pin removal are
        the TUI leg's own steps, and a refused settlement means this frame is
        never sent.
        """
        self._require_short_mode_admissible("clear the history")
        params = encode_reconfigure_request(
            mode=RECONFIGURE_MODE_HISTORY_CLEAR, session_id=self._require_session_id()
        )
        return await self._reconfigure_request(params)

    async def write_config(self, candidate: ConfigWriteParams) -> ReconfigureReply:
        """Run one config transaction in the agent process as a long operation.

        The candidate crosses verbatim and the receipt is the SHIPPED result: the
        persistence x application dimensions, the rejected flag, the failure
        details, the revision, the saved values and the launch-metadata outcome.
        A lost reply is settled by ONE marker re-check and is never resent -- a
        write that proves it completed is reported as completed-with-lost-receipt
        so the caller reads the settings instead of repeating the transaction.
        """
        if self._closed:
            raise RootAgentProcessError("the root agent process is closed")
        if self._death is not None:
            raise RootAgentExitedError(
                self._death.reason, exit_code=self._death.exit_code
            )
        deadline = time.monotonic() + self._config_write_deadline
        request = encode_reconfigure_request(
            mode=RECONFIGURE_MODE_CONFIG_WRITE,
            session_id=self._require_session_id(),
            deadline=deadline,
            config_write=candidate,
        )
        request_id = self._begin_request(
            request, long_method=RECONFIGURE_MODE_CONFIG_WRITE
        )
        try:
            payload = await self._await_request(
                request_id,
                AGENT_SESSION_RECONFIGURE_METHOD,
                request,
                timeout=self._config_write_deadline + self._config_write_reply_grace,
            )
        except TimeoutError:
            reply = await self._settle_config_write_timeout(request_id)
        except RootAgentRequestError as exc:
            self._raise_reconfigure_code(exc)
        else:
            reply = decode_reconfigure_reply(payload)
        self._adopt_reconfigure(reply)
        return reply

    async def plan_handoff(self, params: SessionPlanHandoffParams) -> ReconfigureReply:
        """Run the full consent fence in the agent process as a long operation.

        The consent snapshot crosses verbatim; the agent applies exactly those
        values or fails without starting execution. A lost reply is settled by
        ONE marker re-check -- the operation's own TERMINAL record -- and is
        never resent.
        """
        if self._closed:
            raise RootAgentProcessError("the root agent process is closed")
        if self._death is not None:
            raise RootAgentExitedError(
                self._death.reason, exit_code=self._death.exit_code
            )
        deadline = time.monotonic() + self._handoff_deadline
        request = encode_reconfigure_request(
            mode=RECONFIGURE_MODE_PLAN_HANDOFF, deadline=deadline, handoff=params
        )
        request_id = self._begin_request(
            request, long_method=RECONFIGURE_MODE_PLAN_HANDOFF
        )
        try:
            payload = await self._await_request(
                request_id,
                AGENT_SESSION_RECONFIGURE_METHOD,
                request,
                timeout=self._handoff_deadline + self._handoff_reply_grace,
            )
        except TimeoutError:
            reply = await self._settle_handoff_timeout(request_id)
        except RootAgentRequestError as exc:
            self._raise_reconfigure_code(exc)
        else:
            reply = decode_reconfigure_reply(payload)
        # The handoff rotates the session, so the reply's payload is the state
        # to address from now on -- including the one the re-check read, whose
        # identity the resent-free settlement must honour just the same.
        self._adopt_reconfigure(reply)
        return reply

    async def list_sessions(self, params: SessionListParams) -> SessionListResponse:
        """List stored sessions from the config the agent's own loop holds."""
        payload = await self._request(
            AGENT_SESSION_LIST_METHOD, params.model_dump(mode="json")
        )
        return SessionListResponse.model_validate(payload)

    async def reload_config(self, params: ConfigReloadParams) -> ConfigMutationResponse:
        """Run one config reload transaction in the agent process.

        The agent's orchestrator and backend are the ones the reload applies
        to, so the receipt names the config this session is actually using.
        """
        payload = await self._request(
            AGENT_SESSION_CONFIG_RELOAD_METHOD, params.model_dump(mode="json")
        )
        return ConfigMutationResponse.model_validate(payload)

    async def read_plan_record(self) -> SessionPlanRecord | None:
        """Read the durable plan mode the agent's session metadata carries."""
        payload = await self._request(
            AGENT_SESSION_PLAN_RECORD_READ_METHOD,
            {"session_id": self._require_session_id()},
        )
        record = payload.get("record")
        return None if record is None else SessionPlanRecord.model_validate(record)

    async def write_plan_record(
        self, params: SessionPlanRecordWriteParams
    ) -> SessionPlanRecordWriteResponse:
        """Persist (or remove) the plan mode where the transcript is written."""
        payload = await self._request(
            AGENT_SESSION_PLAN_RECORD_WRITE_METHOD, params.model_dump(mode="json")
        )
        return SessionPlanRecordWriteResponse.model_validate(payload)

    async def read_policy(self, params: PolicyReadParams) -> PolicyReadResponse:
        """Read the policy revision and sources the agent's loop enforces."""
        payload = await self._request(
            AGENT_SESSION_POLICY_READ_METHOD, params.model_dump(mode="json")
        )
        return PolicyReadResponse.model_validate(payload)

    async def read_roots(self, params: RootsReadParams) -> RootsReadResponse:
        """Read the authorized-roots view the agent's own source composes."""
        payload = await self._request(
            AGENT_SESSION_ROOTS_READ_METHOD, params.model_dump(mode="json")
        )
        return RootsReadResponse.model_validate(payload)

    async def save_roots(self, params: RootsSaveParams) -> RootsSaveResponse:
        """Persist one approved root grant where the agent's config lives."""
        payload = await self._request(
            AGENT_SESSION_ROOTS_SAVE_METHOD, params.model_dump(mode="json")
        )
        return RootsSaveResponse.model_validate(payload)

    async def policy_request(
        self, method: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Drive one policy-tree phase in the agent that owns the loop.

        The agent serves the four phases of its own policy node, so a tree
        transaction runs its prepare and commit where that node's authority
        lives rather than on a coordinator copy of it.
        """
        return await self._request(method, params)

    async def _reconfigure_request(self, params: dict[str, Any]) -> ReconfigureReply:
        """Send one short reconfigure mode and adopt the state it leaves."""
        try:
            payload = await self._request(AGENT_SESSION_RECONFIGURE_METHOD, params)
        except RootAgentRequestError as exc:
            self._raise_reconfigure_code(exc)
        reply = decode_reconfigure_reply(payload)
        self._adopt_reconfigure(reply)
        return reply

    def _raise_reconfigure_code(self, exc: RootAgentRequestError) -> NoReturn:
        """Map one typed wire failure onto this end's own refusal classes."""
        if exc.code == AGENT_SESSION_NOT_FOUND_ERROR:
            raise RootAgentSessionNotFound(exc.message) from exc
        if exc.code == RECONFIGURE_CONFLICT_ERROR:
            raise RootAgentReconfigureConflict(exc.message) from exc
        raise exc

    def _require_short_mode_admissible(self, described: str) -> None:
        """Refuse a short reconfigure before a frame is sent, typed.

        Both checks are this end's own state: an open settlement holds the
        agent's fence until it resolves, and a long operation this end is still
        waiting on holds it right now.
        """
        pending = self.pending_compactions()
        if pending:
            raise RootAgentReconfigureConflict(
                f"Cannot {described} while {pending[0].op_id or pending[0].request_id} "
                "is still unsettled. Read the session state and retry once it "
                "settles."
            )
        if self._long_inflight:
            method = next(iter(self._long_inflight.values()))
            raise RootAgentReconfigureConflict(
                f"Cannot {described} while {method} is in flight; read the "
                "session state and retry once it settles."
            )

    async def _settle_handoff_timeout(self, request_id: str) -> ReconfigureReply:
        """Settle one timed-out handoff with the ONE marker re-check.

        The operation's own record survives the handoff, so a payload
        constructed after the deadline carries it: a completed record proves the
        fence reached its terminal flip, and that payload is the state the
        reply would have carried. Anything else claims nothing, and a handoff
        is never resent.
        """
        try:
            snapshot = await self.snapshot(include_history=False)
        except BaseException as exc:
            if isinstance(exc, asyncio.CancelledError):
                raise
            raise RootAgentReconfigureConflict(
                "the plan handoff's one-shot marker re-check failed "
                f"({type(exc).__name__}: {exc}), so nothing is claimed and "
                "nothing is resent; read the session state before retrying"
            ) from exc
        verdict = self.marker_verdict(snapshot, request_id)
        if verdict.state == MARKER_TERMINAL and verdict.outcome == OP_OUTCOME_COMPLETED:
            return ReconfigureReply(
                mode=RECONFIGURE_MODE_PLAN_HANDOFF,
                outcome=RECONFIGURE_OUTCOME_ACCEPTED,
                snapshot=snapshot,
            )
        outcome = f" ({verdict.outcome})" if verdict.outcome else ""
        raise RootAgentReconfigureConflict(
            "the plan handoff's reply was lost and its marker re-check answered "
            f"{verdict.state}{outcome}: nothing is claimed and nothing is "
            "resent; read the session state before retrying"
        )

    async def _settle_config_write_timeout(self, request_id: str) -> ReconfigureReply:
        """Settle one timed-out config write with the ONE marker re-check.

        A TERMINAL completed record proves the transaction ran, but the shipped
        dimensions that came with its reply are gone: the answer says exactly
        that and points at a settings read, so nothing is invented and the write
        is never resent.
        """
        try:
            snapshot = await self.snapshot(include_history=False)
        except BaseException as exc:
            if isinstance(exc, asyncio.CancelledError):
                raise
            raise RootAgentReconfigureConflict(
                "the config write's one-shot marker re-check failed "
                f"({type(exc).__name__}: {exc}), so nothing is claimed and "
                "nothing is resent; read the configuration before retrying"
            ) from exc
        verdict = self.marker_verdict(snapshot, request_id)
        if verdict.state == MARKER_TERMINAL and verdict.outcome == OP_OUTCOME_COMPLETED:
            raise RootAgentReconfigureConflict(
                "The config write ran to completion in the agent process but its "
                "reply was lost, so its persistence and application results "
                "could not be read; nothing is resent. Read the configuration to "
                "confirm what was saved before retrying."
            )
        outcome = f" ({verdict.outcome})" if verdict.outcome else ""
        raise RootAgentReconfigureConflict(
            "the config write's reply was lost and its marker re-check answered "
            f"{verdict.state}{outcome}: nothing is claimed and nothing is "
            "resent; read the session state before retrying"
        )

    def _adopt_reconfigure(self, reply: ReconfigureReply) -> None:
        """Adopt the identity a reconfigure reply left behind.

        A clear rotates the session (and a handoff rotates it too), so the
        agent's own reply is the authority: the snapshot carries the id and the
        directories this end must address from now on. The marker table is NOT
        cleared -- the incarnation does not change -- so any open settlement
        still resolves against the same table.
        """
        snapshot = reply.snapshot
        if snapshot is None:
            return
        self.session_id = snapshot.state.session.id
        self.session_dir = snapshot.session_dir
        self.jobs_dir = snapshot.jobs_dir
        if reply.accounting is not None:
            self.open_accounting = reply.accounting

    def _require_session_id(self) -> str:
        if self.session_id is None:
            raise RootAgentProcessError("the root agent process has no open session")
        return self.session_id

    def refuse_deferred_session_operation(self, method: str) -> NoReturn:
        """Refuse one session operation this release does not carry, before sending.

        Fork, relocate and the rewind pair have no carrier over the process
        wire in this release. A frame would only be refused at the far end, so
        the refusal is minted here and names the same in_process escape.
        """
        raise RootAgentDeferredOperation(deferred_session_message(method))

    async def out_of_band_events(self) -> AsyncGenerator[BaseEvent, None]:
        """Stream the turn-less notices the agent process emits.

        The coordinator's single drain consumer reads this exactly as it reads
        the in-process loop's own stream; closing the handle or the root's death
        ends it instead of leaving it waiting on a pipe nobody owns.
        """
        while True:
            event = await self._notices.get()
            if event is None:
                return
            yield event

    def _end_notices(self) -> None:
        self._notices.put_nowait(None)

    async def close(self) -> None:
        """Retire the live route and terminate the agent process."""
        if self._closed:
            return
        self._closed = True
        for manifest, agent_id in self._routes:
            with suppress(OSError):
                manifest.remove_agent(agent_id)
        self._routes.clear()
        self._host_ports = HostServicePorts()
        self._cancel_host_tasks()
        # The publication sink belongs to the live backend: a closed handle
        # answers no settlement request, so the agent never waits on it.
        self._callback_settlement_sink = None
        self._turn_terminals.clear()
        self._end_notices()
        for task in list(self._notifications):
            task.cancel()
        if self._notifications:
            await asyncio.gather(*self._notifications, return_exceptions=True)
        self._pump.cancel()
        with suppress(asyncio.CancelledError):
            await self._pump
        self._heartbeat.cancel()
        with suppress(asyncio.CancelledError):
            await self._heartbeat
        self._fail_turns("the root agent process closed")
        self._void_pending_compactions("the root agent process closed")
        await self._transport.close()
        # The tree slot returns to the budget only after the child is reaped.
        self._spawned.retire()

    async def _pump_messages(self) -> None:
        failure: str | None = None
        try:
            async for message in self._transport.messages():
                self._dispatch(message)
        except BaseException as exc:
            failure = f"{type(exc).__name__}: {exc}"
        finally:
            if self._closed:
                reason = failure or "the root agent process closed"
                self._fail_requests(reason)
                self._fail_turns(reason)
            else:
                # EOF on the root pipe is root death, not a transport hiccup:
                # whatever the pump ended on, the root can no longer answer.
                # A pipe that closes with the process still running is the
                # immediate signature; a pipe closing behind a process that
                # already exited is the same death the watcher would report.
                self._declare_death(
                    "the root agent pipe closed"
                    if failure is None
                    else f"the root agent pipe closed: {failure}",
                    signature=(
                        RootFailureSignature.DEATH_RECEIPT
                        if self._spawned.process.poll() is not None
                        else RootFailureSignature.PIPE_EOF
                    ),
                )

    async def _watch_liveness(self) -> None:
        """Declare root death when the process is gone but the pipe stays open."""
        while True:
            await asyncio.sleep(self._heartbeat_interval)
            if self._closed or self._death is not None:
                return
            if self._spawned.process.poll() is not None:
                self._declare_death(
                    "the root agent process exited",
                    signature=RootFailureSignature.DEATH_RECEIPT,
                )
                return

    def _declare_death(
        self,
        reason: str,
        *,
        signature: RootFailureSignature = RootFailureSignature.DEATH_RECEIPT,
    ) -> None:
        """Publish one typed root death and fail everything still waiting."""
        if self._death is not None or self._closed:
            return
        exit_code = self._spawned.process.poll()
        death = RootDeath(reason=reason, exit_code=exit_code, signature=signature)
        self._death = death
        self._death_event.set()
        self._end_notices()
        self._host_ports = HostServicePorts()
        self._cancel_host_tasks()
        # The marker table died with the incarnation, so nothing pending can
        # resolve any more: the settlements are voided and enumerated instead.
        self._void_pending_compactions(reason)
        self._record_diagnostic(death.diagnostic, None)
        for future in self._pending.values():
            if not future.done():
                future.set_exception(
                    RootAgentExitedError(
                        reason, exit_code=exit_code, signature=signature
                    )
                )
        for turn_id, channel in self._turns.items():
            channel.put_nowait((
                "ended",
                {
                    "turn_id": turn_id,
                    "outcome": TURN_OUTCOME_FAILED,
                    "error": death.diagnostic,
                },
            ))

    def _dispatch(self, message: Mapping[str, Any]) -> None:
        """Route one agent frame by method presence and the closed tables.

        A method-bearing frame with an id is an agent-origin request: the two
        op-9 host-service methods are served from the turn's host ports, and
        everything else is answered with the typed compat rejection. A
        method-bearing frame without an id is a notification; a frame with no
        method answers a coordinator-origin request. An unlisted notification is
        a recorded diagnostic, never a root-death claim: the loop keeps serving.
        """
        method = message.get("method")
        if isinstance(method, str):
            if "id" in message:
                if method in COORDINATOR_RECEIVING_REQUESTS:
                    self._serve_agent_request(message, method)
                    return
                self._reject_agent_request(message, method)
                return
            if method in COORDINATOR_RECEIVING_NOTIFICATIONS:
                self._deliver_notification(method, _payload(message))
                return
            self._record_diagnostic(
                "the method is not in this direction's closed table",
                None,
                method=method,
            )
            return
        if "id" in message:
            self._settle_request(message)
            return
        self._record_diagnostic(
            "the frame carries no method and answers no request", None
        )

    def _serve_agent_request(self, message: Mapping[str, Any], method: str) -> None:
        """Serve one agent-origin host-service request on its own task."""
        if request_id_origin(message["id"]) != AGENT_ORIGIN:
            # An agent-origin method with a foreign id would settle a future
            # this end never minted: answer it typed instead of correlating it.
            error = WireMethodNotSupportedError(method, direction=AGENT_ORIGIN)
            self._record_diagnostic(
                "the request id is not in the agent-origin namespace",
                message["id"],
                method=method,
            )
            self._send_soon({
                "jsonrpc": "2.0",
                "id": message["id"],
                "error": {"code": type(error).__name__, "message": str(error)},
            })
            return
        serve = (
            self._serve_tool_io_request
            if method == HOST_TOOL_IO_METHOD
            else (
                self._serve_callback_settlement
                if method == AGENT_TURN_CALLBACK_METHOD
                else self._serve_subagent_run
            )
        )
        task = asyncio.create_task(serve(message["id"], _payload(message)))
        self._host_tasks.add(task)
        task.add_done_callback(self._host_tasks.discard)

    def bind_callback_settlement_sink(
        self, sink: Callable[[str, UserInputSettlement], Awaitable[bool]]
    ) -> None:
        """Serve the agent's terminal-callback publications from *sink*.

        The agent awaits this end's answer before its waiter unblocks, so a
        sink that publishes and answers is what keeps the terminal projection
        ordered ahead of the effect it unblocks.
        """
        self._callback_settlement_sink = sink

    async def _serve_callback_settlement(
        self, request_id: object, params: dict[str, Any]
    ) -> None:
        """Publish one terminal callback disposition the agent settled."""
        try:
            callback_id, settlement = decode_user_input_settlement(params)
        except AgentWireError as exc:
            await self._answer_agent_request(request_id, exc)
            return
        sink = self._callback_settlement_sink
        published = False
        if sink is not None:
            try:
                published = await sink(callback_id, settlement)
            except Exception as exc:
                await self._answer_agent_request(request_id, exc)
                return
        await self._send({
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {"published": published},
        })

    def _cancel_host_tasks(self) -> None:
        """Cancel in-flight host calls: they follow their turn's cancellation."""
        for task in list(self._host_tasks):
            task.cancel()
        self._host_tasks.clear()

    async def _serve_tool_io_request(
        self, request_id: object, params: dict[str, Any]
    ) -> None:
        """Serve one tool I/O operation from the turn's host port."""
        tool_io = self._host_ports.tool_io
        if tool_io is None:
            await self._answer_agent_request(
                request_id,
                UnsupportedProcessTurnRequest(
                    "the turn in flight offered no host tool I/O port"
                ),
            )
            return
        try:
            op, arguments = decode_tool_io_request(params)
            value = await _tool_io_call(tool_io, op, arguments)
        except BaseException as exc:
            await self._answer_agent_request(request_id, exc)
            return
        await self._send({
            "jsonrpc": "2.0",
            "id": request_id,
            "result": encode_tool_io_result(op, value),
        })

    async def _serve_subagent_run(
        self, request_id: object, params: dict[str, Any]
    ) -> None:
        """Serve one host subagent run: accept, stream every frame, close once."""
        runner = self._host_ports.subagent_runner
        try:
            run_id, args, context = decode_subagent_run_request(params)
            if runner is None:
                raise UnsupportedProcessTurnRequest(
                    "the turn in flight offered no host subagent runner"
                )
        except BaseException as exc:
            await self._answer_agent_request(request_id, exc)
            return
        await self._send({
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {"run_id": run_id, "accepted": True},
        })
        result: TaskResult | None = None
        error: str | None = None
        try:
            async with aclosing(runner.run(args, _invoke_context(context))) as frames:
                async for frame in frames:
                    await self._notify(
                        HOST_SUBAGENT_EVENT_METHOD,
                        {"run_id": run_id, "frame": encode_host_run_frame(frame)},
                    )
                    if isinstance(frame, TaskResult):
                        result = frame
                        break
        except asyncio.CancelledError:
            # The turn ended or the root is closing: the agent still learns the
            # run is over, so its generator never waits on a channel nobody owns.
            self._send_soon({
                "jsonrpc": "2.0",
                "method": HOST_SUBAGENT_ENDED_METHOD,
                "params": {
                    "run_id": run_id,
                    **encode_host_run_end(error="the host subagent run was cancelled"),
                },
            })
            raise
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
        await self._notify(
            HOST_SUBAGENT_ENDED_METHOD,
            {"run_id": run_id, **encode_host_run_end(result=result, error=error)},
        )

    async def _answer_agent_request(
        self, request_id: object, error: BaseException
    ) -> None:
        await self._send({
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": type(error).__name__, "message": str(error)},
        })

    def _deliver_notification(self, method: str, payload: dict[str, Any]) -> None:
        if method == AGENT_TURN_EVENT_METHOD:
            self._deliver(payload, "event")
            return
        if method == AGENT_NOTICE_METHOD:
            self._deliver_notice(payload)
            return
        if method == AGENT_OP_ADMISSION_METHOD:
            self._record_admission(payload)
            return
        if method == AGENT_TURN_RETRYING_METHOD:
            self._deliver(payload, "retrying")
            return
        if method == AGENT_TURN_ENDED_METHOD:
            self._deliver(payload, "ended")
            return
        # The table admitted the method, so only a table entry that lost its
        # handler reaches here: record it instead of failing the pump.
        self._record_diagnostic(
            "the table lists the method but no handler serves it", None, method=method
        )

    def _record_admission(self, payload: Mapping[str, Any]) -> None:
        """Hold one long operation's admission by the coordinator's request id.

        An admission that arrives after its request already timed out is still
        honored as an ATTRIBUTION input: it names the op id, so a cancel can be
        addressed and the marker re-check has a record to look for. A record
        that already settled is not re-settled; the late frame is a diagnostic.
        """
        try:
            marker = decode_operation_marker(payload)
        except AgentWireError as exc:
            self._record_diagnostic(
                f"an admission frame cannot be admitted: {exc}", None
            )
            return
        self._operations[marker.request_id] = AdmittedOperation(
            request_id=marker.request_id,
            op_id=marker.op_id,
            method=marker.method,
            deadline=marker.deadline,
        )
        settled = self._compactions.get(marker.request_id)
        if settled is not None and not settled.pending:
            self._record_diagnostic(
                "an admission arrived after its request had already settled; "
                "the late frame never re-settles",
                marker.request_id,
                method=AGENT_OP_ADMISSION_METHOD,
            )

    def _reject_agent_request(self, message: Mapping[str, Any], method: str) -> None:
        """Answer an agent-origin request typed; the agent settles, never hangs."""
        error = WireMethodNotSupportedError(method, direction=AGENT_ORIGIN)
        self._record_diagnostic(
            "the method is not in the coordinator's closed agent-request table",
            message["id"],
            method=method,
        )
        self._send_soon({
            "jsonrpc": "2.0",
            "id": message["id"],
            "error": {"code": type(error).__name__, "message": str(error)},
        })

    def _record_diagnostic(
        self, reason: str, request_id: object, *, method: object = None
    ) -> None:
        diagnostic = WireDiagnostic(
            direction=AGENT_ORIGIN, reason=reason, method=method, request_id=request_id
        )
        self._diagnostics.append(diagnostic)
        logger.warning(diagnostic.message())

    def _deliver(self, payload: dict[str, Any], kind: str) -> None:
        channel = self._turns.get(str(payload.get("turn_id")))
        if channel is None:
            return
        channel.put_nowait((kind, payload))

    def _deliver_notice(self, payload: Mapping[str, Any]) -> None:
        """Admit one turn-less notice, or record why it cannot be admitted."""
        raw = payload.get("event")
        if not isinstance(raw, Mapping):
            self._record_diagnostic("a notice frame carries no event", None)
            return
        try:
            event = decode_notice(raw)
        except AgentWireError as exc:
            self._record_diagnostic(f"a notice frame cannot be admitted: {exc}", None)
            return
        self._notices.put_nowait(event)

    def _settle_request(self, message: Mapping[str, Any]) -> None:
        request_id = str(message.get("id"))
        future = self._pending.get(request_id)
        if future is None or future.done():
            # A late reply never re-settles anything, and it is no longer
            # dropped silently: the record names the id it could not answer.
            self._record_diagnostic(
                "the reply answers no request this end is still waiting for",
                message.get("id"),
            )
            return
        error = message.get("error")
        if isinstance(error, Mapping):
            self._consecutive_timeouts = 0
            future.set_exception(
                RootAgentRequestError(
                    str(error.get("code") or "AgentError"),
                    str(error.get("message") or "agent request failed"),
                )
            )
            return
        # Any answer proves the root is serving, so the timeout streak ends:
        # the idle wedge is repeated timeouts with nothing answered between.
        self._consecutive_timeouts = 0
        result = message.get("result")
        future.set_result(result if isinstance(result, dict) else {})

    async def _request(
        self, method: str, params: dict[str, Any], *, timeout: float | None = None
    ) -> dict[str, Any]:
        request_id = self._begin_request(params)
        return await self._await_request(request_id, method, params, timeout=timeout)

    def _begin_request(
        self, params: Mapping[str, Any], *, long_method: str | None = None
    ) -> str:
        """Mint one coordinator request id, recording its absolute deadline.

        The id is minted separately from the wait so a caller whose reply may
        never arrive still correlates: the one-shot marker re-check keys on the
        id it sent, not on whatever the pipe returns. ``long_method`` records a
        long operation as in flight for as long as this end waits on it, which
        is what refuses a short reconfigure mode before a frame is sent.
        """
        if self._closed:
            raise RootAgentProcessError("the root agent process is closed")
        if self._death is not None:
            raise RootAgentExitedError(
                self._death.reason, exit_code=self._death.exit_code
            )
        request_id = coordinator_request_id(next(self._request_ids))
        self._last_request_id = request_id
        deadline = params.get("deadline")
        if isinstance(deadline, int | float) and not isinstance(deadline, bool):
            # The coordinator minted this deadline: keep it so the one-shot
            # marker re-check can judge a payload's construction against it.
            self._operation_deadlines[request_id] = float(deadline)
        if long_method is not None:
            self._long_inflight[request_id] = long_method
        return request_id

    async def _await_request(
        self,
        request_id: str,
        method: str,
        params: dict[str, Any],
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        future: asyncio.Future[dict[str, Any]] = (
            asyncio.get_running_loop().create_future()
        )
        self._pending[request_id] = future
        effective = timeout if timeout is not None else self._request_timeout
        try:
            await self._transport.send({
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params,
            })
            return await asyncio.wait_for(future, timeout=effective)
        except TimeoutError as exc:
            # A live root that did not answer inside its deadline is the op
            # signature; the same thing repeating with the watcher still alive
            # and no op in flight is the idle-wedge residual, never claimed as
            # an abort.
            self._consecutive_timeouts += 1
            signature = (
                RootFailureSignature.IDLE_WEDGE
                if self._consecutive_timeouts >= IDLE_WEDGE_TIMEOUT_STREAK
                else RootFailureSignature.OP_DEADLINE
            )
            self._record_diagnostic(
                str(
                    RootAgentOpTimeoutError(
                        method,
                        effective,
                        signature=signature,
                        occurrences=self._consecutive_timeouts,
                    )
                ),
                request_id,
            )
            raise RootAgentOpTimeoutError(
                method,
                effective,
                signature=signature,
                occurrences=self._consecutive_timeouts,
            ) from exc
        finally:
            self._pending.pop(request_id, None)
            self._long_inflight.pop(request_id, None)

    async def _send(self, message: dict[str, Any]) -> None:
        await self._transport.send(message)

    async def _notify(self, method: str, params: dict[str, Any]) -> None:
        await self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def _send_soon(self, frame: dict[str, Any]) -> None:
        """Send one frame from a synchronous dispatch, tracked for shutdown."""
        task = asyncio.create_task(self._transport.send(frame))
        self._notifications.add(task)
        task.add_done_callback(self._notifications.discard)

    def _schedule_notification(self, method: str, params: dict[str, Any]) -> None:
        if self._closed:
            raise RootAgentProcessError("the root agent process is closed")
        self._send_soon({"jsonrpc": "2.0", "method": method, "params": params})

    def _page_reader(
        self, agent_id: str, before: str | None, limit: int
    ) -> Awaitable[AgentTranscriptGetResponse]:
        return self.read_page(agent_id, before=before, limit=limit)

    async def _finish_turn(self, payload: Mapping[str, Any]) -> None:
        # The terminal frame is retained: a child run's completion source reads
        # the failure evidence the child derived, before the turn raises.
        turn_id = payload.get("turn_id")
        if isinstance(turn_id, str):
            self._turn_terminals[turn_id] = dict(payload)
        outcome = payload.get("outcome")
        if outcome == TURN_OUTCOME_COMPLETED:
            return
        if outcome == TURN_OUTCOME_INTERRUPTED:
            raise asyncio.CancelledError()
        if outcome == TURN_OUTCOME_FAILED:
            if self._death is not None:
                raise RootAgentExitedError(
                    self._death.reason, exit_code=self._death.exit_code
                )
            raise AgentProcessTurnFailed(
                str(payload.get("error") or "the agent turn failed")
            )
        raise RootAgentProcessError(f"unknown turn outcome {outcome!r}")

    def _fail_requests(self, reason: str) -> None:
        for future in self._pending.values():
            if not future.done():
                future.set_exception(RootAgentProcessError(reason))

    def _fail_turns(self, reason: str) -> None:
        for turn_id, channel in self._turns.items():
            channel.put_nowait((
                "ended",
                {"turn_id": turn_id, "outcome": TURN_OUTCOME_FAILED, "error": reason},
            ))


def _mapping(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RootAgentProcessError("agent frame payload is not an object")
    return value


def _payload(message: Mapping[str, Any]) -> dict[str, Any]:
    params = message.get("params")
    return params if isinstance(params, dict) else {}


def _invoke_context(context: Mapping[str, Any]) -> InvokeContext:
    """Rebuild the portable context of one host subagent run.

    Only the identity and location fields crossed the pipe; the coordinator
    supplies its own collaborators, so the runner sees the same ``tool_call_id``
    and session identity the agent's tool call carried.
    """
    session_dir = context.get("session_dir")
    scratchpad_dir = context.get("scratchpad_dir")
    session_id = context.get("session_id")
    return InvokeContext(
        tool_call_id=str(context.get("tool_call_id") or ""),
        session_id=session_id if isinstance(session_id, str) else None,
        session_dir=Path(session_dir) if isinstance(session_dir, str) else None,
        scratchpad_dir=Path(scratchpad_dir)
        if isinstance(scratchpad_dir, str)
        else None,
        is_subagent=bool(context.get("is_subagent")),
    )


async def _tool_io_call(
    tool_io: ToolIOPort, op: str, arguments: Mapping[str, Any]
) -> object:
    """Perform one decoded tool I/O operation on the host's port."""
    if op == "read_lines":
        return await tool_io.read_lines(
            Path(str(arguments.get("path") or "")),
            start_line=int(arguments.get("start_line") or 1),
            limit=int(arguments.get("limit") or 0),
            max_bytes=int(arguments.get("max_bytes") or 0),
        )
    if op == "read_text":
        return await tool_io.read_text(Path(str(arguments.get("path") or "")))
    if op == "write_text":
        newline = arguments.get("newline")
        await tool_io.write_text(
            Path(str(arguments.get("path") or "")),
            str(arguments.get("content") or ""),
            encoding=str(arguments.get("encoding") or "utf-8"),
            newline=newline if isinstance(newline, str) else None,
        )
        return None
    if op == "run_shell":
        request = arguments.get("request")
        if not isinstance(request, Mapping):
            raise UnsupportedProcessTurnRequest(
                "a run_shell tool I/O request carries no shell command"
            )
        return await tool_io.run_shell(decode_shell_command_request(request))
    raise UnsupportedProcessTurnRequest(f"tool I/O op {op!r} is not served")


def root_agent_command() -> list[str]:
    """The module invocation the coordinator spawns as the root agent."""
    return [sys.executable, "-m", ROOT_AGENT_MODULE]


def multi_process_root_selected(environ: Mapping[str, str] | None = None) -> bool:
    """Whether this run selects the multi-process (stdio) route."""
    source = os.environ if environ is None else environ
    selector = source.get(TRANSPORT_ENV_VAR, "").strip().lower()
    return selector in MULTI_PROCESS_TRANSPORT_SELECTORS


async def spawn_root_agent_process(  # noqa: PLR0913 - the spawn surface
    *,
    session_options: BaseModel,
    client_info: BaseModel,
    client_capabilities: BaseModel | None = None,
    manifest: RegistryManifest | None = None,
    agent_id: str | None = None,
    connection_id: str = _DEFAULT_CONNECTION_ID,
    session_id: str | None = None,
    continue_latest: bool = False,
    env: Mapping[str, str] | None = None,
    cwd: str | os.PathLike[str] | None = None,
    command: Sequence[str] | None = None,
    ready_timeout: float = DEFAULT_SPAWN_READY_TIMEOUT,
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
    heartbeat_interval: float = DEFAULT_ROOT_HEARTBEAT_INTERVAL,
    budget: ProcessTreeBudget | None = None,
) -> RootAgentProcess:
    """Spawn the root agent, open its session, and return its live handle.

    The caller supplies the multi-process (stdio) selector in ``env``; a run
    that selects ``in_process`` (or names no transport) keeps its agent in
    this process and must not call this function. ``budget`` is the tree
    budget of the coordinator that owns this process, so the root agent and
    every child it asks for draw on ONE hard cap. ``session_id`` and
    ``continue_latest`` select the persisted-session resume path (a fresh
    process resuming a stored session), which is distinct from live re-attach.
    """
    source = os.environ if env is None else env
    if not multi_process_root_selected(source):
        raise RootAgentSelectionError(
            f"{TRANSPORT_ENV_VAR} does not select the multi-process (stdio) "
            "root agent; the in-process route stays selected"
        )
    # The root agent reads no config, so the parent's one startup executor
    # resolution travels in the spawn env: every shell the child spawns (the
    # bash tool, background jobs, completion checks, the app-server shell)
    # uses the same binary the coordinator resolved.
    spawn_env = publish_shell_executor(source)
    spawned = await spawn_process(
        command or root_agent_command(),
        env=spawn_env,
        cwd=cwd,
        ready_timeout=ready_timeout,
        budget=budget,
    )
    process = RootAgentProcess(
        spawned,
        connection_id=connection_id,
        request_timeout=request_timeout,
        heartbeat_interval=heartbeat_interval,
    )
    try:
        reply = await process.open(
            session_options=session_options,
            client_info=client_info,
            client_capabilities=client_capabilities,
            session_id=session_id,
            continue_latest=continue_latest,
        )
    except BaseException:
        await process.close()
        raise
    if manifest is not None:
        process.register_live_route(manifest, agent_id or reply.session_id)
    return process


__all__ = [
    "AgentProcessTurnFailed",
    "CompactSettlementRecord",
    "RootAgentDeferredOperation",
    "RootAgentExitedError",
    "RootAgentOpTimeoutError",
    "RootAgentProcess",
    "RootAgentProcessError",
    "RootAgentReplaceConflict",
    "RootAgentRequestError",
    "RootAgentSelectionError",
    "RootAgentSessionNotFound",
    "RootDeath",
    "RootFailureSignature",
    "UnsupportedProcessTurnRequest",
    "multi_process_root_selected",
    "root_agent_command",
    "spawn_root_agent_process",
]
