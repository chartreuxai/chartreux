"""Root agent process: one agent loop served over its own standard pipes.

The coordinator spawns this module as a child process. The child announces
readiness with the WP1 bootstrap frame, opens a session through the
agent-side execution port, and then serves turn requests: turn events stream
back as they are produced, the loop records them into the session transcript
through its own session logger as it runs, and transcript pages are served
from the live in-memory view. No transcript file is read for interactive
paging, and no coordinator state is mirrored here.

The loop is reached through the same :class:`TurnExecutionPort` the in-process
escape path uses, so the agent components are identical in both topologies.
"""

from __future__ import annotations

import asyncio
from collections.abc import (
    AsyncGenerator,
    Awaitable,
    Callable,
    Coroutine,
    Mapping,
    Sequence,
)
from contextlib import aclosing, suppress
from dataclasses import dataclass, replace
import itertools
from pathlib import Path
import re
import sys
import time
from typing import TYPE_CHECKING, Any, Protocol, cast
from uuid import uuid4

if TYPE_CHECKING:
    from chartreux.app_server._connection_protocol import ClientCapabilities, ClientInfo

from chartreux.app_server._agent_transcript import (
    read_agent_transcript,
    read_live_agent_transcript,
)
from chartreux.app_server._dispatch import RequestFailure
from chartreux.app_server._execution import SessionExecution, SessionExecutionConflict
from chartreux.app_server._policy_tree import PolicyNodeService
from chartreux.app_server._projection import project_history
from chartreux.app_server._resources import ResourceRequestHandler
from chartreux.app_server._review import ReviewRequestHandler
from chartreux.app_server._root_recovery import transcript_integrity_verdict
from chartreux.app_server._session_backend_port import SessionBackendError
from chartreux.app_server._shell import ShellController
from chartreux.app_server._shell_record import (
    ManualShellOutcome,
    build_manual_shell_record,
)
from chartreux.app_server._shell_requests import _manual_shell_outcome
from chartreux.app_server._state import history_page
from chartreux.app_server._subagent_authority import DangerousCommandAuthority
from chartreux.app_server._turns import _ROOT_EFFECT_SETTLE_GRACE_SECONDS
from chartreux.app_server._utils import now_ms
from chartreux.app_server._workspace import prepare_prompt
from chartreux.app_server.agent_admission import (
    OP_OUTCOME_COMPLETED,
    LongOperationFence,
    OperationMarker,
    OperationMarkerTable,
)
from chartreux.app_server.agent_reconfigure import AgentReconfigureService
from chartreux.app_server.agent_snapshot import (
    SNAPSHOT_HISTORY_LIMIT,
    AgentSessionSnapshot,
    RootOpenAccounting,
    build_open_reply,
    build_session_snapshot,
    owned_user_source,
    session_availability,
    session_dirs,
    session_roots_revision,
)
from chartreux.app_server.agent_steering import AgentSteeringArbiter
from chartreux.app_server.agent_wire import (
    AGENT_AGENTS_CANCEL_METHOD,
    AGENT_AGENTS_INVENTORY_READ_METHOD,
    AGENT_AGENTS_METHODS,
    AGENT_AGENTS_RELEASE_METHOD,
    AGENT_AGENTS_RESUME_METHOD,
    AGENT_AGENTS_UPDATE_METHOD,
    AGENT_AUTHORITY_ALLOW,
    AGENT_AUTHORITY_CHANGED_METHOD,
    AGENT_AUTHORITY_CHECK_METHOD,
    AGENT_AUTHORITY_DECISION_KIND_TOOL_AVAILABILITY,
    AGENT_AUTHORITY_DECISION_KIND_TOOL_INVOCATION,
    AGENT_AUTHORITY_DENY,
    AGENT_CATALOG_PUBLIC_METHODS,
    AGENT_CATALOG_READ_METHODS,
    AGENT_CHILD_TEARDOWN_METHOD,
    AGENT_CHILD_TRANSCRIPT_PAGE_METHOD,
    AGENT_COMPACT_METHOD,
    AGENT_CONFIG_PROXY_WRITE_METHOD,
    AGENT_CONFIG_PUBLIC_METHODS,
    AGENT_CONFIG_SURFACE_METHODS,
    AGENT_DEFERRED_SESSION_METHODS,
    AGENT_DIAGNOSTICS_METHODS,
    AGENT_DIAGNOSTICS_PUBLIC_METHODS,
    AGENT_INJECT_METHOD,
    AGENT_LOOPS_LIST_METHOD,
    AGENT_LOOPS_METHODS,
    AGENT_LOOPS_PUBLIC_METHODS,
    AGENT_MCP_AUTH_REQUIRED_METHOD,
    AGENT_MCP_AUTH_URL_METHOD,
    AGENT_MCP_CATALOG_ADD_METHOD,
    AGENT_MCP_CATALOG_LOGIN_METHOD,
    AGENT_MCP_CATALOG_LOGOUT_METHOD,
    AGENT_MCP_CATALOG_METHODS,
    AGENT_MCP_CATALOG_PUBLIC_METHODS,
    AGENT_MCP_CATALOG_READ_METHOD,
    AGENT_MCP_CATALOG_REFRESH_METHOD,
    AGENT_MCP_CATALOG_REMOVE_METHOD,
    AGENT_MCP_CATALOG_TOGGLE_METHOD,
    AGENT_MCP_OWNER_REKEY_METHOD,
    AGENT_MCP_OWNER_RELEASE_METHOD,
    AGENT_MCP_RELOAD_FAIL_METHOD,
    AGENT_MCP_RELOAD_FINISH_METHOD,
    AGENT_MCP_RELOAD_PREPARE_METHOD,
    AGENT_NOTICE_METHOD,
    AGENT_OP_ADMISSION_METHOD,
    AGENT_OP_CANCEL_METHOD,
    AGENT_OPEN_METHOD,
    AGENT_ORIGIN,
    AGENT_POLICY_DANGEROUS_COMMANDS_METHODS,
    AGENT_POLICY_DANGEROUS_COMMANDS_READ_METHOD,
    AGENT_RECEIVING_NOTIFICATIONS,
    AGENT_RECEIVING_REQUESTS,
    AGENT_RECOVERY_EVIDENCE_METHOD,
    AGENT_REVIEW_METHODS,
    AGENT_REVIEW_PUBLIC_METHODS,
    AGENT_RUN_COMPLETED_METHOD,
    AGENT_RUN_COMPLETIONS_READ_METHOD,
    AGENT_SEQUENCE_FIELD,
    AGENT_SESSION_CONFIG_RELOAD_METHOD,
    AGENT_SESSION_HISTORY_LIST_METHOD,
    AGENT_SESSION_LIST_METHOD,
    AGENT_SESSION_PLAN_MODE_PRESENT_METHOD,
    AGENT_SESSION_POLICY_READ_METHOD,
    AGENT_SESSION_RECONFIGURE_METHOD,
    AGENT_SESSION_RENAME_METHOD,
    AGENT_SESSION_REPLACE_METHOD,
    AGENT_SESSION_ROOTS_READ_METHOD,
    AGENT_SESSION_ROOTS_SAVE_METHOD,
    AGENT_SESSION_SAFETY_READ_METHOD,
    AGENT_SESSION_SAFETY_TRANSITION_METHOD,
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
    AGENT_WORKSPACE_PROMPT_PREPARE_METHOD,
    CHILD_RECEIVING_NOTIFICATIONS,
    COMPACT_SETTLEMENT_CANCELLED_UNPUBLISHED,
    COMPACT_SETTLEMENT_PUBLISHED_SETTLED,
    COMPACT_SETTLEMENT_PUBLISHED_UNSETTLED,
    COORDINATOR_ORIGIN,
    HOST_SUBAGENT_ENDED_METHOD,
    HOST_SUBAGENT_EVENT_METHOD,
    HOST_TOOL_IO_METHOD,
    PARENT_INCARNATION_PAYLOAD_KEY,
    POLICY_METHODS,
    RECONFIGURE_MODE_CONFIG_WRITE,
    RECONFIGURE_MODE_HISTORY_CLEAR,
    RECONFIGURE_MODE_SETTINGS,
    SUBAGENT_RUN_METHOD,
    TURN_OUTCOME_COMPLETED,
    TURN_OUTCOME_FAILED,
    TURN_OUTCOME_INTERRUPTED,
    AgentAuthorityChanged,
    AgentAuthorityCheck,
    AgentAuthorityDecision,
    AgentProtocolRefusal,
    AgentReconfigureConflictError,
    AgentRecoveryEvidence,
    AgentRunCompletion,
    AgentSessionNotFoundError,
    AgentShellRefusedError,
    AgentWireError,
    CompactRequest,
    CompactSettlementReply,
    ReconfigureRequest,
    RemoteUserInputRejectedError,
    RemoteWireRequestError,
    RenameReply,
    ReplaceReply,
    ReplaceRequest,
    ToolIOCapabilities,
    UnsupportedAgentRequestError,
    WireDiagnostic,
    WireMethodNotSupportedError,
    WireTransportClosedError,
    agent_request_id,
    decode_authority_check,
    decode_compact_request,
    decode_event,
    decode_host_run_end,
    decode_host_run_frame,
    decode_host_service_offers,
    decode_inject_request,
    decode_open_request,
    decode_reconfigure_request,
    decode_rename_request,
    decode_replace_request,
    decode_shell_interrupt,
    decode_shell_run_request,
    decode_steer_request,
    decode_tool_io_result,
    decode_turn_request,
    decode_user_input_result,
    deferred_session_message,
    encode_authority_changed,
    encode_authority_decision,
    encode_availability,
    encode_compact_reply,
    encode_event,
    encode_inject_reply,
    encode_notice,
    encode_open_reply,
    encode_operation_marker,
    encode_parent_incarnation,
    encode_reconfigure_reply,
    encode_rename_reply,
    encode_replace_reply,
    encode_retry_reason,
    encode_run_completion,
    encode_run_completions,
    encode_session_snapshot,
    encode_shell_command_request,
    encode_shell_interrupt_reply,
    encode_shell_run_reply,
    encode_steer_settlement,
    encode_subagent_run_request,
    encode_tool_io_request,
    encode_transcript_page,
    encode_user_input_settlement,
    event_live_turns_used,
    implementation_identity,
    request_id_origin,
)
from chartreux.app_server.config import stated_compact_threshold
from chartreux.app_server.models import PublicRecoveryEligibility
from chartreux.app_server.process_bootstrap import (
    ADVERTISED_CAPABILITIES,
    AGENT_WIRE_CAPABILITY_LIVE_CHILD_AUTHORITY,
    bootstrap_requested,
    write_ready_frame,
)
from chartreux.app_server.protocol import (
    AgentsCancelParams,
    AgentsCancelResponse,
    AgentsInventoryReadParams,
    AgentsInventoryReadResponse,
    AgentsReconcileParams,
    AgentsReconcileResponse,
    AgentsReleaseParams,
    AgentsReleaseResponse,
    AgentsResumeParams,
    AgentsResumeResponse,
    AgentSummaryModel,
    AgentTranscriptGetParams,
    AgentTranscriptGetResponse,
    AgentTranscriptState,
    CancelOutcome as PublicCancelOutcome,
    DangerousCommandAuthorityReadParams,
    DangerousCommandRevokeParams,
    MCPAddParams,
    MCPAuthRequiredParams,
    MCPLoginParams,
    MCPLogoutParams,
    MCPReadParams,
    MCPRefreshParams,
    MCPRemoveParams,
    MCPToggleParams,
    PolicyReadParams,
    ProtocolErrorCode,
    ReconcileOutcome as PublicReconcileOutcome,
    ReleaseOutcome as PublicReleaseOutcome,
    ResumeOutcome as PublicResumeOutcome,
    RootsReadParams,
    RootsReadResponse,
    RootsSaveParams,
    RootsSaveResponse,
    RunStopReason as PublicRunStopReason,
    SessionHistoryListParams,
    SessionHistoryListResponse,
    SessionListParams,
    SessionSafetyReadParams,
    SessionSafetyTransitionParams,
    ShellRunParams,
    ShellRunResponse,
    WorkspacePromptPrepareParams,
    WorkspacePromptPrepareResponse,
)
from chartreux.app_server.subagent_ports import (
    ChildPublicationTeardownPort,
    ChildTeardownRequest,
    SubagentAdmission,
)
from chartreux.app_server.transport import (
    BinaryLineReader,
    BinaryLineWriter,
    JsonRpcTransport,
    StdioJsonRpcTransport,
)
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.agent_loop._execution_port import ExecutionTurnRequest
from chartreux.core.agent_loop._loop_types import AgentTurnOptions
from chartreux.core.agent_loop._request_broker import UserInputSettlement
from chartreux.core.agent_loop._safety_authority import AuthorityActor, SafetyLevel
from chartreux.core.config._root_persistence import read_saved_roots
from chartreux.core.config.harness_files import init_harness_files_manager
from chartreux.core.dangerous_commands import (
    OPERATION_REGISTRY,
    REMOTE_EXEC_OPERATION_ID,
    canonical_remote_destination,
)
from chartreux.core.events import (
    AssistantEvent,
    BaseEvent,
    ToolStreamEvent,
    UserInputRequestEvent,
)
from chartreux.core.llm_models import LLMMessage
from chartreux.core.paths import LITERAL_LOG_FILE, LOG_FILE
from chartreux.core.recovery_models import (
    FailureClass,
    RootTurnFailure,
    build_root_turn_failure,
)
from chartreux.core.recovery_observation import (
    observe_exception,
    record_observation,
    turn_observation,
)
from chartreux.core.session.session_loader import SessionLoader
from chartreux.core.subagents import (
    AgentAvailability,
    AgentBusyError,
    AgentClosedError,
    AgentEvictedError,
    AgentResultExpiredError,
    AgentSummary,
    CancelOutcome,
    CancelResult,
    EffectVerification,
    LaunchOutcome,
    ReconcileOutcome,
    ReconcileResult,
    ReleaseAgentOutcome,
    ResumeAgentResult,
    ResumeOutcome,
    RunStatus,
    RunStopReason,
    SubagentRunnerPort,
    TaskArgs,
    TaskResult,
    UnknownAgentError,
    WaitForAgentResult,
    WaitStatus,
    derive_agent_name,
    normalize_task_summary,
    still_running_guidance,
)
from chartreux.core.tools import secret_redaction
from chartreux.core.tools.base import InvokeContext
from chartreux.core.tools.builtins.bash import BashToolConfig
from chartreux.core.tools.classification import shipped_builtin_name
from chartreux.core.tools.io_port import (
    ShellCommandRequest,
    ShellCommandResult,
    ToolIOPort,
)
from chartreux.core.tools.models import ToolPermission
from chartreux.core.tools.utils import instruction_read_ceiling, scratchpad_ceiling
from chartreux.core.utils.retry import RetryObserver, RetryReason
from chartreux.model_display import format_model_display_name
from chartreux.observability.logging import init_file_logging, logger
from chartreux.utils.io import BoundedReadResult, ReadSafeResult

ROOT_AGENT_MODULE = "chartreux.app_server.root_agent"


class RootLoopFactory(Protocol):
    """Builds the loop an ``agent/open`` request asks for, and owns its process.

    A factory may also expose ``open_accounting``: the accounting identity the
    open it just served adopted, for the reply's replay records. A factory that
    can rebind its loop to another stored session exposes ``replace(loop,
    session_id=..., continue_latest=...)``; one that cannot leaves the identity
    replacement typed-refused instead of silently rebinding nothing.
    """

    async def open(self, params: Mapping[str, Any]) -> AgentLoop: ...

    async def close(self) -> None: ...


class HarnessRootLoopFactory:
    """The production factory: one harness process owns one root loop."""

    def __init__(self) -> None:
        from chartreux.app_server._runtime import HarnessProcess

        self._process = HarnessProcess()
        self.open_accounting: RootOpenAccounting | None = None
        # The children this root owns, built with the loop they launch from:
        # the multi-process route gives every child its own process, the
        # in-process escape leaves every child in the coordinator.
        self.child_host: AgentChildHost | None = None

    async def open(self, params: Mapping[str, Any]) -> AgentLoop:
        from chartreux.app_server._connection_protocol import (
            ClientCapabilities,
            ClientInfo,
        )
        from chartreux.app_server._runtime import RootOpenRequest
        from chartreux.app_server.protocol import AgentConfig

        request = decode_open_request(params)
        options = AgentConfig.model_validate(request.agent_config)
        client_info = ClientInfo.model_validate(request.client_info)
        capabilities = ClientCapabilities.model_validate(request.client_capabilities)
        loop = await self._process.open_root(
            RootOpenRequest(
                options=options,
                client_info=client_info,
                client_capabilities=capabilities,
                session_id=request.session_id,
                continue_latest=request.continue_latest,
            )
        )
        self.open_accounting = self._process.last_open_accounting
        # The validated client identity of THIS root's open rides every
        # process spawn's payload: the host owns the spawn, so it owns the
        # identity the child re-validates at its own open.
        self.child_host = agent_child_host(
            loop, client_info=client_info, client_capabilities=capabilities
        )
        return loop

    async def replace(
        self, loop: AgentLoop, *, session_id: str | None, continue_latest: bool = False
    ) -> None:
        """Rebind this process's root loop to another stored session."""
        from chartreux.app_server._runtime import RuntimeSessionNotFoundError

        try:
            await self._process.rebind_root(
                loop, session_id=session_id, continue_latest=continue_latest
            )
        except RuntimeSessionNotFoundError as exc:
            raise AgentSessionNotFoundError(str(exc)) from exc
        self.open_accounting = RootOpenAccounting.from_loop(loop)

    async def close(self) -> None:
        await self._process.close()


_AGENTS_UPDATE_RETRY_LIMIT = 2
"""How many consecutive failed deliveries one pass may retry (G6a).

A mutation that races a dying emission only re-arms the dirty flag, so the
pass re-schedules once its failed attempt has finished; this budget keeps a
persistently broken pipe from spinning -- the retry chain stops here and the
dirty flag waits for the next real mutation, which is the primary trigger.
"""

_AUTHORITY_CHANGED_RETRY_LIMIT = 2
"""How many consecutive failed authority/change deliveries one pass may retry.

The same bounded chain the inventory pass uses: an invalidation that raced a
dying delivery gets its own pass, and a persistently broken pipe stops the
chain with the dirty flag armed for the next policy commit.
"""

_AUTHORITY_CHECK_RESTARTS = 3
"""How many evaluations one decision may take while policy commits interleave.

The decision reads the revision vector, evaluates, and re-reads it: a commit
that lands in between makes the evaluation stale, so it is redone on the
committed policy. A revision that will not stand still for one evaluation is
not decided on -- the answer is a deny, never a decision on a stale vector.
"""


def _authority_request_sequence(request_id: str) -> int | None:
    """The agent-origin sequence number of one authority request id, or None."""
    if request_id_origin(request_id) != AGENT_ORIGIN:
        return None
    # ``agent_request_id`` mints ``{origin}:{sequence}``, so the agent-origin
    # prefix is what separates the sequence from the namespace it belongs to.
    suffix = request_id.removeprefix(f"{AGENT_ORIGIN}:")
    return int(suffix) if suffix.isdigit() else None


@dataclass(frozen=True, slots=True)
class _ChildAuthorityRecord:
    """What the fence last accepted from one child agent's lineage."""

    generation: int
    epoch: str
    request_sequence: int
    revision: dict[str, int]


class _ChildAuthorityFence:
    """Fence child authority requests by epoch, generation, request and vector.

    A request is admissible only when it names this owner's session and
    incarnation, comes from a generation that is not older than the one this
    child already reached, keeps the epoch that generation minted, advances the
    agent-origin request sequence, and presents a revision vector that is
    componentwise non-decreasing against the last accepted one. Everything
    else is refused before any policy is read: a superseded epoch, a replayed
    or reordered request, and a rolled-back vector are all indistinguishable
    from an attempt to decide on stale authority.

    A new generation is a fresh incarnation and takes a fresh epoch; reusing
    the superseded epoch is refused, so an old incarnation can never present
    itself as the new one. No wall-clock value takes part in any of it.
    """

    def __init__(self) -> None:
        self._records: dict[str, _ChildAuthorityRecord] = {}

    def admit(
        self, request: AgentAuthorityCheck, *, session_id: str | None, incarnation: str
    ) -> str | None:
        """The typed refusal for one request, or ``None`` when it is admissible."""
        if session_id is None:
            return "this agent holds no open session to decide for"
        if request.parent_session_id != session_id:
            return "the request names a session this agent does not own"
        if request.parent_incarnation != incarnation:
            return "the request names a superseded owner incarnation"
        sequence = _authority_request_sequence(request.request_id)
        if sequence is None:
            return "the request id is not an agent-origin request id"
        record = self._records.get(request.child_agent_id)
        if record is not None:
            refusal = self._admit_against(record, request, sequence)
            if refusal is not None:
                return refusal
        self._records[request.child_agent_id] = _ChildAuthorityRecord(
            generation=request.child_generation,
            epoch=request.epoch,
            request_sequence=sequence,
            revision=dict(request.revision),
        )
        return None

    @staticmethod
    def _admit_against(  # noqa: PLR0911 - one refusal per fence rule
        record: _ChildAuthorityRecord, request: AgentAuthorityCheck, sequence: int
    ) -> str | None:
        if request.child_generation < record.generation:
            return "the request names a superseded child incarnation"
        if request.child_generation == record.generation:
            if request.epoch != record.epoch:
                return "the request names an epoch this generation never minted"
            if sequence <= record.request_sequence:
                return "the request is late or replayed for this incarnation"
            rolled_back = [
                component
                for component, value in request.revision.items()
                if value < record.revision.get(component, 0)
            ]
            if rolled_back:
                return (
                    "the request presents a rolled-back authority revision "
                    f"({', '.join(sorted(rolled_back))})"
                )
            return None
        if request.epoch == record.epoch:
            return "a new child generation reuses the superseded epoch"
        return None


def _ancestor_authority_revision(manager: Any) -> int | None:
    """The monotone revision of every manager above *manager*, summed.

    A root owner has no ancestors and reports 0. Each manager's own
    authority generation only ever advances on an accepted change, so their
    sum is monotone too. A chain that names an ancestor but cannot produce it
    reports ``None`` -- an unreadable chain is never treated as a root.
    """
    total = 0
    seen: set[int] = set()
    try:
        cursor = manager
        while cursor._parent_authority_getter is not None:
            parent = cursor._parent_authority()
            if parent is None or id(parent) in seen:
                return None
            seen.add(id(parent))
            total += int(parent._authority_generation)
            cursor = parent
    except Exception:
        return None
    return total


class RootAgentServer:
    """Serve one agent loop over a JSON-RPC transport."""

    def __init__(
        self, transport: JsonRpcTransport, *, factory: RootLoopFactory
    ) -> None:
        self._transport = transport
        self._factory = factory
        self._loop: AgentLoop | None = None
        self._active_turn_id: str | None = None
        self._active_turn_task: asyncio.Task[None] | None = None
        self._tasks: set[asyncio.Task[None]] = set()
        self._policy_node: PolicyNodeService | None = None
        self._steering_arbiter: AgentSteeringArbiter | None = None
        self._notices_task: asyncio.Task[None] | None = None
        self._diagnostics: list[WireDiagnostic] = []
        self._agent_request_ids = itertools.count(1)
        self._pending_agent_requests: dict[str, asyncio.Future[dict[str, Any]]] = {}
        # The agent's own event sequence: every event frame relayed to the
        # coordinator is stamped with the next value, and every snapshot
        # payload carries the value current at its build, so the coordinator
        # can order an adopted snapshot against the events it already relayed
        # (the shared agent-sequence cursor; its watermark cannot).
        self._event_sequence = 0
        self._host_runs: dict[str, asyncio.Queue[tuple[str, Mapping[str, Any]]]] = {}
        # The marker table is incarnation-scoped and dies with this process;
        # the fence admits one long operation at a time into it.
        self._markers = OperationMarkerTable(uuid4().hex)
        self._admitted_marker: OperationMarker | None = None
        self._long_ops = LongOperationFence(
            table=self._markers,
            announce=self._announce_admission,
            on_admitted=self._remember_admission,
        )
        # Item 1 child seam: the publication attempts this agent owns. The
        # coordinator keeps the attempt id, so a lost aggregate reply is still
        # resolvable through the teardown request; an agent that owns no
        # publication attempts refuses the request typed.
        self._publication_teardowns: ChildPublicationTeardownPort | None = None
        # The children this process owns, when its route gives them to it: the
        # turn's runner is then the agent's own, not a pipe proxy.
        self._child_host: AgentChildHost | None = None
        # The reconfigure modes mutate the session in place, so the mode bodies
        # that are not long operations serialize here instead of interleaving.
        self._reconfigure_lock = asyncio.Lock()
        # The resource handler answers from the loop it was built over, so it
        # is rebuilt whenever an identity flow swaps that loop out.
        self._resource_handler: ResourceRequestHandler | None = None
        self._resource_loop: AgentLoop | None = None
        self._review_handler: ReviewRequestHandler | None = None
        self._review_loop: AgentLoop | None = None
        # A manual `!` command runs here, where the loop's cwd and its tool
        # config live; one controller owns every run of this process so an
        # interrupt can reach the process it names.
        self._shell_controller: ShellController | None = None
        # The owner-side MCP catalog surface: this process owns the single
        # orchestrator, so every session-scoped mcp_catalog leg the
        # coordinator proxies runs here against the loop. The service is the
        # factory's harness process one when it exposes it (so the registry
        # armed at open resolves through it), else a process-local one.
        self._mcp_catalog_service: Any = None
        self._mcp_catalog_root: Any = None
        self._mcp_catalog_loop: AgentLoop | None = None
        # The typed owner key of the catalog request currently being served,
        # which the authorization-required sink reuses for its gating.
        self._current_owner_key: str = ""
        # The reload plans held agent-side between the prepare and the
        # finish/fail request of the same owner key.
        self._mcp_reload_plans: dict[str, Any] = {}
        # F10: the child-inventory emission state. The host's mutation seam
        # schedules one coalesced publication, exactly like the in-process
        # registry's serialized emission path -- at most one delivery is in
        # flight, and a change observed during a delivery re-arms the pass.
        self._agents_update_dirty = False
        self._agents_update_task: asyncio.Task[None] | None = None
        self._agents_update_failures = 0
        # G1b: the settled-run completions waiting for their own delivery. Each
        # run's completion is delivered once, in settle order (a completion is
        # not a presentation delta, so nothing coalesces it away); a failing
        # send keeps the entry for the next pass, and the host's ledger is the
        # backstop either way.
        self._run_completions: list[AgentRunCompletion] = []
        self._run_completion_task: asyncio.Task[None] | None = None
        # G5: live child authority. This process is the IMMEDIATE parent of the
        # children it owns, so it is the only peer that may answer an
        # ``agent/authority/check``: the fence refuses a superseded epoch, a
        # replayed or reordered request and a rolled-back revision vector
        # before any policy is read, and the decision itself is taken from the
        # loop's CURRENT policy at request time. The incarnation id is fresh
        # for this process and never reused, so a reply minted by a previous
        # owner incarnation is recognisably stale.
        self._authority_incarnation = uuid4().hex
        self._authority_fence = _ChildAuthorityFence()
        self._published_authority_revision: dict[str, int] | None = None
        self._authority_changed_dirty = False
        self._authority_changed_task: asyncio.Task[None] | None = None
        self._authority_changed_failures = 0

    def install_publication_teardowns(self, port: ChildPublicationTeardownPort) -> None:
        """Serve teardown requests for the child publications this agent owns."""
        self._publication_teardowns = port

    def install_child_host(self, host: Any) -> None:
        """Own this root's children here, and serve their attempts' teardowns.

        The host is what makes the turn offer's ``subagent_runner`` real in a
        process-owned root: the run itself is the agent's own (the launch
        authorities live here), and the attempts it leaves are settled through
        the coordinator's ``agent/child/teardown`` request.
        """
        self._child_host = host
        self.install_publication_teardowns(host)
        # F10: this process owns the children, so it also owns their lifecycle
        # emission -- the host's every inventory mutation schedules one
        # coalesced agents/update notification onto the pipe router, so the
        # coordinator can relay it to the client exactly as the in-process
        # registry emits its own inventory.
        host.bind_inventory_publisher(self._schedule_agents_update)
        # G1b: the same ownership makes this process the only source of a
        # finished run's outcome, so each settled background run also hands
        # its typed completion over for delivery to the coordinator's wake
        # owner.
        host.bind_completion_publisher(self._schedule_run_completion)
        # G5b: this process is the child's immediate parent and the only peer
        # that may answer its authority questions, so the host that mints the
        # child's payload and binds its wire client takes this server as its
        # owner. Without it a published child would carry no incarnation and
        # answer no decision.
        host.bind_authority_owner(self)

    @property
    def session_id(self) -> str | None:
        return self._loop.session_id if self._loop is not None else None

    @property
    def child_host(self) -> AgentChildHost | None:
        """The child host this process owns, when its route gives it children."""
        return self._child_host

    @property
    def authority_incarnation(self) -> str:
        """This owner incarnation's epoch: fresh per process, never reused."""
        return self._authority_incarnation

    @property
    def authority_revision(self) -> dict[str, int] | None:
        """The live authority revision vector, or ``None`` when unreadable."""
        return self._authority_revision_vector()

    @property
    def diagnostics(self) -> tuple[WireDiagnostic, ...]:
        """The wire-compat diagnostics this endpoint recorded, oldest first."""
        return tuple(self._diagnostics)

    async def serve(self) -> None:
        try:
            async for message in self._transport.messages():
                await self._dispatch(message)
        finally:
            await self._shutdown()

    async def _dispatch(self, message: Mapping[str, Any]) -> None:
        """Route one frame by method presence and the closed direction tables.

        A method-bearing frame with an id is a request; a method-bearing frame
        without one is a notification; a frame with no method answers an
        agent-origin request. Neither an unlisted method nor an uncorrelated
        answer is a session failure: the request shape is answered typed and the
        notification shape is recorded, so the peer never hangs and the loop
        keeps serving.
        """
        method = message.get("method")
        if isinstance(method, str):
            await self._dispatch_method(method, message)
            return
        if "id" in message:
            self._settle_agent_reply(message)
            return
        self._record_diagnostic(
            "the frame carries no method and answers no request", None
        )

    async def _dispatch_method(self, method: str, message: Mapping[str, Any]) -> None:
        if "id" in message:
            if method not in AGENT_RECEIVING_REQUESTS:
                await self._reject_request(message["id"], method)
                return
            self._track(self._serve_request(message["id"], method, message))
            return
        if method in AGENT_RECEIVING_NOTIFICATIONS:
            self._dispatch_notification(method, _params(message))
            return
        if method in CHILD_RECEIVING_NOTIFICATIONS:
            # G5b: the notification only an owner's per-child wire client sends
            # its child. The root agent process is no child's peer, so this
            # branch is reachable only in the spawned child.
            if method == AGENT_AUTHORITY_CHANGED_METHOD:
                self._receive_authority_changed(_params(message))
                return
            self._record_diagnostic(
                "the child table lists the method but no handler serves it",
                None,
                method=method,
            )
            return
        if method in AGENT_RECEIVING_REQUESTS:
            # A request with no id has no correlation to answer on: record it
            # rather than fail the session, and never execute it unanswered.
            self._record_diagnostic(
                "the request frame carries no id, so it cannot be answered",
                None,
                method=method,
            )
            return
        self._record_diagnostic(
            "the method is not in this direction's closed table", None, method=method
        )

    def _receive_authority_changed(self, params: Mapping[str, Any]) -> None:
        """Handle one owner invalidation delivered to this process (G5b).

        The base server has no live authority client: a root agent process is
        never a child, so an invalidation addressed to it is recorded. The
        child server overrides this to fence and apply the snapshot.
        """
        self._record_diagnostic(
            "this agent holds no live child-authority client",
            None,
            method=AGENT_AUTHORITY_CHANGED_METHOD,
        )

    def _dispatch_notification(  # noqa: PLR0911 - one branch per wire method
        self, method: str, params: Mapping[str, Any]
    ) -> None:
        if method == AGENT_TURN_START_METHOD:
            self._start_turn(params)
            return
        if method == AGENT_TURN_INTERRUPT_METHOD:
            self._interrupt(params)
            return
        if method == AGENT_OP_CANCEL_METHOD:
            self._cancel_operation(params)
            return
        if method == HOST_SUBAGENT_EVENT_METHOD:
            self._deliver_host_run_frame(params, "frame")
            return
        if method == HOST_SUBAGENT_ENDED_METHOD:
            self._deliver_host_run_frame(params, "ended")
            return
        if method == AGENT_TURN_RESOLVE_METHOD:
            self._settle_user_input(self._resolve_user_input, params)
            return
        if method == AGENT_TURN_REJECT_METHOD:
            self._settle_user_input(self._reject_user_input, params)
            return
        if method == AGENT_TURN_RELAY_METHOD:
            self._settle_user_input(self._relay_user_input, params)
            return
        # The table admitted the method, so only a table entry that lost its
        # handler reaches here: warn instead of failing the session.
        self._record_diagnostic(
            "the table lists the method but no handler serves it", None, method=method
        )

    async def _reject_request(self, request_id: object, method: str) -> None:
        """Answer an unlisted request typed; the peer settles instead of hanging."""
        error = WireMethodNotSupportedError(method, direction=COORDINATOR_ORIGIN)
        self._record_diagnostic(
            "the method is not in this direction's closed table",
            request_id,
            method=method,
        )
        await self._send({
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": type(error).__name__, "message": str(error)},
        })

    def _settle_agent_reply(self, message: Mapping[str, Any]) -> None:
        """Settle the agent-origin request a reply answers, or record it."""
        request_id = message.get("id")
        future = (
            self._pending_agent_requests.get(str(request_id))
            if request_id_origin(request_id) == AGENT_ORIGIN
            else None
        )
        if future is None or future.done():
            self._record_diagnostic(
                "the reply answers no request this end is waiting for", request_id
            )
            return
        self._pending_agent_requests.pop(str(request_id), None)
        error = message.get("error")
        if isinstance(error, Mapping):
            future.set_exception(
                RemoteWireRequestError(
                    str(error.get("code") or "AgentPeerError"),
                    str(error.get("message") or "the peer request failed"),
                )
            )
            return
        result = message.get("result")
        future.set_result(result if isinstance(result, dict) else {})

    def register_agent_request(self) -> tuple[str, asyncio.Future[dict[str, Any]]]:
        """Mint an agent-origin request id and register its reply correlation.

        The op-9 host-service channel is the first agent-origin emitter; the
        registry is what settles those requests typed when the pipe closes.
        """
        request_id = agent_request_id(next(self._agent_request_ids))
        future: asyncio.Future[dict[str, Any]] = (
            asyncio.get_running_loop().create_future()
        )
        self._pending_agent_requests[request_id] = future
        return request_id, future

    def _fail_pending_requests(self, reason: str) -> None:
        """Settle every request this end awaits: a closed pipe answers none."""
        pending, self._pending_agent_requests = self._pending_agent_requests, {}
        for future in pending.values():
            if not future.done():
                future.set_exception(WireTransportClosedError(reason))

    async def call_host(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Ask the coordinator for one host service over the same pipe.

        The request is agent-origin: its id comes from this end's namespace, so
        it can never collide with a coordinator-origin id. The reply settles by
        id; a closed pipe settles it typed (``WireTransportClosedError``) rather
        than leaving the agent's turn hanging.
        """
        request_id, future = self.register_agent_request()
        try:
            await self._send({
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params,
            })
            return await future
        finally:
            self._pending_agent_requests.pop(request_id, None)
            if future.done() and not future.cancelled():
                # Mark an abandoned reply retrieved: a wakeup that arrives after
                # the caller stopped waiting is a recorded fact, not a warning.
                future.exception()

    def open_host_run(
        self, run_id: str
    ) -> asyncio.Queue[tuple[str, Mapping[str, Any]]]:
        """Register the frame channel of one agent-origin host subagent run."""
        channel: asyncio.Queue[tuple[str, Mapping[str, Any]]] = asyncio.Queue()
        self._host_runs[run_id] = channel
        return channel

    def close_host_run(self, run_id: str) -> None:
        """Retire one host run's frame channel."""
        self._host_runs.pop(run_id, None)

    def _deliver_host_run_frame(self, params: Mapping[str, Any], kind: str) -> None:
        channel = self._host_runs.get(str(params.get("run_id")))
        if channel is None:
            self._record_diagnostic(
                "the host subagent frame names a run this end did not open", None
            )
            return
        channel.put_nowait((kind, params))

    def _fail_host_runs(self, reason: str) -> None:
        """Close every host run this end awaits: a closed pipe ends them all."""
        runs, self._host_runs = self._host_runs, {}
        for channel in runs.values():
            channel.put_nowait(("ended", {"result": None, "error": reason}))

    def _host_ports_for(
        self, params: Mapping[str, Any]
    ) -> tuple[SubagentRunnerPort | None, ToolIOPort | None]:
        """The agent-side proxies for the host ports a turn offer announced.

        The ports never cross the pipe: the offer names which ones the
        coordinator serves for this turn, and each proxy turns a port call into
        an agent-origin request.
        """
        offers = decode_host_service_offers(params)
        # A process-owned root runs its children itself: the offer is what
        # grants the turn its child capability, and the launch authorities it
        # would resolve against live in this process. An agent with no host of
        # its own still proxies the coordinator's port over the pipe.
        local = self._child_host.runner if self._child_host is not None else None
        runner = local or (WireSubagentRunner(self) if offers.subagent_runner else None)
        tool_io = (
            WireToolIO(self, offers.tool_io) if offers.tool_io is not None else None
        )
        return runner, tool_io

    def _record_diagnostic(
        self, reason: str, request_id: object, *, method: object = None
    ) -> None:
        diagnostic = WireDiagnostic(
            direction=COORDINATOR_ORIGIN,
            reason=reason,
            method=method,
            request_id=request_id,
        )
        self._diagnostics.append(diagnostic)
        logger.warning(diagnostic.message())

    async def _serve_request(
        self, request_id: object, method: str, message: Mapping[str, Any]
    ) -> None:
        try:
            result = await self._handle_request(method, _params(message), request_id)
        except BaseException as exc:
            await self._send({
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": _wire_error_code(exc), "message": str(exc)},
            })
        else:
            # G5: a served request may have committed policy (a reconfigure, a
            # config write, a roots save, a policy phase). The vector is read
            # fresh here, so an unchanged one publishes nothing and a change
            # coalesces into one invalidation notification.
            self._note_authority_committed()
            await self._send({"jsonrpc": "2.0", "id": request_id, "result": result})

    async def _handle_request(  # noqa: PLR0911, PLR0912, PLR0915 - one branch per wire method
        self, method: str, params: Mapping[str, Any], request_id: object = None
    ) -> dict[str, Any]:
        if method == AGENT_OPEN_METHOD:
            if self._loop is not None:
                raise UnsupportedAgentRequestError("the agent is already open")
            request = decode_open_request(params)
            if request.deadline is None:
                return await self._open_loop(params)
            # An open that resumes a stored session is a long operation: it is
            # admitted over the fence, announced, and bounded by the
            # coordinator's absolute deadline, and its reply completes on the
            # same request id either way.
            return await self._long_ops.admit(
                request_id=str(request_id),
                method=method,
                deadline=request.deadline,
                body=lambda: self._open_loop(params),
            )
        if method == AGENT_SESSION_SNAPSHOT_METHOD:
            return self._read_snapshot(params)
        if method == AGENT_COMPACT_METHOD:
            return await self._compact(request_id, params)
        if method == AGENT_SESSION_REPLACE_METHOD:
            return await self._replace(request_id, params)
        if method == AGENT_SESSION_RENAME_METHOD:
            return await self._rename(params)
        if method == AGENT_SESSION_RECONFIGURE_METHOD:
            return await self._reconfigure(request_id, params)
        if method == AGENT_SESSION_LIST_METHOD:
            return await self._list_sessions(params)
        if method == AGENT_SESSION_CONFIG_RELOAD_METHOD:
            return await self._reload_config(params)
        if method == AGENT_SESSION_PLAN_MODE_PRESENT_METHOD:
            return await self._plan_mode_present(params)
        if method == AGENT_SESSION_POLICY_READ_METHOD:
            return await self._read_policy(params)
        if method == AGENT_SESSION_SAFETY_READ_METHOD:
            return self._safety_read(params)
        if method == AGENT_SESSION_SAFETY_TRANSITION_METHOD:
            return self._safety_transition(params)
        if method == AGENT_SESSION_ROOTS_READ_METHOD:
            return await self._read_roots(params)
        if method == AGENT_SESSION_ROOTS_SAVE_METHOD:
            return await self._save_roots(params)
        if method == AGENT_WORKSPACE_PROMPT_PREPARE_METHOD:
            return await self._prepare_prompt(params)
        if method in AGENT_CONFIG_SURFACE_METHODS:
            return await self._config_surface(method, params)
        if method in AGENT_POLICY_DANGEROUS_COMMANDS_METHODS:
            return await self._dangerous_commands_surface(method, params)
        if method in AGENT_AGENTS_METHODS:
            return await self._subagent_management_surface(method, params)
        if method == AGENT_RUN_COMPLETIONS_READ_METHOD:
            return self._read_run_completions(params)
        if method == AGENT_CHILD_TRANSCRIPT_PAGE_METHOD:
            return await self._child_transcript_surface(method, params)
        if method == AGENT_SESSION_HISTORY_LIST_METHOD:
            return self._history_list_page(params)
        if method in AGENT_LOOPS_METHODS:
            return await self._loops_surface(method, params)
        if method in AGENT_DIAGNOSTICS_METHODS:
            return await self._diagnostics_surface(method, params)
        if method in AGENT_CATALOG_READ_METHODS:
            return await self._catalog_read_surface(method, params)
        if method in AGENT_REVIEW_METHODS:
            return self._review_surface(method, params)
        if method in AGENT_MCP_CATALOG_METHODS:
            return await self._mcp_catalog_surface(method, params)
        if method in {AGENT_MCP_OWNER_RELEASE_METHOD, AGENT_MCP_OWNER_REKEY_METHOD}:
            return self._mcp_owner_key_surface(method, params)
        if method in {
            AGENT_MCP_RELOAD_PREPARE_METHOD,
            AGENT_MCP_RELOAD_FINISH_METHOD,
            AGENT_MCP_RELOAD_FAIL_METHOD,
        }:
            return await self._mcp_reload_surface(method, params)
        if method == AGENT_RECOVERY_EVIDENCE_METHOD:
            return await self._recovery_evidence(params)
        if method in AGENT_DEFERRED_SESSION_METHODS:
            raise UnsupportedAgentRequestError(deferred_session_message(method))
        if method == AGENT_TRANSCRIPT_PAGE_METHOD:
            return await self._read_page(params)
        if method == AGENT_TURN_ANSWER_METHOD:
            return {"outcome": await self._answer_user_input(params)}
        if method == AGENT_TURN_REFUSE_METHOD:
            return {"outcome": await self._refuse_user_input(params)}
        if method == AGENT_TURN_KEEP_WAITING_METHOD:
            return {"outcome": await self._keep_waiting_user_input(params)}
        if method == AGENT_TURN_CANCEL_METHOD:
            await self._cancel_user_input(params)
            return {"outcome": "cancelled"}
        if method == AGENT_TURN_STEER_METHOD:
            settlement = await self._steering().steer(decode_steer_request(params))
            return encode_steer_settlement(settlement)
        if method == AGENT_INJECT_METHOD:
            request, manual_shell = decode_inject_request(params)
            events = await self._steering().inject(request, manual_shell)
            return encode_inject_reply(events)
        if method == AGENT_SHELL_RUN_METHOD:
            return await self._run_shell(params)
        if method == AGENT_SHELL_INTERRUPT_METHOD:
            return await self._interrupt_shell(params)
        if method in POLICY_METHODS:
            return await self._policy_service().handle(method, params)
        if method == AGENT_CHILD_TEARDOWN_METHOD:
            return await self._teardown_child_attempt(params)
        if method == AGENT_AUTHORITY_CHECK_METHOD:
            return self._authority_check(params)
        raise UnsupportedAgentRequestError(f"unknown agent method {method!r}")

    def _authority_check(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """Answer one per-decision authority question from an owned child (G5).

        The fence runs first and refuses a superseded epoch, a replayed or
        reordered request and a rolled-back revision vector WITHOUT reading
        policy -- each of those is indistinguishable from an attempt to decide
        on stale authority. An admissible request is then decided from the
        loop's CURRENT policy: the vector is read, the tool invocation is
        evaluated against it, and the vector is read again. A policy commit
        that landed in between makes that evaluation stale, so it is redone on
        the committed policy; a vector that will not stand still is a deny.

        Every other outcome -- no open session, no child host, an unowned
        child, an unknown decision kind, an unknown or custom implementation, a
        mismatched identity, arguments this implementation cannot validate, an
        unreadable revision -- is a deny with its typed reason. Nothing here
        waits on the child, so the answer is produced inside the request's own
        arrival step and the child's timeout governs.
        """
        request = decode_authority_check(params)
        vector, reason = self._stabilized_authority_decision(request)
        return encode_authority_decision(
            AgentAuthorityDecision(
                decision=AGENT_AUTHORITY_DENY
                if reason is not None
                else AGENT_AUTHORITY_ALLOW,
                reason=reason,
                epoch=request.epoch,
                parent_session_id=request.parent_session_id,
                parent_incarnation=self._authority_incarnation,
                child_generation=request.child_generation,
                request_id=request.request_id,
                revision=vector if vector is not None else {},
            )
        )

    def _stabilized_authority_decision(
        self, request: AgentAuthorityCheck
    ) -> tuple[dict[str, int] | None, str | None]:
        """(vector, refusal) for one admitted request, on a stable revision."""
        refusal = self._authority_fence.admit(
            request, session_id=self.session_id, incarnation=self._authority_incarnation
        )
        if refusal is not None:
            return self._authority_revision_vector(), refusal
        for _ in range(_AUTHORITY_CHECK_RESTARTS):
            before = self._authority_revision_vector()
            if before is None:
                return None, "the owner's authority revision is unavailable"
            reason = self._tool_invocation_refusal(request)
            after = self._authority_revision_vector()
            if after is None:
                return None, "the owner's authority revision is unavailable"
            if after == before:
                return after, reason
        return None, "the owner's authority revision moved during evaluation"

    def _tool_invocation_refusal(  # noqa: PLR0911 - one refusal per policy rule
        self, request: AgentAuthorityCheck
    ) -> str | None:
        """The refusal for one tool invocation under the owner's current policy.

        Two decision kinds reach here and share the same name-level ceiling:
        ``tool-invocation`` additionally validates the invocation's arguments
        against the owner's own implementation, while ``tool-availability``
        answers the catalog question and therefore stops after the name-level
        checks (it carries no arguments to validate). Anything that cannot be
        proven from the owner's live authority is a refusal.
        """
        if request.decision_kind not in {
            AGENT_AUTHORITY_DECISION_KIND_TOOL_INVOCATION,
            AGENT_AUTHORITY_DECISION_KIND_TOOL_AVAILABILITY,
        }:
            return f"unknown authority decision kind {request.decision_kind!r}"
        loop = self._loop
        host = self._child_host
        if loop is None or host is None:
            return "this agent owns no children to decide for"
        if host.owned_child(request.child_agent_id) is None:
            return (
                f"{request.child_agent_id!r} is not a child this agent owns; "
                "authority is answered only for an immediate child"
            )
        manager = getattr(loop, "tool_manager", None)
        if manager is None:
            return "the owner holds no tool authority to decide with"
        try:
            selected = manager.delegable_tools.get(request.tool_name)
        except Exception:
            return "the owner's tool authority is unavailable"
        if selected is None:
            return (
                f"{request.tool_name!r} is not delegable under the owner's "
                "current policy"
            )
        if shipped_builtin_name(selected) is None:
            return (
                f"{request.tool_name!r} resolves to an unknown or custom "
                "implementation, which live authority does not admit"
            )
        if implementation_identity(selected) != request.implementation:
            return (
                f"{request.tool_name!r} resolves to a different implementation "
                "than the one the child intends to run"
            )
        try:
            if manager.get_tool_config(request.tool_name).permission is (
                ToolPermission.NEVER
            ):
                return f"{request.tool_name!r} is disabled by the owner's policy"
            if request.decision_kind == AGENT_AUTHORITY_DECISION_KIND_TOOL_AVAILABILITY:
                # The catalog question is name-level by construction: the
                # owner's own selected variant stays the class the child must
                # run, so an advertisement it could not execute is impossible.
                return None
            arguments = selected.validate_arguments(dict(request.arguments))
            context = self._authority_permission_context(
                manager, request.tool_name, selected, arguments
            )
        except Exception:
            return "the owner's tool policy could not be resolved for these arguments"
        if context is not None and context.permission is ToolPermission.NEVER:
            return context.reason or (
                f"the owner's policy denies {request.tool_name!r} for these arguments"
            )
        return None

    def _authority_permission_context(
        self, manager: Any, tool_name: str, selected: type[Any], arguments: Any
    ) -> Any:
        """Resolve one delegated invocation's policy exactly as a parent does.

        The ceiling this decision applies is the DELEGATION axis -- what this
        owner may pass to a child -- never its personal inventory, because an
        orchestrator delegates writes it cannot perform itself. The policy
        instance is built from the owner's own selected variant under the
        owner's own config, so the child's claimed class can only be rejected,
        never used to weaken the owner's policy. Shell launches keep their own
        canonical-bash guard, and file/path tools run the full path-authority
        and request-ceiling resolution.
        """
        if tool_name == "bash_start":
            return manager._delegated_shell_execution_permission(
                arguments, None, selected
            )
        with (
            instruction_read_ceiling(manager.instruction_read_files),
            scratchpad_ceiling(manager.scratchpad_roots),
        ):
            return manager._delegated_tool(tool_name, selected).resolve_permission(
                arguments
            )

    def _authority_revision_vector(self) -> dict[str, int] | None:
        """The owner's monotone authority revision vector, or ``None``.

        ``policy`` carries the loop's accepted-policy revision plus this
        manager's own runtime authority invalidations (session grants,
        instruction-read publication, scratchpad scope); ``registry`` the tool
        registry generation; ``grants`` the dangerous-command policy revision;
        ``ancestors`` the summed revision of every manager above this one. A
        component that cannot be read as a non-negative integer makes the whole
        vector unreadable, and an unreadable vector never decides anything.
        """
        loop = self._loop
        if loop is None:
            return None
        manager = getattr(loop, "tool_manager", None)
        if manager is None:
            return None
        grants = getattr(loop, "_dangerous_command_grants", None)
        ancestors = _ancestor_authority_revision(manager)
        policy = getattr(loop, "_authority_revision", None)
        manager_generation = getattr(manager, "_authority_generation", None)
        if not isinstance(policy, int) or not isinstance(manager_generation, int):
            return None
        if not isinstance(ancestors, int):
            return None
        components = {
            "policy": policy + manager_generation,
            "registry": getattr(manager, "_registry_generation", None),
            "grants": getattr(grants, "current_policy_revision", None),
            "ancestors": ancestors,
        }
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in components.values()
        ):
            return None
        return components

    def _note_authority_committed(self) -> None:
        """Schedule an invalidation when a served request moved the vector.

        Policy commits land through this end's own requests (a reconfigure, a
        config write, a roots save, a policy phase), so the check runs after a
        request is served: an unchanged vector publishes nothing, and a change
        coalesces into one fresh-snapshot notification exactly like F10.
        """
        vector = self._authority_revision_vector()
        if vector is None or vector == self._published_authority_revision:
            return
        self._schedule_authority_changed()

    def _schedule_authority_changed(self) -> None:
        """Arm the dirty flag and start the pass unless one is in flight."""
        self._authority_changed_dirty = True
        task = self._authority_changed_task
        if task is None or task.done():
            self._authority_changed_task = self._track(self._emit_authority_changed())

    def _retry_authority_changed(self) -> None:
        """Re-run a pass whose delivery failed with a commit racing it.

        Mirrors the inventory pass's bounded chain: the in-flight guard must
        see a done task, so the retry re-enters the scheduler on the next
        iteration; the budget stops a persistently broken pipe from spinning.
        """
        if self._authority_changed_failures >= _AUTHORITY_CHANGED_RETRY_LIMIT:
            return
        self._authority_changed_failures += 1
        asyncio.get_running_loop().call_soon(self._schedule_authority_changed)

    async def _emit_authority_changed(self) -> None:
        """Deliver the fresh authority snapshot as an id-less notification.

        Every payload is the vector current at its own build, so a shed or lost
        frame can only delay the next commit's fresher copy: the frame is an
        invalidation, never a delta a child could apply. Shutdown cancels the
        pass with the rest of this end's tasks.
        """
        try:
            while self._authority_changed_dirty:
                self._authority_changed_dirty = False
                vector = self._authority_revision_vector()
                session_id = self.session_id
                if vector is None or session_id is None:
                    return
                await self._notify(
                    AGENT_AUTHORITY_CHANGED_METHOD,
                    encode_authority_changed(
                        AgentAuthorityChanged(
                            session_id=session_id,
                            incarnation=self._authority_incarnation,
                            revision=vector,
                        )
                    ),
                )
                # G5b: the same snapshot goes down every live child's own pipe.
                # The child cannot see this end's coordinator notification, so
                # without this fan-out its catalog cache would only refresh on
                # its next decision. A child that misses it still asks fresh at
                # its next invocation, so the shed frame costs a stale catalog,
                # never a stale allow.
                await self._broadcast_authority_changed(
                    encode_authority_changed(
                        AgentAuthorityChanged(
                            session_id=session_id,
                            incarnation=self._authority_incarnation,
                            revision=vector,
                        )
                    )
                )
                self._published_authority_revision = vector
            self._authority_changed_failures = 0
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            raced = self._authority_changed_dirty
            self._authority_changed_dirty = True
            logger.warning("Dropped an authority-change emission: %s", exc)
            if raced:
                self._retry_authority_changed()

    async def _broadcast_authority_changed(self, params: Mapping[str, Any]) -> None:
        """Hand one invalidation to every live child this owner spawned.

        Failure-tolerant per child: a child whose pipe is gone (or wedged) must
        never stall the owner's emission pass or fail it — the frame is an
        invalidation, and the child's next decision asks again regardless.
        """
        host = self._child_host
        if host is None:
            return
        sends: list[Awaitable[None]] = []
        for handle in host.children():
            client = getattr(getattr(handle, "spawned", None), "wire_client", None)
            notify = getattr(client, "notify_authority_changed", None)
            if notify is not None:
                sends.append(notify(dict(params)))
        if not sends:
            return
        results = await asyncio.gather(*sends, return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException):
                logger.debug("Dropped an authority-change relay to a child: %s", result)

    async def _teardown_child_attempt(
        self, params: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Settle one child publication attempt whose outcome was lost.

        The coordinator minted the attempt id, so it survives a lost aggregate
        reply. The reply carries the class the owner ESTABLISHED — reaped,
        tombstoned, inconclusive — never a claimed rollback, and an agent that
        owns no publication attempts answers the request typed.
        """
        port = self._publication_teardowns
        if port is None:
            raise UnsupportedAgentRequestError(
                "this agent owns no child publication attempts to tear down"
            )
        report = await port.teardown_attempt(ChildTeardownRequest.from_wire(params))
        return report.to_wire()

    async def _open_loop(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """Open the loop this request asks for and answer with its open reply."""
        loop = await self._factory.open(params)
        self._loop = loop
        # The production factory builds the session's child host alongside the
        # loop it belongs to; installing it here is what serves the turn
        # offer's subagent runner and the coordinator's teardown requests.
        host = cast("AgentChildHost | None", getattr(self._factory, "child_host", None))
        if host is not None:
            self.install_child_host(host)
        # The broker settles a user-input request in this process, but the
        # terminal projection belongs to the coordinator: binding this end's
        # publisher is what keeps "published before the waiter unblocks" true
        # across the pipe. A loop with no request broker (a scripted test loop)
        # opens no user-input record, so there is nothing to publish.
        broker = getattr(loop, "request_broker", None)
        bind = getattr(broker, "bind_settlement_publisher", None)
        if bind is not None:
            bind(_WireSettlementPublisher(self))
        self._start_notices(loop)
        # G5: the baseline the invalidation emission diffs against is the policy
        # this open accepted, so nothing is published merely for opening.
        self._published_authority_revision = self._authority_revision_vector()
        accounting = cast(
            "RootOpenAccounting | None", getattr(self._factory, "open_accounting", None)
        )
        return encode_open_reply(build_open_reply(loop, accounting=accounting))

    def _read_snapshot(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """Build the session snapshot at reply-write time and encode it.

        The marker set is read now, when the reply is written, never staged
        earlier: a payload therefore cannot predate an admission that preceded
        its reply, which is what lets its absence prove never-admitted.
        """
        loop = self._require_loop()
        return encode_session_snapshot(
            build_session_snapshot(
                loop,
                markers=self._long_ops.records(),
                incarnation_id=self._markers.incarnation_id,
                history_limit=int(
                    params.get("history_limit") or SNAPSHOT_HISTORY_LIMIT
                ),
                include_history=bool(params.get("include_history", True)),
                agent_sequence=self._event_sequence,
            )
        )

    async def _announce_admission(self, marker: OperationMarker) -> None:
        """Announce one admitted long operation before its body starts."""
        await self._notify(AGENT_OP_ADMISSION_METHOD, encode_operation_marker(marker))

    def _remember_admission(self, marker: OperationMarker) -> None:
        """Hold the record of the operation the fence just admitted."""
        self._admitted_marker = marker

    async def _compact(
        self, request_id: object, params: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Run one manual compaction as a long operation (item 1 op 5).

        The wrapper behavior is the in-process one (`_handler._compact`): the
        run is bounded by an absolute deadline, a boundary that was published
        is settled even when the interruption arrives after it, and nothing is
        ever retried automatically -- the reply is the receipt either way.
        """
        request = decode_compact_request(params)
        if request.deadline is None:
            raise UnsupportedAgentRequestError(
                "a compaction request must carry the coordinator's absolute deadline"
            )
        return await self._long_ops.admit(
            request_id=str(request_id),
            method=AGENT_COMPACT_METHOD,
            deadline=request.deadline,
            body=lambda: self._compact_operation(request),
        )

    async def _compact_operation(self, request: CompactRequest) -> dict[str, Any]:
        """Run the compaction body and answer with its agent-minted receipt.

        A cancel notification and the deadline timer both arrive as
        cancellation at the body's next await point (S9-C2), so the record is
        settled with the class the publication evidence supports before any
        reply is written: pre-publication settles cancelled-unpublished,
        post-publication settles the published class.
        """
        loop = self._require_loop()
        marker = self._admitted_marker
        if marker is None:
            raise UnsupportedAgentRequestError(
                "the compaction ran without an admission record"
            )
        try:
            summary = await loop.compact(request.extra_instructions, manual=True)
        except asyncio.CancelledError:
            settlement = (
                COMPACT_SETTLEMENT_PUBLISHED_UNSETTLED
                if loop.compaction_outcome.published
                else COMPACT_SETTLEMENT_CANCELLED_UNPUBLISHED
            )
            self._markers.settle(marker.op_id, settlement)
            detail = (
                "the coordinator cancelled the compaction before it settled"
                if self._long_ops.cancel_requested(marker.op_id)
                else "the compaction was aborted at its absolute deadline"
            )
            return encode_compact_reply(
                self._unsettled_compact_receipt(loop, settlement, detail)
            )
        except Exception as exc:
            published = loop.compaction_outcome.published
            settlement = (
                COMPACT_SETTLEMENT_PUBLISHED_UNSETTLED
                if published
                else COMPACT_SETTLEMENT_CANCELLED_UNPUBLISHED
            )
            self._markers.settle(marker.op_id, settlement)
            return encode_compact_reply(
                self._unsettled_compact_receipt(
                    loop,
                    settlement,
                    f"the compaction failed ({type(exc).__name__}: {exc})",
                )
            )
        try:
            receipt = self._settled_compact_receipt(loop, summary)
        except Exception as exc:
            # The boundary is live, so only the handoff state is missing: a
            # client reconciles instead of retrying a summarization that
            # already happened.
            settlement = COMPACT_SETTLEMENT_PUBLISHED_UNSETTLED
            self._markers.settle(marker.op_id, settlement)
            return encode_compact_reply(
                self._unsettled_compact_receipt(
                    loop,
                    settlement,
                    "the compaction published its boundary but the handoff "
                    f"state could not be read ({type(exc).__name__}: {exc})",
                )
            )
        self._markers.settle(marker.op_id, COMPACT_SETTLEMENT_PUBLISHED_SETTLED)
        return encode_compact_reply(receipt)

    def _settled_compact_receipt(
        self, loop: AgentLoop, summary: str
    ) -> CompactSettlementReply:
        """The publication receipt: the summary and the identity handoff state."""
        snapshot = build_session_snapshot(
            loop,
            markers=(),
            incarnation_id=self._markers.incarnation_id,
            agent_sequence=self._event_sequence,
        )
        session_dir, _jobs_dir = session_dirs(loop)
        return CompactSettlementReply(
            settlement=COMPACT_SETTLEMENT_PUBLISHED_SETTLED,
            session_id=loop.session_id,
            session_dir=str(session_dir) if session_dir is not None else None,
            summary=summary,
            state=snapshot.state,
            session_log=snapshot.log,
        )

    def _unsettled_compact_receipt(
        self, loop: AgentLoop, settlement: str, detail: str
    ) -> CompactSettlementReply:
        """The receipt a compaction still owes when it could not settle.

        The reconciliation state rides along when it can be read; a receipt
        that cannot carry it still carries the class and what happened, because
        the record -- not the reply -- is what a re-check settles against.
        """
        state = None
        log = None
        try:
            snapshot = build_session_snapshot(
                loop,
                markers=(),
                incarnation_id=self._markers.incarnation_id,
                agent_sequence=self._event_sequence,
            )
        except Exception:
            pass
        else:
            state, log = snapshot.state, snapshot.log
        session_dir, _jobs_dir = session_dirs(loop)
        return CompactSettlementReply(
            settlement=settlement,
            session_id=loop.session_id,
            session_dir=str(session_dir) if session_dir is not None else None,
            state=state,
            session_log=log,
            detail=detail,
        )

    async def _replace(
        self, request_id: object, params: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Rebind this agent's session to a stored one as a long operation.

        The rebind loads a transcript and builds an MCP registry, so it is
        admitted over the fence with the coordinator's absolute deadline and
        announced before any body step runs. A refused, expired or aborted
        admission answers the request id typed and leaves the old identity in
        place; only a committed rebind is answered with a receipt.
        """
        request = decode_replace_request(params)
        if request.deadline is None:
            raise UnsupportedAgentRequestError(
                "a session replace must carry the coordinator's absolute deadline"
            )
        return await self._long_ops.admit(
            request_id=str(request_id),
            method=AGENT_SESSION_REPLACE_METHOD,
            deadline=request.deadline,
            body=lambda: self._rebind_operation(request),
        )

    async def _rebind_operation(self, request: ReplaceRequest) -> dict[str, Any]:
        """Run one rebind and answer with the identity it left behind."""
        loop = self._require_loop()
        marker = self._admitted_marker
        if marker is None:
            raise UnsupportedAgentRequestError(
                "the session replace ran without an admission record"
            )
        rebind = getattr(self._factory, "replace", None)
        if rebind is None:
            raise UnsupportedAgentRequestError(
                "this agent cannot rebind its session to a stored one"
            )
        await rebind(
            loop, session_id=request.session_id, continue_latest=request.continue_latest
        )
        # The rebind committed, and nothing awaits between it and the marker
        # handoff below. The operation's own record is settled here rather than
        # by the fence because a seed must be TERMINAL to cross the rebind
        # (S5-B5); the fence's own settle then finds it already settled, which
        # is a no-op.
        self._markers.settle(marker.op_id, OP_OUTCOME_COMPLETED)
        self._markers.rebind(incarnation_id=uuid4().hex, carry_op_id=marker.op_id)
        snapshot = build_session_snapshot(
            loop,
            markers=self._long_ops.records(),
            incarnation_id=self._markers.incarnation_id,
            agent_sequence=self._event_sequence,
        )
        accounting = cast(
            "RootOpenAccounting | None", getattr(self._factory, "open_accounting", None)
        )
        return encode_replace_reply(
            ReplaceReply(snapshot=snapshot, accounting=accounting)
        )

    async def _rename(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """Apply one manual title here, where the session metadata lives."""
        loop = self._require_loop()
        title = decode_rename_request(params)
        updated_at = await loop.session_logger.apply_manual_title(title)
        persisted = loop.session_logger.title
        if persisted is None:
            raise RuntimeError("The session title was not updated")
        return encode_rename_reply(RenameReply(title=persisted, updated_at=updated_at))

    async def _reconfigure(
        self, request_id: object, params: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Run one reconfigure mode: short here, the fence over the fence.

        A short mode mutates the session directly and is refused typed while a
        long operation holds the fence (one long op per root), so a settings
        update or a clear can never interleave with a compaction, a replace or
        a config write. The config write is admitted as a long operation with
        the coordinator's absolute deadline: its marker record, its deadline
        and its fence are the same ones compaction uses.
        """
        request = decode_reconfigure_request(params)
        if request.mode == RECONFIGURE_MODE_CONFIG_WRITE:
            if request.deadline is None:
                raise UnsupportedAgentRequestError(
                    "a config write must carry the coordinator's absolute deadline"
                )
            return await self._long_ops.admit(
                request_id=str(request_id),
                method=AGENT_SESSION_RECONFIGURE_METHOD,
                deadline=request.deadline,
                body=lambda: self._config_write_operation(request),
            )
        active = self._long_ops.active_op_id
        if active is not None:
            raise AgentReconfigureConflictError(
                f"{request.mode} is refused while operation {active} is in "
                "flight; retry once it settles"
            )
        async with self._reconfigure_lock:
            service = self._reconfigure_service()
            if request.mode == RECONFIGURE_MODE_SETTINGS:
                return encode_reconfigure_reply(service.settings_update(request))
            assert request.mode == RECONFIGURE_MODE_HISTORY_CLEAR
            return encode_reconfigure_reply(await service.history_clear(request))

    async def _config_write_operation(
        self, request: ReconfigureRequest
    ) -> dict[str, Any]:
        """Run one config transaction and answer with its shipped result."""
        async with self._reconfigure_lock:
            return encode_reconfigure_reply(
                await self._reconfigure_service().config_write(request)
            )

    def _reconfigure_service(self) -> AgentReconfigureService:
        """The reconfigure adapter bound to the loop this agent owns now."""
        return AgentReconfigureService(
            self._require_loop(),
            turn_active=lambda: self._active_turn_id is not None,
            snapshot=self._agent_snapshot,
        )

    async def _list_sessions(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """List the sessions this agent's own config resolves on disk.

        The list is a pure projection of the store plus the loop's cwd, so it
        is answered here rather than reconstructed from a coordinator copy of
        a config that would not be the one the session opened with.
        """
        from chartreux.app_server._host import project_session_list

        request = SessionListParams.model_validate(dict(params))
        response = await asyncio.to_thread(
            project_session_list, self._require_loop().config, request
        )
        return response.model_dump(mode="json")

    async def _reload_config(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """Run one config reload transaction in the process that owns the loop.

        The receipt is the shipped mutation response: the reload's prepared and
        applied legs run against this process's orchestrator and backend, so a
        coordinator-side reload would report a config this session is not using.
        """
        dispatched = await self._resources().dispatch("config/reload", dict(params))
        return dispatched.response.model_dump(mode="json")

    async def _plan_mode_present(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """True when the session's RAW metadata carried a ``plan_mode`` key.

        Detection reads the raw metadata dict through the loader choke point,
        never the typed record — the typed ``SessionMetadata`` drops the
        retired key (``extra="ignore"``), so only the raw dict still has it.
        """
        loop = self._require_loop()
        if params.get("session_id") != loop.session_id:
            raise UnsupportedAgentRequestError(
                "the plan-mode presence flag belongs to the root session"
            )
        logger = loop.session_logger
        session_dir = logger.session_dir
        if not logger.enabled or session_dir is None:
            return {"plan_mode_present": False}
        raw_metadata = await asyncio.to_thread(
            SessionLoader.load_raw_metadata, session_dir
        )
        return {"plan_mode_present": "plan_mode" in raw_metadata}

    async def _read_policy(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """Answer the policy read from the authority this agent's loop holds.

        The revision is the same tree token the node's own four-phase service
        reports, so a reader compares like with like across the two topologies.
        """
        request = PolicyReadParams.model_validate(dict(params))
        loop = self._require_loop()
        if request.session_id != loop.session_id:
            raise UnsupportedAgentRequestError("policy requires the root session")
        orchestrator = loop.config_orchestrator
        return {
            "revision": self._policy_service().revision().tree_revision,
            "sources": [
                source.layer_name
                for source in orchestrator.restrictions
                if source.kind == "source"
                and source.identity is not None
                and source.identity.owner == orchestrator.policy_owner
            ],
        }

    def _safety_read(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """Report the loop's current SafetyLevel (v0.8.0 A-P1)."""
        request = SessionSafetyReadParams.model_validate(dict(params))
        loop = self._require_loop()
        if request.session_id != loop.session_id:
            raise UnsupportedAgentRequestError("safety read requires the root session")
        return {"level": loop.safety_level.value}

    def _safety_transition(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """One authority-level transition, gated where the loop lives (A-P1).

        Idle-only: a turn in flight refuses the transition. Entering
        DANGEROUS additionally requires a writable session audit sink — fail
        closed. Revocation refuses while managed background jobs run. The
        interactive ceremony lives TUI-side; this is the server-side floor,
        mirroring the in-process handler's gates.
        """
        from chartreux.app_server._runtime import (
            SessionAuditSinkUnavailableError,
            require_writable_session_audit_sink,
        )

        request = SessionSafetyTransitionParams.model_validate(dict(params))
        loop = self._require_loop()
        if request.session_id != loop.session_id:
            raise UnsupportedAgentRequestError(
                "safety transition requires the root session"
            )
        current = loop.safety_level.value
        try:
            target = SafetyLevel(request.to)
        except ValueError:
            return {
                "level": current,
                "transitioned": False,
                "reason": f"unknown safety level: {request.to!r}",
            }
        if self._active_turn_id is not None:
            return {
                "level": current,
                "transitioned": False,
                "reason": "a turn is active; safety-level transitions are idle-only",
            }
        if target is SafetyLevel.DANGEROUS:
            try:
                require_writable_session_audit_sink(loop.config)
            except SessionAuditSinkUnavailableError as exc:
                return {
                    "level": current,
                    "transitioned": False,
                    "reason": f"no writable session audit sink: {exc}",
                }
        elif loop.active_background_job_count > 0:
            return {
                "level": current,
                "transitioned": False,
                "reason": (
                    "managed background jobs are active; settle them before "
                    "revoking authority"
                ),
            }
        event = loop.authority_state.transition(target, actor=AuthorityActor.HUMAN)
        return {
            "level": loop.safety_level.value,
            "transitioned": event is not None,
            "reason": "",
        }

    async def _read_roots(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """Answer the authorized-roots view from this loop's own source.

        The effective view is the owned user source's roots for this session's
        workspace plus the grants this incarnation holds, so the coordinator
        never composes a path list from a copy it does not own.
        """
        request = RootsReadParams.model_validate(dict(params))
        loop = self._require_loop()
        if request.session_id != loop.session_id:
            raise UnsupportedAgentRequestError("roots require the root session")
        project = loop.cwd
        source = owned_user_source(loop.config_orchestrator)
        saved = await read_saved_roots(loop.config_orchestrator, project=project)
        effective = {
            path
            for entry in source.authorized_roots
            if entry.project == project
            for path in entry.roots
        }
        effective.update(loop._session_root_grants.roots)
        effective.discard(project.resolve())
        return RootsReadResponse(
            revision=session_roots_revision(loop),
            project=str(project),
            roots=[
                str(path)
                for entry in source.authorized_roots
                if entry.project == project
                for path in entry.roots
            ],
            saved_roots=list(saved.roots) if saved.unavailable is None else None,
            user_revision=saved.user_revision,
            effective_roots=sorted(str(path) for path in effective),
        ).model_dump(mode="json")

    async def _save_roots(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """Persist one approved project root grant through this loop's config.

        The save targets this session's own user layer under the orchestrator
        mutation lock and publishes nothing, exactly as the in-process save
        does; the outcome is reported, never raised, so an already-applied
        session grant survives a stale expected revision.
        """
        request = RootsSaveParams.model_validate(dict(params))
        loop = self._require_loop()
        if request.session_id != loop.session_id:
            raise UnsupportedAgentRequestError("roots require the root session")
        result = await loop.config_orchestrator.save_project_root_grant(
            project=loop.cwd,
            root=Path(request.root),
            expected_revision=request.expected_revision,
        )
        return RootsSaveResponse(
            target=result.target,
            persistence=result.persistence,
            application=result.application,
            revision=result.revision,
            error=result.error,
        ).model_dump(mode="json")

    async def _prepare_prompt(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """Prepare one prompt here, where the loop, its model and its cwd live.

        Preparation reads the live model, the session's working directory and
        its session directory, and it writes attachment snapshots, so it runs
        in this process and the reply carries the prompt it built -- a
        coordinator cache could answer with the wrong model or directory.
        The validation and error types are the in-process ones: a preparation
        failure (too many or unsupported images, an unreadable attachment)
        answers typed on the request id as ``PromptPreparationError``, and a
        preparation that times out is an unknown outcome the caller never
        blind-retries.
        """
        request = WorkspacePromptPrepareParams.model_validate(dict(params))
        loop = self._require_loop()
        if request.session_id != loop.session_id:
            raise UnsupportedAgentRequestError(
                "prompt preparation requires the root session"
            )
        prompt = await asyncio.to_thread(
            prepare_prompt, loop, request.message, request.title_content
        )
        return WorkspacePromptPrepareResponse(prompt=prompt).model_dump(mode="json")

    async def _config_surface(
        self, method: str, params: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Answer one configuration projection from the owner this agent holds.

        The fields and settings views are projections of this process's own
        orchestrator, its layered sources and their save revisions, and the
        proxy settings are read from -- and written to -- the env file this
        process resolves; a coordinator copy would report a config the session
        is not running on. The reply is the shipped resource response, so both
        routes hand the client one shape, and a refusal publishes the code the
        in-process handler answers with: a session mismatch is NOT_FOUND, a
        rejected proxy value INVALID_PARAMS, a busy session CONFLICT.
        """
        if method == AGENT_CONFIG_PROXY_WRITE_METHOD and self._active_turn_id:
            # The in-process handler refuses a proxy write while the session is
            # running work. This process knows its own live turn, and a value
            # written under one would be adopted by a request already in
            # flight, so the same refusal answers here under the same code.
            raise AgentProtocolRefusal(
                ProtocolErrorCode.CONFLICT,
                f"Session is busy running turn {self._active_turn_id}",
            )
        try:
            dispatched = await self._resources().dispatch(
                AGENT_CONFIG_PUBLIC_METHODS[method], dict(params)
            )
        except SessionExecutionConflict as exc:
            raise AgentProtocolRefusal(ProtocolErrorCode.CONFLICT, str(exc)) from exc
        except RequestFailure as exc:
            raise AgentProtocolRefusal(exc.code, str(exc)) from exc
        return dispatched.response.model_dump(mode="json")

    def _authority(self) -> DangerousCommandAuthority:
        """The authority collaborator over the one loop this agent owns.

        This process serves exactly the registered session its own loop is,
        so every resolver answers with that loop; a request naming any other
        session id is refused exactly as an unregistered one.
        """
        loop = self._require_loop()

        def resolve_live_loop(session_id: str) -> AgentLoop:
            if session_id != loop.session_id:
                raise ValueError(
                    "Dangerous-command authority requires a live registered session"
                )
            return loop

        return DangerousCommandAuthority(
            resolve_live_loop=resolve_live_loop,
            resolve_session_loop=lambda session_id: (
                loop if session_id == loop.session_id else None
            ),
            resolve_root_loop=lambda: loop,
        )

    async def _dangerous_commands_surface(
        self, method: str, params: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Answer one dangerous-command authority request from the owner.

        The live grant store, the accepted saved grants and the root
        orchestrator's saved-grant read/write all live in this process's
        loop, so the read and the CAS-guarded revoke run here, where the
        authority generation they check is the session's own — never on a
        coordinator copy. A request naming another session is refused
        not-found, an unknown operation or a wrongly scoped destination a
        param refusal, exactly as the in-process handler answers them; a
        stale expected generation is not an error: the revoke reports the
        conflict outcome with the current generation, never a false ack.
        """
        loop = self._require_loop()
        if method == AGENT_POLICY_DANGEROUS_COMMANDS_READ_METHOD:
            request = DangerousCommandAuthorityReadParams.model_validate(dict(params))
            if request.session_id != loop.session_id:
                raise AgentSessionNotFoundError(
                    f"Session not found: {request.session_id}"
                )
            response = await self._authority().read_dangerous_command_authority(
                request.session_id
            )
            return response.model_dump(mode="json")
        request = DangerousCommandRevokeParams.model_validate(dict(params))
        if request.session_id != loop.session_id:
            raise AgentSessionNotFoundError(f"Session not found: {request.session_id}")
        # The same param refusals the in-process backend answers before it
        # delegates, published here under the protocol's own code.
        if request.operation_id not in OPERATION_REGISTRY:
            raise AgentProtocolRefusal(
                ProtocolErrorCode.INVALID_PARAMS, "Unknown dangerous operation"
            )
        if request.destination is not None and (
            request.operation_id != REMOTE_EXEC_OPERATION_ID
            or canonical_remote_destination(request.destination) is None
        ):
            raise AgentProtocolRefusal(
                ProtocolErrorCode.INVALID_PARAMS,
                "A destination-scoped revocation requires network.remote-exec "
                "and one exact canonical destination",
            )
        response = await self._authority().revoke_dangerous_command_grant(
            session_id=request.session_id,
            operation_id=request.operation_id,
            expected_live_generation=request.expected_live_generation,
            expected_disk_revision=request.expected_disk_revision,
            destination=request.destination,
        )
        return response.model_dump(mode="json")

    def _require_wire_session(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """Return the request body after checking the session it addresses.

        The carrier names the session it answers for; a request naming any
        other id is refused exactly as an unregistered one, and the session
        key never reaches the shipped param model.
        """
        session_id = params.get("session_id")
        loop = self._require_loop()
        if session_id != loop.session_id:
            raise AgentProtocolRefusal(
                ProtocolErrorCode.NOT_FOUND, f"Session not found: {session_id}"
            )
        return {key: value for key, value in params.items() if key != "session_id"}

    async def _subagent_management_surface(  # noqa: PLR0911 - one return per method arm
        self, method: str, params: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Answer one subagent-management request from the owner.

        A process-owned root owns its children, so every operation runs here
        against this process's own child handles. The typed outcome sets and
        guards are the in-process registry's; an operation the process-backed
        child records cannot represent answers its typed lifecycle disposition
        and changes nothing, never a false success.
        """
        loop = self._require_loop()
        body = self._require_wire_session(params)
        host = self._child_host
        if method == AGENT_AGENTS_CANCEL_METHOD:
            request = AgentsCancelParams.model_validate(body)
            if host is None:
                return AgentsCancelResponse(
                    outcome=PublicCancelOutcome.UNKNOWN_RUN, run_id=request.run_id
                ).model_dump(mode="json")
            result = await host.cancel_run(
                request.agent_id,
                request.run_id,
                reason=RunStopReason.USER_CANCELLED,
                requester_session_id=loop.session_id,
            )
            return AgentsCancelResponse(
                outcome=PublicCancelOutcome(result.outcome.value),
                run_id=result.run_id,
                stop_reason=(
                    PublicRunStopReason(result.stop_reason.value)
                    if result.stop_reason is not None
                    else None
                ),
            ).model_dump(mode="json")
        if method == AGENT_AGENTS_RELEASE_METHOD:
            request = AgentsReleaseParams.model_validate(body)
            outcome = (
                ReleaseAgentOutcome.UNKNOWN
                if host is None
                else await host.release_guarded(
                    request.agent_id, request.expected_run_id
                )
            )
            from chartreux.app_server._handler import _RELEASE_REASONS

            return AgentsReleaseResponse(
                outcome=PublicReleaseOutcome(outcome.value),
                reason=_RELEASE_REASONS[outcome],
            ).model_dump(mode="json")
        if method == AGENT_AGENTS_RESUME_METHOD:
            request = AgentsResumeParams.model_validate(body)
            idempotency_key = request.idempotency_key or (
                f"{request.agent_id}:{request.expected_run_id}"
            )
            if host is None:
                result = ResumeAgentResult(
                    outcome=ResumeOutcome.NOT_RESUMABLE,
                    agent_id=request.agent_id,
                    run_id=request.expected_run_id,
                    reason="Unknown agent",
                    reason_codes=["not_suspended"],
                )
            else:
                result = await host.resume_agent(
                    request.agent_id,
                    request.expected_run_id,
                    idempotency_key,
                    requester_session_id=loop.session_id,
                )
            return AgentsResumeResponse(
                outcome=PublicResumeOutcome(result.outcome.value),
                agent_id=result.agent_id,
                run_id=result.run_id,
                successor_run_id=result.successor_run_id,
                recovery_operation_id=result.recovery_operation_id,
                retry_at=result.retry_at,
                eligibility=(
                    PublicRecoveryEligibility(result.eligibility.value)
                    if result.eligibility is not None
                    else None
                ),
                reason_codes=result.reason_codes,
                reason=result.reason,
                same_agent_resumes_used=result.same_agent_resumes_used,
            ).model_dump(mode="json")
        if method == AGENT_AGENTS_INVENTORY_READ_METHOD:
            # The typed live-inventory read rides the same owner surface: the
            # process that owns the children projects its own handles as one
            # revised snapshot, so a coordinator reads the owner's inventory
            # and never a coordinator-side copy of it. The escape route keeps
            # no child host, and this process owns no children then, so its
            # honest inventory is the empty one carried with the loop's own
            # identity and revision floor.
            AgentsInventoryReadParams.model_validate(body)
            if host is None:
                return AgentsInventoryReadResponse(
                    session_id=loop.session_id,
                    generation=loop.session_generation,
                    revision=0,
                    agents=[],
                ).model_dump(mode="json")
            return host.inventory_snapshot().model_dump(mode="json")
        request = AgentsReconcileParams.model_validate(body)
        effects = [
            EffectVerification(
                tool_call_id=effect.tool_call_id,
                disposition=effect.disposition,
                note=effect.note,
            )
            for effect in request.verified_effects
        ]
        if host is None:
            result = ReconcileResult(
                outcome=ReconcileOutcome.NOT_PENDING,
                agent_id=request.agent_id,
                run_id=request.expected_run_id,
                transcript_boundary=request.transcript_boundary,
                reason="No reconcile barrier is pending for this agent",
                reason_codes=["no_reconcile_barrier"],
            )
        else:
            result = await host.reconcile_result(
                request.agent_id,
                request.expected_run_id,
                request.transcript_boundary,
                effects,
                requester_session_id=loop.session_id,
            )
        return AgentsReconcileResponse(
            outcome=PublicReconcileOutcome(result.outcome.value),
            agent_id=result.agent_id,
            run_id=result.run_id,
            transcript_boundary=result.transcript_boundary,
            eligibility=(
                PublicRecoveryEligibility(result.eligibility.value)
                if result.eligibility is not None
                else None
            ),
            reason_codes=result.reason_codes,
            reason=result.reason,
        ).model_dump(mode="json")

    async def _child_transcript_surface(
        self, method: str, params: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Read one page of a child transcript from where it lives.

        The page is answered by the owner the child belongs to: a live child
        through its own reader, a dead one from its saved transcript under the
        root session. An agent this process never spawned is not-found, never
        an empty page.
        """
        body = self._require_wire_session(params)
        request = AgentTranscriptGetParams.model_validate(body)
        host = self._child_host
        if host is None:
            raise AgentProtocolRefusal(ProtocolErrorCode.NOT_FOUND, "Agent not found")
        try:
            response = await host.read_transcript_page(
                request.agent_id, before=request.before, limit=request.limit
            )
        except UnknownAgentError as exc:
            raise AgentProtocolRefusal(
                ProtocolErrorCode.NOT_FOUND, "Agent not found"
            ) from exc
        return encode_transcript_page(response)

    def _history_list_page(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """Serve one page of the attached session's live history.

        The authoritative projection -- live messages plus any sealed prefix
        read back from the transcript -- is rebuilt from the loop this agent
        owns, exactly as the in-process route answers it. The host reads
        persisted messages only, so it stays the route for saved and
        non-attached sessions; a request naming any other id is refused
        not-found here, never answered from this loop's projection.
        """
        body = self._require_wire_session(params)
        request = SessionHistoryListParams.model_validate(body)
        history = project_history(self._require_loop())
        page = history_page(
            history,
            turn_id=request.turn_id,
            before=request.cursor if request.sort_direction == "backward" else None,
            after=request.cursor if request.sort_direction == "forward" else None,
            limit=request.limit,
        )
        return SessionHistoryListResponse(
            items=page.entries,
            next_cursor=(
                page.cursor.before
                if request.sort_direction == "backward"
                else page.cursor.after
            ),
            previous_cursor=(
                page.cursor.after
                if request.sort_direction == "backward"
                else page.cursor.before
            ),
        ).model_dump(mode="json")

    async def _loops_surface(
        self, method: str, params: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Answer one scheduled-loop request from the process holding the schedule.

        The loop manager this process owns is the schedule: its fire times, its
        persistence and its idle admission all live here, so a list, create,
        delete or clear runs here through the shipped resource handler instead
        of against a coordinator copy that would be a second schedule. A session
        this agent does not serve is not-found, an invalid interval, prompt or
        id is the handler's own param refusal, and a mutation while this process
        runs a turn answers the conflict the in-process route answers. The
        request is sent exactly once: a mutation that timed out is an unknown
        outcome, never a blind retry.
        """
        loop = self._require_loop()
        body = self._require_wire_session(params)
        if method != AGENT_LOOPS_LIST_METHOD and self._active_turn_id:
            # The in-process route refuses a mutation while the session runs
            # work. This process knows its own live turn, and a loop created
            # or removed under one would race the run in flight.
            raise AgentProtocolRefusal(
                ProtocolErrorCode.CONFLICT,
                f"Session is busy running turn {self._active_turn_id}",
            )
        try:
            dispatched = await self._resources().dispatch(
                AGENT_LOOPS_PUBLIC_METHODS[method],
                {"sessionId": loop.session_id, **body},
            )
        except SessionExecutionConflict as exc:
            raise AgentProtocolRefusal(ProtocolErrorCode.CONFLICT, str(exc)) from exc
        except RequestFailure as exc:
            raise AgentProtocolRefusal(exc.code, str(exc)) from exc
        return dispatched.response.model_dump(mode="json")

    async def _diagnostics_surface(
        self, method: str, params: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Answer one session-scoped diagnostics read from the process that owns it.

        The issues and hook count are projected from the loop this process owns,
        and the debug log page is read through this process's own log reader
        with the request's own limit and offset, so both reads run against the
        state they describe instead of a coordinator copy of it. The page
        crosses exactly as the shipped resource handler bounds and redacts it:
        nothing is merged in from another process's view, and every entry keeps
        the pid provenance its log line recorded. A session this agent does not
        serve is not-found.
        """
        loop = self._require_loop()
        body = self._require_wire_session(params)
        try:
            dispatched = await self._resources().dispatch(
                AGENT_DIAGNOSTICS_PUBLIC_METHODS[method],
                {"sessionId": loop.session_id, **body},
            )
        except RequestFailure as exc:
            raise AgentProtocolRefusal(exc.code, str(exc)) from exc
        return dispatched.response.model_dump(mode="json")

    async def _catalog_read_surface(
        self, method: str, params: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Answer one latent catalog read from the process that resolves it.

        The skill and tool inventories and the live counters are all resolved
        against this process's own config, tool manager and loop, so each read
        runs here instead of being re-derived from a coordinator copy. The
        shipped resource handler answers all of them, so its validation and
        projection apply unchanged; identity is the absent identity this fork's
        resources answer, which is what the response model already admits.
        """
        loop = self._require_loop()
        body = self._require_wire_session(params)
        try:
            dispatched = await self._resources().dispatch(
                AGENT_CATALOG_PUBLIC_METHODS[method],
                {"sessionId": loop.session_id, **body},
            )
        except RequestFailure as exc:
            raise AgentProtocolRefusal(exc.code, str(exc)) from exc
        return dispatched.response.model_dump(mode="json")

    def _review_surface(self, method: str, params: Mapping[str, Any]) -> dict[str, Any]:
        """Answer one review request from the loop that owns the checkpoints.

        Review state is session-local, memory-only state recorded by this
        process's own checkpoint recorder, so the review manager here is the
        only place it exists. The shipped review handler answers the request,
        which keeps its parameter validation, its idle admission for mutations
        and its projection exactly as the in-process route has them.
        """
        loop = self._require_loop()
        body = self._require_wire_session(params)
        if self._review_handler is None or self._review_loop is not loop:
            self._review_handler = ReviewRequestHandler(
                loop.review_manager,
                self._require_named_session,
                self._require_idle_session,
            )
            self._review_loop = loop
        try:
            dispatched = self._review_handler.dispatch(
                AGENT_REVIEW_PUBLIC_METHODS[method],
                {"sessionId": loop.session_id, **body},
            )
        except RequestFailure as exc:
            raise AgentProtocolRefusal(exc.code, str(exc)) from exc
        return dispatched.response.model_dump(mode="json")

    def _require_named_session(self, session_id: str) -> None:
        loop = self._require_loop()
        if session_id != loop.session_id:
            raise AgentProtocolRefusal(
                ProtocolErrorCode.NOT_FOUND, f"Session not found: {session_id}"
            )

    def _require_idle_session(self) -> None:
        """Refuse a mutation while this process is running a turn.

        A review mutation rewrites the workspace the running turn is still
        writing to, so it waits for idle exactly as the in-process execution
        guard makes it wait.
        """
        if self._active_turn_id is not None:
            raise AgentProtocolRefusal(
                ProtocolErrorCode.CONFLICT,
                f"Session is busy running turn {self._active_turn_id}",
            )

    def _mcp_service(self) -> Any:
        """The owner-side MCP catalog service of this process.

        The factory's harness process owns one (so the loop's registry armed
        at open resolves through the same authentication service); a factory
        that exposes none (a scripted test factory) gets a process-local one.
        """
        if self._mcp_catalog_service is None:
            from chartreux.app_server._mcp_auth import MCPAuthenticationService
            from chartreux.app_server.mcp_catalog import MCPCatalogService

            process = getattr(self._factory, "_process", None)
            service = getattr(process, "mcp_catalog", None)
            if service is None:
                authentication = getattr(process, "mcp_authentication", None)
                if authentication is None:
                    authentication = MCPAuthenticationService()
                service = MCPCatalogService(authentication)
            self._mcp_catalog_service = service
        return self._mcp_catalog_service

    def _mcp_owner_root(self, owner_key: str) -> Any:
        """The loop-backed control port the catalog service runs against.

        Rebuilt whenever an identity flow swaps the loop out, exactly like
        the resource handler. The typed owner key files the auth bindings
        under a handle held for exactly as long as the key lives.
        """
        from chartreux.app_server.mcp_catalog import AgentLoopMCPCatalogRoot

        loop = self._require_loop()
        if self._mcp_catalog_root is None or self._mcp_catalog_loop is not loop:
            self._mcp_catalog_root = AgentLoopMCPCatalogRoot(
                loop,
                resources=self._resources(),
                authorization_provider=self._mcp_service().authentication,
                is_busy=lambda: self._active_turn_id is not None,
                authorization_required_sink=self._mcp_authorization_required_sink,
            )
            self._mcp_catalog_loop = loop
        root = self._mcp_catalog_root
        self._mcp_service().authentication.acquire_owner_key(owner_key)
        return root

    async def _mcp_authorization_required_sink(self, name: str, required: Any) -> None:
        """Gate the authorization-required handshake owner-side and relay it.

        The binding and the read_mcp control port live here, so the dedup and
        the needs_auth check run in this process (the coordinator's copy would
        answer for a backend it does not own); the gated notification crosses
        as an agent-emitted frame the coordinator relays to the client.
        """
        loop = self._require_loop()
        root = self._mcp_owner_root(self._current_owner_key)
        params = MCPAuthRequiredParams(
            session_id=loop.session_id,
            name=name,
            descriptor_revision=required.descriptor_revision,
            observed_connection_revision=getattr(
                required, "observed_connection_revision", None
            ),
        )
        runtime_updated = await self._mcp_service().accept_auth_required(
            params, root, owner_key=self._current_owner_key
        )
        if runtime_updated is None:
            return
        await self._notify(
            AGENT_MCP_AUTH_REQUIRED_METHOD, params.model_dump(mode="json")
        )

    async def _mcp_notify_relay(self, method: str, params: Any) -> None:
        """Relay one service notification agent -> coordinator -> client.

        The login's authUrl is published by the service through this notify;
        here it becomes the agent-emitted frame the coordinator's pump
        relays as the client-facing ``mcp_catalog/authUrl``. The authUrl
        payload carries no session id (its model forbids extra keys), so
        only the session-scoped authRequired frame names the session.
        """
        if method == "mcp_catalog/authUrl":
            await self._notify(
                AGENT_MCP_AUTH_URL_METHOD, params.model_dump(mode="json")
            )
            return
        await self._notify(
            method,
            {
                **params.model_dump(mode="json"),
                "session_id": self._require_loop().session_id,
            },
        )

    async def _mcp_catalog_surface(
        self, method: str, params: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Answer one session-scoped MCP catalog request from the owner.

        The agent owns the single orchestrator, so the projection, every
        mutation, login/logout and convergence leg runs here against the
        loop-backed control port; the coordinator holds only projected views
        and revision keys. A session this agent does not serve is not-found,
        and the typed owner key the request carries files the auth bindings
        under a handle held for exactly as long as the key lives.
        """
        body = self._require_wire_session(params)
        owner_key = body.pop("owner_key", None)
        body.pop("sessionId", None)
        if not isinstance(owner_key, str) or not owner_key:
            raise AgentProtocolRefusal(
                ProtocolErrorCode.INVALID_PARAMS, "A typed MCP owner key is required"
            )
        # The wire body carries aliased fields; the dispatch path validates by
        # alias, so the body round-trips through the public params model with
        # the verified session id restored under its field name.
        public_method = AGENT_MCP_CATALOG_PUBLIC_METHODS[method]
        params_model = {
            AGENT_MCP_CATALOG_READ_METHOD: MCPReadParams,
            AGENT_MCP_CATALOG_REFRESH_METHOD: MCPRefreshParams,
            AGENT_MCP_CATALOG_ADD_METHOD: MCPAddParams,
            AGENT_MCP_CATALOG_TOGGLE_METHOD: MCPToggleParams,
            AGENT_MCP_CATALOG_REMOVE_METHOD: MCPRemoveParams,
            AGENT_MCP_CATALOG_LOGIN_METHOD: MCPLoginParams,
            AGENT_MCP_CATALOG_LOGOUT_METHOD: MCPLogoutParams,
        }[method]
        wire_params = params_model.model_validate({
            **body,
            "session_id": self._require_loop().session_id,
        })
        self._current_owner_key = owner_key
        root = self._mcp_owner_root(owner_key)
        try:
            dispatched = await self._mcp_service().dispatch(
                public_method,
                wire_params.model_dump(mode="json", by_alias=True),
                root=root,
                notify=self._mcp_notify_relay,
                owner_key=owner_key,
            )
        except RequestFailure as exc:
            raise AgentProtocolRefusal(exc.code, str(exc)) from exc
        except SessionBackendError as exc:
            raise AgentProtocolRefusal(exc.code, str(exc)) from exc
        return dispatched.response.model_dump(mode="json")

    def _mcp_owner_key_surface(
        self, method: str, params: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Run one typed owner-key lifecycle leg in this process.

        The explicit release on session end replaces the GC weak-drop, and
        the re-key across a session rotation is atomic: the handle is
        unchanged, so the auth bindings survive the rotation instead of
        being silently dropped mid-session.
        """
        self._require_wire_session(params)
        authentication = self._mcp_service().authentication
        if method == AGENT_MCP_OWNER_RELEASE_METHOD:
            key = params.get("owner_key")
            if not isinstance(key, str):
                raise AgentProtocolRefusal(
                    ProtocolErrorCode.INVALID_PARAMS,
                    "A typed MCP owner key is required",
                )
            released = authentication.release_owner_key(key)
            self._mcp_reload_plans.pop(key, None)
            return {"key": key, "released": released, "rekeyed": False}
        old_key = params.get("old_owner_key")
        new_key = params.get("new_owner_key")
        if not isinstance(old_key, str) or not isinstance(new_key, str):
            raise AgentProtocolRefusal(
                ProtocolErrorCode.INVALID_PARAMS,
                "A re-key names the old and the new typed owner key",
            )
        authentication.rekey_owner_key(old_key, new_key)
        plan = self._mcp_reload_plans.pop(old_key, None)
        if plan is not None:
            self._mcp_reload_plans[new_key] = plan
        return {"key": new_key, "released": False, "rekeyed": True}

    async def _mcp_reload_surface(
        self, method: str, params: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Run one config-reload convergence leg in the process that owns the loop.

        prepare reloads the orchestrator and suspends the restrictive changes;
        finish converges and answers the typed converged/partial receipt with
        the runtime the owner resolved; fail restores the restrictive
        suspensions after a failed reload. The plan itself stays agent-side,
        keyed by the owner key, between the legs.
        """
        body = self._require_wire_session(params)
        owner_key = body.pop("owner_key", None)
        if not isinstance(owner_key, str) or not owner_key:
            raise AgentProtocolRefusal(
                ProtocolErrorCode.INVALID_PARAMS, "A typed MCP owner key is required"
            )
        self._current_owner_key = owner_key
        root = self._mcp_owner_root(owner_key)
        service = self._mcp_service()
        try:
            if method == AGENT_MCP_RELOAD_PREPARE_METHOD:
                plan = await service.prepare_config_reload(root, owner_key=owner_key)
                if plan is None:
                    raise AgentProtocolRefusal(
                        ProtocolErrorCode.NOT_IMPLEMENTED,
                        "This agent's session backend cannot prepare MCP convergence",
                    )
                self._mcp_reload_plans[owner_key] = plan
                return {
                    "previous_revision": plan.previous.revision,
                    "restrictive_names": sorted(plan.restrictive_names),
                }
            plan = self._mcp_reload_plans.get(owner_key)
            if method == AGENT_MCP_RELOAD_FINISH_METHOD:
                runtime = await service.finish_config_reload(
                    root, plan, owner_key=owner_key
                )
                self._mcp_reload_plans.pop(owner_key, None)
                return {
                    "converged": runtime is not None,
                    "runtime": (
                        runtime.model_dump(mode="json") if runtime is not None else None
                    ),
                }
            await service.fail_config_reload(root, plan, owner_key=owner_key)
            self._mcp_reload_plans.pop(owner_key, None)
            return {"restored": True}
        except RequestFailure as exc:
            raise AgentProtocolRefusal(exc.code, str(exc)) from exc
        except SessionBackendError as exc:
            raise AgentProtocolRefusal(exc.code, str(exc)) from exc

    async def _recovery_evidence(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """Answer the gate-eligibility evidence request from the owned loop.

        The coordinator-local root-recovery gate's loop-owned predicate inputs
        are built here at reply-write time — the live ``[recovery.root]``
        config, the transcript-integrity verdict projected from the live
        transcript, and the NON-MUTATING availability snapshot — so the wire
        carries one consistent evidence set per decision and no coordinator
        copy of them exists to drift. A request naming any other session is
        the same typed not-found the other carriers answer.
        """
        self._require_wire_session(params)
        loop = self._require_loop()
        evidence = AgentRecoveryEvidence(
            config=loop.config.recovery.root,
            integrity=transcript_integrity_verdict(loop.messages),
            availability=session_availability(loop),
        )
        return {
            "config": evidence.config.model_dump(mode="json"),
            "integrity": evidence.integrity,
            "availability": encode_availability(evidence.availability),
        }

    def _resources(self) -> ResourceRequestHandler:
        """The resource handler bound to the loop this agent owns now."""
        loop = self._require_loop()
        if self._resource_handler is None or self._resource_loop is not loop:
            self._resource_handler = ResourceRequestHandler(
                loop, SessionExecution(), self._drop_resource_notification
            )
            self._resource_loop = loop
        return self._resource_handler

    async def _drop_resource_notification(self, method: str, params: object) -> None:
        """Record a resource notification this route does not publish.

        ``config/reload`` answers with its runtime in the reply, so it emits
        nothing here; a future resource path that does notify is recorded
        rather than silently dropped.
        """
        self._record_diagnostic(
            "the resource handler emitted a notification this route does not carry",
            None,
            method=method,
        )

    def _agent_snapshot(self) -> AgentSessionSnapshot:
        """Build the state a reconfigure reply leaves behind, at reply time."""
        return build_session_snapshot(
            self._require_loop(),
            markers=self._long_ops.records(),
            incarnation_id=self._markers.incarnation_id,
            agent_sequence=self._event_sequence,
        )

    def _cancel_operation(self, params: Mapping[str, Any]) -> None:
        """Cancel the operation this id names; a notification carries no reply.

        The cancel is idempotent and its only assertable outcome is the
        operation's own settlement, so naming an operation this end no longer
        runs is a no-op rather than a diagnostic. A frame that names no
        operation at all is malformed and is recorded.
        """
        op_id = params.get("op_id")
        if not isinstance(op_id, str) or not op_id:
            self._record_diagnostic("a cancel frame carries no operation id", None)
            return
        self._long_ops.cancel(op_id)

    def _policy_service(self) -> PolicyNodeService:
        """The one policy-node service serving this agent's four phases."""
        if self._policy_node is None:
            self._policy_node = PolicyNodeService(self._require_loop())
        return self._policy_node

    def _steering(self) -> AgentSteeringArbiter:
        """The one steering arbiter serving this agent's loop."""
        if self._steering_arbiter is None:
            self._steering_arbiter = AgentSteeringArbiter(
                self._require_loop(),
                active_turn_id=lambda: self._active_turn_id,
                emit_event=self._emit_steer_event,
            )
        return self._steering_arbiter

    async def _emit_steer_event(self, turn_id: str, event: BaseEvent) -> None:
        await self._notify_event(
            AGENT_TURN_EVENT_METHOD, {"turn_id": turn_id, "event": encode_event(event)}
        )

    def _start_notices(self, loop: AgentLoop) -> None:
        """Drain the loop's out-of-band events into turn-less notice frames.

        A loop with no out-of-band source (a scripted test loop) simply emits
        no notices; the serve loop is otherwise unchanged.
        """
        source = getattr(loop, "out_of_band_events", None)
        if source is None:
            return
        self._notices_task = self._track(self._emit_notices(source))

    async def _emit_notices(self, source: Any) -> None:
        """Forward the loop's title, route and background-work events out."""
        try:
            async for event in source():
                try:
                    payload = encode_notice(event)
                except AgentWireError as exc:
                    self._record_diagnostic(
                        f"the out-of-band event does not ride a notice: {exc}", None
                    )
                    continue
                await self._notify_event(AGENT_NOTICE_METHOD, {"event": payload})
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            logger.warning("Dropped the agent's out-of-band notice stream: %s", exc)

    def _start_turn(self, params: Mapping[str, Any]) -> None:
        turn_id = str(params.get("turn_id") or "")
        if not turn_id:
            raise UnsupportedAgentRequestError("a turn request carries no turn id")
        if self._active_turn_id is not None:
            self._track(self._end_rejected_turn(turn_id))
            return
        self._active_turn_id = turn_id
        self._active_turn_task = self._track(self._run_turn(turn_id, params))

    async def _run_turn(self, turn_id: str, params: Mapping[str, Any]) -> None:
        outcome = TURN_OUTCOME_COMPLETED
        error: str | None = None
        fault: BaseException | None = None
        try:
            loop = self._require_loop()
            request = decode_turn_request(params)
            runner, tool_io = self._host_ports_for(params)
            options = request.turn_options or AgentTurnOptions()
            request = replace(
                request,
                subagent_runner=runner,
                tool_io=tool_io,
                turn_options=replace(
                    options, turn_id=turn_id, retry_sink=self._retry_sink(turn_id)
                ),
            )
            async with aclosing(loop.execution_port.run_turn(request)) as events:
                async for event in events:
                    await self._notify_event(
                        AGENT_TURN_EVENT_METHOD,
                        {"turn_id": turn_id, "event": encode_event(event)},
                    )
        except asyncio.CancelledError:
            outcome = TURN_OUTCOME_INTERRUPTED
        except BaseException as exc:
            outcome = TURN_OUTCOME_FAILED
            error = f"{type(exc).__name__}: {exc}"
            fault = exc
        finally:
            failure: RootTurnFailure | None = None
            if fault is not None or outcome is TURN_OUTCOME_INTERRUPTED:
                # R2: the typed terminal projection is captured BEFORE the open
                # user-input requests are cancelled below — absence of
                # callbacks afterwards proves nothing — so the coordinator's
                # gate classifies on pre-finalization evidence and the typed
                # envelope survives the wire on the ended frame. A capture
                # failure never masks the turn's own outcome.
                failure = await self._capture_turn_failure(
                    turn_id,
                    fault,
                    stop_reason=(
                        "user_cancelled"
                        if outcome is TURN_OUTCOME_INTERRUPTED
                        else None
                    ),
                )
            self._active_turn_id = None
            self._active_turn_task = None
            # A request still open when its turn ends has no waiter left to
            # answer it: close it here, exactly as the in-process turn closes
            # its callbacks, so the client never keeps a ghost prompt and the
            # terminal disposition is published before the ended frame.
            with suppress(Exception):
                await self._cancel_open_user_inputs("Turn ended")
            ended: dict[str, Any] = {
                "turn_id": turn_id,
                "outcome": outcome,
                "error": error,
            }
            if failure is not None:
                ended["failure"] = failure.model_dump(mode="json")
            with suppress(Exception):
                await self._notify(AGENT_TURN_ENDED_METHOD, ended)

    async def _capture_turn_failure(
        self, turn_id: str, fault: BaseException | None, *, stop_reason: str | None
    ) -> RootTurnFailure | None:
        """Build the typed terminal projection for the coordinator's gate (R2).

        Mirrors the in-process capture at the terminalization boundary: the
        failure is classified from the recorded observation BEFORE any
        message-only projection discards it, and the interaction/effect
        barriers are read while the turn's records are still open. Total by
        contract: a capture failure never masks the primary failure.
        """
        try:
            loop = self._loop
            if loop is None:
                return None
            if fault is not None and isinstance(fault, Exception):
                # Same capture phase as the in-process boundary (turn, not
                # turn_terminal): a measured transport attempt maps to
                # LLM_CALL, which is what confirms the transient origin.
                record_observation(
                    observe_exception(fault, phase="turn", turn_id=turn_id)
                )
            barrier, barrier_detail = self._interaction_barrier(loop)
            failure = build_root_turn_failure(
                observation=turn_observation(turn_id),
                turn_id=turn_id,
                committed=loop.committed_model,
                stop_reason=stop_reason,
                effects_settled=self._effects_settled(loop, turn_id),
                interaction_barrier=barrier,
                barrier_detail=barrier_detail,
            )
            if (
                failure.failure_class is FailureClass.INCOMPLETE_STREAM
                and not failure.effects_settled
                and await loop.settle_turn_tool_effects(
                    turn_id, grace=_ROOT_EFFECT_SETTLE_GRACE_SECONDS
                )
            ):
                failure = failure.model_copy(update={"effects_settled": True})
            return failure
        except Exception:
            logger.exception("root turn failure projection capture failed")
            return None

    def _interaction_barrier(self, loop: AgentLoop) -> tuple[bool, str]:
        """The interaction barrier read BEFORE callback cancellation (R2):
        an open approval/question record in the loop's own broker. The
        coordinator-side barriers (queued work, handoff, stop) are read by
        the coordinator's gate from its own session view.
        """
        broker = getattr(loop, "request_broker", None)
        records = getattr(broker, "records", None)
        if broker is None or records is None:
            return False, ""
        if any(not record.core_resolved for record in records().values()):
            return True, "an approval or user question callback is still open"
        return False, ""

    def _effects_settled(self, loop: AgentLoop, turn_id: str) -> bool:
        """Whether every live tool invocation of the turn reached a terminal
        state (a live task — including one waiting on an approval — is an
        unsettled effect).
        """
        runner = getattr(loop, "_tool_batch_runner", None)
        invocations = getattr(runner, "invocations", None)
        if not invocations:
            return True
        return not any(
            not invocation.task.done()
            for (identity, _call_id), invocation in invocations.items()
            if identity == turn_id
        )

    async def _cancel_open_user_inputs(self, reason: str) -> None:
        """Close every user-input request this agent leaves unanswered."""
        loop = self._loop
        broker = getattr(loop, "request_broker", None)
        records = getattr(broker, "records", None)
        if broker is None or records is None:
            return
        for request_id, record in list(records().items()):
            if record.core_resolved or record.future.done():
                continue
            await broker.cancel(request_id, RemoteUserInputRejectedError(reason))

    def _interrupt(self, params: Mapping[str, Any]) -> None:
        """Cancel the turn this frame names; a turn this end is not running is
        a no-op, because a notification carries no reply to settle.
        """
        turn_id = str(params.get("turn_id") or "")
        if not turn_id:
            self._record_diagnostic("an interrupt frame carries no turn id", None)
            return
        task = self._active_turn_task
        if task is None or self._active_turn_id != turn_id:
            return
        task.cancel()

    async def _run_shell(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """Run one queued manual ``!`` command where the loop lives.

        The command is the user's own input, so nothing here consults the
        model's permission machinery: the spawn binds the pinned executor
        contract, the credential scrub is that funnel's, and every outcome --
        the empty-command refusal, a spawn failure, an exit, a timeout, an
        interrupt -- renders through the one record builder, whose redaction
        runs before the text leaves this process.
        """
        request = decode_shell_run_request(params)
        loop = self._require_loop()
        cwd = self._shell_cwd(loop, request.cwd)
        max_output_bytes = self._max_shell_output_bytes(loop)
        if not request.command.strip():
            # The one refusal lane: nothing executes, and the record still
            # becomes the user's turn like any other command's.
            refused = build_manual_shell_record(
                ManualShellOutcome(
                    operation_id=request.operation_id,
                    command=request.command,
                    cwd=cwd,
                    status="refused",
                    created_at=now_ms(),
                    error="Shell command cannot be empty",
                ),
                max_output_bytes=max_output_bytes,
            )
            return encode_shell_run_reply(refused.text, refused.context)
        run_params = ShellRunParams(
            session_id=loop.session_id,
            operation_id=request.operation_id,
            command=request.command,
            cwd=cwd,
            timeout_seconds=request.timeout_seconds,
        )
        response: ShellRunResponse | None = None
        failure: Exception | None = None
        started_at = time.monotonic()
        created_at = now_ms()
        try:
            response = secret_redaction.redact_model(
                await self._shell(loop).run(run_params)
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            failure = exc
        record = build_manual_shell_record(
            _manual_shell_outcome(
                run_params,
                response=response,
                failure=failure,
                output_text="",
                duration_ms=(time.monotonic() - started_at) * 1000,
                created_at=created_at,
            ),
            max_output_bytes=max_output_bytes,
        )
        return encode_shell_run_reply(record.text, record.context)

    async def _interrupt_shell(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """Stop the command this request names; one not running runs nothing."""
        operation_id = decode_shell_interrupt(params)
        controller = self._shell_controller
        interrupted = (
            False if controller is None else await controller.interrupt(operation_id)
        )
        return encode_shell_interrupt_reply(interrupted)

    def _shell(self, loop: AgentLoop) -> ShellController:
        """The one controller that owns this process's manual command runs."""
        if self._shell_controller is None:
            self._shell_controller = ShellController(Path(loop.cwd))
        return self._shell_controller

    def _shell_cwd(self, loop: AgentLoop, requested: str | None) -> str:
        """Execute where the user points; only a missing directory is an error."""
        root = Path(loop.cwd).resolve()
        cwd = Path(requested).expanduser().resolve() if requested else root
        if not cwd.is_dir():
            raise AgentShellRefusedError(
                f"Shell working directory does not exist: {cwd}"
            )
        return str(cwd)

    def _max_shell_output_bytes(self, loop: AgentLoop) -> int:
        """The per-stream cap the record's output sections carry."""
        bash_config = loop.tool_manager.get_tool_config("bash")
        if isinstance(bash_config, BashToolConfig):
            return bash_config.max_output_bytes
        return 16_000

    async def _end_rejected_turn(self, turn_id: str) -> None:
        await self._notify(
            AGENT_TURN_ENDED_METHOD,
            {
                "turn_id": turn_id,
                "outcome": TURN_OUTCOME_FAILED,
                "error": "a turn is already running on this agent",
            },
        )

    def _retry_sink(self, turn_id: str) -> RetryObserver:
        async def forward(reason: RetryReason) -> None:
            await self._notify(
                AGENT_TURN_RETRYING_METHOD,
                {"turn_id": turn_id, "reason": encode_retry_reason(reason)},
            )

        return forward

    def _settle_user_input(
        self, settle: Callable[[Mapping[str, Any]], None], params: Mapping[str, Any]
    ) -> None:
        """Settle a host answer without letting a stale answer kill the agent."""
        try:
            settle(params)
        except Exception as exc:
            logger.warning("Dropped an agent user-input answer: %s", exc)

    def _resolve_user_input(self, params: Mapping[str, Any]) -> None:
        loop = self._require_loop()
        loop.execution_port.resolve_user_input(
            str(params.get("request_id")), decode_user_input_result(params["result"])
        )

    def _reject_user_input(self, params: Mapping[str, Any]) -> None:
        loop = self._require_loop()
        loop.execution_port.reject_user_input(
            str(params.get("request_id")),
            RemoteUserInputRejectedError(str(params.get("message") or "rejected")),
        )

    def _relay_user_input(self, params: Mapping[str, Any]) -> None:
        """Adopt a host-relayed user-input request; its record lives here."""
        loop = self._require_loop()
        event_payload = params.get("event")
        if not isinstance(event_payload, Mapping):
            raise UnsupportedAgentRequestError("a relay frame carries no event")
        event = decode_event(event_payload)
        if not isinstance(event, UserInputRequestEvent):
            raise UnsupportedAgentRequestError("a relay frame carries no request event")
        loop.request_broker.relay_user_input_request(event)

    async def _answer_user_input(self, params: Mapping[str, Any]) -> str:
        """Arbitrated answer: the agent-side broker owns the decision."""
        loop = self._require_loop()
        return await loop.request_broker.settle_answer(
            str(params.get("request_id")), decode_user_input_result(params["result"])
        )

    async def _refuse_user_input(self, params: Mapping[str, Any]) -> str:
        """Arbitrated rejection: the agent-side broker owns the decision."""
        loop = self._require_loop()
        return await loop.request_broker.settle_rejection(
            str(params.get("request_id")),
            RemoteUserInputRejectedError(str(params.get("message") or "rejected")),
        )

    async def _keep_waiting_user_input(self, params: Mapping[str, Any]) -> str:
        loop = self._require_loop()
        return await loop.request_broker.keep_waiting(str(params.get("request_id")))

    async def _cancel_user_input(self, params: Mapping[str, Any]) -> None:
        loop = self._require_loop()
        await loop.request_broker.cancel(
            str(params.get("request_id")),
            RemoteUserInputRejectedError(str(params.get("message") or "cancelled")),
        )

    async def _read_page(self, params: Mapping[str, Any]) -> dict[str, Any]:
        loop = self._require_loop()
        agent_id = params.get("agent_id")
        if agent_id != loop.session_id:
            raise UnsupportedAgentRequestError("unknown agent")
        response = await asyncio.to_thread(
            read_live_agent_transcript,
            cast("list[LLMMessage]", loop.messages),
            before=params.get("before"),
            limit=int(params.get("limit") or 50),
        )
        return encode_transcript_page(response)

    def _require_loop(self) -> AgentLoop:
        if self._loop is None:
            raise UnsupportedAgentRequestError("the agent is not open")
        return self._loop

    def _track(self, coroutine: Coroutine[Any, Any, None]) -> asyncio.Task[None]:
        task = asyncio.create_task(coroutine)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def _notify(self, method: str, params: dict[str, Any]) -> None:
        await self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def _next_event_sequence(self) -> int:
        """Mint the agent-sequence stamp for one relayed event frame."""
        self._event_sequence += 1
        return self._event_sequence

    async def _notify_event(self, method: str, params: dict[str, Any]) -> None:
        """Relay one agent event to the coordinator, sequence-stamped."""
        params[AGENT_SEQUENCE_FIELD] = self._next_event_sequence()
        await self._notify(method, params)

    def _schedule_agents_update(self) -> None:
        """Schedule one coalesced child-inventory publication (F10).

        The host's mutation seam calls this synchronously, so scheduling never
        awaits the pipe: a mutation that arrives while a publication is in
        flight only marks the state dirty, and the running pass publishes a
        fresh snapshot at its next loop, never a stale one -- the same
        coalescing contract the in-process registry's emission guard holds.
        """
        self._start_agents_update_pass()

    def _start_agents_update_pass(self) -> None:
        """Arm the dirty flag and start the pass unless one is in flight."""
        self._agents_update_dirty = True
        if self._agents_update_task is None or self._agents_update_task.done():
            self._agents_update_task = self._track(self._emit_agents_updates())

    def _retry_agents_update(self) -> None:
        """Re-run a pass whose delivery failed with a mutation racing it (G6a).

        A failed delivery only re-arms the dirty flag, so on its own the
        racing mutation's snapshot waited for some later mutation. The retry
        re-enters the same coalescing scheduler once this failed pass has
        finished -- the in-flight guard must see a done task, hence the
        deferred start -- and the budget bounds the chain: a persistently
        broken pipe stops retrying here and the dirty flag stays armed for the
        next real mutation.
        """
        if self._child_host is None:
            # Shutdown already released the children: no snapshot to rebuild.
            return
        if self._agents_update_failures >= _AGENTS_UPDATE_RETRY_LIMIT:
            return
        self._agents_update_failures += 1
        asyncio.get_running_loop().call_soon(self._start_agents_update_pass)

    async def _emit_agents_updates(self) -> None:
        """Deliver the dirty inventory snapshots as unsequenced notifications.

        Every payload is a fresh full snapshot of the owner's honest subset --
        ``AgentsInventoryReadResponse``, the same projection the inventory
        read serves -- so a shed or lost frame can only delay the next
        mutation's fresher copy, never strand a delta. A closed or failing
        pipe re-arms the dirty flag instead of unwinding the mutation; a
        mutation that raced the failing delivery buys one bounded retry, and a
        failure with nothing dirty waits for the next mutation. Shutdown
        cancels the pass.
        """
        try:
            while self._agents_update_dirty:
                self._agents_update_dirty = False
                host = self._child_host
                if host is None:
                    return
                await self._notify(
                    AGENT_AGENTS_UPDATE_METHOD,
                    host.inventory_snapshot().model_dump(mode="json"),
                )
            self._agents_update_failures = 0
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            # Read the flag BEFORE re-arming it: it is the tell that a
            # mutation landed during this failed delivery and still needs its
            # own pass.
            raced = self._agents_update_dirty
            self._agents_update_dirty = True
            logger.warning("Dropped an agents/update emission: %s", exc)
            if raced:
                self._retry_agents_update()

    def _schedule_run_completion(self, completion: AgentRunCompletion) -> None:
        """Schedule one settled run's completion for delivery (G1b).

        The host's settle seam calls this synchronously, so a settle never
        awaits the pipe. Unlike the inventory snapshots, completions are
        queued rather than coalesced: every settled run is its own fact and
        its own wake, so the queue is drained in settle order and a failing
        send leaves the entry for the next pass.
        """
        self._run_completions.append(completion)
        if self._run_completion_task is None or self._run_completion_task.done():
            self._run_completion_task = self._track(self._emit_run_completions())

    async def _emit_run_completions(self) -> None:
        """Deliver the queued completions as id-less typed notifications.

        Each frame carries the settled run's own record -- the same typed
        fields the recovery read replays -- so the coordinator's wake owner
        never reconstructs an outcome from a snapshot. A failed send keeps its
        entry at the head of the queue: the next settle retries it, and the
        owner's ledger remains the recovery source either way.
        """
        try:
            while self._run_completions:
                completion = self._run_completions[0]
                await self._notify(
                    AGENT_RUN_COMPLETED_METHOD, encode_run_completion(completion)
                )
                self._run_completions.pop(0)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            logger.warning("Dropped a run-completion emission: %s", exc)

    def _read_run_completions(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """Answer the recovery ledger read from the process that owns the runs.

        The ledger is the notification carrier's backstop: a coordinator whose
        wake frame was lost re-reads it and wakes only the runs its own
        delivered set does not already name.
        """
        self._require_wire_session(params)
        host = self._child_host
        if host is None:
            return encode_run_completions(())
        return encode_run_completions(host._run_completion_ledger())

    async def _send(self, message: dict[str, Any]) -> None:
        await self._transport.send(message)

    async def _shutdown(self) -> None:
        self._fail_host_runs("the peer pipe closed")
        self._fail_pending_requests("the peer pipe closed")
        with suppress(BaseException):
            await self._cancel_open_user_inputs("App server closed")
        current = asyncio.current_task()
        for task in list(self._tasks):
            if task is not current and not task.done():
                task.cancel()
        for task in list(self._tasks):
            if task is not current:
                with suppress(BaseException):
                    await task
        host, self._child_host = self._child_host, None
        if host is not None:
            with suppress(BaseException):
                await host.close()
        shell, self._shell_controller = self._shell_controller, None
        if shell is not None:
            # A command still running when the coordinator goes away belongs to
            # a user turn nobody can receive any more: stop it, don't orphan it.
            with suppress(BaseException):
                await shell.close()
        loop, self._loop = self._loop, None
        if loop is not None:
            await loop.aclose()
        await self._factory.close()


class HostServiceCallError(AgentWireError):
    """A host service the coordinator could not serve, typed for the loop."""


class _WireSettlementPublisher:
    """Publish one terminal user-input disposition where the client can see it.

    The request record lives here, so this process is the one that settles it;
    the projection is the coordinator's, so the terminal state is published
    there. The broker awaits this hook BEFORE its waiter unblocks, which is what
    keeps a reconnect (or a client answering a dead request) from seeing a ghost
    open callback. A publication that cannot be delivered never loses the
    settlement itself -- the waiter still settles -- so the failure is recorded
    and the settlement goes on.
    """

    def __init__(self, server: RootAgentServer) -> None:
        self._server = server

    async def publish_user_input_settlement(
        self, request_id: str, settlement: UserInputSettlement
    ) -> None:
        try:
            await self._server.call_host(
                AGENT_TURN_CALLBACK_METHOD,
                encode_user_input_settlement(request_id, settlement),
            )
        except Exception as exc:
            logger.warning(
                "The terminal callback %s was not published: %s", request_id, exc
            )

    async def after_user_input_settlement(
        self, request_id: str, settlement: UserInputSettlement
    ) -> None:
        """The resumed status is part of the published patch; nothing to add."""
        return None


class WireSubagentRunner:
    """The agent's subagent-runner port, served by the coordinator over the pipe.

    ``run`` streams one run's frames: every frame arrives as an id-less
    ``host/subagent/event`` notification for this run id, and exactly one
    ``host/subagent/ended`` notification closes the generator. A run the
    coordinator cannot serve raises typed instead of streaming an empty result,
    and a closed pipe settles the stream with the transport error.
    """

    def __init__(self, server: RootAgentServer) -> None:
        self._server = server

    def run(
        self, args: TaskArgs, ctx: InvokeContext
    ) -> AsyncGenerator[ToolStreamEvent | TaskResult, None]:
        return self._run(args, ctx)

    async def _run(
        self, args: TaskArgs, ctx: InvokeContext
    ) -> AsyncGenerator[ToolStreamEvent | TaskResult, None]:
        run_id = uuid4().hex
        channel = self._server.open_host_run(run_id)
        try:
            await self._server.call_host(
                SUBAGENT_RUN_METHOD, encode_subagent_run_request(run_id, args, ctx)
            )
            while True:
                kind, payload = await channel.get()
                if kind == "frame":
                    frame = decode_host_run_frame(_frame_payload(payload))
                    yield frame
                    if isinstance(frame, TaskResult):
                        return
                    continue
                result, error = decode_host_run_end(payload)
                if result is not None:
                    yield result
                    return
                raise HostServiceCallError(
                    error or f"the host subagent run {run_id} ended without a result"
                )
        finally:
            self._server.close_host_run(run_id)


class WireToolIO:
    """The agent's tool I/O port, served by the coordinator's host port.

    The capability flags are the ones the turn offer announced (they describe
    the coordinator's port statically); every operation is one agent-origin
    ``host/tool/io`` request answered by id.
    """

    def __init__(
        self, server: RootAgentServer, capabilities: ToolIOCapabilities
    ) -> None:
        self._server = server
        self._capabilities = capabilities

    @property
    def supports_read(self) -> bool:
        return self._capabilities.supports_read

    @property
    def supports_write(self) -> bool:
        return self._capabilities.supports_write

    @property
    def supports_terminal(self) -> bool:
        return self._capabilities.supports_terminal

    async def read_lines(
        self, path: Path, *, start_line: int, limit: int, max_bytes: int
    ) -> BoundedReadResult:
        value = await self._call(
            "read_lines",
            {
                "path": str(path),
                "start_line": start_line,
                "limit": limit,
                "max_bytes": max_bytes,
            },
        )
        return cast("BoundedReadResult", value)

    async def read_text(self, path: Path) -> ReadSafeResult:
        value = await self._call("read_text", {"path": str(path)})
        return cast("ReadSafeResult", value)

    async def write_text(
        self,
        path: Path,
        content: str,
        *,
        encoding: str = "utf-8",
        newline: str | None = None,
    ) -> None:
        await self._call(
            "write_text",
            {
                "path": str(path),
                "content": content,
                "encoding": encoding,
                "newline": newline,
            },
        )

    async def run_shell(self, request: ShellCommandRequest) -> ShellCommandResult:
        value = await self._call(
            "run_shell", {"request": encode_shell_command_request(request)}
        )
        return cast("ShellCommandResult", value)

    async def _call(self, op: str, params: Mapping[str, Any]) -> object:
        payload = await self._server.call_host(
            HOST_TOOL_IO_METHOD, encode_tool_io_request(op, params)
        )
        return decode_tool_io_result(op, payload)


def _frame_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    frame = payload.get("frame")
    if not isinstance(frame, Mapping):
        raise HostServiceCallError("a host subagent event carries no frame payload")
    return frame


# --- item 1 seam 2: the child processes this root agent owns -----------------

DEFAULT_CHILD_ROLLBACK_WAIT = 5.0
"""How long a conservative teardown waits for an attempt's own rollback."""


class AgentChildHostError(RuntimeError):
    """The child host cannot serve this turn's child launch."""


class _UnavailableChildFactory:
    """The in-process child factory a process-owned root never reaches.

    The host exists only on the multi-process route, where every child is its
    own process; a selection that would build a child in this process answers
    typed instead of silently abandoning the isolation rule.
    """

    async def open(self, spec: Any) -> AgentLoop:
        raise AgentChildHostError(
            "this root agent spawned a child on the in-process route; the "
            "in-process escape builds its children in the coordinator"
        )


_MAX_STORED_CHILD_RESULTS = 32
"""How many finished child results this host keeps retrievable (count bound)."""

_MAX_COMPLETION_LEDGER = 32
"""How many settled-run completions the recovery ledger keeps (count bound).

The ledger is the notification carrier's backstop, not a history: it holds the
completions a lost frame could still be recovered from, oldest evicted first.
"""

_CHILD_TELEMETRY_TIMEOUT = 5.0
"""How long one run's child-telemetry read may take before it is abandoned.

The read is an ordinary request/reply to a child that has just finished its
turn, so it answers immediately in practice; the bound keeps a wedged child
from delaying its own settle. A read that fails or times out leaves the
telemetry unset -- unknown stays unknown, never zero.
"""

_CHILD_TELEMETRY_TICK_SECONDS = 1.0
"""How often a RUNNING child's telemetry is observed, and the cadence bound (H1).

A running row's model, context and thinking are the child's own facts, so the
host has to ask the child while it runs; one read per second keeps the row
within a second of the child's truth without turning the wire into a poll
loop. The tick is a floor, not a promise: a read is itself bounded by
``_CHILD_TELEMETRY_TIMEOUT`` and the loop never overlaps two reads, so the
worst-case staleness of a live row is one tick plus one read. A slower tick
would let a browser row drift visibly (and its elapsed measure freeze between
emissions); a faster one buys nothing a client can show, because the emission
coalescer collapses the mutations anyway.
"""


@dataclass(frozen=True)
class _StoredChildResult:
    """One finished run's result, kept separately from its terminal metadata.

    The count bound on stored results and the TTL/cap on terminal metadata are
    independent lifetimes, exactly as in the in-process registry: metadata may
    expire while the result stays retrievable, and a result may expire while
    its closed row stays visible.
    """

    result: TaskResult
    completed_at: float


@dataclass(frozen=True)
class _RunTarget:
    """The resolved target of one result read or wait.

    ``handle`` is the live or terminal handle when this host still holds one,
    and ``result`` is the run's retrievable result (None while it runs, and
    for a foreground run whose result left with its own tool response).
    """

    agent_id: str
    run_id: str
    handle: AgentChildHandle | None
    result: TaskResult | None


def _observed_effective_thinking(observed: AgentSessionSnapshot) -> str | None:
    """The thinking level a child's own config reports for its next request (H1).

    The child's snapshot carries the same estimate its ``/status`` renders: the
    resolved semantic level when the deployment can be encoded, and the level
    the child's active model requested when the estimate cannot be resolved --
    the same declared-level answer the in-process record reports. A snapshot
    with no config projection, or one built by a peer that predates the field,
    reports nothing, and the row keeps rendering unknown.
    """
    config = getattr(getattr(observed, "runtime", None), "config", None)
    if config is None:
        return None
    estimate = getattr(config, "effective_thinking", None)
    if estimate is not None:
        semantic = getattr(estimate, "semantic", None)
        if getattr(estimate, "known", False) and semantic is not None:
            return str(semantic)
        requested = getattr(estimate, "requested", None)
        if requested is not None:
            return str(requested)
        return None
    thinking = getattr(getattr(config, "active_model", None), "thinking", None)
    return str(thinking) if thinking is not None else None


def _observed_compaction_threshold(observed: AgentSessionSnapshot) -> int | None:
    """The compaction trigger the child's own config states (H3).

    The trigger rides the same config view the child publishes with its session
    snapshot -- the value its own auto-compaction middleware compares tokens
    with -- so the row measures usage where that child would compact. A
    snapshot with no config projection, a peer that predates the field, and a
    value the strict codec cannot vouch for all leave the row's denominator
    unknown rather than borrowing a window this host cannot claim.
    """
    config = getattr(getattr(observed, "runtime", None), "config", None)
    active = getattr(config, "active_model", None)
    return stated_compact_threshold(getattr(active, "auto_compact_threshold", None))


_ATTEMPT_HANDLE = re.compile(r"(?:^|-)attempt-([0-9a-f]{32})\Z")


def _short_display_handle(agent_id: str) -> str:
    """The short DISPLAY form of a process-topology handle (R2).

    A published handle is ``{identity}-attempt-<uuid4 hex>``, which reads as
    noise in a human-facing sentence. The short form keeps the attempt marker
    and the attempt uuid's first eight characters; anything that is not
    exactly that shape is returned unchanged, so an unexpected handle is never
    mangled. This is the app-server twin of the TUI's
    ``short_agent_display_id`` (a process layer never imports a widget module);
    both are pinned to the same answers, because the rule must not drift
    between the ack a human reads and the row a browser draws.

    Display-only: the full handle stays in every field and sentence that acts
    on it -- the ack's structured ``agent_id``, its ``agent_id=...``
    continuation instruction, wake text, ``check_agents`` rows and tool
    arguments -- and two attempts sharing eight characters can only ever
    collide on screen, never in a lookup.
    """
    if not agent_id:
        return agent_id
    match = _ATTEMPT_HANDLE.search(agent_id)
    if match is None:
        return agent_id
    return f"attempt-{match.group(1)[:8]}"


@dataclass
class AgentChildHandle:
    """One child process this root agent spawned, keyed by its attempt id.

    ``attempt_id`` is minted here, so a lost aggregate outcome still names its
    attempt: the teardown request the coordinator sends after a lost reply
    resolves through this record. A RETAINED child keeps its process across
    re-tasks; a non-retained one is reaped at completion but keeps a bounded
    terminal record (``closed_at``/``terminal_reason``) projecting as a closed
    row until its metadata TTL or explicit release removes it.

    The telemetry fields are HOST-OBSERVED facts only, because the row that
    renders them must not invent one: the run's start and end are this
    process's monotonic stamps, the task summaries are the task the host was
    asked to run, the turn count is the child's own count of the messages its
    run committed (relayed while the run proceeds, settled by its terminal
    carrier at the end, and never a count of the frames the host saw), and the
    model, context and thinking are read from the child's own session snapshot
    while it runs. A fact this process never observed stays ``None`` and
    renders as unknown.
    """

    agent_id: str
    identity: str
    attempt_id: str
    spawned: Any
    session_id: str
    child_dir: str | None
    retained: bool
    slot: Any
    # The stable per-root display sequence minted at publication: the display
    # name is ``{identity}-{seq}`` because the agent_id itself is identity plus
    # the attempt uuid. None on a handle built outside publication, which keeps
    # the handle-derived fallback.
    seq: int | None = None
    linked: bool = False
    closed: bool = False
    result: TaskResult | None = None
    # The run this handle is (or last was) executing: the management surface
    # pins cancel and guarded release to it, exactly as the registry pins them
    # to a run record. A background run keeps its task here so a cancel can
    # interrupt it without dropping the retained child's process.
    run_id: str | None = None
    task: asyncio.Task[None] | None = None
    # A FOREGROUND run in flight (H6). The runner awaits that run on the
    # caller's own tool stream, so no task ever lands on the handle; without
    # this marker the busy state would read the child idle for the whole run
    # and a concurrent re-task would be admitted onto it. Set before the run's
    # first await and cleared after its settle.
    foreground_run: bool = False
    requested_stop_reason: Any = None
    # The re-task that claimed this idle handle across its publication (J1):
    # the claiming attempt's id, held only while that publication runs. It is
    # what makes "refuse a racing re-task" and "commit this re-task's
    # retention" one atomic step, so two re-tasks of one idle handle admit
    # exactly one run without either mutating the other's handle.
    reuse_claim: str | None = None
    # Host-observed terminal state: the settled outcome every projection and
    # closed row reads. ``idle_since`` is the retained-idle anchor the idle
    # TTL and cap measure from, and ``idle_seconds_at_close`` freezes the idle
    # segment a retirement measured before it cleared that anchor.
    last_run_status: RunStatus | None = None
    last_turns_used: int | None = None
    closed_at: float | None = None
    terminal_reason: str | None = None
    idle_since: float | None = None
    idle_seconds_at_close: float | None = None
    # Run timing (monotonic stamps taken at the run's admission and settle) and
    # the task summaries the runs were launched with.
    run_started_at: float | None = None
    run_finished_at: float | None = None
    initial_task_summary: str | None = None
    current_task_summary: str | None = None
    # What the child itself reported in its session snapshot: the model it
    # committed to, the context it measured and the thinking level its own
    # config resolves. Unset until (and unless) a snapshot read answers.
    effective_model: str | None = None
    base_model: str | None = None
    active_provider: str | None = None
    effective_thinking: str | None = None
    context_tokens: int | None = None
    context_window: int | None = None
    # The child's OWN automatic-compaction trigger, read from the same snapshot
    # (H3). The row renders usage against it, never against the window: the
    # child's compaction fires there. Unset when the child reported none.
    context_threshold: int | None = None

    @property
    def alive(self) -> bool:
        return not self.closed and bool(getattr(self.spawned, "alive", True))

    @property
    def running(self) -> bool:
        """Whether a run is in flight, whichever routing owns it.

        A background run is the handle's own task. A foreground one is awaited
        by the runner on the caller's stream and has no task here, so its own
        marker decides -- and it must, because a second run on one child is
        never admitted over either routing.
        """
        if self.foreground_run:
            return True
        return self.task is not None and not self.task.done()


class AgentChildHost:  # noqa: PLR0904 - the child-host port surface
    """The children one root agent process spawns, links and tears down.

    The root agent owns the session, so it also owns the child launch: the
    candidate resolves against this process's own orchestrator and tool
    authorities, the working-set handoff is rendered from this loop, and the
    parent link is written to this loop's transcript. A child is a process of
    its own (``ProcessSubagentSpawn``) and the host holds the attempt table the
    coordinator's teardown request resolves against.

    The lifecycle mirrors the in-process registry's three independent
    lifetimes. A run settles into a terminal outcome the host observed
    (completed, failed or cancelled), and the result is captured BEFORE the
    process is reaped, so a non-retained child's result survives its process.
    Terminal metadata (a closed row: availability, last-run status, turns,
    latest run id) lives under ``[subagents].tombstone_ttl_seconds`` and
    ``[subagents].max_tombstones`` (oldest first); a zero TTL keeps no
    metadata at all. Stored results live under their own count bound, so
    metadata may expire while the result stays retrievable, and a result may
    expire (answering the typed expired error) while its closed row stays
    visible. Explicit release purges both, and a closed handle refuses a
    re-task with the registry's typed closed error. Retained-idle children are
    retired by ``[subagents].idle_ttl_seconds`` and
    ``[subagents].max_idle_agents`` (zero disables each); the retirement is
    silent on this topology until a completion carrier exists, so a retired
    child simply leaves an ``evicted`` row.

    The rows carry the telemetry this process actually observed (G3) and
    nothing else: run timing from the host's own monotonic stamps, the task
    summaries it launched, the turns the child reported on its terminal
    carrier, and the model and context the child's own snapshot answers. A
    field with no observation source stays unset, so the browser renders
    unknown rather than a fabricated value.

    Root death: a dead root process emits nothing more. The rows the
    coordinator last received therefore linger on the client until a later
    mutation of a live root re-snapshots the inventory -- presentation state
    is only ever changed by an emission, never inferred from silence.
    """

    def __init__(
        self,
        loop: AgentLoop,
        *,
        spawn: Any = None,
        environ: Mapping[str, str] | None = None,
        rollback_wait: float | None = None,
        client_info: ClientInfo | None = None,
        client_capabilities: ClientCapabilities | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._loop = loop
        self._spawn = spawn if spawn is not None else _default_child_spawn(environ)
        self._rollback_wait = (
            DEFAULT_CHILD_ROLLBACK_WAIT if rollback_wait is None else rollback_wait
        )
        self._clock = time.monotonic if clock is None else clock
        # The validated client identity this root opened with rides every
        # fresh process spawn's payload: the child validates it at its own
        # open (``ClientInfo.model_validate``), and a payload without the
        # two keys fails that open deterministically. A host built with no
        # identity (never in production: ``RootLoopFactory.open`` supplies
        # one) keeps the legacy payload shape.
        self._client_info = client_info
        self._client_capabilities = client_capabilities
        self._admission = SubagentAdmission()
        self._cap = loop.config.subagents.max_running_subagents
        # A retained child is held by its stable agent id; every attempt is
        # held by the attempt id the coordinator would name after a lost reply.
        self._children: dict[str, AgentChildHandle] = {}
        self._attempts: dict[str, AgentChildHandle] = {}
        # G2: the display-name sequence. Monotone for this root and never
        # reused, so two children of one identity stay distinguishable and a
        # re-task or a terminal projection never renumbers the handle.
        self._agent_sequence = 0
        # Settled non-retained children keep a bounded terminal record here
        # (closed rows), and their results live in a separately bounded store
        # keyed by (agent, run) -- two lifetimes, exactly as in-process.
        self._terminal: dict[str, AgentChildHandle] = {}
        self._results: dict[tuple[str, str], _StoredChildResult] = {}
        self._expired_results: set[tuple[str, str]] = set()
        self._wait_leases: dict[tuple[str, str], int] = {}
        self._retention_reaper: asyncio.Task[None] | None = None
        self._retention_deadline: float | None = None
        self._tasks: set[asyncio.Task[None]] = set()
        self._runner = AgentChildRunner(self)
        # The agent-side inventory revision (v0.7.9 point 5): ONE monotone
        # counter advanced by a synchronous increment in the same no-await
        # window as each handle mutation, so a snapshot never observes a
        # revision torn from the inventory it projected. The single event
        # loop makes the increment atomic; the coordinator's revision floor
        # compares it only within the loop incarnation the response also
        # names, so a rebind or replacement generation establishes a fresh
        # floor instead of comparing across generations.
        self._inventory_revision = 0
        # F10: the owner-side lifecycle emission. A process-backed root owns
        # its children and their lifecycle moments; the publisher the serving
        # harness binds here is what carries every inventory mutation onto
        # the coordinator's client-facing agents/update. A host built with no
        # publisher keeps the revision discipline and publishes nothing.
        self._inventory_publisher: Callable[[], None] | None = None
        # G1b: the completion carrier. A settled BACKGROUND run is a fact only
        # this process knows, so its settle publishes one typed completion to
        # the serving harness (which forwards it to the coordinator's wake
        # owner) and keeps it in this bounded ledger, which the recovery read
        # replays when a notification frame was lost.
        self._completion_publisher: Callable[[AgentRunCompletion], None] | None = None
        self._completions: list[AgentRunCompletion] = []
        # G5b: the owner this host publishes its children for. A process child
        # asks its spawner for a live decision, so the child's wire client must
        # be bound to the process that holds the policy -- which is the
        # ``RootAgentServer`` this host was installed on, never the host
        # itself. A host built without one (only tests construct one directly)
        # publishes no incarnation, so its children run unbound and deny.
        self._authority_owner: Any = None

    def bind_inventory_publisher(self, publisher: Callable[[], None]) -> None:
        """Bind the emission callback this host's inventory mutations notify.

        The publisher is a synchronous scheduler (the serving harness's
        coalesced agents/update pass), never the send itself, so the mutation
        window stays synchronous and a callback can never block a spawn or a
        release.
        """
        self._inventory_publisher = publisher

    def bind_completion_publisher(
        self, publisher: Callable[[AgentRunCompletion], None]
    ) -> None:
        """Bind the callback one settled background run's completion notifies.

        Like the inventory publisher it is a synchronous scheduler: the run's
        settle records the completion and hands it over without awaiting the
        pipe, so the settle window stays synchronous and a failing send can
        never unwind a child's lifecycle. A host built with no publisher keeps
        the ledger and publishes nothing.
        """
        self._completion_publisher = publisher

    def bind_authority_owner(self, owner: Any) -> None:
        """Bind the process that answers this host's children (G5b).

        The owner supplies the incarnation every spawned child must name in
        its questions and the endpoint its wire client is bound to. A host
        built without one (tests construct one directly) mints no incarnation
        and binds no endpoint, so its children run unbound and deny every
        decision -- the same fail-closed answer an owner too old to carry the
        key gets.
        """
        self._authority_owner = owner

    def _advance_inventory_revision(self) -> None:
        """Advance the monotone handle-inventory revision (held region).

        Every mutation of this host's handle-holding state -- publication,
        re-task run pins, terminal results, cancel stop-reasons, release and
        teardown -- calls this in the mutation's synchronous window, so any
        post-snapshot mutation yields a strictly higher revision for the
        next snapshot.
        """
        self._inventory_revision += 1
        publisher = self._inventory_publisher
        if publisher is None:
            return
        try:
            publisher()
        except BaseException as exc:
            # The revision stays advanced and the mutation stays settled: a
            # failed scheduling hook is a lost emission the next mutation
            # re-delivers, never a broken child lifecycle.
            logger.warning("Dropped an inventory-mutation emission: %s", exc)

    @property
    def _subagent_config(self) -> Any:
        """The ``[subagents]`` retention section in force for this loop."""
        return self._loop.config.subagents

    def _claim_reuse(self, handle: AgentChildHandle, attempt_id: str) -> None:
        """Synchronously claim an idle retained handle for one re-task (J1).

        The claim is taken in the same no-await window as the busy check, so
        of two re-tasks that both observe the handle idle exactly one is
        admitted: the loser observes the claim here and is refused with the
        registry's typed busy error BEFORE publication can commit its
        retention or its link onto the handle the winner's run admits under.
        A running or terminal handle is refused with the same typed errors
        the in-process admission raises. The token is the claiming attempt's
        id, so only that attempt can release the claim.
        """
        if handle.closed:
            raise self._closed_handle_error(handle)
        if handle.running or handle.reuse_claim is not None:
            raise AgentBusyError("Agent is already running a task")
        handle.reuse_claim = attempt_id

    def _release_reuse_claim(self, handle: AgentChildHandle, attempt_id: str) -> None:
        """Release this attempt's reuse claim, and only its own."""
        if handle.reuse_claim == attempt_id:
            handle.reuse_claim = None

    def _apply_reuse_retention(self, handle: AgentChildHandle, retained: bool) -> None:
        """Commit the re-task's retention decision onto a reused handle.

        An explicit ``retain`` argument (including ``retain=false``, which
        makes the continuation the instance's last run) is the only source
        that can change a committed policy; the idle anchor resets because
        the handle is about to run again.

        The handle's running state is re-read HERE, at the moment of
        mutation: a handle that was admitted or closed since the caller's
        check keeps the policy of the run it is executing, and the racing
        re-task is refused instead of rewriting a live run's settle.
        """
        if handle.closed or handle.running:
            raise AgentBusyError("Agent is already running a task")
        if handle.retained == retained and handle.idle_since is None:
            return
        handle.retained = retained
        handle.idle_since = None
        self._advance_inventory_revision()

    # -- terminal metadata and stored results -------------------------------
    #
    # Three independent lifetimes, as in the in-process registry: the process
    # dies at settle, the closed row lives under the metadata TTL/cap, and the
    # result lives under its own count bound.

    def _remember_terminal(self, handle: AgentChildHandle) -> None:
        """Store one closed row under the metadata TTL and oldest-first cap.

        A zero ``tombstone_ttl_seconds`` keeps no metadata at all: the handle
        becomes unknown once reaped while its result stays retrievable, exactly
        like the in-process tombstone build.
        """
        config = self._subagent_config
        if config.tombstone_ttl_seconds > 0:
            self._terminal[handle.agent_id] = handle
        cap = config.max_tombstones
        while cap > 0 and len(self._terminal) > cap:
            oldest_id = min(
                self._terminal,
                key=lambda agent_id: (
                    self._terminal[agent_id].closed_at or 0.0,
                    agent_id,
                ),
            )
            self._terminal.pop(oldest_id, None)
        self._rearm_retention_reaper()

    def _expire_terminal_metadata(self) -> list[str]:
        """Prune expired closed rows; stored results are untouched.

        Returns the pruned agent ids. This is the retention reaper's body and
        the maintenance hook a test can drive against an injected clock; the
        rows leaving the projection advance the inventory revision so the F10
        emission carries the removal.
        """
        ttl = self._subagent_config.tombstone_ttl_seconds
        if ttl <= 0:
            return []
        now = self._clock()
        expired = [
            agent_id
            for agent_id, handle in self._terminal.items()
            if handle.closed_at is not None and now >= handle.closed_at + ttl
        ]
        for agent_id in expired:
            self._terminal.pop(agent_id, None)
        if expired:
            self._advance_inventory_revision()
        return expired

    def _store_result(
        self, handle: AgentChildHandle, run_id: str, result: TaskResult
    ) -> None:
        """Keep one finished run's result under the count bound (insertion)."""
        key = (handle.agent_id, run_id)
        self._results[key] = _StoredChildResult(
            result=result, completed_at=self._clock()
        )
        self._expired_results.discard(key)
        self._expire_results(keep=key)
        self._advance_inventory_revision()

    def _expire_results(self, *, keep: tuple[str, str]) -> None:
        """Evict the oldest unleased results past the count bound.

        An evicted result leaves its expiry marker behind, so a later read of
        that exact run answers the typed expired error while an agent this
        host never held stays unknown. A wait lease pins the result its waiter
        is about to read, exactly as the registry excludes leased keys.
        """
        excess = len(self._results) - _MAX_STORED_CHILD_RESULTS
        if excess <= 0:
            return
        candidates = [
            (key, stored)
            for key, stored in self._results.items()
            if key != keep and not self._wait_leases.get(key)
        ]
        candidates.sort(key=lambda item: (item[1].completed_at, item[0][0], item[0][1]))
        for key, _stored in candidates[:excess]:
            self._results.pop(key, None)
            self._expired_results.add(key)

    def _latest_stored_result(self, agent_id: str) -> tuple[str, TaskResult] | None:
        """The agent's newest retrievable stored result, if any."""
        candidates = [
            (run_id, stored)
            for (owner, run_id), stored in self._results.items()
            if owner == agent_id
        ]
        if not candidates:
            return None
        run_id, stored = max(
            candidates, key=lambda item: (item[1].completed_at, item[0])
        )
        return run_id, stored.result

    def _purge_results(self, agent_id: str) -> bool:
        """Drop one agent's stored results, markers, and leases.

        Returns whether anything was purged, so a release of an
        already-metadata-less agent whose result outlived its row still
        answers released (the in-process result purge keeps the same
        unknown-handle distinction).
        """
        keys = {key for key in self._results if key[0] == agent_id}
        keys.update(key for key in self._expired_results if key[0] == agent_id)
        keys.update(key for key in self._wait_leases if key[0] == agent_id)
        for key in keys:
            self._results.pop(key, None)
            self._expired_results.discard(key)
            self._wait_leases.pop(key, None)
        return bool(keys)

    def _next_retention_delay(self) -> float | None:
        """The earliest metadata/idle deadline, in seconds from now."""
        config = self._subagent_config
        now = self._clock()
        deadlines: list[float] = []
        if config.tombstone_ttl_seconds > 0:
            deadlines.extend(
                handle.closed_at + config.tombstone_ttl_seconds
                for handle in self._terminal.values()
                if handle.closed_at is not None
            )
        if config.idle_ttl_seconds > 0:
            deadlines.extend(
                handle.idle_since + config.idle_ttl_seconds
                for handle in self._children.values()
                if handle.idle_since is not None
            )
        if not deadlines:
            return None
        return max(0.0, min(deadlines) - now)

    def _rearm_retention_reaper(self) -> None:
        """Schedule one retention pass at the earliest live deadline."""
        delay = self._next_retention_delay()
        if delay is None:
            return
        deadline = self._clock() + delay
        existing = self._retention_reaper
        if (
            existing is not None
            and not existing.done()
            and self._retention_deadline is not None
            and self._retention_deadline <= deadline
        ):
            return
        if existing is not None and not existing.done():
            existing.cancel()
        self._retention_reaper = asyncio.create_task(
            self._run_retention_reaper(delay), name="root-agent-child-retention-reaper"
        )
        self._retention_deadline = deadline

    async def _run_retention_reaper(self, delay: float) -> None:
        """One timer pass: expire rows, retire idle retained children, rearm."""
        await asyncio.sleep(delay)
        self._retention_reaper = None
        self._retention_deadline = None
        self._expire_terminal_metadata()
        await self._retire_idle_retained()
        self._rearm_retention_reaper()

    async def _retire_idle_retained(self) -> None:
        """Retire retained-idle children past the idle TTL or the idle cap.

        ``idle_ttl_seconds=0`` disables TTL retirement and
        ``max_idle_agents=0`` disables the cap; the cap retires oldest first
        by idle anchor, exactly as the registry's reaper selects victims. A
        retired child is reaped and leaves an ``evicted`` row -- silent on
        this topology, because no completion carrier reaches the parent yet.
        """
        config = self._subagent_config
        now = self._clock()
        idle = [
            handle
            for handle in self._children.values()
            if handle.idle_since is not None
        ]
        if not idle:
            return
        victims: list[AgentChildHandle] = []
        if config.idle_ttl_seconds > 0:
            victims.extend(
                handle
                for handle in idle
                if now - (handle.idle_since or now) >= config.idle_ttl_seconds
            )
        selected = {id(handle) for handle in victims}
        cap = config.max_idle_agents
        if cap > 0:
            remaining = sorted(
                (handle for handle in idle if id(handle) not in selected),
                key=lambda handle: (handle.idle_since or now, handle.agent_id),
            )
            if len(remaining) > cap:
                victims.extend(remaining[: len(remaining) - cap])
        for handle in victims:
            await self._close_terminal(handle, reason="evicted")

    @property
    def runner(self) -> SubagentRunnerPort:
        """The turn's subagent runner: every child runs in its own process."""
        return self._runner

    @property
    def budget(self) -> Any:
        """The hard child-tree cap this host's spawns draw their slots from."""
        return getattr(self._spawn, "budget", None)

    @property
    def spawn_port(self) -> Any:
        """The spawn adapter this host launches every child through."""
        return self._spawn

    def children(self) -> tuple[AgentChildHandle, ...]:
        """The retained children this host holds, oldest first."""
        return tuple(self._children.values())

    def attempt(self, attempt_id: str) -> AgentChildHandle | None:
        """The attempt this host holds under *attempt_id*, if it still owns it."""
        return self._attempts.get(attempt_id)

    def owned_child(self, agent_id: str) -> AgentChildHandle | None:
        """The live or terminal child this host holds under *agent_id*.

        Authority is answered only for an IMMEDIATE child, so the decision
        endpoint resolves the requesting agent id here first: a frame from a
        process this owner never published is not a child of it, whatever else
        the frame claims.
        """
        handle = self._children.get(agent_id) or self._terminal.get(agent_id)
        if handle is not None:
            return handle
        for attempt in self._attempts.values():
            if attempt.agent_id == agent_id:
                return attempt
        return None

    @staticmethod
    def _require_live_authority_capability(spawned: Any) -> None:
        """Refuse a process child that did not announce live child authority.

        An in-process child carries no wire peer and is not affected. A process
        child whose ready frame lacks the mandatory capability is a version
        skew: the publication fails typed and the attempt's rollback closes the
        process, rather than serving a child on a static authority snapshot no
        live decision could revoke (G5, fail-closed).
        """
        wire_client = getattr(spawned, "wire_client", None)
        if wire_client is None:
            return
        announced = getattr(wire_client, "advertised_capabilities", frozenset())
        if AGENT_WIRE_CAPABILITY_LIVE_CHILD_AUTHORITY not in announced:
            raise AgentChildHostError(
                "the spawned child did not announce the mandatory "
                "live-child-authority capability; restart both ends on the same "
                "chartreux version"
            )

    def _bind_child_authority(self, spawned: Any) -> None:
        """Bind this owner's authority endpoint into a published child (G5b).

        The child's only wire peer is the client that spawned it, so that
        client must serve the child's questions from the process that holds the
        live policy. An in-process child carries no wire peer and needs no
        endpoint; a client that cannot take a binding is left unbound, which
        answers the child's question typed and therefore denies.
        """
        owner = self._authority_owner
        if owner is None:
            return
        wire_client = getattr(spawned, "wire_client", None)
        bind = getattr(wire_client, "bind_child_authority_endpoint", None)
        if bind is None:
            return
        bind(owner._authority_check)

    async def publish(  # noqa: PLR0915 - one statement per publication stage
        self,
        args: TaskArgs,
        ctx: InvokeContext,
        *,
        attempt_id: str,
        retained: bool,
        reuse: AgentChildHandle | None = None,
    ) -> AgentChildHandle:
        """Run one aggregate publication attempt and answer its handle.

        The stages run in order and a failure names the stage that failed: the
        admission fence, the candidate resolution, the working-set handoff, the
        permit refusal a process child cannot consume, the spawn, the durable
        attribution, and the parent link.
        """
        from chartreux.app_server._subagent_launch import (
            ChildPublicationError,
            ChildPublicationStage,
        )
        from chartreux.app_server._subagent_transcript import (
            CLIENT_CAPABILITIES_PAYLOAD_KEY,
            CLIENT_INFO_PAYLOAD_KEY,
            INHERITED_WORKSPACE_PAYLOAD_KEY,
            encode_child_policy_payload,
            encode_inherited_workspace,
        )
        from chartreux.app_server.subagent_ports import (
            CHILD_HANDOFF_PAYLOAD_KEY,
            child_spawn_spec,
        )

        stage = ChildPublicationStage.ADMISSION
        slot: Any = None
        spawned: Any = None
        try:
            # G5: live child authority is mandatory on BOTH sides, so a process
            # that cannot advertise it publishes no child at all. This is the
            # owner's own half of the contract; the child's half is checked on
            # its ready frame below. Neither is ever downgraded to the static
            # inherited policy.
            if (
                AGENT_WIRE_CAPABILITY_LIVE_CHILD_AUTHORITY
                not in ADVERTISED_CAPABILITIES
            ):
                raise AgentChildHostError(
                    "this process does not advertise the mandatory "
                    "live-child-authority capability, so a process child is "
                    "refused rather than served on a static authority snapshot"
                )
            slot = self._admission.reserve(
                cap=self._cap,
                depth=1 if ctx.is_subagent else 0,
                reuse_agent_id=args.agent_id,
                retained_idle=list(self._children),
            )
            agent_name = self._resolved_identity(args).replace(":", "-")
            if reuse is not None and reuse.alive:
                # A retained child keeps its process: the re-task only links
                # the new tool call to the process that already exists. The
                # retention decision still applies, so an explicit
                # ``retain=false`` continuation closes this handle at settle.
                # Both the commit and the link happen inside the claim the
                # runner took, so a re-task refused while this one publishes
                # can never change the policy or the linkage of the run this
                # publication is about to admit. The claim itself is released by
                # this publication's own ``finally``, on every exit.
                self._apply_reuse_retention(reuse, retained)
                stage = ChildPublicationStage.LINKAGE
                await self._link(reuse, ctx, agent_name, reuse.attempt_id)
                self._admission.release(slot)
                return reuse
            stage = ChildPublicationStage.HANDOFF
            root_session_dir = self._loop.session_logger.session_dir
            if root_session_dir is None:
                raise AgentChildHostError(
                    "a process-backed child needs the root session directory: "
                    "its transcript belongs under <root>/agents"
                )
            handoff = self._loop.file_state_handoff_message()
            payload: dict[str, Any] = (
                {CHILD_HANDOFF_PAYLOAD_KEY: handoff.content}
                if handoff is not None and handoff.content
                else {}
            )
            # The child validates its own ClientInfo at open, so the identity
            # this root opened with rides the payload; a host built with none
            # keeps the legacy shape (disclosed: production always sends one).
            if self._client_info is not None:
                from chartreux.app_server._connection_protocol import ClientCapabilities

                payload[CLIENT_INFO_PAYLOAD_KEY] = self._client_info.model_dump(
                    mode="json"
                )
                payload[CLIENT_CAPABILITIES_PAYLOAD_KEY] = (
                    self._client_capabilities or ClientCapabilities()
                ).model_dump(mode="json")
            # A child loop is constructed as a subagent, so it needs the
            # parent workspace the in-process route hands over through
            # ``child_runtime_policy``: it rides the payload the same way,
            # and the child's open fails loudly without it.
            payload[INHERITED_WORKSPACE_PAYLOAD_KEY] = encode_inherited_workspace(
                self._loop.tool_manager.workspace
            )
            # F11: the parent's accepted policy rides the payload the same
            # unconditionally (a child loop is only legal with a policy):
            # restrictions, mode restrictions, dispatch root grants and the
            # budget-bearing session options, all read from this parent loop.
            payload.update(encode_child_policy_payload(self._loop))
            # G5b: the owner incarnation this child must name in every
            # authority question, so its requests are fenced against a
            # superseded owner. A host with no owner bound publishes none, and
            # the child then answers no decision rather than deciding unfenced.
            owner = self._authority_owner
            if owner is not None:
                payload[PARENT_INCARNATION_PAYLOAD_KEY] = encode_parent_incarnation(
                    owner.authority_incarnation
                )
            stage = ChildPublicationStage.PERMIT
            if args.override_cooldown:
                raise AgentChildHostError(
                    "a process-backed child cannot consume a cooldown-override "
                    "permit in this release; the reservation was released"
                )
            spec = child_spawn_spec(
                agent_id=f"{agent_name}-{attempt_id}",
                parent_session_id=self._loop.session_id,
                root_session_dir=root_session_dir,
                agent_name=agent_name,
                payload=payload,
                attempt_id=attempt_id,
            )
            stage = ChildPublicationStage.SPAWN
            spawned = await self._spawn.spawn(spec)
            self._require_live_authority_capability(spawned)
            # G5b: the child's questions must reach THIS process's live policy,
            # so the answering endpoint is bound into the wire client that owns
            # the child's pipe before its first turn can ask anything.
            self._bind_child_authority(spawned)
            stage = ChildPublicationStage.PERSISTENCE
            session_id = getattr(spawned, "session_id", None)
            if not isinstance(session_id, str) or not session_id:
                raise AgentChildHostError(
                    "the spawned child announced no session id, so its first "
                    "durable record cannot be attributed to this attempt"
                )
            handle = AgentChildHandle(
                agent_id=getattr(spawned, "agent_id", spec.agent_id),
                identity=agent_name,
                attempt_id=attempt_id,
                spawned=spawned,
                session_id=session_id,
                child_dir=getattr(spawned, "child_session_dir", None),
                retained=retained,
                slot=slot,
                seq=self._allocate_agent_sequence(),
                # The task the host was asked to run rides the handle the same
                # way the in-process record captures it at admission: the
                # initial summary is the instance's first task and survives a
                # re-task, the current one follows every run.
                initial_task_summary=self._task_summary(args),
            )
            self._attempts[attempt_id] = handle
            self._advance_inventory_revision()
            stage = ChildPublicationStage.LINKAGE
            await self._link(handle, ctx, agent_name, attempt_id)
            if retained:
                self._children[handle.agent_id] = handle
                self._advance_inventory_revision()
            return handle
        except BaseException as exc:
            failed = exc.stage if isinstance(exc, ChildPublicationError) else stage
            raise await self._fail_attempt(
                failed, attempt_id, exc, slot=slot, spawned=spawned
            ) from exc
        finally:
            # K2/H6: the claim this attempt took is released on EVERY exit of
            # the publication -- including the fresh-spawn path a retained
            # child that died while idle takes, which runs no reuse branch at
            # all. A handle this attempt will not run therefore never keeps the
            # claim, so it never answers a permanent busy refusal afterwards.
            if reuse is not None:
                self._release_reuse_claim(reuse, attempt_id)

    async def _fail_attempt(
        self,
        stage: Any,
        attempt_id: str,
        exc: BaseException,
        *,
        slot: Any,
        spawned: Any,
    ) -> Any:
        """Settle a failed attempt conservatively and name its stage."""
        from chartreux.app_server._subagent_launch import ChildPublicationError
        from chartreux.app_server.child_agent_process import teardown_spawned_attempt

        retraction = None
        if spawned is not None:
            report = await teardown_spawned_attempt(
                spawned, attempt_id=attempt_id, wait_seconds=self._rollback_wait
            )
            retraction = report.outcome
            if report.settled:
                self._attempts.pop(attempt_id, None)
                self._advance_inventory_revision()
        if slot is not None:
            self._admission.release(slot)
        return ChildPublicationError(
            stage,
            attempt_id,
            retraction=retraction,
            detail=f"{type(exc).__name__}: {exc}",
        )

    async def _link(
        self,
        handle: AgentChildHandle,
        ctx: InvokeContext,
        agent_name: str,
        attempt_id: str,
    ) -> None:
        """Record the parent link on this loop; a conflict names its stage."""
        from chartreux.app_server._subagent_launch import (
            ChildPublicationError,
            ChildPublicationStage,
            record_process_child_link,
        )

        try:
            await record_process_child_link(
                self._loop,
                session_id=handle.session_id,
                session_dir=handle.child_dir,
                tool_call_id=ctx.tool_call_id,
                agent=agent_name,
                attempt_id=attempt_id,
            )
            handle.linked = True
        except ChildPublicationError:
            raise
        except BaseException as exc:
            raise ChildPublicationError(
                ChildPublicationStage.LINKAGE,
                attempt_id,
                detail=f"{type(exc).__name__}: {exc}",
            ) from exc

    def _allocate_agent_sequence(self) -> int:
        """Mint one published handle's stable display sequence (G2)."""
        self._agent_sequence += 1
        return self._agent_sequence

    @staticmethod
    def _display_name(handle: AgentChildHandle) -> str:
        """The process-topology display name ``{identity}-{sequence}`` (G2).

        The handle's agent_id is the wire handle -- identity plus the attempt
        uuid -- so deriving the name from it re-prefixes the identity and
        prints the uuid (``researcher-researcher-attempt-<hex>``). The host
        names a published handle by the stable per-root sequence instead: a
        number unique within this root, unchanged across a re-task and across
        the terminal projection. The child's committed model never reaches
        this process, so the documented model segment is omitted rather than
        guessed; a handle that bypassed publication has no sequence and keeps
        the handle-derived fallback.
        """
        if handle.seq is None:
            return derive_agent_name(handle.agent_id, handle.identity, None, None, None)
        return f"{handle.identity}-{handle.seq}"

    def _resolved_identity(self, args: TaskArgs) -> str:
        """The launch identity the child commits to, resolved where it can be.

        The canonical resolution reads the parent's own orchestrator, tool
        inventory and route table, so it runs here, in the process that holds
        them, exactly as an in-process launch resolves it.
        """
        from chartreux.core.agents.manager import resolve_identity_candidate

        candidate = resolve_identity_candidate(
            args,
            parent_orchestrator=self._loop.config_orchestrator,
            tool_inventory={
                name: object() for name in self._loop.tool_manager.registered_tools
            },
            authorized_tool_names=frozenset(self._loop.tool_manager.delegable_tools),
            session_model=self._loop.config.active_model,
            user_prompts_dirs=self._loop.harness_files.user_prompts_dirs,
        )
        return candidate.identity

    async def run_child(
        self,
        handle: AgentChildHandle,
        args: TaskArgs,
        ctx: InvokeContext,
        *,
        run_id: str | None = None,
        background: bool = False,
    ) -> TaskResult:
        """Run one turn in a spawned child and answer its result.

        The result is captured and stored BEFORE the run settles, so the
        settle (which reaps a non-retained child and publishes its closed row)
        never races the result write: the row a client observes already carries
        its outcome and turns, and a later read returns the same result. A
        retained child keeps its process (and its place in the cap) after the
        run; a non-retained one is reaped at the same settle but leaves its
        terminal metadata. The run id is recorded on the handle before the turn
        starts, so a cancel or guarded release arriving mid-run pins to the run
        it names. Only a background run's result is retained: a foreground
        result leaves with its own tool response.

        Turn count (G3): the carrier's ``turns_used`` settles the count. The
        frames this loop iterates are NOT turns -- a streaming loop emits one
        ``AssistantEvent`` per chunk and a tool-only step emits none -- so the
        count is never derived from them; what the run's owner stamps beside
        each relayed event (``event_live_turns_used``) is its own running count
        of committed messages, and it keeps a RUNNING row honest. The terminal
        frame still wins at settle, and a run that reports no count anywhere
        leaves the count unproven.

        Live observation (H1): the run's timing stamps and the child's own
        snapshot (model, context, thinking) are observed while it runs, on a
        bounded non-overlapping tick, so the row a browser shows carries them
        before the run ends; the same read is taken once more at settle. Every
        observation is fenced by the run id AFTER its read suspends -- a
        re-tasked or closed handle must never be overwritten by a late
        snapshot -- and a read that fails or is discarded leaves the fields
        alone instead of zeroing them.
        """
        from chartreux.app_server.subagent_ports import SubagentCompletion
        from chartreux.core.subagents import prepare_subagent_prompt

        run_id = run_id or uuid4().hex
        handle.run_id = run_id
        handle.requested_stop_reason = None
        handle.result = None
        handle.last_run_status = None
        handle.last_turns_used = None
        handle.idle_since = None
        handle.idle_seconds_at_close = None
        handle.run_started_at = self._clock()
        handle.run_finished_at = None
        handle.current_task_summary = self._task_summary(args)
        self._advance_inventory_revision()
        parts: list[str] = []
        turns: int | None = None
        outcome = TURN_OUTCOME_FAILED
        error: str | None = None
        cancelled = False
        observation = self._start_child_telemetry_observation(handle, run_id)
        # H6: a foreground run is the caller's own stream, so nothing but this
        # marker can tell an admission that the child is busy. It is set before
        # the run's first await and cleared only ACROSS the settle, so the run
        # is busy from its first frame until it has fully settled.
        handle.foreground_run = not background
        try:
            request = ExecutionTurnRequest(
                message=prepare_subagent_prompt(args.task, ctx),
                turn_options=AgentTurnOptions(turn_id=run_id),
            )
            async with aclosing(
                handle.spawned.completion_source.stream(request)
            ) as frames:
                async for frame in frames:
                    if isinstance(frame, SubagentCompletion):
                        outcome = frame.outcome
                        error = frame.error
                        turns = frame.turns_used
                        continue
                    self._record_live_turns(
                        handle, run_id, event_live_turns_used(frame)
                    )
                    if isinstance(frame, AssistantEvent):
                        if frame.content:
                            parts.append(frame.content)
            # The child's own facts are read once more while its process is
            # still here and still idle (the settle below may reap it). The
            # read stays in this body, not the finally: a cancelled or failed
            # run must reach its own settle without an extra await, and its
            # telemetry simply stays unobserved. The tick is stopped first so
            # the two reads never overlap.
            await self._stop_child_telemetry_observation(observation)
            await self._observe_child_telemetry(handle, run_id=run_id)
        except asyncio.CancelledError:
            cancelled = True
            raise
        finally:
            try:
                await self._stop_child_telemetry_observation(observation)
                result = await self._finish_child_run(
                    handle,
                    run_id,
                    parts=parts,
                    turns=turns,
                    outcome=outcome,
                    error=error,
                    cancelled=cancelled,
                    background=background,
                )
            finally:
                handle.foreground_run = False
        return result

    async def _finish_child_run(
        self,
        handle: AgentChildHandle,
        run_id: str,
        *,
        parts: list[str],
        turns: int | None,
        outcome: str,
        error: str | None,
        cancelled: bool,
        background: bool,
    ) -> TaskResult:
        """Close one run: capture its result, project its outcome, settle it.

        Runs in the ``finally`` of ``run_child``, so every exit path -- a
        completed run, a failed one, and a cancellation -- settles exactly
        once, with the result captured BEFORE the settle (which may reap the
        process) so a closed row and its stored result agree.
        """
        handle.run_finished_at = self._clock()
        response = "\n\n".join(parts)
        if error:
            response = f"{response}\n\n{error}".strip()
        result = TaskResult(
            response=response,
            # ``turns_used`` is a required field of the core model, so an
            # unreported count answers 0 -- the shipped launch-ack convention
            # -- and never a frame count. The projection reads only the proven
            # count (``_projected_turns``), so this 0 is never rendered.
            turns_used=turns if turns is not None else 0,
            completed=not cancelled and outcome == TURN_OUTCOME_COMPLETED,
            stop_reason=(
                handle.requested_stop_reason
                if cancelled and handle.requested_stop_reason is not None
                else None
            ),
            agent_id=(handle.agent_id if handle.retained or background else None),
            run_id=run_id if handle.retained or background else None,
        )
        handle.last_turns_used = turns
        handle.last_run_status = (
            RunStatus.CANCELLED
            if cancelled
            else RunStatus.COMPLETED
            if outcome == TURN_OUTCOME_COMPLETED
            else RunStatus.FAILED
        )
        if background:
            handle.result = result
            self._store_result(handle, run_id, result)
        await self.settle_run(handle, background=background)
        return result

    @staticmethod
    def _task_summary(args: TaskArgs) -> str | None:
        """The bounded summary of the task this run was launched with.

        The in-process route captures the same value at record/run admission
        with the same normalization, so the two topologies project the same
        text for the same launch.
        """
        return normalize_task_summary(args.task_summary, fallback=args.task)

    def _start_child_telemetry_observation(
        self, handle: AgentChildHandle, run_id: str
    ) -> asyncio.Task[None] | None:
        """Start this run's live observation tick, when there is anything to ask.

        A spawn whose adapter exposes no wire client (a test double, a
        pre-wire handle) has no observation source, so a run of one starts no
        task at all rather than spinning one that can never answer.
        """
        read = getattr(getattr(handle.spawned, "wire_client", None), "snapshot", None)
        if read is None:
            return None
        return asyncio.create_task(
            self._child_telemetry_observation_loop(handle, run_id),
            name=f"child-telemetry-{handle.agent_id}",
        )

    async def _child_telemetry_observation_loop(
        self, handle: AgentChildHandle, run_id: str
    ) -> None:
        """Observe one running child once per tick until the run ends (H1).

        The reads are strictly sequential -- each observation is awaited
        before the next sleep -- so two snapshots can never be in flight for
        one run, and the loop stops as soon as the run it was started for is
        no longer the handle's run (a re-task) or the handle is terminal.
        """
        while True:
            await asyncio.sleep(_CHILD_TELEMETRY_TICK_SECONDS)
            if self._observation_is_stale(handle, run_id):
                return
            await self._observe_child_telemetry(handle, run_id=run_id)

    async def _stop_child_telemetry_observation(
        self, task: asyncio.Task[None] | None
    ) -> None:
        """Cancel and await the tick, on every exit path of a run.

        Idempotent, so the normal path can stop it before the settle read and
        the ``finally`` can still call it unconditionally. The task's own
        cancellation is swallowed; a cancellation arriving from OUTSIDE this
        call (the run itself being cancelled) is re-raised, so an in-flight
        cancel of the parent task is never eaten here.
        """
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            if not task.cancelled():
                raise
        except Exception:
            # An observation is never load-bearing: a failed tick is a lost
            # update the next one re-delivers, not a failed child run.
            logger.debug("Child telemetry observer failed", exc_info=True)

    @staticmethod
    def _observation_is_stale(handle: AgentChildHandle, run_id: str) -> bool:
        """Whether a read started for ``run_id`` no longer describes the handle."""
        return handle.closed or handle.run_id != run_id

    def _record_live_turns(
        self, handle: AgentChildHandle, run_id: str, turns_used: int | None
    ) -> None:
        """Project one relayed frame's run-progress count onto the handle (H1).

        The value is the run's owner's own count of committed assistant
        messages, so it is a host-observed fact like the terminal carrier's --
        not a count of the frames that carried it. ``None`` (a frame from a
        peer that reports no progress, or a malformed value) changes nothing,
        and an unchanged count is not a mutation, so a stream of chunks costs
        one revision advance, not one per frame.
        """
        if turns_used is None or self._observation_is_stale(handle, run_id):
            return
        if handle.last_turns_used == turns_used:
            return
        handle.last_turns_used = turns_used
        self._advance_inventory_revision()

    async def _observe_child_telemetry(
        self, handle: AgentChildHandle, *, run_id: str
    ) -> None:
        """Read the child's own snapshot for the facts only it can report (G3).

        The model, the context and the thinking level are the child's, not
        this process's: the child builds its own loop and commits its own
        model, so the host asks it -- the child's session snapshot is the same
        typed read the coordinator already speaks, and its ``committed_model``
        is the identity the in-process record reads from its loop, while its
        config's own effective-thinking estimate is what its ``/status``
        reports. The read is bounded and failure-tolerant (a transport failure
        or an unresponsive child leaves the fields unset), and it changes
        nothing when the child reports nothing.

        The fence is taken AFTER the read suspends and BEFORE any mutation:
        the run can be re-tasked or the handle closed inside that window, and
        a late snapshot must never write over the newer run's rows. Only a
        read that passes the fence is a mutation, so only that read advances
        the inventory revision -- a failed or discarded one leaves it alone.
        """
        client = getattr(handle.spawned, "wire_client", None)
        read = getattr(client, "snapshot", None)
        if read is None:
            return
        try:
            observed: AgentSessionSnapshot = await asyncio.wait_for(
                read(include_history=False), _CHILD_TELEMETRY_TIMEOUT
            )
        except Exception as exc:
            logger.debug("Child telemetry read failed for %s: %s", handle.agent_id, exc)
            return
        if self._observation_is_stale(handle, run_id):
            logger.debug(
                "Discarded a stale child telemetry read for %s", handle.agent_id
            )
            return
        committed = observed.committed_model
        if committed is not None:
            handle.effective_model = format_model_display_name(
                committed.provider, committed.wire_name
            )
            handle.base_model = committed.base_model
            handle.active_provider = committed.provider
        else:
            # The child committed no concrete identity; its own active-model
            # alias is still the model it resolves, and it is exactly the
            # fallback the in-process record uses when nothing is committed.
            handle.effective_model = observed.state.session.model
        thinking = _observed_effective_thinking(observed)
        if thinking is not None:
            handle.effective_thinking = thinking
        stats = observed.runtime.stats
        handle.context_tokens = stats.context_tokens
        handle.context_window = observed.runtime.context_window
        # The child's own compaction trigger rides the config view it publishes
        # with the same snapshot, so a row measures usage at the point this
        # child would compact -- not at the window this host cannot claim.
        handle.context_threshold = _observed_compaction_threshold(observed)
        self._advance_inventory_revision()

    async def settle_run(
        self, handle: AgentChildHandle, *, background: bool = False
    ) -> None:
        """Settle one run: keep a retained child, close and remember the rest.

        A retained child stays resident and becomes retained-idle, which is
        what the idle TTL and idle cap measure from; a non-retained one is
        reaped, but its handle moves into the bounded terminal map so the
        closed row (and the result captured before this settle) survives the
        process. ``background`` marks the run whose outcome the parent model is
        only told about here: a foreground run's result leaves with its own
        tool response, so only a background settle publishes the completion
        carrier.
        """
        if handle.retained:
            self._admission.mark_retained_idle(handle.agent_id)
            self._admission.release(handle.slot)
            handle.idle_since = self._clock()
            self._advance_inventory_revision()
            if background:
                self._publish_run_completion(handle)
            await self._retire_idle_retained()
            self._rearm_retention_reaper()
            return
        await self._close_terminal(handle)
        if background:
            self._publish_run_completion(handle)

    def _publish_run_completion(self, handle: AgentChildHandle) -> None:
        """Record and publish one settled background run's completion (G1b).

        The completion names the run the result store answers for -- the same
        ``(agent_id, run_id)`` pair ``get_agent_result`` takes -- beside the
        display name the inventory row carries, so the parent's wake text and
        its next read agree. It is recorded BEFORE the publisher is called:
        the ledger is what the recovery read replays when the notification
        frame is lost, so a settle is never invisible to the coordinator. A
        handle whose run recorded no observed outcome publishes nothing.
        """
        run_id = handle.run_id
        status = handle.last_run_status
        if run_id is None or status is None:
            return
        completion = AgentRunCompletion(
            session_id=self._loop.session_id,
            agent_id=handle.agent_id,
            run_id=run_id,
            name=self._display_name(handle),
            status=status.value,
            stop_reason=(
                handle.requested_stop_reason.value
                if handle.requested_stop_reason is not None
                else None
            ),
            result_available=(handle.agent_id, run_id) in self._results,
        )
        self._completions.append(completion)
        del self._completions[:-_MAX_COMPLETION_LEDGER]
        publisher = self._completion_publisher
        if publisher is None:
            return
        try:
            publisher(completion)
        except BaseException as exc:
            logger.warning("Dropped a run-completion emission: %s", exc)

    def _purge_completions(self, agent_id: str) -> None:
        """Drop one released agent's ledgered completions (G1b).

        An explicit release removes both of the handle's lifetimes, and the
        ledger describes exactly the results that release purges: keeping an
        entry would let a later sweep wake the parent for a run whose result
        the release already made unknown.
        """
        self._completions = [
            item for item in self._completions if item.agent_id != agent_id
        ]

    def _run_completion_ledger(self) -> tuple[AgentRunCompletion, ...]:
        """The bounded ledger of this process's settled background runs."""
        return tuple(self._completions)

    async def _close_terminal(
        self, handle: AgentChildHandle, *, reason: str = "closed"
    ) -> None:
        """Reap one child's process, keep its closed row, free its capacity.

        A retirement measures the idle segment it is closing before it clears
        the anchor: the evicted row reports that duration, exactly as the
        in-process tombstone carries the eviction's idle time rather than
        losing it with the anchor.
        """
        if handle.closed:
            return
        closed_at = self._clock()
        if handle.idle_since is not None:
            handle.idle_seconds_at_close = max(0.0, closed_at - handle.idle_since)
        handle.closed = True
        handle.closed_at = closed_at
        handle.terminal_reason = reason
        handle.idle_since = None
        self._children.pop(handle.agent_id, None)
        self._attempts.pop(handle.attempt_id, None)
        self._remember_terminal(handle)
        self._advance_inventory_revision()
        try:
            with suppress(Exception):
                await handle.spawned.close()
        finally:
            self._admission.release(handle.slot)
            self._admission.clear_retained(handle.agent_id)

    async def release(self, handle: AgentChildHandle) -> None:
        """Purge one child's terminal metadata and results, and reap it.

        An explicit release removes BOTH lifetimes: the closed row and the
        stored results with their expiry markers, so a later read of the
        handle is unknown rather than expired. The completion ledger entry
        describing those results goes with them, so no later sweep can wake
        the parent for a run the release made unknowable. A released handle is
        reaped exactly once even when it was already closed.
        """
        self._children.pop(handle.agent_id, None)
        self._attempts.pop(handle.attempt_id, None)
        self._terminal.pop(handle.agent_id, None)
        self._purge_results(handle.agent_id)
        self._purge_completions(handle.agent_id)
        self._advance_inventory_revision()
        await self._reap(handle)

    async def _reap(self, handle: AgentChildHandle) -> None:
        """Close a handle's process once and free its capacity and budget slot."""
        if handle.closed:
            self._admission.release(handle.slot)
            self._admission.clear_retained(handle.agent_id)
            return
        handle.closed = True
        try:
            with suppress(Exception):
                await handle.spawned.close()
        finally:
            self._admission.release(handle.slot)
            self._admission.clear_retained(handle.agent_id)

    def admit_background(
        self, handle: AgentChildHandle, args: TaskArgs, ctx: InvokeContext
    ) -> TaskResult:
        """Acknowledge a background launch and run its turn out of band.

        The acknowledgment's metadata is HONEST UNKNOWN (R2): the spawn has
        only just been scheduled, so this process has observed nothing about
        the child's model, provider or thinking level, and the in-process
        route's populated values come from a record the child itself published
        at admission. The keys are present with ``None`` values rather than
        absent, so a reader can tell "unknown yet" from "this route never
        carries it", and no value is ever guessed from the parent.

        A handle whose run is still active refuses the re-task with the
        registry's typed busy error, before any admission state moves: the
        running run keeps its run id and its task, so the refusal never
        overwrites the row state of a run that is still proceeding. This is
        the check the in-process admission makes against a non-idle record,
        and it is synchronous with the run-id mint, so two re-tasks racing
        for the same idle handle still admit exactly one run.

        A CLOSED handle refuses with the registry's typed closed error
        instead (G1a): its process is gone, so admitting a run would mint an
        identity for a child nothing can serve. The refusal is the one the
        reuse admission raises, and it moves no state either.
        """
        if handle.closed:
            raise self._closed_handle_error(handle)
        if handle.running:
            raise AgentBusyError("Agent is already running a task")
        # Every admission is its own run, including a re-task of a retained
        # handle: a reused run id would overwrite the stored result of the run
        # it replaced and make the earlier result unretrievable.
        run_id = uuid4().hex
        handle.run_id = run_id
        task = asyncio.create_task(
            self._run_background(handle, args, ctx, run_id=run_id),
            name=f"root-agent-child-{handle.agent_id}",
        )
        handle.task = task
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        self._advance_inventory_revision()
        return TaskResult(
            response=(
                # The parenthetical is display-only; the continuation
                # instruction below keeps the FULL handle, because that is the
                # string a re-task actually has to pass back.
                f"Background agent {self._display_name(handle)} "
                f"({_short_display_handle(handle.agent_id)}) launched and "
                "running in its own process"
                + (
                    "; this agent is retained and stays resident after the run "
                    f'completes (continue it with agent_id="{handle.agent_id}")'
                    if handle.retained
                    else "; this agent is not retained and exits at completion"
                )
            ),
            turns_used=0,
            completed=True,
            status="launched",
            launch_outcome=LaunchOutcome.LAUNCHED,
            agent_id=handle.agent_id,
            run_id=run_id,
            name=self._display_name(handle),
            metadata={
                "base_model": None,
                "active_provider": None,
                "effective_model": None,
                "effective_thinking": None,
            },
        )

    async def _run_background(
        self,
        handle: AgentChildHandle,
        args: TaskArgs,
        ctx: InvokeContext,
        *,
        run_id: str,
    ) -> None:
        try:
            await self.run_child(handle, args, ctx, run_id=run_id, background=True)
        except asyncio.CancelledError:
            # The cancelled run kept its partial result and settled inside
            # run_child; the task's cancellation still propagates to its owner.
            raise
        except BaseException as exc:
            logger.warning("A background child run failed: %s", exc)

    # -- the subagent-management surface -----------------------------------
    #
    # A process-owned root owns its children, so these operations run here
    # against the handles it spawned and never against a coordinator copy.
    # The typed outcome sets are the in-process registry's; a disposition this
    # host's records cannot represent answers its own typed outcome rather than
    # a false success, and a repeated cancel or release stays safe.

    @property
    def session_id(self) -> str:
        """The root session id that parents this host's children."""
        return self._loop.session_id

    def inventory_snapshot(self) -> AgentsInventoryReadResponse:
        """Project the live handle inventory as ONE typed, revised snapshot.

        The revision and the projected inventory are read together in one
        synchronous pass, so the revision describes EXACTLY the returned
        inventory. The projection is the process-backed owner's honest subset
        of the same wire model the coordinator registry projects: every
        retained child, every running non-retained attempt, and every live
        closed or evicted row, availability pinned to the run the handle
        recorded, the host-observed telemetry (run timing, task summaries,
        turns, and the model and context the child reported), and -- of the
        states a process-backed record cannot carry -- a suspension or
        compaction is never fabricated. The identity is the loop incarnation
        ``(session_id, session_generation)`` the revision floor compares
        within.
        """
        return AgentsInventoryReadResponse(
            session_id=self._loop.session_id,
            generation=self._loop.session_generation,
            revision=self._inventory_revision,
            agents=[
                self._inventory_summary(handle) for handle in self._inventory_handles()
            ],
        )

    def _inventory_handles(self) -> list[AgentChildHandle]:
        """The handles a live inventory read projects: live and terminal.

        Retained and running handles come first in spawn order, followed by
        the live terminal metadata (closed/evicted rows). Deduplication is by
        (agent, attempt), so one handle is never projected twice.
        """
        handles: list[AgentChildHandle] = []
        seen: set[tuple[str, str]] = set()
        for handle in (
            *self._children.values(),
            *self._attempts.values(),
            *self._terminal.values(),
        ):
            key = (handle.agent_id, handle.attempt_id)
            if key in seen:
                continue
            seen.add(key)
            if handle.closed or handle.retained or handle.running:
                handles.append(handle)
        return handles

    @staticmethod
    def _settled_status(result: TaskResult | None) -> str | None:
        """The last-run status a handle's own result implies.

        The fallback for a handle that carries a result but no recorded
        ``last_run_status`` -- a directly constructed handle and a foreground
        run whose result the caller already took -- so the projection keeps
        answering the status the completion implies.
        """
        if result is None:
            return None
        return RunStatus.COMPLETED.value if result.completed else RunStatus.FAILED.value

    def _inventory_summary(self, handle: AgentChildHandle) -> AgentSummaryModel:
        """One handle's wire projection; the process-backed honest subset.

        A live handle projects running/idle with the run it holds; a terminal
        handle projects the closed row the owner observed -- availability
        ``closed`` (or ``evicted`` for a retired retained child), the settled
        last-run status and turns, its latest run id, the typed stop reason,
        and whether that run's result has expired. Fields a process-backed
        record cannot observe (suspension, compaction) stay unset rather than
        fabricated.

        The telemetry a row renders is projected only from what this host
        observed (G3): the run's elapsed time from its monotonic stamps, the
        idle segment from the retained-idle anchor (frozen at retirement for an
        evicted row), the task summaries from the launches this host served,
        the live turn count the run's owner stamped on its relayed events (H1),
        and the model, context and thinking from the child's own snapshot. A
        value with no observation source stays unset and renders as unknown.
        """
        now = self._clock()
        running = handle.running
        evicted = handle.closed and handle.terminal_reason == "evicted"
        if handle.closed:
            availability = (
                AgentAvailability.EVICTED.value
                if evicted
                else AgentAvailability.CLOSED.value
            )
        else:
            availability = (
                RunStatus.RUNNING.value if running else AgentAvailability.IDLE.value
            )
        return AgentSummaryModel(
            agent_id=handle.agent_id,
            identity=handle.identity,
            name=self._display_name(handle),
            availability=availability,
            current_run_id=handle.run_id if running else None,
            current_run_status=RunStatus.RUNNING.value if running else None,
            turns_used=self._projected_turns(handle),
            last_run_status=(
                handle.last_run_status.value
                if handle.last_run_status is not None
                else self._settled_status(handle.result)
            ),
            latest_run_id=handle.run_id,
            stop_reason=(
                PublicRunStopReason(handle.requested_stop_reason.value)
                if handle.requested_stop_reason is not None
                else None
            ),
            result_expired=(
                handle.run_id is not None
                and (handle.agent_id, handle.run_id) in self._expired_results
            ),
            initial_task_summary=handle.initial_task_summary,
            current_task_summary=handle.current_task_summary,
            idle_seconds=self._projected_idle_seconds(
                handle, now=now, evicted=evicted, running=running
            ),
            run_elapsed_seconds=self._projected_run_elapsed(handle, now=now),
            effective_model=handle.effective_model,
            base_model=handle.base_model,
            active_provider=handle.active_provider,
            effective_thinking=handle.effective_thinking,
            context_tokens=handle.context_tokens,
            context_window=handle.context_window,
            context_threshold=handle.context_threshold,
        )

    @staticmethod
    def _projected_turns(handle: AgentChildHandle) -> int | None:
        """The latest run's turn count, only when its provenance is proven.

        The count is always the child's own: the run-progress value its owner
        stamps on each relayed event while the run proceeds (H1), replaced by
        its terminal carrier's value at settle. A frame count is never
        substituted: the stream a child sends is not a turn stream. A handle
        this host never ran (no run stamp) answers its result's own count,
        exactly as ``_settled_status`` reads that result for the status a
        directly constructed record can represent.
        """
        if handle.last_turns_used is not None:
            return handle.last_turns_used
        if handle.run_started_at is None and handle.result is not None:
            return handle.result.turns_used
        return None

    @staticmethod
    def _projected_run_elapsed(handle: AgentChildHandle, *, now: float) -> float | None:
        """The latest run's elapsed seconds from its monotonic stamps.

        While the run is live the stamp is the run's start, so successive
        snapshots report a growing duration; once it settles the end stamp
        freezes the same measure, so a closed row's last-run duration never
        drifts.
        """
        started = handle.run_started_at
        if started is None:
            return None
        end = handle.run_finished_at if handle.run_finished_at is not None else now
        return max(0.0, end - started)

    @staticmethod
    def _projected_idle_seconds(
        handle: AgentChildHandle, *, now: float, evicted: bool, running: bool
    ) -> float | None:
        """The idle segment a row reports, in the state it is in.

        A retained-idle child measures from its anchor; a retired child keeps
        the segment its retirement measured; a running child has no idle
        segment, and a plain close keeps none (the in-process close-on-
        completion tombstone reports the same zero).
        """
        if evicted:
            return handle.idle_seconds_at_close
        if handle.closed:
            return 0.0
        if running or handle.idle_since is None:
            return None
        return max(0.0, now - handle.idle_since)

    async def check_agents(self) -> list[AgentSummary]:
        """Project the handle inventory as the tool surface's summaries.

        The process-backed boundary is the inventory read's honest subset: a
        handle carries no suspension, compaction or recovery state, so those
        core fields stay unset, and the identity is carried by the agent id,
        the canonical identity and its derived display name (never a second
        handle). The host-observed telemetry (run timing, task summaries, turns
        with their proven provenance, and the model, thinking and context the
        child reported) rides here exactly as it rides the wire row:
        ``check_agents`` and the wire inventory read project the SAME handles
        through the SAME projection, closed rows included. The current
        projection supplies every core field the process route can observe --
        including the thinking level and the compaction threshold that
        denominator the row renders usage against -- so the tool surface and
        the wire row never disagree about the same handle.
        """
        return [
            AgentSummary(
                agent_id=summary.agent_id,
                identity=summary.identity,
                name=summary.name,
                availability=AgentAvailability(summary.availability),
                current_run_id=summary.current_run_id,
                current_run_status=(
                    RunStatus(summary.current_run_status)
                    if summary.current_run_status is not None
                    else None
                ),
                turns_used=summary.turns_used,
                last_run_status=(
                    RunStatus(summary.last_run_status)
                    if summary.last_run_status is not None
                    else None
                ),
                latest_run_id=summary.latest_run_id,
                stop_reason=(
                    RunStopReason(summary.stop_reason)
                    if summary.stop_reason is not None
                    else None
                ),
                initial_task_summary=summary.initial_task_summary,
                current_task_summary=summary.current_task_summary,
                idle_seconds=summary.idle_seconds,
                run_elapsed_seconds=summary.run_elapsed_seconds,
                effective_model=summary.effective_model,
                base_model=summary.base_model,
                active_provider=summary.active_provider,
                # J2: the wire row carries these, so the tool surface must
                # too -- the row a caller reads here is the row the browser
                # renders, and an unobserved value stays None (fail-closed)
                # exactly as it does on the wire.
                effective_thinking=summary.effective_thinking,
                context_tokens=summary.context_tokens,
                context_window=summary.context_window,
                context_threshold=summary.context_threshold,
            )
            for summary in (
                self._inventory_summary(handle) for handle in self._inventory_handles()
            )
        ]

    def _manageable(self, agent_id: str) -> AgentChildHandle | None:
        """The child this host holds under *agent_id*: live, closed, or attempted."""
        handle = self._children.get(agent_id)
        if handle is not None:
            return handle
        handle = self._terminal.get(agent_id)
        if handle is not None:
            return handle
        return next(
            (item for item in self._attempts.values() if item.agent_id == agent_id),
            None,
        )

    @staticmethod
    def _closed_handle_error(handle: AgentChildHandle) -> AgentClosedError:
        """The registry's typed reuse refusal for a terminal handle."""
        if handle.terminal_reason == "evicted":
            return AgentEvictedError(
                f"Agent evicted: {handle.agent_id}; start a new agent"
            )
        return AgentClosedError(
            "This instance is closed; launch a new task without agent_id. "
            "Use retain=true when a resident continuation is required."
        )

    def _resolve_run_target(self, agent_id: str, run_id: str | None) -> _RunTarget:
        """Resolve the exact run a result read or wait targets, or the typed error.

        A latest-run read resolves to the handle's recorded run, or -- once
        the terminal metadata expired -- to the newest stored result, so a
        result outlives its row. A pinned run resolves from the result store
        first, then the handle's own retained result. An evicted result keeps
        its expiry marker and answers the typed expired error; a run this host
        never recorded (or a released one) stays typed unknown.
        """
        handle = self._manageable(agent_id)
        if run_id is None:
            run_id = handle.run_id if handle is not None else None
            if run_id is None:
                latest = self._latest_stored_result(agent_id)
                if latest is not None:
                    return _RunTarget(agent_id, latest[0], None, latest[1])
                if handle is None:
                    raise UnknownAgentError(f"Unknown agent: {agent_id}")
                raise UnknownAgentError(f"Unknown run {run_id} for agent {agent_id}")
        key = (agent_id, run_id)
        if key in self._expired_results:
            raise AgentResultExpiredError(
                f"Result expired for run {run_id} on agent {agent_id}"
            )
        stored = self._results.get(key)
        if stored is not None:
            return _RunTarget(agent_id, run_id, handle, stored.result)
        if handle is not None and handle.run_id == run_id:
            return _RunTarget(agent_id, run_id, handle, handle.result)
        raise UnknownAgentError(f"Unknown run {run_id} for agent {agent_id}")

    async def cancel_run(  # noqa: PLR0911 - one branch per typed disposition
        self,
        agent_id: str,
        run_id: str | None = None,
        *,
        reason: RunStopReason,
        requester_session_id: str,
    ) -> CancelResult:
        """Stop-request one run this host owns, with the registry's outcomes.

        The request pins to the run the handle recorded, so a stale run id is
        an unknown run rather than a stop of whatever runs now. A background
        run is interrupted in place; a retained child keeps its process, so a
        later re-task stays possible.
        """
        handle = self._manageable(agent_id)
        if handle is None:
            return CancelResult(outcome=CancelOutcome.UNKNOWN_RUN, run_id=run_id)
        if reason is not RunStopReason.USER_CANCELLED and (
            requester_session_id != self._loop.session_id
        ):
            return CancelResult(outcome=CancelOutcome.FORBIDDEN, run_id=run_id)
        target = run_id or handle.run_id
        if run_id is not None and handle.run_id is not None and run_id != handle.run_id:
            return CancelResult(outcome=CancelOutcome.UNKNOWN_RUN, run_id=run_id)
        if not handle.running:
            if handle.result is not None:
                return CancelResult(
                    outcome=CancelOutcome.ALREADY_FINISHING,
                    run_id=target,
                    stop_reason=handle.requested_stop_reason,
                )
            return CancelResult(outcome=CancelOutcome.NOT_RUNNING, run_id=target)
        if handle.requested_stop_reason is not None:
            return CancelResult(
                outcome=CancelOutcome.ALREADY_STOPPING,
                run_id=target,
                stop_reason=handle.requested_stop_reason,
            )
        if handle.task is None:
            # A foreground run is in flight on the caller's own stream, so this
            # host holds no task to interrupt: the stop stays unavailable, and
            # no stop reason is recorded for a request nothing would act on.
            return CancelResult(outcome=CancelOutcome.NOT_RUNNING, run_id=target)
        handle.requested_stop_reason = reason
        self._advance_inventory_revision()
        assert handle.task is not None
        handle.task.cancel()
        return CancelResult(
            outcome=CancelOutcome.STOP_REQUESTED, run_id=target, stop_reason=reason
        )

    async def release_guarded(
        self, agent_id: str, expected_run_id: str | None = None
    ) -> ReleaseAgentOutcome:
        """Remove a retained child's handle or one closed row under guarded rails.

        A running background run or a latest-run mismatch is a conflict that
        released nothing; an agent this host no longer holds (released,
        metadata-expired, or never seen) is unknown. A handle an in-flight
        re-task has claimed is a conflict too: the run that claim admits has no
        handle to release yet, so a release answering success there would purge
        the resident the very next admission runs on. Releasing a live handle
        answers ``RELEASED`` and removes it; releasing a closed/evicted row
        answers ``EVICTED`` and purges its metadata together with its stored
        results, so a repeat answers unknown, never a second release.
        """
        handle = self._manageable(agent_id)
        if handle is None:
            return ReleaseAgentOutcome.UNKNOWN
        if handle.running or handle.reuse_claim is not None:
            return ReleaseAgentOutcome.CONFLICT
        if (
            expected_run_id is not None
            and handle.run_id is not None
            and expected_run_id != handle.run_id
        ):
            return ReleaseAgentOutcome.CONFLICT
        terminal = handle.closed
        await self.release(handle)
        return ReleaseAgentOutcome.EVICTED if terminal else ReleaseAgentOutcome.RELEASED

    async def release_agent(self, agent_id: str) -> ReleaseAgentOutcome:
        """Purge one child under the model tool's unguarded release rails.

        Unguarded by design (the tool may release active instances): a run
        still executing is interrupted and joined first, so the reaped child
        cannot outlive the release, and the release removes BOTH lifetimes --
        the closed row (``EVICTED``) and every stored result with its expiry
        markers, so a later read is unknown rather than expired. A live
        handle answers ``RELEASED``; an agent whose metadata already expired
        but whose result survives is still purged and answers ``RELEASED``
        (the in-process result purge keeps the same distinction); a handle
        this host never held answers the typed unknown.

        A handle an in-flight re-task holds claimed is refused with the
        registry's typed busy error: the claim IS that re-task's admission, and
        purging the handle it is about to run on would answer a release the very
        next admission contradicts. The refusal moves nothing, so the
        publication finishes and a later release retires the run it admitted.
        """
        handle = self._manageable(agent_id)
        if handle is None:
            if not self._purge_results(agent_id):
                raise UnknownAgentError(f"Unknown agent: {agent_id}")
            self._purge_completions(agent_id)
            self._advance_inventory_revision()
            return ReleaseAgentOutcome.RELEASED
        if handle.reuse_claim is not None:
            raise AgentBusyError("Agent is already running a task")
        if handle.closed:
            await self.release(handle)
            return ReleaseAgentOutcome.EVICTED
        task = handle.task
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        await self.release(handle)
        return ReleaseAgentOutcome.RELEASED

    async def resume_agent(
        self,
        agent_id: str,
        expected_run_id: str,
        idempotency_key: str,
        *,
        requester_session_id: str,
        override_cooldown: bool = False,
    ) -> ResumeAgentResult:
        """Answer warm resume typed: this host keeps no suspended run.

        A process-backed child that fails is reaped or retained as a completed
        handle — there is no warm suspended session to restore — so the typed
        disposition names that instead of scheduling a run this process could
        not restore. The failed run's result and pinned waits stay immutable.
        ``override_cooldown`` carries the port's full signature but changes
        nothing here: the forced-attempt permit is minted only for a ledger
        episode, and this host has no suspension to consult on the way to
        the typed not-resumable disposition.
        """
        handle = self._manageable(agent_id)
        if (
            handle is not None
            and handle.run_id is not None
            and handle.run_id != expected_run_id
        ):
            return ResumeAgentResult(
                outcome=ResumeOutcome.NOT_RESUMABLE,
                agent_id=agent_id,
                run_id=expected_run_id,
                reason="expected_run_id does not match the agent's latest run",
                reason_codes=["stale_expected_run_id"],
            )
        return ResumeAgentResult(
            outcome=ResumeOutcome.NOT_RESUMABLE,
            agent_id=agent_id,
            run_id=expected_run_id,
            reason=(
                "Process-backed children are not preserved for warm resume; "
                "replacement remains permitted"
            ),
            reason_codes=["not_suspended"],
        )

    async def reconcile_result(
        self,
        agent_id: str,
        expected_run_id: str,
        transcript_boundary: str,
        verified_effects: list[EffectVerification],
        *,
        requester_session_id: str,
    ) -> ReconcileResult:
        """Answer a barrier submission typed: no barrier is pending here.

        Process-backed children never suspend into a reconcile barrier, so a
        submission changes nothing and says so — stale or duplicate-looking
        evidence never clears a barrier this host does not hold.
        """
        return ReconcileResult(
            outcome=ReconcileOutcome.NOT_PENDING,
            agent_id=agent_id,
            run_id=expected_run_id,
            transcript_boundary=transcript_boundary,
            reason="No reconcile barrier is pending for this agent",
            reason_codes=["no_reconcile_barrier"],
        )

    async def get_agent_result(
        self, agent_id: str, run_id: str | None = None
    ) -> TaskResult | None:
        """Read one run's retained result, nonblocking, with the registry's errors.

        A run still executing answers None; a finished background run answers
        the result the host captured before the settle -- the latest run when
        no run id is given, and any earlier retained run when one is (a
        re-task's older results stay retrievable under their own count bound).
        A result evicted by that bound answers the typed expired error even
        while its closed row is still visible; an agent or run this host never
        recorded -- or one already released -- answers the typed unknown. A
        foreground run's result left with its own tool response and is
        retained nowhere on this host, so a read of that run keeps the expired
        error rather than a fabricated copy.
        """
        target = self._resolve_run_target(agent_id, run_id)
        handle = target.handle
        if handle is not None and handle.running and handle.run_id == target.run_id:
            return None
        if target.result is None:
            raise AgentResultExpiredError(
                f"Result expired for run {target.run_id} on agent {agent_id}"
            )
        return target.result

    def _still_running_disposition(
        self, agent_id: str, handle: AgentChildHandle, wait_started_at: float
    ) -> WaitForAgentResult:
        """One still-running wait disposition for a handle-owned run."""
        assert handle.run_id is not None
        elapsed = max(0.0, time.monotonic() - wait_started_at)
        return WaitForAgentResult(
            status=WaitStatus.STILL_RUNNING,
            agent_id=agent_id,
            run_id=handle.run_id,
            elapsed_seconds=elapsed,
            run_status=RunStatus.RUNNING,
            turns_used=0,
            guidance=still_running_guidance(
                agent_id=agent_id,
                run_id=handle.run_id,
                elapsed_seconds=elapsed,
                run_status=RunStatus.RUNNING,
                turns_used=0,
            ),
        )

    async def wait_for_agent(
        self, agent_id: str, run_id: str | None = None, *, timeout: float | None = None
    ) -> WaitForAgentResult:
        """Block until a run this host owns completes, with the typed disposition.

        ``completed`` carries the run's retained TaskResult; a run that
        already finished (or whose handle closed while the wait was issued)
        resolves immediately from the stored result. A bounded timeout
        unblocks the waiter only (the run's task is shielded, so the timeout
        never cancels the run). A run stopped while waited resolves to the
        partial result it kept instead of unwinding the wait, and a finished
        run whose result is not retained keeps the typed expired error, never
        a fabricated one. The still-running progress a handle can represent is
        its run identity and running status: the in-flight turn count is not
        tracked on a process handle, so the disposition carries zero.
        """
        if timeout is not None and timeout <= 0:
            raise ValueError("timeout must be positive or None")
        target = self._resolve_run_target(agent_id, run_id)
        wait_started_at = time.monotonic()
        handle = target.handle
        if (
            handle is not None
            and handle.running
            and handle.run_id == target.run_id
            and handle.task is not None
        ):
            task = handle.task
            key = (agent_id, target.run_id)
            self._wait_leases[key] = self._wait_leases.get(key, 0) + 1
            try:
                try:
                    await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
                except TimeoutError:
                    return self._still_running_disposition(
                        agent_id, handle, wait_started_at
                    )
                except asyncio.CancelledError:
                    current = asyncio.current_task()
                    if current is not None and current.cancelling():
                        raise
                    if not task.done():
                        raise
                    # The pinned run itself was stopped; it kept its partial
                    # result, so the wait resolves instead of unwinding.
            finally:
                remaining = self._wait_leases.get(key, 0) - 1
                if remaining > 0:
                    self._wait_leases[key] = remaining
                else:
                    self._wait_leases.pop(key, None)
            # The settle may have closed the handle; re-resolve so the read
            # sees the stored result the settle captured.
            target = self._resolve_run_target(agent_id, target.run_id)
        if target.result is None:
            raise AgentResultExpiredError(
                f"Result expired for run {target.run_id} on agent {agent_id}"
            )
        return WaitForAgentResult(status=WaitStatus.COMPLETED, result=target.result)

    async def read_transcript_page(
        self, agent_id: str, *, before: str | None = None, limit: int = 50
    ) -> AgentTranscriptGetResponse:
        """Read one page of a child's transcript from where it lives.

        A live child answers through its own page reader over its own pipe;
        once the process is gone, its saved transcript under the child
        directory is read instead. An agent this host never spawned is
        not-found.
        """
        handle = self._manageable(agent_id)
        if handle is None:
            raise UnknownAgentError(f"Unknown agent: {agent_id}")
        wire_client = getattr(handle.spawned, "wire_client", None)
        if handle.alive and wire_client is not None:
            return await wire_client.read_page(agent_id, before=before, limit=limit)
        return await asyncio.to_thread(self._read_saved_page, handle, before, limit)

    def _read_saved_page(
        self, handle: AgentChildHandle, before: str | None, limit: int
    ) -> AgentTranscriptGetResponse:
        """Read a dead child's saved page, contained under the root session."""
        child_dir = handle.child_dir
        root_dir = self._loop.session_logger.session_dir
        if child_dir is None or root_dir is None:
            return AgentTranscriptGetResponse(
                state=AgentTranscriptState.NO_SAVED_TRANSCRIPT
            )
        child_path = Path(child_dir)
        try:
            child_path.relative_to(Path(root_dir))
        except ValueError as exc:
            raise AgentChildHostError(
                "the child transcript is outside the root session directory"
            ) from exc
        return read_agent_transcript(
            child_path, lambda: True, before=before, limit=limit
        )

    async def teardown_attempt(self, request: ChildTeardownRequest) -> Any:
        """Settle one publication attempt this agent owns; never claim success.

        The coordinator minted nothing here, but it holds the attempt id the
        agent answered with, so a lost aggregate outcome still names its
        attempt: an attempt still running is cancelled, given the bounded
        rollback wait, and only then adjudicated — the typed inconclusive class
        when it cannot be settled, never a claimed rollback.
        """
        from chartreux.app_server.child_agent_process import (
            DEFAULT_CHILD_ROLLBACK_WAIT,
            teardown_spawned_attempt,
        )
        from chartreux.app_server.subagent_ports import (
            ChildTeardownOutcome,
            ChildTeardownReport,
        )

        handle = self._attempts.get(request.attempt_id)
        if handle is None:
            return ChildTeardownReport(
                attempt_id=request.attempt_id,
                outcome=ChildTeardownOutcome.ATTEMPT_UNKNOWN,
                agent_id=request.agent_id,
                detail="this agent holds no attempt under that id",
            )
        report = await teardown_spawned_attempt(
            handle.spawned,
            attempt_id=request.attempt_id,
            reason=request.reason,
            wait_seconds=(
                DEFAULT_CHILD_ROLLBACK_WAIT
                if request.wait_seconds is None
                else request.wait_seconds
            ),
        )
        if report.settled:
            await self.release(handle)
        return report

    async def close(self) -> None:
        """Reap every child this host still owns, retained or not."""
        tasks, self._tasks = list(self._tasks), set()
        for task in tasks:
            task.cancel()
        for task in tasks:
            with suppress(BaseException):
                await task
        for handle in (
            list(self._children.values())
            + list(self._attempts.values())
            + list(self._terminal.values())
        ):
            await self.release(handle)
        self._results.clear()
        self._expired_results.clear()
        self._wait_leases.clear()
        # A child settling inside the cancellation above still re-arms the
        # retention reaper, so the reaper is detached only once no settle can
        # follow.
        reaper = self._retention_reaper
        self._retention_reaper = None
        self._retention_deadline = None
        if reaper is not None and not reaper.done():
            reaper.cancel()
            with suppress(BaseException):
                await reaper


class AgentChildRunner:
    """The turn's subagent runner: every ``task`` call spawns a child process.

    The turn offer is what grants this turn its children; the runner is the
    agent's own, because the launch authorities (the orchestrator, the tool
    inventory, the transcript, the session logger) live in this process. A
    background launch is acknowledged and keeps running here; a foreground one
    blocks the ``task`` call until the child's terminal frame.
    """

    def __init__(self, host: AgentChildHost) -> None:
        self._host = host

    def run(
        self, args: TaskArgs, ctx: InvokeContext
    ) -> AsyncGenerator[ToolStreamEvent | TaskResult, None]:
        return self._run(args, ctx)

    async def _run(
        self, args: TaskArgs, ctx: InvokeContext
    ) -> AsyncGenerator[ToolStreamEvent | TaskResult, None]:
        from chartreux.app_server._subagent_launch import new_publication_attempt_id
        from chartreux.app_server.subagent_ports import (
            MAX_SUBAGENT_DEPTH,
            SubagentDepthDeniedError,
        )

        if ctx.is_subagent:
            raise SubagentDepthDeniedError(depth=1, max_depth=MAX_SUBAGENT_DEPTH)
        reuse = next(
            (
                child
                for child in self._host.children()
                if args.agent_id is not None and child.agent_id == args.agent_id
            ),
            None,
        )
        if args.agent_id is not None and reuse is None:
            # A named handle this host no longer holds is never silently
            # replaced by a fresh child: a closed row refuses with the
            # registry's typed reuse error and an unknown id stays unknown.
            closed = self._host._manageable(args.agent_id)
            if closed is not None:
                raise self._host._closed_handle_error(closed)
            raise UnknownAgentError(
                f"Unknown agent_id instance handle: {args.agent_id}. "
                "Check the agent_id with check_agents"
            )
        if reuse is not None and not reuse.closed and reuse.running:
            # The in-process admission refuses a non-idle record with the same
            # typed error, and the refusal is made HERE as well: refusing
            # before publication leaves the running run's link, retention and
            # inventory row exactly as they were.
            raise AgentBusyError("Agent is already running a task")
        attempt_id = new_publication_attempt_id()
        if reuse is not None:
            # The claim is the admission for a re-task: it refuses a busy,
            # terminal or already-claimed handle in the same no-await window,
            # so two re-tasks of one idle handle admit exactly one run and the
            # refused one never reaches publication. Without it the loser
            # could commit its retention and its link onto the handle while
            # the winner's publication is suspended. The publication owns the
            # claim from here and releases it on every exit, so no failure path
            # out of it can strand one.
            self._host._claim_reuse(reuse, attempt_id)
        # Precedence matches the in-process admission: an explicit retain
        # argument wins, a reuse inherits the instance's committed policy, and
        # a fresh launch takes the [subagents].retain_agents default.
        if args.retain is not None:
            retained = bool(args.retain)
        elif reuse is not None:
            retained = reuse.retained
        else:
            retained = bool(self._host._subagent_config.retain_agents)
        handle = await self._host.publish(
            args, ctx, attempt_id=attempt_id, retained=retained, reuse=reuse
        )
        if args.background:
            yield self._host.admit_background(handle, args, ctx)
            return
        yield await self._host.run_child(handle, args, ctx)

    async def check_agents(self) -> list[AgentSummary]:
        """Answer the management-surface cast the loop makes of this runner.

        ``_build_invoke_context`` casts the turn's subagent runner to
        ``SubagentManagementPort``; a process-owned root supplies this runner,
        so the cast must find a working inventory projection here rather than
        an AttributeError from a run-only object.
        """
        return await self._host.check_agents()

    async def cancel_run(
        self,
        agent_id: str,
        run_id: str | None = None,
        *,
        reason: RunStopReason,
        requester_session_id: str,
    ) -> CancelResult:
        """The casted runner answers CancelAgent with the host's stop disposition."""
        return await self._host.cancel_run(
            agent_id, run_id, reason=reason, requester_session_id=requester_session_id
        )

    async def release_agent(self, agent_id: str) -> ReleaseAgentOutcome:
        """The casted runner answers ReleaseAgent with the host's unguarded release."""
        return await self._host.release_agent(agent_id)

    async def resume_agent(
        self,
        agent_id: str,
        expected_run_id: str,
        idempotency_key: str,
        *,
        requester_session_id: str,
        override_cooldown: bool = False,
    ) -> ResumeAgentResult:
        """The casted runner answers ResumeAgent with the host's typed resume."""
        return await self._host.resume_agent(
            agent_id,
            expected_run_id,
            idempotency_key,
            requester_session_id=requester_session_id,
            override_cooldown=override_cooldown,
        )

    async def reconcile_result(
        self,
        agent_id: str,
        expected_run_id: str,
        transcript_boundary: str,
        verified_effects: Sequence[EffectVerification],
        *,
        requester_session_id: str,
    ) -> ReconcileResult:
        """The casted runner answers ReconcileResult with the host's barrier answer."""
        return await self._host.reconcile_result(
            agent_id,
            expected_run_id,
            transcript_boundary,
            list(verified_effects),
            requester_session_id=requester_session_id,
        )

    async def wait_for_agent(
        self, agent_id: str, run_id: str | None = None, *, timeout: float | None = None
    ) -> WaitForAgentResult:
        """The casted runner answers WaitForAgent with the host's typed wait."""
        return await self._host.wait_for_agent(agent_id, run_id, timeout=timeout)

    async def get_agent_result(
        self, agent_id: str, run_id: str | None = None
    ) -> TaskResult | None:
        """The casted runner answers GetAgentResult with the host's retained result."""
        return await self._host.get_agent_result(agent_id, run_id)


def _default_child_spawn(environ: Mapping[str, str] | None) -> Any:
    """The spawn adapter a process-owned root selects for its children."""
    from chartreux.app_server.subagent_ports import subagent_spawn_port

    return subagent_spawn_port(environ, child_factory=_UnavailableChildFactory())


def agent_child_host(
    loop: AgentLoop,
    *,
    spawn: Any = None,
    environ: Mapping[str, str] | None = None,
    rollback_wait: float | None = None,
    client_info: ClientInfo | None = None,
    client_capabilities: ClientCapabilities | None = None,
) -> AgentChildHost | None:
    """The child host for one root agent process, or None on the escape route.

    A run that selects the in-process escape keeps every child inside the
    coordinator, so the agent installs no host at all and the offer it receives
    is served over the pipe by the coordinator's own port.
    """
    if spawn is None:
        from chartreux.app_server.transport import resolve_transport_selection

        if resolve_transport_selection(environ=environ).kind == "in_process":
            return None
    return AgentChildHost(
        loop,
        spawn=spawn,
        rollback_wait=rollback_wait,
        client_info=client_info,
        client_capabilities=client_capabilities,
    )


def _params(message: Mapping[str, Any]) -> Mapping[str, Any]:
    params = message.get("params")
    return params if isinstance(params, Mapping) else {}


def _wire_error_code(exc: BaseException) -> str:
    """The wire code one failed request publishes.

    Most refusals are named by their class, which IS their code. One that
    carries the app server's own protocol code publishes that code instead, so
    the coordinator reads the failure the in-process route answers rather than
    a generic one.
    """
    if isinstance(exc, AgentProtocolRefusal):
        return exc.code.value
    return type(exc).__name__


async def serve_root_agent(
    *,
    factory: RootLoopFactory,
    reader: BinaryLineReader | None = None,
    writer: BinaryLineWriter | None = None,
    announce_ready: bool = False,
) -> None:
    """Serve a root agent over the given pipes, announcing readiness first."""
    if reader is None or writer is None:
        transport: JsonRpcTransport = StdioJsonRpcTransport.from_standard_streams()
        ready_writer: BinaryLineWriter = sys.stdout.buffer
    else:
        transport = StdioJsonRpcTransport(reader, writer)
        ready_writer = writer
    if announce_ready:
        write_ready_frame(ready_writer)
    try:
        await RootAgentServer(transport, factory=factory).serve()
    finally:
        with suppress(Exception):
            await transport.close()


def main() -> None:
    from chartreux.core.config import load_dotenv_values

    init_harness_files_manager("user", "project")
    init_file_logging(LOG_FILE.path, repair_path=LITERAL_LOG_FILE.path)
    load_dotenv_values()
    logger.info("Root agent process starting")
    asyncio.run(
        serve_root_agent(
            factory=HarnessRootLoopFactory(), announce_ready=bootstrap_requested()
        )
    )


if __name__ == "__main__":
    main()
