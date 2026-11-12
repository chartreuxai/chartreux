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
from collections.abc import AsyncGenerator, Callable, Coroutine, Mapping
from contextlib import aclosing, suppress
from dataclasses import dataclass, replace
import itertools
from pathlib import Path
import sys
import time
from typing import Any, Protocol, cast
from uuid import uuid4

from chartreux.app_server._agent_transcript import read_live_agent_transcript
from chartreux.app_server._execution import SessionExecution
from chartreux.app_server._plan_record import plan_mode_record, session_plan_record
from chartreux.app_server._policy_tree import PolicyNodeService
from chartreux.app_server._resources import ResourceRequestHandler
from chartreux.app_server._shell import ShellController
from chartreux.app_server._shell_record import (
    ManualShellOutcome,
    build_manual_shell_record,
)
from chartreux.app_server._shell_requests import _manual_shell_outcome
from chartreux.app_server._utils import now_ms
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
    session_dirs,
    session_roots_revision,
)
from chartreux.app_server.agent_steering import AgentSteeringArbiter
from chartreux.app_server.agent_wire import (
    AGENT_CHILD_TEARDOWN_METHOD,
    AGENT_COMPACT_METHOD,
    AGENT_DEFERRED_SESSION_METHODS,
    AGENT_INJECT_METHOD,
    AGENT_NOTICE_METHOD,
    AGENT_OP_ADMISSION_METHOD,
    AGENT_OP_CANCEL_METHOD,
    AGENT_OPEN_METHOD,
    AGENT_ORIGIN,
    AGENT_RECEIVING_NOTIFICATIONS,
    AGENT_RECEIVING_REQUESTS,
    AGENT_SESSION_CONFIG_RELOAD_METHOD,
    AGENT_SESSION_LIST_METHOD,
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
    COMPACT_SETTLEMENT_CANCELLED_UNPUBLISHED,
    COMPACT_SETTLEMENT_PUBLISHED_SETTLED,
    COMPACT_SETTLEMENT_PUBLISHED_UNSETTLED,
    COORDINATOR_ORIGIN,
    HOST_SUBAGENT_ENDED_METHOD,
    HOST_SUBAGENT_EVENT_METHOD,
    HOST_TOOL_IO_METHOD,
    POLICY_METHODS,
    RECONFIGURE_MODE_CONFIG_WRITE,
    RECONFIGURE_MODE_HISTORY_CLEAR,
    RECONFIGURE_MODE_PLAN_HANDOFF,
    RECONFIGURE_MODE_SETTINGS,
    SUBAGENT_RUN_METHOD,
    TURN_OUTCOME_COMPLETED,
    TURN_OUTCOME_FAILED,
    TURN_OUTCOME_INTERRUPTED,
    AgentReconfigureConflictError,
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
    encode_compact_reply,
    encode_event,
    encode_inject_reply,
    encode_notice,
    encode_open_reply,
    encode_operation_marker,
    encode_reconfigure_reply,
    encode_rename_reply,
    encode_replace_reply,
    encode_retry_reason,
    encode_session_snapshot,
    encode_shell_command_request,
    encode_shell_interrupt_reply,
    encode_shell_run_reply,
    encode_steer_settlement,
    encode_subagent_run_request,
    encode_tool_io_request,
    encode_transcript_page,
    encode_user_input_settlement,
    request_id_origin,
)
from chartreux.app_server.process_bootstrap import (
    bootstrap_requested,
    write_ready_frame,
)
from chartreux.app_server.protocol import (
    PolicyReadParams,
    RootsReadParams,
    RootsReadResponse,
    RootsSaveParams,
    RootsSaveResponse,
    SessionListParams,
    SessionPlanRecordWriteParams,
    ShellRunParams,
    ShellRunResponse,
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
from chartreux.core.config._root_persistence import read_saved_roots
from chartreux.core.config.harness_files import init_harness_files_manager
from chartreux.core.events import (
    AssistantEvent,
    BaseEvent,
    ToolStreamEvent,
    UserInputRequestEvent,
)
from chartreux.core.llm_models import LLMMessage
from chartreux.core.paths import LITERAL_LOG_FILE, LOG_FILE
from chartreux.core.subagents import (
    LaunchOutcome,
    SubagentRunnerPort,
    TaskArgs,
    TaskResult,
)
from chartreux.core.tools import secret_redaction
from chartreux.core.tools.base import InvokeContext
from chartreux.core.tools.builtins.bash import BashToolConfig
from chartreux.core.tools.io_port import (
    ShellCommandRequest,
    ShellCommandResult,
    ToolIOPort,
)
from chartreux.core.utils.retry import RetryObserver, RetryReason
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
        self.child_host = agent_child_host(loop)
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
        # A manual `!` command runs here, where the loop's cwd and its tool
        # config live; one controller owns every run of this process so an
        # interrupt can reach the process it names.
        self._shell_controller: ShellController | None = None

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

    @property
    def session_id(self) -> str | None:
        return self._loop.session_id if self._loop is not None else None

    @property
    def child_host(self) -> AgentChildHost | None:
        """The child host this process owns, when its route gives it children."""
        return self._child_host

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
                "error": {"code": type(exc).__name__, "message": str(exc)},
            })
        else:
            await self._send({"jsonrpc": "2.0", "id": request_id, "result": result})

    async def _handle_request(  # noqa: PLR0911, PLR0912 - one branch per wire method
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
        if method == AGENT_SESSION_PLAN_RECORD_READ_METHOD:
            return await self._read_plan_record(params)
        if method == AGENT_SESSION_PLAN_RECORD_WRITE_METHOD:
            return await self._write_plan_record(params)
        if method == AGENT_SESSION_POLICY_READ_METHOD:
            return await self._read_policy(params)
        if method == AGENT_SESSION_ROOTS_READ_METHOD:
            return await self._read_roots(params)
        if method == AGENT_SESSION_ROOTS_SAVE_METHOD:
            return await self._save_roots(params)
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
        raise UnsupportedAgentRequestError(f"unknown agent method {method!r}")

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
            loop, markers=(), incarnation_id=self._markers.incarnation_id
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
                loop, markers=(), incarnation_id=self._markers.incarnation_id
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
        """Run one reconfigure mode: short here, the plan fence over the fence.

        A short mode mutates the session directly and is refused typed while a
        long operation holds the fence (one long op per root), so a settings
        update or a clear can never interleave with a compaction, a replace or
        a handoff. The plan handoff is admitted as a long operation with the
        coordinator's absolute deadline: its marker record, its deadline and
        its fence are the same ones compaction uses.
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
        if request.mode == RECONFIGURE_MODE_PLAN_HANDOFF:
            if request.deadline is None:
                raise UnsupportedAgentRequestError(
                    "a plan handoff must carry the coordinator's absolute deadline"
                )
            return await self._long_ops.admit(
                request_id=str(request_id),
                method=AGENT_SESSION_RECONFIGURE_METHOD,
                deadline=request.deadline,
                body=lambda: self._plan_handoff_operation(request),
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

    async def _plan_handoff_operation(
        self, request: ReconfigureRequest
    ) -> dict[str, Any]:
        """Run one plan handoff and answer with the fence outcome it reached."""
        async with self._reconfigure_lock:
            return encode_reconfigure_reply(
                await self._reconfigure_service().plan_handoff(request)
            )

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

    async def _read_plan_record(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """Read the durable plan mode the session's own metadata carries."""
        loop = self._require_loop()
        if params.get("session_id") != loop.session_id:
            raise UnsupportedAgentRequestError(
                "the plan record belongs to the root session"
            )
        logger = loop.session_logger
        metadata = logger.session_metadata
        record = (
            None
            if metadata is None or metadata.plan_mode is None
            else session_plan_record(metadata.plan_mode)
        )
        return {"record": None if record is None else record.model_dump(mode="json")}

    async def _write_plan_record(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """Persist (or, with a null record, remove) the durable plan mode.

        The reply carries the record as READ BACK, so a session whose logging is
        disabled answers null instead of echoing a mode no resume could restore.
        """
        request = SessionPlanRecordWriteParams.model_validate(dict(params))
        if request.session_id != self._require_loop().session_id:
            raise UnsupportedAgentRequestError(
                "the plan record belongs to the root session"
            )
        logger = self._require_loop().session_logger
        record = None if request.record is None else plan_mode_record(request.record)
        if record is None:
            await logger.clear_plan_mode()
        else:
            await logger.persist_plan_mode(record)
        metadata = logger.session_metadata
        persisted = (
            None
            if metadata is None or metadata.plan_mode is None
            else session_plan_record(metadata.plan_mode)
        )
        return {
            "record": None if persisted is None else persisted.model_dump(mode="json")
        }

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
        await self._notify(
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
                await self._notify(AGENT_NOTICE_METHOD, {"event": payload})
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
                    await self._notify(
                        AGENT_TURN_EVENT_METHOD,
                        {"turn_id": turn_id, "event": encode_event(event)},
                    )
        except asyncio.CancelledError:
            outcome = TURN_OUTCOME_INTERRUPTED
        except BaseException as exc:
            outcome = TURN_OUTCOME_FAILED
            error = f"{type(exc).__name__}: {exc}"
        finally:
            self._active_turn_id = None
            self._active_turn_task = None
            # A request still open when its turn ends has no waiter left to
            # answer it: close it here, exactly as the in-process turn closes
            # its callbacks, so the client never keeps a ghost prompt and the
            # terminal disposition is published before the ended frame.
            with suppress(Exception):
                await self._cancel_open_user_inputs("Turn ended")
            with suppress(Exception):
                await self._notify(
                    AGENT_TURN_ENDED_METHOD,
                    {"turn_id": turn_id, "outcome": outcome, "error": error},
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


@dataclass
class AgentChildHandle:
    """One child process this root agent spawned, keyed by its attempt id.

    ``attempt_id`` is minted here, so a lost aggregate outcome still names its
    attempt: the teardown request the coordinator sends after a lost reply
    resolves through this record. A RETAINED child keeps its process across
    re-tasks; a non-retained one is reaped at completion.
    """

    agent_id: str
    identity: str
    attempt_id: str
    spawned: Any
    session_id: str
    child_dir: str | None
    retained: bool
    slot: Any
    linked: bool = False
    closed: bool = False
    result: TaskResult | None = None

    @property
    def alive(self) -> bool:
        return not self.closed and bool(getattr(self.spawned, "alive", True))


class AgentChildHost:
    """The children one root agent process spawns, links and tears down.

    The root agent owns the session, so it also owns the child launch: the
    candidate resolves against this process's own orchestrator and tool
    authorities, the working-set handoff is rendered from this loop, and the
    parent link is written to this loop's transcript. A child is a process of
    its own (``ProcessSubagentSpawn``) and the host holds the attempt table the
    coordinator's teardown request resolves against.
    """

    def __init__(
        self,
        loop: AgentLoop,
        *,
        spawn: Any = None,
        environ: Mapping[str, str] | None = None,
        rollback_wait: float | None = None,
    ) -> None:
        self._loop = loop
        self._spawn = spawn if spawn is not None else _default_child_spawn(environ)
        self._rollback_wait = (
            DEFAULT_CHILD_ROLLBACK_WAIT if rollback_wait is None else rollback_wait
        )
        self._admission = SubagentAdmission()
        self._cap = loop.config.subagents.max_running_subagents
        # A retained child is held by its stable agent id; every attempt is
        # held by the attempt id the coordinator would name after a lost reply.
        self._children: dict[str, AgentChildHandle] = {}
        self._attempts: dict[str, AgentChildHandle] = {}
        self._tasks: set[asyncio.Task[None]] = set()
        self._runner = AgentChildRunner(self)

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

    async def publish(
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
        from chartreux.app_server.subagent_ports import (
            CHILD_HANDOFF_PAYLOAD_KEY,
            child_spawn_spec,
        )

        stage = ChildPublicationStage.ADMISSION
        slot: Any = None
        spawned: Any = None
        try:
            slot = self._admission.reserve(
                cap=self._cap,
                depth=1 if ctx.is_subagent else 0,
                reuse_agent_id=args.agent_id,
                retained_idle=list(self._children),
            )
            agent_name = self._resolved_identity(args).replace(":", "-")
            if reuse is not None and reuse.alive:
                # A retained child keeps its process: the re-task only links
                # the new tool call to the process that already exists.
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
            payload = (
                {CHILD_HANDOFF_PAYLOAD_KEY: handoff.content}
                if handoff is not None and handoff.content
                else {}
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
            )
            self._attempts[attempt_id] = handle
            stage = ChildPublicationStage.LINKAGE
            await self._link(handle, ctx, agent_name, attempt_id)
            if retained:
                self._children[handle.agent_id] = handle
            return handle
        except BaseException as exc:
            failed = exc.stage if isinstance(exc, ChildPublicationError) else stage
            raise await self._fail_attempt(
                failed, attempt_id, exc, slot=slot, spawned=spawned
            ) from exc

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
        self, handle: AgentChildHandle, args: TaskArgs, ctx: InvokeContext
    ) -> TaskResult:
        """Run one turn in a spawned child and answer its result.

        A retained child keeps its process (and its place in the cap) after the
        run; a non-retained one is reaped here, so its process exits with the
        run that produced the result.
        """
        from chartreux.app_server.subagent_ports import SubagentCompletion
        from chartreux.core.subagents import prepare_subagent_prompt

        run_id = uuid4().hex
        request = ExecutionTurnRequest(
            message=prepare_subagent_prompt(args.task, ctx),
            turn_options=AgentTurnOptions(turn_id=run_id),
        )
        parts: list[str] = []
        turns = 0
        outcome = TURN_OUTCOME_FAILED
        error: str | None = None
        try:
            async with aclosing(
                handle.spawned.completion_source.stream(request)
            ) as frames:
                async for frame in frames:
                    if isinstance(frame, SubagentCompletion):
                        outcome = frame.outcome
                        error = frame.error
                        continue
                    if isinstance(frame, AssistantEvent):
                        turns += 1
                        if frame.content:
                            parts.append(frame.content)
        finally:
            await self.settle_run(handle)
        response = "\n\n".join(parts)
        if error:
            response = f"{response}\n\n{error}".strip()
        return TaskResult(
            response=response,
            turns_used=turns,
            completed=outcome == TURN_OUTCOME_COMPLETED,
            agent_id=handle.agent_id if handle.retained else None,
            run_id=run_id if handle.retained else None,
        )

    async def settle_run(self, handle: AgentChildHandle) -> None:
        """Release the run's capacity: a retained child stays, others are reaped."""
        if handle.retained:
            self._admission.mark_retained_idle(handle.agent_id)
            self._admission.release(handle.slot)
            return
        await self.release(handle)

    async def release(self, handle: AgentChildHandle) -> None:
        """Reap one child's process and free its capacity and budget slot."""
        if handle.closed:
            return
        handle.closed = True
        self._children.pop(handle.agent_id, None)
        self._attempts.pop(handle.attempt_id, None)
        try:
            with suppress(Exception):
                await handle.spawned.close()
        finally:
            self._admission.release(handle.slot)
            self._admission.clear_retained(handle.agent_id)

    def admit_background(
        self, handle: AgentChildHandle, args: TaskArgs, ctx: InvokeContext
    ) -> TaskResult:
        """Acknowledge a background launch and run its turn out of band."""
        task = asyncio.create_task(
            self._run_background(handle, args, ctx),
            name=f"root-agent-child-{handle.agent_id}",
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return TaskResult(
            response=(
                f"Background agent {handle.identity} ({handle.agent_id}) launched "
                "and running in its own process"
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
            run_id=uuid4().hex,
        )

    async def _run_background(
        self, handle: AgentChildHandle, args: TaskArgs, ctx: InvokeContext
    ) -> None:
        try:
            handle.result = await self.run_child(handle, args, ctx)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            logger.warning("A background child run failed: %s", exc)

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
        for handle in list(self._children.values()) + list(self._attempts.values()):
            await self.release(handle)


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
        retained = bool(args.retain) or (
            args.agent_id is not None
            and args.agent_id in {child.agent_id for child in self._host.children()}
        )
        reuse = next(
            (
                child
                for child in self._host.children()
                if args.agent_id is not None and child.agent_id == args.agent_id
            ),
            None,
        )
        handle = await self._host.publish(
            args,
            ctx,
            attempt_id=new_publication_attempt_id(),
            retained=retained,
            reuse=reuse,
        )
        if args.background:
            yield self._host.admit_background(handle, args, ctx)
            return
        yield await self._host.run_child(handle, args, ctx)


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
    return AgentChildHost(loop, spawn=spawn, rollback_wait=rollback_wait)


def _params(message: Mapping[str, Any]) -> Mapping[str, Any]:
    params = message.get("params")
    return params if isinstance(params, Mapping) else {}


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
