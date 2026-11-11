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
from collections.abc import AsyncGenerator, Awaitable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, replace
import itertools
import os
import subprocess
import sys
from typing import Any, cast
from uuid import uuid4

from pydantic import BaseModel

from chartreux.app_server.agent_wire import (
    AGENT_OPEN_METHOD,
    AGENT_TRANSCRIPT_PAGE_METHOD,
    AGENT_TURN_ANSWER_METHOD,
    AGENT_TURN_CANCEL_METHOD,
    AGENT_TURN_ENDED_METHOD,
    AGENT_TURN_EVENT_METHOD,
    AGENT_TURN_KEEP_WAITING_METHOD,
    AGENT_TURN_REFUSE_METHOD,
    AGENT_TURN_REJECT_METHOD,
    AGENT_TURN_RELAY_METHOD,
    AGENT_TURN_RESOLVE_METHOD,
    AGENT_TURN_RETRYING_METHOD,
    AGENT_TURN_START_METHOD,
    TURN_OUTCOME_COMPLETED,
    TURN_OUTCOME_FAILED,
    TURN_OUTCOME_INTERRUPTED,
    decode_event,
    decode_retry_reason,
    decode_transcript_page,
    encode_event,
    encode_turn_request,
    encode_user_input_result,
)
from chartreux.app_server.process_bootstrap import (
    DEFAULT_SPAWN_READY_TIMEOUT,
    ProcessTreeBudget,
    SpawnedProcess,
    spawn_process,
)
from chartreux.app_server.protocol import AgentTranscriptGetResponse
from chartreux.app_server.registry_manifest import AgentRoute, RegistryManifest
from chartreux.app_server.root_agent import ROOT_AGENT_MODULE
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
    UserInputSettleOutcome,
)
from chartreux.core.events import BaseEvent, UserInputRequestEvent
from chartreux.core.utils.retry import RetryObserver

DEFAULT_REQUEST_TIMEOUT = 30.0

DEFAULT_ROOT_HEARTBEAT_INTERVAL = 2.0
"""The root pipe's liveness heartbeat period (plan.md numeric bounds, WP7 [7C])."""

_DEFAULT_CONNECTION_ID = "root-agent"


class RootAgentProcessError(RuntimeError):
    """Base class for failures of the root agent process handle."""


@dataclass(frozen=True, slots=True)
class RootDeath:
    """One observed death of the root agent process, typed."""

    reason: str
    exit_code: int | None = None


class RootAgentExitedError(RootAgentProcessError):
    """The root agent process died: its pipe closed or its process is gone.

    Root death is a typed lifecycle observation, never a transport error: the
    handle stops serving requests and turns, and the coordinator disposes of
    the child tree before anything is respawned or resumed.
    """

    def __init__(self, reason: str, *, exit_code: int | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.exit_code = exit_code


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


type _Frame = tuple[str, dict[str, Any]]


class RootAgentProcess:
    """A spawned root agent serving the agent-side execution port."""

    def __init__(
        self,
        spawned: SpawnedProcess,
        *,
        connection_id: str = _DEFAULT_CONNECTION_ID,
        incarnation_id: str | None = None,
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
        heartbeat_interval: float = DEFAULT_ROOT_HEARTBEAT_INTERVAL,
    ) -> None:
        if request_timeout <= 0:
            raise ValueError("request_timeout must be positive")
        if heartbeat_interval <= 0:
            raise ValueError("heartbeat_interval must be positive")
        self._spawned = spawned
        self._transport = spawned.transport
        self.connection_id = connection_id
        self.incarnation_id = incarnation_id or str(uuid4())
        self.session_id: str | None = None
        self._request_timeout = request_timeout
        self._request_ids = itertools.count(1)
        self._pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._turns: dict[str, asyncio.Queue[_Frame]] = {}
        self._retry_sinks: dict[str, RetryObserver | None] = {}
        self._notifications: set[asyncio.Task[None]] = set()
        self._routes: list[tuple[RegistryManifest, str]] = []
        self._active_turn_id: str | None = None
        self._closed = False
        self._death: RootDeath | None = None
        self._death_event = asyncio.Event()
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

    async def wait_for_death(self) -> RootDeath:
        """Wait until root death is declared and return the typed record."""
        await self._death_event.wait()
        death = self._death
        if death is None:
            raise RootAgentProcessError("the root agent process is not dead")
        return death

    async def open(
        self,
        *,
        session_options: BaseModel,
        client_info: BaseModel,
        client_capabilities: BaseModel | None = None,
    ) -> str:
        """Open the session in the agent process and return its session id."""
        return await self._open_request(
            AGENT_OPEN_METHOD,
            {
                "agent_config": session_options.model_dump(mode="json", by_alias=True),
                "client_info": client_info.model_dump(mode="json", by_alias=True),
                "client_capabilities": (
                    client_capabilities.model_dump(mode="json", by_alias=True)
                    if client_capabilities is not None
                    else {}
                ),
            },
        )

    async def _open_request(self, method: str, params: dict[str, Any]) -> str:
        """Send an open request and adopt the session id it announces."""
        result = await self._request(method, params)
        session_id = result.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise RootAgentRequestError(
                "MalformedOpenResponse", "the agent opened without a session id"
            )
        self.session_id = session_id
        return session_id

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
        if request.subagent_runner is not None or request.tool_io is not None:
            raise UnsupportedProcessTurnRequest(
                "host-side subagent and tool I/O ports have no wire form yet"
            )
        turn_id = request.turn_options.turn_id if request.turn_options else None
        turn_id = turn_id or str(uuid4())
        channel: asyncio.Queue[_Frame] = asyncio.Queue()
        self._turns[turn_id] = channel
        self._retry_sinks[turn_id] = (
            request.turn_options.retry_sink if request.turn_options else None
        )
        self._active_turn_id = turn_id
        try:
            await self._transport.send({
                "jsonrpc": "2.0",
                "method": AGENT_TURN_START_METHOD,
                "params": {
                    "turn_id": turn_id,
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

    async def close(self) -> None:
        """Retire the live route and terminate the agent process."""
        if self._closed:
            return
        self._closed = True
        for manifest, agent_id in self._routes:
            with suppress(OSError):
                manifest.remove_agent(agent_id)
        self._routes.clear()
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
                self._declare_death(
                    "the root agent pipe closed"
                    if failure is None
                    else f"the root agent pipe closed: {failure}"
                )

    async def _watch_liveness(self) -> None:
        """Declare root death when the process is gone but the pipe stays open."""
        while True:
            await asyncio.sleep(self._heartbeat_interval)
            if self._closed or self._death is not None:
                return
            if self._spawned.process.poll() is not None:
                self._declare_death("the root agent process exited")
                return

    def _declare_death(self, reason: str) -> None:
        """Publish one typed root death and fail everything still waiting."""
        if self._death is not None or self._closed:
            return
        exit_code = self._spawned.process.poll()
        self._death = RootDeath(reason=reason, exit_code=exit_code)
        self._death_event.set()
        for future in self._pending.values():
            if not future.done():
                future.set_exception(RootAgentExitedError(reason, exit_code=exit_code))
        for turn_id, channel in self._turns.items():
            channel.put_nowait((
                "ended",
                {"turn_id": turn_id, "outcome": TURN_OUTCOME_FAILED, "error": reason},
            ))

    def _dispatch(self, message: Mapping[str, Any]) -> None:
        if "id" in message:
            self._settle_request(message)
            return
        method = message.get("method")
        params = message.get("params")
        payload = params if isinstance(params, dict) else {}
        if method == AGENT_TURN_EVENT_METHOD:
            self._deliver(payload, "event")
            return
        if method == AGENT_TURN_RETRYING_METHOD:
            self._deliver(payload, "retrying")
            return
        if method == AGENT_TURN_ENDED_METHOD:
            self._deliver(payload, "ended")
            return
        raise RootAgentProcessError(f"unknown agent frame {method!r}")

    def _deliver(self, payload: dict[str, Any], kind: str) -> None:
        channel = self._turns.get(str(payload.get("turn_id")))
        if channel is None:
            return
        channel.put_nowait((kind, payload))

    def _settle_request(self, message: Mapping[str, Any]) -> None:
        request_id = str(message.get("id"))
        future = self._pending.get(request_id)
        if future is None or future.done():
            return
        error = message.get("error")
        if isinstance(error, Mapping):
            future.set_exception(
                RootAgentRequestError(
                    str(error.get("code") or "AgentError"),
                    str(error.get("message") or "agent request failed"),
                )
            )
            return
        result = message.get("result")
        future.set_result(result if isinstance(result, dict) else {})

    async def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if self._closed:
            raise RootAgentProcessError("the root agent process is closed")
        if self._death is not None:
            raise RootAgentExitedError(
                self._death.reason, exit_code=self._death.exit_code
            )
        request_id = str(next(self._request_ids))
        future: asyncio.Future[dict[str, Any]] = (
            asyncio.get_running_loop().create_future()
        )
        self._pending[request_id] = future
        try:
            await self._transport.send({
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params,
            })
            return await asyncio.wait_for(future, timeout=self._request_timeout)
        finally:
            self._pending.pop(request_id, None)

    def _schedule_notification(self, method: str, params: dict[str, Any]) -> None:
        if self._closed:
            raise RootAgentProcessError("the root agent process is closed")

        async def deliver() -> None:
            await self._transport.send({
                "jsonrpc": "2.0",
                "method": method,
                "params": params,
            })

        task = asyncio.create_task(deliver())
        self._notifications.add(task)
        task.add_done_callback(self._notifications.discard)

    def _page_reader(
        self, agent_id: str, before: str | None, limit: int
    ) -> Awaitable[AgentTranscriptGetResponse]:
        return self.read_page(agent_id, before=before, limit=limit)

    async def _finish_turn(self, payload: Mapping[str, Any]) -> None:
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
    every child it asks for draw on ONE hard cap.
    """
    source = os.environ if env is None else env
    if not multi_process_root_selected(source):
        raise RootAgentSelectionError(
            f"{TRANSPORT_ENV_VAR} does not select the multi-process (stdio) "
            "root agent; the in-process route stays selected"
        )
    spawned = await spawn_process(
        command or root_agent_command(),
        env=source,
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
        session_id = await process.open(
            session_options=session_options,
            client_info=client_info,
            client_capabilities=client_capabilities,
        )
    except BaseException:
        await process.close()
        raise
    if manifest is not None:
        process.register_live_route(manifest, agent_id or session_id)
    return process


__all__ = [
    "AgentProcessTurnFailed",
    "RootAgentExitedError",
    "RootAgentProcess",
    "RootAgentProcessError",
    "RootAgentRequestError",
    "RootAgentSelectionError",
    "RootDeath",
    "UnsupportedProcessTurnRequest",
    "multi_process_root_selected",
    "root_agent_command",
    "spawn_root_agent_process",
]
