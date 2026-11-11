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
from collections.abc import Callable, Coroutine, Mapping
from contextlib import aclosing, suppress
from dataclasses import replace
import sys
from typing import Any, Protocol, cast

from chartreux.app_server._agent_transcript import read_live_agent_transcript
from chartreux.app_server._policy_tree import PolicyNodeService
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
    POLICY_METHODS,
    TURN_OUTCOME_COMPLETED,
    TURN_OUTCOME_FAILED,
    TURN_OUTCOME_INTERRUPTED,
    RemoteUserInputRejectedError,
    UnsupportedAgentRequestError,
    decode_event,
    decode_turn_request,
    decode_user_input_result,
    encode_event,
    encode_retry_reason,
    encode_transcript_page,
)
from chartreux.app_server.process_bootstrap import (
    bootstrap_requested,
    write_ready_frame,
)
from chartreux.app_server.transport import (
    BinaryLineReader,
    BinaryLineWriter,
    JsonRpcTransport,
    StdioJsonRpcTransport,
)
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.agent_loop._loop_types import AgentTurnOptions
from chartreux.core.config.harness_files import init_harness_files_manager
from chartreux.core.events import UserInputRequestEvent
from chartreux.core.llm_models import LLMMessage
from chartreux.core.paths import LITERAL_LOG_FILE, LOG_FILE
from chartreux.core.utils.retry import RetryObserver, RetryReason
from chartreux.observability.logging import init_file_logging, logger

ROOT_AGENT_MODULE = "chartreux.app_server.root_agent"


class RootLoopFactory(Protocol):
    """Builds the loop an ``agent/open`` request asks for, and owns its process."""

    async def open(self, params: Mapping[str, Any]) -> AgentLoop: ...

    async def close(self) -> None: ...


class HarnessRootLoopFactory:
    """The production factory: one harness process owns one root loop."""

    def __init__(self) -> None:
        from chartreux.app_server._runtime import HarnessProcess

        self._process = HarnessProcess()

    async def open(self, params: Mapping[str, Any]) -> AgentLoop:
        from chartreux.app_server._connection_protocol import (
            ClientCapabilities,
            ClientInfo,
        )
        from chartreux.app_server._runtime import RootOpenRequest
        from chartreux.app_server.protocol import AgentConfig

        options = AgentConfig.model_validate(params.get("agent_config") or {})
        client_info = ClientInfo.model_validate(params.get("client_info") or {})
        capabilities = ClientCapabilities.model_validate(
            params.get("client_capabilities") or {}
        )
        return await self._process.open_root(
            RootOpenRequest(
                options=options,
                client_info=client_info,
                client_capabilities=capabilities,
            )
        )

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
        self._tasks: set[asyncio.Task[None]] = set()
        self._policy_node: PolicyNodeService | None = None

    @property
    def session_id(self) -> str | None:
        return self._loop.session_id if self._loop is not None else None

    async def serve(self) -> None:
        try:
            async for message in self._transport.messages():
                self._dispatch(message)
        finally:
            await self._shutdown()

    def _dispatch(self, message: Mapping[str, Any]) -> None:
        method = message.get("method")
        if not isinstance(method, str):
            raise UnsupportedAgentRequestError("agent frame carries no method")
        if "id" in message:
            self._track(self._serve_request(message["id"], method, message))
            return
        params = _params(message)
        if method == AGENT_TURN_START_METHOD:
            self._start_turn(params)
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
        raise UnsupportedAgentRequestError(f"unknown agent frame {method!r}")

    async def _serve_request(
        self, request_id: object, method: str, message: Mapping[str, Any]
    ) -> None:
        try:
            result = await self._handle_request(method, _params(message))
        except BaseException as exc:
            await self._send({
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": type(exc).__name__, "message": str(exc)},
            })
        else:
            await self._send({"jsonrpc": "2.0", "id": request_id, "result": result})

    async def _handle_request(  # noqa: PLR0911 - one branch per wire method
        self, method: str, params: Mapping[str, Any]
    ) -> dict[str, Any]:
        if method == AGENT_OPEN_METHOD:
            if self._loop is not None:
                raise UnsupportedAgentRequestError("the agent is already open")
            self._loop = await self._factory.open(params)
            return {"session_id": self._loop.session_id}
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
        if method in POLICY_METHODS:
            return await self._policy_service().handle(method, params)
        raise UnsupportedAgentRequestError(f"unknown agent method {method!r}")

    def _policy_service(self) -> PolicyNodeService:
        """The one policy-node service serving this agent's four phases."""
        if self._policy_node is None:
            self._policy_node = PolicyNodeService(self._require_loop())
        return self._policy_node

    def _start_turn(self, params: Mapping[str, Any]) -> None:
        turn_id = str(params.get("turn_id") or "")
        if not turn_id:
            raise UnsupportedAgentRequestError("a turn request carries no turn id")
        if self._active_turn_id is not None:
            self._track(self._end_rejected_turn(turn_id))
            return
        self._active_turn_id = turn_id
        self._track(self._run_turn(turn_id, params))

    async def _run_turn(self, turn_id: str, params: Mapping[str, Any]) -> None:
        outcome = TURN_OUTCOME_COMPLETED
        error: str | None = None
        try:
            loop = self._require_loop()
            request = decode_turn_request(params)
            options = request.turn_options or AgentTurnOptions()
            request = replace(
                request,
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
            with suppress(Exception):
                await self._notify(
                    AGENT_TURN_ENDED_METHOD,
                    {"turn_id": turn_id, "outcome": outcome, "error": error},
                )

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
        current = asyncio.current_task()
        for task in list(self._tasks):
            if task is not current and not task.done():
                task.cancel()
        for task in list(self._tasks):
            if task is not current:
                with suppress(BaseException):
                    await task
        loop, self._loop = self._loop, None
        if loop is not None:
            await loop.aclose()
        await self._factory.close()


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
