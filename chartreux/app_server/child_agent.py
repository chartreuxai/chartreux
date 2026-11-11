"""Child agent process: one subagent loop served over its own standard pipes.

The coordinator spawns this module for each subagent (WP5 [5A]). The child
announces readiness with the WP1 bootstrap frame, opens through the
``subagent/open`` frame — which carries the spawn spec — and then serves the
same agent-wire turn protocol as the root agent process: turn events stream
back in production order and the terminal ``agent/turn/ended`` frame closes
the run, which the coordinator's completion source surfaces as its final
frame.

A child process is always fresh, never resumed as a session
(design.md:176-180): its prior transcripts persist as records under the root
session and are reconciled by the coordinator registry ([5C]). Child-owned
transcripts under ``<root>/agents/*`` and the propagated policy tree land with
[5C]/[5D]; this entrypoint only serves the wire.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from contextlib import aclosing, suppress
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
import sys
from typing import Any, cast

from chartreux.app_server._agent_transcript import read_live_agent_transcript
from chartreux.app_server._subagent_transcript import (
    AGENT_NAME_PAYLOAD_KEY,
    ChildResultRecord,
    ChildStartMarker,
    ChildTranscriptError,
    ChildTranscriptStore,
    child_result_digest,
    child_transcript_logging,
    child_transcript_root_for,
    root_session_dir_from_spec,
)
from chartreux.app_server.agent_wire import (
    AGENT_TURN_ENDED_METHOD,
    AGENT_TURN_EVENT_METHOD,
    SUBAGENT_OPEN_METHOD,
    TURN_OUTCOME_COMPLETED,
    TURN_OUTCOME_FAILED,
    TURN_OUTCOME_INTERRUPTED,
    UnsupportedAgentRequestError,
    decode_turn_request,
    encode_event,
    encode_transcript_page,
)
from chartreux.app_server.process_bootstrap import (
    bootstrap_requested,
    write_ready_frame,
)
from chartreux.app_server.root_agent import RootAgentServer, RootLoopFactory
from chartreux.app_server.subagent_ports import spawn_spec_from_wire
from chartreux.app_server.transport import (
    BinaryLineReader,
    BinaryLineWriter,
    JsonRpcTransport,
    StdioJsonRpcTransport,
)
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.agent_loop._loop_types import AgentTurnOptions
from chartreux.core.config.harness_files import init_harness_files_manager
from chartreux.core.llm_models import LLMMessage
from chartreux.core.paths import LITERAL_LOG_FILE, LOG_FILE
from chartreux.observability.logging import init_file_logging, logger

CHILD_AGENT_MODULE = "chartreux.app_server.child_agent"


class ChildAgentServer(RootAgentServer):
    """Serve one child agent loop; ``subagent/open`` carries the spawn spec.

    The child OWNS its transcript ([5C]): its loop's SessionLogger writes
    ``<root>/agents/<agent>_<ts>_<id>``, and this server writes the durable
    start marker before a run executes and the terminal result record (result
    id + digest) save-before-emit. The coordinator never writes these records;
    it reads them to adjudicate a crash.
    """

    def __init__(
        self, transport: JsonRpcTransport, *, factory: RootLoopFactory
    ) -> None:
        super().__init__(transport, factory=factory)
        self._agent_id: str | None = None
        self._root_session_dir: Path | None = None
        self._transcript_store: ChildTranscriptStore | None = None

    async def _handle_request(
        self, method: str, params: Mapping[str, Any]
    ) -> dict[str, Any]:
        if method == SUBAGENT_OPEN_METHOD:
            if self._loop is not None:
                raise UnsupportedAgentRequestError("the child agent is already open")
            # Validate the spec before building the loop: a malformed open
            # frame fails typed instead of opening a nameless child.
            spec = spawn_spec_from_wire(params)
            self._loop = await self._factory.open(params)
            self._agent_id = spec.agent_id
            session_dir = self._loop.session_logger.session_dir
            self._root_session_dir = child_transcript_root_for(
                session_dir, root_session_dir_from_spec(spec.payload)
            )
            self._transcript_store = self._open_transcript_store()
            return {
                "agent_id": spec.agent_id,
                "session_id": self._loop.session_id,
                "session_dir": str(session_dir) if session_dir is not None else None,
            }
        return await super()._handle_request(method, params)

    def _open_transcript_store(self) -> ChildTranscriptStore | None:
        """The child-owned marker store for this transcript dir, when usable."""
        loop = self._loop
        root = self._root_session_dir
        if loop is None or root is None:
            return None
        session_dir = loop.session_logger.session_dir
        if session_dir is None:
            return None
        try:
            return ChildTranscriptStore(session_dir, root_session_dir=root)
        except ChildTranscriptError as exc:
            logger.warning("Child transcript markers disabled: %s", exc)
            return None

    async def _run_turn(self, turn_id: str, params: Mapping[str, Any]) -> None:
        """Run one turn with the child's durable start and terminal records."""
        self._record_start(turn_id)
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
            # Save-before-emit: the terminal result record is durable before
            # the completion frame is announced, so a crash right after this
            # point reconciles as COMPLETED, not as an unknown outcome.
            self._record_result(turn_id, outcome=outcome, error=error)
            with suppress(Exception):
                await self._notify(
                    AGENT_TURN_ENDED_METHOD,
                    {"turn_id": turn_id, "outcome": outcome, "error": error},
                )

    async def _read_page(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """Serve a live page; the child answers to its agent id and session id."""
        loop = self._require_loop()
        agent_id = params.get("agent_id")
        if agent_id not in {loop.session_id, self._agent_id}:
            raise UnsupportedAgentRequestError("unknown agent")
        response = await asyncio.to_thread(
            read_live_agent_transcript,
            cast("list[LLMMessage]", loop.messages),
            before=params.get("before"),
            limit=int(params.get("limit") or 50),
        )
        return encode_transcript_page(response)

    def _record_start(self, turn_id: str) -> None:
        store = self._transcript_store
        loop = self._loop
        if store is None or loop is None:
            return
        try:
            store.record_start(
                ChildStartMarker(
                    agent_id=self._agent_id or loop.session_id,
                    session_id=loop.session_id,
                    run_id=turn_id,
                    started_at=datetime.now(UTC).isoformat(),
                )
            )
        except ChildTranscriptError as exc:
            logger.warning("Failed to record the child start marker: %s", exc)

    def _record_result(self, turn_id: str, *, outcome: str, error: str | None) -> None:
        store = self._transcript_store
        loop = self._loop
        if store is None or loop is None:
            return
        agent_id = self._agent_id or loop.session_id
        try:
            store.record_result(
                ChildResultRecord(
                    agent_id=agent_id,
                    session_id=loop.session_id,
                    run_id=turn_id,
                    result_id=f"{agent_id}:{turn_id}",
                    digest=child_result_digest(
                        agent_id=agent_id,
                        session_id=loop.session_id,
                        run_id=turn_id,
                        outcome=outcome,
                        error=error,
                    ),
                    outcome=outcome,
                    completed_at=datetime.now(UTC).isoformat(),
                    error=error,
                )
            )
        except ChildTranscriptError as exc:
            logger.warning("Failed to record the child terminal result: %s", exc)


class HarnessChildLoopFactory:
    """The production factory: one child process owns one child loop."""

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

        spec = spawn_spec_from_wire(params)
        payload = spec.payload
        root_session_dir = root_session_dir_from_spec(payload)
        if root_session_dir is None:
            # Fail closed: a child's transcript belongs under the root
            # session's agents/ subtree. Falling back to the configured save
            # dir would put it outside the root and break reconciliation.
            raise UnsupportedAgentRequestError(
                "a child spawn spec carries no root session directory"
            )
        options = AgentConfig.model_validate(payload.get("agent_config") or {})
        client_info = ClientInfo.model_validate(payload.get("client_info") or {})
        capabilities = ClientCapabilities.model_validate(
            payload.get("client_capabilities") or {}
        )
        raw_name = payload.get(AGENT_NAME_PAYLOAD_KEY)
        agent_name = (
            raw_name
            if isinstance(raw_name, str) and raw_name
            else spec.agent_id.replace(":", "-")
        )
        return await self._process.open_root(
            RootOpenRequest(
                options=options,
                client_info=client_info,
                client_capabilities=capabilities,
                # The child's own transcript dir, under the root session's
                # agents/ subtree: the child owns the only writer.
                session_logging=child_transcript_logging(root_session_dir, agent_name),
                is_subagent=True,
                parent_session_id=spec.parent_session_id,
            )
        )

    async def close(self) -> None:
        await self._process.close()


async def serve_child_agent(
    *,
    factory: RootLoopFactory,
    reader: BinaryLineReader | None = None,
    writer: BinaryLineWriter | None = None,
    announce_ready: bool = False,
) -> None:
    """Serve a child agent over the given pipes, announcing readiness first."""
    if reader is None or writer is None:
        transport: JsonRpcTransport = StdioJsonRpcTransport.from_standard_streams()
        ready_writer: BinaryLineWriter = sys.stdout.buffer
    else:
        transport = StdioJsonRpcTransport(reader, writer)
        ready_writer = writer
    if announce_ready:
        write_ready_frame(ready_writer)
    try:
        await ChildAgentServer(transport, factory=factory).serve()
    finally:
        with suppress(Exception):
            await transport.close()


def main() -> None:
    from chartreux.core.config import load_dotenv_values

    init_harness_files_manager("user", "project")
    init_file_logging(LOG_FILE.path, repair_path=LITERAL_LOG_FILE.path)
    load_dotenv_values()
    logger.info("Child agent process starting")
    asyncio.run(
        serve_child_agent(
            factory=HarnessChildLoopFactory(), announce_ready=bootstrap_requested()
        )
    )


if __name__ == "__main__":
    main()
