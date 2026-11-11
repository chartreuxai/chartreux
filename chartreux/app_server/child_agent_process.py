"""Coordinator-side process-backed adapter for the subagent-spawn port.

Each spawn gives the subagent its own OS process via the WP1 bootstrap
handshake (:func:`process_bootstrap.spawn_process`) running the
:mod:`chartreux.app_server.child_agent` entrypoint. ``subagent/open`` carries
the spawn spec, turn requests stream events back in production order, and the
terminal ``agent/turn/ended`` frame becomes the completion source's final
frame — the same agent wire the root agent process speaks, so both topologies
run the same agent components.

A spawned child is always fresh, never resumed (design.md:176-180); the
handle's ``close`` terminates and reaps the process and returns its place in
the coordinator's process-tree budget. Admission, capacity, and the event feed
that consumes the completion source land with [5B]; the physical tree cap is a
separate hard bound the adapter can never exceed ([7A]).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Mapping, Sequence
from contextlib import aclosing
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from chartreux.app_server._subagent_recovery import (
    ChildTranscriptReconciler,
    ChildTranscriptReconciliation,
)
from chartreux.app_server._subagent_transcript import (
    ChildTranscriptError,
    root_session_dir_from_spec,
)
from chartreux.app_server.agent_wire import (
    SUBAGENT_OPEN_METHOD,
    TURN_OUTCOME_COMPLETED,
    TURN_OUTCOME_FAILED,
    TURN_OUTCOME_INTERRUPTED,
)
from chartreux.app_server.child_agent import CHILD_AGENT_MODULE
from chartreux.app_server.process_bootstrap import (
    DEFAULT_SPAWN_READY_TIMEOUT,
    ProcessTreeBudget,
    ProcessTreeShutdown,
    ProcessTreeShutdownReport,
    SpawnedProcess,
    spawn_process,
)
from chartreux.app_server.registry_manifest import RegistryManifest
from chartreux.app_server.root_agent_process import (
    DEFAULT_REQUEST_TIMEOUT,
    AgentProcessTurnFailed,
    RootAgentProcess,
    RootAgentRequestError,
)
from chartreux.app_server.subagent_ports import (
    SpawnedSubagent,
    SubagentCompletion,
    SubagentCompletionSource,
    SubagentFrame,
    SubagentSpawnDeniedError,
    SubagentSpawnPort,
    SubagentSpawnSpec,
    spawn_spec_to_wire,
)
from chartreux.app_server.transport import (
    MULTI_PROCESS_TRANSPORT_SELECTORS,
    TRANSPORT_ENV_VAR,
)
from chartreux.core.agent_loop._execution_port import ExecutionTurnRequest


class ChildAgentProcess(RootAgentProcess):
    """A spawned child agent process; the open frame carries the spawn spec."""

    def __init__(
        self,
        spawned: SpawnedProcess,
        *,
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
    ) -> None:
        super().__init__(spawned, request_timeout=request_timeout)
        self.child_session_dir: str | None = None
        self.root_session_dir: str | None = None

    async def policy_request(
        self, method: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Carry one policy-tree frame to this child (WP5 [5D]).

        The child owns its policy authority; the coordinator-side transaction
        only drives the four phases, so a :class:`ProcessPolicyNode` wraps this
        client and the child's own :class:`PolicyNodeService` answers.
        """
        return await self._request(method, params)

    async def open_subagent(self, spec: SubagentSpawnSpec) -> str:
        """Open the child agent and adopt the session id it announces.

        The child reports the transcript directory it created for itself; the
        coordinator keeps it for live paging and crash reconciliation and
        never writes into it.
        """
        result = await self._request(SUBAGENT_OPEN_METHOD, spawn_spec_to_wire(spec))
        session_id = result.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise RootAgentRequestError(
                "MalformedOpenResponse", "the agent opened without a session id"
            )
        self.session_id = session_id
        session_dir = result.get("session_dir")
        self.child_session_dir = (
            session_dir if isinstance(session_dir, str) and session_dir else None
        )
        return session_id


class ProcessSubagentCompletionSource:
    """Stream one child run over the wire, ending with the completion frame."""

    def __init__(self, agent_id: str, process: ChildAgentProcess) -> None:
        self._agent_id = agent_id
        self._process = process

    @property
    def agent_id(self) -> str:
        return self._agent_id

    def stream(
        self, request: ExecutionTurnRequest
    ) -> AsyncGenerator[SubagentFrame, None]:
        return self._stream(request)

    async def _stream(
        self, request: ExecutionTurnRequest
    ) -> AsyncGenerator[SubagentFrame, None]:
        try:
            async with aclosing(self._process.run_turn(request)) as events:
                async for event in events:
                    yield event
        except asyncio.CancelledError:
            yield SubagentCompletion(outcome=TURN_OUTCOME_INTERRUPTED)
            return
        except AgentProcessTurnFailed as exc:
            yield SubagentCompletion(outcome=TURN_OUTCOME_FAILED, error=str(exc))
            return
        yield SubagentCompletion(outcome=TURN_OUTCOME_COMPLETED)


class ProcessSpawnedSubagent:
    """A spawned subagent living in its own OS process."""

    def __init__(self, agent_id: str, process: ChildAgentProcess) -> None:
        self._agent_id = agent_id
        self._process = process
        self._completion_source = ProcessSubagentCompletionSource(agent_id, process)

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @property
    def process(self) -> subprocess.Popen[bytes]:
        return self._process.process

    @property
    def alive(self) -> bool:
        """Whether this child's process is running (WP7 [7C] death detection)."""
        return self._process.alive

    @property
    def completion_source(self) -> SubagentCompletionSource:
        return self._completion_source

    async def close(self) -> None:
        await self._process.close()

    def reconcile_death(self) -> ChildTranscriptReconciliation:
        """Adjudicate this child after its process died, without replaying.

        Reads — never writes — the durable markers the child left in its own
        transcript directory, so a crash reconciles as never executed,
        started-with-an-unknown-outcome, or completed with an unconsumed
        result.
        """
        child_dir = self._process.child_session_dir
        root_session_dir = self._process.root_session_dir
        if child_dir is None or root_session_dir is None:
            raise ChildTranscriptError(
                f"Child {self._agent_id} left no transcript directory to reconcile"
            )
        return ChildTranscriptReconciler(Path(root_session_dir)).reconcile(
            Path(child_dir), agent_id=self._agent_id
        )


def child_agent_command() -> list[str]:
    """The module invocation the coordinator spawns as a child agent."""
    return [sys.executable, "-m", CHILD_AGENT_MODULE]


def multi_process_child_selected(environ: Mapping[str, str] | None = None) -> bool:
    """Whether this run selects the multi-process (stdio) route."""
    source = os.environ if environ is None else environ
    selector = source.get(TRANSPORT_ENV_VAR, "").strip().lower()
    return selector in MULTI_PROCESS_TRANSPORT_SELECTORS


class ProcessSubagentSpawn(SubagentSpawnPort):
    """Spawn adapter that gives each subagent its own OS process."""

    def __init__(
        self,
        *,
        env: Mapping[str, str] | None = None,
        cwd: str | os.PathLike[str] | None = None,
        ready_timeout: float = DEFAULT_SPAWN_READY_TIMEOUT,
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
        command: Sequence[str] | None = None,
        manifest: RegistryManifest | None = None,
        budget: ProcessTreeBudget | None = None,
    ) -> None:
        self._env = env
        self._cwd = cwd
        self._ready_timeout = ready_timeout
        self._request_timeout = request_timeout
        self._command = command
        self._manifest = manifest
        # One budget per coordinator-side spawn adapter: every child this
        # coordinator spawns draws on the same hard tree cap (design.md:448).
        self._budget = budget if budget is not None else ProcessTreeBudget()
        self._tree_shutdown: ProcessTreeShutdown | None = None

    async def shutdown_tree(self) -> ProcessTreeShutdownReport:
        """Stop the whole spawned-child tree in design order (WP7 [7B]).

        Admission closes before the first signal, every live child is
        terminated (grace, then killed), its pipes close, and the tree is
        reaped — all inside one 10 s bound. A spawn arriving meanwhile is
        denied typed by the tree budget, and the budget reopens only once the
        old tree is reaped.
        """
        if self._tree_shutdown is None:
            self._tree_shutdown = ProcessTreeShutdown(self._budget)
        return await self._tree_shutdown.shutdown()

    async def spawn(self, spec: SubagentSpawnSpec) -> SpawnedSubagent:
        """Spawn one child agent process and open it with the spawn spec.

        A run that selects ``in_process`` (or names no transport) never
        reaches a physical spawn: it gets the typed denial, matching the root
        agent process's selection guard. Each spawn is a FRESH incarnation:
        the child creates its own transcript directory and the manifest route
        carries a new incarnation id, so a later child process is never
        resumed as the earlier one. An exhausted tree budget raises the typed
        :class:`SpawnCapExceededError` before any process starts.
        """
        if not spec.agent_id:
            raise ValueError("a spawn spec carries an agent id")
        source = os.environ if self._env is None else self._env
        if not multi_process_child_selected(source):
            raise SubagentSpawnDeniedError(
                f"{TRANSPORT_ENV_VAR} does not select the multi-process (stdio) "
                "route; the in-process adapter stays selected"
            )
        spawned = await spawn_process(
            self._command or child_agent_command(),
            env=source,
            cwd=self._cwd,
            ready_timeout=self._ready_timeout,
            budget=self._budget,
        )
        process = ChildAgentProcess(spawned, request_timeout=self._request_timeout)
        root_session_dir = root_session_dir_from_spec(spec.payload)
        process.root_session_dir = (
            str(root_session_dir) if root_session_dir is not None else None
        )
        try:
            await process.open_subagent(spec)
            if self._manifest is not None:
                # Live pages for the child are proxied through the [2C-2a]
                # live-page port: the coordinator's manifest row routes to the
                # child process's own page reader, never to a disk copy.
                process.register_live_route(self._manifest, spec.agent_id)
        except BaseException:
            await process.close()
            raise
        return ProcessSpawnedSubagent(spec.agent_id, process)

    async def respawn(
        self, spec: SubagentSpawnSpec, *, previous: SpawnedSubagent | None = None
    ) -> SpawnedSubagent:
        """Respawn a child as a FRESH incarnation, never a resume (WP7 [7C]).

        The previous child is reaped FIRST, so the new process never overlaps
        it and its tree-budget slot is free before the new spawn is admitted.
        The spec must not name an existing child session directory: a respawn
        is a new process with its own transcript record, never a resume of the
        earlier incarnation (design.md:176-180).
        """
        if spec.session_dir:
            raise SubagentSpawnDeniedError(
                "a respawn spawns a fresh child; the spawn spec must not name an "
                "existing child session directory (a child process is never resumed)"
            )
        if previous is not None:
            await previous.close()
        return await self.spawn(spec)


__all__ = [
    "ChildAgentProcess",
    "ProcessSpawnedSubagent",
    "ProcessSubagentCompletionSource",
    "ProcessSubagentSpawn",
    "child_agent_command",
    "multi_process_child_selected",
]
