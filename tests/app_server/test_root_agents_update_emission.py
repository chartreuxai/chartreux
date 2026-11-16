"""F10 pins: the owner's child lifecycle reaches the client's agents/update.

The browser rows the TUI renders come exclusively from the ``AgentsUpdate``
event (``agent_browser.py`` consumes nothing else, and no read populates rows
at mount), but on the process topology the root agent process owns the
children and the coordinator registry that owns the in-process emission holds
none of them -- so the browser showed zero rows for live root-owned children
(panel round 2, seats agent-65/agent-67; confirmed by the user's manual test).

The fix is the owner-side emission arm: the root process publishes one
id-less ``agent/agents/update`` notification per child-inventory mutation
carrying its full honest snapshot (the same ``AgentsInventoryReadResponse``
the inventory read serves), the coordinator's notification pump relays it
onto the client-facing ``agents/update`` through the sink the live backend
binds when the root's ready frame advertised the capability, and the
client's event stream delivers it exactly as the in-process registry's own
emission would be. These pins hold every leg of that chain and prove the
root-owned child produces a visible browser row on the process topology.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Iterable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from chartreux.app_server import _runtime as runtime
from chartreux.app_server.agent_wire import (
    AGENT_AGENTS_UPDATE_METHOD,
    AGENT_RUN_COMPLETED_METHOD,
    AGENT_RUN_COMPLETIONS_READ_METHOD,
    COORDINATOR_RECEIVING_NOTIFICATIONS,
    AgentRunCompletion,
    AgentWireError,
    decode_run_completion,
    encode_run_completion,
    required_agent_capabilities,
)
from chartreux.app_server.client import AppServerClient
from chartreux.app_server.events import AgentsUpdate, TurnStarted
from chartreux.app_server.process_bootstrap import (
    ADVERTISED_CAPABILITIES,
    AGENT_WIRE_CAPABILITY_LIVE_CHILD_AUTHORITY,
    AGENT_WIRE_CAPABILITY_ROOT_AGENTS_UPDATE,
    AGENT_WIRE_CAPABILITY_ROOT_RUN_COMPLETIONS,
    encode_ready_frame,
    parse_ready_frame,
)
from chartreux.app_server.process_session_backend import completion_notification_text
from chartreux.app_server.protocol import AgentsInventoryReadResponse, ReleaseOutcome
from chartreux.app_server.root_agent import AgentChildHost, RootAgentServer
from chartreux.app_server.subagent_ports import (
    TURN_OUTCOME_COMPLETED,
    SubagentCompletion,
)
from chartreux.app_server.transport import memory_transport_pair
from chartreux.core.config.models import SessionLoggingConfig
from chartreux.core.events import AssistantEvent
from chartreux.core.session_types import CommittedModelIdentity
from chartreux.core.subagents import RunStatus, RunStopReason, TaskArgs
from chartreux.core.tools.base import InvokeContext
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.app_server import attach_test_app_server_session
from tests.stubs.fake_backend import FakeBackend

_UPDATE_TIMEOUT = 10.0
"""Bounded patience for one mutation's emission, generous under load."""


def test_the_ready_frame_advertises_the_root_agents_update_capability() -> None:
    """F10: this build announces the emission arm its relays depend on.

    The live backend binds the update sink only when the root's ready frame
    advertised the capability, so the advertised set must carry it and the
    encode/parse round trip must preserve it -- the same contract the
    live-inventory capability holds.
    """
    assert AGENT_WIRE_CAPABILITY_ROOT_AGENTS_UPDATE in ADVERTISED_CAPABILITIES
    assert (
        AGENT_WIRE_CAPABILITY_ROOT_AGENTS_UPDATE
        in parse_ready_frame(encode_ready_frame()).capabilities
    )


def test_the_coordinator_admits_the_agents_update_notification() -> None:
    """F10: the agent-origin update frame must pass the closed dispatch table.

    An unlisted notification is a recorded diagnostic and never reaches the
    pump's handlers, so the method has to sit in the coordinator's receiving
    table for the relay to fire at all.
    """
    assert AGENT_AGENTS_UPDATE_METHOD in COORDINATOR_RECEIVING_NOTIFICATIONS


async def _close_spawned() -> None:
    return None


class _BlockingSource:
    """One completion source whose run blocks until its frames are posted."""

    def __init__(self) -> None:
        self._release = asyncio.Event()
        self._frames: list[Any] = []

    def complete(self, frames: list[Any]) -> None:
        self._frames = frames
        self._release.set()

    def stream(self, request: Any) -> AsyncGenerator[Any, None]:
        return self._iter(request)

    async def _iter(self, request: Any) -> AsyncGenerator[Any, None]:
        await self._release.wait()
        for frame in self._frames:
            yield frame


class _ScriptingSpawnAdapter:
    """The spawn adapter that hands each published child its run source."""

    agent_id = "recorded"

    def __init__(self, source: Any, *, client: Any = None) -> None:
        self._source = source
        self._client = client
        self.specs: list[Any] = []

    async def spawn(self, spec: Any) -> Any:
        self.specs.append(spec)
        return SimpleNamespace(
            agent_id=spec.agent_id,
            session_id="child-session-1",
            child_session_dir=None,
            completion_source=self._source,
            wire_client=self._client,
            close=_close_spawned,
        )


class _SnapshottingClient:
    """A spawned child's wire client answering its own session snapshot (G3)."""

    def __init__(
        self, committed: Any, *, context_tokens: int, thinking: str | None = None
    ) -> None:
        self._committed = committed
        self._context_tokens = context_tokens
        self._thinking = thinking
        self.reads = 0
        # G5: a process child is published only when its ready frame announced
        # the mandatory live-child-authority capability, so this fake peer
        # advertises it exactly as the production child does.
        self.advertised_capabilities = frozenset({
            AGENT_WIRE_CAPABILITY_LIVE_CHILD_AUTHORITY
        })

    async def snapshot(self, *, include_history: bool = True) -> Any:
        assert include_history is False
        self.reads += 1
        config = (
            SimpleNamespace(
                effective_thinking=SimpleNamespace(
                    requested=self._thinking, semantic=self._thinking, known=True
                ),
                active_model=SimpleNamespace(thinking=self._thinking),
            )
            if self._thinking is not None
            else None
        )
        return SimpleNamespace(
            committed_model=self._committed,
            state=SimpleNamespace(session=SimpleNamespace(model=None)),
            runtime=SimpleNamespace(
                stats=SimpleNamespace(
                    context_tokens=self._context_tokens,
                    last_turn_prompt_tokens=None,
                    last_turn_cached_tokens=None,
                    last_turn_cached_tokens_reported=False,
                ),
                context_window=32_000,
                config=config,
            ),
        )


def _process_root(tmp_path: Path) -> Any:
    """The loop shape a process-owned root actually opens (F6/E1 pattern)."""
    return build_test_agent_loop(
        config=build_test_vibe_config(
            session_logging=SessionLoggingConfig(
                enabled=True,
                save_dir=str(tmp_path / "sessions"),
                session_prefix="session",
            )
        ),
        backend=FakeBackend([[mock_llm_chunk(content="ack")]]),
    )


def _task_args() -> TaskArgs:
    return TaskArgs(task="Slow inspection", identity="coder", retain=True)


def _ctx(loop: Any, call_id: str) -> InvokeContext:
    return InvokeContext(tool_call_id=call_id, session_id=loop.session_id)


async def _next_agents_update(stream: Any) -> AgentsUpdate:
    """Drain the client event stream until one agents/update lands."""
    while True:
        event = await asyncio.wait_for(stream.__anext__(), timeout=_UPDATE_TIMEOUT)
        if isinstance(event, AgentsUpdate):
            return event


async def _next_update_frame(
    frames: AsyncIterator[dict[str, Any]],
) -> AgentsInventoryReadResponse:
    """Drain the agent wire until one agents/update notification lands.

    The frame must be id-less (a notification, never a request the peer
    would answer) and its payload must decode as the owner's honest
    snapshot type -- the same projection the inventory read serves.
    """
    while True:
        message = await asyncio.wait_for(frames.__anext__(), timeout=_UPDATE_TIMEOUT)
        if message.get("method") == AGENT_AGENTS_UPDATE_METHOD:
            assert "id" not in message
            return AgentsInventoryReadResponse.model_validate(message["params"])


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_the_owner_publishes_each_mutation_as_an_id_less_snapshot(
    tmp_path: Path,
) -> None:
    """F10: every host inventory mutation emits one wire notification.

    The serving harness binds the host's mutation seam as the publisher, so
    a retained child's spawn, its background admission and its release each
    carry a fresh full snapshot over ``agent/agents/update`` -- and a
    snapshot never resurrects a child the release already removed.
    """
    loop = _process_root(tmp_path)
    host = AgentChildHost(loop, spawn=_ScriptingSpawnAdapter(_BlockingSource()))
    transport, peer = memory_transport_pair()
    server = RootAgentServer(transport, factory=cast(Any, SimpleNamespace()))
    server.install_child_host(host)
    serving = asyncio.create_task(server.serve())
    frames = peer.messages()
    try:
        handle = await host.publish(
            _task_args(), _ctx(loop, "task-1"), attempt_id="attempt-1", retained=True
        )
        update = await _next_update_frame(frames)
        assert update.session_id == loop.session_id
        assert update.agents and update.agents[0].agent_id == handle.agent_id
        assert update.agents[0].availability == "idle"

        host.admit_background(handle, _task_args(), _ctx(loop, "task-1"))
        running = await _next_update_frame(frames)
        states = {agent.agent_id: agent for agent in running.agents}
        assert states[handle.agent_id].availability == "running"

        await host.release_agent(handle.agent_id)
        # The unguarded release interrupts and joins the live run first, and
        # that cancel-mutation legitimately publishes a pre-pop snapshot -- so
        # drain until the post-release snapshot carries the removal.
        released: AgentsInventoryReadResponse | None = None
        for _ in range(5):
            frame = await _next_update_frame(frames)
            if handle.agent_id not in {agent.agent_id for agent in frame.agents}:
                released = frame
                break
        assert released is not None
        assert handle.agent_id not in {agent.agent_id for agent in released.agents}
    finally:
        await transport.close()
        await peer.close()
        await asyncio.gather(serving, return_exceptions=True)
        await host.close()
        await loop.aclose()


async def _close_session_stack(
    session: Any, client: Any, harness: Any, loop: Any, host: Any
) -> None:
    for close in (session.close, client.close, harness.close):
        try:
            await close()
        except BaseException:
            pass
    try:
        await host.close()
    except BaseException:
        pass
    await loop.aclose()


@pytest.mark.asyncio
@pytest.mark.timeout(90)
async def test_root_owned_children_populate_the_client_agents_update_stream(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """F10: a root-owned child produces a visible browser row on the process topology.

    The full chain rides end-to-end on the production routing: the owner's
    mutation emits ``agent/agents/update`` over the agent wire, the
    coordinator's pump relays it onto the client-facing ``agents/update``
    through the sink the capability gate bound, and the event stream the
    TUI browser consumes (``handle_agents_update`` is its only row source)
    delivers the child's row for every lifecycle moment -- spawn, run,
    settle, release. On the pre-F10 base no update ever arrives and the
    first drain times out.
    """
    from tests.app_server.backend_contract.conftest import install_memory_root_agent

    loop = _process_root(tmp_path)
    source = _BlockingSource()
    host = AgentChildHost(loop, spawn=_ScriptingSpawnAdapter(source))
    install_memory_root_agent(monkeypatch, loop, child_host=host)
    client_transport, server_transport = memory_transport_pair()
    harness = await runtime.create_harness_server(
        server_transport, transport_kind="stdio"
    )
    client = AppServerClient(client_transport, run_peer=harness.serve)
    session = await attach_test_app_server_session(client)
    stream = session.events()
    try:
        handle = await host.publish(
            _task_args(), _ctx(loop, "task-1"), attempt_id="attempt-e2e", retained=True
        )
        spawn_update = await _next_agents_update(stream)
        rows = {agent.agent_id: agent for agent in spawn_update.agents}
        assert handle.agent_id in rows
        assert rows[handle.agent_id].availability == "idle"
        # G2: the row the browser renders carries the stable per-root name,
        # not the handle-derived ``coder-coder-attempt-...`` shape.
        assert rows[handle.agent_id].name == "coder-1"

        host.admit_background(handle, _task_args(), _ctx(loop, "task-1"))
        run_update = await _next_agents_update(stream)
        rows = {agent.agent_id: agent for agent in run_update.agents}
        assert rows[handle.agent_id].availability == "running"

        source.complete([SubagentCompletion(outcome=TURN_OUTCOME_COMPLETED)])
        settled_update = await _next_agents_update(stream)
        rows = {agent.agent_id: agent for agent in settled_update.agents}
        assert rows[handle.agent_id].availability == "idle"
        assert rows[handle.agent_id].last_run_status == "completed"

        ack = await session.release_agent(handle.agent_id)
        assert ack.outcome is ReleaseOutcome.RELEASED
        released_row = False
        for _ in range(5):
            released_update = await _next_agents_update(stream)
            if handle.agent_id not in {
                agent.agent_id for agent in released_update.agents
            }:
                released_row = True
                break
        assert released_row
    finally:
        await _close_session_stack(session, client, harness, loop, host)


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_the_relay_drops_undecodable_and_foreign_snapshots_without_wedging(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """F10: the relay's guards drop bad frames and never poison the stream.

    The relay is a presentation-delta consumer, not a request handler: an
    undecodable payload and a snapshot that names another session both drop
    as lost deltas, and the first real mutation afterwards still reaches the
    client -- the pump stays unwedged.
    """
    from tests.app_server.backend_contract.conftest import install_memory_root_agent

    loop = _process_root(tmp_path)
    host = AgentChildHost(loop, spawn=_ScriptingSpawnAdapter(_BlockingSource()))
    install_memory_root_agent(monkeypatch, loop, child_host=host)
    client_transport, server_transport = memory_transport_pair()
    harness = await runtime.create_harness_server(
        server_transport, transport_kind="stdio"
    )
    client = AppServerClient(client_transport, run_peer=harness.serve)
    session = await attach_test_app_server_session(client)
    stream = session.events()
    try:
        backend = cast(Any, harness._server._root)
        await backend._relay_agents_update(
            AGENT_AGENTS_UPDATE_METHOD, {"not": "a snapshot"}
        )
        foreign = AgentsInventoryReadResponse(
            session_id="some-other-session", generation=1, revision=1, agents=[]
        )
        await backend._relay_agents_update(
            AGENT_AGENTS_UPDATE_METHOD, foreign.model_dump(mode="json")
        )
        handle = await host.publish(
            _task_args(),
            _ctx(loop, "task-1"),
            attempt_id="attempt-guard",
            retained=True,
        )
        update = await _next_agents_update(stream)
        assert handle.agent_id in {agent.agent_id for agent in update.agents}
    finally:
        await _close_session_stack(session, client, harness, loop, host)


# ---------------------------------------------------------------------------
# G1b: the settled-run completion carrier and the parent wake
# ---------------------------------------------------------------------------

_WAKE_TIMEOUT = 20.0
"""Bounded patience for one wake turn, generous under a loaded parallel run."""


def _completion(
    *,
    session_id: str = "session-1",
    agent_id: str = "agent-1",
    run_id: str = "run-1",
    name: str = "Coder",
    status: str = RunStatus.COMPLETED.value,
    stop_reason: str | None = None,
    result_available: bool = True,
) -> AgentRunCompletion:
    return AgentRunCompletion(
        session_id=session_id,
        agent_id=agent_id,
        run_id=run_id,
        name=name,
        status=status,
        stop_reason=stop_reason,
        result_available=result_available,
    )


def test_the_ready_frame_advertises_the_root_run_completions_capability() -> None:
    """G1b: this build announces the completion carrier its wake depends on.

    The live backend binds the completion sink only when the root's ready frame
    advertised the capability -- exactly the gate the F10 update sink sits
    behind -- so the advertised set must carry it and the encode/parse round
    trip must preserve it.
    """
    assert AGENT_WIRE_CAPABILITY_ROOT_RUN_COMPLETIONS in ADVERTISED_CAPABILITIES
    assert (
        AGENT_WIRE_CAPABILITY_ROOT_RUN_COMPLETIONS
        in parse_ready_frame(encode_ready_frame()).capabilities
    )


def test_the_coordinator_admits_the_run_completed_notification() -> None:
    """G1b: the agent-origin completion frame must pass the closed table.

    An unlisted notification is a recorded diagnostic and never reaches the
    pump's handlers, so the method sits in the coordinator's receiving table --
    where an older peer drops it as an unknown notification instead of breaking
    (the additive-carrier contract the F10 frame holds).
    """
    assert AGENT_RUN_COMPLETED_METHOD in COORDINATOR_RECEIVING_NOTIFICATIONS
    assert required_agent_capabilities(AGENT_RUN_COMPLETIONS_READ_METHOD, {}) == (
        frozenset({AGENT_WIRE_CAPABILITY_ROOT_RUN_COMPLETIONS})
    )


def test_the_completion_codec_round_trips_and_refuses_an_inexact_frame() -> None:
    """G1b: the frame's key set is exact -- a skewed frame is refused typed.

    The wake drives a parent turn, so a frame that adds or omits a field is a
    version skew between peers and is refused, never repaired into a run the
    owner did not report.
    """
    completion = _completion(stop_reason=RunStopReason.BUDGET_EXCEEDED.value)
    assert decode_run_completion(encode_run_completion(completion)) == completion

    with pytest.raises(AgentWireError):
        decode_run_completion({**encode_run_completion(completion), "extra": 1})

    trimmed = encode_run_completion(completion)
    del trimmed["stop_reason"]
    with pytest.raises(AgentWireError):
        decode_run_completion(trimmed)


def test_the_wake_text_matches_the_in_process_notification_wording() -> None:
    """G1b: the process-topology wake says exactly what the registry says.

    The in-process registry's delivered notification is the pinned wording; the
    coordinator-side builder must reproduce it byte for byte, status words
    included, so the model reads one notification shape on both topologies.
    """
    assert completion_notification_text(_completion()) == (
        "Background agent Coder (agent-1, run run-1) completed."
        " Use get_agent_result to retrieve the result."
        " Retrieve this run by its run_id; continue independent work"
        " or end your turn for other running agents rather than"
        " waiting by default."
    )
    statuses = {
        RunStopReason.BUDGET_EXCEEDED.value: (
            RunStatus.COMPLETED.value,
            "budget-stopped (budget exceeded)",
        ),
        RunStopReason.BUDGET_UNVERIFIABLE.value: (
            RunStatus.FAILED.value,
            "budget-stopped (budget unverifiable)",
        ),
        RunStopReason.USER_CANCELLED.value: (
            RunStatus.CANCELLED.value,
            "cancelled (by user)",
        ),
        RunStopReason.RETASKED.value: (RunStatus.CANCELLED.value, "retasked"),
        RunStopReason.ERROR.value: (RunStatus.CANCELLED.value, "cancelled"),
    }
    for stop_reason, (status, word) in statuses.items():
        text = completion_notification_text(
            _completion(status=status, stop_reason=stop_reason)
        )
        assert text.startswith(f"Background agent Coder (agent-1, run run-1) {word}.")
    assert completion_notification_text(
        _completion(status=RunStatus.FAILED.value)
    ).startswith("Background agent Coder (agent-1, run run-1) failed.")
    assert completion_notification_text(
        _completion(
            status=RunStatus.COMPLETED.value, stop_reason=RunStopReason.ERROR.value
        )
    ).startswith("Background agent Coder (agent-1, run run-1) completed.")


async def _next_completion_frame(
    frames: AsyncIterator[dict[str, Any]],
) -> AgentRunCompletion:
    """Drain the agent wire until one run-completion notification lands.

    The frame must be id-less (a notification, never a request the peer would
    answer) and its payload must decode as the typed completion.
    """
    while True:
        message = await asyncio.wait_for(frames.__anext__(), timeout=_UPDATE_TIMEOUT)
        if message.get("method") == AGENT_RUN_COMPLETED_METHOD:
            assert "id" not in message
            return decode_run_completion(message["params"])


async def _wait_for_model_text(model: Any, text: str) -> None:
    """Wait until one completion request carried ``text`` verbatim."""
    deadline = asyncio.get_running_loop().time() + _WAKE_TIMEOUT
    while asyncio.get_running_loop().time() < deadline:
        for messages in model.requests_messages:
            if any(text in str(message.content) for message in messages):
                return
        await asyncio.sleep(0.02)
    raise AssertionError(f"no model request carried the wake text: {text!r}")


async def _assert_no_model_request(model: Any, *, baseline: int) -> None:
    """Hold for a bounded window proving no further turn reached the model."""
    deadline = asyncio.get_running_loop().time() + 1.5
    while asyncio.get_running_loop().time() < deadline:
        assert len(model.requests_messages) == baseline
        await asyncio.sleep(0.02)


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_the_owner_emits_one_id_less_completion_per_settled_background_run(
    tmp_path: Path,
) -> None:
    """G1b: the settle seam publishes one typed, id-less completion.

    A background run settles where its result is captured, so the owner is the
    only process that can report it: one ``agent/run/completed`` notification
    carries the run's typed record -- session, agent, run, display name, the
    observed status and stop reason, and whether that run is still retrievable
    -- and the ledger the recovery read replays holds the same frame. The run
    the frame names is exactly what ``get_agent_result`` answers for.
    """
    loop = _process_root(tmp_path)
    source = _BlockingSource()
    host = AgentChildHost(loop, spawn=_ScriptingSpawnAdapter(source))
    transport, peer = memory_transport_pair()
    server = RootAgentServer(transport, factory=cast(Any, SimpleNamespace()))
    server.install_child_host(host)
    serving = asyncio.create_task(server.serve())
    frames = peer.messages()
    try:
        handle = await host.publish(
            _task_args(),
            _ctx(loop, "task-1"),
            attempt_id="attempt-completion",
            retained=True,
        )
        host.admit_background(handle, _task_args(), _ctx(loop, "task-1"))
        assert handle.task is not None
        source.complete([SubagentCompletion(outcome=TURN_OUTCOME_COMPLETED)])
        await asyncio.gather(handle.task, return_exceptions=True)

        completion = await _next_completion_frame(frames)
        assert completion.session_id == loop.session_id
        assert completion.agent_id == handle.agent_id
        assert completion.run_id == handle.run_id
        row = {item.agent_id: item for item in host.inventory_snapshot().agents}[
            handle.agent_id
        ]
        assert completion.name == row.name
        assert completion.status == RunStatus.COMPLETED.value
        assert completion.stop_reason is None
        assert completion.result_available is True

        ledger = host._run_completion_ledger()
        assert [item for item in ledger if item.run_id == handle.run_id] == [completion]
        result = await host.get_agent_result(handle.agent_id, handle.run_id)
        assert result is not None
        assert result.completed
        assert result.run_id == handle.run_id
    finally:
        await transport.close()
        await peer.close()
        await asyncio.gather(serving, return_exceptions=True)
        await host.close()
        await loop.aclose()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_the_emitted_row_and_completion_name_the_stable_sequence(
    tmp_path: Path,
) -> None:
    """G2: the browser row and the wake carry the clean per-root name.

    The row the TUI browser renders reads the owner's projected display name,
    and the completion's wake text names the same run, so both must carry the
    stable ``{identity}-{sequence}`` name rather than the handle-derived
    ``{identity}-{identity}-attempt-{uuid}`` shape the process path produced
    from its agent_id. The agent_id and run_id the model must use stay exactly
    as minted on both frames.
    """
    loop = _process_root(tmp_path)
    source = _BlockingSource()
    host = AgentChildHost(loop, spawn=_ScriptingSpawnAdapter(source))
    transport, peer = memory_transport_pair()
    server = RootAgentServer(transport, factory=cast(Any, SimpleNamespace()))
    server.install_child_host(host)
    serving = asyncio.create_task(server.serve())
    frames = peer.messages()
    try:
        handle = await host.publish(
            _task_args(),
            _ctx(loop, "task-1"),
            attempt_id="attempt-display-name",
            retained=True,
        )
        spawn_update = await _next_update_frame(frames)
        assert spawn_update.agents[0].name == "coder-1"

        host.admit_background(handle, _task_args(), _ctx(loop, "task-1"))
        source.complete([SubagentCompletion(outcome=TURN_OUTCOME_COMPLETED)])
        assert handle.task is not None
        await asyncio.gather(handle.task, return_exceptions=True)

        completion = await _next_completion_frame(frames)
    finally:
        await transport.close()
        await peer.close()
        await asyncio.gather(serving, return_exceptions=True)
        await host.close()
        await loop.aclose()

    assert completion.name == "coder-1"
    assert "attempt" not in completion.name
    assert completion.agent_id == handle.agent_id
    assert completion.agent_id.startswith("coder-attempt-")
    assert completion.run_id == handle.run_id


class _WakeStack:
    """The production stdio stack a completion wakes, over the memory wire."""

    def __init__(
        self,
        *,
        loop: Any,
        model: Any,
        source: Any,
        host: AgentChildHost,
        harness: Any,
        client: Any,
        session: Any,
    ) -> None:
        self.loop = loop
        self.model = model
        self.source = source
        self.host = host
        self.harness = harness
        self.client = client
        self.session = session
        # The process session backend (the wake owner) and its root process.
        self.backend = cast(Any, harness._server._root)
        self.process = cast(Any, self.backend._process)

    async def close(self) -> None:
        await _close_session_stack(
            self.session, self.client, self.harness, self.loop, self.host
        )

    async def settle_background(self, *, call_id: str) -> Any:
        """Publish, launch and settle one retained background child."""
        handle = await self.host.publish(
            _task_args(),
            _ctx(self.loop, call_id),
            attempt_id=f"attempt-{call_id}",
            retained=True,
        )
        self.host.admit_background(handle, _task_args(), _ctx(self.loop, call_id))
        assert handle.task is not None
        self.source.complete([SubagentCompletion(outcome=TURN_OUTCOME_COMPLETED)])
        await asyncio.gather(handle.task, return_exceptions=True)
        return handle


async def _wake_stack(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    turns: int = 3,
    model: Any = None,
) -> _WakeStack:
    """Build the memory-wire stdio stack with a multi-turn parent model."""
    from tests.app_server.backend_contract.conftest import install_memory_root_agent

    if model is None:
        model = FakeBackend([
            [mock_llm_chunk(content=f"ack-{index}")] for index in range(turns)
        ])
    loop = build_test_agent_loop(
        config=build_test_vibe_config(
            session_logging=SessionLoggingConfig(
                enabled=True,
                save_dir=str(tmp_path / "sessions"),
                session_prefix="session",
            )
        ),
        backend=model,
    )
    source = _BlockingSource()
    host = AgentChildHost(loop, spawn=_ScriptingSpawnAdapter(source))
    install_memory_root_agent(monkeypatch, loop, child_host=host)
    client_transport, server_transport = memory_transport_pair()
    harness = await runtime.create_harness_server(
        server_transport, transport_kind="stdio"
    )
    client = AppServerClient(client_transport, run_peer=harness.serve)
    session = await attach_test_app_server_session(client)
    return _WakeStack(
        loop=loop,
        model=model,
        source=source,
        host=host,
        harness=harness,
        client=client,
        session=session,
    )


async def _next_turn_started(stream: Any) -> TurnStarted:
    """Drain the client event stream until one turn starts."""
    while True:
        event = await asyncio.wait_for(stream.__anext__(), timeout=_WAKE_TIMEOUT)
        if isinstance(event, TurnStarted):
            return event


@pytest.mark.asyncio
@pytest.mark.timeout(90)
async def test_a_settled_background_run_wakes_the_idle_parent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """G1b: the process topology wakes an idle parent exactly as the registry does.

    The owner's completion frame crosses the agent wire, the coordinator's arm
    hands it to the wake owner the capability gate bound the sink to, and -- the
    parent being idle -- the backend starts its own injected turn carrying the
    byte-identical notification text. That turn is the model's next request, and
    the run it names answers ``get_agent_result`` at the same key.
    """
    stack = await _wake_stack(monkeypatch, tmp_path)
    stream = stack.session.events()
    try:
        handle = await stack.settle_background(call_id="task-1")
        started = await _next_turn_started(stream)
        assert started.turn.id
        completion = (await stack.process.read_run_completions(stack.loop.session_id))[
            -1
        ]
        assert (completion.agent_id, completion.run_id) == (
            handle.agent_id,
            handle.run_id,
        )
        await _wait_for_model_text(
            stack.model, completion_notification_text(completion)
        )
        result = await stack.host.get_agent_result(handle.agent_id, handle.run_id)
        assert result is not None and result.completed
    finally:
        await stack.close()


@pytest.mark.asyncio
@pytest.mark.timeout(90)
async def test_a_duplicate_completion_wakes_the_parent_only_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """G1b: at-most-once per settled run, even when the frame is redelivered.

    The delivered set is keyed (agent_id, run_id) and re-derived from the
    owner's own inventory and ledger at install, so a duplicate frame -- or a
    resume sweep re-reading the same ledger -- never double-wakes the parent.
    """
    stack = await _wake_stack(monkeypatch, tmp_path)
    stream = stack.session.events()
    try:
        await stack.settle_background(call_id="task-1")
        await _next_turn_started(stream)
        completion = (await stack.process.read_run_completions(stack.loop.session_id))[
            -1
        ]
        # The first wake's own request is the baseline: the duplicate below
        # must not add one.
        await _wait_for_model_text(
            stack.model, completion_notification_text(completion)
        )
        baseline = len(stack.model.requests_messages)
        await stack.backend._handle_run_completion(completion)
        await _assert_no_model_request(stack.model, baseline=baseline)
        assert (
            completion.agent_id,
            completion.run_id,
        ) in stack.backend._delivered_wakes
    finally:
        await stack.close()


@pytest.mark.asyncio
@pytest.mark.timeout(90)
async def test_a_frame_lost_to_the_pipe_is_recovered_by_the_turn_terminal_sweep(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """G1b: the notification is at-most-once, so the sweep is the backstop.

    With the sink unbound the owner still records the settled run in its
    ledger, and nothing wakes the parent. The next turn terminal -- the point
    where the coordinator is about to admit work anyway -- re-reads that ledger
    and wakes the run its delivered set does not name. The contract is
    at-most-once plus the sweep, never exactly-once framing.
    """
    stack = await _wake_stack(monkeypatch, tmp_path, turns=4)
    try:
        identity = await stack.host.publish(
            _task_args(),
            _ctx(stack.loop, "task-1"),
            attempt_id="attempt-1",
            retained=True,
        )
        stack.host.admit_background(identity, _task_args(), _ctx(stack.loop, "task-1"))
        # The frame is lost: the owner's notification has nowhere to land. The
        # sink stays unbound for the settle, so the emission is dropped exactly
        # as a frame lost between the peers would be.
        assert stack.process._run_completion_sink is not None
        stack.process._run_completion_sink = None
        stack.source.complete([SubagentCompletion(outcome=TURN_OUTCOME_COMPLETED)])
        assert identity.task is not None
        await asyncio.gather(identity.task, return_exceptions=True)
        assert len(stack.model.requests_messages) == 0
        assert await stack.process.read_run_completions(stack.loop.session_id)

        # A user turn's terminal is the sweep's admission point.
        consumed = [event async for event in stack.session.act("hello")]
        assert consumed
        completion = (await stack.process.read_run_completions(stack.loop.session_id))[
            -1
        ]
        await _wait_for_model_text(
            stack.model, completion_notification_text(completion)
        )
    finally:
        await stack.close()


@pytest.mark.asyncio
@pytest.mark.timeout(90)
async def test_a_cancelled_foreign_or_retired_frame_never_wakes_the_parent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """G1b: the wake's fences hold on every arm.

    A user-cancelled run is announced without starting a turn (the in-process
    gate), a frame naming another session is dropped, a malformed frame is
    dropped the way the agents/update relay drops a snapshot it cannot vouch
    for -- silently, because the owner's ledger is the recovery source -- and
    the stream keeps working afterwards. A retired generation wakes nothing
    even when a valid frame still arrives.
    """
    stack = await _wake_stack(monkeypatch, tmp_path)
    try:
        cancelled = _completion(
            session_id=stack.loop.session_id,
            stop_reason=RunStopReason.USER_CANCELLED.value,
            status=RunStatus.CANCELLED.value,
        )
        await stack.backend._handle_run_completion(cancelled)
        await _assert_no_model_request(stack.model, baseline=0)
        assert (
            completion_notification_text(cancelled) in stack.backend._pending_wake_texts
        )

        await stack.backend._handle_run_completion(
            _completion(session_id="some-other-session")
        )
        await _assert_no_model_request(stack.model, baseline=0)

        stack.process._route_sink_frame(AGENT_RUN_COMPLETED_METHOD, {"nope": True})
        assert not stack.process.diagnostics
        await _assert_no_model_request(stack.model, baseline=0)

        # The malformed drop never wedges the sink: a valid frame after it
        # still wakes the parent.
        valid = _completion(
            session_id=stack.loop.session_id, run_id="run-after-malformed"
        )
        await stack.backend._handle_run_completion(valid)
        await _wait_for_model_text(stack.model, completion_notification_text(valid))

        stack.backend.retire()
        await stack.backend._handle_run_completion(
            _completion(session_id=stack.loop.session_id, run_id="run-after-retire")
        )
        await _assert_no_model_request(
            stack.model, baseline=len(stack.model.requests_messages)
        )
    finally:
        await stack.close()


class _GatedBackend(FakeBackend):
    """A model whose completions wait until the test opens the gate.

    The steer arm needs a parent turn that is genuinely in flight; holding the
    provider call is what makes "the turn is still running" a fact rather than a
    race the test happens to win.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.gate = asyncio.Event()

    async def complete(self, **kwargs: Any) -> Any:
        await self.gate.wait()
        return await super().complete(**kwargs)

    async def complete_streaming(self, **kwargs: Any) -> AsyncGenerator[Any, None]:
        await self.gate.wait()
        async for chunk in super().complete_streaming(**kwargs):
            yield chunk


async def _wait_until(predicate: Any, *, timeout: float = _WAKE_TIMEOUT) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.02)
    raise AssertionError("the expected state was never reached")


async def _wait_for_transcript_text(stack: _WakeStack, text: str) -> None:
    """Wait until the agent's own transcript carries ``text`` verbatim.

    A steer commits its message into the running turn's transcript, which is
    what the loop reads at its next model boundary; the transcript (not a
    provider request) is therefore the steer's observable effect.
    """
    deadline = asyncio.get_running_loop().time() + _WAKE_TIMEOUT
    while asyncio.get_running_loop().time() < deadline:
        snapshot = await stack.process.snapshot()
        transcript = " ".join(
            str(getattr(block, "text", ""))
            for entry in (snapshot.state.history or [])
            for block in entry.content
        )
        if text in transcript:
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"the transcript never carried the wake text: {text!r}")


@pytest.mark.asyncio
@pytest.mark.timeout(90)
async def test_a_completion_steers_the_turn_already_in_flight(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """G1b: a busy parent is steered, never given a second concurrent turn.

    A turn in flight is the parent's attention: the wake is offered to that turn
    as a steer (``agent/turn/steer``) and turn admission stays untouched, which
    is the in-process registry's own choice. A steer the turn can no longer take
    -- it settled underneath the request -- falls back to the held text instead
    of dropping the wake, and the held text rides the next turn the parent runs.
    """
    model = _GatedBackend([
        [mock_llm_chunk(content="first")],
        [mock_llm_chunk(content="second")],
    ])
    stack = await _wake_stack(monkeypatch, tmp_path, model=model)
    try:
        turn = asyncio.create_task(_drain(stack.session.act("hold on")))
        await _wait_until(lambda: stack.backend._active_turn is not None)
        await stack.settle_background(call_id="task-1")
        completion = (await stack.process.read_run_completions(stack.loop.session_id))[
            -1
        ]
        text = completion_notification_text(completion)
        await _wait_until(
            lambda: (
                (completion.agent_id, completion.run_id)
                in stack.backend._delivered_wakes
            )
        )
        # The live turn absorbed the wake: no concurrent turn was admitted.
        assert stack.backend._turn_starts == 1
        model.gate.set()
        await turn
        if stack.backend._pending_wake_texts:
            # The steer lost the race to the turn's own terminal: the wake is
            # held for the next turn rather than dropped.
            assert stack.backend._pending_wake_texts == [text]
            await _drain(stack.session.act("next"))
        await _wait_for_transcript_text(stack, text)
        assert stack.backend._pending_wake_texts == []
    finally:
        await stack.close()


async def _drain(events: Any) -> list[Any]:
    return [event async for event in events]


@pytest.mark.asyncio
@pytest.mark.timeout(90)
async def test_a_wake_that_cannot_start_a_turn_rides_the_promoted_turn(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """G1b: the queued-turn decision is inject-on-promote, never a new turn.

    When a turn cannot be started (the parent is busy, or the queue already
    holds work), the wake text is held and delivered into the next promoted
    turn as an injected message -- the in-process pending injected message's
    own semantics. It never becomes a queue item the user sees and never a
    turn of its own.
    """
    from chartreux.app_server.models import SessionTextContentBlock
    from chartreux.app_server.protocol import TurnEnqueueParams, TurnUserInputEntry

    stack = await _wake_stack(monkeypatch, tmp_path)
    try:
        stack.backend._turn_queue.enqueue(
            TurnEnqueueParams(
                session_id=stack.loop.session_id,
                entries=[
                    TurnUserInputEntry(
                        entry_id="queued-1",
                        content=[SessionTextContentBlock(text="queued work")],
                    )
                ],
            )
        )
        completion = _completion(session_id=stack.loop.session_id)
        await stack.backend._handle_run_completion(completion)
        text = completion_notification_text(completion)
        assert stack.backend._pending_wake_texts == [text]
        assert stack.backend._turn_starts == 0

        await stack.backend._promote_next()
        await _wait_for_model_text(stack.model, text)
        await _wait_for_model_text(stack.model, "queued work")
        assert stack.backend._turn_starts == 1
        assert stack.backend._pending_wake_texts == []
    finally:
        await stack.close()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_the_child_telemetry_rides_the_emitted_inventory_snapshot(
    tmp_path: Path,
) -> None:
    """G3: the observed telemetry reaches the browser through the F10 emission.

    The emission carries whatever inventory the mutation's revision names, so
    the run's timing, its task summaries, the child-reported turn count and
    the child's own model/context must appear on the emitted row -- no read
    and no extra surface feeds the browser. The pre-G3 base emits the same
    frames with all of these unset.
    """
    loop = _process_root(tmp_path)
    source = _BlockingSource()
    committed = CommittedModelIdentity(
        base_model="alpha-model",
        provider="alpha",
        wire_name="alpha-wire",
        catalog_revision="test-fixture",
    )
    host = AgentChildHost(
        loop,
        spawn=_ScriptingSpawnAdapter(
            source, client=_SnapshottingClient(committed, context_tokens=99)
        ),
    )
    transport, peer = memory_transport_pair()
    server = RootAgentServer(transport, factory=cast(Any, SimpleNamespace()))
    server.install_child_host(host)
    serving = asyncio.create_task(server.serve())
    frames = peer.messages()
    try:
        args = _task_args()
        handle = await host.publish(
            args, _ctx(loop, "task-1"), attempt_id="attempt-telemetry", retained=True
        )
        await _next_update_frame(frames)
        host.admit_background(handle, args, _ctx(loop, "task-1"))
        await _next_update_frame(frames)
        source.complete([
            AssistantEvent(content="inspected"),
            SubagentCompletion(outcome=TURN_OUTCOME_COMPLETED, turns_used=1),
        ])
        assert handle.task is not None
        await asyncio.gather(handle.task, return_exceptions=True)
        emitted = None
        for _ in range(6):
            frame = await _next_update_frame(frames)
            row = next(
                (agent for agent in frame.agents if agent.agent_id == handle.agent_id),
                None,
            )
            if row is not None and row.effective_model is not None:
                emitted = row
                break
    finally:
        await transport.close()
        await peer.close()
        await asyncio.gather(serving, return_exceptions=True)
        await host.close()
        await loop.aclose()

    assert emitted is not None
    assert emitted.effective_model == "alpha/alpha-wire"
    assert emitted.base_model == "alpha-model"
    assert emitted.active_provider == "alpha"
    assert emitted.context_tokens == 99
    assert emitted.context_window == 32_000
    assert emitted.turns_used == 1
    assert emitted.initial_task_summary == "Slow inspection"
    assert emitted.current_task_summary == "Slow inspection"
    assert emitted.run_elapsed_seconds is not None


# ---------------------------------------------------------------------------
# H1: the live observation is a mutation, so the emission carries it mid-run
# ---------------------------------------------------------------------------

_LIVE_EMISSION_TIMEOUT = 5.0
"""Bounded patience for one live row to reach the root pipe (ticked at 0.02s)."""


class _HeldRunSource:
    """A run that relays its first frames, then holds until released.

    The hold is the whole point of the H1 emission pin: the child is still
    running, so the only way an emitted snapshot can name its model is the
    periodic observation -- the terminal carrier has not arrived.
    """

    def __init__(self, frames: Iterable[Any]) -> None:
        self._frames = list(frames)
        self._release = asyncio.Event()

    def release(self) -> None:
        self._release.set()

    def stream(self, request: Any) -> AsyncGenerator[Any, None]:
        return self._iter()

    async def _iter(self) -> AsyncGenerator[Any, None]:
        for frame in self._frames:
            yield frame
        await self._release.wait()
        yield SubagentCompletion(outcome=TURN_OUTCOME_COMPLETED, turns_used=1)


def _relayed_frame(event: Any, turns: int) -> Any:
    """One event as the child's pipe delivers it, carrying its run progress.

    The pre-H1 codec takes no progress argument at all, so such a peer can
    only send the plain event; the pin has to fail on the cadence then, not on
    a constructor mismatch.
    """
    from chartreux.app_server import agent_wire

    try:
        encoded = agent_wire.encode_event(event, live_turns_used=turns)
    except TypeError:
        return event
    return agent_wire.decode_event(encoded)


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_a_running_child_emits_its_telemetry_before_it_settles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """H1: a held run's telemetry reaches the browser through an emission.

    G3 read the child once, at settle, so every mutation the browser saw
    during a run carried empty telemetry. The cadence turns each successful
    read into an inventory mutation like any other, so the same F10 emission
    must deliver a snapshot naming the child's model, context and thinking --
    and the turn count its relayed frame reported -- while ``handle.task`` is
    still in flight. On the pre-H1 base no read happens before the settle, so
    no such frame ever lands and this pin times out.
    """
    from chartreux.app_server import root_agent

    monkeypatch.setattr(root_agent, "_CHILD_TELEMETRY_TICK_SECONDS", 0.02)
    loop = _process_root(tmp_path)
    committed = CommittedModelIdentity(
        base_model="alpha-model",
        provider="alpha",
        wire_name="alpha-wire",
        catalog_revision="test-fixture",
    )
    client = _SnapshottingClient(committed, context_tokens=77, thinking="high")
    source = _HeldRunSource([_relayed_frame(AssistantEvent(content="inspecting"), 1)])
    host = AgentChildHost(loop, spawn=_ScriptingSpawnAdapter(source, client=client))
    transport, peer = memory_transport_pair()
    server = RootAgentServer(transport, factory=cast(Any, SimpleNamespace()))
    server.install_child_host(host)
    serving = asyncio.create_task(server.serve())
    frames = peer.messages()
    live_row = None
    try:
        args = _task_args()
        handle = await host.publish(
            args,
            _ctx(loop, "task-1"),
            attempt_id="attempt-live-emission",
            retained=True,
        )
        await _next_update_frame(frames)
        host.admit_background(handle, args, _ctx(loop, "task-1"))
        await _next_update_frame(frames)
        assert handle.task is not None
        deadline = asyncio.get_running_loop().time() + _LIVE_EMISSION_TIMEOUT
        while asyncio.get_running_loop().time() < deadline:
            frame = await _next_update_frame(frames)
            row = next(
                (agent for agent in frame.agents if agent.agent_id == handle.agent_id),
                None,
            )
            if row is not None and row.effective_model is not None:
                live_row = row
                break
        assert not handle.task.done(), "the run settled before its live row was emitted"
        source.release()
        await asyncio.gather(handle.task, return_exceptions=True)
    finally:
        await transport.close()
        await peer.close()
        await asyncio.gather(serving, return_exceptions=True)
        await host.close()
        await loop.aclose()

    assert live_row is not None, "no emitted snapshot carried the live telemetry"
    assert live_row.availability == RunStatus.RUNNING.value
    assert live_row.effective_model == "alpha/alpha-wire"
    assert live_row.base_model == "alpha-model"
    assert live_row.active_provider == "alpha"
    assert live_row.context_tokens == 77
    assert live_row.context_window == 32_000
    assert live_row.effective_thinking == "high"
    assert live_row.turns_used == 1


# ---------------------------------------------------------------------------
# G6a: a mutation that races a dying emission schedules its own retry
# ---------------------------------------------------------------------------

_RETRY_TIMEOUT = 3.0
"""Bounded patience for one scripted delivery.

Every write here is an in-memory hop (a loop turn plus the retry's deferred
start), so the delivery lands at once; the bound only has to keep the pre-G6a
base -- which never retries -- from hanging the pin.
"""


class _ScriptedRootPipe:
    """The root pipe the G6a pins drive: park a write, break it, or record it.

    ``park`` names the write attempts (1-based, in the order the server makes
    them) that hold inside ``send`` -- the delivery in flight -- until the pin
    breaks the pipe, which then raises the broken-pipe error a closed
    coordinator socket produces. Every other attempt lands on the recorded
    stream, where the pin reads the snapshot the server actually delivered.
    """

    def __init__(self, *, park: Iterable[int] = ()) -> None:
        self.attempts = 0
        self.frames: list[dict[str, Any]] = []
        self._park = set(park)
        self._parked: asyncio.Queue[int] = asyncio.Queue()
        self._delivered: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._gates: dict[int, asyncio.Event] = {}

    async def send(self, message: dict[str, Any]) -> None:
        self.attempts += 1
        attempt = self.attempts
        if attempt in self._park:
            gate = asyncio.Event()
            self._gates[attempt] = gate
            self._parked.put_nowait(attempt)
            await gate.wait()
            raise BrokenPipeError("the coordinator closed the root pipe")
        self.frames.append(message)
        self._delivered.put_nowait(message)

    async def wait_parked(self, attempt: int) -> None:
        """Wait until the server is parked inside the named write attempt."""
        while True:
            seen = await asyncio.wait_for(self._parked.get(), timeout=_RETRY_TIMEOUT)
            if seen == attempt:
                return

    def break_pipe(self, attempt: int) -> None:
        """Fail the parked write attempt the way a closed pipe does."""
        self._gates[attempt].set()

    async def next_frame(self) -> AgentsInventoryReadResponse:
        """Drain the delivered stream until one agents/update lands."""
        message = await asyncio.wait_for(self._delivered.get(), timeout=_RETRY_TIMEOUT)
        assert message.get("method") == AGENT_AGENTS_UPDATE_METHOD
        assert "id" not in message
        return AgentsInventoryReadResponse.model_validate(message["params"])

    def messages(self) -> AsyncIterator[dict[str, Any]]:
        return self._never()

    async def _never(self) -> AsyncIterator[dict[str, Any]]:
        while True:
            await asyncio.sleep(3600.0)
            yield {}

    async def close(self) -> None:
        return None


async def _close_root_stack(server: RootAgentServer, host: Any, loop: Any) -> None:
    """Cancel the server's passes, reap the host's children, close the loop."""
    for task in list(server._tasks):
        task.cancel()
    for task in list(server._tasks):
        try:
            await task
        except BaseException:
            pass
    try:
        await host.close()
    except BaseException:
        pass
    await loop.aclose()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_a_mutation_racing_a_failed_delivery_gets_its_own_pass(
    tmp_path: Path,
) -> None:
    """G6a: a failed delivery still delivers the snapshot a mutation raced it with.

    The first write parks inside ``send`` -- the delivery in flight. A second
    child is published while it is parked, so that mutation only re-arms the
    dirty flag (the single-in-flight guard), and then the pipe breaks. The
    pass must reschedule itself instead of waiting for some later mutation:
    the fresh full snapshot naming BOTH children lands with no further
    mutation. On the pre-G6a base the failure only re-arms the flag, so this
    pin times out.
    """
    loop = _process_root(tmp_path)
    args = _task_args()
    host = AgentChildHost(loop, spawn=_ScriptingSpawnAdapter(_BlockingSource()))
    pipe = _ScriptedRootPipe(park={1})
    server = RootAgentServer(pipe, factory=cast(Any, SimpleNamespace()))
    server.install_child_host(host)
    try:
        first = await host.publish(
            args, _ctx(loop, "task-1"), attempt_id="attempt-raced-1", retained=True
        )
        await pipe.wait_parked(1)
        # The mutation that races the delivery: it only marks the state dirty,
        # so no second write may start while the first is still in flight.
        second = await host.publish(
            args, _ctx(loop, "task-2"), attempt_id="attempt-raced-2", retained=True
        )
        assert pipe.attempts == 1
        assert server._agents_update_dirty is True
        pipe.break_pipe(1)

        update = await pipe.next_frame()
        rows = {agent.agent_id: agent for agent in update.agents}
        assert set(rows) == {first.agent_id, second.agent_id}
        assert update.revision >= 2
    finally:
        await _close_root_stack(server, host, loop)


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_a_persistently_broken_pipe_stops_retrying(tmp_path: Path) -> None:
    """G6a: the raced retry chain is bounded, and a later mutation recovers.

    Every write the pass attempts parks and then fails, and the pin keeps a
    mutation racing each parked delivery -- the arrangement that buys one
    retry after another. The chain stops after ``_AGENTS_UPDATE_RETRY_LIMIT``
    retries, so the server makes three write attempts in total and no fourth,
    even though the dirty flag is still armed. A later mutation against a
    healed pipe then delivers the coalesced fresh snapshot, so the bound costs
    no update.
    """
    # The bound is the server's own budget; importing it here keeps this
    # module importable against the pre-G6a base, where the pin then fails on
    # behaviour (one attempt, no retry) rather than on collection.
    from chartreux.app_server.root_agent import _AGENTS_UPDATE_RETRY_LIMIT

    loop = _process_root(tmp_path)
    args = _task_args()
    host = AgentChildHost(loop, spawn=_ScriptingSpawnAdapter(_BlockingSource()))
    pipe = _ScriptedRootPipe(park={1, 2, 3})
    server = RootAgentServer(pipe, factory=cast(Any, SimpleNamespace()))
    server.install_child_host(host)
    try:
        await host.publish(
            args, _ctx(loop, "task-1"), attempt_id="attempt-bound-1", retained=True
        )
        await pipe.wait_parked(1)
        await host.publish(
            args, _ctx(loop, "task-2"), attempt_id="attempt-bound-2", retained=True
        )
        pipe.break_pipe(1)
        await pipe.wait_parked(2)
        await host.publish(
            args, _ctx(loop, "task-3"), attempt_id="attempt-bound-3", retained=True
        )
        pipe.break_pipe(2)
        await pipe.wait_parked(3)
        await host.publish(
            args, _ctx(loop, "task-4"), attempt_id="attempt-bound-4", retained=True
        )
        pipe.break_pipe(3)

        await asyncio.sleep(0.05)
        assert pipe.attempts == 1 + _AGENTS_UPDATE_RETRY_LIMIT
        assert pipe.frames == []

        last = await host.publish(
            args, _ctx(loop, "task-5"), attempt_id="attempt-bound-5", retained=True
        )
        update = await pipe.next_frame()
        assert last.agent_id in {agent.agent_id for agent in update.agents}
    finally:
        await _close_root_stack(server, host, loop)
