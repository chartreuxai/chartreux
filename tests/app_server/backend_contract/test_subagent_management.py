"""Backend-contract wire tests for the subagent-management surface.

``agents/{cancel,release,resume,reconcile}`` and ``agent/transcript/get`` are
carried to the process that owns the children. On the stdio route that owner is
the root agent's child host, so the nodes here run on both transports: the
unknown-agent node pins the identical typed outcome set, and the owner-state
node exercises the process-backed handles (a retained completed child) that the
in-process registry represents with its own records.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
import pytest_asyncio

from chartreux.app_server.protocol import (
    AgentTranscriptGetResponse,
    AgentTranscriptState,
    AppServerResponseError,
    CancelOutcome,
    ClientCapabilities,
    ProtocolErrorCode,
    ReconcileOutcome,
    ReleaseOutcome,
    ResumeOutcome,
    SessionOptions,
)
from chartreux.app_server.root_agent import AgentChildHandle, AgentChildHost
from chartreux.core.subagents import TaskResult
from tests.app_server.backend_contract.conftest import (
    BackendContractConnection,
    connect_backend_contract_stdio_host,
    install_memory_root_agent,
)
from tests.conftest import build_test_agent_loop


class _FakeWireClient:
    """A live child's own page reader, answering one empty saved-empty page."""

    async def read_page(
        self, agent_id: str, *, before: str | None = None, limit: int = 50
    ) -> AgentTranscriptGetResponse:
        return AgentTranscriptGetResponse(
            state=AgentTranscriptState.AVAILABLE, entries=[], has_more=False
        )


class _FakeSpawned:
    """A spawned child handle stand-in: live, with a page reader and no dir."""

    alive = True
    child_session_dir: str | None = None
    root_session_dir: str | None = None

    def __init__(self) -> None:
        self.wire_client = _FakeWireClient()

    async def close(self) -> None:
        self.alive = False


class _FakeSpawnPort:
    """A spawn adapter the injected handles never draw from."""

    budget: Any = None

    async def spawn(self, spec: Any) -> Any:
        raise AssertionError("the injected handles are published directly")


def _retained_child(host: AgentChildHost) -> AgentChildHandle:
    handle = AgentChildHandle(
        agent_id="child",
        identity="coder",
        attempt_id="attempt-1",
        spawned=_FakeSpawned(),
        session_id="child-session",
        child_dir=None,
        retained=True,
        slot=object(),
    )
    handle.run_id = "run-1"
    handle.result = TaskResult(
        response="done", turns_used=1, completed=True, agent_id="child", run_id="run-1"
    )
    host._children["child"] = handle
    return handle


@pytest_asyncio.fixture
async def stdio_child_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[tuple[BackendContractConnection, Any, AgentChildHost]]:
    """A stdio session whose root agent owns one retained completed child."""
    loop = build_test_agent_loop()
    child_host = AgentChildHost(loop, spawn=_FakeSpawnPort())
    install_memory_root_agent(monkeypatch, loop, child_host=child_host)
    connection = await connect_backend_contract_stdio_host(
        session_options=SessionOptions(), capabilities=ClientCapabilities()
    )
    session = await connection.host.open_session()
    try:
        yield connection, session, child_host
    finally:
        await session.close()
        await connection.host.close()
        await loop.aclose()


@pytest.mark.asyncio
async def test_unknown_agent_answers_the_same_typed_outcomes_on_the_selected_transport(
    backend_contract_transport_session: Any,
) -> None:
    """Every management route answers its typed disposition for an absent agent.

    The client reads one contract whichever owner answered: cancel is an
    unknown run, release an unknown handle, resume not-resumable, reconcile
    not-pending, and a transcript read is a not-found refusal — never a false
    success and never a silent empty page.
    """
    session = backend_contract_transport_session
    cancel = await session.cancel_agent("missing-agent", "run-1")
    assert cancel.outcome is CancelOutcome.UNKNOWN_RUN
    assert cancel.run_id == "run-1"

    release = await session.release_agent("missing-agent", "run-1")
    assert release.outcome is ReleaseOutcome.UNKNOWN

    resume = await session.resume_agent("missing-agent", "run-1")
    assert resume.outcome is ResumeOutcome.NOT_RESUMABLE

    reconcile = await session.reconcile_result("missing-agent", "run-1", "turn-1", [])
    assert reconcile.outcome is ReconcileOutcome.NOT_PENDING

    with pytest.raises(AppServerResponseError) as exc_info:
        client = await session._ensure_attached()
        await client.request("agent/transcript/get", {"agentId": "missing-agent"})
    assert exc_info.value.error.code is ProtocolErrorCode.NOT_FOUND


@pytest.mark.asyncio
async def test_the_stdio_owner_answers_from_the_child_handles_it_spawned(
    stdio_child_owner: tuple[BackendContractConnection, Any, AgentChildHost],
) -> None:
    """The process-backed owner preserves the registry's guards and dispositions.

    A completed retained child is not running (a cancel is already-finishing),
    a stale run pin conflicts instead of releasing, the live page is read
    through the child's own reader, warm resume is typed as not-suspended, no
    reconcile barrier exists, and a release removes the handle so a repeat is
    unknown rather than a second release.
    """
    _connection, session, host = stdio_child_owner
    _retained_child(host)

    cancel = await session.cancel_agent("child", "run-1")
    assert cancel.outcome is CancelOutcome.ALREADY_FINISHING

    stale = await session.cancel_agent("child", "stale-run")
    assert stale.outcome is CancelOutcome.UNKNOWN_RUN

    conflict = await session.release_agent("child", "stale-run")
    assert conflict.outcome is ReleaseOutcome.CONFLICT

    page = await host.read_transcript_page("child", limit=10)
    assert page.state is AgentTranscriptState.AVAILABLE

    resume = await session.resume_agent("child", "run-1")
    assert resume.outcome is ResumeOutcome.NOT_RESUMABLE
    assert "not_suspended" in resume.reason_codes

    reconcile = await session.reconcile_result("child", "run-1", "turn-1", [])
    assert reconcile.outcome is ReconcileOutcome.NOT_PENDING

    released = await session.release_agent("child", "run-1")
    assert released.outcome is ReleaseOutcome.RELEASED
    assert released.reason is not None

    repeated = await session.release_agent("child", "run-1")
    assert repeated.outcome is ReleaseOutcome.UNKNOWN


@pytest.mark.asyncio
async def test_a_dead_child_reads_its_saved_page_instead_of_a_live_route(
    stdio_child_owner: tuple[BackendContractConnection, Any, AgentChildHost],
) -> None:
    """A child that is gone is paged from disk, never from a dead live route.

    The owner holds no live route for the exited process and its transcript
    directory is unavailable here, so the read says so typed rather than
    serving an empty page as if the transcript were intact.
    """
    connection, _session, host = stdio_child_owner
    handle = _retained_child(host)
    handle.spawned.alive = False
    handle.child_dir = None

    response = await connection.client.request(
        "agent/transcript/get", {"agentId": "child"}
    )
    assert response["state"] == AgentTranscriptState.NO_SAVED_TRANSCRIPT.value
