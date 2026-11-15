from __future__ import annotations

from pathlib import Path

import pytest

from chartreux.acp.agent import ChartreuxAcpAgent
from chartreux.acp.exceptions import InvalidRequestError
from chartreux.app_server.local import LocalHarnessOptions
from chartreux.app_server.protocol import SessionOptions
from chartreux.app_server.session import AppServerSession
from chartreux.app_server.transport import IN_PROCESS_TRANSPORT, STDIO_TRANSPORT
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.checkpoints import FileSnapshot, FileState
from tests.conftest import build_test_agent_loop
from tests.stubs.app_server import start_test_app_server
from tests.stubs.fake_client import FakeClient

ACP_TRANSPORTS = (IN_PROCESS_TRANSPORT, STDIO_TRANSPORT)


@pytest.mark.asyncio
async def test_review_extensions_delegate_to_app_server(tmp_path: Path) -> None:
    loops: list[AgentLoop] = []

    async def start_session(options: LocalHarnessOptions) -> AppServerSession:
        loop = build_test_agent_loop()
        loops.append(loop)
        return await AppServerSession.start(
            start_test_app_server(loop),
            client_info=options.client.info,
            capabilities=options.client.capabilities,
            session_options=options.session_options,
        )

    agent = ChartreuxAcpAgent(session_starter=start_session)
    client = FakeClient()
    agent.on_connect(client)
    client.on_connect(agent)
    response = await agent.new_session(cwd=str(tmp_path), mcp_servers=[])
    session_id = response.session_id
    loop = loops[0]
    path = tmp_path / "review.txt"
    path.write_text("before\n", encoding="utf-8")
    turn_id = len(loop.messages)
    loop.checkpoint_recorder.create_checkpoint()
    loop.checkpoint_recorder.add_snapshot(
        FileSnapshot(path=str(path.resolve()), state=FileState(path.read_bytes()))
    )
    path.write_text("after\n", encoding="utf-8")
    loop.checkpoint_recorder.seal_turn()
    owner = {"kind": "agent", "turnId": turn_id}

    try:
        state = await agent.ext_method("review/state", {"sessionId": session_id})
        baseline = await agent.ext_method(
            "review/baseline", {"sessionId": session_id, "path": str(path)}
        )
        turn_diff = await agent.ext_method(
            "review/turnDiff",
            {"sessionId": session_id, "path": str(path), "owner": owner},
        )
        hunks = await agent.ext_method(
            "review/hunks", {"sessionId": session_id, "path": str(path), "owner": owner}
        )
    finally:
        await agent.close()

    assert state["files"][0]["regions"][0]["owner"] == owner
    assert state["scopes"][0]["owner"] == owner
    assert baseline == {"content": "before\n"}
    assert turn_diff == {
        "status": "modified",
        "baseline": "before\n",
        "current": "after\n",
    }
    assert hunks == {
        "hunks": [
            {
                "side": "additions",
                "line": 0,
                "regions": [
                    {
                        "versionIndex": state["files"][0]["regions"][0]["versionIndex"],
                        "ordinal": state["files"][0]["regions"][0]["ordinal"],
                    }
                ],
            }
        ]
    }


@pytest.mark.parametrize(
    ("method", "expected_content"),
    [("review/approve", "after\n"), ("review/revert", "before\n")],
)
@pytest.mark.asyncio
async def test_review_mutation_extensions_delegate_to_app_server(
    tmp_path: Path, method: str, expected_content: str
) -> None:
    loops: list[AgentLoop] = []

    async def start_session(options: LocalHarnessOptions) -> AppServerSession:
        loop = build_test_agent_loop()
        loops.append(loop)
        return await AppServerSession.start(
            start_test_app_server(loop),
            client_info=options.client.info,
            capabilities=options.client.capabilities,
            session_options=options.session_options,
        )

    agent = ChartreuxAcpAgent(session_starter=start_session)
    client = FakeClient()
    agent.on_connect(client)
    client.on_connect(agent)
    response = await agent.new_session(cwd=str(tmp_path), mcp_servers=[])
    path = tmp_path / "review.txt"
    path.write_text("before\n", encoding="utf-8")
    loop = loops[0]
    loop.checkpoint_recorder.create_checkpoint()
    loop.checkpoint_recorder.add_snapshot(
        FileSnapshot(path=str(path.resolve()), state=FileState(path.read_bytes()))
    )
    path.write_text("after\n", encoding="utf-8")
    loop.checkpoint_recorder.seal_turn()

    try:
        result = await agent.ext_method(
            method, {"sessionId": response.session_id, "target": {"kind": "all"}}
        )
    finally:
        await agent.close()

    assert result == {}
    assert path.read_text(encoding="utf-8") == expected_content


@pytest.mark.asyncio
async def test_review_mutation_rejects_invalid_target(tmp_path: Path) -> None:
    agent = ChartreuxAcpAgent()
    client = FakeClient()
    agent.on_connect(client)
    client.on_connect(agent)
    response = await agent.new_session(cwd=str(tmp_path), mcp_servers=[])

    try:
        with pytest.raises(InvalidRequestError):
            await agent.ext_method(
                "review/approve",
                {"sessionId": response.session_id, "target": {"kind": "bogus"}},
            )
    finally:
        await agent.close()


# -- both-transport pins ------------------------------------------------------
#
# Review state is session-local, memory-only state of the loop's checkpoint
# recorder, so it exists only where that loop runs. These nodes run on both
# transports and assert the same projected answer, and the mapping pin asserts
# the ACP call reaches the review method it names. The stdio leg is the
# production topology over a memory wire; the spawned-child legs land with the
# PTY harness (C5c).


def _record_checkpoint(loop: AgentLoop, path: Path, after: str) -> dict[str, object]:
    """Record one turn's edit of ``path`` and return its owner reference.

    The pre-state is snapshotted into the open turn, the file is then written,
    and sealing reads the post-state from disk -- the order the loop's own
    recorder runs in, so the edit is attributed to the turn rather than to a
    between-turn manual edit.
    """
    turn_id = len(loop.messages)
    loop.checkpoint_recorder.create_checkpoint()
    loop.checkpoint_recorder.add_snapshot(
        FileSnapshot(path=str(path.resolve()), state=FileState(path.read_bytes()))
    )
    path.write_text(after, encoding="utf-8")
    loop.checkpoint_recorder.seal_turn()
    return {"kind": "agent", "turnId": turn_id}


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ACP_TRANSPORTS, ids=ACP_TRANSPORTS)
async def test_review_reads_answer_on_both_transports(
    transport: str, acp_sessions, tmp_path: Path
) -> None:
    harness = await acp_sessions(
        transport, session_options=SessionOptions(cwd=str(tmp_path))
    )
    response = await harness.agent.new_session(cwd=str(tmp_path), mcp_servers=[])
    session_id = response.session_id
    path = tmp_path / "review.txt"
    path.write_text("before\n", encoding="utf-8")
    owner = _record_checkpoint(harness.loop, path, "after\n")

    state = await harness.agent.ext_method("review/state", {"sessionId": session_id})
    baseline = await harness.agent.ext_method(
        "review/baseline", {"sessionId": session_id, "path": str(path)}
    )
    turn_diff = await harness.agent.ext_method(
        "review/turnDiff", {"sessionId": session_id, "path": str(path), "owner": owner}
    )
    hunks = await harness.agent.ext_method(
        "review/hunks", {"sessionId": session_id, "path": str(path), "owner": owner}
    )

    assert state["files"][0]["regions"][0]["owner"] == owner
    assert state["scopes"][0]["owner"] == owner
    assert baseline == {"content": "before\n"}
    assert turn_diff == {
        "status": "modified",
        "baseline": "before\n",
        "current": "after\n",
    }
    assert hunks == {
        "hunks": [
            {
                "side": "additions",
                "line": 0,
                "regions": [
                    {
                        "versionIndex": state["files"][0]["regions"][0]["versionIndex"],
                        "ordinal": state["files"][0]["regions"][0]["ordinal"],
                    }
                ],
            }
        ]
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ACP_TRANSPORTS, ids=ACP_TRANSPORTS)
@pytest.mark.parametrize(
    ("method", "expected_content"),
    [("review/approve", "after\n"), ("review/revert", "before\n")],
)
async def test_review_mutations_apply_on_both_transports(
    transport: str, method: str, expected_content: str, acp_sessions, tmp_path: Path
) -> None:
    harness = await acp_sessions(
        transport, session_options=SessionOptions(cwd=str(tmp_path))
    )
    response = await harness.agent.new_session(cwd=str(tmp_path), mcp_servers=[])
    path = tmp_path / "review.txt"
    path.write_text("before\n", encoding="utf-8")
    _record_checkpoint(harness.loop, path, "after\n")

    result = await harness.agent.ext_method(
        method, {"sessionId": response.session_id, "target": {"kind": "all"}}
    )

    assert result == {}
    assert path.read_text(encoding="utf-8") == expected_content


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ACP_TRANSPORTS, ids=ACP_TRANSPORTS)
async def test_review_state_crosses_as_its_own_method_on_both_transports(
    transport: str, acp_sessions, acp_request_recorder, tmp_path: Path
) -> None:
    """The ACP review call reaches the review method it names, not a sibling."""
    harness = await acp_sessions(
        transport, session_options=SessionOptions(cwd=str(tmp_path))
    )
    response = await harness.agent.new_session(cwd=str(tmp_path), mcp_servers=[])
    recorded = acp_request_recorder(harness.server)

    await harness.agent.ext_method("review/state", {"sessionId": response.session_id})

    assert recorded.issued("review/state") == 1
