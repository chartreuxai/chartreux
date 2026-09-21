from __future__ import annotations

from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from chartreux.acp.agent import ChartreuxAcpAgent
from chartreux.acp.exceptions import InvalidRequestError, SessionNotFoundError
from chartreux.app_server.local import LocalHarnessOptions
from chartreux.app_server.session import AppServerSession
from chartreux.core.trusted_folders import trusted_folders_manager
from tests.conftest import build_test_agent_loop
from tests.stubs.app_server import start_test_app_server
from tests.stubs.fake_backend import FakeBackend
from tests.stubs.fake_client import FakeClient


@pytest.fixture
def trust_agent(backend: FakeBackend) -> ChartreuxAcpAgent:
    loops: dict[Path, Any] = {}

    async def start_session(options: LocalHarnessOptions) -> AppServerSession:
        cwd = Path(options.session_options.cwd or Path.cwd())
        loop = build_test_agent_loop(backend=backend, cwd=cwd, enable_streaming=True)
        loops[cwd.resolve()] = loop
        return await AppServerSession.start(
            start_test_app_server(loop),
            client_info=options.client.info,
            capabilities=options.client.capabilities,
            session_options=options.session_options,
        )

    agent = ChartreuxAcpAgent(session_starter=start_session)
    cast(Any, agent)._test_loops = loops
    client = FakeClient()
    agent.on_connect(client)
    client.on_connect(agent)
    return agent


async def _new_session(agent: ChartreuxAcpAgent, cwd: Path) -> str:
    return (await agent.new_session(cwd=str(cwd), mcp_servers=[])).session_id


@pytest.mark.asyncio
@pytest.mark.parametrize("decision", ["trust_repo", "trust_cwd", "decline"])
async def test_trust_decision_requires_existing_session(
    trust_agent: ChartreuxAcpAgent, tmp_path: Path, decision: str
) -> None:
    with pytest.raises(InvalidRequestError, match="valid sessionId"):
        await trust_agent.ext_method(
            "trust/decision", {"decision": decision, "cwd": str(tmp_path)}
        )

    assert trusted_folders_manager.is_trusted(tmp_path) is not True
    assert trusted_folders_manager.is_explicitly_untrusted(tmp_path) is not True


@pytest.mark.asyncio
async def test_trust_decision_rejects_unknown_session(
    trust_agent: ChartreuxAcpAgent, tmp_path: Path
) -> None:
    with pytest.raises(SessionNotFoundError):
        await trust_agent.ext_method(
            "trust/decision",
            {"sessionId": "missing", "decision": "trust_cwd", "cwd": str(tmp_path)},
        )


@pytest.mark.asyncio
async def test_trust_decision_rejects_foreign_cwd_without_writing(
    trust_agent: ChartreuxAcpAgent, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_cwd = tmp_path / "session"
    foreign_cwd = tmp_path / "foreign"
    session_cwd.mkdir()
    foreign_cwd.mkdir()
    session_id = await _new_session(trust_agent, session_cwd)
    decide_trust = AsyncMock()
    monkeypatch.setattr(
        trust_agent.sessions[session_id].app_server.resources.workspace,
        "decide_trust",
        decide_trust,
    )

    with pytest.raises(InvalidRequestError, match="session working directory"):
        await trust_agent.ext_method(
            "trust/decision",
            {"sessionId": session_id, "decision": "decline", "cwd": str(foreign_cwd)},
        )

    decide_trust.assert_not_awaited()
    assert trusted_folders_manager.is_explicitly_untrusted(foreign_cwd) is not True


@pytest.mark.asyncio
@pytest.mark.parametrize("session_key", ["sessionId", "session_id"])
async def test_trust_decision_accepts_canonical_session_cwd(
    trust_agent: ChartreuxAcpAgent, tmp_path: Path, session_key: str
) -> None:
    session_cwd = tmp_path / "workspace"
    session_cwd.mkdir()
    (session_cwd / "AGENTS.md").write_text("# local instructions\n")
    alias = session_cwd / ".." / session_cwd.name
    session_id = await _new_session(trust_agent, session_cwd)

    result = await trust_agent.ext_method(
        "trust/decision",
        {session_key: session_id, "decision": "trust_cwd", "cwd": str(alias)},
    )

    assert result == {"trust_status": "trusted", "details": None}
    assert trusted_folders_manager.is_trusted(session_cwd) is True


@pytest.mark.asyncio
async def test_trust_decision_uses_current_cwd_after_session_relocation(
    trust_agent: ChartreuxAcpAgent, tmp_path: Path
) -> None:
    original = tmp_path / "original"
    relocated = original / "relocated"
    original.mkdir()
    relocated.mkdir()
    (relocated / "AGENTS.md").write_text("# relocated instructions\n")
    session_id = await _new_session(trust_agent, original)
    session = trust_agent.sessions[session_id]
    cast(Any, trust_agent)._test_loops[original.resolve()].cwd = relocated.resolve()
    session.app_server.state.session.cwd = str(relocated)

    with pytest.raises(InvalidRequestError, match="session working directory"):
        await trust_agent.ext_method(
            "trust/decision",
            {"sessionId": session_id, "decision": "trust_cwd", "cwd": str(original)},
        )

    result = await trust_agent.ext_method(
        "trust/decision",
        {"sessionId": session_id, "decision": "trust_cwd", "cwd": str(relocated)},
    )
    assert result["trust_status"] == "trusted"
    assert trusted_folders_manager.is_trusted(relocated) is True
    assert trusted_folders_manager.is_trusted(original) is not True


@pytest.mark.asyncio
async def test_trust_decision_is_bound_to_selected_session(
    trust_agent: ChartreuxAcpAgent, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    first_id = await _new_session(trust_agent, first)
    second_id = await _new_session(trust_agent, second)
    decide_first = AsyncMock()
    decide_second = AsyncMock()
    monkeypatch.setattr(
        trust_agent.sessions[first_id].app_server.resources.workspace,
        "decide_trust",
        decide_first,
    )
    monkeypatch.setattr(
        trust_agent.sessions[second_id].app_server.resources.workspace,
        "decide_trust",
        decide_second,
    )

    with pytest.raises(InvalidRequestError, match="session working directory"):
        await trust_agent.ext_method(
            "trust/decision",
            {"sessionId": first_id, "decision": "trust_cwd", "cwd": str(second)},
        )

    decide_first.assert_not_awaited()
    decide_second.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"sessionId": 42, "decision": "trust_cwd"}, "valid sessionId"),
        ({"sessionId": "SESSION", "decision": []}, "Unknown trust decision"),
        ({"sessionId": "SESSION", "decision": "unknown"}, "Unknown trust decision"),
        (
            {"sessionId": "SESSION", "decision": "trust_cwd", "cwd": 42},
            "cwd must be a string",
        ),
        (
            {"sessionId": "SESSION", "decision": "trust_cwd", "cwd": "bad\0cwd"},
            "cwd is invalid",
        ),
    ],
)
async def test_trust_decision_rejects_malformed_arguments(
    trust_agent: ChartreuxAcpAgent,
    tmp_path: Path,
    params: dict[str, object],
    message: str,
) -> None:
    session_id = await _new_session(trust_agent, tmp_path)
    request = {
        key: session_id if value == "SESSION" else value
        for key, value in params.items()
    }

    with pytest.raises(InvalidRequestError, match=message):
        await trust_agent.ext_method("trust/decision", request)


@pytest.mark.asyncio
async def test_trust_status_remains_readable_without_session(
    trust_agent: ChartreuxAcpAgent, tmp_path: Path
) -> None:
    result = await trust_agent.ext_method("trust/status", {"cwd": str(tmp_path)})

    assert result["trust_status"] == "untrusted"
