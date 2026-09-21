from __future__ import annotations

from pathlib import Path

import pytest

from chartreux.acp.agent import ChartreuxAcpAgent
from chartreux.acp.exceptions import (
    InvalidRequestError,
    NotImplementedMethodError,
    SessionNotFoundError,
)
from chartreux.app_server.models import IdentityEntityView, IdentityView
from chartreux.app_server.protocol import (
    AppServerResponseError,
    ProtocolError,
    ProtocolErrorCode,
)


async def _new_session(agent: ChartreuxAcpAgent) -> str:
    return (await agent.new_session(cwd=str(Path.cwd()), mcp_servers=[])).session_id


@pytest.mark.asyncio
async def test_identity_read_routes_through_session_identity_resource(
    acp_agent_loop: ChartreuxAcpAgent, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_id = await _new_session(acp_agent_loop)
    identity = IdentityView(
        id="user-1",
        email="user@example.test",
        first_name="Test",
        workspace=IdentityEntityView(id="workspace-1", name="Workspace"),
    )

    async def read() -> IdentityView:
        return identity

    monkeypatch.setattr(
        acp_agent_loop.sessions[session_id].app_server.resources.identity, "read", read
    )

    assert await acp_agent_loop.ext_method(
        "identity/read", {"sessionId": session_id}
    ) == {
        "id": "user-1",
        "email": "user@example.test",
        "firstName": "Test",
        "lastName": None,
        "workspace": {"id": "workspace-1", "name": "Workspace"},
        "organization": None,
    }


@pytest.mark.asyncio
async def test_identity_read_handles_missing_invalid_and_absent_identity(
    acp_agent_loop: ChartreuxAcpAgent, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(InvalidRequestError, match="identity request"):
        await acp_agent_loop.ext_method("identity/read", {})
    with pytest.raises(SessionNotFoundError):
        await acp_agent_loop.ext_method("identity/read", {"sessionId": "missing"})

    session_id = await _new_session(acp_agent_loop)

    async def read() -> None:
        return None

    monkeypatch.setattr(
        acp_agent_loop.sessions[session_id].app_server.resources.identity, "read", read
    )
    assert (
        await acp_agent_loop.ext_method("identity/read", {"sessionId": session_id})
        == {}
    )


@pytest.mark.asyncio
async def test_identity_read_maps_resource_errors_and_rejects_unsupported_methods(
    acp_agent_loop: ChartreuxAcpAgent, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_id = await _new_session(acp_agent_loop)

    async def read() -> None:
        raise AppServerResponseError(
            ProtocolError(
                code=ProtocolErrorCode.INVALID_PARAMS, message="identity failed"
            )
        )

    monkeypatch.setattr(
        acp_agent_loop.sessions[session_id].app_server.resources.identity, "read", read
    )
    with pytest.raises(InvalidRequestError, match="identity failed"):
        await acp_agent_loop.ext_method("identity/read", {"sessionId": session_id})
    with pytest.raises(NotImplementedMethodError):
        await acp_agent_loop.ext_method("identity/write", {"sessionId": session_id})
