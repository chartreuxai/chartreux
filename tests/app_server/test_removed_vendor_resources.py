"""Removed vendor resources must not remain advertised or callable."""

from __future__ import annotations

import pytest

from chartreux.acp import models as acp_models
from chartreux.app_server import models, protocol
from chartreux.app_server.protocol import (
    SERVER_METHODS,
    AppServerResponseError,
    ClientInfo,
    ProtocolErrorCode,
)
from tests.conftest import build_test_agent_loop
from tests.stubs.app_server import attach_test_app_server_session, start_test_app_server

PLUGIN_METHODS = ("plugin/info", "plugin/reload", "plugin_catalog/read")

PROJECT_LINKS_METHODS = (
    "projectLinks/create",
    "projectLinks/inspectRoot",
    "projectLinks/link",
    "projectLinks/list",
    "projectLinks/picker/load",
    "projectLinks/picker/loadMore",
    "projectLinks/resolveRoot",
    "projectLinks/save",
    "projectLinks/unlink",
)


@pytest.mark.parametrize(
    "method", [*PROJECT_LINKS_METHODS, *PLUGIN_METHODS, "unknown/method"]
)
def test_unknown_methods_are_not_advertised(method: str) -> None:
    assert method not in SERVER_METHODS


def test_project_links_contract_is_absent_but_config_schema_is_retained() -> None:
    assert not any(name.startswith("ProjectLink") for name in vars(protocol))
    assert not any(name.startswith("ProjectLinks") for name in vars(acp_models))
    assert hasattr(acp_models, "ConfigSchemaResponse")


def test_account_contract_is_absent_but_identity_is_retained() -> None:
    assert "account/read" not in SERVER_METHODS
    assert "identity/read" in SERVER_METHODS
    assert not any(name.startswith("Account") for name in vars(models))
    assert not any(name.startswith("Account") for name in vars(protocol))
    identity = models.IdentityView(id="provider-user", first_name="Local")
    assert identity.name == "Local"
    assert protocol.IdentityReadResponse(identity=identity).identity == identity


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method", [*PROJECT_LINKS_METHODS, *PLUGIN_METHODS, "unknown/method"]
)
async def test_unknown_methods_are_rejected_before_attachment(method: str) -> None:
    client = start_test_app_server(build_test_agent_loop())
    try:
        await client.initialize(ClientInfo(name="removed-resource-test", version="0"))
        await client.notify("initialized")
        with pytest.raises(AppServerResponseError) as excinfo:
            await client.request(method, {})
        assert excinfo.value.error.code is ProtocolErrorCode.METHOD_NOT_FOUND
    finally:
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method", [*PROJECT_LINKS_METHODS, *PLUGIN_METHODS, "unknown/method"]
)
async def test_unknown_methods_are_rejected_after_attachment(method: str) -> None:
    client = start_test_app_server(build_test_agent_loop())
    session = await attach_test_app_server_session(client)
    try:
        with pytest.raises(AppServerResponseError) as excinfo:
            await client.request(method, {"sessionId": session.session_id})
        assert excinfo.value.error.code is ProtocolErrorCode.METHOD_NOT_FOUND
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_removed_account_request_before_attachment_is_unknown() -> None:
    client = start_test_app_server(build_test_agent_loop())
    try:
        await client.initialize(ClientInfo(name="removed-resource-test", version="0"))
        await client.notify("initialized")
        with pytest.raises(AppServerResponseError) as excinfo:
            await client.request("account/read", {"sessionId": "not-attached"})
        assert excinfo.value.error.code is ProtocolErrorCode.METHOD_NOT_FOUND
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_removed_account_request_does_not_break_attached_resources() -> None:
    client = start_test_app_server(build_test_agent_loop())
    session = await attach_test_app_server_session(client)
    try:
        assert not hasattr(session.resources, "account")
        assert session.resources.identity.current is None
        before = session.resources.config.current
        with pytest.raises(AppServerResponseError) as excinfo:
            await client.request("account/read", {"sessionId": session.session_id})
        assert excinfo.value.error.code is ProtocolErrorCode.METHOD_NOT_FOUND
        await session.resources.runtime.refresh()
        assert session.resources.config.current == before
    finally:
        await session.close()
