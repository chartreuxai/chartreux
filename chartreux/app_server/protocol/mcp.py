"""MCP catalog read/refresh/toggle/add/remove and auth wire models."""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from chartreux.app_server._model import ProtocolModel
from chartreux.app_server.models import MCPState
from chartreux.app_server.protocol.runtime import RuntimeSnapshot
from chartreux.utils.mcp import MCPAddTransport


class MCPReadParams(ProtocolModel):
    session_id: str


class MCPReadResponse(ProtocolModel):
    mcp: MCPState


class MCPRefreshParams(ProtocolModel):
    session_id: str


class MCPToggleParams(ProtocolModel):
    session_id: str | None = None
    name: str
    disabled: bool
    tool_name: str | None = None


class MCPAddParams(ProtocolModel):
    session_id: str | None = None
    url: str
    name: str | None = None
    scopes: list[str] = Field(default_factory=list)
    transport: MCPAddTransport = "streamable-http"


class MCPAddResponse(ProtocolModel):
    name: str
    url: str
    created: bool
    runtime: RuntimeSnapshot | None = None


class MCPCatalogMutationResponse(ProtocolModel):
    runtime: RuntimeSnapshot | None = None


class MCPRemoveParams(ProtocolModel):
    session_id: str | None = None
    name: str


class MCPRemoveResponse(ProtocolModel):
    name: str
    removed: bool
    runtime: RuntimeSnapshot | None = None


class MCPLogoutParams(ProtocolModel):
    session_id: str | None = None
    name: str


class MCPLoginParams(ProtocolModel):
    session_id: str | None = None
    name: str


class MCPAuthUrlParams(ProtocolModel):
    NOTIFICATION_METHOD: ClassVar[str] = "mcp_catalog/authUrl"

    name: str
    url: str


class MCPAuthRequiredParams(ProtocolModel):
    NOTIFICATION_METHOD: ClassVar[str] = "mcp_catalog/authRequired"

    session_id: str
    name: str
    descriptor_revision: str
    observed_connection_revision: str | None = None
