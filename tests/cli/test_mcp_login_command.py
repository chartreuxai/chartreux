from __future__ import annotations

from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock

import pytest

from chartreux.app_server.models import MCPSourceStatus, MCPSourceSummary, MCPState
from chartreux.app_server.protocol import (
    AppServerResponseError,
    ProtocolError,
    ProtocolErrorCode,
)
from chartreux.cli.textual_ui.widgets.messages import ErrorMessage
from tests.conftest import build_test_chartreux_app, build_test_vibe_config


async def _empty_login(_name: str) -> AsyncGenerator[object, None]:
    return
    yield  # pragma: no cover - makes this an async generator


def _server(name: str) -> MCPSourceSummary:
    return MCPSourceSummary(
        name=name, transport="streamable-http", status=MCPSourceStatus.NEEDS_AUTH
    )


@pytest.mark.asyncio
async def test_mcp_login_server_uses_oauth_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = build_test_chartreux_app(config=build_test_vibe_config())
    await app.prepare()

    monkeypatch.setattr(
        app.app_server.resources.runtime, "wait_until_ready", AsyncMock()
    )
    monkeypatch.setattr(
        app.app_server.resources.mcp,
        "read",
        AsyncMock(return_value=MCPState(sources=[_server("linear")])),
    )

    login_calls: list[str] = []

    def fake_login(name: str) -> AsyncGenerator[object, None]:
        login_calls.append(name)
        return _empty_login(name)

    monkeypatch.setattr(app.app_server.resources.mcp, "login", fake_login)

    switched: list[object] = []

    async def switch_from_input(widget: object, scroll: bool = False) -> None:
        switched.append(widget)

    monkeypatch.setattr(app, "_switch_from_input", switch_from_input)
    monkeypatch.setattr(app, "_mount_and_scroll", AsyncMock())

    await app._mcp_login("linear")

    assert login_calls == ["linear"]
    assert switched == []


@pytest.mark.asyncio
async def test_mcp_login_surfaces_login_error(monkeypatch: pytest.MonkeyPatch) -> None:
    app = build_test_chartreux_app(config=build_test_vibe_config())
    await app.prepare()

    error = AppServerResponseError(
        ProtocolError(code=ProtocolErrorCode.INTERNAL_ERROR, message="transient boom")
    )

    async def failing_login(_name: str) -> AsyncGenerator[object, None]:
        raise error
        yield  # pragma: no cover - makes this an async generator

    monkeypatch.setattr(app.app_server.resources.mcp, "login", failing_login)

    mounted: list[object] = []

    async def mount_and_scroll(widget: object, after: object | None = None) -> None:
        mounted.append(widget)

    monkeypatch.setattr(app, "_mount_and_scroll", mount_and_scroll)

    await app._mcp_login("gmail")

    assert any(
        isinstance(w, ErrorMessage) and "transient boom" in str(w._error)
        for w in mounted
    )
