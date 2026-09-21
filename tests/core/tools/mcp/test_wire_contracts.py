from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from chartreux.core.events import ToolResultEvent
from chartreux.core.tools.base import BaseToolConfig, BaseToolState, InvokeContext
from chartreux.core.tools.mcp import tools
from chartreux.core.tools.mcp.pool import MCPConnectionPool
from chartreux.core.tools.remote import MCPTool, MCPToolResult, RemoteTool, _OpenArgs


@pytest.mark.parametrize("status", [True, False, None])
@pytest.mark.parametrize("shape", [dict, SimpleNamespace])
def test_error_status_preserves_model_content_and_ui_failure(status, shape):
    payload = {
        "content": [{"type": "text", "text": "remote details"}],
        "futureField": 1,
    }
    if status is not None:
        payload["isError"] = status
    result = tools._parse_call_result("server", "remote", shape(**payload))
    assert result.ok is (status is not True)
    assert result.text == "remote details"
    for cls in (
        tools.create_mcp_http_proxy_tool_class(
            url="https://mcp.invalid", remote=RemoteTool(name="remote")
        ),
        tools.create_mcp_stdio_proxy_tool_class(
            command=["fake"], remote=RemoteTool(name="remote")
        ),
    ):
        assert issubclass(cls, MCPTool)
        display = cls.get_result_display(
            ToolResultEvent(
                tool_name=cls.get_name(),
                tool_class=cls,
                tool_call_id="test",
                result=result,
            )
        )
        assert display.success is (status is not True)


def test_text_and_structured_content_are_both_preserved():
    result = tools._parse_call_result(
        "s",
        "t",
        {
            "content": [{"type": "text", "text": "explanation"}],
            "structuredContent": {"count": 2},
        },
    )
    assert result.text == "explanation"
    assert result.structured == {"count": 2}


@pytest.mark.parametrize(
    "kind", ["image", "audio", "resource", "resource_link", "future_block"]
)
def test_unsupported_content_has_type_only_omission_notice(kind):
    result = tools._parse_call_result(
        "s",
        "t",
        {
            "content": [
                {
                    "type": kind,
                    "data": "synthetic-private-payload",
                    "uri": "private://value",
                }
            ]
        },
    )
    assert result.text is not None
    assert kind in result.text
    assert "omit" in result.text.lower() or "unsupported" in result.text.lower()
    assert "synthetic-private-payload" not in result.text
    assert "private://value" not in result.text


def test_empty_result_is_tolerated():
    result = tools._parse_call_result("s", "t", {})
    assert result.ok
    assert result.text is None
    assert result.structured is None


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["http", "stdio", "pooled-stdio"])
@pytest.mark.parametrize("payload", [{}, {"nullable": None}, {"nullable": "value"}])
async def test_proxy_preserves_explicit_null_but_not_omitted_arguments(
    monkeypatch, transport, payload
):
    remote = RemoteTool(name="remote")
    call = AsyncMock(return_value=MCPToolResult(server="s", tool="remote"))
    if transport == "http":
        cls = tools.create_mcp_http_proxy_tool_class(
            url="https://mcp.invalid", remote=remote
        )
        monkeypatch.setattr(tools, "call_tool_http", call)
    else:
        cls = tools.create_mcp_stdio_proxy_tool_class(command=["fake"], remote=remote)
        monkeypatch.setattr(tools, "call_tool_stdio", call)
    ctx = InvokeContext(tool_call_id="test")
    if transport == "pooled-stdio":
        pool = MCPConnectionPool()
        monkeypatch.setattr(pool, "call_tool", call)
        ctx = InvokeContext(tool_call_id="test", mcp_pool=pool)
    tool = cls(lambda: BaseToolConfig(), BaseToolState())
    results = [result async for result in tool.run(_OpenArgs(**payload), ctx)]
    assert len(results) == 1
    actual = (
        call.call_args.kwargs["arguments"]
        if transport == "pooled-stdio"
        else call.call_args.args[2]
    )
    assert actual == payload


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["discovery", "execution"])
@pytest.mark.parametrize(
    "location",
    ["https://other.invalid/mcp", "https://mcp.invalid/next", "http://mcp.invalid/mcp"],
)
async def test_redirect_never_forwards_custom_secrets(monkeypatch, operation, location):
    requests = []

    def respond(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(307, headers={"Location": location})
        return httpx.Response(403)

    def client(**kwargs):
        kwargs.pop("verify")
        return httpx.AsyncClient(transport=httpx.MockTransport(respond), **kwargs)

    @asynccontextmanager
    async def stream(url, *, http_client):
        response = await http_client.post(url, json={"method": "initialize"})
        response.raise_for_status()
        yield None, None, None

    monkeypatch.setattr(tools, "ChartreuxAsyncHTTPClient", client)
    monkeypatch.setattr(tools, "build_ssl_context", lambda: True)
    monkeypatch.setattr(tools, "streamable_http_client", stream)
    headers = {"X-Tenant-Secret": "synthetic-only", "Authorization": "Bearer synthetic"}
    with pytest.raises(httpx.HTTPStatusError):
        if operation == "discovery":
            await tools.list_tools_http("https://mcp.invalid/mcp", headers=headers)
        else:
            await tools.call_tool_http(
                "https://mcp.invalid/mcp", "mutate", {}, headers=headers
            )
    assert requests[0].headers["X-Tenant-Secret"] == "synthetic-only"
    assert requests[0].headers["Authorization"] == "Bearer synthetic"
    assert len(requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["http", "stdio"])
@pytest.mark.parametrize("repeated", [False, True])
async def test_discovery_follows_pages_and_bounds_repeated_cursor(
    monkeypatch, transport, repeated
):
    cursors = []

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def initialize(self):
            pass

        async def list_tools(self, cursor=None, **kwargs):
            cursors.append(cursor)
            # Yield so the outer deadline can interrupt a broken unbounded loop.
            await asyncio.sleep(0)
            assert len(cursors) <= 4, "repeated cursor was not bounded"
            return SimpleNamespace(
                tools=[
                    {"name": "first" if cursor is None else "second", "futureField": 1}
                ],
                nextCursor="next" if cursor is None or repeated else None,
            )

    session = Session()

    @asynccontextmanager
    async def client(*args, **kwargs):
        yield object()

    @asynccontextmanager
    async def stream(*args, **kwargs):
        yield None, None, None

    monkeypatch.setattr(tools, "create_vibe_mcp_http_client", client)
    monkeypatch.setattr(tools, "streamable_http_client", stream)
    monkeypatch.setattr(tools, "ClientSession", lambda *args, **kwargs: session)
    monkeypatch.setattr(tools, "enter_stdio_session", AsyncMock(return_value=session))

    async def discover():
        if transport == "http":
            return await tools.list_tools_http(
                "https://mcp.invalid", startup_timeout_sec=1
            )
        return await tools.list_tools_stdio(["fake"], startup_timeout_sec=1)

    async with asyncio.timeout(2):
        if repeated:
            with pytest.raises((ValueError, RuntimeError), match="(?i)cursor|paginat"):
                await discover()
            assert cursors == [None, "next"]
        else:
            result = await discover()
            assert [tool.name for tool in result] == ["first", "second"]
            assert cursors == [None, "next"]


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["http", "stdio"])
async def test_discovery_uses_one_total_deadline_across_initialization_and_pages(
    monkeypatch, transport
):
    real_timeout = asyncio.timeout
    deadlines, budgets, phases = [], [], []

    def timeout(delay):
        budgets.append(delay)
        context = real_timeout(delay)
        deadlines.append(context)
        return context

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            phases.append("closed")

        async def initialize(self):
            phases.append("initialized")
            assert len(deadlines) == 1

        async def list_tools(self, cursor=None):
            phases.append(cursor)
            assert len(deadlines) == 1
            if cursor is not None:
                # Expire the original budget deterministically, without sleeps.
                deadlines[0].reschedule(asyncio.get_running_loop().time() - 1)
                await asyncio.Event().wait()
            return SimpleNamespace(tools=[{"name": "first"}], nextCursor="next")

    session = Session()

    @asynccontextmanager
    async def client(*args, **kwargs):
        yield object()

    @asynccontextmanager
    async def stream(*args, **kwargs):
        yield None, None, None

    async def enter(stack, *args, **kwargs):
        await stack.enter_async_context(session)
        await session.initialize()
        return session

    monkeypatch.setattr(tools.asyncio, "timeout", timeout)
    monkeypatch.setattr(tools, "create_vibe_mcp_http_client", client)
    monkeypatch.setattr(tools, "streamable_http_client", stream)
    monkeypatch.setattr(tools, "ClientSession", lambda *args, **kwargs: session)
    monkeypatch.setattr(tools, "enter_stdio_session", enter)
    with pytest.raises(TimeoutError):
        if transport == "http":
            await tools.list_tools_http("https://mcp.invalid", startup_timeout_sec=10)
        else:
            await tools.list_tools_stdio(["fake"], startup_timeout_sec=10)
    assert budgets == [10]
    assert phases == ["initialized", None, "next", "closed"]
