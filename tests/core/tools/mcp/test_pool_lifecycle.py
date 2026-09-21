from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import anyio
from mcp.shared.exceptions import McpError
from mcp.types import CONNECTION_CLOSED, ErrorData
import pytest

from chartreux.core.tools.mcp.pool import MCPConnectionPool, _StdioConnection
from chartreux.core.tools.mcp.tools import build_stdio_params


def _result():
    return SimpleNamespace(content=[], structuredContent=None, isError=False)


@pytest.mark.asyncio
async def test_cancelled_queued_call_never_dispatches(monkeypatch):
    started, release, queued = asyncio.Event(), asyncio.Event(), asyncio.Event()
    calls = []

    async def call(name, arguments, **kwargs):
        calls.append(name)
        if name == "active":
            started.set()
            await release.wait()
        return _result()

    monkeypatch.setattr(
        "chartreux.core.tools.mcp.pool.enter_stdio_session",
        AsyncMock(return_value=SimpleNamespace(call_tool=call)),
    )
    conn = _StdioConnection(build_stdio_params(["fake"]), None)
    original_put = conn._requests.put

    async def put(request):
        await original_put(request)
        if request is not None and request.tool_name == "cancelled":
            queued.set()

    monkeypatch.setattr(conn._requests, "put", put)
    active = asyncio.create_task(conn.call_tool("active", {}, None))
    cancelled = None
    try:
        await asyncio.wait_for(started.wait(), 2)
        cancelled = asyncio.create_task(conn.call_tool("cancelled", {}, None))
        await asyncio.wait_for(queued.wait(), 2)
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        release.set()
        await asyncio.wait_for(active, 2)
        await asyncio.wait_for(conn.call_tool("barrier", {}, None), 2)
        assert calls == ["active", "barrier"]
    finally:
        release.set()
        for task in (active, cancelled):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(
            *(t for t in (active, cancelled) if t), return_exceptions=True
        )
        await conn.aclose()


@pytest.mark.asyncio
async def test_active_cancellation_stops_worker_operation(monkeypatch):
    started, stopped, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def call(*args, **kwargs):
        started.set()
        try:
            await release.wait()
            return _result()
        finally:
            stopped.set()

    monkeypatch.setattr(
        "chartreux.core.tools.mcp.pool.enter_stdio_session",
        AsyncMock(return_value=SimpleNamespace(call_tool=call)),
    )
    pool = MCPConnectionPool()
    task = asyncio.create_task(
        pool.call_tool(command=["fake"], tool_name="mutate", arguments={})
    )
    try:
        await asyncio.wait_for(started.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        # A bounded wait for worker cancellation, not a sleep used for ordering.
        await asyncio.wait_for(stopped.wait(), 2)
    finally:
        release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await pool.aclose()


@pytest.mark.asyncio
async def test_shutdown_refuses_new_calls_and_is_repeatable(monkeypatch):
    closing, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def close_session():
        closing.set()
        await release.wait()

    async def call(name, arguments, **kwargs):
        calls.append(name)
        return _result()

    async def enter(stack, *args, **kwargs):
        stack.push_async_callback(close_session)
        return SimpleNamespace(call_tool=call)

    monkeypatch.setattr("chartreux.core.tools.mcp.pool.enter_stdio_session", enter)
    pool = MCPConnectionPool()
    await pool.call_tool(command=["fake"], tool_name="initial", arguments={})
    close = asyncio.create_task(pool.aclose())
    try:
        await asyncio.wait_for(closing.wait(), 2)
        with pytest.raises(RuntimeError):
            await asyncio.wait_for(
                pool.call_tool(command=["fake"], tool_name="late", arguments={}), 2
            )
        assert calls == ["initial"]
    finally:
        release.set()
        await asyncio.wait_for(close, 2)
        await pool.aclose()
    with pytest.raises(RuntimeError):
        await pool.call_tool(command=["fake"], tool_name="after-close", arguments={})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        anyio.BrokenResourceError("response lost after commit"),
        McpError(ErrorData(code=CONNECTION_CLOSED, message="Connection closed")),
        ExceptionGroup("transport", [anyio.ClosedResourceError()]),
    ],
)
async def test_committed_mutation_is_not_replayed_after_lost_response(
    monkeypatch, failure
):
    mutations, entered, closed = [], [], []

    async def enter(stack, *args, **kwargs):
        generation = len(entered)
        entered.append(generation)

        async def close():
            closed.append(generation)

        async def call(name, arguments, **kwargs):
            mutations.append(arguments["operation"])
            if generation == 0:
                raise failure
            return _result()

        stack.push_async_callback(close)
        return SimpleNamespace(call_tool=call)

    monkeypatch.setattr("chartreux.core.tools.mcp.pool.enter_stdio_session", enter)
    pool = MCPConnectionPool()
    try:
        with pytest.raises(type(failure)):
            await pool.call_tool(
                command=["fake"], tool_name="mutate", arguments={"operation": "first"}
            )
        assert mutations == ["first"]
        assert closed == [0]
        await pool.call_tool(
            command=["fake"],
            tool_name="mutate",
            arguments={"operation": "explicit-next"},
        )
        assert mutations == ["first", "explicit-next"]
        assert entered == [0, 1]
    finally:
        await pool.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["cancel", "close", "worker-death"])
async def test_session_contexts_unwind_in_owner_task_and_pending_calls_settle(
    monkeypatch, action
):
    started, queued = asyncio.Event(), asyncio.Event()
    owners, exits, dispatched = [], [], []

    @asynccontextmanager
    async def owned():
        owner = asyncio.current_task()
        owners.append(owner)
        async with anyio.create_task_group():
            yield
            assert asyncio.current_task() is owner
        exits.append(asyncio.current_task())

    async def enter(stack, *args, **kwargs):
        await stack.enter_async_context(owned())

        async def call(name, *args, **kwargs):
            dispatched.append(name)
            started.set()
            await asyncio.Event().wait()

        return SimpleNamespace(call_tool=call)

    monkeypatch.setattr("chartreux.core.tools.mcp.pool.enter_stdio_session", enter)
    conn = _StdioConnection(build_stdio_params(["fake"]), None)
    original_put = conn._requests.put

    async def put(req):
        await original_put(req)
        if req.tool_name == "pending":
            queued.set()

    monkeypatch.setattr(conn._requests, "put", put)
    active = asyncio.create_task(conn.call_tool("active", {}, None))
    pending = None
    try:
        await asyncio.wait_for(started.wait(), 2)
        pending = asyncio.create_task(conn.call_tool("pending", {}, None))
        await asyncio.wait_for(queued.wait(), 2)
        if action == "cancel":
            active.cancel()
        elif action == "close":
            await conn.aclose()
        else:
            assert conn._worker is not None
            conn._worker.cancel()
        outcomes = await asyncio.wait_for(
            asyncio.gather(active, pending, return_exceptions=True), 2
        )
        assert all(isinstance(result, BaseException) for result in outcomes)
        await conn.aclose()
        assert owners == exits
        assert dispatched == ["active"]
    finally:
        active.cancel()
        if pending is not None:
            pending.cancel()
        await conn.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [RuntimeError("initialization failed"), TimeoutError("initialization timeout")],
)
async def test_initialization_failure_closes_partial_context_and_next_call_reconnects(
    monkeypatch, failure
):
    entered, closed = [], []

    async def enter(stack, *args, **kwargs):
        owner = asyncio.current_task()
        entered.append(owner)

        async def close():
            assert asyncio.current_task() is owner
            closed.append(owner)

        stack.push_async_callback(close)
        if len(entered) == 1:
            raise failure
        return SimpleNamespace(call_tool=AsyncMock(return_value=_result()))

    monkeypatch.setattr("chartreux.core.tools.mcp.pool.enter_stdio_session", enter)
    pool = MCPConnectionPool()
    try:
        with pytest.raises(type(failure)):
            await pool.call_tool(command=["fake"], tool_name="first", arguments={})
        assert closed == entered
        await pool.call_tool(command=["fake"], tool_name="next", arguments={})
        assert len(entered) == 2
    finally:
        await pool.aclose()
    assert entered == closed


@pytest.mark.asyncio
async def test_cleanup_exception_is_reported_and_never_retried(monkeypatch):
    closes = []

    async def enter(stack, *args, **kwargs):
        async def close():
            closes.append(asyncio.current_task())
            raise RuntimeError("synthetic close failure")

        stack.push_async_callback(close)
        return SimpleNamespace(call_tool=AsyncMock(return_value=_result()))

    monkeypatch.setattr("chartreux.core.tools.mcp.pool.enter_stdio_session", enter)
    pool = MCPConnectionPool()
    await pool.call_tool(command=["fake"], tool_name="initial", arguments={})
    conn = next(iter(pool._conns.values()))
    for _ in range(2):
        with pytest.raises(ExceptionGroup, match="MCP pool cleanup failed"):
            await pool.aclose()
        assert conn in pool._conns.values()
        assert not pool.cleanup_complete
    assert closes == [conn._worker]
    assert conn._stack is not None


@pytest.mark.asyncio
async def test_cancelled_close_waiter_does_not_cancel_worker_cleanup(monkeypatch):
    closing, release = asyncio.Event(), asyncio.Event()

    async def enter(stack, *args, **kwargs):
        async def close():
            closing.set()
            await release.wait()

        stack.push_async_callback(close)
        return SimpleNamespace(call_tool=AsyncMock(return_value=_result()))

    monkeypatch.setattr("chartreux.core.tools.mcp.pool.enter_stdio_session", enter)
    pool = MCPConnectionPool()
    await pool.call_tool(command=["fake"], tool_name="initial", arguments={})
    conn = next(iter(pool._conns.values()))
    waiter = asyncio.create_task(conn.aclose())
    try:
        await asyncio.wait_for(closing.wait(), 2)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert conn._worker is not None and not conn._worker.done()
        release.set()
        await pool.aclose()
        assert pool.cleanup_complete
    finally:
        release.set()
        await asyncio.gather(waiter, return_exceptions=True)
        await pool.aclose()


@pytest.mark.asyncio
async def test_sdk_tool_timeout_is_not_replayed_and_preserves_session(monkeypatch):
    calls = AsyncMock(
        side_effect=[McpError(ErrorData(code=408, message="Timed out")), _result()]
    )
    enter = AsyncMock(return_value=SimpleNamespace(call_tool=calls))
    monkeypatch.setattr("chartreux.core.tools.mcp.pool.enter_stdio_session", enter)
    pool = MCPConnectionPool()
    try:
        with pytest.raises(McpError):
            await pool.call_tool(
                command=["fake"], tool_name="first", arguments={}, tool_timeout_sec=0.1
            )
        assert calls.await_count == 1
        assert calls.call_args.kwargs["read_timeout_seconds"] == timedelta(seconds=0.1)
        await pool.call_tool(command=["fake"], tool_name="next", arguments={})
        assert enter.await_count == 1
    finally:
        await pool.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("resist_in", ["operation", "cleanup"])
async def test_close_timeout_reports_failure_and_retains_worker(monkeypatch, resist_in):
    started, cancelled, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    owners, exits = [], []
    monkeypatch.setattr("chartreux.core.tools.mcp.pool._CLOSE_TIMEOUT_SEC", 0.05)

    async def enter(stack, *args, **kwargs):
        owner = asyncio.current_task()
        owners.append(owner)

        async def close():
            if resist_in == "cleanup":
                cancelled.set()
                while not release.is_set():
                    try:
                        await release.wait()
                    except asyncio.CancelledError:
                        pass
            assert asyncio.current_task() is owner
            exits.append(owner)

        async def call(*args, **kwargs):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                if resist_in == "operation":
                    await release.wait()
                raise

        stack.push_async_callback(close)
        return SimpleNamespace(call_tool=call)

    monkeypatch.setattr("chartreux.core.tools.mcp.pool.enter_stdio_session", enter)
    pool = MCPConnectionPool()
    active = asyncio.create_task(
        pool.call_tool(command=["fake"], tool_name="mutate", arguments={})
    )
    try:
        await asyncio.wait_for(started.wait(), 2)
        conn = next(iter(pool._conns.values()))
        worker = conn._worker
        assert worker is not None
        with pytest.raises(TimeoutError, match="MCP.*cleanup"):
            await pool.aclose()
        assert cancelled.is_set()
        assert not worker.done()
        assert conn in pool._conns.values()
        with pytest.raises(TimeoutError):
            await conn.aclose()
        with pytest.raises(RuntimeError, match="closed"):
            await pool.call_tool(command=["fake"], tool_name="late", arguments={})
        release.set()
        await asyncio.gather(active, return_exceptions=True)
        await pool.aclose()
        assert worker.done()
        assert owners == exits
        assert pool._conns == {}
    finally:
        release.set()
        await asyncio.gather(active, return_exceptions=True)
        await pool.aclose()
