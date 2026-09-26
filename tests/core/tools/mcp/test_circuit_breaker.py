from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import anyio
from mcp.shared.exceptions import McpError
from mcp.types import ErrorData
import pytest

from chartreux.core.tools.base import BaseToolConfig, BaseToolState, InvokeContext
from chartreux.core.tools.mcp.pool import MCPConnectionPool
from chartreux.core.tools.mcp.tools import (
    MCPServerCircuitBreaker,
    MCPServerCooldownError,
    _is_call_timeout,
    _OpenArgs,
    create_mcp_http_proxy_tool_class,
    create_mcp_stdio_proxy_tool_class,
)
from chartreux.core.tools.remote import MCPToolResult, RemoteTool


def _result():
    return SimpleNamespace(content=[], structuredContent=None, isError=False)


def _timeout_error() -> McpError:
    # The SDK converts a read_timeout_seconds overrun into McpError code 408.
    return McpError(ErrorData(code=408, message="Timed out"))


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _expire_cooldown(breaker: MCPServerCircuitBreaker) -> None:
    """Patch the clock: move the cooldown deadline into the past."""
    until = breaker._cooldown_until
    assert until is not None
    breaker._cooldown_until = until - (MCPServerCircuitBreaker.COOLDOWN_SEC + 1)


class TestCallTimeoutClassification:
    def test_only_the_sdk_timeout_signal_counts(self):
        assert _is_call_timeout(_timeout_error())
        assert _is_call_timeout(ExceptionGroup("wrapped", [_timeout_error()]))
        assert not _is_call_timeout(
            McpError(ErrorData(code=401, message="Unauthorized"))
        )
        assert not _is_call_timeout(McpError(ErrorData(code=-32000, message="boom")))
        assert not _is_call_timeout(RuntimeError("normal tool failure"))
        assert not _is_call_timeout(TimeoutError("cleanup wait"))


class TestBreakerStateMachine:
    def test_cumulative_timeouts_and_single_flight_probe(self):
        clock = FakeClock()
        breaker = MCPServerCircuitBreaker(clock=clock)
        for _ in range(breaker.TIMEOUT_THRESHOLD):
            breaker.record_timeout(breaker.admission("srv"))
        with pytest.raises(MCPServerCooldownError):
            breaker.admission("srv")
        clock.advance(breaker.COOLDOWN_SEC + 1)
        probe = breaker.admission("srv")
        with pytest.raises(MCPServerCooldownError) as exc:
            breaker.admission("srv")
        assert exc.value.retry_in == pytest.approx(breaker.PROBE_RETRY_SEC)
        breaker.record_probe_incomplete(probe)
        replacement = breaker.admission("srv")
        breaker.record_success(replacement)
        breaker.admission("srv")

    def test_stale_completion_cannot_release_probe_or_clear_cooldown(self):
        clock = FakeClock()
        breaker = MCPServerCircuitBreaker(clock=clock)
        old = breaker.admission("srv")
        for _ in range(breaker.TIMEOUT_THRESHOLD):
            breaker.record_timeout(breaker.admission("srv"))
        breaker.record_success(old)
        with pytest.raises(MCPServerCooldownError):
            breaker.admission("srv")
        clock.advance(breaker.COOLDOWN_SEC + 1)
        probe = breaker.admission("srv")
        breaker.record_probe_incomplete(old)
        breaker.record_timeout(old)
        breaker.record_success(old)
        with pytest.raises(MCPServerCooldownError):
            breaker.admission("srv")
        breaker.record_timeout(probe)
        with pytest.raises(MCPServerCooldownError):
            breaker.admission("srv")

    def test_old_probe_cancel_cannot_release_new_probe(self):
        clock = FakeClock()
        breaker = MCPServerCircuitBreaker(clock=clock)
        for _ in range(breaker.TIMEOUT_THRESHOLD):
            breaker.record_timeout(breaker.admission("srv"))
        clock.advance(breaker.COOLDOWN_SEC + 1)
        old_probe = breaker.admission("srv")
        breaker.record_probe_incomplete(old_probe)
        breaker.record_success(old_probe)
        assert breaker._cooldown_until is not None
        new_probe = breaker.admission("srv")
        breaker.record_probe_incomplete(old_probe)
        breaker.record_timeout(old_probe)
        breaker.record_success(old_probe)
        with pytest.raises(MCPServerCooldownError):
            breaker.admission("srv")
        breaker.record_success(new_probe)
        breaker.admission("srv")


class TestPoolCircuitBreaker:
    @staticmethod
    def _hang_pool(monkeypatch, calls, *, clock=None):
        monkeypatch.setattr(
            "chartreux.core.tools.mcp.pool.enter_stdio_session",
            AsyncMock(return_value=SimpleNamespace(call_tool=calls)),
        )
        if clock is not None:
            return MCPConnectionPool(clock=clock)
        return MCPConnectionPool()

    @pytest.mark.asyncio
    async def test_repeated_timeouts_trip_and_fail_fast(self, monkeypatch):
        clock = FakeClock()
        calls = AsyncMock(side_effect=_timeout_error())
        pool = self._hang_pool(monkeypatch, calls, clock=clock)
        try:
            for _ in range(MCPServerCircuitBreaker.TIMEOUT_THRESHOLD):
                with pytest.raises(McpError):
                    await pool.call_tool(
                        command=["fake"],
                        tool_name="hang",
                        arguments={},
                        tool_timeout_sec=0.1,
                        server_name="hung",
                    )
            assert calls.await_count == MCPServerCircuitBreaker.TIMEOUT_THRESHOLD
            with pytest.raises(MCPServerCooldownError, match="hung") as exc_info:
                await pool.call_tool(
                    command=["fake"],
                    tool_name="hang",
                    arguments={},
                    tool_timeout_sec=0.1,
                    server_name="hung",
                )
            assert exc_info.value.retry_in == pytest.approx(
                MCPServerCircuitBreaker.COOLDOWN_SEC
            )
            # Failed fast: the hanging server saw no further dispatches.
            assert calls.await_count == MCPServerCircuitBreaker.TIMEOUT_THRESHOLD
        finally:
            await pool.aclose()

    @pytest.mark.asyncio
    async def test_recovery_probe_readmits_after_cooldown(self, monkeypatch):
        clock = FakeClock()
        calls = AsyncMock(side_effect=_timeout_error())
        session = SimpleNamespace(call_tool=calls)
        monkeypatch.setattr(
            "chartreux.core.tools.mcp.pool.enter_stdio_session",
            AsyncMock(return_value=session),
        )
        pool = MCPConnectionPool(clock=clock)
        try:
            for _ in range(MCPServerCircuitBreaker.TIMEOUT_THRESHOLD):
                with pytest.raises(McpError):
                    await pool.call_tool(
                        command=["fake"],
                        tool_name="hang",
                        arguments={},
                        server_name="hung",
                    )
            # The injected clock moves the cooldown window into the past.
            clock.advance(MCPServerCircuitBreaker.COOLDOWN_SEC + 1)
            calls.side_effect = None
            calls.return_value = _result()
            result = await pool.call_tool(
                command=["fake"], tool_name="hang", arguments={}, server_name="hung"
            )
            assert result.ok is True
            assert calls.await_count == MCPServerCircuitBreaker.TIMEOUT_THRESHOLD + 1
            # Re-admitted: later calls dispatch normally.
            await pool.call_tool(
                command=["fake"], tool_name="hang", arguments={}, server_name="hung"
            )
            assert calls.await_count == MCPServerCircuitBreaker.TIMEOUT_THRESHOLD + 2
        finally:
            await pool.aclose()

    @pytest.mark.asyncio
    async def test_success_resets_the_timeout_counter(self, monkeypatch):
        outcomes = [
            _timeout_error(),
            _timeout_error(),
            _result(),
            _timeout_error(),
            _timeout_error(),
            _timeout_error(),
        ]
        calls = AsyncMock(side_effect=outcomes)
        pool = self._hang_pool(monkeypatch, calls)
        try:
            expectations = [McpError, McpError, None, McpError, McpError, McpError]
            for expectation in expectations:
                if expectation is None:
                    await pool.call_tool(
                        command=["fake"], tool_name="t", arguments={}, server_name="srv"
                    )
                else:
                    with pytest.raises(expectation):
                        await pool.call_tool(
                            command=["fake"],
                            tool_name="t",
                            arguments={},
                            server_name="srv",
                        )
            assert calls.await_count == len(outcomes)
            with pytest.raises(MCPServerCooldownError):
                await pool.call_tool(
                    command=["fake"], tool_name="t", arguments={}, server_name="srv"
                )
            assert calls.await_count == len(outcomes)
        finally:
            await pool.aclose()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "failure",
        [
            McpError(ErrorData(code=-32000, message="tool error")),
            McpError(ErrorData(code=401, message="Unauthorized")),
            anyio.BrokenResourceError("transport died"),
        ],
    )
    async def test_non_timeout_failures_never_trip(self, monkeypatch, failure):
        calls = AsyncMock(side_effect=failure)
        pool = self._hang_pool(monkeypatch, calls)
        try:
            for _ in range(MCPServerCircuitBreaker.TIMEOUT_THRESHOLD + 2):
                with pytest.raises(type(failure)):
                    await pool.call_tool(
                        command=["fake"], tool_name="t", arguments={}, server_name="srv"
                    )
            # Every call dispatched: no cooldown was ever applied.
            assert calls.await_count == MCPServerCircuitBreaker.TIMEOUT_THRESHOLD + 2
        finally:
            await pool.aclose()

    @pytest.mark.asyncio
    async def test_breakers_are_isolated_per_server(self, monkeypatch):
        """One hung server's cooldown must not fail-fast a healthy peer."""
        hang_calls = AsyncMock(side_effect=_timeout_error())
        ok_calls = AsyncMock(return_value=_result())
        sessions = iter([
            SimpleNamespace(call_tool=hang_calls),
            SimpleNamespace(call_tool=ok_calls),
        ])

        async def fake_enter_stdio_session(stack, params, *, init_timeout):
            return next(sessions)

        monkeypatch.setattr(
            "chartreux.core.tools.mcp.pool.enter_stdio_session",
            fake_enter_stdio_session,
        )
        pool = MCPConnectionPool(clock=FakeClock())
        try:
            for _ in range(MCPServerCircuitBreaker.TIMEOUT_THRESHOLD):
                with pytest.raises(McpError):
                    await pool.call_tool(
                        command=["fake"],
                        tool_name="t",
                        arguments={},
                        server_name="hung",
                    )
            with pytest.raises(MCPServerCooldownError, match="hung"):
                await pool.call_tool(
                    command=["fake"], tool_name="t", arguments={}, server_name="hung"
                )
            # The healthy peer shares the command but has its own breaker key:
            # it keeps dispatching while the hung server is cooling down.
            result = await pool.call_tool(
                command=["fake"], tool_name="t", arguments={}, server_name="healthy"
            )
            assert result.ok is True
            assert ok_calls.await_count == 1
            assert hang_calls.await_count == MCPServerCircuitBreaker.TIMEOUT_THRESHOLD
        finally:
            await pool.aclose()

    @pytest.mark.asyncio
    async def test_recovery_probe_blocks_concurrent_calls(self, monkeypatch):
        clock = FakeClock()
        calls = AsyncMock(side_effect=_timeout_error())
        session = SimpleNamespace(call_tool=calls)
        monkeypatch.setattr(
            "chartreux.core.tools.mcp.pool.enter_stdio_session",
            AsyncMock(return_value=session),
        )
        pool = MCPConnectionPool(clock=clock)
        release = asyncio.Event()
        try:
            for _ in range(MCPServerCircuitBreaker.TIMEOUT_THRESHOLD):
                with pytest.raises(McpError):
                    await pool.call_tool(
                        command=["fake"],
                        tool_name="hang",
                        arguments={},
                        server_name="hung",
                    )
            clock.advance(MCPServerCircuitBreaker.COOLDOWN_SEC + 1)

            started = asyncio.Event()

            async def hang(name, arguments, **kwargs):
                started.set()
                await release.wait()
                return _result()

            session.call_tool = hang
            probe = asyncio.create_task(
                pool.call_tool(
                    command=["fake"], tool_name="hang", arguments={}, server_name="hung"
                )
            )
            await asyncio.wait_for(started.wait(), 2)
            with pytest.raises(MCPServerCooldownError) as exc_info:
                await pool.call_tool(
                    command=["fake"], tool_name="hang", arguments={}, server_name="hung"
                )
            # The probe is in flight: the blocked call reports an honest
            # probe window instead of "retry in 0s".
            assert exc_info.value.retry_in == pytest.approx(
                MCPServerCircuitBreaker.PROBE_RETRY_SEC
            )
            release.set()
            result = await asyncio.wait_for(probe, 2)
            assert result.ok is True
        finally:
            release.set()
            await pool.aclose()


class TestHttpProxyCircuitBreaker:
    @pytest.mark.asyncio
    async def test_old_completion_does_not_release_live_probe(self, monkeypatch):
        clock = FakeClock()
        cls: Any = create_mcp_http_proxy_tool_class(
            url="https://mcp.example", remote=RemoteTool(name="echo"), alias="ex"
        )
        cls._breaker = MCPServerCircuitBreaker(clock=clock)
        old_started = asyncio.Event()
        release_old = asyncio.Event()
        probe_started = asyncio.Event()
        release_probe = asyncio.Event()

        async def fake_call(url, tool_name, arguments, **kwargs):
            if arguments.get("old"):
                old_started.set()
                await release_old.wait()
            if arguments.get("probe"):
                probe_started.set()
                await release_probe.wait()
            return MCPToolResult(server=url, tool=tool_name)

        monkeypatch.setattr("chartreux.core.tools.mcp.tools.call_tool_http", fake_call)
        old = asyncio.create_task(cls._call_remote({"old": True}))
        try:
            await asyncio.wait_for(old_started.wait(), 2)
            for _ in range(cls._breaker.TIMEOUT_THRESHOLD):
                cls._breaker.record_timeout(cls._breaker.admission("ex"))
            clock.advance(cls._breaker.COOLDOWN_SEC + 1)
            probe = asyncio.create_task(cls._call_remote({"probe": True}))
            try:
                await asyncio.wait_for(probe_started.wait(), 2)
                release_old.set()
                await asyncio.wait_for(old, 2)
                with pytest.raises(MCPServerCooldownError):
                    await cls._call_remote({})
            finally:
                release_probe.set()
                await asyncio.wait_for(probe, 2)
        finally:
            release_old.set()

    @pytest.mark.asyncio
    async def test_http_timeouts_trip_fail_fast_and_readmit(self, monkeypatch):
        tool_cls: Any = create_mcp_http_proxy_tool_class(
            url="https://mcp.example", remote=RemoteTool(name="echo"), alias="ex"
        )
        calls = []

        async def fake_call_tool_http(url, tool_name, arguments, **kwargs):
            calls.append(tool_name)
            if len(calls) <= MCPServerCircuitBreaker.TIMEOUT_THRESHOLD:
                raise _timeout_error()
            return MCPToolResult(server=url, tool=tool_name, ok=True, text="done")

        monkeypatch.setattr(
            "chartreux.core.tools.mcp.tools.call_tool_http", fake_call_tool_http
        )
        for _ in range(MCPServerCircuitBreaker.TIMEOUT_THRESHOLD):
            with pytest.raises(McpError):
                await tool_cls._call_remote({})
        with pytest.raises(MCPServerCooldownError, match="ex"):
            await tool_cls._call_remote({})
        assert len(calls) == MCPServerCircuitBreaker.TIMEOUT_THRESHOLD
        # Patch the clock: the cooldown expires and the probe re-admits.
        _expire_cooldown(tool_cls._breaker)
        result = await tool_cls._call_remote({})
        assert result.ok is True
        assert len(calls) == MCPServerCircuitBreaker.TIMEOUT_THRESHOLD + 1


class TestStdioFallbackCircuitBreaker:
    @pytest.mark.asyncio
    async def test_no_pool_fallback_trips_and_fails_fast(self, monkeypatch):
        cls = create_mcp_stdio_proxy_tool_class(
            command=["srv"], remote=RemoteTool(name="t"), alias="local"
        )
        tool = cls(lambda: BaseToolConfig(), BaseToolState())
        ctx = InvokeContext(tool_call_id="1", mcp_pool=None)
        calls = []

        async def fake_call_tool_stdio(command, tool_name, arguments, **kwargs):
            calls.append(tool_name)
            raise _timeout_error()

        monkeypatch.setattr(
            "chartreux.core.tools.mcp.tools.call_tool_stdio", fake_call_tool_stdio
        )

        async def invoke():
            results = []
            async for event in tool.run(_OpenArgs(), ctx):
                results.append(event)
            return results

        for _ in range(MCPServerCircuitBreaker.TIMEOUT_THRESHOLD):
            with pytest.raises(Exception, match="MCP stdio call failed"):
                await invoke()
        assert len(calls) == MCPServerCircuitBreaker.TIMEOUT_THRESHOLD
        with pytest.raises(Exception, match="cooling down"):
            await invoke()
        # Failed fast: the one-shot fallback saw no further dispatches.
        assert len(calls) == MCPServerCircuitBreaker.TIMEOUT_THRESHOLD
