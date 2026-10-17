"""MCP OAuth loopback callback handler tests (happy path, port-in-use, missing code, wait timeout and its port release, cancellation, idle/silent/slow connections, oversized-request rejection, connection cap)."""

from __future__ import annotations

import asyncio
from contextlib import suppress
import socket
from unittest.mock import patch

import pytest

from chartreux.core.auth.mcp_oauth import (
    LoopbackCallbackHandler,
    MCPOAuthCallbackTimeout,
    MCPOAuthError,
    MCPOAuthPortInUse,
)
from tests.core.auth._mcp_oauth_support import (
    _connect_when_bound,
    _free_port,
    _send_callback,
)


class TestLoopbackCallbackHandler:
    @pytest.mark.asyncio
    async def test_happy_path(self) -> None:
        port = _free_port()
        handler = LoopbackCallbackHandler(port=port, server_alias="demo")

        async def driver() -> bytes:
            return await _send_callback(port, "code=AUTH_CODE_123&state=STATE_XYZ")

        serve_task = asyncio.create_task(handler.serve_once())
        driver_task = asyncio.create_task(driver())
        code, state = await serve_task
        response = await driver_task

        assert code == "AUTH_CODE_123"
        assert state == "STATE_XYZ"
        assert b"200 OK" in response
        assert b"Login complete" in response

    @pytest.mark.asyncio
    async def test_port_in_use_raises(self) -> None:
        port = _free_port()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
        sock.bind(("127.0.0.1", port))
        sock.listen(1)
        try:
            handler = LoopbackCallbackHandler(port=port, server_alias="demo")
            with pytest.raises(MCPOAuthPortInUse) as exc_info:
                await handler.serve_once()
            assert exc_info.value.port == port
            assert exc_info.value.server_alias == "demo"
            assert "redirect_port" in str(exc_info.value)
        finally:
            sock.close()

    @pytest.mark.asyncio
    async def test_missing_code_raises(self) -> None:
        port = _free_port()
        handler = LoopbackCallbackHandler(port=port, server_alias="demo")

        async def driver() -> bytes:
            return await _send_callback(port, "error=access_denied&state=S")

        serve_task = asyncio.create_task(handler.serve_once())
        driver_task = asyncio.create_task(driver())
        with pytest.raises(MCPOAuthError, match="missing 'code'"):
            await serve_task
        response = await driver_task
        assert b"400 Bad Request" in response

    @pytest.mark.asyncio
    async def test_callback_wait_times_out_and_releases_port(self) -> None:
        port = _free_port()
        handler = LoopbackCallbackHandler(port=port, server_alias="demo")

        with patch("chartreux.core.auth.mcp_oauth._CALLBACK_TIMEOUT_SECONDS", 0.2):
            with pytest.raises(MCPOAuthCallbackTimeout, match="has been released"):
                await handler.serve_once()

        # The port must be immediately reusable by the next login attempt.
        retry = LoopbackCallbackHandler(port=port, server_alias="demo")
        serve_task = asyncio.create_task(retry.serve_once())
        driver_task = asyncio.create_task(
            _send_callback(port, "code=AUTH_CODE_123&state=STATE_XYZ")
        )
        code, state = await serve_task
        await driver_task
        assert code == "AUTH_CODE_123"
        assert state == "STATE_XYZ"

    @pytest.mark.asyncio
    async def test_cancelling_the_callback_wait_releases_port(self) -> None:
        port = _free_port()
        handler = LoopbackCallbackHandler(port=port, server_alias="demo")
        serve_task = asyncio.create_task(handler.serve_once())

        # Give the loopback server a moment to bind before cancelling the wait,
        # exactly like a user interrupting a login in progress.
        await asyncio.sleep(0.05)
        serve_task.cancel()
        with suppress(asyncio.CancelledError):
            await serve_task
        # The wait really was cancelled mid-flight, not completed early.
        assert serve_task.cancelled()

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", port))

    @pytest.mark.asyncio
    async def test_idle_accepted_connection_does_not_extend_callback_timeout(
        self,
    ) -> None:
        port = _free_port()
        handler = LoopbackCallbackHandler(port=port, server_alias="demo")
        with patch("chartreux.core.auth.mcp_oauth._CALLBACK_TIMEOUT_SECONDS", 0.2):
            task = asyncio.create_task(handler.serve_once())
            reader, writer = await _connect_when_bound(port)
            try:
                with pytest.raises(MCPOAuthCallbackTimeout):
                    await asyncio.wait_for(task, 1)
                assert await reader.read() == b""
            finally:
                writer.close()
                await writer.wait_closed()
        retry = asyncio.create_task(
            LoopbackCallbackHandler(port=port, server_alias="demo").serve_once()
        )
        response = await _send_callback(port, "code=SECOND")
        assert b"200 OK" in response
        assert await retry == ("SECOND", None)

    @pytest.mark.asyncio
    async def test_silent_and_empty_connections_do_not_abort_late_callback(
        self,
    ) -> None:
        port = _free_port()
        task = asyncio.create_task(
            LoopbackCallbackHandler(port=port, server_alias="demo").serve_once()
        )
        silent_reader, silent_writer = await _connect_when_bound(port)
        _, empty_writer = await _connect_when_bound(port)
        try:
            empty_writer.close()
            await empty_writer.wait_closed()
            # Use production deadlines: the silent connection outlives the
            # per-connection timeout, but the login stays open for the browser.
            assert await asyncio.wait_for(silent_reader.read(), 7) == b""
            assert not task.done()
            response = await _send_callback(port, "code=LATE&state=OK")
            assert b"200 OK" in response
            assert await task == ("LATE", "OK")
        finally:
            silent_writer.close()
            await silent_writer.wait_closed()
            if not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task

    @pytest.mark.asyncio
    async def test_slow_trickle_has_one_deadline(self) -> None:
        port = _free_port()
        with patch("chartreux.core.auth.mcp_oauth._CALLBACK_CONNECTION_SECONDS", 0.1):
            task = asyncio.create_task(
                LoopbackCallbackHandler(port=port, server_alias="demo").serve_once()
            )
            reader, writer = await _connect_when_bound(port)
            try:
                writer.write(b"GET /callback?code=ok HTTP/1.1\r\n")
                await writer.drain()
                with pytest.raises(MCPOAuthError, match="slow request"):
                    await asyncio.wait_for(task, 1)
                assert await reader.read() == b""
            finally:
                writer.close()
                await writer.wait_closed()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "raw_request",
        [
            b"x" * 5000,
            b"GET /callback?code=x HTTP/1.1\r\n" + b"X: y\r\n" * 70,
            b"bad\r\n\r\n",
        ],
    )
    async def test_bad_or_oversized_request_is_rejected(
        self, raw_request: bytes
    ) -> None:
        port = _free_port()
        task = asyncio.create_task(
            LoopbackCallbackHandler(port=port, server_alias="demo").serve_once()
        )
        reader, writer = await _connect_when_bound(port)
        try:
            writer.write(raw_request)
            await writer.drain()
            with pytest.raises(MCPOAuthError):
                await asyncio.wait_for(task, 1)
            await reader.read()
        finally:
            writer.close()
            await writer.wait_closed()

    @pytest.mark.asyncio
    async def test_connection_cap_rejects_excess_without_consuming_callback(
        self,
    ) -> None:
        port = _free_port()
        with patch("chartreux.core.auth.mcp_oauth._CALLBACK_CONNECTIONS", 1):
            task = asyncio.create_task(
                LoopbackCallbackHandler(port=port, server_alias="demo").serve_once()
            )
            _, idle = await _connect_when_bound(port)
            reader, extra = await _connect_when_bound(port)
            try:
                assert await asyncio.wait_for(reader.read(), 1) == b""
                assert not task.done()
            finally:
                idle.close()
                extra.close()
                await idle.wait_closed()
                await extra.wait_closed()
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
