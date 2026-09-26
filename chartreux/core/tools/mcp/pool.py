from __future__ import annotations

import asyncio
from collections.abc import Callable
import contextlib
from dataclasses import dataclass
from datetime import timedelta
import hashlib
import time
from typing import TYPE_CHECKING, Any

import anyio

from chartreux.core.tools.mcp.tools import (
    MCPServerCircuitBreaker,
    MCPToolResult,
    _is_call_timeout as is_call_timeout,
    _parse_call_result as parse_call_result,
    build_stdio_params,
    enter_stdio_session,
)
from chartreux.core.tools.secret_redaction import (
    ScrubPolicy,
    bind_policy,
    current_policy,
)
from chartreux.observability.logging import logger

if TYPE_CHECKING:
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters


# Errors that indicate the stdio transport (subprocess / pipe) is gone and the
# session must be respawned. Tool-level errors and timeouts are deliberately
# excluded: they are legitimate server responses, not dead connections.
_TRANSPORT_ERRORS: tuple[type[BaseException], ...] = (
    anyio.ClosedResourceError,
    anyio.BrokenResourceError,
    BrokenPipeError,
    ConnectionError,
    EOFError,
)

# Bound local cancellation/cleanup waits. Cancellation cannot undo remote effects.
_CLOSE_TIMEOUT_SEC = 5.0


def _transport_failed(exc: BaseException) -> bool:
    from mcp.shared.exceptions import McpError
    from mcp.types import CONNECTION_CLOSED

    if isinstance(exc, BaseExceptionGroup):
        return any(_transport_failed(child) for child in exc.exceptions)
    return isinstance(exc, _TRANSPORT_ERRORS) or (
        isinstance(exc, McpError) and exc.error.code == CONNECTION_CLOSED
    )


def stdio_key(command: list[str], env: dict[str, str] | None, cwd: str | None) -> str:
    # \0 and \x01 delimit the three identity fields so distinct inputs cannot
    # collide (e.g. ["a b"] vs ["a", "b"], or command vs env boundaries).
    env_part = "\0".join(f"{k}={v}" for k, v in sorted((env or {}).items()))
    raw = "\0".join(command) + "\x01" + env_part + "\x01" + (cwd or "")
    return hashlib.blake2s(raw.encode("utf-8"), digest_size=16).hexdigest()


@dataclass
class _Request:
    tool_name: str
    arguments: dict[str, Any]
    call_timeout: timedelta | None
    future: asyncio.Future[Any]
    policy: ScrubPolicy


class _StdioConnection:
    """A single long-lived stdio MCP session owned by one dedicated task.

    The MCP ``stdio_client`` and ``ClientSession`` context managers open anyio
    task groups bound to the task that enters them, so the session must be
    entered, used, and exited all within the same task. A single worker task
    owns the session for its whole lifetime and services calls from a queue;
    callers submit a request and await its future. Because the worker handles
    one request at a time, calls to the same server are serialized (stateful
    servers never see interleaved requests). A failed operation is never replayed;
    a later explicit call can establish a new session.
    """

    def __init__(
        self, params: StdioServerParameters, init_timeout: timedelta | None
    ) -> None:
        self._params = params
        self._init_timeout = init_timeout
        self._requests: asyncio.Queue[_Request | None] = asyncio.Queue()
        self._worker: asyncio.Task[None] | None = None
        self._session: ClientSession | None = None
        self._stack: contextlib.AsyncExitStack | None = None
        self._inflight: _Request | None = None
        self._closed = False
        self._closing_session = False
        self._cleanup_error: RuntimeError | None = None

    def retire(self, *, drain_active: bool = False) -> None:
        if self._closed:
            return
        self._closed = True
        worker = self._worker
        if drain_active:
            # All admitted requests (including queued work) retain their
            # admission policy; the sentinel closes the worker after them.
            if worker is not None and not worker.done():
                self._requests.put_nowait(None)
            return
        if (
            worker is not None
            and not worker.done()
            and not worker.cancelling()
            and not self._closing_session
        ):
            worker.cancel()

    def _ensure_worker(self) -> None:
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._run())

    async def call_tool(
        self, tool_name: str, arguments: dict[str, Any], call_timeout: timedelta | None
    ) -> Any:
        if self._closed:
            raise RuntimeError("MCP stdio connection closed")
        self._ensure_worker()
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        req = _Request(tool_name, arguments, call_timeout, future, current_policy())
        try:
            self._requests.put_nowait(req)
            return await future
        except asyncio.CancelledError:
            future.cancel()
            if self._inflight is req:
                self.retire()
                try:
                    await self._wait_closed(_CLOSE_TIMEOUT_SEC)
                except Exception:
                    # Preserve caller cancellation, but report incomplete cleanup.
                    # The pool still owns this connection and joins it on shutdown.
                    logger.warning("MCP cancellation cleanup incomplete", exc_info=True)
            raise

    async def _run(self) -> None:
        try:
            while True:
                req = await self._requests.get()
                if req is None:
                    return
                if req.future.cancelled():
                    continue
                self._inflight = req
                try:
                    with bind_policy(req.policy):
                        result = await self._handle(req)
                except Exception as exc:
                    if not req.future.done():
                        req.future.set_exception(exc)
                    self._inflight = None
                else:
                    if not req.future.done():
                        req.future.set_result(result)
                    self._inflight = None
        finally:
            try:
                await self._close_session()
            finally:
                self._fail_pending()

    async def _handle(self, req: _Request) -> Any:
        session = await self._ensure_session()
        if req.future.cancelled():
            raise asyncio.CancelledError
        try:
            return await session.call_tool(
                req.tool_name, req.arguments, read_timeout_seconds=req.call_timeout
            )
        except Exception as exc:
            if _transport_failed(exc):
                logger.debug("MCP stdio transport died; operation will not be replayed")
                await self._close_session()
            raise

    async def _ensure_session(self) -> ClientSession:
        if self._session is not None:
            return self._session
        stack = contextlib.AsyncExitStack()

        self._stack = stack
        try:
            session = await enter_stdio_session(
                stack, self._params, init_timeout=self._init_timeout
            )
        except BaseException:
            await self._close_session()
            raise
        self._session = session
        return session

    async def _close_session(self) -> None:
        stack = self._stack
        self._session = None
        if stack is not None and self._cleanup_error is None:
            self._closing_session = True
            try:
                # Unwind in the entering task, without cancelling SDK process
                # termination halfway through. Only the owner's join is bounded.
                await stack.aclose()
            except BaseException as exc:
                self._closed = True
                self._cleanup_error = RuntimeError("MCP stdio cleanup failed")
                self._cleanup_error.__cause__ = exc
                raise self._cleanup_error
            else:
                self._stack = None
            finally:
                self._closing_session = False

    def _fail_pending(self) -> None:
        err = RuntimeError("MCP stdio connection closed")
        if self._inflight is not None and not self._inflight.future.done():
            self._inflight.future.set_exception(err)
        self._inflight = None
        while not self._requests.empty():
            try:
                req = self._requests.get_nowait()
            except asyncio.QueueEmpty:
                break
            if req is not None and not req.future.done():
                req.future.set_exception(err)

    async def aclose(self) -> None:
        self.retire()
        await self._wait_closed(_CLOSE_TIMEOUT_SEC)

    async def _wait_closed(self, timeout: float | None) -> None:
        worker = self._worker
        if worker is not None:
            # asyncio.wait neither cancels the worker on timeout nor conflates
            # its cancellation with cancellation of the task joining it.
            _, pending = await asyncio.wait({worker}, timeout=timeout)
            if pending:
                raise TimeoutError("MCP stdio cleanup still pending")
            with contextlib.suppress(asyncio.CancelledError):
                worker.result()
        if self._cleanup_error is not None:
            raise self._cleanup_error
        self._fail_pending()


class MCPConnectionPool:
    """Session-scoped pool of persistent stdio MCP connections.

    Owned by an ``AgentLoop`` and created lazily in that loop's event loop on the
    first call (discovery runs in a throwaway loop, so connections must not be
    shared with it). Connections live until ``aclose`` is called at session end.
    """

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        policy: ScrubPolicy | None = None,
    ) -> None:
        self._policy = policy or current_policy()
        self._clock = clock
        self._conns: dict[str, _StdioConnection] = {}
        # Per-server hung-server cooldowns; in-memory for the pool's lifetime,
        # mirroring the LLM-side AvailabilityRegistry (never persisted).
        self._breakers: dict[str, MCPServerCircuitBreaker] = {}
        self._creation_lock = asyncio.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None

    def retire(self, *, drain_active: bool = False) -> None:
        """Stop admissions; optionally drain admitted requests for about 5 seconds.

        After the bounded drain deadline, workers still running are cancelled.
        """
        self._closed = True
        for conn in self._conns.values():
            conn.retire(drain_active=drain_active)
        if self._close_task is None or (
            self._close_task.done() and self._close_task.exception() is not None
        ):
            self._close_task = asyncio.create_task(self._close_connections())
            self._close_task.add_done_callback(self._report_close_failure)

    @staticmethod
    def _report_close_failure(task: asyncio.Task[None]) -> None:
        if not task.cancelled() and task.exception() is not None:
            logger.warning("MCP pool cleanup failed; connections retained")

    @property
    def cleanup_complete(self) -> bool:
        return (
            self._close_task is not None
            and self._close_task.done()
            and not self._close_task.cancelled()
            and self._close_task.exception() is None
        )

    def _bind_loop(self) -> None:
        loop = asyncio.get_running_loop()
        if self._loop is loop:
            return
        if self._loop is not None:
            if any(
                conn._stack is not None
                or (conn._worker is not None and not conn._worker.done())
                for conn in self._conns.values()
            ):
                raise RuntimeError("MCP pool cleanup required on its owning event loop")
            # Only fully unwound connections can be discarded across loops.
            logger.debug(
                "MCP pool bound to a new event loop; dropping %d stale connection(s)",
                len(self._conns),
            )
            for conn in self._conns.values():
                conn._closed = True
            self._conns.clear()
        self._loop = loop

    async def _get_or_create(
        self,
        key: str,
        command: list[str],
        env: dict[str, str] | None,
        cwd: str | None,
        startup_timeout_sec: float | None,
    ) -> _StdioConnection:
        async with self._creation_lock:
            if self._closed:
                raise RuntimeError("MCP connection pool closed")
            if (conn := self._conns.get(key)) is not None:
                if not conn._closed:
                    return conn
                await conn.aclose()
                if self._closed:
                    raise RuntimeError("MCP connection pool closed")
            with bind_policy(self._policy):
                params = build_stdio_params(command, env=env, cwd=cwd)
            init_timeout = (
                timedelta(seconds=startup_timeout_sec) if startup_timeout_sec else None
            )
            conn = _StdioConnection(params, init_timeout)
            self._conns[key] = conn
            return conn

    async def call_tool(
        self,
        *,
        command: list[str],
        tool_name: str,
        arguments: dict[str, Any],
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        startup_timeout_sec: float | None = None,
        tool_timeout_sec: float | None = None,
        server_name: str | None = None,
    ) -> MCPToolResult:
        if self._closed:
            raise RuntimeError("MCP connection pool closed")
        self._bind_loop()
        key = repr((stdio_key(command, env, cwd), server_name))
        breaker = self._breakers.get(key)
        if breaker is None:
            breaker = MCPServerCircuitBreaker(clock=self._clock)
            self._breakers[key] = breaker
        # Fail fast (cooldown error naming the server) before any session work.
        admission = breaker.admission(server_name or " ".join(command))
        call_timeout = timedelta(seconds=tool_timeout_sec) if tool_timeout_sec else None
        try:
            conn = await self._get_or_create(
                key, command, env, cwd, startup_timeout_sec
            )
            with bind_policy(self._policy):
                result = await conn.call_tool(tool_name, arguments, call_timeout)
            parsed = parse_call_result("stdio:" + " ".join(command), tool_name, result)
        except BaseException as exc:
            if is_call_timeout(exc):
                breaker.record_timeout(admission)
            else:
                # Auth/tool/transport failures and cancellation have their own
                # handling; they never trip the breaker, only release a probe.
                breaker.record_probe_incomplete(admission)
            raise
        breaker.record_success(admission)
        return parsed

    async def aclose(self) -> None:
        self.retire()
        if self._close_task is not None:
            _, pending = await asyncio.wait(
                {self._close_task}, timeout=_CLOSE_TIMEOUT_SEC * 2
            )
            if pending:
                raise TimeoutError(
                    "MCP pool cleanup still pending; connections retained"
                )
            try:
                self._close_task.result()
            except BaseExceptionGroup as exc:
                if len(exc.exceptions) == 1 and isinstance(
                    exc.exceptions[0], TimeoutError
                ):
                    raise exc.exceptions[0] from exc
                raise

    async def _close_connections(self) -> None:
        conns = list(self._conns.items())
        outcomes = await asyncio.gather(
            *(conn._wait_closed(_CLOSE_TIMEOUT_SEC) for _, conn in conns),
            return_exceptions=True,
        )
        errors = []
        for (key, conn), outcome in zip(conns, outcomes, strict=True):
            if isinstance(outcome, BaseException):
                if isinstance(outcome, TimeoutError):
                    conn.retire()
                    worker = conn._worker
                    if worker is not None and not worker.done():
                        worker.cancel()
                        worker.add_done_callback(self._report_close_failure)
                errors.append(outcome)
            else:
                self._conns.pop(key, None)
        if errors:
            raise BaseExceptionGroup("MCP pool cleanup failed", errors)
        self._loop = None
