from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
import json
import math
import os
import subprocess
import sys
import time
from typing import IO, Any, Literal, Protocol, cast


class InvalidJsonRpcMessage(RuntimeError):
    pass


TRANSPORT_ENV_VAR = "CHARTREUX_TRANSPORT"
IN_PROCESS_TRANSPORT = "in_process"
STDIO_TRANSPORT = "stdio"
SUPPORTED_TRANSPORT_SELECTORS = frozenset({IN_PROCESS_TRANSPORT, STDIO_TRANSPORT})
# The multi-process (stdio) route's selector spellings: the wired name plus the
# legacy experimental spelling, which spawned children and older configs may
# still carry.
MULTI_PROCESS_TRANSPORT_SELECTORS = frozenset({STDIO_TRANSPORT, "multi_process"})
# Recognized legacy spelling that no longer selects anything on its own; the
# wired multi-process route is selected by 'stdio'.
EXPERIMENTAL_TRANSPORT_SELECTORS = frozenset({"multi_process"})


class TransportSelectionError(ValueError):
    """The run's transport selection is unusable."""


class MalformedTransportSelectorError(TransportSelectionError):
    """The selector is not a transport name this installation recognizes."""


class UnsupportedTransportSelectorError(TransportSelectionError):
    """The selector names a known transport that is not wired yet."""


@dataclass(frozen=True, slots=True)
class TransportSelection:
    """The selected transport and which source selected it."""

    kind: Literal["in_process", "stdio"]
    selector: str
    source: Literal["default", "environment", "config"]


def parse_transport_selector(value: str) -> Literal["in_process", "stdio"]:
    """Validate one transport selector value.

    Diagnostics name the setting and its supported spellings but never echo
    the supplied value, which may be a mistyped secret pasted into a shell.
    """
    selector = value.strip().lower()
    if selector in SUPPORTED_TRANSPORT_SELECTORS:
        return cast(Literal["in_process", "stdio"], selector)
    if selector in EXPERIMENTAL_TRANSPORT_SELECTORS:
        raise UnsupportedTransportSelectorError(
            f"{TRANSPORT_ENV_VAR} uses the legacy multi-process spelling; the "
            "wired selector is 'stdio' (escape: 'in_process')."
        )
    raise MalformedTransportSelectorError(
        f"{TRANSPORT_ENV_VAR} must name a transport; the supported selectors "
        "are 'stdio' and 'in_process'."
    )


def resolve_transport_selection(
    *,
    multi_process: bool | None = None,
    environ: Mapping[str, str] | None = None,
    interactive: bool = False,
) -> TransportSelection:
    """Select this run's app-server transport.

    An explicit environment selector outranks the config escape key. The
    config escape (``multi_process: false``) forces the in-process route. With
    neither set, an interactive run defaults to the wired stdio route, while
    a programmatic (``-p``/``--prompt``) run stays in-process.
    """
    source = os.environ if environ is None else environ
    raw = source.get(TRANSPORT_ENV_VAR)
    if raw is not None and raw.strip():
        kind = parse_transport_selector(raw)
        return TransportSelection(kind=kind, selector=kind, source="environment")
    if multi_process is False:
        return TransportSelection(
            kind=IN_PROCESS_TRANSPORT, selector=IN_PROCESS_TRANSPORT, source="config"
        )
    if interactive:
        return TransportSelection(
            kind=STDIO_TRANSPORT, selector=STDIO_TRANSPORT, source="default"
        )
    return TransportSelection(
        kind=IN_PROCESS_TRANSPORT, selector=IN_PROCESS_TRANSPORT, source="default"
    )


class JsonRpcTransport(Protocol):
    async def send(self, message: dict[str, Any]) -> None: ...

    def messages(self) -> AsyncIterator[dict[str, Any]]: ...

    async def close(self) -> None: ...


class BinaryLineReader(Protocol):
    def readline(self, /) -> bytes: ...


class BinaryLineWriter(Protocol):
    def write(self, data: bytes, /) -> int: ...

    def flush(self, /) -> None: ...


class StdioJsonRpcTransport:
    def __init__(
        self,
        reader: BinaryLineReader,
        writer: BinaryLineWriter,
        *,
        outbox_size: int = 256,
    ) -> None:
        if outbox_size < 1:
            raise ValueError("outbox_size must be positive")
        self._reader = reader
        self._writer = writer
        self._outbox: asyncio.Queue[
            tuple[bytes, asyncio.Future[None] | None] | None
        ] = asyncio.Queue(maxsize=outbox_size)
        self._writer_task: asyncio.Task[None] | None = None
        self._close_task: asyncio.Task[None] | None = None
        self._abort_task: asyncio.Task[None] | None = None
        self._closed = False

    @classmethod
    def from_standard_streams(cls) -> StdioJsonRpcTransport:
        return cls(sys.stdin.buffer, sys.stdout.buffer)

    async def send(self, message: dict[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("JSON-RPC transport is closed")
        writer_task = self._ensure_writer()
        line = json.dumps(message, separators=(",", ":")).encode() + b"\n"
        await self._enqueue((line, None), writer_task)

    async def send_confirmed(self, message: dict[str, Any]) -> asyncio.Future[None]:
        if self._closed:
            raise RuntimeError("JSON-RPC transport is closed")
        writer_task = self._ensure_writer()
        line = json.dumps(message, separators=(",", ":")).encode() + b"\n"
        receipt: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        await self._enqueue((line, receipt), writer_task)
        return receipt

    async def messages(self) -> AsyncIterator[dict[str, Any]]:
        while raw := await asyncio.to_thread(self._reader.readline):
            yield _decode_message(raw)

    async def close(self) -> None:
        if self._close_task is None:
            self._closed = True
            self._close_task = asyncio.create_task(self._finish_close())
        await asyncio.shield(self._close_task)

    async def abortive_close(self) -> None:
        """Close the channel without draining, so the peer observes EOF now.

        This is the abortive counterpart of :meth:`close`. Frames still in the
        outbox are discarded rather than written, because a relay path that
        cannot carry its queued teardown within its bound must not let the peer
        wait for a drain that will never come.
        """
        if self._abort_task is None:
            self._closed = True
            self._abort_task = asyncio.create_task(self._finish_abortive_close())
        await asyncio.shield(self._abort_task)

    async def _finish_abortive_close(self) -> None:
        writer_task = self._writer_task
        self._discard_outbox()
        if writer_task is not None and not writer_task.done():
            writer_task.cancel()
        self._close_channel()
        if writer_task is not None:
            with suppress(asyncio.CancelledError):
                await writer_task

    def _discard_outbox(self) -> int:
        discarded = 0
        while True:
            try:
                item = self._outbox.get_nowait()
            except asyncio.QueueEmpty:
                return discarded
            if item is not None:
                discarded += 1
                receipt = item[1]
                if receipt is not None and not receipt.done():
                    receipt.cancel()

    def _close_channel(self) -> None:
        raw = getattr(self._writer, "raw", self._writer)
        close = getattr(raw, "close", None)
        if close is not None:
            with suppress(OSError):
                close()

    def _ensure_writer(self) -> asyncio.Task[None]:
        if self._writer_task is None:
            self._writer_task = asyncio.create_task(self._write_messages())
        self._check_writer(self._writer_task)
        return self._writer_task

    async def _finish_close(self) -> None:
        writer_task = self._writer_task
        if writer_task is None:
            return
        await self._enqueue(None, writer_task, allow_writer_exit=True)
        await asyncio.shield(writer_task)

    async def _enqueue(
        self,
        item: tuple[bytes, asyncio.Future[None] | None] | None,
        writer_task: asyncio.Task[None],
        *,
        allow_writer_exit: bool = False,
    ) -> None:
        self._check_writer(writer_task, allow_stopped=allow_writer_exit)
        put_task = asyncio.create_task(self._outbox.put(item))
        try:
            done, _ = await asyncio.wait(
                (put_task, writer_task), return_when=asyncio.FIRST_COMPLETED
            )
            if writer_task in done:
                self._check_writer(writer_task, allow_stopped=allow_writer_exit)
            await put_task
            self._check_writer(writer_task, allow_stopped=allow_writer_exit)
        finally:
            if not put_task.done():
                put_task.cancel()
                try:
                    await put_task
                except asyncio.CancelledError:
                    pass

    @staticmethod
    def _check_writer(
        writer_task: asyncio.Task[None], *, allow_stopped: bool = False
    ) -> None:
        if not writer_task.done():
            return
        try:
            writer_task.result()
        except asyncio.CancelledError as exc:
            raise RuntimeError("JSON-RPC writer was cancelled") from exc
        if not allow_stopped:
            raise RuntimeError("JSON-RPC writer stopped unexpectedly")

    async def _write_messages(self) -> None:
        try:
            while (item := await self._outbox.get()) is not None:
                line, receipt = item
                try:
                    await asyncio.to_thread(self._write_line, line)
                except BaseException as exc:
                    if receipt is not None and not receipt.done():
                        if isinstance(exc, asyncio.CancelledError):
                            receipt.cancel()
                        else:
                            receipt.set_exception(exc)
                    raise
                if receipt is not None and not receipt.done():
                    receipt.set_result(None)
        finally:
            self._discard_outbox()

    def _write_line(self, line: bytes) -> None:
        self._writer.write(line)
        self._writer.flush()


DEFAULT_TERMINATE_GRACE = 2.0

PROCESS_POLL_INTERVAL = 0.02
"""How often a bounded stop re-checks a process before escalating."""


@dataclass(frozen=True, slots=True)
class ProcessStopOutcome:
    """What one bounded terminate/kill → close-IPC → reap sequence achieved.

    ``reaped`` is the only field that says the process is gone; a stop that
    could not reap inside its bound reports ``reaped=False`` and the caller
    must never read that as a clean stop.
    """

    terminated: bool = False
    killed: bool = False
    ipc_closed: bool = False
    reaped: bool = False


class StoppableProcess(Protocol):
    """The process surface one bounded stop needs (``subprocess.Popen`` fits)."""

    stdin: IO[bytes] | None
    stdout: IO[bytes] | None

    def poll(self) -> int | None: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...


def signal_terminate(process: StoppableProcess) -> bool:
    """Send SIGTERM; False when the process was already gone."""
    if process.poll() is not None:
        return False
    with suppress(OSError):
        process.terminate()
        return True
    return False


def signal_kill(process: StoppableProcess) -> bool:
    """Send SIGKILL; False when the process was already gone."""
    if process.poll() is not None:
        return False
    with suppress(OSError):
        process.kill()
        return True
    return False


def close_process_ipc(process: StoppableProcess) -> bool:
    """Close the child's standard pipes; False when it had none to close."""
    return _close_pipe(process.stdin) or _close_pipe(process.stdout)


def process_exited_within(
    process: StoppableProcess,
    *,
    deadline: float | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    """Poll until the process exits or ``deadline`` passes on the clock.

    Polling one interval at a time (never a blocking ``wait``) keeps every
    stop caller bounded: an expired deadline returns the current liveness
    instead of waiting for a child that may never exit.
    """
    while process.poll() is None:
        if deadline is not None:
            remaining = deadline - clock()
            if remaining <= 0:
                return False
            sleep(min(PROCESS_POLL_INTERVAL, remaining))
        else:
            sleep(PROCESS_POLL_INTERVAL)
    return True


def stop_process(
    process: StoppableProcess,
    *,
    grace: float = DEFAULT_TERMINATE_GRACE,
    deadline: float | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> ProcessStopOutcome:
    """Stop one child: SIGTERM (grace then kill) → close IPC → reap.

    Closing stdin first is what gives the child its EOF drain window; when it
    does not exit inside ``grace`` it is signalled, then killed, and only then
    are the remaining pipes closed. Every wait is bounded by ``deadline`` (or
    by ``grace`` when no total bound is supplied), so a wedged child is
    reported unreaped instead of blocking the shutdown forever.
    """
    _close_pipe(process.stdin)
    exited = process_exited_within(
        process,
        deadline=_min_deadline(clock() + grace, deadline),
        clock=clock,
        sleep=sleep,
    )
    terminated = False
    killed = False
    if not exited:
        terminated = signal_terminate(process)
        exited = process_exited_within(
            process,
            deadline=_min_deadline(clock() + grace, deadline),
            clock=clock,
            sleep=sleep,
        )
    if not exited:
        killed = signal_kill(process)
        exited = process_exited_within(
            process,
            deadline=_min_deadline(clock() + grace, deadline),
            clock=clock,
            sleep=sleep,
        )
    _close_pipe(process.stdout)
    return ProcessStopOutcome(
        terminated=terminated, killed=killed, ipc_closed=True, reaped=exited
    )


def _min_deadline(first: float, second: float | None) -> float:
    return first if second is None else min(first, second)


class ProcessPipeJsonRpcTransport(StdioJsonRpcTransport):
    """JSON-RPC over a spawned child process's standard pipes.

    Draining close is inherited unchanged; once the outbox has drained this
    adapter closes the child's stdin so the child observes EOF, then stops it
    with one bounded SIGTERM→SIGKILL escalation, closes the remaining pipes,
    and reaps it. The typed outcome records what the stop actually achieved.
    """

    def __init__(
        self,
        reader: BinaryLineReader,
        writer: BinaryLineWriter,
        *,
        process: subprocess.Popen[bytes] | None = None,
        terminate_grace: float = DEFAULT_TERMINATE_GRACE,
        outbox_size: int = 256,
    ) -> None:
        super().__init__(reader, writer, outbox_size=outbox_size)
        try:
            valid_grace = math.isfinite(terminate_grace) and terminate_grace > 0
        except TypeError as exc:
            raise ValueError(
                "terminate_grace must be a finite positive number"
            ) from exc
        if not valid_grace:
            raise ValueError("terminate_grace must be a finite positive number")
        self._process = process
        self._terminate_grace = terminate_grace
        self._process_shutdown = False
        self._stop_outcome: ProcessStopOutcome | None = None

    @property
    def stop_outcome(self) -> ProcessStopOutcome | None:
        """What the bounded stop achieved; None until the process is stopped."""
        return self._stop_outcome

    async def close(self) -> None:
        try:
            await super().close()
        finally:
            if not self._process_shutdown:
                self._process_shutdown = True
                await asyncio.to_thread(self._shutdown_child)

    async def abortive_close(self) -> None:
        """Discard the outbox, close the child's stdin, then reap it.

        Closing the pipe is what gives the child EOF; reaping shares the
        draining close's terminate-and-wait grace, so an abort never waits on
        the outbox it just discarded.
        """
        try:
            await super().abortive_close()
        finally:
            if not self._process_shutdown:
                self._process_shutdown = True
                await asyncio.to_thread(self._abort_child)

    def _abort_child(self) -> None:
        if self._process is not None:
            with suppress(OSError):
                self._process.terminate()
        self._shutdown_child()

    def _shutdown_child(self) -> None:
        if self._process is not None:
            self._stop_outcome = terminate_process(
                self._process, grace=self._terminate_grace
            )


def terminate_process(
    process: subprocess.Popen[bytes], *, grace: float = DEFAULT_TERMINATE_GRACE
) -> ProcessStopOutcome:
    """Close the child's pipes, then terminate and reap it within ``grace``."""
    return stop_process(process, grace=grace)


def _close_pipe(stream: IO[bytes] | None) -> bool:
    if stream is None:
        return False
    with suppress(OSError):
        stream.close()
        return True
    return False


class MemoryJsonRpcTransport:
    def __init__(
        self,
        incoming: asyncio.Queue[str | None],
        outgoing: asyncio.Queue[str | None],
        *,
        incoming_closed: asyncio.Event | None = None,
        outgoing_closed: asyncio.Event | None = None,
    ) -> None:
        self._incoming = incoming
        self._outgoing = outgoing
        self._incoming_closed = incoming_closed or asyncio.Event()
        self._outgoing_closed = outgoing_closed or asyncio.Event()
        self._closed = False

    async def send(self, message: dict[str, Any]) -> None:
        if self._closed or self._outgoing_closed.is_set():
            raise RuntimeError("JSON-RPC transport is closed")
        put = asyncio.create_task(
            self._outgoing.put(json.dumps(message, separators=(",", ":")))
        )
        closed = asyncio.create_task(self._outgoing_closed.wait())
        try:
            done, _ = await asyncio.wait(
                (put, closed), return_when=asyncio.FIRST_COMPLETED
            )
            if closed in done and self._outgoing_closed.is_set() and not put.done():
                raise RuntimeError("JSON-RPC transport is closed")
            await put
        finally:
            for task in (put, closed):
                if not task.done():
                    task.cancel()
            await asyncio.gather(put, closed, return_exceptions=True)

    async def messages(self) -> AsyncIterator[dict[str, Any]]:
        while True:
            try:
                raw = self._incoming.get_nowait()
            except asyncio.QueueEmpty:
                if self._incoming_closed.is_set():
                    return
                receive = asyncio.create_task(self._incoming.get())
                closed = asyncio.create_task(self._incoming_closed.wait())
                try:
                    done, _ = await asyncio.wait(
                        (receive, closed), return_when=asyncio.FIRST_COMPLETED
                    )
                    if receive not in done:
                        return
                    raw = receive.result()
                finally:
                    for task in (receive, closed):
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(receive, closed, return_exceptions=True)
            if raw is None:
                return
            yield _decode_message(raw)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._incoming_closed.set()
        self._outgoing_closed.set()


def memory_transport_pair() -> tuple[MemoryJsonRpcTransport, MemoryJsonRpcTransport]:
    client_incoming: asyncio.Queue[str | None] = asyncio.Queue(maxsize=256)
    server_incoming: asyncio.Queue[str | None] = asyncio.Queue(maxsize=256)
    client_closed = asyncio.Event()
    server_closed = asyncio.Event()
    return (
        MemoryJsonRpcTransport(
            client_incoming,
            server_incoming,
            incoming_closed=server_closed,
            outgoing_closed=client_closed,
        ),
        MemoryJsonRpcTransport(
            server_incoming,
            client_incoming,
            incoming_closed=client_closed,
            outgoing_closed=server_closed,
        ),
    )


def _decode_message(raw: str | bytes) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise InvalidJsonRpcMessage("Malformed JSON-RPC message") from exc
    if not isinstance(value, dict):
        raise InvalidJsonRpcMessage("JSON-RPC messages must be objects")
    return value
