from __future__ import annotations

import asyncio
import os
from pathlib import Path
import shlex

from chartreux.core.hooks.config import HookConfig
from chartreux.core.hooks.models import HookExecutionResult, HookInvocation
from chartreux.core.tools.secret_redaction import scrub_child_env
from chartreux.core.utils import kill_async_subprocess
from chartreux.utils.io import decode_console_safe

_MAX_OUTPUT_BYTES = 1024 * 1024


async def _read_capped(
    stream: asyncio.StreamReader | None, limit: int = _MAX_OUTPUT_BYTES
) -> bytes:
    if stream is None:
        return b""
    chunks: list[bytes] = []
    remaining = limit
    while remaining > 0:
        chunk = await stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    while True:
        chunk = await stream.read(65536)
        if not chunk:
            break
    return b"".join(chunks)


class HookExecutor:
    def __init__(self, *, cwd: Path | None = None) -> None:
        self._cwd = (cwd or Path.cwd()).resolve()

    async def run(
        self, hook: HookConfig, invocation: HookInvocation
    ) -> HookExecutionResult:
        stdin_data = invocation.model_dump_json().encode()

        try:
            argv = shlex.split(hook.command, comments=False, posix=True)
        except ValueError as e:
            return HookExecutionResult(
                hook_name=hook.name,
                exit_code=1,
                stdout="",
                stderr=str(e),
                timed_out=False,
            )

        if not argv or not argv[0]:
            return HookExecutionResult(
                hook_name=hook.name,
                exit_code=1,
                stdout="",
                stderr="hook command is empty",
                timed_out=False,
            )

        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
                cwd=self._cwd,
                # Hooks load from project-writable config, so an injected hook
                # must not inherit chartreux's credential variables.
                env=scrub_child_env(os.environ),
            )
        except (OSError, ValueError) as e:
            return HookExecutionResult(
                hook_name=hook.name,
                exit_code=1,
                stdout="",
                stderr=f"Failed to start: {e}",
                timed_out=False,
            )

        try:
            stdin = process.stdin
            if stdin is None:
                await kill_async_subprocess(process)
                return HookExecutionResult(
                    hook_name=hook.name,
                    exit_code=1,
                    stdout="",
                    stderr="Failed to start: stdin stream unavailable",
                    timed_out=False,
                )

            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                self._run_process(process, stdin, stdin_data), timeout=hook.timeout
            )

            stdout = decode_console_safe(stdout_bytes).strip()
            stderr = decode_console_safe(stderr_bytes).strip()
            return HookExecutionResult(
                hook_name=hook.name,
                exit_code=process.returncode,
                stdout=stdout,
                stderr=stderr,
                timed_out=False,
            )
        except TimeoutError:
            await kill_async_subprocess(process)
            return HookExecutionResult(
                hook_name=hook.name,
                exit_code=None,
                stdout="",
                stderr="",
                timed_out=True,
            )
        except BaseException:
            if process.returncode is None:
                await kill_async_subprocess(process)
            raise

    async def _run_process(
        self,
        process: asyncio.subprocess.Process,
        stdin: asyncio.StreamWriter,
        stdin_data: bytes,
    ) -> tuple[bytes, bytes]:
        try:
            await self._write_stdin(stdin, stdin_data)
        except (BrokenPipeError, ConnectionResetError):
            pass

        stdout_bytes, stderr_bytes = await asyncio.gather(
            _read_capped(process.stdout), _read_capped(process.stderr)
        )
        await process.wait()
        return stdout_bytes, stderr_bytes

    async def _write_stdin(
        self, stdin: asyncio.StreamWriter, stdin_data: bytes
    ) -> None:
        stdin.write(stdin_data)
        try:
            await stdin.drain()
        finally:
            stdin.close()
