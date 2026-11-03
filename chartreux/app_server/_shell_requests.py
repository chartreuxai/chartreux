from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
import time
from typing import Any
from uuid import uuid4

from chartreux.app_server._dispatch import (
    DispatchResult,
    RequestFailure,
    method_not_found,
)
from chartreux.app_server._execution import SessionExecution, SessionExecutionKind
from chartreux.app_server._model import validate_wire
from chartreux.app_server._shell import (
    ShellController,
    shell_effect_cancelled,
    shell_effect_detail,
    shell_effect_error,
    shell_effect_state,
)
from chartreux.app_server._turns import TurnController
from chartreux.app_server._utils import now_ms
from chartreux.app_server.models import TextContentBlock
from chartreux.app_server.protocol import (
    ContextInjectParams,
    ProtocolErrorCode,
    SessionShellCommandParams,
    SessionShellCommandResponse,
    ShellRunParams,
    ShellRunResponse,
)
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.llm_models import ManualShellContext
from chartreux.core.tools import secret_redaction
from chartreux.core.tools.builtins._remote_exec import RemoteExecDenial
from chartreux.core.tools.builtins._shell_permission_resolver import (
    remote_exec_plan_for_command,
)
from chartreux.core.tools.builtins.bash import BashToolConfig
from chartreux.core.tools.utils import effective_scratchpad_roots
from chartreux.core.utils.shell import terminal_shell_text_refusal


class ShellRequestHandler:
    def __init__(
        self,
        agent_loop: AgentLoop,
        turns: TurnController,
        execution: SessionExecution,
        require_attached: Callable[[str], None],
        current_event_id: Callable[[str], int],
    ) -> None:
        self._agent_loop = agent_loop
        self._turns = turns
        self._execution = execution
        self._require_attached = require_attached
        self._current_event_id = current_event_id
        self._shell = ShellController(agent_loop.cwd)

    async def dispatch(self, method: str, raw_params: dict[str, Any]) -> DispatchResult:
        if method != "session/shellCommand":
            raise method_not_found(method)
        params = validate_wire(SessionShellCommandParams, raw_params)
        if params.action == "interrupt":
            response = await self._interrupt(params)
        else:
            response = await self._run(params)
        return DispatchResult(response)

    async def close(self) -> None:
        await self._shell.close()

    async def _run(
        self, params: SessionShellCommandParams
    ) -> SessionShellCommandResponse:
        self._require_attached(params.session_id)
        command = params.command or ""
        if not command.strip():
            raise RequestFailure(
                ProtocolErrorCode.INVALID_PARAMS, "Shell command cannot be empty"
            )
        # F1pkg1(a)/(c) terminal consumption contract: the session shell
        # transport is non-attesting — it runs shell text under the ambient
        # executor and no client attests one, so expansion refuses before any
        # effect starts and before the executor is reached. Non-expanding
        # text runs through the same spawn_shell_command that consumes the
        # pinned executor contract at execution in this process (_shell.py),
        # so the approved semantics equal the actual spawn.
        refusal = terminal_shell_text_refusal(command)
        if refusal is not None:
            raise RequestFailure(ProtocolErrorCode.FORBIDDEN, refusal)
        # S4b1b + S4b2a: the session shell spawns in this process, so its
        # executor is the server itself — attested by construction. A
        # standalone ssh invocation with a positive destination (literal IP,
        # or a hostname resolved uniquely non-local) runs with the pinned
        # plan (resolved address + fixed host-key identity in the actual
        # argv); every other ssh spelling hard-refuses here, before any
        # effect starts. The manual `!` path is direct user authority, so no
        # model-side grant flow applies — but the pin still lands at the spawn.
        try:
            path_authority = self._agent_loop.tool_manager.path_authority("bash")
            remote_plan = remote_exec_plan_for_command(
                command,
                cwd=self._agent_loop.cwd,
                workspace=self._agent_loop.tool_manager.workspace,
                authority=path_authority,
                scratchpad_roots=effective_scratchpad_roots(path_authority),
                workspace_roots=frozenset(
                    self._agent_loop.tool_manager.workspace.authorized_roots
                ),
            )
        except RemoteExecDenial as exc:
            raise RequestFailure(ProtocolErrorCode.FORBIDDEN, str(exc)) from exc
        if remote_plan is not None:
            command = remote_plan.command
        operation_id = params.operation_id or str(uuid4())
        run_params = ShellRunParams(
            session_id=params.session_id,
            operation_id=operation_id,
            command=command,
            timeout_seconds=params.timeout_seconds or 30.0,
            cwd=self._workspace_cwd(params.cwd),
        )
        execution = self._execution.begin(SessionExecutionKind.SHELL, operation_id)
        output: list[str] = []
        started_at = time.monotonic()
        created_at = now_ms()
        duration_ms = 0.0

        async def observe_output(chunk: str) -> None:
            output.append(chunk)
            await self._turns.append_effect_output(operation_id, chunk)

        try:
            await self._turns.start_effect(
                session_id=params.session_id,
                entry_id=operation_id,
                title="shell",
                detail=shell_effect_detail(command),
            )
            try:
                response = secret_redaction.redact_model(
                    await self._shell.run(run_params, observe_output)
                )
            except asyncio.CancelledError:
                await self._turns.complete_effect(
                    operation_id,
                    shell_effect_cancelled(
                        output_text="".join(output),
                        duration_ms=(time.monotonic() - started_at) * 1000,
                    ),
                )
                raise
            except Exception as exc:
                await self._turns.complete_effect(
                    operation_id,
                    shell_effect_error(
                        exc,
                        output_text="".join(output),
                        duration_ms=(time.monotonic() - started_at) * 1000,
                    ),
                )
                raise
            duration_ms = (time.monotonic() - started_at) * 1000
            await self._turns.complete_effect(
                operation_id,
                shell_effect_state(
                    response, output_text="".join(output), duration_ms=duration_ms
                ),
            )
        finally:
            self._execution.finish(execution)

        bash_config = self._agent_loop.tool_manager.get_tool_config("bash")
        max_output_bytes = (
            bash_config.max_output_bytes
            if isinstance(bash_config, BashToolConfig)
            else 16_000
        )
        await self._turns.inject(
            ContextInjectParams(
                session_id=params.session_id,
                input=[
                    TextContentBlock(
                        text=_manual_shell_context(
                            response, max_output_bytes=max_output_bytes
                        )
                    )
                ],
            ),
            manual_shell=ManualShellContext(
                operation_id=response.operation_id,
                command=response.command,
                cwd=response.cwd,
                stdout=response.stdout,
                stderr=response.stderr,
                output_text="".join(output),
                exit_code=response.exit_code,
                timed_out=response.timed_out,
                interrupted=response.interrupted,
                duration_ms=duration_ms,
                created_at=created_at,
            ),
        )
        return SessionShellCommandResponse(
            last_event_id=self._current_event_id(params.session_id)
        )

    async def _interrupt(
        self, params: SessionShellCommandParams
    ) -> SessionShellCommandResponse:
        self._require_attached(params.session_id)
        if params.operation_id is None:
            raise RequestFailure(
                ProtocolErrorCode.INVALID_PARAMS,
                "operation_id is required for action='interrupt'",
            )
        await self._shell.interrupt(params.operation_id)
        return SessionShellCommandResponse(
            last_event_id=self._current_event_id(params.session_id)
        )

    def _workspace_cwd(self, requested_cwd: str | None) -> str:
        root = self._agent_loop.cwd.resolve()
        cwd = Path(requested_cwd).expanduser().resolve() if requested_cwd else root
        if not cwd.is_dir():
            raise RequestFailure(
                ProtocolErrorCode.INVALID_PARAMS,
                f"Shell working directory does not exist: {cwd}",
            )
        try:
            cwd.relative_to(root)
        except ValueError as exc:
            raise RequestFailure(
                ProtocolErrorCode.FORBIDDEN,
                f"Shell working directory is outside the workspace: {cwd}",
            ) from exc
        return str(cwd)


def _manual_shell_context(result: ShellRunResponse, *, max_output_bytes: int) -> str:
    stdout = _cap_output(result.stdout, max_output_bytes)
    stderr = _cap_output(result.stderr, max_output_bytes)
    sections = [
        "Manual `!` command result from the user. Use this as context only.",
        f"Command: `{result.command}`",
        f"Working directory: `{result.cwd}`",
    ]
    if result.timed_out:
        sections.append("Status: timed out")
    elif result.interrupted:
        sections.append("Status: interrupted by user")
    else:
        sections.append(f"Exit code: {result.exit_code}")
    if stdout:
        sections.append(f"Stdout:\n```text\n{stdout.rstrip()}\n```")
    if stderr:
        sections.append(f"Stderr:\n```text\n{stderr.rstrip()}\n```")
    if not stdout and not stderr:
        sections.append("Output:\n```text\n(no output)\n```")
    return "\n\n".join(sections)


def _cap_output(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n... [truncated]"
