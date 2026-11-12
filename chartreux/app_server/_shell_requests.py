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
    restored_shell_effect_state,
    shell_effect_cancelled,
    shell_effect_detail,
    shell_effect_error,
    shell_effect_state,
    shell_result_status,
)
from chartreux.app_server._shell_record import (
    ManualShellOutcome,
    ManualShellRecord,
    build_manual_shell_record,
)
from chartreux.app_server._turns import TurnController
from chartreux.app_server._utils import now_ms
from chartreux.app_server.protocol import (
    ProtocolErrorCode,
    SessionShellCommandParams,
    SessionShellCommandResponse,
    ShellRunParams,
    ShellRunResponse,
    TurnShellInputEntry,
)
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.tools import secret_redaction
from chartreux.core.tools.builtins.bash import BashToolConfig


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
        # The queue owns when a queued shell input runs; this handler owns how.
        turns.bind_shell_executor(self._execute_queued_shell)

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
        cwd = self._command_cwd(params.cwd)
        operation_id = params.operation_id or str(uuid4())
        if not command.strip():
            # Item 5 B.1: the empty-input no-op is the one remaining refusal,
            # and it still produces the one record. Nothing executes, but the
            # record becomes the user's turn like any other command's.
            await self._record_empty_command(params.session_id, operation_id, cwd)
            return SessionShellCommandResponse(
                last_event_id=self._current_event_id(params.session_id)
            )
        # A manual `!` command is the user's own input, not model authority: it
        # carries no permission machinery and is never denied for being early.
        # It takes its FIFO place in the turn queue beside queued prompts; at
        # idle the queue runs it once and promotes its record as a user turn.
        completion = self._turns.enqueue_shell_input(
            session_id=params.session_id,
            operation_id=operation_id,
            command=command,
            cwd=cwd,
            timeout_seconds=params.timeout_seconds or 30.0,
        )
        try:
            await completion
        except asyncio.CancelledError:
            # The request went away before its turn: a command that never ran
            # is dropped unexecuted rather than fired later for nobody.
            self._turns.discard_queued_shell_input(operation_id)
            raise
        return SessionShellCommandResponse(
            last_event_id=self._current_event_id(params.session_id)
        )

    async def _record_empty_command(
        self, session_id: str, operation_id: str, cwd: str
    ) -> None:
        """Refuse an empty command through the one record, never a wire error.

        The no-op executes nothing, so the record is ready before the queue sees
        it; the effect tile is completed here so the live timeline matches what
        a rebuild projects from the record metadata.
        """
        await self._turns.start_effect(
            session_id=session_id,
            entry_id=operation_id,
            title="shell",
            detail=shell_effect_detail(""),
        )
        record = build_manual_shell_record(
            ManualShellOutcome(
                operation_id=operation_id,
                command="",
                cwd=cwd,
                status="refused",
                created_at=now_ms(),
                error="Shell command cannot be empty",
            ),
            max_output_bytes=self._max_output_bytes(),
        )
        await self._turns.complete_effect(
            operation_id, restored_shell_effect_state(record.context)
        )
        self._turns.enqueue_shell_record(
            session_id=session_id, operation_id=operation_id, record=record
        )

    async def _execute_queued_shell(
        self, entry: TurnShellInputEntry, session_id: str
    ) -> ManualShellRecord:
        """Run one queued command's effect; the queue owns its user turn.

        The spawn still binds the pinned executor contract, the environment is
        scrubbed, and the record is capped; every outcome reaches the record
        builder, so an infra failure or interrupt is a record, not a denial.
        """
        run_params = ShellRunParams(
            session_id=session_id,
            operation_id=entry.operation_id,
            command=entry.command,
            timeout_seconds=entry.timeout_seconds or 30.0,
            cwd=entry.cwd,
        )
        execution = self._execution.begin(
            SessionExecutionKind.SHELL, entry.operation_id
        )
        output: list[str] = []
        started_at = time.monotonic()
        created_at = now_ms()
        duration_ms = 0.0
        response: ShellRunResponse | None = None
        failure: Exception | None = None

        async def observe_output(chunk: str) -> None:
            output.append(chunk)
            await self._turns.append_effect_output(entry.operation_id, chunk)

        try:
            await self._turns.start_effect(
                session_id=session_id,
                entry_id=entry.operation_id,
                title="shell",
                detail=shell_effect_detail(entry.command),
            )
            try:
                response = secret_redaction.redact_model(
                    await self._shell.run(run_params, observe_output)
                )
            except asyncio.CancelledError:
                await self._turns.complete_effect(
                    entry.operation_id,
                    shell_effect_cancelled(
                        output_text="".join(output),
                        duration_ms=(time.monotonic() - started_at) * 1000,
                    ),
                )
                raise
            except Exception as exc:
                failure = exc
                await self._turns.complete_effect(
                    entry.operation_id,
                    shell_effect_error(
                        exc,
                        output_text="".join(output),
                        duration_ms=(time.monotonic() - started_at) * 1000,
                    ),
                )
            duration_ms = (time.monotonic() - started_at) * 1000
            if response is not None:
                await self._turns.complete_effect(
                    entry.operation_id,
                    shell_effect_state(
                        response, output_text="".join(output), duration_ms=duration_ms
                    ),
                )
        finally:
            self._execution.finish(execution)

        return build_manual_shell_record(
            _manual_shell_outcome(
                run_params,
                response=response,
                failure=failure,
                output_text="".join(output),
                duration_ms=duration_ms,
                created_at=created_at,
            ),
            max_output_bytes=self._max_output_bytes(),
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
        # Before execution an interrupt removes the queued command unexecuted;
        # once it runs, the interrupt belongs to the running shell process and
        # becomes the interrupted record its user turn carries.
        if not self._turns.discard_queued_shell_input(params.operation_id):
            await self._shell.interrupt(params.operation_id)
        return SessionShellCommandResponse(
            last_event_id=self._current_event_id(params.session_id)
        )

    def _command_cwd(self, requested_cwd: str | None) -> str:
        """Execute where the user points; only a missing directory is an error."""
        root = self._agent_loop.cwd.resolve()
        cwd = Path(requested_cwd).expanduser().resolve() if requested_cwd else root
        if not cwd.is_dir():
            raise RequestFailure(
                ProtocolErrorCode.INVALID_PARAMS,
                f"Shell working directory does not exist: {cwd}",
            )
        return str(cwd)

    def _max_output_bytes(self) -> int:
        bash_config = self._agent_loop.tool_manager.get_tool_config("bash")
        if isinstance(bash_config, BashToolConfig):
            return bash_config.max_output_bytes
        return 16_000


def _manual_shell_outcome(
    run_params: ShellRunParams,
    *,
    response: ShellRunResponse | None,
    failure: Exception | None,
    output_text: str,
    duration_ms: float,
    created_at: int,
) -> ManualShellOutcome:
    if response is None:
        return ManualShellOutcome(
            operation_id=run_params.operation_id,
            command=run_params.command,
            cwd=run_params.cwd or "",
            status="failed",
            output_text=output_text,
            duration_ms=duration_ms,
            created_at=created_at,
            error=(str(failure) or type(failure).__name__) if failure else None,
        )
    return ManualShellOutcome(
        operation_id=response.operation_id,
        command=response.command,
        cwd=response.cwd,
        status=shell_result_status(response),
        exit_code=response.exit_code,
        timed_out=response.timed_out,
        interrupted=response.interrupted,
        stdout=response.stdout,
        stderr=response.stderr,
        output_text=output_text,
        duration_ms=duration_ms,
        created_at=created_at,
    )
