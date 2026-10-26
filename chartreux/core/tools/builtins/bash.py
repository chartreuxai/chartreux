from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import final

from pydantic import BaseModel, Field, computed_field, model_validator

from chartreux.core.env import project_env_for_cwd
from chartreux.core.events import ToolResultEvent, ToolStreamEvent
from chartreux.core.tools.base import (
    BaseTool,
    BaseToolConfig,
    BaseToolState,
    InvokeContext,
    ToolError,
    ToolPermission,
)
from chartreux.core.tools.builtins._shell_permission_resolver import (
    ShellPermissionResolver,
    _analyze_guardrail_source as _analyze_guardrail_source,
    _collect_outside_dirs as _collect_outside_dirs,
    _denied as _denied,
    _expand_guardrail_commands as _expand_guardrail_commands,
    _expand_guardrail_parts as _expand_guardrail_parts,
    _get_default_denylist as _get_default_denylist,
    _get_default_denylist_standalone as _get_default_denylist_standalone,
    _get_parser as _get_parser,
    _GuardrailPart as _GuardrailPart,
    _has_unrecognized_nested_shell as _has_unrecognized_nested_shell,
    _matches_pattern as _matches_pattern,
    _operand_access as _operand_access,
    _operand_guardrail_cwds as _operand_guardrail_cwds,
    _scoped_guardrail_cwds as _scoped_guardrail_cwds,
    _split_command_tokens as _split_command_tokens,
    _UnsafeShellSyntax as _UnsafeShellSyntax,
    _update_guardrail_cwds as _update_guardrail_cwds,
    shell_command_write_paths,
)
from chartreux.core.tools.io_port import ShellCommandRequest
from chartreux.core.tools.permissions import PermissionContext
from chartreux.core.tools.ui import ToolCallDisplay, ToolResultDisplay, ToolUIData
from chartreux.core.tools.utils import (
    DEFAULT_SENSITIVE_PATTERNS,
    PathAuthority as PathAuthority,
)
from chartreux.core.utils import kill_async_subprocess
from chartreux.core.utils.shell import spawn_shell_command
from chartreux.utils.io import decode_console_safe
from chartreux.utils.tool_presentation import ToolEffectKind


class BashToolConfig(BaseToolConfig):
    @model_validator(mode="before")
    @classmethod
    def _reject_removed_allowlist(cls, value: object) -> object:
        if isinstance(value, dict) and "allowlist" in value:
            raise ValueError(
                "[tools.bash].allowlist was removed in v0.1; remove this key. "
                "The shell resolver's hard guards are the policy."
            )
        return value

    allowlist: list[str] = Field(
        default_factory=list,
        exclude=True,
        description="Removed legacy field; excluded so internal defaults cannot reintroduce it.",
    )
    permission: ToolPermission = ToolPermission.ALWAYS
    max_output_bytes: int = Field(
        default=16_000, description="Maximum bytes to capture from stdout and stderr."
    )
    default_timeout: int = Field(
        default=300, description="Default timeout for commands in seconds."
    )
    denylist: list[str] = Field(
        default_factory=_get_default_denylist,
        description="Command prefixes that are automatically denied",
    )
    denylist_standalone: list[str] = Field(
        default_factory=_get_default_denylist_standalone,
        description="Commands that are denied only when run without arguments",
    )
    sensitive_patterns: list[str] = Field(
        # v0.5.9 P2-d: sudo left the shipped sensitive prefixes — the
        # `sudo -u <user>` wrapper is modeled transparently in the executor
        # grammar (the wrapped command is re-analyzed), and every other
        # sudo form fails closed there. A user-configured "sudo" pattern
        # still denies the wrapper's outer command outright.
        default=[],
        description="Command prefixes denied before other resolver checks.",
    )


class BashArgs(BaseModel):
    command: str = Field(description="The shell command to execute")
    timeout: int | None = Field(
        default=None, description="Override the default command timeout."
    )


class CapturedShellResult(BaseModel):
    """Result of a shell that captures stdout and stderr as two separate pipes."""

    command: str
    shell: str = ""
    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""
    # P2-i R6: the resolved paths the command's write-classified operands
    # name (redirect targets, write-option values, write-classified command
    # operands), surfaced so the submission secret scan collects
    # bash-written artifacts alongside file-tool paths. Excluded from
    # `model_dump`: the post-tool hook payload and the persisted transcript
    # keep their shape; the accumulator reads the attributes on the
    # in-process result.
    write_paths: list[str] = Field(default_factory=list, exclude=True)
    # True when the command's write surface could not be established; the
    # submission gate treats an unknown footprint as unscannable content.
    write_paths_unknown: bool = Field(default=False, exclude=True)

    # `model_dump` of a tool result is the `post_tool` hook payload, so dropping
    # this key outright would break hooks that read `tool_output.returncode`.
    @computed_field(description="Deprecated alias for `exit_code`.")
    @property
    def returncode(self) -> int:
        return self.exit_code


class FailedShellCommand(ToolError):
    """ToolError from a shell command that failed or timed out.

    A failed run may still have written files (partial execution), so it
    carries the command's write-classified paths and its unknown-write-
    surface flag; the tool executor copies them onto the error result
    event, and the submission scan collects the footprint exactly as a
    successful run's would be.
    """

    def __init__(
        self,
        message: str,
        *,
        write_paths: list[str] | None = None,
        write_paths_unknown: bool = False,
    ) -> None:
        super().__init__(message)
        self.write_paths = list(write_paths or [])
        self.write_paths_unknown = write_paths_unknown


class CancelledShellCommand(asyncio.CancelledError):
    """CancelledError from a shell run cancelled mid-flight.

    A cancelled run may still have written files (partial execution), so
    it carries the command's write-classified paths and its unknown-write-
    surface flag; the tool executor copies them onto the cancelled result
    event, and the submission scan collects the footprint exactly as a
    successful run's would be.
    """

    def __init__(
        self, *, write_paths: list[str] | None = None, write_paths_unknown: bool = False
    ) -> None:
        super().__init__()
        self.write_paths = list(write_paths or [])
        self.write_paths_unknown = write_paths_unknown


def completed_shell_result(
    *,
    command: str,
    stdout: str,
    stderr: str,
    exit_code: int,
    shell: str = "",
    write_paths: list[str] | None = None,
    write_paths_unknown: bool = False,
) -> CapturedShellResult:
    if exit_code != 0:
        message = f"Command failed: {command!r}\nReturn code: {exit_code}"
        if stderr:
            message += f"\nStderr: {stderr}"
        if stdout:
            message += f"\nStdout: {stdout}"
        raise FailedShellCommand(
            message, write_paths=write_paths, write_paths_unknown=write_paths_unknown
        )

    return CapturedShellResult(
        command=command,
        shell=shell,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        write_paths=list(write_paths or []),
        write_paths_unknown=write_paths_unknown,
    )


def _shell_output_byte_limit(max_chars: int) -> int:
    # UTF-32 can use four bytes per displayed character; retain the BOM too.
    return max(0, max_chars) * 4 + 4


async def _drain_shell_stream(
    stream: asyncio.StreamReader | None, max_chars: int
) -> bytes:
    if stream is None:
        return b""
    limit = _shell_output_byte_limit(max_chars)
    collected = bytearray()
    while chunk := await stream.read(64 * 1024):
        if len(collected) < limit:
            collected.extend(chunk[: limit - len(collected)])
    return bytes(collected)


class Bash(
    BaseTool[BashArgs, CapturedShellResult, BashToolConfig, BaseToolState],
    ToolUIData[BashArgs, CapturedShellResult],
):
    effect_kind = ToolEffectKind.SHELL

    @classmethod
    def path_sensitive_patterns(cls, config: BaseToolConfig) -> tuple[str, ...]:
        # sensitive_patterns in Bash config is command-prefix policy, not globs.
        return tuple(DEFAULT_SENSITIVE_PATTERNS)

    @classmethod
    def format_call_display(cls, args: BashArgs) -> ToolCallDisplay:
        return ToolCallDisplay(
            summary=f"bash: {args.command}",
            verb="Running",
            message=args.command,
            settled_verb="Ran",
            settled_message=args.command,
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, CapturedShellResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )

        return ToolResultDisplay(success=True, verb="Ran", message=event.result.command)

    @classmethod
    def get_status_text(cls) -> str:
        return "Running command"

    def resolve_permission(self, args: BashArgs) -> PermissionContext | None:
        return ShellPermissionResolver(
            self.config,
            self.cwd,
            self.workspace,
            self.path_authority,
            self.dangerous_command_authorities_getter,
            # The same trust-gated environment run() injects into the
            # subprocess; the egress rule consults it for injected
            # non-loopback proxies (v0.5.9 R1).
            injected_env=project_env_for_cwd(self.cwd),
        ).resolve_permission(args)

    @final
    def _build_timeout_error(
        self,
        command: str,
        timeout: int,
        write_paths: list[str] | None = None,
        write_paths_unknown: bool = False,
    ) -> ToolError:
        # R7: a timed-out run may still have written files before the kill,
        # so the error carries the run's write footprint for the submission
        # scan, surfaced exactly as a successful run's would be.
        return FailedShellCommand(
            f"Command timed out after {timeout}s: {command!r}",
            write_paths=write_paths,
            write_paths_unknown=write_paths_unknown,
        )

    async def run(
        self, args: BashArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[ToolStreamEvent | CapturedShellResult, None]:
        timeout = args.timeout or self.config.default_timeout
        max_bytes = self.config.max_output_bytes
        # The project environment loads when the working folder is inside a
        # trusted folder: the trusted folder's .env plus $CHARTREUX_HOME/.env,
        # re-read on mtime change. Its values are masked by the redaction
        # backstop, so scripts find keys via os.environ without the model
        # ever seeing them.
        project_env = project_env_for_cwd(self.cwd)
        # P2-i R6: surface the command's write-classified paths (redirect
        # targets, write-option values, write-classified command operands)
        # onto the result, so the submission secret scan collects
        # bash-written artifacts alongside file-tool paths.
        write_paths, write_paths_unknown = shell_command_write_paths(
            args.command,
            cwd=self.cwd,
            workspace=self.workspace,
            authority=self.path_authority,
        )

        if (
            ctx is not None
            and ctx.tool_io is not None
            and ctx.tool_io.supports_terminal
            and ctx.session_id is not None
        ):
            try:
                result = await ctx.tool_io.run_shell(
                    ShellCommandRequest(
                        session_id=ctx.session_id,
                        tool_call_id=ctx.tool_call_id,
                        command=args.command,
                        cwd=self.cwd,
                        timeout=timeout,
                        max_output_bytes=max_bytes,
                        env=project_env or None,
                    )
                )
            except TimeoutError:
                raise self._build_timeout_error(
                    args.command,
                    timeout,
                    write_paths=write_paths,
                    write_paths_unknown=write_paths_unknown,
                ) from None
            except asyncio.CancelledError as exc:
                # A cancelled run may still have written files; the executor's
                # cancellation handler copies the run's write footprint off
                # the cancellation onto the cancelled result event.
                raise CancelledShellCommand(
                    write_paths=write_paths, write_paths_unknown=write_paths_unknown
                ) from exc
            except Exception as exc:
                raise ToolError(
                    f"Error running command {args.command!r}: {exc}"
                ) from exc
            yield completed_shell_result(
                command=args.command,
                stdout=result.stdout[:max_bytes],
                stderr=result.stderr[:max_bytes],
                exit_code=result.returncode,
                write_paths=write_paths,
                write_paths_unknown=write_paths_unknown,
            )
            return

        proc = None
        completed = False
        try:
            proc = await spawn_shell_command(
                args.command, cwd=self.cwd, extra_env=project_env
            )

            try:
                stdout_bytes, stderr_bytes, _ = await asyncio.wait_for(
                    asyncio.gather(
                        _drain_shell_stream(proc.stdout, max_bytes),
                        _drain_shell_stream(proc.stderr, max_bytes),
                        proc.wait(),
                    ),
                    timeout=timeout,
                )
            except TimeoutError:
                await kill_async_subprocess(proc)
                raise self._build_timeout_error(
                    args.command,
                    timeout,
                    write_paths=write_paths,
                    write_paths_unknown=write_paths_unknown,
                )

            stdout, stderr = await asyncio.gather(
                asyncio.to_thread(decode_console_safe, stdout_bytes),
                asyncio.to_thread(decode_console_safe, stderr_bytes),
            )
            stdout = stdout[:max_bytes]
            stderr = stderr[:max_bytes]

            completed = True
            yield completed_shell_result(
                command=args.command,
                stdout=stdout,
                stderr=stderr,
                exit_code=proc.returncode or 0,
                write_paths=write_paths,
                write_paths_unknown=write_paths_unknown,
            )

        except asyncio.CancelledError as exc:
            # Same footprint retention as the tool_io path above: the
            # executor's cancellation handler copies the run's write
            # footprint off the cancellation onto the result event.
            raise CancelledShellCommand(
                write_paths=write_paths, write_paths_unknown=write_paths_unknown
            ) from exc
        except ToolError:
            raise
        except Exception as exc:
            raise ToolError(f"Error running command {args.command!r}: {exc}") from exc
        finally:
            if proc is not None:
                await kill_async_subprocess(
                    proc, kill_exited_process_group=not completed
                )
