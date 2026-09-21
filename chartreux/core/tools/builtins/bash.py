from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from functools import lru_cache
from pathlib import Path
import shlex
from typing import final

from pydantic import BaseModel, Field, computed_field, model_validator
from tree_sitter import Language, Node, Parser
import tree_sitter_bash as tsbash

from chartreux.core.events import ToolResultEvent, ToolStreamEvent
from chartreux.core.scratchpad import is_scratchpad_path
from chartreux.core.tools.base import (
    BaseTool,
    BaseToolConfig,
    BaseToolState,
    InvokeContext,
    ToolError,
    ToolPermission,
)
from chartreux.core.tools.builtins._shell_command_policy import (
    analyze_shell_command_policy,
    matches_command_prefix,
    path_candidates,
)
from chartreux.core.tools.builtins._shell_permission_analysis import (
    analyze_shell_command,
)
from chartreux.core.tools.io_port import ShellCommandRequest
from chartreux.core.tools.permissions import PermissionContext
from chartreux.core.tools.ui import ToolCallDisplay, ToolResultDisplay, ToolUIData
from chartreux.core.tools.utils import (
    DEFAULT_SENSITIVE_PATTERNS,
    ambient_workspace,
    is_path_within_workdir,
    matches_sensitive_pattern,
    resolve_tool_path,
)
from chartreux.core.utils import kill_async_subprocess
from chartreux.core.utils.shell import spawn_shell_command
from chartreux.core.workspace import Workspace
from chartreux.utils.io import decode_console_safe
from chartreux.utils.tool_presentation import ToolEffectKind


@lru_cache(maxsize=1)
def _get_parser() -> Parser:
    return Parser(Language(tsbash.language()))


def _wrapped_guardrail_commands(command: str) -> list[str]:
    """Extract statically visible commands invoked through shell wrappers."""
    tokens = _split_command_tokens(command)
    if not tokens:
        return []
    if tokens[0] == "eval":
        evaluated = " ".join(tokens[1:])
        return list(analyze_shell_command(evaluated).command_parts) if evaluated else []
    if tokens[0] in {"command", "builtin"}:
        index = 1
        while index < len(tokens):
            if tokens[index] == "--":
                index += 1
                break
            if tokens[index].startswith("-"):
                index += 1
                continue
            break
        return [shlex.join(tokens[index:])] if index < len(tokens) else []
    if tokens[0] != "exec":
        return []
    index = 1
    while index < len(tokens):
        token = tokens[index]
        if token == "--":
            index += 1
            break
        if token == "-a":
            index += 2
            continue
        if token.startswith("-"):
            index += 1
            continue
        break
    return [" ".join(tokens[index:])] if index < len(tokens) else []


def _expand_guardrail_commands(command_parts: list[str]) -> list[str]:
    expanded: list[str] = []
    pending = list(command_parts)
    seen: set[str] = set()
    while pending:
        part = pending.pop(0)
        if part in seen:
            continue
        seen.add(part)
        expanded.append(part)
        pending.extend(_wrapped_guardrail_commands(part))
    return expanded


_READ_ONLY_COMMANDS_POSIX = [
    "basename",
    "cat",
    "comm",
    "cut",
    "date",
    "diff",
    "dirname",
    "du",
    "file",
    "find",
    "fmt",
    "fold",
    "grep",
    "head",
    "join",
    "less",
    "ls",
    "md5sum",
    "more",
    "nl",
    "od",
    "paste",
    "pwd",
    "readlink",
    "sha1sum",
    "sha256sum",
    "shasum",
    "sort",
    "stat",
    "sum",
    "tac",
    "tail",
    "tr",
    "uname",
    "uniq",
    "wc",
    "which",
]


def _get_default_denylist() -> list[str]:
    common = ["gdb", "pdb", "passwd"]
    return common + [
        "nano",
        "vim",
        "vi",
        "emacs",
        "bash -i",
        "sh -i",
        "zsh -i",
        "fish -i",
        "dash -i",
        "screen",
        "tmux",
    ]


def _get_default_denylist_standalone() -> list[str]:
    return [
        "python",
        "python3",
        "ipython",
        "bash",
        "sh",
        "nohup",
        "vi",
        "vim",
        "emacs",
        "nano",
        "su",
    ]


_MUTATING_PATH_COMMANDS = {"cd", "chmod", "chown", "cp", "mkdir", "mv", "rm", "touch"}

# Inspect recognized reader and mutator operands before permitting execution.
# This bounded heuristic is not a containment promise for arbitrary programs.
_PATH_COMMANDS = _MUTATING_PATH_COMMANDS | set(_READ_ONLY_COMMANDS_POSIX)


def _split_command_tokens(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        return command.split()


def _collect_outside_dirs(
    command_parts: list[str],
    *,
    workspace: Workspace | None = None,
    scratchpad_dir: Path | None = None,
) -> set[str]:
    """Collect parent directories referenced outside the workdir.

    Iterates file-manipulating commands (see _PATH_COMMANDS) and inspects
    their arguments as candidate paths. Skips flags (-r, --recursive) and
    chmod mode strings (+x). For any argument outside accepted Workspace roots,
    adds the parent directory (or the path itself when it is a directory).
    The return shape is retained for shared callers; Bash treats any result as
    a denial, not a request for an invocation grant.

    Only invoked under POSIX-shell semantics, where "/" is a valid path separator.
    """
    workspace = workspace or ambient_workspace()
    resolved_cwd = workspace.cwd

    def is_within_workdir(path: str) -> bool:
        return is_path_within_workdir(path, workspace=workspace)

    dirs: set[str] = set()
    for part in command_parts:
        tokens = _split_command_tokens(part)
        command = Path(tokens[0]).name if tokens else None
        if not command:
            continue
        for token in path_candidates(
            tokens, inspect_positional_paths=command in _PATH_COMMANDS
        ):
            # Bare filenames can be symlinks outside the accepted workspace too.
            if token == "<redirect>":
                continue
            path_token = token
            if is_within_workdir(path_token):
                continue
            resolved = resolve_tool_path(path_token, resolved_cwd)
            if (
                scratchpad_dir is not None
                and scratchpad_dir.resolve() == scratchpad_dir.absolute()
                and is_scratchpad_path(str(resolved), scratchpad_dir=scratchpad_dir)
                and (workspace.ceiling is None or workspace.ceiling.allows(resolved))
            ):
                continue
            parent = str(resolved) if resolved.is_dir() else str(resolved.parent)
            dirs.add(parent)
    return dirs


def _extract_redirect_paths(command: str) -> list[str]:
    """Inspect literal file redirects using the existing Bash grammar.

    Heredoc bodies, expansions and script contents remain opaque. This is an
    accident guard, not a shell interpreter or a containment guarantee.
    """
    tree = _get_parser().parse(command.encode("utf-8"))
    paths: list[str] = []

    def visit(node: Node) -> None:
        if node.type == "file_redirect":
            destination = node.child_by_field_name("destination")
            if (
                destination is not None
                and destination.type in {"word", "string", "raw_string"}
                and destination.text is not None
            ):
                tokens = _split_command_tokens(destination.text.decode("utf-8"))
                # Descriptor duplication is not filesystem access.
                duplication = any(child.type in {">&", "<&"} for child in node.children)
                if len(tokens) == 1 and not (
                    duplication and (tokens[0].isdigit() or tokens[0] == "-")
                ):
                    paths.append(tokens[0])
        for child in node.children:
            visit(child)

    visit(tree.root_node)
    return paths


def _matches_pattern(command: str, pattern: str) -> bool:
    return command == pattern or command.startswith(pattern + " ")


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
    permission: ToolPermission = ToolPermission.ASK
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
        default=["sudo"],
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

    # `model_dump` of a tool result is the `post_tool` hook payload, so dropping
    # this key outright would break hooks that read `tool_output.returncode`.
    @computed_field(description="Deprecated alias for `exit_code`.")
    @property
    def returncode(self) -> int:
        return self.exit_code


def completed_shell_result(
    *, command: str, stdout: str, stderr: str, exit_code: int, shell: str = ""
) -> CapturedShellResult:
    if exit_code != 0:
        message = f"Command failed: {command!r}\nReturn code: {exit_code}"
        if stderr:
            message += f"\nStderr: {stderr}"
        if stdout:
            message += f"\nStdout: {stdout}"
        raise ToolError(message)

    return CapturedShellResult(
        command=command, shell=shell, exit_code=exit_code, stdout=stdout, stderr=stderr
    )


class Bash(
    BaseTool[BashArgs, CapturedShellResult, BashToolConfig, BaseToolState],
    ToolUIData[BashArgs, CapturedShellResult],
):
    effect_kind = ToolEffectKind.SHELL

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

    def _find_denylist_match(self, command: str) -> str | None:
        tokens = _split_command_tokens(command)
        return next(
            (
                pattern
                for pattern in self.config.denylist
                if matches_command_prefix(tokens, _split_command_tokens(pattern))
            ),
            None,
        )

    def _is_standalone_denylisted(self, command: str) -> bool:
        tokens = _split_command_tokens(command)
        if len(tokens) != 1:
            return False
        executable = tokens[0]
        command_name = Path(executable).name
        return (
            command_name in self.config.denylist_standalone
            or executable in self.config.denylist_standalone
        )

    def _is_sensitive(self, command: str) -> bool:
        tokens = _split_command_tokens(command)
        return any(
            matches_command_prefix(tokens, _split_command_tokens(pattern))
            for pattern in self.config.sensitive_patterns
        )

    def _resolve_guardrail_permission(
        self, command_parts: list[str]
    ) -> PermissionContext | None:
        expanded = _expand_guardrail_commands(command_parts)
        for part in expanded:
            tokens = _split_command_tokens(part)
            if tokens and Path(tokens[0]).name == "env":
                return PermissionContext(
                    permission=ToolPermission.NEVER,
                    reason="Command denied: env wrapper cannot be safely inspected",
                )
            if matched := self._find_denylist_match(part):
                return PermissionContext(
                    permission=ToolPermission.NEVER,
                    reason=f"Command denied: '{part}' matches denylist pattern '{matched}'. Do not attempt to run this command.",
                )
            if self._is_standalone_denylisted(part):
                return PermissionContext(
                    permission=ToolPermission.NEVER,
                    reason=f"Command denied: '{part}' is not allowed as a standalone command. Do not attempt to run this command.",
                )
        for part in expanded:
            if self._is_sensitive(part):
                return PermissionContext(
                    permission=ToolPermission.NEVER,
                    reason="Command denied by a sensitive command rule",
                )
            if analyze_shell_command_policy(
                _split_command_tokens(part)
            ).requires_approval:
                reason = (
                    "Command denied: find execution predicates are not permitted"
                    if Path(_split_command_tokens(part)[0]).name == "find"
                    else f"Command denied: side-effecting options are not permitted: '{part}'"
                )
                return PermissionContext(permission=ToolPermission.NEVER, reason=reason)
        return None

    def _resolve_preconditions(self) -> PermissionContext | None:
        if self.config.permission == ToolPermission.NEVER:
            return PermissionContext(
                permission=ToolPermission.NEVER, reason="Tool denied: bash"
            )
        if not self.workspace.allows(self.cwd):
            return PermissionContext(
                permission=ToolPermission.NEVER,
                reason="Shell cwd is outside the authorized workspace",
            )
        return None

    def _resolve_path_permission(
        self, path_parts: list[str]
    ) -> PermissionContext | None:
        outside_dirs = _collect_outside_dirs(
            path_parts, workspace=self.workspace, scratchpad_dir=self.scratchpad_dir
        )
        for part in path_parts:
            tokens = _split_command_tokens(part)
            if not tokens:
                continue
            command = Path(tokens[0]).name
            for token in path_candidates(
                tokens, inspect_positional_paths=command in _PATH_COMMANDS
            ):
                if token == "<redirect>":
                    continue
                resolved = resolve_tool_path(token, self.workspace.cwd)
                if matches_sensitive_pattern(str(resolved), DEFAULT_SENSITIVE_PATTERNS):
                    return PermissionContext(
                        permission=ToolPermission.NEVER,
                        reason="Sensitive file access denied (bash)",
                    )
        if outside_dirs:
            return PermissionContext(
                permission=ToolPermission.NEVER,
                reason="Shell path is outside the authorized workspace; only an explicit user scope change can authorize it",
            )
        return None

    def resolve_permission(self, args: BashArgs) -> PermissionContext | None:
        precondition = self._resolve_preconditions()
        if precondition is not None:
            return precondition

        analysis = analyze_shell_command(args.command)
        command_parts = list(analysis.command_parts)
        expanded_command_parts = _expand_guardrail_commands(command_parts)
        if "shell analysis failed" in analysis.approval_reasons:
            guardrail_permission = PermissionContext(
                permission=ToolPermission.NEVER,
                reason=f"Command denied: {analysis.approval_label}",
            )
        else:
            guardrail_permission = self._resolve_guardrail_permission(command_parts)
        if guardrail_permission is not None:
            return guardrail_permission

        if any(
            _split_command_tokens(part)
            and _split_command_tokens(part)[0] in {"eval", "exec"}
            for part in expanded_command_parts
        ):
            return PermissionContext(
                permission=ToolPermission.NEVER,
                reason="Command denied: eval and exec cannot be safely inspected",
            )

        redirect_paths = _extract_redirect_paths(args.command)
        path_parts = expanded_command_parts + [
            f"cat {shlex.quote(path)}" for path in redirect_paths
        ]
        path_permission = self._resolve_path_permission(path_parts)
        if path_permission is not None:
            return path_permission

        if analysis.requires_approval:
            return PermissionContext(
                permission=ToolPermission.NEVER,
                reason=f"Command denied: {analysis.approval_label}",
            )
        return PermissionContext(permission=ToolPermission.ALWAYS)

    @final
    def _build_timeout_error(self, command: str, timeout: int) -> ToolError:
        return ToolError(f"Command timed out after {timeout}s: {command!r}")

    async def run(
        self, args: BashArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[ToolStreamEvent | CapturedShellResult, None]:
        timeout = args.timeout or self.config.default_timeout
        max_bytes = self.config.max_output_bytes

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
                    )
                )
            except TimeoutError:
                raise self._build_timeout_error(args.command, timeout) from None
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                raise ToolError(
                    f"Error running command {args.command!r}: {exc}"
                ) from exc
            yield completed_shell_result(
                command=args.command,
                stdout=result.stdout[:max_bytes],
                stderr=result.stderr[:max_bytes],
                exit_code=result.returncode,
            )
            return

        proc = None
        try:
            proc = await spawn_shell_command(args.command, cwd=self.cwd)

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
            except TimeoutError:
                await kill_async_subprocess(proc)
                raise self._build_timeout_error(args.command, timeout)

            stdout = (
                decode_console_safe(stdout_bytes)[:max_bytes] if stdout_bytes else ""
            )
            stderr = (
                decode_console_safe(stderr_bytes)[:max_bytes] if stderr_bytes else ""
            )

            yield completed_shell_result(
                command=args.command,
                stdout=stdout,
                stderr=stderr,
                exit_code=proc.returncode or 0,
            )

        except (ToolError, asyncio.CancelledError):
            raise
        except Exception as exc:
            raise ToolError(f"Error running command {args.command!r}: {exc}") from exc
        finally:
            if proc is not None:
                await kill_async_subprocess(proc)
