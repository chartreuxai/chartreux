from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Iterator
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re
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
    ShellCommandPolicy,
    analyze_shell_command_policy,
    git_repository_config_risk,
    inline_interpreter_switch,
    matches_command_prefix,
    path_candidates,
)
from chartreux.core.tools.builtins._shell_permission_analysis import (
    DANGEROUS_ENV_NAMES,
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


_SHELLS = frozenset({"ash", "bash", "sh", "dash", "hush", "zsh"})
_BUSYBOX_SHELL_APPLETS = frozenset({"sh", "ash", "bash", "dash", "hush"})
_MAX_SHELL_DEPTH = 8
_MAX_EXPANDED_COMMANDS = 256
_MAX_ANALYZED_BYTES = 64 * 1024


class _UnsafeShellSyntax(ValueError):
    pass


@dataclass(frozen=True)
class _GuardrailPart:
    text: str
    scope: tuple[int, ...] = ()


def _shell_source(tokens: list[str]) -> str | None:
    """Locate a literal -c operand using a deliberately narrow shell argv grammar."""
    if not tokens or Path(tokens[0]).name not in _SHELLS:
        return None
    index = 1
    while index < len(tokens):
        token = tokens[index]
        if token == "--":
            if any(candidate == "-c" for candidate in tokens[index + 1 :]):
                raise _UnsafeShellSyntax("unsupported shell -- -c option form")
            index += 1
            break
        if token == "-c":
            if index + 1 >= len(tokens):
                raise _UnsafeShellSyntax("shell -c has no literal source")
            return tokens[index + 1]
        if (
            token.startswith("-")
            and not token.startswith("--")
            and len(token) > len("-c")
        ):
            flags = token[1:]
            if "c" in flags:
                if flags[-1] != "c" or index + 1 >= len(tokens):
                    raise _UnsafeShellSyntax("unsupported shell -c option form")
                return tokens[index + 1]
            if set(flags) <= set("ilrsuvxenfC"):
                index += 1
                continue
        if token in {
            "-i",
            "-l",
            "-r",
            "-s",
            "-u",
            "-v",
            "-x",
            "-e",
            "-n",
            "-f",
            "-C",
            "--noprofile",
            "--norc",
            "--posix",
            "--login",
            "--interactive",
            "--noediting",
        }:
            index += 1
            continue
        if token in {"-O", "+O", "-o", "+o", "--rcfile", "--init-file"}:
            index += 2
            continue
        if token.startswith("-"):
            raise _UnsafeShellSyntax("unsupported shell option before -c")
        break
    return None


def _wrapper_executable(tokens: list[str], name: str) -> list[str]:
    """Locate the executable using only modeled wrapper argv forms."""
    values = {
        "timeout": {"-s", "--signal", "-k", "--kill-after"},
        "stdbuf": {"-i", "-o", "-e"},
        "nice": {"-n", "--adjustment"},
        "ionice": {"-c", "--class", "-n", "--classdata"},
        "taskset": {"-c", "--cpu-list"},
        "time": {"-f", "--format", "-o", "--output"},
        "flock": {"-w", "--wait", "-E", "--conflict-exit-code"},
    }.get(name, set())
    flags = {
        "nohup": set(),
        "setsid": {"-c", "-f", "-w", "--ctty", "--fork", "--wait"},
        "nice": set(),
        "ionice": {"-t", "--ignore", "-p", "-P", "-u"},
        "taskset": {"-a", "--all-tasks", "-p", "--pid"},
        "time": {
            "-a",
            "--append",
            "-p",
            "--portability",
            "-v",
            "--verbose",
            "-q",
            "--quiet",
        },
        "flock": {
            "-s",
            "--shared",
            "-x",
            "--exclusive",
            "-u",
            "--unlock",
            "-n",
            "--nonblock",
            "-o",
            "--close",
            "-F",
            "--no-fork",
        },
    }.get(name, set())
    index = 1
    while index < len(tokens):
        token = tokens[index]
        if token == "--":
            index += 1
            break
        if token in values:
            if index + 1 >= len(tokens):
                raise _UnsafeShellSyntax(f"missing {name} wrapper option value")
            index += 2
            continue
        if token in flags or (name == "nice" and re.fullmatch(r"-\d+", token)):
            index += 1
            continue
        if name == "stdbuf" and re.fullmatch(r"-[ioe].+", token):
            index += 1
            continue
        if name in {"nice", "ionice", "taskset", "time", "flock"} and any(
            token.startswith(option) and len(token) > len(option)
            for option in values
            if len(option) == len("-n") and option.startswith("-")
        ):
            index += 1
            continue
        if any(
            token.startswith(option + "=")
            for option in values
            if option.startswith("--")
        ):
            index += 1
            continue
        if token.startswith("-"):
            raise _UnsafeShellSyntax(f"unsupported {name} wrapper option")
        break
    if name == "timeout":
        index += 1  # duration
    if (
        name == "taskset"
        and index < len(tokens)
        and re.fullmatch(r"(?:0[xX])?[0-9a-fA-F,]+", tokens[index])
    ):
        index += 1  # CPU affinity mask
    if name == "flock":
        index += 1  # lock file or descriptor
    if index >= len(tokens):
        raise _UnsafeShellSyntax(f"missing {name} wrapped executable")
    return [shlex.join(tokens[index:])]


def _has_unrecognized_nested_shell(tokens: list[str]) -> bool:
    """Detect shell -c behind an unmodeled executable (including script -c)."""
    return any(
        re.search(r"(?:^|\s)(?:ash|bash|sh|dash|zsh)\s+-[a-zA-Z]*c(?:\s|$)", token)
        for token in tokens[1:]
    ) or any(
        Path(token).name in _SHELLS
        and any(
            option == "-c" or re.fullmatch(r"-[a-zA-Z]*c", option)
            for option in tokens[index + 1 :]
        )
        for index, token in enumerate(tokens[1:], 1)
    )


def _wrapped_guardrail_commands(command: str) -> tuple[list[str], str | None]:
    """Return actual executable positions, and literal nested shell source."""
    try:
        tokens = shlex.split(command)
    except ValueError as exc:
        raise _UnsafeShellSyntax("malformed shell quoting") from exc
    if not tokens:
        return [], None
    name = Path(tokens[0]).name
    if name == "eval":
        source = " ".join(tokens[1:])
        return list(analyze_shell_command(source).command_parts) if source else [], None
    if name in {"command", "builtin", "exec"}:
        index = 1
        while index < len(tokens):
            token = tokens[index]
            if token == "--":
                index += 1
                break
            if name == "exec" and token == "-a":
                index += 2
                continue
            if token.startswith("-"):
                index += 1
                continue
            break
        return ([shlex.join(tokens[index:])] if index < len(tokens) else []), None
    if name == "script" and "-c" in tokens[1:]:
        raise _UnsafeShellSyntax("unsupported script -c shell wrapper")
    if name == "busybox":
        if not tokens[1:] or tokens[1] not in _BUSYBOX_SHELL_APPLETS:
            raise _UnsafeShellSyntax("unsupported busybox applet")
        # The first operand selects the applet; parse its shell argv and source
        # exactly as a direct shell invocation, subject to the same limits.
        return [shlex.join(tokens[1:])], None
    if name in {
        "nohup",
        "setsid",
        "stdbuf",
        "timeout",
        "nice",
        "ionice",
        "taskset",
        "time",
        "flock",
    }:
        return _wrapper_executable(tokens, name), None
    if name not in _SHELLS and _has_unrecognized_nested_shell(tokens):
        raise _UnsafeShellSyntax(f"unsupported wrapper before shell -c ({name})")
    return [], _shell_source(tokens)


def _expand_guardrail_parts(
    command_parts: list[str],
) -> tuple[list[_GuardrailPart], list[str], list[str]]:
    """Expand in occurrence order, preserving child-shell cwd isolation."""
    expanded: list[_GuardrailPart] = []
    redirects: list[str] = []
    errors: list[str] = []
    byte_count = 0
    next_scope = 0

    def visit(part: str, scope: tuple[int, ...], depth: int) -> None:
        nonlocal byte_count, next_scope
        byte_count += len(part.encode("utf-8"))
        if byte_count > _MAX_ANALYZED_BYTES or len(expanded) >= _MAX_EXPANDED_COMMANDS:
            raise _UnsafeShellSyntax(
                "shell analysis budget exceeded (bytes or commands)"
            )
        expanded.append(_GuardrailPart(part, scope))
        wrapped, source = _wrapped_guardrail_commands(part)
        if source is not None:
            if depth >= _MAX_SHELL_DEPTH:
                raise _UnsafeShellSyntax("shell nesting depth limit exceeded")
            # Quoted source is parsed afresh: shlex cannot model substitutions.
            analysis = analyze_shell_command(source, nested_source=True)
            if analysis.approval_reasons:
                errors.extend(analysis.approval_reasons)
            redirects.extend(_extract_redirect_paths(source))
            next_scope += 1
            child_scope = (*scope, next_scope)
            for child in analysis.command_parts:
                visit(child, child_scope, depth + 1)
        for child in wrapped:
            if depth >= _MAX_SHELL_DEPTH:
                raise _UnsafeShellSyntax("shell nesting depth limit exceeded")
            # Wrapper expansion introduces an executable position absent from
            # the original AST; inspect its lookup/startup mutations as well.
            errors.extend(
                reason
                for reason in analyze_shell_command(child).approval_reasons
                if "command lookup modification" in reason
                or "dangerous environment assignment" in reason
            )
            visit(child, scope, depth + 1)

    for part in command_parts:
        visit(part, (), 0)
    return expanded, redirects, errors


def _expand_guardrail_commands(command_parts: list[str]) -> list[str]:
    return [part.text for part in _expand_guardrail_parts(command_parts)[0]]


# POSIX shell builtins that change the working directory for later commands.
# PowerShell's push-location/set-location aliases have no POSIX counterpart.
_WORKING_DIRECTORY_COMMANDS = {"cd", "pushd"}
_WORKING_DIRECTORY_POP_COMMANDS = {"popd"}
_WORKING_DIRECTORY_TOKEN_COUNT = 2
_SHELL_GLOB_CHARACTERS = frozenset("*?[")


def _update_guardrail_cwds(tokens: list[str], possible_cwds: set[Path]) -> bool:
    """Track every statically possible cwd; return whether it became unknown.

    The set is monotonic to account for uncertain command success and branches.
    """
    if not tokens:
        return False
    command = Path(tokens[0]).name
    if command in _WORKING_DIRECTORY_POP_COMMANDS:
        # The cwd set is monotonic, so a plain pop can only return to a location
        # already recorded by an earlier push. Named/option-bearing stacks are
        # not statically knowable and therefore fail closed.
        return len(tokens) != 1
    if command not in _WORKING_DIRECTORY_COMMANDS:
        return False
    if command == "pushd" and len(tokens) == 1:
        # Swapping an existing directory stack cannot introduce a new path.
        return False
    if (
        len(tokens) != _WORKING_DIRECTORY_TOKEN_COUNT
        or tokens[1].startswith("-")
        or any(character in tokens[1] for character in _SHELL_GLOB_CHARACTERS)
    ):
        return True
    possible_cwds.update(
        resolve_tool_path(tokens[1], cwd) for cwd in tuple(possible_cwds)
    )
    return False


def _scoped_guardrail_cwds(
    expanded: list[_GuardrailPart], cwd: Path
) -> Iterator[tuple[_GuardrailPart, list[str], set[Path], bool]]:
    """Traverse ordered commands with child shells isolated from their parent."""
    scope_cwds: dict[tuple[int, ...], set[Path]] = {(): {cwd}}
    scope_unknown: dict[tuple[int, ...], bool] = {(): False}
    for entry in expanded:
        scope = entry.scope
        if scope not in scope_cwds:
            parent = scope[:-1]
            scope_cwds[scope] = set(scope_cwds[parent])
            scope_unknown[scope] = scope_unknown[parent]
        possible_cwds = scope_cwds[scope]
        tokens = _split_command_tokens(entry.text)
        scope_unknown[scope] |= _update_guardrail_cwds(tokens, possible_cwds)
        yield entry, tokens, possible_cwds, scope_unknown[scope]


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
        # Network-client speed bumps; denylist and path policy control known
        # network exfiltration routes (redaction cannot observe pipe/socket
        # transit). These controls harden against known payload shapes, not
        # complete egress containment.
        "curl",
        "wget",
        "nc",
        "ncat",
        "socat",
        # Inline interpreter code. Enumerated per executable because prefix
        # matching normalizes only the executable token, so a "python -c"
        # pattern does not match "python3 -c".
        "python -c",
        "python3 -c",
        "pypy -c",
        "pypy3 -c",
        "node -e",
        "perl -e",
        "ruby -e",
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

# File-content readers beyond the read-only set: dumpers, encoders, and
# archivers whose positional operands name files. Inspected so the
# sensitive-file deny fires on operands like `~/.chartreux/.env`; these tools
# have legitimate uses, so the sensitive-path check, not a blanket deny, is
# the gate. sed's leading script operand is excluded by path_candidates;
# arbitrary awk scripts are not modeled here.
_FILE_CONTENT_COMMANDS = {
    "base64",
    "gunzip",
    "gzip",
    "hd",
    "hexdump",
    "iconv",
    "openssl",
    "strings",
    "dd",
    "install",
    "rsync",
    "sed",
    "tar",
    "xxd",
    "zcat",
}

# Inspect recognized reader and mutator operands before permitting execution.
# This bounded heuristic is not a containment promise for arbitrary programs.
_PATH_COMMANDS = (
    _MUTATING_PATH_COMMANDS | set(_READ_ONLY_COMMANDS_POSIX) | _FILE_CONTENT_COMMANDS
)


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
        name = Path(tokens[0]).name if tokens else ""
        shell_options = []
        if name in _SHELLS:
            for token in tokens[1:]:
                if token == "--":
                    continue
                if token == "-c" or (
                    token.startswith("-")
                    and not token.startswith("--")
                    and "c" in token[1:]
                ):
                    shell_options.append(token)
                    break
                if not token.startswith("-"):
                    break
                shell_options.append(token)
        shell_interactive = any(
            token == "--interactive"
            or (
                token.startswith("-")
                and not token.startswith("--")
                and "i" in token[1:]
            )
            for token in shell_options
        )
        interpreter = inline_interpreter_switch(tokens)
        return next(
            (
                pattern
                for pattern in self.config.denylist
                if matches_command_prefix(tokens, _split_command_tokens(pattern))
                or pattern == interpreter
                or (shell_interactive and pattern == f"{name} -i")
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

    def _repository_config_risk(
        self, tokens: list[str], policy: ShellCommandPolicy, possible_cwds: set[Path]
    ) -> str | None:
        """The repository-config execution vector for this Git reader, if any.

        Git's -C/--git-dir/--work-tree globals relocate the repository before
        the reader runs, so their values join the statically tracked directories
        as candidate repository roots. Only directories are candidates: other
        path values (e.g. --pathspec-from-file) name files, not repositories.
        """
        cwds = set(possible_cwds)
        for cwd in possible_cwds:
            current = cwd
            index = 1
            while index < len(tokens) and tokens[index].startswith("-"):
                token = tokens[index]
                if token == "-C" and index + 1 < len(tokens):
                    current = resolve_tool_path(tokens[index + 1], current)
                    index += 2
                elif token.startswith("-C") and token != "-C":
                    current = resolve_tool_path(token[2:], current)
                    index += 1
                elif token in {"--git-dir", "--work-tree"}:
                    index += 2
                else:
                    index += 1
            cwds.add(current)
        for value in policy.option_path_values:
            for cwd in possible_cwds:
                resolved = resolve_tool_path(value, cwd)
                if resolved.is_dir():
                    cwds.add(resolved)
        for cwd in cwds:
            if risk := git_repository_config_risk(tokens, cwd=cwd):
                return risk
        return None

    def _resolve_guardrail_permission(  # noqa: PLR0911
        self,
        command_parts: list[str],
        expanded_parts: list[_GuardrailPart] | None = None,
    ) -> PermissionContext | None:
        expanded = expanded_parts or _expand_guardrail_parts(command_parts)[0]
        for entry in expanded:
            part = entry.text
            tokens = _split_command_tokens(part)
            if (
                tokens
                and Path(tokens[0]).name == "unset"
                and any(
                    name in DANGEROUS_ENV_NAMES or name.startswith("GIT_")
                    for name in tokens[1:]
                )
            ):
                return PermissionContext(
                    permission=ToolPermission.NEVER,
                    reason="Command denied: unsetting a protected shell environment variable",
                )
            if tokens and Path(tokens[0]).name == "env":
                return PermissionContext(
                    permission=ToolPermission.NEVER,
                    reason="Command denied: env wrapper cannot be safely inspected",
                )
            try:
                matched = self._find_denylist_match(part)
            except ValueError as exc:
                return PermissionContext(
                    permission=ToolPermission.NEVER, reason=f"Command denied: {exc}"
                )
            if matched:
                return PermissionContext(
                    permission=ToolPermission.NEVER,
                    reason=f"Command denied: '{part}' matches denylist pattern '{matched}'. Do not attempt to run this command.",
                )
            if self._is_standalone_denylisted(part):
                return PermissionContext(
                    permission=ToolPermission.NEVER,
                    reason=f"Command denied: '{part}' is not allowed as a standalone command. Do not attempt to run this command.",
                )
        for entry, tokens, possible_cwds, cwd_unknown in _scoped_guardrail_cwds(
            expanded, self.cwd
        ):
            part = entry.text
            if self._is_sensitive(part):
                return PermissionContext(
                    permission=ToolPermission.NEVER,
                    reason="Command denied by a sensitive command rule",
                )
            policy = analyze_shell_command_policy(tokens)
            if policy.requires_approval:
                reason = (
                    "Command denied: find execution predicates are not permitted"
                    if tokens and Path(tokens[0]).name == "find"
                    else f"Command denied: side-effecting options are not permitted: '{part}'"
                )
                return PermissionContext(permission=ToolPermission.NEVER, reason=reason)
            if not policy.inspect_git_repository:
                continue
            if cwd_unknown:
                return PermissionContext(
                    permission=ToolPermission.NEVER,
                    reason=(
                        "Command denied: the working directory at this git command "
                        "is not statically known, so its repository git config "
                        "cannot be inspected"
                    ),
                )
            if risk := self._repository_config_risk(tokens, policy, possible_cwds):
                return PermissionContext(
                    permission=ToolPermission.NEVER, reason=f"Command denied: {risk}"
                )
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
        self, path_parts: list[_GuardrailPart]
    ) -> PermissionContext | None:
        for _entry, tokens, possible_cwds, cwd_unknown in _scoped_guardrail_cwds(
            path_parts, self.cwd
        ):
            if not tokens:
                continue
            command = Path(tokens[0]).name
            for token in path_candidates(
                tokens, inspect_positional_paths=command in _PATH_COMMANDS
            ):
                if token == "<redirect>":
                    continue
                # Shell globs can expand into hidden sensitive files. Quoting is
                # lost in shlex's argv view, so deny ambiguous patterns outright.
                if (
                    command == "find"
                    and token in tokens[2:]
                    and any(
                        tokens[index] == token
                        and tokens[index - 1] in {"-name", "-iname", "-path", "-ipath"}
                        for index in range(2, len(tokens))
                    )
                ):
                    continue
                if any(character in token for character in _SHELL_GLOB_CHARACTERS):
                    return PermissionContext(
                        permission=ToolPermission.NEVER,
                        reason="Shell path glob cannot be safely inspected",
                    )
                if (
                    cwd_unknown
                    and not Path(token).is_absolute()
                    and not token.startswith("~")
                ):
                    return PermissionContext(
                        permission=ToolPermission.NEVER,
                        reason="Shell file operand has a working directory that is not statically known",
                    )
                for cwd in possible_cwds:
                    resolved = resolve_tool_path(token, cwd)
                    if matches_sensitive_pattern(
                        str(resolved), DEFAULT_SENSITIVE_PATTERNS
                    ):
                        return PermissionContext(
                            permission=ToolPermission.NEVER,
                            reason="Sensitive file access denied (bash)",
                        )
                    if is_path_within_workdir(str(resolved), workspace=self.workspace):
                        continue
                    if (
                        self.scratchpad_dir is not None
                        and self.scratchpad_dir.resolve()
                        == self.scratchpad_dir.absolute()
                        and is_scratchpad_path(
                            str(resolved), scratchpad_dir=self.scratchpad_dir
                        )
                        and (
                            self.workspace.ceiling is None
                            or self.workspace.ceiling.allows(resolved)
                        )
                    ):
                        continue
                    return PermissionContext(
                        permission=ToolPermission.NEVER,
                        reason="Shell path is outside the authorized workspace; only an explicit user scope change can authorize it",
                    )
        return None

    def resolve_permission(self, args: BashArgs) -> PermissionContext | None:  # noqa: PLR0911
        precondition = self._resolve_preconditions()
        if precondition is not None:
            return precondition

        if (
            len(args.command.encode("utf-8", errors="surrogatepass"))
            > _MAX_ANALYZED_BYTES
        ):
            return PermissionContext(
                permission=ToolPermission.NEVER,
                reason="Command denied: shell analysis budget exceeded (bytes)",
            )
        analysis = analyze_shell_command(args.command)
        command_parts = list(analysis.command_parts)
        try:
            expanded, inner_redirects, inner_errors = _expand_guardrail_parts(
                command_parts
            )
        except _UnsafeShellSyntax as exc:
            return PermissionContext(
                permission=ToolPermission.NEVER, reason=f"Command denied: {exc}"
            )
        expanded_command_parts = [entry.text for entry in expanded]
        if "shell analysis failed" in analysis.approval_reasons:
            guardrail_permission = PermissionContext(
                permission=ToolPermission.NEVER,
                reason=f"Command denied: {analysis.approval_label}",
            )
        else:
            guardrail_permission = self._resolve_guardrail_permission(
                command_parts, expanded
            )
        if guardrail_permission is not None:
            return guardrail_permission
        if inner_errors:
            return PermissionContext(
                permission=ToolPermission.NEVER,
                reason=f"Command denied: nested shell syntax requiring approval: {', '.join(sorted(set(inner_errors)))}",
            )

        if any(
            _split_command_tokens(part)
            and _split_command_tokens(part)[0] in {"eval", "exec"}
            for part in expanded_command_parts
        ):
            return PermissionContext(
                permission=ToolPermission.NEVER,
                reason="Command denied: eval and exec cannot be safely inspected",
            )

        redirect_paths = _extract_redirect_paths(args.command) + inner_redirects
        path_parts = expanded + [
            _GuardrailPart(f"cat {shlex.quote(path)}") for path in redirect_paths
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
                raise self._build_timeout_error(args.command, timeout)

            stdout, stderr = await asyncio.gather(
                asyncio.to_thread(decode_console_safe, stdout_bytes),
                asyncio.to_thread(decode_console_safe, stderr_bytes),
            )
            stdout = stdout[:max_bytes]
            stderr = stderr[:max_bytes]

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
