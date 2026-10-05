from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Iterator
from dataclasses import dataclass
from functools import lru_cache
import glob
from pathlib import Path
import re
import shlex
from typing import final

from pydantic import BaseModel, Field, computed_field, model_validator
from tree_sitter import Language, Node, Parser
import tree_sitter_bash as tsbash

from chartreux.core.events import ToolResultEvent, ToolStreamEvent
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
    executor_boundary,
    git_metadata_paths,
    git_repository_config_risk,
    inline_interpreter_switch,
    matches_command_prefix,
    path_candidates,
)
from chartreux.core.tools.builtins._shell_diagnostics import (
    Diagnostic,
    render_diagnostic,
)
from chartreux.core.tools.builtins._shell_permission_analysis import (
    DANGEROUS_ENV_NAMES,
    ShellPermissionAnalysis,
    _has_active_bracket_glob,
    analyze_shell_command,
    argv_permission_reasons,
)
from chartreux.core.tools.io_port import ShellCommandRequest
from chartreux.core.tools.permissions import PermissionContext
from chartreux.core.tools.ui import ToolCallDisplay, ToolResultDisplay, ToolUIData
from chartreux.core.tools.utils import (
    DEFAULT_SENSITIVE_PATTERNS,
    PathAccess,
    PathAuthority,
    ambient_workspace,
    resolve_tool_path,
)
from chartreux.core.utils import kill_async_subprocess
from chartreux.core.utils.shell import spawn_shell_command
from chartreux.core.workspace import Workspace
from chartreux.utils.io import decode_console_safe
from chartreux.utils.tool_presentation import ToolEffectKind


def _denied(diagnostic: Diagnostic) -> PermissionContext:
    return PermissionContext(
        permission=ToolPermission.NEVER, reason=render_diagnostic(diagnostic)
    )


@lru_cache(maxsize=1)
def _get_parser() -> Parser:
    return Parser(Language(tsbash.language()))


_SHELLS = frozenset({"ash", "bash", "sh", "dash", "hush", "zsh"})
_MAX_SHELL_DEPTH = 8
_MAX_EXPANDED_COMMANDS = 256
_MAX_ANALYZED_BYTES = 64 * 1024


class _UnsafeShellSyntax(ValueError):
    pass


@dataclass(frozen=True)
class _GuardrailPart:
    text: str
    scope: tuple[int, ...] = ()
    access: PathAccess | None = None
    cwd_overrides: tuple[str, ...] = ()
    inspect_executable: bool = False


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


def _expand_guardrail_parts(  # noqa: PLR0915
    command_parts: list[str],
) -> tuple[list[_GuardrailPart], list[str], list[Diagnostic]]:
    """Expand in occurrence order, preserving child-shell cwd isolation."""
    expanded: list[_GuardrailPart] = []
    redirects: list[str] = []
    errors: list[Diagnostic] = []
    byte_count = 0
    next_scope = 0

    def visit(
        part: str,
        scope: tuple[int, ...],
        depth: int,
        cwd_overrides: tuple[str, ...] = (),
        *,
        inspect_executable: bool = False,
    ) -> None:
        nonlocal byte_count, next_scope
        byte_count += len(part.encode("utf-8"))
        if byte_count > _MAX_ANALYZED_BYTES or len(expanded) >= _MAX_EXPANDED_COMMANDS:
            raise _UnsafeShellSyntax(
                "shell analysis budget exceeded (bytes or commands)"
            )
        expanded.append(
            _GuardrailPart(
                part,
                scope,
                cwd_overrides=cwd_overrides,
                inspect_executable=inspect_executable,
            )
        )
        try:
            tokens = shlex.split(part)
        except ValueError as exc:
            raise _UnsafeShellSyntax("malformed shell quoting") from exc
        try:
            inline_interpreter_switch(tokens)  # Preserve legacy option diagnostics.
            boundary = executor_boundary(tokens, admission=True)
        except ValueError as exc:
            raise _UnsafeShellSyntax(str(exc)) from exc
        if boundary is None:
            if _has_unrecognized_nested_shell(tokens):
                name = Path(tokens[0]).name
                raise _UnsafeShellSyntax(
                    f"unsupported wrapper before shell -c ({name})"
                )
            return
        if boundary.boundary_kind == "shell_source":
            if depth >= _MAX_SHELL_DEPTH:
                raise _UnsafeShellSyntax("shell nesting depth limit exceeded")
            for source in boundary.inline_payloads:
                analysis = _analyze_guardrail_source(source, nested_source=True)
                errors.extend(analysis.approval_diagnostic.related)
                redirects.extend(_extract_redirect_paths(source))
                next_scope += 1
                child_scope = (*scope, next_scope)
                for child in analysis.command_parts:
                    visit(child, child_scope, depth + 1)
                for path in _extract_redirect_paths(source):
                    if path not in _REDIRECTION_DEVICES:
                        byte_count += len(path.encode("utf-8"))
                        if (
                            byte_count > _MAX_ANALYZED_BYTES
                            or len(expanded) >= _MAX_EXPANDED_COMMANDS
                        ):
                            raise _UnsafeShellSyntax(
                                "shell analysis budget exceeded (bytes or commands)"
                            )
                        expanded.append(
                            _GuardrailPart(
                                f"cat {shlex.quote(path)}",
                                child_scope,
                                PathAccess.WRITE,
                            )
                        )
        if boundary.inner_command_range is not None or boundary.module_command:
            if depth >= _MAX_SHELL_DEPTH:
                raise _UnsafeShellSyntax("shell nesting depth limit exceeded")
            start, end = boundary.inner_command_range or (0, 0)
            child_tokens = list(boundary.module_command) or tokens[start:end]
            errors.extend(
                Diagnostic(
                    "analysis", command_part=shlex.join(child_tokens), detail=reason
                )
                for reason in argv_permission_reasons(
                    child_tokens, nested_source=bool(scope)
                )
            )
            child_scope = scope
            if boundary.child_execution or boundary.cwd_overrides:
                next_scope += 1
                child_scope = (*scope, next_scope)
            # shlex.join is only a lossless argv serialization for the policy
            # helpers. This argv is NEVER parsed as shell source or an AST.
            visit(
                shlex.join(child_tokens),
                child_scope,
                depth + 1,
                boundary.cwd_overrides,
                inspect_executable=True,
            )

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
            for directory in entry.cwd_overrides:
                scope_cwds[scope] = {
                    resolve_tool_path(directory, parent_cwd)
                    for parent_cwd in scope_cwds[scope]
                }
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


_MUTATING_PATH_COMMANDS = {
    "cd",
    "chmod",
    "chown",
    "cp",
    "mkdir",
    "mv",
    "rm",
    "tee",
    "touch",
}

# Package managers accept local package/directory operands as well as names.
# Use the same bounded positional-path inspection as other path commands;
# this does not model package scripts, manifests, or every option's operands.
_PACKAGE_PATH_COMMANDS = {
    "bun",
    "cargo",
    "go",
    "npm",
    "npx",
    "pip",
    "pip3",
    "pipx",
    "pnpm",
    "poetry",
    "uv",
    "yarn",
}

# File-content commands beyond the read-only set, including encoders,
# archivers, and mutators whose positional operands name files. Inspected so
# sensitive and outside-workspace operands are denied rather than treating
# the command name as evidence of read-only behavior. Sed's policy excludes
# its script operand and inspects literal r/w targets and derived backups;
# arbitrary awk script effects and external sed script contents remain opaque.
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
    _MUTATING_PATH_COMMANDS
    | _PACKAGE_PATH_COMMANDS
    | set(_READ_ONLY_COMMANDS_POSIX)
    | _FILE_CONTENT_COMMANDS
)


def _split_command_tokens(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        return command.split()


# Only commands whose modeled operands cannot be output files receive exact
# instruction-read capabilities. E.g. sort -o, uniq's second operand, sed and
# archivers are deliberately not classified as readers.
_INSTRUCTION_READ_COMMANDS = frozenset({
    "cat",
    "head",
    "tail",
    "wc",
    "grep",
    "nl",
    "tac",
    "stat",
    "file",
    "ls",
    "md5sum",
    "sha1sum",
    "sha256sum",
    "shasum",
    "sum",
    "strings",
    "zcat",
})


def _operand_access(command: str) -> PathAccess:
    if command in _INSTRUCTION_READ_COMMANDS:
        return PathAccess.READ
    if command in {"mkdir", "tee", "touch"}:
        return PathAccess.WRITE
    # Mixed-source/destination commands and opaque effects are mutators: an
    # input-looking operand must not smuggle in an exact-file read capability.
    return PathAccess.MUTATE


def _operand_guardrail_cwds(
    token: str, tokens: list[str], possible_cwds: set[Path]
) -> set[Path]:
    """Directory selectors are parent-relative; executor operands are child-relative."""
    boundary = executor_boundary(tokens)
    if boundary is None or not boundary.cwd_overrides:
        return possible_cwds
    if token in boundary.cwd_overrides:
        policy = analyze_shell_command_policy(tokens)
        occurrences = (*policy.option_path_values, *(policy.positional_values or ()))
        if occurrences.count(token) <= len(boundary.cwd_overrides):
            return possible_cwds
        # Path candidates are strings, so identical selector/operand spellings
        # must be checked in both coordinate systems, not guessed by identity.
        child_cwds = possible_cwds
        for directory in boundary.cwd_overrides:
            child_cwds = {resolve_tool_path(directory, cwd) for cwd in child_cwds}
        return possible_cwds | child_cwds
    for directory in boundary.cwd_overrides:
        possible_cwds = {resolve_tool_path(directory, cwd) for cwd in possible_cwds}
    return possible_cwds


def _collect_outside_dirs(
    command_parts: list[str],
    *,
    workspace: Workspace | None = None,
    scratchpad_dir: Path | None = None,
    authority: PathAuthority | None = None,
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
    workspace = (
        authority.workspace
        if authority is not None
        else workspace or ambient_workspace()
    )
    resolved_cwd = workspace.cwd
    # The legacy helper reports boundaries only unless a full authority is
    # supplied. Keep its return contract without a parallel path evaluator.
    authority = authority or PathAuthority(
        "bash", ToolPermission.ALWAYS, (), (), (), workspace, scratchpad_dir
    )

    dirs: set[str] = set()
    try:
        expanded, _, errors = _expand_guardrail_parts(command_parts)
    except _UnsafeShellSyntax:
        return {str(resolved_cwd)}
    if errors:
        return {str(resolved_cwd)}
    for entry, tokens, possible_cwds, unknown in _scoped_guardrail_cwds(
        expanded, resolved_cwd
    ):
        command = Path(tokens[0]).name if tokens else None
        if not command:
            continue
        for token in path_candidates(
            tokens, inspect_positional_paths=command in _PATH_COMMANDS
        ):
            # Bare filenames can be symlinks outside the accepted workspace too.
            if token == "<redirect>":
                continue
            if unknown and not Path(token).is_absolute():
                dirs.add(str(resolved_cwd))
                continue
            for cwd in _operand_guardrail_cwds(token, tokens, possible_cwds):
                resolved = resolve_tool_path(token, cwd)
                if authority.allows(resolved, entry.access or _operand_access(command)):
                    continue
                parent = str(resolved) if resolved.is_dir() else str(resolved.parent)
                dirs.add(parent)
    return dirs


_REDIRECTION_DEVICES = frozenset({"/dev/null", "/dev/stdout", "/dev/stderr"})


def _analyze_guardrail_source(
    command: str, *, nested_source: bool = False
) -> ShellPermissionAnalysis:
    """Treat only literal standard-device redirects as workspace-neutral sinks."""
    try:
        source = command.encode("utf-8")
    except UnicodeEncodeError:
        return analyze_shell_command(command, nested_source=nested_source)
    replacements: list[tuple[int, int]] = []

    def visit(node: Node) -> None:
        if node.type == "file_redirect":
            destination = node.child_by_field_name("destination")
            if destination is not None and destination.text is not None:
                tokens = _split_command_tokens(destination.text.decode("utf-8"))
                if len(tokens) == 1 and tokens[0] in _REDIRECTION_DEVICES:
                    replacements.append((destination.start_byte, destination.end_byte))
        for child in node.children:
            visit(child)

    visit(_get_parser().parse(source).root_node)
    for start, end in sorted(replacements, reverse=True):
        source = source[:start] + b"__standard_device_sink__" + source[end:]
    return analyze_shell_command(source.decode("utf-8"), nested_source=nested_source)


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
                or (interpreter == "python3 -c" and pattern == "python -c")
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
            if entry.access is not None:
                continue  # Synthetic redirect operands are not executable commands.
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
                return _denied(Diagnostic("protected_env"))
            try:
                matched = self._find_denylist_match(part)
            except ValueError as exc:
                return _denied(
                    Diagnostic("command", detail=str(exc), command_part=part)
                )
            if matched:
                return _denied(Diagnostic("denylist", matched, part))
            if self._is_standalone_denylisted(part):
                return _denied(Diagnostic("standalone", command_part=part))
        for entry, tokens, possible_cwds, cwd_unknown in _scoped_guardrail_cwds(
            expanded, self.cwd
        ):
            if entry.access is not None:
                continue
            part = entry.text
            if self._is_sensitive(part):
                return _denied(Diagnostic("sensitive_command"))
            policy = analyze_shell_command_policy(tokens)
            if policy.requires_approval:
                diagnostic = (
                    Diagnostic(
                        "command", detail="find execution predicates are not permitted"
                    )
                    if tokens and Path(tokens[0]).name == "find"
                    else policy.denial_diagnostic(part)
                )
                return _denied(diagnostic)
            if not policy.inspect_git_repository:
                continue
            if cwd_unknown:
                return _denied(Diagnostic("git_cwd_unknown"))
            if risk := self._repository_config_risk(tokens, policy, possible_cwds):
                return _denied(Diagnostic("command", detail=risk, command_part=part))
        return None

    def _resolve_preconditions(self) -> PermissionContext | None:
        if self.config.permission == ToolPermission.NEVER:
            return _denied(Diagnostic("tool_denied"))
        if not self.workspace.allows(self.cwd):
            return _denied(Diagnostic("cwd_outside"))
        return None

    def _resolve_path_permission(
        self, path_parts: list[_GuardrailPart]
    ) -> PermissionContext | None:
        authority = self.path_authority
        for entry, tokens, possible_cwds, cwd_unknown in _scoped_guardrail_cwds(
            path_parts, self.cwd
        ):
            if not tokens:
                continue
            command = Path(tokens[0]).name
            access = entry.access or _operand_access(command)
            for token in path_candidates(
                tokens,
                inspect_positional_paths=command in _PATH_COMMANDS,
                inspect_executable=entry.inspect_executable,
            ):
                if token == "<redirect>":
                    continue
                # Only listing globs with a literal, scoped parent and safe
                # current matches are modeled; other path globs fail closed.
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
                if any(
                    character in token for character in "*?"
                ) or _has_active_bracket_glob(token.encode()):
                    # Python glob and shell bracket expressions differ. Dot-
                    # leading patterns may also expand to . or .. in shells.
                    supported_glob = "[" not in token and not Path(
                        token
                    ).name.startswith(".")
                    if (
                        command == "ls"
                        and not cwd_unknown
                        and not Path(token).is_absolute()
                        and not token.startswith("~")
                        and supported_glob
                    ):
                        parent = str(Path(token).parent)
                        if not any(
                            character in parent for character in _SHELL_GLOB_CHARACTERS
                        ) and all(
                            self.workspace.allows(resolve_tool_path(parent, cwd))
                            and authority.allows(resolve_tool_path(parent, cwd), access)
                            and authority.allows(resolve_tool_path(token, cwd), access)
                            and all(
                                self.workspace.allows(Path(match).resolve())
                                and authority.allows(Path(match).resolve(), access)
                                for match in glob.glob(str(cwd / token))
                            )
                            for cwd in possible_cwds
                        ):
                            continue
                    return _denied(Diagnostic("path_glob", token, entry.text))
                if (
                    cwd_unknown
                    and not Path(token).is_absolute()
                    and not token.startswith("~")
                ):
                    return _denied(Diagnostic("operand_cwd_unknown", token, entry.text))
                for cwd in _operand_guardrail_cwds(token, tokens, possible_cwds):
                    resolved = resolve_tool_path(token, cwd)
                    decision = authority.resolve(str(resolved), access)
                    if (
                        decision is not None
                        and decision.permission == ToolPermission.NEVER
                    ):
                        # Keep the shell's established outside-scope diagnostic.
                        if (
                            decision.reason
                            and decision.reason.startswith("File ")
                            and "outside" in decision.reason
                        ):
                            return _denied(
                                Diagnostic("path_outside", token, entry.text)
                            )
                        return _denied(
                            Diagnostic("text", token, entry.text, decision.reason)
                        )
        return None

    def _redirect_metadata_permission(
        self, redirect_paths: list[str], expanded: list[_GuardrailPart]
    ) -> PermissionContext | None:
        if not redirect_paths:
            return None
        redirect_cwds = {self.cwd}
        metadata: set[Path] = set()
        # Include the shell's repository even for a redirect-only command or
        # when the first command changes directories.
        parts = [_GuardrailPart(""), *expanded]
        for _, tokens, possible_cwds, _ in _scoped_guardrail_cwds(parts, self.cwd):
            redirect_cwds.update(possible_cwds)
            for cwd in possible_cwds:
                directories = git_metadata_paths(tokens, cwd=cwd)
                if directories is None:
                    return _denied(Diagnostic("metadata_unlocated"))
                metadata.update(directories)
        for path in redirect_paths:
            if ".git" in Path(path).parts or any(
                ".git" in (resolved := resolve_tool_path(path, cwd)).parts
                or any(resolved.is_relative_to(directory) for directory in metadata)
                for cwd in redirect_cwds
            ):
                return _denied(Diagnostic("metadata_redirect", path))
        return None

    def resolve_permission(self, args: BashArgs) -> PermissionContext | None:  # noqa: PLR0911
        precondition = self._resolve_preconditions()
        if precondition is not None:
            return precondition

        if (
            len(args.command.encode("utf-8", errors="surrogatepass"))
            > _MAX_ANALYZED_BYTES
        ):
            return _denied(Diagnostic("byte_budget"))
        analysis = _analyze_guardrail_source(args.command)
        command_parts = list(analysis.command_parts)
        try:
            expanded, inner_redirects, inner_errors = _expand_guardrail_parts(
                command_parts
            )
        except _UnsafeShellSyntax as exc:
            return _denied(
                Diagnostic("command", detail=str(exc), command_part=args.command)
            )
        expanded_command_parts = [entry.text for entry in expanded]
        if "shell analysis failed" in analysis.approval_reasons:
            guardrail_permission = _denied(analysis.approval_diagnostic)
        else:
            guardrail_permission = self._resolve_guardrail_permission(
                command_parts, expanded
            )
        if guardrail_permission is not None:
            return guardrail_permission
        if inner_errors:
            return _denied(Diagnostic("nested_analysis", related=tuple(inner_errors)))

        if any(
            _split_command_tokens(part)
            and _split_command_tokens(part)[0] in {"eval", "exec"}
            for part in expanded_command_parts
        ):
            return _denied(Diagnostic("eval_exec"))

        redirect_paths = _extract_redirect_paths(args.command) + inner_redirects
        # Redirection destinations are file operands, not a blanket veto on
        # unrelated Git readers in the same pipeline or command list.
        if metadata_permission := self._redirect_metadata_permission(
            redirect_paths, expanded
        ):
            return metadata_permission
        path_parts = expanded + [
            _GuardrailPart(f"cat {shlex.quote(path)}", access=PathAccess.WRITE)
            for path in _extract_redirect_paths(args.command)
            if path not in _REDIRECTION_DEVICES
        ]
        path_permission = self._resolve_path_permission(path_parts)
        if path_permission is not None:
            return path_permission

        if analysis.requires_approval:
            return _denied(analysis.approval_diagnostic)
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
        completed = False
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

            completed = True
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
                await kill_async_subprocess(
                    proc, kill_exited_process_group=not completed
                )
