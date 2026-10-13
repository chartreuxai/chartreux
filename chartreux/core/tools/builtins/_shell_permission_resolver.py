from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from functools import lru_cache
import glob
from pathlib import Path
import re
import shlex
from typing import TYPE_CHECKING, Protocol

from tree_sitter import Language, Parser
import tree_sitter_bash as tsbash

from chartreux.core.dangerous_commands import (
    DangerousCommandAuthority,
    DangerousCommandProposal,
    RepositoryScope,
    operations_for_ids,
)
from chartreux.core.tools.base import ToolPermission
from chartreux.core.tools.builtins._shell_command_policy import (
    ShellCommandPolicy,
    analyze_shell_command_policy,
    canonical_git_command,
    classify_dangerous_git_command,
    executor_boundary,
    git_cwd_relocations,
    git_metadata_paths,
    git_push_destination_risk,
    git_repository_config_risk,
    git_work_tree_writing_subcommand,
    inline_interpreter_switch,
    is_dangerous_git_subcommand,
    matches_command_prefix,
    path_candidates,
    resolve_git_work_tree,
    resolve_repository_scope,
)
from chartreux.core.tools.builtins._shell_diagnostics import (
    DenialClass,
    Diagnostic,
    aggregate_denial_class,
    render_diagnostic,
)
from chartreux.core.tools.builtins._shell_permission_analysis import (
    _REDIRECTION_DEVICES,
    DANGEROUS_ENV_NAMES,
    ShellPermissionAnalysis,
    _has_active_bracket_glob,
    aggregate_approval_diagnostic,
    analyze_shell_command,
    argv_permission_diagnostics,
)
from chartreux.core.tools.permissions import PermissionContext
from chartreux.core.tools.utils import (
    PathAccess,
    PathAuthority,
    ambient_workspace,
    effective_scratchpad_roots,
    resolve_tool_path,
)
from chartreux.core.workspace import Workspace

if TYPE_CHECKING:
    from chartreux.core.tools.builtins.bash import BashToolConfig


class ShellCommandArgs(Protocol):
    command: str


class DangerousCommandAuthorityProvider(Protocol):
    """Runtime-owned dangerous-command grants (session and once lifetimes).

    This is the port the runtime approval lifecycle implements; the resolver
    only consumes immutable snapshots. The resolver never mutates grants,
    persists nothing, and executes nothing: an implementation returns the
    authorities currently in force, each keyed by operation and
    repository/worktree scope, and owns its own lifetimes (a once grant
    disappears from the snapshot after the invocation it authorized).
    """

    def dangerous_command_authorities(
        self,
    ) -> tuple[DangerousCommandAuthority, ...]: ...


def _denied(diagnostic: Diagnostic) -> PermissionContext:
    return PermissionContext(
        permission=ToolPermission.NEVER, reason=render_diagnostic(diagnostic)
    )


def _dangerous_policy_reason(context: PermissionContext) -> Diagnostic:
    """The policy-class reason a dangerous-command denial contributes to a call.

    A denial carrying a grantable proposal names the missing operations; a
    hard guard (unmodeled, unclassifiable, or unlocatable git form) is stated
    as uninspectable. Either way the reason is policy-class, so a call that
    also has unmodeled syntax aggregates to a policy denial.
    """
    proposal = context.dangerous_command_proposal
    if proposal is not None:
        return Diagnostic(
            "analysis",
            detail=(
                "a dangerous git operation requiring approval ("
                + ", ".join(proposal.operations)
                + ")"
            ),
        )
    return Diagnostic(
        "analysis", detail="a dangerous git command that cannot be safely inspected"
    )


def _authority_denial(
    decision: PermissionContext, token: str, command_part: str
) -> PermissionContext:
    """The denial for a path-authority NEVER decision.

    Keeps the shell's established outside-scope diagnostic.
    """
    if (
        decision.reason
        and decision.reason.startswith("File ")
        and "outside" in decision.reason
    ):
        return _denied(Diagnostic("path_outside", token, command_part))
    return _denied(Diagnostic("text", token, command_part, decision.reason))


def _scope_sort_key(scope: RepositoryScope) -> tuple[str, str]:
    """Deterministic ordering for candidate repository scopes."""
    return (
        str(scope.git_dir),
        str(scope.work_tree) if scope.work_tree is not None else "",
    )


@lru_cache(maxsize=1)
def _get_parser() -> Parser:
    return Parser(Language(tsbash.language()))


_SHELLS = frozenset({"ash", "bash", "sh", "dash", "hush", "zsh"})
_MAX_SHELL_DEPTH = 8
_MAX_EXPANDED_COMMANDS = 256
_MAX_ANALYZED_BYTES = 64 * 1024


class _UnsafeShellSyntax(ValueError):
    """A fail-closed expansion failure carrying its denial class.

    The class is decided where the failure is raised. The default is POLICY:
    an unclassified expansion failure is a policy denial and names no
    alternative.
    """

    def __init__(
        self,
        message: str,
        *,
        denial_class: DenialClass = DenialClass.POLICY,
        alternative: str | None = None,
    ) -> None:
        super().__init__(message)
        self.denial_class = denial_class
        self.alternative = alternative


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
    *,
    scratchpad_roots: frozenset[Path] = frozenset(),
    redirect_operands: tuple[tuple[str, ...], ...] = (),
    workspace_roots: frozenset[Path] | None = None,
) -> tuple[list[_GuardrailPart], list[str], list[Diagnostic]]:
    """Expand in occurrence order, preserving child-shell cwd isolation.

    Literal redirect targets become synthetic WRITE operands placed at their
    occurrence position — immediately before the command part they redirect —
    so each one resolves against the working directory in force at that point
    in the chain, never a later cd's directory.
    """
    expanded: list[_GuardrailPart] = []
    redirects: list[str] = []
    errors: list[Diagnostic] = []
    byte_count = 0
    next_scope = 0

    def append_redirect_operand(target: str, scope: tuple[int, ...]) -> None:
        nonlocal byte_count
        if target in _REDIRECTION_DEVICES:
            return  # Device sinks are not filesystem targets.
        byte_count += len(target.encode("utf-8"))
        if byte_count > _MAX_ANALYZED_BYTES or len(expanded) >= _MAX_EXPANDED_COMMANDS:
            raise _UnsafeShellSyntax(
                "shell analysis budget exceeded (bytes or commands)"
            )
        expanded.append(
            _GuardrailPart(f"cat {shlex.quote(target)}", scope, PathAccess.WRITE)
        )

    def visit(
        part: str,
        scope: tuple[int, ...],
        depth: int,
        cwd_overrides: tuple[str, ...] = (),
        *,
        inspect_executable: bool = False,
        redirect_targets: tuple[str, ...] = (),
    ) -> None:
        nonlocal byte_count, next_scope
        for target in redirect_targets:
            append_redirect_operand(target, scope)
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
            raise _UnsafeShellSyntax(
                "malformed shell quoting",
                denial_class=DenialClass.SYNTAX,
                alternative="write the command with well-formed shell quoting",
            ) from exc
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
                analysis = _analyze_guardrail_source(
                    source,
                    nested_source=True,
                    scratchpad_roots=scratchpad_roots,
                    workspace_roots=workspace_roots,
                )
                errors.extend(analysis.approval_diagnostic.related)
                nested_operands = analysis.redirect_operands
                redirects.extend(
                    target for group in nested_operands for target in group
                )
                next_scope += 1
                child_scope = (*scope, next_scope)
                nested_parts = list(analysis.command_parts)
                for index, child in enumerate(nested_parts):
                    targets = (
                        nested_operands[index] if index < len(nested_operands) else ()
                    )
                    visit(child, child_scope, depth + 1, redirect_targets=targets)
                if len(nested_operands) > len(nested_parts):
                    # Targets that follow the nested source's last command.
                    for target in nested_operands[len(nested_parts)]:
                        append_redirect_operand(target, child_scope)
        if boundary.inner_command_range is not None or boundary.module_command:
            if depth >= _MAX_SHELL_DEPTH:
                raise _UnsafeShellSyntax("shell nesting depth limit exceeded")
            start, end = boundary.inner_command_range or (0, 0)
            child_tokens = list(boundary.module_command) or tokens[start:end]
            errors.extend(
                replace(reason, command_part=shlex.join(child_tokens))
                for reason in argv_permission_diagnostics(
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

    for index, part in enumerate(command_parts):
        targets = redirect_operands[index] if index < len(redirect_operands) else ()
        visit(part, (), 0, redirect_targets=targets)
    if len(redirect_operands) > len(command_parts):
        # Targets that follow the last command of the whole chain.
        for target in redirect_operands[len(command_parts)]:
            append_redirect_operand(target, ())
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
    # The analysis appends a <redirect> marker to redirected commands; it is
    # not a shell operand, so it must not disturb cd/pushd/popd token counts.
    tokens = [token for token in tokens if token != "<redirect>"]
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
        # Recognized dangerous Git forms (push, checkout, restore, stash
        # drop/clear, switch --discard-changes/-f, reflog expire/delete) are
        # no longer denylist entries: they are classified per operation by
        # the dangerous-command flow and denied pending typed authority.
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


# Mutating path commands: their positional operands are write-classified and
# inspected, so they are also checked against protected Git metadata. ln
# creates the hard-link/symlink aliases a later write can reach metadata
# through (both operands are checked: source and link path); rmdir/unlink are
# rm's directory/single-file twins.
_MUTATING_PATH_COMMANDS = {
    "cd",
    "chmod",
    "chown",
    "cp",
    "ln",
    "mkdir",
    "mv",
    "rm",
    "rmdir",
    "tee",
    "touch",
    "unlink",
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
# patch/truncate/unzip are in-place writers whose positional operands (and
# unzip's -d extraction destination) name the written files. shred and
# fallocate are in-place mutators whose positional operands name the
# shredded or grown files; ar's positional archive operand is created or
# replaced and bsdtar/pax are tar-family archivers (their -f archive and
# bsdtar's -C destination are extracted by their write-operand tables);
# cpio's policy extracts only its write operands (the pass-through
# destination positional, the -D directory, and the copy-out archive files).
# objcopy is an in-place mutator whose positional operands name the copied
# input and the written output (its option values surface as positional
# candidates through the fallback scan, which can only over-block). The
# compression families join the same surface: bzip2/bunzip2/bzcat,
# xz/unxz/xzcat, and lzma/unlzma/lzcat/lzmainfo replace or annotate their
# positional operand in place (gzip/gunzip/zcat were already modeled; the
# zstd family is governed by its own write-operand policy, which also
# extracts -o/--output-dir-* operands). mktemp creates a file at every
# template positional; its -p/--tmpdir directories are extracted by its
# policy, and the separate -p spelling surfaces as a positional candidate
# through the fallback scan.
_FILE_CONTENT_COMMANDS = {
    "ar",
    "base64",
    "bsdtar",
    "bunzip2",
    "bzcat",
    "bzip2",
    "cpio",
    "dd",
    "fallocate",
    "gunzip",
    "gzip",
    "hd",
    "hexdump",
    "iconv",
    "install",
    "lzma",
    "lzcat",
    "lzmainfo",
    "mktemp",
    "objcopy",
    "openssl",
    "pax",
    "patch",
    "rsync",
    "sed",
    "shred",
    "strings",
    "tar",
    "truncate",
    "unlzma",
    "unzip",
    "unxz",
    "xz",
    "xzcat",
    "xxd",
    "zcat",
}

# Script-loading builtins read their operand as shell source in the current
# shell, so the sourced file's path joins the inspected surface like a
# reader's operand: classified READ (never write/mutate), so a script outside
# workspace+scratchpad is denied while benign in-workspace and scratchpad
# scripts stay allowed.
_SOURCE_PATH_COMMANDS = {".", "source"}

# Inspect recognized reader and mutator operands before permitting execution.
# This bounded heuristic is not a containment promise for arbitrary programs.
_PATH_COMMANDS = (
    _MUTATING_PATH_COMMANDS
    | _PACKAGE_PATH_COMMANDS
    | set(_READ_ONLY_COMMANDS_POSIX)
    | _FILE_CONTENT_COMMANDS
    | _SOURCE_PATH_COMMANDS
)


def _split_command_tokens(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        return command.split()


def _part_command(tokens: list[str]) -> str:
    """The command a guardrail part runs, keeping the dot builtin's bare name.

    ``Path(".").name`` is empty, so dot sourcing needs its raw token to join
    the path-inspected surface.
    """
    return Path(tokens[0]).name or tokens[0]


# Only commands whose modeled operands cannot be output files receive exact
# instruction-read capabilities. E.g. sort -o, uniq's second operand, sed and
# archivers are deliberately not classified as readers. The script-loading
# builtins read their operand as shell source, so they are readers too: an
# outside-workspace script is denied like any other outside read operand.
_INSTRUCTION_READ_COMMANDS = frozenset({
    ".",
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
    "source",
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


def _operand_resolution_cwds(
    token: str,
    tokens: list[str],
    possible_cwds: set[Path],
    *,
    git_write_values: frozenset[str],
    git_write_cwds: frozenset[Path] | None,
) -> set[Path] | frozenset[Path]:
    """The working directories one path operand resolves against.

    Git's raw write operands resolve against the directories its own -C
    sequence relocates the tracked cwds to — git applies -C before the
    subcommand runs, so `git -C .git archive -o config HEAD` resolves
    `config` inside the metadata dir. Every other operand resolves against
    the tracked cwds unchanged.
    """
    if token in git_write_values and git_write_cwds is not None:
        return git_write_cwds
    return _operand_guardrail_cwds(token, tokens, possible_cwds)


# Metadata files whose content selects the programs git runs or the repository
# identity itself. A hard link is invisible to path resolution, so write-time
# containment compares (device, inode) identities against exactly these files.
_METADATA_CRITICAL_FILES = ("config", "config.worktree", "HEAD", "commondir", "index")


def _metadata_hardlink_inodes(
    directories: frozenset[Path],
) -> frozenset[tuple[int, int]]:
    """(device, inode) identities of the critical files inside metadata dirs."""
    inodes: set[tuple[int, int]] = set()
    for directory in directories:
        for name in _METADATA_CRITICAL_FILES:
            try:
                status = (directory / name).stat()
            except OSError:
                continue
            inodes.add((status.st_dev, status.st_ino))
        try:
            hooks = list((directory / "hooks").iterdir())
        except OSError:
            continue
        for hook in hooks:
            try:
                status = hook.stat()
            except OSError:
                continue
            inodes.add((status.st_dev, status.st_ino))
    return frozenset(inodes)


def _is_hardlinked_metadata(resolved: Path, inodes: frozenset[tuple[int, int]]) -> bool:
    """Whether an existing path is a hard link to a critical metadata file.

    Path resolution follows symlinks but never hard links, so an alias
    created outside this guard (pre-existing, or made by another process) is
    invisible to containment; comparing identities catches the write-through
    at the moment it happens. Metadata files outside the critical set remain
    a documented residual (ADR 0019).
    """
    if not inodes:
        return False
    try:
        status = resolved.stat()
    except OSError:
        return False
    return (status.st_dev, status.st_ino) in inodes


@dataclass(frozen=True)
class _MetadataGuard:
    """Protected Git metadata identity for one shell resolution.

    ``cwds`` is the union of every statically tracked working directory (the
    union the redirect guard resolves against); ``directories`` is the union
    of git metadata directories across those cwds (``git_metadata_paths``);
    ``inodes`` carries the critical-file identities a hard link could alias.
    """

    cwds: frozenset[Path]
    directories: frozenset[Path]
    inodes: frozenset[tuple[int, int]]


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
        command = _part_command(tokens) if tokens else None
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


_analysis_cache: ContextVar[dict[tuple[str, bool], ShellPermissionAnalysis] | None] = (
    ContextVar("shell_launch_analysis", default=None)
)


@contextmanager
def shell_analysis_scope() -> Iterator[None]:
    """Reuse pure syntax analysis within one launch, never cache authority."""
    if _analysis_cache.get() is not None:
        yield
        return
    token = _analysis_cache.set({})
    try:
        yield
    finally:
        _analysis_cache.reset(token)


def _analyze_guardrail_source(
    command: str,
    *,
    nested_source: bool = False,
    scratchpad_roots: frozenset[Path] = frozenset(),
    workspace_roots: frozenset[Path] | None = None,
) -> ShellPermissionAnalysis:
    """Analyze shell source, reusing pure syntax analysis within one launch.

    The cache key stays (command, nested_source); the scratchpad roots and the
    workspace roots are deliberately not part of it. Within one launch the
    local resolver and the parent invocation guard can hold different
    authorities, so a cached analysis may carry another caller's roots — a
    spurious approval reason at worst (the false-denial direction). A cached
    scratchpad exemption can never authorize a write outside
    workspace+scratchpad: the resolver's path layer independently gates every
    literal redirect target as a WRITE against the caller's own authority.
    The workspace roots gate only the sed simulation's READ — which file the
    analysis may read at all — never an authorization.
    """
    cache = _analysis_cache.get()
    key = (command, nested_source)
    if cache is None:
        return analyze_shell_command(
            command,
            nested_source=nested_source,
            scratchpad_roots=scratchpad_roots,
            workspace_roots=workspace_roots,
        )
    if key not in cache:
        cache[key] = analyze_shell_command(
            command,
            nested_source=nested_source,
            scratchpad_roots=scratchpad_roots,
            workspace_roots=workspace_roots,
        )
    return cache[key]


def _matches_pattern(command: str, pattern: str) -> bool:
    return command == pattern or command.startswith(pattern + " ")


class ShellPermissionResolver:
    """Shared foreground/managed-launch shell policy, with captured authority."""

    def __init__(
        self,
        config: BashToolConfig,
        cwd: Path,
        workspace: Workspace,
        path_authority: PathAuthority,
        dangerous_command_authorities: Callable[
            [], tuple[DangerousCommandAuthority, ...]
        ]
        | None = None,
    ) -> None:
        self.config = config
        self.cwd = cwd
        self.workspace = workspace
        self.path_authority = path_authority
        # Live getter: authority changes must invalidate cached decisions, so
        # the snapshot is read per resolution, never captured at construction.
        self._dangerous_command_authorities = dangerous_command_authorities or (
            lambda: ()
        )

    def _authorities(self) -> tuple[DangerousCommandAuthority, ...]:
        """The immutable dangerous-command authority snapshot for this call."""
        try:
            return self._dangerous_command_authorities()
        except Exception:
            # A broken authority source fails closed: no authority, no grant.
            return ()

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
        canonical = canonical_git_command(tokens)
        return next(
            (
                pattern
                for pattern in self.config.denylist
                if matches_command_prefix(tokens, _split_command_tokens(pattern))
                or (
                    canonical is not None
                    and matches_command_prefix(
                        canonical, _split_command_tokens(pattern)
                    )
                )
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
        expanded = (
            expanded_parts
            or _expand_guardrail_parts(
                command_parts,
                workspace_roots=frozenset(self.workspace.authorized_roots),
            )[0]
        )
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

    def _check_resolved_operand(
        self,
        *,
        token: str,
        resolved: Path,
        access: PathAccess,
        command_part: str,
        authority: PathAuthority,
        metadata_guard: _MetadataGuard | None,
    ) -> PermissionContext | None:
        """The authority decision plus metadata containment for one operand."""
        decision = authority.resolve(str(resolved), access)
        if decision is not None and decision.permission == ToolPermission.NEVER:
            return _authority_denial(decision, token, command_part)
        if metadata_guard is not None and (
            hit := self._operand_metadata_hit(token, resolved, metadata_guard)
        ):
            return _denied(Diagnostic(hit, token, command_part))
        return None

    def _ls_listing_glob_permitted(
        self,
        *,
        token: str,
        possible_cwds: set[Path],
        cwd_unknown: bool,
        access: PathAccess,
        authority: PathAuthority,
    ) -> bool:
        """Whether an ls listing glob stays within every scoped cwd."""
        if cwd_unknown or Path(token).is_absolute() or token.startswith("~"):
            return False
        parent = str(Path(token).parent)
        if any(character in parent for character in _SHELL_GLOB_CHARACTERS):
            return False
        return all(
            self.workspace.allows(resolve_tool_path(parent, cwd))
            and authority.allows(resolve_tool_path(parent, cwd), access)
            and authority.allows(resolve_tool_path(token, cwd), access)
            and all(
                self.workspace.allows(Path(match).resolve())
                and authority.allows(Path(match).resolve(), access)
                for match in glob.glob(str(cwd / token))
            )
            for cwd in possible_cwds
        )

    def _resolve_path_permission(
        self, path_parts: list[_GuardrailPart]
    ) -> PermissionContext | None:
        authority = self.path_authority
        metadata_guard: _MetadataGuard | None = None
        metadata_unlocated = False
        for entry, tokens, possible_cwds, cwd_unknown in _scoped_guardrail_cwds(
            path_parts, self.cwd
        ):
            if not tokens:
                continue
            command = _part_command(tokens)
            access = entry.access or _operand_access(command)
            # Git's own selectors locate the repository: its -C/--git-dir/
            # --work-tree operands are classified and risk-inspected by the
            # dangerous-command and repository-config flows. Git's RAW WRITE
            # operands (archive --output, a writing config --file, apply
            # --directory, an init/worktree-add/move or submodule-add
            # destination) are governed
            # like any other write operand and join the metadata guard; they
            # resolve against the directories git's own -C sequence
            # relocates the tracked cwds to, because git applies -C before
            # the subcommand runs. Every other write- or mutate-classified
            # path operand of every command must stay out of the protected
            # metadata of every repository the call can touch.
            git_write_values, git_write_cwds, git_denial = self._git_entry_governance(
                tokens=tokens,
                path_parts=path_parts,
                possible_cwds=possible_cwds,
                cwd_unknown=cwd_unknown,
                entry=entry,
            )
            if git_denial is not None:
                return git_denial
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
                # Bash never pathname-expands a redirect destination, so a
                # literal glob-looking target is a literal filename: the
                # path-glob heuristic fires for command operands only, where
                # the shell really globs. Synthetic redirect operands are the
                # only parts that carry an access classification.
                if entry.access is None and (
                    any(character in token for character in "*?")
                    or _has_active_bracket_glob(token.encode())
                ):
                    # Python glob and shell bracket expressions differ. Dot-
                    # leading patterns may also expand to . or .. in shells.
                    supported_glob = "[" not in token and not Path(
                        token
                    ).name.startswith(".")
                    if (
                        command == "ls"
                        and supported_glob
                        and self._ls_listing_glob_permitted(
                            token=token,
                            possible_cwds=possible_cwds,
                            cwd_unknown=cwd_unknown,
                            access=access,
                            authority=authority,
                        )
                    ):
                        continue
                    return _denied(Diagnostic("path_glob", token, entry.text))
                if (
                    cwd_unknown
                    and not Path(token).is_absolute()
                    and not token.startswith("~")
                ):
                    return _denied(Diagnostic("operand_cwd_unknown", token, entry.text))
                guard_metadata = access is not PathAccess.READ and (
                    command != "git" or token in git_write_values
                )
                operand_cwds = _operand_resolution_cwds(
                    token,
                    tokens,
                    possible_cwds,
                    git_write_values=git_write_values,
                    git_write_cwds=git_write_cwds,
                )
                for cwd in operand_cwds:
                    resolved = resolve_tool_path(token, cwd)
                    if guard_metadata:
                        if metadata_guard is None and not metadata_unlocated:
                            metadata_guard = self._git_metadata_guard(path_parts)
                            metadata_unlocated = metadata_guard is None
                        if metadata_unlocated:
                            # A repository is present but its metadata cannot
                            # be located; the write cannot be verified safe.
                            return _denied(Diagnostic("metadata_unlocated"))
                    if denied := self._check_resolved_operand(
                        token=token,
                        resolved=resolved,
                        access=access,
                        command_part=entry.text,
                        authority=authority,
                        metadata_guard=metadata_guard if guard_metadata else None,
                    ):
                        return denied
        return None

    def _git_entry_governance(
        self,
        *,
        tokens: list[str],
        path_parts: list[_GuardrailPart],
        possible_cwds: set[Path],
        cwd_unknown: bool,
        entry: _GuardrailPart,
    ) -> tuple[frozenset[str], frozenset[Path] | None, PermissionContext | None]:
        """Git's raw write operands, their -C relocation, and work-tree denial.

        Returns the write-path operand values the operand guard must cover
        (archive --output, a writing config --file, apply --directory, an
        init/worktree-add/move or submodule-add destination), the
        directories git's own -C sequence relocates the tracked cwds to
        (None when the call carries no -C, so the operands resolve against
        the tracked cwds unchanged), and the
        denial for a tree-writing subcommand whose resolved work tree lands
        inside the git metadata dir; the denial is None for every other
        invocation. A non-git entry yields no values, no cwds, and no denial.
        """
        if Path(tokens[0]).name != "git":
            return frozenset(), None, None
        policy = analyze_shell_command_policy(tokens)
        write_values = frozenset(policy.write_path_values)
        relocated: frozenset[Path] | None = None
        if write_values:
            relocated_cwds = git_cwd_relocations(tokens, possible_cwds)
            relocated = (
                frozenset(relocated_cwds) if relocated_cwds is not None else None
            )
        if git_work_tree_writing_subcommand(tokens) is None:
            return write_values, relocated, None
        denial = self._git_work_tree_denial(
            tokens=tokens,
            path_parts=path_parts,
            possible_cwds=possible_cwds,
            cwd_unknown=cwd_unknown,
            entry=entry,
        )
        return write_values, relocated, denial

    def _git_work_tree_denial(
        self,
        *,
        tokens: list[str],
        path_parts: list[_GuardrailPart],
        possible_cwds: set[Path],
        cwd_unknown: bool,
        entry: _GuardrailPart,
    ) -> PermissionContext | None:
        """Deny tree-writing git calls whose work tree resolves into metadata.

        A work tree inside the git metadata dir is never legitimate: the
        subcommand writes tracked (or patched) paths beneath it, so the write
        lands in protected metadata — the probed routine ACE chain that
        planted an executable hook. The resolved work tree is governed like a
        write operand: workspace authority first, then metadata containment.
        """
        if cwd_unknown:
            # The work tree cannot be statically resolved; fail closed.
            return _denied(Diagnostic("operand_cwd_unknown", "<work tree>", entry.text))
        guard = self._git_metadata_guard(path_parts)
        if guard is None:
            # A repository is present but its metadata cannot be located; the
            # write cannot be verified safe.
            return _denied(Diagnostic("metadata_unlocated"))
        authority = self.path_authority
        for cwd in possible_cwds:
            work_tree = resolve_git_work_tree(tokens, cwd)
            if work_tree is None:
                return _denied(Diagnostic("metadata_unlocated"))
            decision = authority.resolve(str(work_tree), PathAccess.MUTATE)
            if decision is not None and decision.permission == ToolPermission.NEVER:
                return _authority_denial(decision, str(work_tree), entry.text)
            if ".git" in work_tree.parts or any(
                work_tree.is_relative_to(directory) for directory in guard.directories
            ):
                return _denied(
                    Diagnostic("metadata_worktree", str(work_tree), entry.text)
                )
        return None

    def _git_metadata_guard(
        self, expanded: list[_GuardrailPart]
    ) -> _MetadataGuard | None:
        """Locate protected Git metadata across every tracked working directory.

        The union mirrors the redirect guard's: the shell's own repository is
        included even for a redirect-only command or when the first command
        changes directories, and executor cwd overrides contribute their child
        directories. None means a repository is present but its metadata
        cannot be located, so callers fail closed.
        """
        cwds: set[Path] = {self.cwd}
        metadata: set[Path] = set()
        # Include the shell's repository even for a redirect-only command or
        # when the first command changes directories.
        parts = [_GuardrailPart(""), *expanded]
        for _, tokens, possible_cwds, _ in _scoped_guardrail_cwds(parts, self.cwd):
            cwds.update(possible_cwds)
            for cwd in possible_cwds:
                directories = git_metadata_paths(tokens, cwd=cwd)
                if directories is None:
                    return None
                metadata.update(directories)
        directories = frozenset(metadata)
        return _MetadataGuard(
            frozenset(cwds), directories, _metadata_hardlink_inodes(directories)
        )

    @staticmethod
    def _operand_metadata_hit(
        token: str, resolved: Path, guard: _MetadataGuard
    ) -> str | None:
        """The diagnostic code when one write-classified operand reaches metadata.

        Containment covers any tracked repository's metadata (resolved
        against every tracked cwd, including executor child directories);
        the literal ``.git`` component check catches metadata spelled into the
        operand itself; the inode comparison catches a pre-existing hard link
        to a critical metadata file, which path resolution cannot see.
        """
        if ".git" in Path(token).parts or ".git" in resolved.parts:
            return "metadata_operand"
        if any(resolved.is_relative_to(directory) for directory in guard.directories):
            return "metadata_operand"
        if _is_hardlinked_metadata(resolved, guard.inodes):
            return "metadata_hardlink"
        return None

    def _redirect_metadata_permission(
        self, redirect_paths: list[str], expanded: list[_GuardrailPart]
    ) -> PermissionContext | None:
        if not redirect_paths:
            return None
        guard = self._git_metadata_guard(expanded)
        if guard is None:
            return _denied(Diagnostic("metadata_unlocated"))
        for path in redirect_paths:
            if ".git" in Path(path).parts:
                return _denied(Diagnostic("metadata_redirect", path))
            hit: str | None = None
            for cwd in guard.cwds:
                resolved = resolve_tool_path(path, cwd)
                if ".git" in resolved.parts or any(
                    resolved.is_relative_to(directory)
                    for directory in guard.directories
                ):
                    hit = "metadata_redirect"
                    break
                if _is_hardlinked_metadata(resolved, guard.inodes):
                    # A pre-existing hard link writes through to the metadata
                    # file; path resolution cannot see it, the inode can.
                    hit = "metadata_hardlink"
                    break
            if hit is not None:
                return _denied(Diagnostic(hit, path))
        return None

    def _resolve_dangerous_command_permission(  # noqa: PLR0911
        self, args: ShellCommandArgs, expanded: list[_GuardrailPart]
    ) -> PermissionContext | None:
        """Classify the whole expanded call and deny missing dangerous authority.

        Runs last: a dangerous_command denial is returned only when it is the
        sole missing authority, so it never masks tool disablement, explicit
        denylist entries, sensitivity or protected metadata, path restrictions,
        or any other separately-denied segment (in either order). Unmodeled or
        unlocatable git forms are hard-denied with no grantable proposal, and
        so is a push whose bare-name destination operand is not a configured
        remote of every candidate repository: git resolves an unconfigured
        destination as a URL, so the push would run the destination's
        receive-pack and hooks.
        """
        authorities = self._authorities()
        missing: dict[str, RepositoryScope] = {}
        for entry, tokens, possible_cwds, cwd_unknown in _scoped_guardrail_cwds(
            expanded, self.cwd
        ):
            if entry.access is not None or not tokens:
                continue  # Synthetic redirect operands are not executable commands.
            classification = classify_dangerous_git_command(tokens)
            if classification.kind == "routine":
                continue
            if classification.kind == "hard_denied":
                if not is_dangerous_git_subcommand(tokens):
                    # Unmodeled builtins and unrecognized subcommands carry no
                    # dangerous operation: the existing policy layer governs
                    # them, and its enforcement is unchanged.
                    continue
                return _denied(
                    Diagnostic(
                        "dangerous_guard",
                        detail=classification.reason,
                        command_part=entry.text,
                    )
                )
            grantable = classification.grantable
            if grantable is None:
                # The frozen classifier attaches the payload exactly when the
                # kind is grantable; a violation fails closed, never allows.
                return _denied(
                    Diagnostic(
                        "dangerous_guard",
                        detail="this git command could not be classified",
                        command_part=entry.text,
                    )
                )
            if cwd_unknown:
                return _denied(
                    Diagnostic(
                        "dangerous_guard",
                        detail=(
                            "the working directory at this git command is not "
                            "statically known, so its repository cannot be located"
                        ),
                        command_part=entry.text,
                    )
                )
            # The command may run in any statically possible directory, so
            # every candidate repository scope must carry the authority.
            scopes: set[RepositoryScope] = set()
            for cwd in possible_cwds:
                scope = resolve_repository_scope(cwd, grantable.relocation)
                if scope is None:
                    return _denied(
                        Diagnostic(
                            "dangerous_guard",
                            detail=(
                                "the repository for this git command cannot be located"
                            ),
                            command_part=entry.text,
                        )
                    )
                scopes.add(scope)
            # A bare-name push destination resolves only through the
            # repository's configured remotes: git resolves any destination
            # that is not a configured remote name as a URL, so an
            # unconfigured name would push to a path and run the
            # destination's receive-pack and hooks. Every repository the
            # command may run in must resolve the destination.
            for scope in sorted(scopes, key=_scope_sort_key):
                if risk := git_push_destination_risk(grantable.call, scope):
                    return _denied(
                        Diagnostic(
                            "dangerous_guard", detail=risk, command_part=entry.text
                        )
                    )
            for operation_id in grantable.operation_ids:
                covered = all(
                    any(
                        operation_id in authority.operations
                        and authority.scope == scope
                        for authority in authorities
                    )
                    for scope in scopes
                )
                if not covered and operation_id not in missing:
                    # Candidate scopes come from a set; keep the proposal's
                    # representative scope deterministic.
                    missing[operation_id] = min(scopes, key=_scope_sort_key)
        if not missing:
            return None
        operations = operations_for_ids(missing)
        proposal = DangerousCommandProposal(
            command=tuple(_split_command_tokens(args.command)),
            project=self.cwd,
            scope=next(iter(missing.values())),
            operations=tuple(operation.operation_id for operation in operations),
            consequence="; ".join(operation.consequence for operation in operations),
        )
        return PermissionContext(
            permission=ToolPermission.NEVER,
            reason=render_diagnostic(
                Diagnostic(
                    "dangerous_command",
                    ", ".join(operation.operation_id for operation in operations),
                    args.command,
                    "; ".join(operation.consequence for operation in operations),
                )
            ),
            denial_kind="dangerous_command",
            dangerous_command_proposal=proposal,
        )

    def resolve_permission(self, args: ShellCommandArgs) -> PermissionContext | None:  # noqa: PLR0911
        precondition = self._resolve_preconditions()
        if precondition is not None:
            return precondition

        if (
            len(args.command.encode("utf-8", errors="surrogatepass"))
            > _MAX_ANALYZED_BYTES
        ):
            return _denied(Diagnostic("byte_budget"))
        scratchpad_roots = effective_scratchpad_roots(self.path_authority)
        workspace_roots = frozenset(self.workspace.authorized_roots)
        analysis = _analyze_guardrail_source(
            args.command,
            scratchpad_roots=scratchpad_roots,
            workspace_roots=workspace_roots,
        )
        command_parts = list(analysis.command_parts)
        try:
            expanded, inner_redirects, inner_errors = _expand_guardrail_parts(
                command_parts,
                scratchpad_roots=scratchpad_roots,
                redirect_operands=analysis.redirect_operands,
                workspace_roots=workspace_roots,
            )
        except _UnsafeShellSyntax as exc:
            return _denied(
                Diagnostic(
                    "command",
                    detail=str(exc),
                    command_part=args.command,
                    denial_class=exc.denial_class,
                    alternative=exc.alternative,
                )
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
            return _denied(
                Diagnostic(
                    "nested_analysis",
                    related=tuple(inner_errors),
                    # Fail closed: any policy or unknown inner finding makes
                    # the nested denial a policy denial with no alternative.
                    denial_class=aggregate_denial_class(tuple(inner_errors)),
                )
            )

        if any(
            _split_command_tokens(part)
            and _split_command_tokens(part)[0] in {"eval", "exec"}
            for part in expanded_command_parts
        ):
            return _denied(Diagnostic("eval_exec"))

        redirect_paths = [
            target for group in analysis.redirect_operands for target in group
        ] + inner_redirects
        # Redirection destinations are file operands, not a blanket veto on
        # unrelated Git readers in the same pipeline or command list. The
        # metadata guard sees the original literal targets here, before any
        # scratchpad-target authorization can take effect below.
        if metadata_permission := self._redirect_metadata_permission(
            redirect_paths, expanded
        ):
            return metadata_permission
        # Redirect destinations already became synthetic WRITE operands at
        # their occurrence positions inside `expanded`; each resolves against
        # the working directory tracked at that point in the chain.
        path_permission = self._resolve_path_permission(expanded)
        if path_permission is not None:
            return path_permission

        if analysis.requires_approval:
            # The whole-call denial class is decided BEFORE rendering: a call
            # that also lacks a dangerous-command authority carries a
            # policy-class reason, so the aggregate is a policy denial that
            # names no alternative — suggesting one invites the refused
            # action through another tool (AM-F). The authorization outcome
            # is unchanged: the call was already denied, and the
            # dangerous-command flow below still reports only the sole
            # missing authority.
            dangerous_permission = self._resolve_dangerous_command_permission(
                args, expanded
            )
            if dangerous_permission is not None:
                return _denied(
                    aggregate_approval_diagnostic(
                        analysis, _dangerous_policy_reason(dangerous_permission)
                    )
                )
            return _denied(analysis.approval_diagnostic)

        # The dangerous-command flow runs last: it only ever reports the sole
        # missing authority, after every other restriction has been checked.
        dangerous_permission = self._resolve_dangerous_command_permission(
            args, expanded
        )
        if dangerous_permission is not None:
            return dangerous_permission
        return PermissionContext(permission=ToolPermission.ALWAYS)
