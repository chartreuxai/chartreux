"""The pure git-policy half of the shell-policy family: subcommand tables and _git_policy.

The ordinary (non-dangerous) git policy: the reader subcommand table
(``_GIT_SUBCOMMAND_TABLE``), the destructive-guard chain
(``_legacy_clean_flags``, ``_git_destructive_policy``), the operand tables
for archive output, apply, init, worktree add/move, and submodule add (with
the ``--unsafe-paths`` hard-guard reason), the mode-aware symbolic-ref
policy (``_symbolic_ref_policy``), ``_git_destination_operands``,
and ``_git_policy`` itself, which builds the ``ShellCommandPolicy`` result
for one git invocation. The module is pure: no filesystem access, no
subprocess, no grant mutation. It imports only ``_shell_argv``,
``_shell_git_support``, and ``_shell_policy_types`` -- never the facade, so
the import graph stays acyclic. ``_shell_command_policy`` (the facade)
re-exports every name defined here, so import paths and object identity are
unchanged by the split.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from chartreux.core.tools.builtins._shell_argv import (
    _has_long,
    _options_before_double_dash,
    _OptionTable,
    _scan_options,
)
from chartreux.core.tools.builtins._shell_git_support import (
    _GIT_BOOLEAN_GLOBALS,
    _GIT_GLOBAL_TABLE,
    _GIT_NONREADER_SUBCOMMANDS,
    _GIT_READER_SUBCOMMANDS,
    _GIT_UNMODELED_BUILTINS,
    _effective_clean_flags,
    _git_global_options,
    _scan_git_config_call,
    _scan_git_symbolic_ref_call,
)
from chartreux.core.tools.builtins._shell_policy_types import ShellCommandPolicy

if TYPE_CHECKING:
    from chartreux.core.tools.builtins._shell_git_support import _GitSymbolicRefCall

_GIT_SUBCOMMAND_TABLE = _OptionTable(
    frozenset({
        "--anchored",
        "--color-moved-ws",
        "--contents",
        "--diff-filter",
        "--diff-algorithm",
        "--diff-merges",
        "--ignore-matching-lines",
        "--inter-hunk-context",
        "--line-prefix",
        "--max-count",
        "--output",
        "--output-indicator-context",
        "--output-indicator-new",
        "--output-indicator-old",
        "--pathspec-from-file",
        "--rotate-to",
        "--skip-to",
        "--src-prefix",
        "--dst-prefix",
        "--stat-count",
        "--stat-graph-width",
        "--stat-name-width",
        "--word-diff-regex",
    }),
    frozenset("GLSO"),
    frozenset({
        "--abbrev",
        "--color",
        "--color-moved",
        "--dirstat",
        "--find-copies",
        "--find-renames",
        "--format",
        "--pretty",
        "--stat",
        "--submodule",
        "--unified",
        "--word-diff",
    }),
    frozenset("U"),
    boolean_long=frozenset({
        "--ext-diff",
        "--help",
        "--remerge-diff",
        "--show-signature",
        "--textconv",
    }),
)


def _legacy_clean_flags(effective_options: list[str]) -> tuple[bool, bool]:
    """Resolve raw git clean option tokens into effective (force, dry-run).

    The runtime policy layer scans raw tokens, not the classifier's
    resolved options: long spellings map directly, and a short cluster
    contributes its ``f``/``n`` characters in order.
    """
    options: list[str] = []
    for token in effective_options:
        if token in {"--dry-run", "--no-dry-run", "--force", "--no-force"}:
            options.append(token)
        elif token.startswith("-") and not token.startswith("--"):
            options.extend(f"-{short}" for short in token[1:] if short in {"f", "n"})
    return _effective_clean_flags(tuple(options))


def _git_destructive_policy(args: list[str]) -> bool:
    index = 0
    while index < len(args) and args[index].startswith("-"):
        token = args[index]
        if token == "--":
            index += 1
            break
        name = token.partition("=")[0]
        if name in _GIT_BOOLEAN_GLOBALS or token in {"-h", "-p", "-P", "-v"}:
            index += 1
            continue
        if name in _GIT_GLOBAL_TABLE.long_values or token[:2] in {"-C", "-c"}:
            has_attached_value = (
                "=" in token if token.startswith("--") else len(token) > len(token[:2])
            )
            index += 1 if has_attached_value else 2
            continue
        if name in _GIT_GLOBAL_TABLE.optional_long_values:
            index += 1
            continue
        return False
    if index >= len(args):
        return False
    subcommand = args[index]
    options = _options_before_double_dash(args[index + 1 :])
    if subcommand == "reset":
        return "--hard" in options
    if subcommand == "clean":
        effective_options: list[str] = []
        option_index = 0
        subcommand_args = args[index + 1 :]
        while option_index < len(subcommand_args):
            token = subcommand_args[option_index]
            if token == "--":
                break
            effective_options.append(token)
            if token in {"-e", "--exclude"}:
                option_index += 2
            else:
                option_index += 1
        # git resolves repeated boolean options in order and accepts --no-
        # spellings of them, so the last dry-run/force spelling decides:
        # --dry-run --no-dry-run is a destructive clean, --force --no-force
        # is not. Presence alone would misread both.
        force, dry_run = _legacy_clean_flags(effective_options)
        return force and not dry_run
    return False


# git archive's --output/-o operand names the written archive: a raw
# file-write operand of an otherwise exempted subcommand, governed like any
# other write operand.
_GIT_ARCHIVE_OUTPUT_TABLE = _OptionTable(
    long_values=frozenset({"--output"}), short_values=frozenset({"o"})
)


def _git_archive_output_values(args: list[str]) -> tuple[str, ...]:
    """The --output/-o targets of one git archive call, in every spelling."""
    parsed = _scan_options(args, _GIT_ARCHIVE_OUTPUT_TABLE)
    return parsed.values_for("--output", "-o")


# git apply's --directory relocates the patch's write root: every patched
# path (after -p stripping) lands beneath it, so the value is a raw write
# operand of an otherwise exempted subcommand, governed like any other.
# --output redirects the patched result into a file instead of patching the
# work-tree paths in place, so its value is a raw write operand too.
_GIT_APPLY_TABLE = _OptionTable(
    long_values=frozenset({"--directory", "--output"}),
    boolean_long=frozenset({"--unsafe-paths"}),
)


# --unsafe-paths disables the path refusal git itself applies outside the
# work tree — including paths inside .git metadata, which git otherwise
# rejects as "invalid path" — and no legitimate agent workflow needs it.
_GIT_APPLY_UNSAFE_PATHS_REASON = (
    "git apply --unsafe-paths disables the path refusal git itself "
    "applies outside the work tree, including paths inside protected "
    "Git metadata: no legitimate agent workflow needs it, and no "
    "permission setting can authorize it"
)


# git init's positional operand is the directory the call creates, and
# --separate-git-dir names the git dir it creates; both are write operands.
# --shared takes an optional attached value only, so it never consumes the
# positional directory.
_GIT_INIT_TABLE = _OptionTable(
    long_values=frozenset({
        "--initial-branch",
        "--object-format",
        "--ref-format",
        "--separate-git-dir",
        "--template",
    }),
    short_values=frozenset({"b"}),
    optional_long_values=frozenset({"--shared"}),
    boolean_long=frozenset({"--bare", "--quiet"}),
)


# git worktree add's first positional operand is the new work tree's path;
# -b/-B and --reason take values, every other option is a boolean (--orphan
# included: it selects an unborn branch and takes no value).
_GIT_WORKTREE_ADD_TABLE = _OptionTable(
    long_values=frozenset({"--reason"}),
    short_values=frozenset({"b", "B"}),
    boolean_long=frozenset({
        "--checkout",
        "--detach",
        "--force",
        "--guess-remote",
        "--lock",
        "--orphan",
        "--quiet",
        "--relative-paths",
        "--track",
    }),
)


# git worktree move's grammar is `git worktree move [-f] <worktree>
# <new-path>`: -f/--force is the only option, and the SECOND positional
# names the directory the call relocates the work tree to.
_GIT_WORKTREE_MOVE_TABLE = _OptionTable(boolean_long=frozenset({"--force"}))


# git submodule add's grammar is `git submodule add [options] <repository>
# [<path>]`: the SECOND positional names the directory the call creates
# the submodule work tree at, and --separate-git-dir names the git dir it
# creates for it (the init twin). -b/--branch, --name, --depth,
# --reference, and --separate-git-dir take values; every other option is a
# boolean.
_GIT_SUBMODULE_ADD_TABLE = _OptionTable(
    long_values=frozenset({
        "--branch",
        "--depth",
        "--name",
        "--reference",
        "--separate-git-dir",
    }),
    short_values=frozenset({"b"}),
    boolean_long=frozenset({"--dissociate", "--force", "--progress", "--quiet"}),
)


def _git_destination_operands(  # noqa: PLR0911
    subcommand: str, args: list[str], index: int
) -> tuple[list[str], str | None]:
    """The write operands a git subcommand creates or relocates its writes to.

    ``git apply --directory`` names the patch's write root (every patched
    path lands beneath it after ``-p`` stripping) and ``--output`` redirects
    the patched result into a file instead of patching the work-tree paths
    in place, so both values are raw write operands of an otherwise exempted
    subcommand, governed like any other. Without either, the write root is
    still derivable: git applies paths relative to the current working
    directory, so the cwd tree root is the write root — the resolver's write
    collection derives it per call through
    :func:`git_apply_cwd_root_write` (the policy layer keeps the recorded
    operand values only, so the recorded policy surface is unchanged).
    Returns the write operands and, for ``git apply --unsafe-paths``, the
    hard-guard reason: the flag disables the path refusal git itself applies
    outside the work tree, including inside protected metadata.
    """
    rest = args[index + 1 :]
    if subcommand == "apply":
        apply_parsed = _scan_options(rest, _GIT_APPLY_TABLE)
        write_paths = list(apply_parsed.values_for("--directory"))
        write_paths.extend(apply_parsed.values_for("--output"))
        if _has_long(apply_parsed, "--unsafe-paths"):
            return write_paths, _GIT_APPLY_UNSAFE_PATHS_REASON
        return write_paths, None
    if subcommand == "init":
        init_parsed = _scan_options(rest, _GIT_INIT_TABLE)
        write_paths = [init_parsed.positionals[0]] if init_parsed.positionals else []
        write_paths.extend(init_parsed.values_for("--separate-git-dir"))
        return write_paths, None
    if subcommand == "worktree" and args[index + 1 : index + 2] == ["add"]:
        add_parsed = _scan_options(args[index + 2 :], _GIT_WORKTREE_ADD_TABLE)
        write_paths = [add_parsed.positionals[0]] if add_parsed.positionals else []
        return write_paths, None
    if subcommand == "worktree" and args[index + 1 : index + 2] == ["move"]:
        move_parsed = _scan_options(args[index + 2 :], _GIT_WORKTREE_MOVE_TABLE)
        # The first positional selects the work tree to move; the second
        # names its new location.
        write_paths = (
            [move_parsed.positionals[1]] if len(move_parsed.positionals) > 1 else []
        )
        return write_paths, None
    if subcommand == "submodule" and args[index + 1 : index + 2] == ["add"]:
        add_parsed = _scan_options(args[index + 2 :], _GIT_SUBMODULE_ADD_TABLE)
        # The first positional is the source repository (a read); the
        # second names the destination the call creates.
        write_paths = (
            [add_parsed.positionals[1]] if len(add_parsed.positionals) > 1 else []
        )
        write_paths.extend(add_parsed.values_for("--separate-git-dir"))
        return write_paths, None
    return [], None


def git_apply_cwd_root_write(tokens: list[str]) -> bool:
    """Whether one git invocation is a plain ``git apply`` writing the cwd root.

    Plain ``git apply`` patches the paths the diff names, relative to the
    current working directory (after git's own ``-C`` relocations), so the
    cwd tree root is the write root the resolver's write collection resolves
    against the relocated cwds. The policy layer keeps the recorded operand
    values only (``--directory``/``--output``), so this predicate — not a
    cwd-root entry in the policy's write-path values — carries the derived
    root, and the recorded policy surface is unchanged.
    """
    args = tokens[1:]
    globals_result = _git_global_options(args)
    if globals_result is None:
        return False
    index, _paths = globals_result
    if index >= len(args) or args[index] != "apply":
        return False
    apply_parsed = _scan_options(args[index + 1 :], _GIT_APPLY_TABLE)
    return not (
        apply_parsed.values_for("--directory") or apply_parsed.values_for("--output")
    )


def _symbolic_ref_policy(
    call: _GitSymbolicRefCall, paths: list[str]
) -> ShellCommandPolicy:
    """The mode-aware ``git symbolic-ref`` policy.

    The read form (exactly one operand, no write marker) resolves like a
    reader; the write forms (a ref update or deletion) and unscannable
    forms are not granted by the read modeling and keep the
    unmodeled-builtin denial.
    """
    if call.reads:
        return ShellCommandPolicy(
            inspect_git_repository=True,
            option_path_values=tuple(paths),
            positional_values=call.operands,
        )
    return ShellCommandPolicy(
        requires_approval=True,
        denial_reason=(
            "unmodeled git builtin 'symbolic-ref' write form; denied "
            "conservatively; permission settings cannot override this guard"
            if call.writes
            else "unmodeled git builtin 'symbolic-ref' form; denied "
            "conservatively; permission settings cannot override this guard"
        ),
    )


def _git_policy(args: list[str]) -> ShellCommandPolicy:
    if _git_destructive_policy(args):
        return ShellCommandPolicy(requires_approval=True)
    globals_result = _git_global_options(args)
    if globals_result is None:
        return ShellCommandPolicy(requires_approval=True)
    index, paths = globals_result
    if index >= len(args):
        return ShellCommandPolicy(option_path_values=tuple(paths))
    subcommand = args[index]
    if subcommand not in _GIT_READER_SUBCOMMANDS:
        if subcommand == "symbolic-ref":
            # Modeled by mode, not blanket-read: the read/write verdict
            # lives in _symbolic_ref_policy.
            return _symbolic_ref_policy(
                _scan_git_symbolic_ref_call(args[index + 1 :]), paths
            )
        option_paths = list(paths)
        write_paths: list[str] = []
        hard_guard_reason: str | None = None
        if subcommand == "archive":
            # git archive --output/-o writes the named file directly.
            write_paths.extend(_git_archive_output_values(args[index + 1 :]))
        elif subcommand == "config":
            # A writing git config --file/-f names the written file, so it
            # is a write-classified operand; a read names a file git only
            # reads, which stays authority-scoped like a selector. An
            # unscannable form fails closed as a write.
            call = _scan_git_config_call(args[index + 1 :])
            file_values = [
                value for option, value in call.values if option in {"--file", "-f"}
            ]
            if call.error is not None or call.mutates or call.edits:
                write_paths.extend(file_values)
            else:
                option_paths.extend(file_values)
        elif (
            subcommand in {"apply", "init"}
            or (
                subcommand == "worktree"
                and args[index + 1 : index + 2] in (["add"], ["move"])
            )
            or (subcommand == "submodule" and args[index + 1 : index + 2] == ["add"])
        ):
            # Destinations the call creates or relocates its writes to
            # (apply --directory, init/worktree-add/move/submodule-add
            # destinations); the helper also reports the --unsafe-paths
            # hard guard.
            write_paths, hard_guard_reason = _git_destination_operands(
                subcommand, args, index
            )
        denial_reason = (
            f"unmodeled git builtin {subcommand!r}; denied conservatively; permission settings cannot override this guard"
            if subcommand in _GIT_UNMODELED_BUILTINS
            else f"unrecognized git subcommand {subcommand!r}; aliases and external helpers cannot be inspected"
        )
        if hard_guard_reason is not None:
            denial_reason = hard_guard_reason
        return ShellCommandPolicy(
            requires_approval=(
                subcommand not in _GIT_NONREADER_SUBCOMMANDS
                or hard_guard_reason is not None
            ),
            denial_reason=denial_reason,
            option_path_values=tuple(option_paths),
            write_path_values=tuple(write_paths),
        )
    parsed = _scan_options(args[index + 1 :], _GIT_SUBCOMMAND_TABLE)
    # --remerge-diff and --diff-merges=remerge re-run repository-configured
    # merge drivers; --show-signature runs the configured gpg program.
    diff_merge_values = {
        value.casefold() for value in parsed.values_for("--diff-merges")
    }
    requires_approval = (
        _has_long(parsed, "--output")
        or any(
            token
            in {
                "--ext-diff",
                "--help",
                "--remerge-diff",
                "--show-signature",
                "--textconv",
            }
            for token in parsed.options
        )
        or bool(diff_merge_values & {"r", "remerge"})
    )
    option_paths = list(paths)
    option_paths.extend(parsed.values_for("--contents", "--pathspec-from-file"))
    option_paths.extend(parsed.values_for("-O"))
    return ShellCommandPolicy(
        requires_approval=requires_approval,
        inspect_positional_paths=subcommand == "diff"
        and "--no-index" in parsed.options,
        # Plain readers stay permitted for ordinary repositories; the Bash
        # resolver separately inspects repository-owned git config for
        # executable helpers and denies the reader when one is active.
        inspect_git_repository=True,
        option_path_values=tuple(option_paths),
        positional_values=parsed.positionals,
    )
