"""The shared git-support leaves of the shell-policy family.

The symbols more than one of {ordinary policy, classifier, inspection,
facade} consume: git's global-option tables (``_GIT_GLOBAL_TABLE``,
``_GIT_BOOLEAN_GLOBALS``) and subcommand sets (``_GIT_READER_SUBCOMMANDS``,
``_GIT_PAGING_SUBCOMMANDS``, ``_GIT_NONREADER_SUBCOMMANDS``,
``_GIT_UNMODELED_BUILTINS``), ``_effective_clean_flags``, the runtime global
walker ``_git_global_options`` and its classifier mirror ``_scan_git_globals``
(two distinct functions with deliberately different validation: the runtime
tolerates a dangling ``-C``, the classifier hard-denies), plus
``canonical_git_command``, ``push_destination_operands``, the strict option
scanner stack (``_OptionTokenScan``, ``_StrictGitOptions``,
``_GitGlobalScan``, the ``_scan_*`` token scanners,
``_scan_classified_git_options``), the ``git config`` scan contract shared by
classifier and runtime policy (``_GIT_CONFIG_TABLE`` with its action sets,
``_GitConfigCall``, ``_scan_git_config_call``), and
``_GIT_SUBCOMMAND_INDEX``. This module imports only ``_shell_argv`` and
``chartreux.core.dangerous_commands``; ``_shell_command_policy`` (the facade)
re-exports every name defined here, so import paths and object identity are
unchanged by the split.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from chartreux.core.dangerous_commands import GitRepositoryRelocation
from chartreux.core.tools.builtins._shell_argv import (
    _match_long,
    _OptionTable,
    _scan_options,
)

if TYPE_CHECKING:
    from chartreux.core.dangerous_commands import ModeledGitCall


_GIT_GLOBAL_TABLE = _OptionTable(
    frozenset({
        "--config-env",
        "--git-dir",
        "--namespace",
        "--resolve-git-dir",
        "--super-prefix",
        "--work-tree",
    }),
    frozenset("Cc"),
    frozenset({"--exec-path"}),
)
_GIT_BOOLEAN_GLOBALS = frozenset({
    "--bare",
    "--glob-pathspecs",
    "--help",
    "--html-path",
    "--icase-pathspecs",
    "--info-path",
    "--literal-pathspecs",
    "--man-path",
    "--no-advice",
    "--no-optional-locks",
    "--no-pager",
    "--no-replace-objects",
    "--noglob-pathspecs",
    "--paginate",
    "--verbose",
    "--version",
})


def _effective_clean_flags(options: tuple[str, ...]) -> tuple[bool, bool]:
    """Effective (force, dry-run) state of git clean under last-wins semantics.

    git's parse-options resolves repeated boolean options in order and
    accepts a ``--no-`` spelling of every boolean, so the final spelling
    decides: ``--dry-run --no-dry-run`` leaves the clean destructive, and
    ``--force --no-force`` leaves it inert. Presence alone would misread
    both. ``options`` holds resolved option tokens — long options by
    canonical name, short options one character each, so a ``-fdn`` cluster
    contributes ``-f``, ``-d``, and ``-n`` in order.
    """
    force = False
    dry_run = False
    for option in options:
        if option in {"--force", "-f"}:
            force = True
        elif option == "--no-force":
            force = False
        if option in {"--dry-run", "-n"}:
            dry_run = True
        elif option == "--no-dry-run":
            dry_run = False
    return force, dry_run


_GIT_READER_SUBCOMMANDS = frozenset({
    "blame",
    "diff",
    "log",
    "show",
    "status",
    "whatchanged",
    "branch",
    "tag",
    "grep",
    "reflog",
    "stash",
    "shortlog",
    "rev-parse",
    "ls-files",
    "ls-tree",
    "describe",
    "show-ref",
    "for-each-ref",
    "rev-list",
    "cat-file",
    "count-objects",
    "diff-files",
    "diff-index",
    "diff-tree",
    "merge-base",
    "name-rev",
    "verify-pack",
    "check-attr",
    "check-ignore",
    "check-ref-format",
    "check-mailmap",
    "column",
    "get-tar-commit-id",
    "var",
})

_GIT_PAGING_SUBCOMMANDS = _GIT_READER_SUBCOMMANDS

# Explicitly recognized non-reader builtins retain their existing policy.
# Other subcommands remain gated, whether builtins or potential external helpers.
_GIT_NONREADER_SUBCOMMANDS = frozenset({
    "add",
    "am",
    "apply",
    "bisect",
    "checkout",
    "cherry-pick",
    "clean",
    "clone",
    "commit",
    "config",
    "fetch",
    "init",
    "merge",
    "mv",
    "pull",
    "push",
    "rebase",
    "remote",
    "reset",
    "restore",
    "revert",
    "rm",
    "switch",
    "worktree",
    "submodule",
    "help",
    "version",
    "archive",
    "notes",
})


# Known builtins without modeled operand/side-effect rules remain gated, but
# are not aliases. Do not treat these as readers (e.g. fsck --lost-found writes).
_GIT_UNMODELED_BUILTINS = frozenset({
    "bundle",
    "fsck",
    "gc",
    "maintenance",
    "reflog",
    "prune",
    "repack",
    "pack-objects",
    "index-pack",
    "unpack-objects",
    "update-index",
    "update-ref",
    "symbolic-ref",
    "hash-object",
    "write-tree",
    "read-tree",
    "commit-tree",
    "mktree",
    "merge-base",
    "merge-file",
    "merge-index",
    "merge-tree",
    "name-rev",
    "describe",
    "fast-export",
    "fast-import",
    "filter-branch",
    "format-patch",
    "send-email",
    "request-pull",
    "shortlog",
    "sparse-checkout",
    "multi-pack-index",
    "pack-refs",
    "verify-pack",
    "verify-commit",
    "verify-tag",
    "count-objects",
    "rerere",
    "replace",
    "credential",
    "credential-cache",
    "credential-store",
    "daemon",
    "http-backend",
    "upload-pack",
    "receive-pack",
    "upload-archive",
    "for-each-repo",
    "bugreport",
    "diagnose",
    "scalar",
    "whatchanged",
})


def _git_global_options(args: list[str]) -> tuple[int, list[str]] | None:
    """Walk git's global options; fail closed on unknown or config overrides.

    Returns the index of the first non-global token together with the global
    path values (-C/--git-dir/--work-tree), or None when the option sequence
    itself must be denied.
    """
    index = 0
    paths: list[str] = []
    while index < len(args) and args[index].startswith("-"):
        token = args[index]
        name = token.partition("=")[0]
        if token in {"-h", "-p", "-P", "-v", "--"} or name in _GIT_BOOLEAN_GLOBALS:
            index += 1
            continue
        parsed = _scan_options(args[index:], _GIT_GLOBAL_TABLE)
        known = (
            _match_long(
                token,
                _GIT_GLOBAL_TABLE.long_values | _GIT_GLOBAL_TABLE.optional_long_values,
            )
            if token.startswith("--")
            else token[:2] in {"-C", "-c"}
        )
        if not known:
            return None
        if name in {"--config-env"} or token.startswith("-c"):
            return None
        value_pairs = parsed.values[:1]
        if value_pairs and value_pairs[0][0] in {"-C", "--git-dir", "--work-tree"}:
            paths.append(value_pairs[0][1])
        consumed_value = bool(value_pairs) and (
            (
                token.startswith("--")
                and "=" not in token
                and not token.startswith("--exec-path")
            )
            or token in {"-C", "-c"}
        )
        index += 2 if consumed_value else 1
    return index, paths


def canonical_git_command(tokens: list[str]) -> list[str] | None:
    """Canonical git command form with recognized global options elided.

    The canonical form keeps the executable and the full subcommand suffix:
    ``git -C . stash drop`` becomes ``git stash drop``. Returns None when the
    executable is not git or the leading global options are unrecognized; the
    policy layer denies those spellings, so declining to normalize is safe.
    """
    if not tokens or Path(tokens[0]).name != "git":
        return None
    globals_result = _git_global_options(tokens[1:])
    if globals_result is None:
        return None
    index, _paths = globals_result
    return [tokens[0], *tokens[1 + index :]]


def push_destination_operands(call: ModeledGitCall) -> tuple[str, ...]:
    """The destination operands of a modeled push call.

    The first positional is always the repository, never a refspec, and
    ``--repo`` overrides it, so both name the destination a push runs
    receive-pack and hooks on.
    """
    destinations = list(call.operands[:1])
    destinations.extend(
        value for option, value in call.option_values if option == "--repo"
    )
    return tuple(destinations)


# git config is a permitted routine command, so the classifier scans it
# leniently: git itself rejects unmodeled options, and the key check below
# looks at every operand, so a misread option consumption can only
# over-block, never miss a write. Value options consume their operand
# exactly as git's parser does (-f/--file, -t/--type, --blob, --default,
# --comment); every other documented option is a boolean.
_GIT_CONFIG_TABLE = _OptionTable(
    frozenset({"--blob", "--comment", "--default", "--file", "--type"}),
    frozenset({"f", "t"}),
    boolean_long=frozenset({
        "--add",
        "--all",
        "--edit",
        "--fixed-value",
        "--get",
        "--get-all",
        "--get-color",
        "--get-colorbool",
        "--get-regexp",
        "--get-urlmatch",
        "--global",
        "--includes",
        "--list",
        "--local",
        "--name-only",
        "--null",
        "--replace-all",
        "--show-names",
        "--show-origin",
        "--show-scope",
        "--system",
        "--unset",
        "--unset-all",
        "--worktree",
    }),
)
_GIT_CONFIG_BOOLEAN_SHORTS = frozenset("elz")
# Read actions report configuration; they never write it.
_GIT_CONFIG_READ_OPTIONS = frozenset({
    "--get",
    "--get-all",
    "--get-color",
    "--get-colorbool",
    "--get-regexp",
    "--get-urlmatch",
    "--list",
    "-l",
})
# Write actions record a value.
_GIT_CONFIG_WRITE_LONGS = frozenset({"--add", "--replace-all"})
# Removal actions only ever remove configuration.
_GIT_CONFIG_UNSET_LONGS = frozenset({"--unset", "--unset-all"})
# Section operations act on section names, not keys.
_GIT_CONFIG_SECTION_LONGS = frozenset({"--rename-section", "--remove-section"})
# The newer subcommand spellings (git config set/get/list/...): the first
# operand is the action word, not a key.
_GIT_CONFIG_SUBCOMMANDS = frozenset({
    "edit",
    "get",
    "list",
    "remove-section",
    "rename-section",
    "set",
    "unset",
})
_GIT_CONFIG_READ_SUBCOMMANDS = frozenset({"get", "list"})
_GIT_CONFIG_MUTATION_SUBCOMMANDS = frozenset({
    "remove-section",
    "rename-section",
    "set",
    "unset",
})


@dataclass(frozen=True)
class _OptionTokenScan:
    """Scan result for one classified option token."""

    options: tuple[str, ...] = ()
    values: tuple[tuple[str, str], ...] = ()
    consumed: int = 1
    error: str | None = None


@dataclass(frozen=True)
class _StrictGitOptions:
    """Strictly scanned options of one classified git subcommand."""

    options: tuple[str, ...]
    values: tuple[tuple[str, str], ...]
    operands: tuple[str, ...]
    end_of_options: bool
    error: str | None


@dataclass(frozen=True)
class _GitGlobalScan:
    """Structured walk of git's global options for classification."""

    index: int
    relocation: GitRepositoryRelocation
    end_of_options: bool
    error: str | None


def _scan_required_long_value(
    option: str, token: str, args: list[str], index: int
) -> _OptionTokenScan:
    """Consume one required-value long option, in = or separated form."""
    _, separator, attached = token.partition("=")
    if separator:
        if not attached:
            return _OptionTokenScan(
                error=f"option {option!r} requires a non-empty value"
            )
        return _OptionTokenScan(options=(option,), values=((option, attached),))
    if index + 1 >= len(args):
        return _OptionTokenScan(error=f"option {option!r} requires a value")
    return _OptionTokenScan(
        options=(option,), values=((option, args[index + 1]),), consumed=2
    )


def _scan_long_option_token(
    token: str,
    args: list[str],
    index: int,
    table: _OptionTable,
    *,
    strict: bool,
    hard_guard_longs: frozenset[str],
) -> _OptionTokenScan:
    """Resolve one long option token against a classified subcommand's grammar."""
    name = token.partition("=")[0]
    if hard_guard_longs and _match_long(token, hard_guard_longs) is not None:
        return _OptionTokenScan(error=f"option {name!r} selects the program git runs")
    all_long = table.long_values | table.optional_long_values | table.boolean_long
    option = _match_long(token, all_long)
    if option is None:
        if strict:
            return _OptionTokenScan(error=f"unknown or ambiguous option {name!r}")
        return _OptionTokenScan(options=(name,))
    if option in table.boolean_long and "=" in token and strict:
        return _OptionTokenScan(error=f"option {option!r} takes no value")
    if option in table.long_values:
        return _scan_required_long_value(option, token, args, index)
    # Optional values attach only in = form; an empty attached value is valid
    # (e.g. --force-with-lease=). A boolean option given a value in lenient
    # mode lands here too and is recorded, not consumed.
    values = ((option, token.partition("=")[2]),) if "=" in token else ()
    return _OptionTokenScan(options=(option,), values=values)


def _scan_short_option_token(
    token: str,
    args: list[str],
    index: int,
    table: _OptionTable,
    *,
    boolean_shorts: frozenset[str],
    strict: bool,
) -> _OptionTokenScan:
    """Resolve one short-option token (a cluster) against the grammar."""
    options: list[str] = []
    values: list[tuple[str, str]] = []
    error: str | None = None
    consumed = 1
    chars = token[1:]
    for offset, short in enumerate(chars):
        name = f"-{short}"
        if short in table.short_values:
            attached = chars[offset + 1 :]
            options.append(name)
            if attached:
                values.append((name, attached))
            elif index + 1 < len(args):
                values.append((name, args[index + 1]))
                consumed = 2
            else:
                error = f"option {name!r} requires a value"
            break
        if short in table.optional_short_values:
            attached = chars[offset + 1 :]
            options.append(name)
            if attached:
                values.append((name, attached))
            break
        if short in boolean_shorts:
            options.append(name)
            continue
        if strict:
            error = f"unknown or ambiguous option {name!r}"
            break
        options.append(name)
    return _OptionTokenScan(tuple(options), tuple(values), consumed, error)


def _scan_classified_git_options(
    args: list[str],
    table: _OptionTable,
    *,
    boolean_shorts: frozenset[str] = frozenset(),
    hard_guard_longs: frozenset[str] = frozenset(),
    strict: bool = True,
) -> _StrictGitOptions:
    """Scan one classified git subcommand's arguments.

    Unlike _scan_options, classification must never misread a consumed
    operand as a flag, so every option is resolved against the subcommand's
    modeled grammar. Strict mode (push, switch, reset, clean) fails closed on
    any unknown option, ambiguous abbreviation, value given to a boolean
    option, empty required value, or dangling required value. Lenient mode
    (checkout, restore, stash, reflog) records unknown options without
    consuming them: git itself rejects unmodeled options, and the consequence
    of those subcommands does not depend on their options.
    """
    options: list[str] = []
    values: list[tuple[str, str]] = []
    operands: list[str] = []
    end_of_options = False
    error: str | None = None
    index = 0
    while index < len(args) and error is None:
        token = args[index]
        if end_of_options:
            operands.append(token)
            index += 1
        elif token == "--":
            end_of_options = True
            index += 1
        elif token.startswith("--"):
            step = _scan_long_option_token(
                token,
                args,
                index,
                table,
                strict=strict,
                hard_guard_longs=hard_guard_longs,
            )
            error = step.error
            options.extend(step.options)
            values.extend(step.values)
            index += step.consumed
        elif token.startswith("-") and token != "-":
            step = _scan_short_option_token(
                token, args, index, table, boolean_shorts=boolean_shorts, strict=strict
            )
            error = step.error
            options.extend(step.options)
            values.extend(step.values)
            index += step.consumed
        else:
            operands.append(token)
            index += 1
    return _StrictGitOptions(
        tuple(options), tuple(values), tuple(operands), end_of_options, error
    )


def _scan_git_globals(args: list[str]) -> _GitGlobalScan:
    """Walk git's global options, recording repository relocation operands.

    The recognized grammar mirrors _git_global_options exactly, so the
    classifier agrees with runtime enforcement on which spellings are known.
    Unlike the runtime walker, classification also fails closed on a dangling
    or empty required value and keeps -C/--git-dir/--work-tree operands
    structured.
    """
    index = 0
    cwd_paths: list[str] = []
    git_dir: str | None = None
    work_tree: str | None = None
    end_of_options = False
    error: str | None = None
    while index < len(args) and args[index].startswith("-"):
        token = args[index]
        name = token.partition("=")[0]
        if token in {"-h", "-p", "-P", "-v"} or name in _GIT_BOOLEAN_GLOBALS:
            index += 1
            continue
        if token == "--":
            end_of_options = True
            index += 1
            continue
        parsed = _scan_options(args[index:], _GIT_GLOBAL_TABLE)
        known = (
            _match_long(
                token,
                _GIT_GLOBAL_TABLE.long_values | _GIT_GLOBAL_TABLE.optional_long_values,
            )
            if token.startswith("--")
            else token[:2] in {"-C", "-c"}
        )
        if not known:
            error = f"unknown or ambiguous global option {name!r}"
            break
        if name == "--config-env" or token.startswith("-c"):
            error = f"global option {name!r} overrides git configuration"
            break
        value_pairs = parsed.values[:1]
        if value_pairs and not value_pairs[0][1]:
            error = f"global option {value_pairs[0][0]!r} requires a non-empty value"
            break
        if not value_pairs:
            # A required-value global at the end of the invocation has no operand.
            required = (
                _match_long(token, _GIT_GLOBAL_TABLE.long_values)
                if token.startswith("--")
                else token[:2] == "-C"
            )
            if required is not None:
                error = f"global option {name!r} requires a value"
                break
        elif value_pairs[0][0] == "-C":
            cwd_paths.append(value_pairs[0][1])
        elif value_pairs[0][0] == "--git-dir":
            git_dir = value_pairs[0][1]
        elif value_pairs[0][0] == "--work-tree":
            work_tree = value_pairs[0][1]
        consumed_value = bool(value_pairs) and (
            (
                token.startswith("--")
                and "=" not in token
                and not token.startswith("--exec-path")
            )
            or token in {"-C", "-c"}
        )
        index += 2 if consumed_value else 1
    relocation = GitRepositoryRelocation(
        cwd_paths=tuple(cwd_paths), git_dir=git_dir, work_tree=work_tree
    )
    return _GitGlobalScan(index, relocation, end_of_options, error)


@dataclass(frozen=True)
class _GitConfigCall:
    """One scanned ``git config`` call, shared by classifier and runtime policy.

    Both layers must agree on which calls write: the classifier guards the
    push-effecting and program-selection keys, and the runtime policy governs
    the ``--file`` destination of a writing call. ``operands`` holds the
    positional operands with the subcommand word removed.
    """

    options: frozenset[str]
    operands: tuple[str, ...]
    subcommand: str | None
    values: tuple[tuple[str, str], ...]
    error: str | None
    reads: bool
    edits: bool
    mutates: bool
    section_operation: bool
    renames: bool


def _scan_git_config_call(args: list[str]) -> _GitConfigCall:
    """Scan one git config call into its write-relevant shape.

    The scan is lenient — git itself rejects unmodeled options — and the
    key check looks at every operand, not only the first, so an option
    whose value consumption is misread can only over-block, never miss a
    write. The only scan errors are a dangling or empty required value at
    the end of the invocation, which leaves every earlier --file value
    already extracted.
    """
    scan = _scan_classified_git_options(
        args, _GIT_CONFIG_TABLE, boolean_shorts=_GIT_CONFIG_BOOLEAN_SHORTS, strict=False
    )
    options = set(scan.options)
    operands = list(scan.operands)
    subcommand: str | None = None
    if operands and operands[0] in _GIT_CONFIG_SUBCOMMANDS:
        subcommand = operands.pop(0)
    reads = bool(options & _GIT_CONFIG_READ_OPTIONS) or (
        subcommand in _GIT_CONFIG_READ_SUBCOMMANDS
    )
    edits = bool(options & {"--edit", "-e"}) or subcommand == "edit"
    section_operation = bool(options & _GIT_CONFIG_SECTION_LONGS) or (
        subcommand in {"rename-section", "remove-section"}
    )
    mutates = (
        len(operands) > 1
        or bool(options & _GIT_CONFIG_WRITE_LONGS)
        or bool(options & _GIT_CONFIG_UNSET_LONGS)
        or bool(options & _GIT_CONFIG_SECTION_LONGS)
        or subcommand in _GIT_CONFIG_MUTATION_SUBCOMMANDS
    )
    return _GitConfigCall(
        options=frozenset(options),
        operands=tuple(operands),
        subcommand=subcommand,
        values=scan.values,
        error=scan.error,
        reads=reads,
        edits=edits,
        mutates=mutates,
        section_operation=section_operation,
        renames=subcommand == "rename-section" or "--rename-section" in options,
    )


# Position of the subcommand token in a canonical git command.
_GIT_SUBCOMMAND_INDEX = 1
