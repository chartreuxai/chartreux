from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import re

from chartreux.core.dangerous_commands import (
    GitRepositoryRelocation as GitRepositoryRelocation,
)
from chartreux.core.tools.builtins._shell_argv import (
    _COMBINED_SHORT_OPTION_MIN_LENGTH as _COMBINED_SHORT_OPTION_MIN_LENGTH,
    _SHORT_OPTION_LENGTH as _SHORT_OPTION_LENGTH,
    _argument_matches as _argument_matches,
    _command_name,
    _has_long,
    _has_short,
    _has_short_in_cluster,
    _is_combined_short_options as _is_combined_short_options,
    _match_long,
    _options_before_double_dash,
    _OptionTable,
    _ParsedOptions as _ParsedOptions,
    _scan_options,
    matches_command_prefix as matches_command_prefix,
)
from chartreux.core.tools.builtins._shell_diagnostics import Diagnostic
from chartreux.core.tools.builtins._shell_executor_grammar import (
    _INLINE_SWITCHES as _INLINE_SWITCHES,
    _MAX_EXECUTOR_DEPTH as _MAX_EXECUTOR_DEPTH,
    _PACKAGE_PATH_COMMANDS,
    _PACKAGE_PATH_OPTIONS,
    EXECUTOR_REGISTRY,
    ArgumentRange as ArgumentRange,
    BoundaryKind as BoundaryKind,
    ExecutorBoundary as ExecutorBoundary,
    ExecutorGrammar as ExecutorGrammar,
    _executor_option as _executor_option,
    _executor_package_policy,
    _module_executor_command as _module_executor_command,
    _package_local_path,
    _package_module_tokens,
    _package_requirement_path,
    _short_interpreter_option as _short_interpreter_option,
    executor_boundary,
    executor_owned_tokens,
    inline_interpreter_switch as inline_interpreter_switch,
    is_package_command,
    package_argument_indexes as package_argument_indexes,
)
from chartreux.core.tools.builtins._shell_git_classifier import (
    _DANGEROUS_GIT_SUBCOMMANDS as _DANGEROUS_GIT_SUBCOMMANDS,
    _GIT_CHECKOUT_TABLE as _GIT_CHECKOUT_TABLE,
    _GIT_CLEAN_BOOLEAN_SHORTS as _GIT_CLEAN_BOOLEAN_SHORTS,
    _GIT_CLEAN_TABLE as _GIT_CLEAN_TABLE,
    _GIT_CONFIG_BRANCH_PUSH_EFFECTING_KEYS as _GIT_CONFIG_BRANCH_PUSH_EFFECTING_KEYS,
    _GIT_CONFIG_CORE_PROGRAM_KEYS as _GIT_CONFIG_CORE_PROGRAM_KEYS,
    _GIT_CONFIG_CORE_PROGRAM_SELECTION_KEYS as _GIT_CONFIG_CORE_PROGRAM_SELECTION_KEYS,
    _GIT_CONFIG_DIFF_PROGRAM_SELECTION_KEYS as _GIT_CONFIG_DIFF_PROGRAM_SELECTION_KEYS,
    _GIT_CONFIG_FILTER_PROGRAM_SELECTION_KEYS as _GIT_CONFIG_FILTER_PROGRAM_SELECTION_KEYS,
    _GIT_CONFIG_INCLUDE_SECTIONS as _GIT_CONFIG_INCLUDE_SECTIONS,
    _GIT_CONFIG_INTERACTIVE_PROGRAM_SELECTION_KEYS as _GIT_CONFIG_INTERACTIVE_PROGRAM_SELECTION_KEYS,
    _GIT_CONFIG_MAN_PROGRAM_SELECTION_KEYS as _GIT_CONFIG_MAN_PROGRAM_SELECTION_KEYS,
    _GIT_CONFIG_MERGE_PROGRAM_SELECTION_KEYS as _GIT_CONFIG_MERGE_PROGRAM_SELECTION_KEYS,
    _GIT_CONFIG_REMOTE_PUSH_EFFECTING_KEYS as _GIT_CONFIG_REMOTE_PUSH_EFFECTING_KEYS,
    _GIT_CONFIG_SEQUENCE_PROGRAM_SELECTION_KEYS as _GIT_CONFIG_SEQUENCE_PROGRAM_SELECTION_KEYS,
    _GIT_CONFIG_URL_PUSH_EFFECTING_KEYS as _GIT_CONFIG_URL_PUSH_EFFECTING_KEYS,
    _GIT_PUSH_BOOLEAN_SHORTS as _GIT_PUSH_BOOLEAN_SHORTS,
    _GIT_PUSH_FORCE_OPTIONS as _GIT_PUSH_FORCE_OPTIONS,
    _GIT_PUSH_HARD_GUARD_LONGS as _GIT_PUSH_HARD_GUARD_LONGS,
    _GIT_PUSH_TABLE as _GIT_PUSH_TABLE,
    _GIT_REFLOG_TABLE as _GIT_REFLOG_TABLE,
    _GIT_REMOTE_BOOLEAN_SHORTS as _GIT_REMOTE_BOOLEAN_SHORTS,
    _GIT_REMOTE_MUTATION_SUBCOMMANDS as _GIT_REMOTE_MUTATION_SUBCOMMANDS,
    _GIT_REMOTE_TABLE as _GIT_REMOTE_TABLE,
    _GIT_RESET_BOOLEAN_SHORTS as _GIT_RESET_BOOLEAN_SHORTS,
    _GIT_RESET_TABLE as _GIT_RESET_TABLE,
    _GIT_RESTORE_TABLE as _GIT_RESTORE_TABLE,
    _GIT_STASH_TABLE as _GIT_STASH_TABLE,
    _GIT_SWITCH_BOOLEAN_SHORTS as _GIT_SWITCH_BOOLEAN_SHORTS,
    _GIT_SWITCH_DISCARD_OPTIONS as _GIT_SWITCH_DISCARD_OPTIONS,
    _GIT_SWITCH_TABLE as _GIT_SWITCH_TABLE,
    _GIT_WORKTREE_TABLES as _GIT_WORKTREE_TABLES,
    _PUSH_URL_SCHEME as _PUSH_URL_SCHEME,
    _ROUTINE as _ROUTINE,
    _STASH_REFLOG_OPERATION_IDS as _STASH_REFLOG_OPERATION_IDS,
    _classify_git_config as _classify_git_config,
    _classify_git_hard_guard_subcommand as _classify_git_hard_guard_subcommand,
    _classify_git_push as _classify_git_push,
    _classify_git_remaining as _classify_git_remaining,
    _classify_git_remote as _classify_git_remote,
    _classify_git_stash_or_reflog as _classify_git_stash_or_reflog,
    _classify_git_switch as _classify_git_switch,
    _classify_git_worktree_subcommand as _classify_git_worktree_subcommand,
    _dispatch_git_subcommand as _dispatch_git_subcommand,
    _fail_closed_on_end_of_options as _fail_closed_on_end_of_options,
    _git_config_key_is_program_selection as _git_config_key_is_program_selection,
    _git_config_key_is_push_effecting as _git_config_key_is_push_effecting,
    _git_config_program_section_reason as _git_config_program_section_reason,
    _git_config_program_selection_reason as _git_config_program_selection_reason,
    _git_config_push_effecting_reason as _git_config_push_effecting_reason,
    _git_config_rename_guard_reason as _git_config_rename_guard_reason,
    _git_config_section_is_program_selection as _git_config_section_is_program_selection,
    _git_config_section_is_push_effecting as _git_config_section_is_push_effecting,
    _git_config_section_operation_guard_reason as _git_config_section_operation_guard_reason,
    _git_config_write_guard_reason as _git_config_write_guard_reason,
    _grantable_classification as _grantable_classification,
    _hard_denied as _hard_denied,
    _push_destination_fail_closed_reason as _push_destination_fail_closed_reason,
    _push_destination_is_explicit as _push_destination_is_explicit,
    _push_operation_ids as _push_operation_ids,
    _refspec_fail_closed_reason as _refspec_fail_closed_reason,
    classify_dangerous_git_command as classify_dangerous_git_command,
    is_dangerous_git_subcommand as is_dangerous_git_subcommand,
)
from chartreux.core.tools.builtins._shell_git_inspection import (
    _FALSE_GIT_CONFIG_VALUES as _FALSE_GIT_CONFIG_VALUES,
    _GIT_CONFIG_ENTRY as _GIT_CONFIG_ENTRY,
    _GIT_CONFIG_SECTION_HEADER as _GIT_CONFIG_SECTION_HEADER,
    _GIT_TREE_WRITING_SUBCOMMANDS as _GIT_TREE_WRITING_SUBCOMMANDS,
    _INCLUDE_DIRECTIVE_NAMES,
    _git_config_entries,
    _git_config_paths as _git_config_paths,
    _git_config_remote_url_names as _git_config_remote_url_names,
    _git_dir_config_paths as _git_dir_config_paths,
    _git_index_hook_risk,
    _git_value_is_active,
    _parse_git_config_entries as _parse_git_config_entries,
    git_configured_remote_names as git_configured_remote_names,
    git_cwd_relocations as git_cwd_relocations,
    git_metadata_paths as git_metadata_paths,
    git_push_destination_risk as git_push_destination_risk,
    git_work_tree_writing_subcommand as git_work_tree_writing_subcommand,
    resolve_git_work_tree as resolve_git_work_tree,
    resolve_repository_scope as resolve_repository_scope,
)
from chartreux.core.tools.builtins._shell_git_policy import (
    _GIT_APPLY_TABLE as _GIT_APPLY_TABLE,
    _GIT_APPLY_UNSAFE_PATHS_REASON as _GIT_APPLY_UNSAFE_PATHS_REASON,
    _GIT_ARCHIVE_OUTPUT_TABLE as _GIT_ARCHIVE_OUTPUT_TABLE,
    _GIT_INIT_TABLE as _GIT_INIT_TABLE,
    _GIT_SUBCOMMAND_TABLE as _GIT_SUBCOMMAND_TABLE,
    _GIT_SUBMODULE_ADD_TABLE as _GIT_SUBMODULE_ADD_TABLE,
    _GIT_WORKTREE_ADD_TABLE as _GIT_WORKTREE_ADD_TABLE,
    _GIT_WORKTREE_MOVE_TABLE as _GIT_WORKTREE_MOVE_TABLE,
    _git_archive_output_values as _git_archive_output_values,
    _git_destination_operands as _git_destination_operands,
    _git_destructive_policy as _git_destructive_policy,
    _git_policy,
    _legacy_clean_flags as _legacy_clean_flags,
)
from chartreux.core.tools.builtins._shell_git_support import (
    _GIT_BOOLEAN_GLOBALS as _GIT_BOOLEAN_GLOBALS,
    _GIT_CONFIG_BOOLEAN_SHORTS as _GIT_CONFIG_BOOLEAN_SHORTS,
    _GIT_CONFIG_MUTATION_SUBCOMMANDS as _GIT_CONFIG_MUTATION_SUBCOMMANDS,
    _GIT_CONFIG_READ_OPTIONS as _GIT_CONFIG_READ_OPTIONS,
    _GIT_CONFIG_READ_SUBCOMMANDS as _GIT_CONFIG_READ_SUBCOMMANDS,
    _GIT_CONFIG_SECTION_LONGS as _GIT_CONFIG_SECTION_LONGS,
    _GIT_CONFIG_SUBCOMMANDS as _GIT_CONFIG_SUBCOMMANDS,
    _GIT_CONFIG_TABLE as _GIT_CONFIG_TABLE,
    _GIT_CONFIG_UNSET_LONGS as _GIT_CONFIG_UNSET_LONGS,
    _GIT_CONFIG_WRITE_LONGS as _GIT_CONFIG_WRITE_LONGS,
    _GIT_GLOBAL_TABLE as _GIT_GLOBAL_TABLE,
    _GIT_NONREADER_SUBCOMMANDS as _GIT_NONREADER_SUBCOMMANDS,
    _GIT_PAGING_SUBCOMMANDS,
    _GIT_READER_SUBCOMMANDS,
    _GIT_SUBCOMMAND_INDEX,
    _GIT_UNMODELED_BUILTINS as _GIT_UNMODELED_BUILTINS,
    _effective_clean_flags as _effective_clean_flags,
    _git_global_options,
    _GitConfigCall as _GitConfigCall,
    _GitGlobalScan as _GitGlobalScan,
    _OptionTokenScan as _OptionTokenScan,
    _scan_classified_git_options as _scan_classified_git_options,
    _scan_git_config_call as _scan_git_config_call,
    _scan_git_globals as _scan_git_globals,
    _scan_long_option_token as _scan_long_option_token,
    _scan_required_long_value as _scan_required_long_value,
    _scan_short_option_token as _scan_short_option_token,
    _StrictGitOptions as _StrictGitOptions,
    canonical_git_command as canonical_git_command,
    push_destination_operands as push_destination_operands,
)
from chartreux.core.tools.builtins._shell_policy_types import ShellCommandPolicy

# GNU chmod's octal-mode value ceiling; larger values fail as "mode too large".
_CHMOD_MAX_NUMERIC_MODE = 0o7777
_MAX_LESS_RESUMED_OPTIONS = 256


_TABLES = {
    "date": _OptionTable(
        frozenset({"--date", "--file", "--reference", "--set", "--rfc-3339"}),
        frozenset("dfrs"),
        frozenset({"--iso-8601"}),
        frozenset("I"),
    ),
    "diff": _OptionTable(
        frozenset({
            "--changed-group-format",
            "--exclude-from",
            "--from-file",
            "--horizon-lines",
            "--ignore-matching-lines",
            "--label",
            "--line-format",
            "--new-group-format",
            "--new-line-format",
            "--old-group-format",
            "--old-line-format",
            "--show-function-line",
            "--starting-file",
            "--tabsize",
            "--to-file",
            "--unchanged-group-format",
            "--unchanged-line-format",
            "--width",
        }),
        frozenset("FILSWX"),
    ),
    "du": _OptionTable(
        frozenset({
            "--block-size",
            "--exclude",
            "--exclude-from",
            "--files0-from",
            "--max-depth",
            "--threshold",
            "--time-style",
        }),
        frozenset("BdtX"),
        frozenset({"--time"}),
    ),
    "wc": _OptionTable(frozenset({"--files0-from"})),
    "grep": _OptionTable(
        frozenset({
            "--after-context",
            "--before-context",
            "--binary-files",
            "--context",
            "--directories",
            "--exclude",
            "--exclude-dir",
            "--exclude-from",
            "--file",
            "--group-separator",
            "--include",
            "--label",
            "--max-count",
            "--regexp",
        }),
        frozenset("ABCDdefm"),
        boolean_long=frozenset({"--binary"}),
    ),
    "file": _OptionTable(
        frozenset({
            "--exclude",
            "--exclude-quiet",
            "--files-from",
            "--magic-file",
            "--parameter",
            "--separator",
        }),
        frozenset("eFfmP"),
    ),
    "sort": _OptionTable(
        frozenset({
            "--batch-size",
            "--buffer-size",
            "--compress-program",
            "--files0-from",
            "--key",
            "--output",
            "--parallel",
            "--random-source",
            "--sort",
            "--temporary-directory",
            "--field-separator",
        }),
        frozenset("koSTt"),
    ),
    "tree": _OptionTable(
        frozenset({
            "--charset",
            "--filelimit",
            "--gitfile",
            "--output",
            "--sort",
            "--timefmt",
        }),
        frozenset("HILPTo"),
    ),
}


_BSD_DATE_COMPONENT_LENGTHS = frozenset({2, 4, 6, 8, 10, 12})
_BSD_DATE_SECONDS_LENGTH = 2


def _date_short_option_state(token: str) -> tuple[bool, bool, bool]:
    """Return -j seen, -f seen, and whether the next argument is consumed."""
    has_no_set = False
    has_input_format = False
    value_options = frozenset({"d", "f", "r", "v", "z"})
    for index, option in enumerate(token[1:]):
        has_no_set = has_no_set or option == "j"
        has_input_format = has_input_format or option == "f"
        if option == "I":
            break
        if option in value_options:
            return has_no_set, has_input_format, index + 2 == len(token)
    return has_no_set, has_input_format, False


def _matches_bsd_date_setting_operand(value: str) -> bool:
    date_part, separator, seconds = value.partition(".")
    return (
        date_part.isdigit()
        and len(date_part) in _BSD_DATE_COMPONENT_LENGTHS
        and (
            not separator
            or (len(seconds) == _BSD_DATE_SECONDS_LENGTH and seconds.isdigit())
        )
    )


def _date_has_setting_operand(args: list[str]) -> bool:
    """Detect BSD date's positional clock-setting forms.

    GNU date has no valid non-option operand other than an output format, so
    conservatively applying this grammar on every POSIX platform only turns
    otherwise-invalid GNU invocations into denials.
    """
    has_no_set = False
    has_input_format = False
    positional: list[str] = []
    skip_next = False
    options_ended = False
    long_value_options = frozenset({"--date", "--file", "--reference", "--rfc-3339"})

    for token in args:
        if skip_next:
            skip_next = False
            continue
        if options_ended:
            if not token.startswith("+"):
                positional.append(token)
            continue
        if token == "--":
            options_ended = True
            continue
        if token.startswith("+"):
            continue
        if token.startswith("--"):
            option, separator, _value = token.partition("=")
            if not separator and option in long_value_options:
                skip_next = True
            continue
        if token.startswith("-") and token != "-":
            token_no_set, token_input_format, skip_next = _date_short_option_state(
                token
            )
            has_no_set = has_no_set or token_no_set
            has_input_format = has_input_format or token_input_format
            continue
        positional.append(token)

    if has_no_set or not positional:
        return False
    if has_input_format:
        return True

    # BSD's positional setter uses [[[[[cc]yy]mm]dd]HH]MM[.ss]. Avoid denying
    # arbitrary invalid GNU operands merely because they are positional.
    return any(_matches_bsd_date_setting_operand(value) for value in positional)


def _standard_policy(command: str, args: list[str]) -> ShellCommandPolicy:
    table = _TABLES[command]
    parsed = _scan_options(args, table)
    approval_long: tuple[str, ...] = ()
    approval_short = frozenset()
    path_options: tuple[str, ...] = ()
    inspect = False
    if command == "sort":
        approval_long = (
            "--compress-program",
            "--files0-from",
            "--output",
            "--temporary-directory",
        )
        approval_short = frozenset("oT")
        path_options = ("--random-source",)
    elif command == "grep":
        path_options = ("--exclude-from", "--file", "-f")
    elif command == "file":
        approval_long = (
            "--compile",
            "--files-from",
            "--uncompress",
            "--uncompress-noreport",
        )
        approval_short = frozenset("CfzZ")
        path_options = ("--files-from", "-f", "--magic-file", "-m")
    elif command == "du":
        # --files0-from reads a list of paths from file content, which static
        # path analysis cannot see.
        approval_long = ("--files0-from",)
        path_options = ("--files0-from", "--exclude-from", "-X")
    elif command == "wc":
        approval_long = ("--files0-from",)
        path_options = ("--files0-from",)
    elif command == "date":
        approval_long = ("--set",)
        approval_short = frozenset("s")
        path_options = ("--file", "-f", "--reference", "-r")
    elif command == "diff":
        path_options = ("--exclude-from", "-X", "--from-file", "--to-file")
    elif command == "tree":
        approval_long = ("--output",)
        approval_short = frozenset("o")
        path_options = ("--gitfile",)
        inspect = True
    path_values = parsed.values_for(*path_options)
    if command == "file":
        path_values = tuple(
            path
            for option, value in parsed.values
            if option in path_options
            for path in value.split(":")
        )
    if command == "tree":
        # tree resumes short-cluster scanning after a value-taking short, so
        # any "o" in a cluster reaches the output-file option.
        requires_approval = _has_long(parsed, *approval_long) or _has_short_in_cluster(
            parsed, approval_short
        )
    else:
        requires_approval = _has_long(parsed, *approval_long) or _has_short(
            parsed, approval_short, table
        )
    if command == "date":
        requires_approval = requires_approval or _date_has_setting_operand(args)
    return ShellCommandPolicy(
        requires_approval=requires_approval,
        inspect_positional_paths=inspect,
        option_path_values=path_values,
        positional_values=parsed.positionals,
    )


_CHECKSUM_TABLE = _OptionTable(frozenset({"--check"}))


def _checksum_policy(args: list[str]) -> ShellCommandPolicy:
    """md5sum/sha1sum/sha256sum/shasum --check reads paths from file content."""
    parsed = _scan_options(args, _CHECKSUM_TABLE)
    return ShellCommandPolicy(
        requires_approval=_has_long(parsed, "--check")
        or _has_short(parsed, frozenset("c"), _CHECKSUM_TABLE)
    )


def _uniq_policy(args: list[str]) -> ShellCommandPolicy:
    # A second positional operand is uniq's output file, the positional
    # counterpart of sort --output.
    operand_count_with_output = 2
    positional_count = 0
    skip_next = False
    options_ended = False
    for token in args:
        if skip_next:
            skip_next = False
            continue
        if options_ended:
            positional_count += 1
            continue
        if token == "--":
            options_ended = True
            continue
        if token.startswith("--"):
            option, separator, _value = token.partition("=")
            if not separator and option in {
                "--check-chars",
                "--skip-chars",
                "--skip-fields",
            }:
                skip_next = True
            continue
        is_legacy_plus_option = token.startswith("+") and token[1:].isdigit()
        if (token.startswith("-") and token != "-") or is_legacy_plus_option:
            for index, option in enumerate(token[1:]):
                if option in {"f", "s", "w"}:
                    if index + 2 == len(token):
                        skip_next = True
                    break
            continue
        positional_count += 1
    return ShellCommandPolicy(
        requires_approval=positional_count >= operand_count_with_output
    )


_LESS_LONG_VALUE_OPTIONS = frozenset({
    "--autosave",
    "--buffers",
    "--cmd",
    "--color",
    "--emouse",
    "--end-prompt",
    "--jump-target",
    "--lesskey-content",
    "--lesskey-context",
    "--lesskey-file",
    "--lesskey-src",
    "--log-file",
    "--max-back-scroll",
    "--max-forw-scroll",
    "--pattern",
    "--prompt",
    "--quotes",
    "--rscroll",
    "--shift",
    "--tabs",
    "--tag",
    "--tag-file",
    "--window",
})
_LESS_APPROVAL_LONG_OPTIONS = frozenset({
    "--autosave",
    "--cmd",
    "--lesskey-content",
    "--lesskey-context",
    "--lesskey-file",
    "--lesskey-src",
    "--log-file",
    "--tag",
    "--tag-file",
})
_LESS_SHORT_VALUE_OPTIONS = frozenset('"#DbhjkopPtTxyz')
_LESS_SHORT_NUMERIC_VALUE_OPTIONS = frozenset("#bhjxyz")
_LESS_LONG_NUMERIC_VALUE_OPTIONS = frozenset({
    "--buffers",
    "--header",
    "--jump-target",
    "--line-num-width",
    "--match-shift",
    "--max-back-scroll",
    "--max-forw-scroll",
    "--modelines",
    "--shift",
    "--status-col-width",
    "--tabs",
    "--wheel-lines",
    "--window",
})
_LESS_LONG_STRING_VALUE_OPTIONS = _LESS_LONG_VALUE_OPTIONS | frozenset({
    "--intr",
    "--search-options",
})
_LESS_APPROVAL_SHORT_OPTIONS = frozenset({"k", "o", "O", "t", "T"})


def _matches_long_option(token: str, option: str) -> bool:
    return _match_long(token, frozenset({option})) is not None


def _normalize_less_long_option(token: str) -> str:
    # less accepts ``--+name`` as its long reset/default spelling. String-valued
    # option handlers can still consume their supplied configuration value.
    if token.startswith("--+"):
        token = "--" + token[3:]
    option, separator, value = token.partition("=")
    return option.lower() + separator + value


def _less_startup_requires_approval(token: str) -> bool:
    command = token[2:] if token.startswith("++") else token[1:]
    if command in {"g", "G"} or (
        command and all("0" <= character <= "9" for character in command)
    ):
        return False
    if not (command.startswith(("/", "?")) and command.isprintable()):
        return True
    return _less_string_value_requires_approval(command[1:])


def _less_numeric_value_requires_approval(value: str) -> bool:
    """Check options that less resumes parsing after a numeric value."""
    index = 0
    if (
        len(value) > 1
        and value[0] == "-"
        and ("0" <= value[1] <= "9" or value[1] == ".")
    ):
        index = 1
    while index < len(value) and (
        "0" <= value[index] <= "9" or value[index] in {".", ","}
    ):
        index += 1
    suffix = value[index:]
    if not suffix:
        return False
    if not suffix.startswith(("-", "+")):
        suffix = "-" + suffix
    return _less_policy([suffix]).requires_approval


def _less_string_value_requires_approval(value: str) -> bool:
    """Check the option suffix after less's attached-string ``$`` terminator."""
    _, separator, suffix = value.partition("$")
    if not separator or not suffix:
        return False
    if not suffix.startswith(("-", "+")):
        suffix = "-" + suffix
    return _less_policy([suffix]).requires_approval


def _less_long_option_requires_approval(token: str) -> bool:
    if any(
        _matches_long_option(token, option) for option in _LESS_APPROVAL_LONG_OPTIONS
    ):
        return True
    option, separator, attached_value = token.partition("=")
    if not separator:
        return False
    if any(
        _matches_long_option(option, numeric_option)
        for numeric_option in _LESS_LONG_NUMERIC_VALUE_OPTIONS
    ):
        return _less_numeric_value_requires_approval(attached_value)
    if any(
        _matches_long_option(option, string_option)
        for string_option in _LESS_LONG_STRING_VALUE_OPTIONS
    ):
        return _less_string_value_requires_approval(attached_value)
    # New less releases may add string-valued options. Their attached values
    # share the ``$`` terminator grammar, so fail closed on a risky resumed suffix
    # even before the option name is added to the version-specific table.
    if "$" in attached_value:
        return _less_string_value_requires_approval(attached_value)
    return False


def _less_short_option_action(token: str) -> tuple[bool, bool]:
    """Return whether this token requires approval and consumes the next token."""
    index = 1
    while index < len(token):
        remainder = token[index:]
        if remainder.startswith("--"):
            normalized = _normalize_less_long_option(remainder)
            return _less_long_option_requires_approval(normalized), False

        option = token[index]
        if option == "$":
            index += 1
            continue
        if option == "+":
            return _less_startup_requires_approval(remainder), False
        if "0" <= option <= "9":
            return _less_numeric_value_requires_approval(remainder), False
        if option in _LESS_APPROVAL_SHORT_OPTIONS:
            return True, False
        if option not in _LESS_SHORT_VALUE_OPTIONS:
            index += 1
            continue
        attached_value = token[index + 1 :]
        if option in _LESS_SHORT_NUMERIC_VALUE_OPTIONS:
            requires_approval = bool(
                attached_value and _less_numeric_value_requires_approval(attached_value)
            )
        else:
            requires_approval = bool(
                attached_value and _less_string_value_requires_approval(attached_value)
            )
        return requires_approval, not attached_value
    return False, False


def _less_policy(args: list[str]) -> ShellCommandPolicy:
    """less's option grammar can start shell commands and lesskey programs.

    Ported from upstream v2.25.7: startup commands (``+!cmd``), lesskey files,
    tag files, and option values whose ``$`` terminator resumes option parsing
    are all code-execution or config-loading vectors that token-shape analysis
    cannot model.
    """
    if any(
        not argument.isprintable() or argument.count("$") > _MAX_LESS_RESUMED_OPTIONS
        for argument in args
    ):
        # Resumed options recursively re-enter this parser; bound the work.
        return ShellCommandPolicy(requires_approval=True)

    for argument in args:
        _, separator, resumed = argument.partition("$")
        if not separator or not (resumed := resumed.lstrip()):
            continue
        if not resumed.startswith(("-", "+")):
            resumed = "-" + resumed
        if _less_policy([resumed]).requires_approval:
            return ShellCommandPolicy(requires_approval=True)

    skip_next = False
    option_tokens = [
        (token, argument == "--") for argument in args for token in argument.split()
    ]
    for token, ends_options in option_tokens:
        if token == "--" and ends_options:
            # less ends option scanning at "--" instead of feeding it to a
            # value-taking option, which then errors for lack of a value
            # (verified against less 668); a later "-o" is a filename.
            break
        if skip_next:
            skip_next = False
            continue
        if token.startswith("+") and _less_startup_requires_approval(token):
            return ShellCommandPolicy(requires_approval=True)
        if token.startswith("--"):
            normalized = _normalize_less_long_option(token)
            if _less_long_option_requires_approval(normalized):
                return ShellCommandPolicy(requires_approval=True)
            # Only exact, version-stable spellings may hide the following token.
            # An ambiguous abbreviation or an option unknown to an older less
            # release does not consume its apparent value, which can expose a
            # following -k/--lesskey-* option to less instead.
            if "=" not in normalized and normalized in _LESS_LONG_VALUE_OPTIONS:
                skip_next = True
            continue
        if not token.startswith("-"):
            continue
        requires_approval, skip_next = _less_short_option_action(token)
        if requires_approval:
            return ShellCommandPolicy(requires_approval=True)
    return ShellCommandPolicy()


_FIND_VALUE_PREDICATES = frozenset({
    "-amin",
    "-anewer",
    "-atime",
    "-cmin",
    "-cnewer",
    "-context",
    "-ctime",
    "-fstype",
    "-gid",
    "-group",
    "-ilname",
    "-iname",
    "-inum",
    "-ipath",
    "-iregex",
    "-iwholename",
    "-links",
    "-lname",
    "-maxdepth",
    "-mindepth",
    "-mmin",
    "-mtime",
    "-name",
    "-newer",
    "-path",
    "-perm",
    "-printf",
    "-regex",
    "-regextype",
    "-samefile",
    "-size",
    "-type",
    "-uid",
    "-used",
    "-user",
    "-wholename",
    "-xtype",
})
_FIND_SIDE_EFFECTS = frozenset({
    "-delete",
    "-exec",
    "-execdir",
    "-fls",
    "-fprint",
    "-fprintf",
    "-fprint0",
    "-ok",
    "-okdir",
})


def _find_policy(args: list[str]) -> ShellCommandPolicy:
    index = 0
    while index < len(args) and (
        args[index] in {"-H", "-L", "-P"} or re.fullmatch(r"-O[0-3]", args[index])
    ):
        index += 1
    if index < len(args) and args[index] == "--":
        index += 1
    while index < len(args) and not (
        args[index].startswith("-") or args[index] in {"!", "(", ")"}
    ):
        index += 1
    while index < len(args):
        token = args[index]
        if token in _FIND_SIDE_EFFECTS:
            return ShellCommandPolicy(requires_approval=True)
        if token in _FIND_VALUE_PREDICATES or re.fullmatch(
            r"-newer[aBcmt][aBcmt]", token
        ):
            index += 2
        else:
            index += 1
    return ShellCommandPolicy()


def _rm_policy(args: list[str]) -> ShellCommandPolicy:
    for token in _options_before_double_dash(args):
        if (
            token.startswith("--rec")
            and "--recursive".startswith(token)
            or token.startswith("-")
            and not token.startswith("--")
            and any(flag in token[1:] for flag in "rR")
        ):
            return ShellCommandPolicy(
                requires_approval=True,
                diagnostic=Diagnostic("recursive_rm", offending_token=token),
            )
    return ShellCommandPolicy()


def _recursive_option(args: list[str]) -> bool:
    return any(
        token == "--recursive"
        or (token.startswith("-") and not token.startswith("--") and "R" in token[1:])
        for token in _options_before_double_dash(args)
    )


def _chmod_policy(args: list[str]) -> ShellCommandPolicy:
    if not _recursive_option(args):
        return ShellCommandPolicy()
    options_ended = False
    mode = ""
    for token in args:
        if token == "--" and not options_ended:
            options_ended = True
            continue
        if not options_ended and token.startswith("-"):
            continue
        mode = token
        break
    # GNU chmod accepts any number of leading-zero-padded octal digits and
    # rejects only values above 07777 ("mode too large"), so the last digit is
    # always the "others" permission class ("00007" means mode 0007).
    numeric_permissive = (
        bool(re.fullmatch(r"[0-7]+", mode))
        and int(mode, 8) <= _CHMOD_MAX_NUMERIC_MODE
        and int(mode[-1]) & 2 != 0
    )
    symbolic_permissive = any(
        (not who or "a" in who or "o" in who) and "w" in permissions
        for who, operator, permissions in re.findall(
            r"(?:^|,)([ugoa]*)([+=-])([^,]*)", mode
        )
        if operator in {"+", "="}
    )
    return ShellCommandPolicy(
        requires_approval=numeric_permissive or symbolic_permissive
    )


def _chown_policy(args: list[str]) -> ShellCommandPolicy:
    return ShellCommandPolicy(requires_approval=_recursive_option(args))


def git_repository_config_risk(  # noqa: PLR0911, PLR0912
    tokens: list[str], *, cwd: Path
) -> str | None:
    """The vector through which an otherwise-benign Git reader runs repo code.

    Adapts upstream v2.25.7's ``git_repository_requires_approval`` to
    chartreux's deny-only resolver: the returned string is the denial reason
    rather than an approval request. Upstream's ``git_repository_identity``
    (a stable approval-scope key) has no consumer here and is not ported.
    """
    policy = analyze_shell_command_policy(tokens)
    if not policy.inspect_git_repository or len(tokens) <= _GIT_SUBCOMMAND_INDEX:
        return None
    globals_result = _git_global_options(tokens[1:])
    if globals_result is None or globals_result[0] >= len(tokens) - 1:
        return None
    subcommand = tokens[1 + globals_result[0]].casefold()
    pager_enabled = True
    for option in tokens[1 : 1 + globals_result[0]]:
        if option in {"--no-pager", "-P"}:
            pager_enabled = False
        elif option in {"--paginate", "-p"}:
            pager_enabled = True

    def vector(key: str, remediation: str | None = None) -> str:
        # The resolver is deny-only with no passthrough, so the broadest
        # vectors carry the remediation the reader cannot discover itself.
        reason = f"repository git config may execute arbitrary code via {key}"
        return reason if remediation is None else f"{reason}; {remediation}"

    entries = _git_config_entries(cwd)
    if entries is None:
        return "repository git config is unreadable; failing closed"
    for section, key, _value in entries:
        # Includes can hide any of the executable settings checked below.
        if section in {"include", "includeif"}:
            return vector(
                f"{section}.{key}",
                f"audit or remove the [{_INCLUDE_DIRECTIVE_NAMES[section]}] "
                "directive in .git/config",
            )
    for section, key, value in entries:
        if not _git_value_is_active(value):
            continue
        if pager_enabled and subcommand in _GIT_PAGING_SUBCOMMANDS:
            if section == "core" and key == "pager":
                return vector("core.pager")
            if section == "pager" and key == subcommand:
                return vector(f"pager.{subcommand}")
    if subcommand in _GIT_READER_SUBCOMMANDS:
        for section, key, value in entries:
            if not _git_value_is_active(value):
                continue
            if section == "core" and key == "fsmonitor":
                return vector(
                    "core.fsmonitor",
                    "audit or remove the core.fsmonitor setting in .git/config",
                )
            if section == "filter" and key in {"clean", "process"}:
                return vector(
                    f"filter.{key}",
                    "audit or remove the [filter] command in .git/config",
                )
    if subcommand in _GIT_READER_SUBCOMMANDS:
        for section, key, value in entries:
            if not _git_value_is_active(value):
                continue
            if section == "diff" and key in {"command", "external", "textconv"}:
                return vector(f"diff.{key}")
    if subcommand == "log":
        for section, key, value in entries:
            if not _git_value_is_active(value):
                continue
            # Remerge output can invoke repository-owned merge drivers, and a
            # signature verifier runs the configured gpg program.
            if section == "merge" and key == "driver":
                return vector("merge.driver")
            if section == "gpg" and key == "program":
                return vector("gpg.program")
    if risk := _git_index_hook_risk(cwd, entries):
        return risk
    return None


_SED_TABLE = _OptionTable(
    long_values=frozenset({"--expression", "--file", "--line-length"}),
    short_values=frozenset("efl"),
    optional_long_values=frozenset({"--in-place"}),
    optional_short_values=frozenset("i"),
    boolean_long=frozenset({
        "--debug",
        "--follow-symlinks",
        "--help",
        "--null-data",
        "--posix",
        "--quiet",
        "--regexp-extended",
        "--sandbox",
        "--separate",
        "--silent",
        "--unbuffered",
        "--version",
    }),
)


def _sed_delimited_end(script: str, index: int, delimiter: str) -> int:
    while index < len(script):
        if script[index] == "\\":
            index += 2
        elif script[index] == delimiter:
            return index + 1
        else:
            index += 1
    raise ValueError("unterminated sed expression")


def _sed_script_paths(script: str) -> tuple[str, ...]:  # noqa: PLR0912
    """Inspect literal file commands without interpreting substitutions as code."""
    paths: list[str] = []
    index = 0
    while index < len(script):
        character = script[index]
        if character.isspace() or character in ";{},!+$" or character.isdigit():
            # `+` is the GNU relative-address joiner (`,+N`), the form the
            # statically expanded locate-then-print idiom produces. A `+`
            # outside an address is a sed runtime error, never a write, and
            # this scan continues linearly, so any later file command is
            # still found. The `first~step` address stays unmodeled.
            index += 1
            continue
        if character == "/":  # Regex address.
            index = _sed_delimited_end(script, index + 1, "/")
            continue
        # Alternate address delimiter.
        if character == "\\" and index + 1 < len(script):
            delimiter = script[index + 1]
            index = _sed_delimited_end(script, index + 2, delimiter)
            continue
        index += 1
        if character in "sy":
            if index >= len(script):
                raise ValueError("missing sed delimiter")
            delimiter = script[index]
            index = _sed_delimited_end(script, index + 1, delimiter)
            index = _sed_delimited_end(script, index, delimiter)
            if character == "y":
                continue
            # Substitution flags include a file-writing 'w' selector.
            while index < len(script) and script[index] not in ";\n}#":
                if script[index] == "w":
                    character = "w"
                    index += 1
                    break
                if script[index] not in "0123456789giImMp \t\r":
                    raise ValueError(
                        "unmodeled sed substitution flag (shell execution is not permitted)"
                    )
                index += 1
            else:
                continue
        if character in "rRwW":
            # GNU sed consumes the rest of the line as the filename, even ';'.
            end = script.find("\n", index)
            end = len(script) if end == -1 else end
            target = script[index:end].lstrip(" \t")
            if "\\" in target:
                raise ValueError("escaped sed filename cannot be safely inspected")
            if target:
                paths.append(target)
            index = end
        elif character in "aic#":
            # Text and comments are not sed commands.
            end = script.find("\n", index)
            index = len(script) if end == -1 else end
        elif character in "btT:":
            while index < len(script) and script[index] not in ";\n":
                index += 1
        elif character not in "dDgGhHlnNpPqx=z":
            # Unknown syntax may hide a file command; do not guess its boundary.
            raise ValueError("unsupported sed script syntax")
    return tuple(paths)


def _sed_policy(args: list[str]) -> ShellCommandPolicy:
    parsed = _scan_options(args, _SED_TABLE)
    has_script_option = _has_long(parsed, "--expression", "--file") or _has_short(
        parsed, frozenset("ef"), _SED_TABLE
    )
    inputs = parsed.positionals if has_script_option else parsed.positionals[1:]
    scripts = parsed.values_for("--expression", "-e")
    if not has_script_option:
        scripts = parsed.positionals[:1]
    if _has_long(parsed, "--file") or _has_short(parsed, frozenset("f"), _SED_TABLE):
        return ShellCommandPolicy(
            requires_approval=True,
            denial_reason="external sed scripts cannot be analyzed; inline the script instead",
        )
    paths: list[str] = []
    requires_approval = False
    denial_reason = None
    try:
        for script in scripts:
            paths.extend(_sed_script_paths(script))
    except ValueError as exc:
        requires_approval = True
        denial_reason = str(exc)
    for suffix in parsed.values_for("--in-place", "-i"):
        if "/" in suffix:
            requires_approval = True
        paths.extend(
            suffix.replace("*", operand) if "*" in suffix else operand + suffix
            for operand in inputs
            if suffix
        )
    return ShellCommandPolicy(
        requires_approval=requires_approval,
        denial_reason=denial_reason,
        mutates_files=_has_long(parsed, "--in-place")
        or _has_short(parsed, frozenset("i"), _SED_TABLE),
        inspect_positional_paths=True,
        option_path_values=tuple(paths),
        positional_values=inputs,
    )


def _package_policy(args: list[str], *, pip: bool = False) -> ShellCommandPolicy:
    table = _OptionTable(
        long_values=_PACKAGE_PATH_OPTIONS | frozenset({"--python"}),
        short_values=frozenset("tercdwo") if pip else frozenset("o"),
    )
    parsed = _scan_options(args, table)
    try:
        option_paths = tuple(
            _package_local_path(value)
            for value in parsed.values_for(
                *_PACKAGE_PATH_OPTIONS, "-t", "-e", "-r", "-c", "-d", "-w", "-o"
            )
        )
        positional_paths = tuple(
            path
            for value in parsed.positionals
            if (path := _package_requirement_path(value)) is not None
        )
    except ValueError:
        return ShellCommandPolicy(
            requires_approval=True, denial_reason="unmodeled local package URL"
        )
    return ShellCommandPolicy(
        inspect_positional_paths=True,
        option_path_values=option_paths,
        positional_values=positional_paths,
    )


# Compilers and linkers whose -o/--output operand is the written program: the
# output file is a write-classified path operand, so it is workspace-scoped and
# checked against protected Git metadata like every other write operand. The
# assembler (as) writes its object through -o as well; it spells no long form
# (--output would abbreviate --output-target), so the modeled long form is
# inert for it.
_COMPILER_OUTPUT_COMMANDS = frozenset({
    "gcc",
    "cc",
    "clang",
    "g++",
    "c++",
    "clang++",
    "rustc",
    "ld",
    "as",
})
_COMPILER_OUTPUT_TABLE = _OptionTable(
    long_values=frozenset({"--output"}), short_values=frozenset("o")
)


def _compiler_output_policy(args: list[str]) -> ShellCommandPolicy:
    parsed = _scan_options(args, _COMPILER_OUTPUT_TABLE)
    return ShellCommandPolicy(option_path_values=parsed.values_for("--output", "-o"))


# Write-operand options of the commands whose positional operands are already
# write-classified (the mutating path commands and the file-content commands).
# Every modeled option names a file or directory the command writes, so each
# extracted value is a write-classified path operand — in the separate form
# (-C .git), the attached short form (-C.git), and the attached long form
# (--output=.git/config) alike. Options that name read targets (tar -T,
# patch -i, rsync --exclude-from) are deliberately absent: only write
# operands are extracted here.
_WRITE_OPERAND_TABLES: dict[str, _OptionTable] = {
    # cp/mv/install/ln -t/--target-directory names the destination directory.
    "cp": _OptionTable(
        long_values=frozenset({"--target-directory"}), short_values=frozenset({"t"})
    ),
    "mv": _OptionTable(
        long_values=frozenset({"--target-directory"}), short_values=frozenset({"t"})
    ),
    "install": _OptionTable(
        long_values=frozenset({"--target-directory"}), short_values=frozenset({"t"})
    ),
    "ln": _OptionTable(
        long_values=frozenset({"--target-directory"}), short_values=frozenset({"t"})
    ),
    # tee --output-error[=MODE] carries the write-error mode; GNU's modes
    # (warn, exit, ...) are not paths, so extracting the attached value is
    # conservative over-blocking, never a missed write operand.
    "tee": _OptionTable(optional_long_values=frozenset({"--output-error"})),
    # patch writes the patched result (-o/--output), the rejects
    # (-r/--reject-file), and runs under a relocated directory
    # (-d/--directory); backup names are prefixed with -B/--prefix and
    # -Y/--basename-prefix.
    "patch": _OptionTable(
        long_values=frozenset({
            "--basename-prefix",
            "--directory",
            "--output",
            "--prefix",
            "--reject-file",
        }),
        short_values=frozenset({"B", "Y", "d", "o", "r"}),
    ),
    # tar -f/--file names the archive (written when creating, read when
    # extracting — the operand is write-classified either way, matching the
    # positional scan), -C/--directory the extraction destination, and
    # --add-file a member added to the archive.
    "tar": _OptionTable(
        long_values=frozenset({"--add-file", "--directory", "--file"}),
        short_values=frozenset({"C", "f"}),
    ),
    # unzip -d names the extraction destination.
    "unzip": _OptionTable(short_values=frozenset({"d"})),
    # iconv -o/--output names the written conversion result.
    "iconv": _OptionTable(
        long_values=frozenset({"--output"}), short_values=frozenset({"o"})
    ),
    # gpg's -o/--output names the written output file (the decrypted or
    # signed result): every spelling is real, including the attached short
    # form (-oFILE) and --output=FILE.
    "gpg": _OptionTable(
        long_values=frozenset({"--output"}), short_values=frozenset({"o"})
    ),
    # xmllint's -o/--output names the written XML result. The attached
    # spellings (-oFILE, --output=FILE) are rejected by xmllint itself, so
    # extracting them is a conservative superset, never a missed operand.
    "xmllint": _OptionTable(
        long_values=frozenset({"--output"}), short_values=frozenset({"o"})
    ),
    # rsync's directory/file options name where transferred, partial,
    # backup, and compared files and the log are written.
    "rsync": _OptionTable(
        long_values=frozenset({
            "--backup-dir",
            "--compare-dest",
            "--copy-dest",
            "--link-dest",
            "--log-file",
            "--partial-dir",
            "--temp-dir",
        }),
        short_values=frozenset({"T"}),
    ),
    # bsdtar is tar-compatible: -f/--file names the archive (written when
    # creating, read when extracting — write-classified either way, matching
    # tar) and -C/--directory the extraction destination.
    "bsdtar": _OptionTable(
        long_values=frozenset({"--directory", "--file"}),
        short_values=frozenset({"C", "f"}),
    ),
    # pax -f names the archive, written in write mode (-w) and read in read
    # mode (-r); write-classified either way, matching tar.
    "pax": _OptionTable(short_values=frozenset({"f"})),
    # ar's positional archive operand is created or replaced by the mutating
    # operations (inspected positionally, like tar's), and --output names
    # the extraction destination directory.
    "ar": _OptionTable(long_values=frozenset({"--output"})),
}


def _write_operand_policy(
    table: _OptionTable,
) -> Callable[[list[str]], ShellCommandPolicy]:
    """One command's write-operand extraction over its modeled option table.

    The table carries only write-operand options, so every value the scan
    consumes is a write-classified path operand. Unknown options never
    consume a following token, so a misread option consumption can only
    leave an extra candidate, never hide a write operand.
    """

    def policy(args: list[str]) -> ShellCommandPolicy:
        parsed = _scan_options(args, table)
        return ShellCommandPolicy(
            option_path_values=tuple(value for _, value in parsed.values)
        )

    return policy


def _openssl_output_policy(args: list[str]) -> ShellCommandPolicy:
    """openssl's -out/--out operand names the written output file.

    OpenSSL spells its output option with one or two dashes and accepts the
    value separately or attached with ``=`` (``-out FILE``, ``-out=FILE``,
    ``--out FILE``, ``--out=FILE``), which the short-cluster scanner cannot
    model; extract every spelling directly. Only the exact ``-out``/``--out``
    prefixes match, so longer options (``-outform``) never do.
    """
    values: list[str] = []
    for index, token in enumerate(args):
        if token in {"-out", "--out"} and index + 1 < len(args):
            values.append(args[index + 1])
        elif token.startswith(("-out=", "--out=")):
            values.append(token.partition("=")[2])
    return ShellCommandPolicy(option_path_values=tuple(values))


# split's value-taking options keep their operands (a SIZE, a SEPARATOR) out
# of the positional scan: only the trailing PREFIX operand names the written
# output files (PREFIXaa, PREFIXab, ...).
_SPLIT_TABLE = _OptionTable(
    long_values=frozenset({
        "--additional-suffix",
        "--bytes",
        "--filter",
        "--line-bytes",
        "--lines",
        "--number",
        "--separator",
        "--suffix-length",
    }),
    short_values=frozenset({"C", "a", "b", "l", "n", "t"}),
    optional_long_values=frozenset({"--hex-suffixes", "--numeric-suffixes"}),
)


def _split_output_policy(args: list[str]) -> ShellCommandPolicy:
    """split's PREFIX operand names the written output files.

    ``split [OPTION]... [INPUT [PREFIX]]`` writes PREFIXaa, PREFIXab, ...;
    with a single positional the input is read and the prefix defaults to
    ``x`` in the working directory, so only a second positional is a write
    target. The input operand is a read and stays governed by the workspace
    authority alone, exactly like every other reader operand.
    """
    parsed = _scan_options(args, _SPLIT_TABLE)
    positionals = parsed.positionals
    prefix = (positionals[-1],) if len(positionals) > 1 else ()
    return ShellCommandPolicy(inspect_positional_paths=True, positional_values=prefix)


# csplit's --prefix/-f operand names the written output files (PREFIX00,
# PREFIX01, ...); the positional operands are the input file and match
# patterns, not write targets.
_CSPLIT_TABLE = _OptionTable(
    long_values=frozenset({"--prefix"}), short_values=frozenset({"f"})
)


def _csplit_output_policy(args: list[str]) -> ShellCommandPolicy:
    parsed = _scan_options(args, _CSPLIT_TABLE)
    return ShellCommandPolicy(option_path_values=parsed.values_for("--prefix", "-f"))


# cpio's write operands: the -D/--directory directory the copy-in and
# pass-through modes change into before writing (the extraction/copy
# destination root), the -F/--file and -O archive files of copy-out mode,
# and the pass-through destination positional (modeled by _cpio_policy).
# Member names inside an archive stay uninspected (the documented member
# residual, same as tar). Options that name read targets (-I's archive, -E's
# pattern file) and non-path values (-H's format, -M's message, -W's flag,
# -C's io size, --block-size) are consumed but never extracted; cpio's
# optional-value -R/--owner is deliberately unmodeled, so its value stays a
# positional candidate — an unmodeled option can only over-block, never
# hide a write operand.
_CPIO_TABLE = _OptionTable(
    long_values=frozenset({
        "--block-size",
        "--directory",
        "--file",
        "--format",
        "--io-size",
        "--message",
        "--pattern-file",
        "--warning",
    }),
    short_values=frozenset({"C", "D", "E", "F", "H", "I", "M", "O", "W"}),
    boolean_long=frozenset({
        "--absolute-filenames",
        "--append",
        "--dereference",
        "--device-independent",
        "--dot",
        "--extract",
        "--force-local",
        "--ignore-devno",
        "--ignore-dirnlink",
        "--link",
        "--list",
        "--make-directories",
        "--no-absolute-filenames",
        "--no-preserve-owner",
        "--null",
        "--numeric-uid-gid",
        "--only-verify-crc",
        "--pass-through",
        "--preserve-modification-time",
        "--quiet",
        "--renumber-inodes",
        "--reproducible",
        "--reset-access-time",
        "--swap-bytes",
        "--swap-halfwords",
        "--to-stdout",
        "--unconditional",
        "--verbose",
    }),
)


def _cpio_policy(args: list[str]) -> ShellCommandPolicy:
    """cpio's write operands: pass-through destinations and archive files.

    Copy-pass mode (-p/--pass-through, -l/--link) copies the files named on
    standard input into its positional destination directory, so every
    positional is a write operand. Copy-in mode (-i) extracts archive members
    into the -D/--directory directory (or the cwd without it — the member
    residual), and copy-out mode (-o) writes the archive to -F/--file or -O.
    """
    parsed = _scan_options(args, _CPIO_TABLE)
    write_values = list(parsed.values_for("--directory", "-D", "--file", "-F", "-O"))
    passthrough = _has_long(parsed, "--pass-through", "--link") or _has_short(
        parsed, frozenset({"p", "l"}), _CPIO_TABLE
    )
    if passthrough:
        # Copy-pass mode's positional operand is the destination directory.
        return ShellCommandPolicy(
            inspect_positional_paths=True,
            option_path_values=tuple(write_values),
            positional_values=parsed.positionals,
        )
    return ShellCommandPolicy(
        option_path_values=tuple(write_values), positional_values=()
    )


# strip writes its result to -o when given and mutates every positional
# operand in place otherwise, so both are write operands: strip preserves
# the executable bit, so `strip -o .git/hooks/pre-commit evil` plants a
# hook exactly like the modeled `ld -o`. strip spells no long --output
# (--output abbreviates --output-target, which the table models so the
# abbreviation consumes its operand like strip itself), and its other
# value-taking options name formats, sections, and symbols — never files
# the command writes — so they are consumed but never extracted.
_STRIP_TABLE = _OptionTable(
    long_values=frozenset({
        "--input-target",
        "--keep-section",
        "--keep-symbol",
        "--output-target",
        "--remove-relocations",
        "--remove-section",
        "--strip-symbol",
        "--target",
    }),
    short_values=frozenset({"F", "I", "K", "N", "O", "R", "o"}),
)


def _strip_policy(args: list[str]) -> ShellCommandPolicy:
    parsed = _scan_options(args, _STRIP_TABLE)
    return ShellCommandPolicy(
        inspect_positional_paths=True,
        option_path_values=parsed.values_for("-o"),
        positional_values=parsed.positionals,
    )


# scp's grammar is `scp [options] source... destination`, where an operand
# carrying a colon before any slash is a REMOTE spec (outside this guard:
# it writes a remote host, never local metadata). A local-to-local copy is
# a mutating path command like ln: BOTH operands are write-classified and
# inspected, so a destination resolving into metadata is denied on
# containment and a source resolving into metadata is denied the same way
# (the ln rule). The value-taking options are modeled so their operands
# (ports, identity files, ssh options) never surface as positional
# candidates.
_SCP_TABLE = _OptionTable(
    short_values=frozenset({"c", "D", "F", "i", "J", "l", "o", "P", "S"})
)


def _scp_policy(args: list[str]) -> ShellCommandPolicy:
    parsed = _scan_options(args, _SCP_TABLE)
    return ShellCommandPolicy(
        inspect_positional_paths=True, positional_values=parsed.positionals
    )


# zip's first positional operand is the archive the call creates or
# replaces; the remaining positionals name the files it stores (reads,
# but classified like cp's source operand: a mutating path command
# inspects every operand). The value-taking options (the -b scratch
# directory, -i/-x patterns, -n suffixes, -P password) are modeled so
# their operands never surface as positional candidates.
_ZIP_TABLE = _OptionTable(short_values=frozenset({"P", "b", "i", "n", "x"}))


def _zip_policy(args: list[str]) -> ShellCommandPolicy:
    parsed = _scan_options(args, _ZIP_TABLE)
    return ShellCommandPolicy(
        inspect_positional_paths=True, positional_values=parsed.positionals
    )


# 7z's grammar is `7z <command> [switches] <archive> [files...]`: every
# positional operand is write-classified (the archive is created or
# replaced by the mutating commands and read by the listing ones —
# write-classified either way, matching tar's -f; the command letter and
# the stored-file operands are classified like cp's source operand).
# The -o switch names the extraction destination and is ATTACHED-ONLY in
# 7z (`-o.git` extracts into .git; `-o .git` is rejected by 7z itself, and
# `-o=.git` creates a directory literally named "=.git"), so it is an
# optional short value: the attached spelling extracts, and a separate
# value stays a positional candidate the scan still inspects.
_SEVENZIP_TABLE = _OptionTable(optional_short_values=frozenset({"o"}))


def _sevenzip_policy(args: list[str]) -> ShellCommandPolicy:
    parsed = _scan_options(args, _SEVENZIP_TABLE)
    return ShellCommandPolicy(
        inspect_positional_paths=True,
        option_path_values=parsed.values_for("-o"),
        positional_values=parsed.positionals,
    )


# unar extracts into -o's directory (default: the current directory) and
# spells the option in four forms: `-o DIR`, `-o=DIR`, and the single-dash
# long `-output-directory DIR` / `-output-directory=DIR`. The attached
# short form (`-oDIR`) is rejected by unar itself; the scanner's cluster
# rules cannot model single-dash long options, so the forms are extracted
# directly. Only these options name files unar writes: the positional
# operands name the archive and the members to extract (reads).
def _unar_output_policy(args: list[str]) -> ShellCommandPolicy:
    values: list[str] = []
    for index, token in enumerate(args):
        if token in {"-o", "-output-directory"} and index + 1 < len(args):
            values.append(args[index + 1])
        elif token.startswith(("-o=", "-output-directory=")):
            values.append(token.partition("=")[2])
    return ShellCommandPolicy(option_path_values=tuple(values))


# zstd (and its twins zstdmt/unzstd/zstdcat, which are the same binary)
# mutates every positional operand in place — bzip2/xz/lzma style — and
# additionally writes to -o's file (separate form only) and to the
# --output-dir-flat/--output-dir-mirror directories, in both the `=` and
# the separate spelling, so those options are extracted as write
# operands. The remaining value-taking options (levels, sizes, formats)
# name no written file and only ever over-block as leaked candidates.
_ZSTD_TABLE = _OptionTable(
    long_values=frozenset({"--output-dir-flat", "--output-dir-mirror"}),
    short_values=frozenset({"o"}),
)


def _zstd_policy(args: list[str]) -> ShellCommandPolicy:
    parsed = _scan_options(args, _ZSTD_TABLE)
    return ShellCommandPolicy(
        inspect_positional_paths=True,
        option_path_values=parsed.values_for(
            "-o", "--output-dir-flat", "--output-dir-mirror"
        ),
        positional_values=parsed.positionals,
    )


# mktemp creates a file at every template positional and in the directory
# -p/--tmpdir names (the long form takes its directory only attached:
# --tmpdir=DIR). The resolver's file-content surface inspects the
# template positionals; this policy adds the -p/--tmpdir directories,
# whose separate spelling also surfaces as a positional candidate and
# whose attached spelling only the scan can reach. --suffix names no
# directory (a slash in it fails), so it is consumed but not extracted.
_MKTEMP_TABLE = _OptionTable(
    long_values=frozenset({"--suffix"}),
    short_values=frozenset({"p"}),
    optional_long_values=frozenset({"--tmpdir"}),
)


def _mktemp_policy(args: list[str]) -> ShellCommandPolicy:
    parsed = _scan_options(args, _MKTEMP_TABLE)
    return ShellCommandPolicy(option_path_values=parsed.values_for("-p", "--tmpdir"))


_COMMAND_POLICIES = {
    "chmod": _chmod_policy,
    "chown": _chown_policy,
    "date": lambda args: _standard_policy("date", args),
    "diff": lambda args: _standard_policy("diff", args),
    "du": lambda args: _standard_policy("du", args),
    "file": lambda args: _standard_policy("file", args),
    "find": _find_policy,
    "git": _git_policy,
    "gsed": _sed_policy,
    "grep": lambda args: _standard_policy("grep", args),
    "less": _less_policy,
    "md5sum": _checksum_policy,
    "more": _less_policy,
    "rm": _rm_policy,
    "sed": _sed_policy,
    "sha1sum": _checksum_policy,
    "sha256sum": _checksum_policy,
    "shasum": _checksum_policy,
    "sort": lambda args: _standard_policy("sort", args),
    "tree": lambda args: _standard_policy("tree", args),
    "uniq": _uniq_policy,
    "wc": lambda args: _standard_policy("wc", args),
    "xargs": lambda args: ShellCommandPolicy(requires_approval=True),
}
_COMMAND_POLICIES.update({
    name: _compiler_output_policy for name in _COMPILER_OUTPUT_COMMANDS
})
_COMMAND_POLICIES.update({
    name: _write_operand_policy(table) for name, table in _WRITE_OPERAND_TABLES.items()
})
_COMMAND_POLICIES["openssl"] = _openssl_output_policy
_COMMAND_POLICIES["split"] = _split_output_policy
_COMMAND_POLICIES["csplit"] = _csplit_output_policy
_COMMAND_POLICIES["cpio"] = _cpio_policy
_COMMAND_POLICIES["strip"] = _strip_policy
_COMMAND_POLICIES["scp"] = _scp_policy
_COMMAND_POLICIES["zip"] = _zip_policy
_COMMAND_POLICIES.update({name: _sevenzip_policy for name in ("7z", "7za", "7zr")})
_COMMAND_POLICIES["unar"] = _unar_output_policy
_COMMAND_POLICIES.update({
    name: _zstd_policy for name in ("zstd", "zstdcat", "zstdmt", "unzstd")
})
_COMMAND_POLICIES["mktemp"] = _mktemp_policy


def analyze_shell_command_policy(tokens: list[str]) -> ShellCommandPolicy:
    boundary = executor_boundary(tokens)
    if boundary and boundary.module_command:
        # Expanded module argv is inspected as its own command, with its own
        # executor boundary. The interpreter does not own that child's source.
        return ShellCommandPolicy()
    if (
        boundary
        and boundary.consumed_ranges
        and (grammar := EXECUTOR_REGISTRY.get(_command_name(tokens[0])))
        and (grammar.keyword or grammar.package_executable)
    ):
        return _executor_package_policy(tokens, boundary)
    tokens = _package_module_tokens(executor_owned_tokens(tokens))
    if not tokens:
        return ShellCommandPolicy()
    name = _command_name(tokens[0])
    if name in _PACKAGE_PATH_COMMANDS:
        return _package_policy(
            tokens[1:],
            pip=name in {"pip", "pip3"} or (name == "uv" and "pip" in tokens[1:]),
        )
    policy = _COMMAND_POLICIES.get(name)
    return policy(tokens[1:]) if policy else ShellCommandPolicy()


def _command_operand_paths(command: str, tokens: list[str]) -> list[str]:
    """Native ``keyword=value`` operands of dd that name files.

    dd's if=/of= operands name its input and output files. Other commands'
    write operands come from their option tables (see _WRITE_OPERAND_TABLES
    and the git policies), which extract separate, attached-short, and
    attached-long spellings alike.
    """
    if command == "dd":
        return [token[3:] for token in tokens[1:] if token.startswith(("if=", "of="))]
    return []


def path_candidates(
    tokens: list[str],
    *,
    inspect_positional_paths: bool,
    inspect_executable: bool = False,
) -> tuple[str, ...]:
    if not tokens:
        return ()
    executable_paths = (
        [tokens[0]] if "/" in tokens[0] or tokens[0].startswith("~") else []
    )
    boundary = executor_boundary(tokens)
    if boundary is not None and (
        boundary.inner_command_range is not None
        or boundary.module_command
        or boundary.inline_payload_ranges
        or boundary.native_targets
    ):
        owned = executor_owned_tokens(tokens)
        policy = analyze_shell_command_policy(tokens)
        # Compiled targets and interpreter scripts are files/packages, never
        # commands. Only the executor's own operands are inspected at this layer.
        # Registered outer wrappers retain their legacy executable treatment;
        # an executable selected inside another executor is always inspected.
        candidates = [
            *(executable_paths if inspect_executable else []),
            *policy.option_path_values,
            *policy.write_path_values,
        ]
        candidates.extend(policy.positional_values or ())
        candidates.extend(boundary.cwd_overrides)
        if boundary.native_targets and not is_package_command(owned):
            candidates.extend(boundary.native_targets)
        return tuple(dict.fromkeys(candidates))
    policy = analyze_shell_command_policy(tokens)
    candidates = [
        *executable_paths,
        *policy.option_path_values,
        *policy.write_path_values,
    ]
    if not (inspect_positional_paths or policy.inspect_positional_paths):
        return tuple(candidates)
    command = _command_name(tokens[0])
    candidates.extend(_command_operand_paths(command, tokens))
    if policy.positional_values is not None:
        candidates.extend(policy.positional_values)
        return tuple(candidates)
    options_ended = False
    for token in tokens[1:]:
        if token == "--":
            options_ended = True
            continue
        if not options_ended and token.startswith("-"):
            continue
        if command == "chmod" and token.startswith("+"):
            continue
        candidates.append(token)
    return tuple(candidates)
