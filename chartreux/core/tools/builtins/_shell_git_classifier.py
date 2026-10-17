"""The pure, typed dangerous-git classifier of the shell-policy family.

One whole expanded call is ``routine``, ``hard_denied``, or ``grantable``.
The classifier never mutates grants, persists nothing, and executes nothing;
the runtime enforcement in the facade is unchanged. Ambiguous or unmodeled
forms fail closed: never grantable, and never an ordinary git.push.

The module holds the classifier's tables and per-subcommand verdicts
(``_GIT_PUSH_TABLE`` through ``_classify_git_remaining``), the
``git config`` reason family (the ``_git_config_*`` guards and
``_classify_git_config``), and the dispatch surface
(``_classify_git_remote``, ``_dispatch_git_subcommand``,
``classify_dangerous_git_command``, ``_DANGEROUS_GIT_SUBCOMMANDS``,
``is_dangerous_git_subcommand``). This module imports only
``_shell_argv``, ``_shell_git_support``, and
``chartreux.core.dangerous_commands``; ``_shell_command_policy`` (the
facade) re-exports every name defined here, so import paths and object
identity are unchanged by the split.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from chartreux.core.dangerous_commands import (
    DangerousCommandClassification,
    GrantableGitCall,
    ModeledGitCall,
    operations_for_ids,
)
from chartreux.core.tools.builtins._shell_argv import _command_name, _OptionTable
from chartreux.core.tools.builtins._shell_git_support import (
    _GIT_NONREADER_SUBCOMMANDS,
    _GIT_READER_SUBCOMMANDS,
    _GIT_UNMODELED_BUILTINS,
    _effective_clean_flags,
    _scan_classified_git_options,
    _scan_git_config_call,
    _scan_git_globals,
    canonical_git_command,
    push_destination_operands,
)

if TYPE_CHECKING:
    from chartreux.core.dangerous_commands import GitRepositoryRelocation
    from chartreux.core.tools.builtins._shell_git_support import _GitGlobalScan


_GIT_PUSH_TABLE = _OptionTable(
    frozenset({"--push-option", "--repo"}),
    frozenset("o"),
    frozenset({"--force-with-lease", "--signed"}),
    boolean_long=frozenset({
        "--all",
        "--atomic",
        "--delete",
        "--dry-run",
        "--follow-tags",
        "--force",
        "--force-if-includes",
        "--ipv4",
        "--ipv6",
        "--mirror",
        "--no-atomic",
        "--no-dry-run",
        "--no-force",
        "--no-progress",
        "--no-quiet",
        "--no-set-upstream",
        "--no-thin",
        "--no-verbose",
        "--no-verify",
        "--porcelain",
        "--progress",
        "--prune",
        "--quiet",
        "--set-upstream",
        "--tags",
        "--thin",
        "--verbose",
        "--verify",
    }),
)
_GIT_PUSH_BOOLEAN_SHORTS = frozenset("46fnquv")
_GIT_PUSH_FORCE_OPTIONS = frozenset({
    "--force",
    "--force-if-includes",
    "--force-with-lease",
    "-f",
})
# --receive-pack/--exec select the program git runs for a local remote, so a
# push carrying either is never grantable.
_GIT_PUSH_HARD_GUARD_LONGS = frozenset({"--exec", "--receive-pack"})
# A URL scheme prefix (https://, ssh://, git://, ...) names a destination
# directly, exactly like a path operand does.
_PUSH_URL_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]*://")


def _push_destination_is_explicit(destination: str) -> bool:
    """Whether a push destination operand is a path or URL, not a remote name.

    A bare name (``origin``, ``upstream``) resolves through the repository's
    configured remotes, which the grant covers; a path or URL names its
    destination directly, outside the source repository the grant scopes.
    A remote name can never contain ``:`` (git's refname validation rejects
    it), so an scp-like ``host:repo.git`` operand is a URL destination, not
    a remote name.
    """
    if "/" in destination or ":" in destination or destination.startswith((".", "~")):
        return True
    return _PUSH_URL_SCHEME.match(destination) is not None


def _push_destination_fail_closed_reason(call: ModeledGitCall) -> str | None:
    """Why a push destination is not a grantable remote name, or None.

    The destination operands name the destination a push runs receive-pack
    and hooks on, so an explicit path or URL destination is never
    grantable. A bare name is only name-shaped: the classifier stays pure
    and cannot inspect the repository, so the resolver verifies the name
    against the repository's configured remotes — git resolves any
    destination that is not a configured remote name as a URL.
    """
    for destination in push_destination_operands(call):
        if _push_destination_is_explicit(destination):
            return (
                f"destination {destination!r} is a path or URL, not a remote "
                "name: a push to it runs the destination's receive-pack and "
                "hooks, outside the repository the grant covers"
            )
    return None


_GIT_SWITCH_TABLE = _OptionTable(
    frozenset({"--create", "--force-create", "--orphan"}),
    frozenset("cC"),
    frozenset({"--recurse-submodules"}),
    boolean_long=frozenset({
        "--detach",
        "--discard-changes",
        "--force",
        "--guess",
        "--merge",
        "--no-create",
        "--no-guess",
        "--no-track",
        "--quiet",
        "--track",
    }),
)
_GIT_SWITCH_BOOLEAN_SHORTS = frozenset("dfgmq")
_GIT_SWITCH_DISCARD_OPTIONS = frozenset({"--discard-changes", "--force", "-f"})

# checkout/restore are scanned leniently: every form is consequential and git
# itself rejects unmodeled options, so unknown options are recorded without
# failing the classification. Only their value-taking options matter, to keep
# consumed operands out of the positional operands.
_GIT_CHECKOUT_TABLE = _OptionTable(
    frozenset({"--conflict", "--orphan", "--pathspec-from-file"}),
    frozenset("bB"),
    frozenset({"--recurse-submodules"}),
)
_GIT_RESTORE_TABLE = _OptionTable(
    frozenset({"--pathspec-from-file", "--source"}), frozenset("s")
)
_GIT_WORKTREE_TABLES = {"checkout": _GIT_CHECKOUT_TABLE, "restore": _GIT_RESTORE_TABLE}

# stash's value-taking options must consume before the sub-subcommand is
# read: `git stash -m drop` is a stash push with message "drop", not a drop.
_GIT_STASH_TABLE = _OptionTable(
    frozenset({"--message", "--pathspec-from-file"}), frozenset("m")
)
_GIT_REFLOG_TABLE = _OptionTable()
_STASH_REFLOG_OPERATION_IDS = {
    ("stash", "drop"): "git.stash.drop",
    ("stash", "clear"): "git.stash.clear",
    ("reflog", "expire"): "git.reflog.expire",
    ("reflog", "delete"): "git.reflog.delete",
}

# reset/clean are scanned strictly: their hard-guard flags accept unambiguous
# abbreviations (`git reset --ha` really hard-resets), so an unrecognized
# option could hide one and must fail closed.
_GIT_RESET_TABLE = _OptionTable(
    frozenset({"--pathspec-from-file"}),
    boolean_long=frozenset({
        "--hard",
        "--keep",
        "--merge",
        "--mixed",
        "--quiet",
        "--soft",
    }),
)
_GIT_RESET_BOOLEAN_SHORTS = frozenset("q")
_GIT_CLEAN_TABLE = _OptionTable(
    frozenset({"--exclude"}),
    frozenset("e"),
    boolean_long=frozenset({
        "--dry-run",
        "--force",
        "--interactive",
        "--no-dry-run",
        "--no-force",
        "--no-interactive",
        "--no-quiet",
        "--quiet",
    }),
)
_GIT_CLEAN_BOOLEAN_SHORTS = frozenset("dfnqxXi")


# git remote is scanned leniently: only the sub-subcommand decides the
# classification, and git itself rejects unmodeled options. The global
# grammar is the single boolean -v/--verbose; every other option belongs to
# the sub-subcommand, whose own operands never matter once the form is a
# hard guard.
_GIT_REMOTE_TABLE = _OptionTable(boolean_long=frozenset({"--verbose"}))
_GIT_REMOTE_BOOLEAN_SHORTS = frozenset("v")
# git remote mutations redefine where a granted push lands: add/set-url
# point a bare destination name at an arbitrary URL or path, rename
# retargets an existing remote, and remove (rm) turns a configured bare
# name back into a URL git resolves as a path.
_GIT_REMOTE_MUTATION_SUBCOMMANDS = frozenset({
    "add",
    "set-url",
    "rename",
    "remove",
    "rm",
})

_ROUTINE = DangerousCommandClassification(kind="routine")


def _hard_denied(reason: str) -> DangerousCommandClassification:
    return DangerousCommandClassification(kind="hard_denied", reason=reason)


def _grantable_classification(
    operation_ids: tuple[str, ...],
    call: ModeledGitCall,
    relocation: GitRepositoryRelocation,
) -> DangerousCommandClassification:
    operations = operations_for_ids(operation_ids)
    consequence = "; ".join(operation.consequence for operation in operations)
    return DangerousCommandClassification(
        kind="grantable",
        grantable=GrantableGitCall(operations, consequence, call, relocation),
    )


def _fail_closed_on_end_of_options(
    classification: DangerousCommandClassification, *, end_of_options: bool
) -> DangerousCommandClassification:
    """An end-of-options marker is never a modeled form of a grantable call."""
    if classification.kind == "grantable" and end_of_options:
        return _hard_denied("end-of-options marker is not a modeled form")
    return classification


def _refspec_fail_closed_reason(refspec: str) -> str | None:
    """Why a push refspec is not a modeled form, or None when it is."""
    if not refspec:
        return "empty refspec"
    if any(char in refspec for char in "*?["):
        return f"glob refspec {refspec!r} is not a modeled form"
    if refspec.count(":") > 1:
        return f"refspec {refspec!r} is not a valid src:dst form"
    _, separator, dst = refspec.partition(":")
    if separator and not dst:
        return f"refspec {refspec!r} has an empty destination"
    body = refspec.removeprefix("+")
    if not body:
        return f"refspec {refspec!r} is not a valid form"
    return None


def _push_operation_ids(*, force: bool, delete: bool, mirror: bool) -> tuple[str, ...]:
    """Operation IDs a push requires; a compound push never inherits git.push."""
    ids = []
    if force:
        ids.append("git.push.force")
    if delete:
        ids.append("git.push.delete")
    if mirror:
        ids.append("git.push.mirror")
    return tuple(ids) if ids else ("git.push",)


def _classify_git_push(
    args: list[str], globals_scan: _GitGlobalScan
) -> DangerousCommandClassification:
    scan = _scan_classified_git_options(
        args,
        _GIT_PUSH_TABLE,
        boolean_shorts=_GIT_PUSH_BOOLEAN_SHORTS,
        hard_guard_longs=_GIT_PUSH_HARD_GUARD_LONGS,
    )
    if scan.error is not None:
        return _hard_denied(f"git push: {scan.error}")
    options = set(scan.options)
    force = bool(options & _GIT_PUSH_FORCE_OPTIONS)
    # --prune removes remote refs without a local counterpart, so it is a
    # deletion mechanism like --delete, never ordinary publication.
    delete = "--delete" in options or "--prune" in options
    mirror = "--mirror" in options
    call = ModeledGitCall("push", scan.options, scan.values, scan.operands)
    # A push to an explicit path or URL destination is never grantable: the
    # grant scopes the source repository, and the push runs the destination's
    # receive-pack and hooks.
    destination_reason = _push_destination_fail_closed_reason(call)
    if destination_reason is not None:
        return _hard_denied(f"git push: {destination_reason}")
    # The first positional is always the repository, never a refspec.
    for refspec in scan.operands[1:]:
        reason = _refspec_fail_closed_reason(refspec)
        if reason is not None:
            return _hard_denied(f"git push: {reason}")
        if refspec.startswith("+"):
            force = True
        if refspec.removeprefix("+").startswith(":"):
            delete = True
    result = _grantable_classification(
        _push_operation_ids(force=force, delete=delete, mirror=mirror),
        call,
        globals_scan.relocation,
    )
    return _fail_closed_on_end_of_options(
        result, end_of_options=globals_scan.end_of_options or scan.end_of_options
    )


def _classify_git_worktree_subcommand(
    subcommand: str,
    operation_id: str,
    table: _OptionTable,
    args: list[str],
    globals_scan: _GitGlobalScan,
) -> DangerousCommandClassification:
    """checkout/restore: every form is consequential, so all forms are grantable."""
    scan = _scan_classified_git_options(args, table, strict=False)
    if scan.error is not None:
        return _hard_denied(f"git {subcommand}: {scan.error}")
    call = ModeledGitCall(subcommand, scan.options, scan.values, scan.operands)
    result = _grantable_classification((operation_id,), call, globals_scan.relocation)
    return _fail_closed_on_end_of_options(
        result, end_of_options=globals_scan.end_of_options or scan.end_of_options
    )


def _classify_git_switch(
    args: list[str], globals_scan: _GitGlobalScan
) -> DangerousCommandClassification:
    scan = _scan_classified_git_options(
        args, _GIT_SWITCH_TABLE, boolean_shorts=_GIT_SWITCH_BOOLEAN_SHORTS
    )
    if scan.error is not None:
        return _hard_denied(f"git switch: {scan.error}")
    if not set(scan.options) & _GIT_SWITCH_DISCARD_OPTIONS:
        return _ROUTINE
    call = ModeledGitCall("switch", scan.options, scan.values, scan.operands)
    result = _grantable_classification(
        ("git.switch.discard",), call, globals_scan.relocation
    )
    return _fail_closed_on_end_of_options(
        result, end_of_options=globals_scan.end_of_options or scan.end_of_options
    )


def _classify_git_stash_or_reflog(
    subcommand: str, args: list[str], globals_scan: _GitGlobalScan
) -> DangerousCommandClassification:
    """stash drop/clear and reflog expire/delete are grantable; other forms are not."""
    table = _GIT_STASH_TABLE if subcommand == "stash" else _GIT_REFLOG_TABLE
    scan = _scan_classified_git_options(args, table, strict=False)
    if scan.error is not None:
        return _hard_denied(f"git {subcommand}: {scan.error}")
    operation_id = _STASH_REFLOG_OPERATION_IDS.get((
        subcommand,
        scan.operands[0] if scan.operands else "",
    ))
    if operation_id is None:
        return _ROUTINE
    call = ModeledGitCall(subcommand, scan.options, scan.values, scan.operands)
    result = _grantable_classification((operation_id,), call, globals_scan.relocation)
    return _fail_closed_on_end_of_options(
        result, end_of_options=globals_scan.end_of_options or scan.end_of_options
    )


def _classify_git_hard_guard_subcommand(
    subcommand: str, args: list[str]
) -> DangerousCommandClassification:
    """reset/clean: destructive forms are hard guards no grant can lift."""
    if subcommand == "reset":
        scan = _scan_classified_git_options(
            args, _GIT_RESET_TABLE, boolean_shorts=_GIT_RESET_BOOLEAN_SHORTS
        )
        dangerous = "--hard" in scan.options
        guard = "git reset --hard"
    else:
        scan = _scan_classified_git_options(
            args, _GIT_CLEAN_TABLE, boolean_shorts=_GIT_CLEAN_BOOLEAN_SHORTS
        )
        force, dry_run = _effective_clean_flags(scan.options)
        dangerous = force and not dry_run
        guard = "forced non-dry-run git clean"
    if scan.error is not None:
        return _hard_denied(f"git {subcommand}: {scan.error}")
    if dangerous:
        return _hard_denied(f"{guard} is a hard guard; no grant can authorize it")
    return _ROUTINE


def _classify_git_remaining(subcommand: str) -> DangerousCommandClassification:
    """Readers and unclassified nonreaders stay routine; the rest are hard guards."""
    if subcommand in _GIT_UNMODELED_BUILTINS:
        return _hard_denied(
            f"unmodeled git builtin {subcommand!r}; denied conservatively"
        )
    if (
        subcommand not in _GIT_READER_SUBCOMMANDS
        and subcommand not in _GIT_NONREADER_SUBCOMMANDS
    ):
        return _hard_denied(
            f"unrecognized git subcommand {subcommand!r}; "
            "aliases and external helpers cannot be inspected"
        )
    return _ROUTINE


# remote.<name> keys whose write redefines where a granted push lands or
# the program a push runs. url.<base>.insteadOf is guarded alongside
# pushInsteadOf because git applies it to the URL of a configured remote
# when pushing, not only when fetching.
_GIT_CONFIG_REMOTE_PUSH_EFFECTING_KEYS = frozenset({
    "mirror",
    "push",
    "url",
    "pushurl",
    "receivepack",
    "uploadpack",
    "proxy",
    "vcs",
})
# branch.<name> keys that select the destination of a bare push.
_GIT_CONFIG_BRANCH_PUSH_EFFECTING_KEYS = frozenset({"remote", "pushremote"})
# url.<base> keys that rewrite the URL a push uses.
_GIT_CONFIG_URL_PUSH_EFFECTING_KEYS = frozenset({"insteadof", "pushinsteadof"})
# include.path and includeIf.<condition>.path author a file git folds into
# configuration resolution, which can set any push-effecting key for a remote
# the resolver verified.
_GIT_CONFIG_INCLUDE_SECTIONS = frozenset({"include", "includeif"})
# core keys that select a program a push runs: the ssh command (even a
# --dry-run), the git:// proxy, and the ask-pass program.
_GIT_CONFIG_CORE_PROGRAM_KEYS = frozenset({"sshcommand", "gitproxy", "askpass"})
# core/sequence keys that select a program git runs on routine subcommands,
# not only on a push: core.hooksPath selects the directory git takes hooks
# from, core.editor the editor git runs for commit/tag/rebase without -m,
# sequence.editor the editor an interactive rebase runs, and core.fsmonitor
# the filesystem-monitor daemon git runs for status-bearing subcommands
# such as add and commit.
_GIT_CONFIG_CORE_PROGRAM_SELECTION_KEYS = frozenset({
    "editor",
    "fsmonitor",
    "hookspath",
})
_GIT_CONFIG_SEQUENCE_PROGRAM_SELECTION_KEYS = frozenset({"editor"})
# The man section selects the program `git help` runs: man.<tool>.path names
# the man viewer's executable for the tool man.viewer selects (git 2.53
# rejects an arbitrary man.viewer command string, so the path key is the
# live vector), and interactive.diffFilter names the pager-style filter
# git runs for `git add -p` diffs.
_GIT_CONFIG_MAN_PROGRAM_SELECTION_KEYS = frozenset({"path", "viewer"})
_GIT_CONFIG_INTERACTIVE_PROGRAM_SELECTION_KEYS = frozenset({"difffilter"})
# The driver sections select programs the same way on routine subcommands:
# filter.<name>.clean and filter.<name>.process run for files with that
# filter (git add, git checkout), merge.<name>.driver runs for files with
# that merge driver (git merge), and diff.<name>.command,
# diff.<name>.textconv, and diff.<name>.external run for diffs with that
# driver (git diff, git log -p).
_GIT_CONFIG_FILTER_PROGRAM_SELECTION_KEYS = frozenset({"clean", "process"})
_GIT_CONFIG_MERGE_PROGRAM_SELECTION_KEYS = frozenset({"driver"})
_GIT_CONFIG_DIFF_PROGRAM_SELECTION_KEYS = frozenset({"command", "external", "textconv"})


def _git_config_key_is_push_effecting(key: str) -> bool:  # noqa: PLR0911
    """Whether one git config key changes where a granted push lands.

    The guarded surface is every key whose write redefines the push
    destination or the program a push runs: ``remote.<name>.url`` and
    ``remote.<name>.pushurl`` define where a push to that remote lands,
    ``remote.<name>.receivepack`` and ``remote.<name>.uploadpack`` select
    the program git runs for the remote (receivepack is the config twin of
    the hard-guarded ``--receive-pack`` option; uploadpack is its
    fetch-side twin, guarded for symmetry), ``remote.<name>.proxy`` selects
    the proxy program git runs, ``remote.<name>.vcs`` selects the
    ``git-remote-<vcs>`` helper that ``git remote update`` executes,
    ``url.<base>.insteadOf`` and ``url.<base>.pushInsteadOf`` rewrite the
    URL a push uses, ``branch.<name>.remote`` and ``branch.<name>.pushremote``
    select the destination of a bare push (path-capable),
    ``remote.<name>.mirror`` mirrors every push to that remote,
    ``remote.<name>.push`` installs the refspecs a refspec-less push uses
    (forced or deleting ones included), and ``remote.pushDefault`` (also
    spelled ``remote.<name>.pushDefault``) selects the remote a bare push
    targets. The program-execution family is guarded the same way:
    ``core.sshCommand`` selects the program git runs for an ssh push (even
    a ``--dry-run``), ``core.gitProxy`` the program it runs for a
    ``git://`` push, ``core.askPass`` the program it runs to ask for a
    credential, and the whole ``credential`` section installs the
    credential-lookup program a push runs (``credential.helper`` and
    ``credential.<url>.helper``). Finally, ``include.path`` and
    ``includeIf.<condition>.path`` author a file git folds into
    configuration resolution, which can set any of these keys for a remote
    the resolver verified. Git folds section and key names
    case-insensitively, so the guard does too; remote, branch, and url-base
    names are case-sensitive, but the guard fails closed on any spelling.
    """
    *sections, name = key.split(".")
    if not sections:
        return False
    folded = name.casefold()
    section = sections[0].casefold()
    if section == "remote":
        if folded in _GIT_CONFIG_REMOTE_PUSH_EFFECTING_KEYS:
            # The name-bearing keys carry the remote name.
            return bool(sections[1:])
        return folded == "pushdefault"
    if section == "branch":
        return folded in _GIT_CONFIG_BRANCH_PUSH_EFFECTING_KEYS and bool(sections[1:])
    if section == "url":
        return folded in _GIT_CONFIG_URL_PUSH_EFFECTING_KEYS and bool(sections[1:])
    if section in _GIT_CONFIG_INCLUDE_SECTIONS:
        # include.path and includeIf.<condition>.path are the only include
        # keys; the whole section is guarded so no spelling slips through.
        return True
    if section == "credential":
        # credential.helper and credential.<url>.helper install a
        # credential-lookup program; the whole section is guarded so no
        # helper spelling slips through.
        return True
    if section == "core":
        return folded in _GIT_CONFIG_CORE_PROGRAM_KEYS
    return False


def _git_config_section_is_push_effecting(section: str) -> bool:
    """Whether one section operand of a section operation is push-effecting.

    A section rename or removal relocates or deletes whole sections, so it
    can set or relocate any key the guarded surface covers: the remote
    definitions, the url rewrites, the branch push destinations, the
    program-execution keys, the credential section, and the include
    directives.
    """
    folded = section.casefold()
    return folded in {
        "remote",
        "url",
        "branch",
        "credential",
        "include",
        "includeif",
    } or folded.startswith((
        "remote.",
        "url.",
        "branch.",
        "credential.",
        "include.",
        "includeif.",
    ))


def _git_config_key_is_program_selection(key: str) -> bool:  # noqa: PLR0911
    """Whether one git config key selects a program git runs on routine subcommands.

    ``core.hooksPath`` selects the directory git runs hooks from, so a granted
    push — or any hook-running subcommand, including readers that refresh the
    index — executes the hooks there; ``core.editor`` selects the program
    ``git commit`` (without ``-m``), ``git tag -a``, and ``git rebase`` run;
    ``sequence.editor`` selects the program an interactive rebase runs; and
    ``core.fsmonitor`` selects the daemon git runs for status-bearing
    subcommands (``git add``, ``git commit``). The driver sections select
    programs the same way on routine subcommands:
    ``filter.<name>.clean`` and ``filter.<name>.process`` run for files with
    that filter (``git add``, ``git checkout``), ``merge.<name>.driver`` runs
    for files with that merge driver (``git merge``), and
    ``diff.<name>.command``, ``diff.<name>.textconv``, and
    ``diff.<name>.external`` run for diffs with that driver (``git diff``,
    ``git log -p``). The ``man`` section selects the program ``git help``
    runs: ``man.<tool>.path`` names the man viewer's executable for the tool
    ``man.viewer`` selects, and ``interactive.diffFilter`` names the filter
    git runs for ``git add -p`` diffs. Git folds section and key names
    case-insensitively, so the guard does too.
    """
    *sections, name = key.split(".")
    if not sections:
        return False
    folded = name.casefold()
    section = sections[0].casefold()
    if section in {"core", "sequence"}:
        keys = (
            _GIT_CONFIG_CORE_PROGRAM_SELECTION_KEYS
            if section == "core"
            else _GIT_CONFIG_SEQUENCE_PROGRAM_SELECTION_KEYS
        )
        return folded in keys
    if section == "filter":
        return folded in _GIT_CONFIG_FILTER_PROGRAM_SELECTION_KEYS and bool(
            sections[1:]
        )
    if section == "merge":
        return folded in _GIT_CONFIG_MERGE_PROGRAM_SELECTION_KEYS and bool(sections[1:])
    if section == "diff":
        return folded in _GIT_CONFIG_DIFF_PROGRAM_SELECTION_KEYS and bool(sections[1:])
    if section == "man":
        # man.viewer selects the viewer tool and man.<tool>.path names its
        # executable; both keys are guarded (a bare man.path or a
        # man.<tool>.viewer spelling is not a real git key, but guarding it
        # fails closed).
        return folded in _GIT_CONFIG_MAN_PROGRAM_SELECTION_KEYS
    if section == "interactive":
        return folded in _GIT_CONFIG_INTERACTIVE_PROGRAM_SELECTION_KEYS
    return False


def _git_config_section_is_program_selection(section: str) -> bool:
    """Whether one section operand of a rename reaches a program-selection section.

    Renaming a section to (or from) ``core``, ``sequence``, ``filter``,
    ``merge``, ``diff``, ``man``, or ``interactive`` moves its keys into or
    out of the sections whose ``hooksPath``/``editor``/``fsmonitor``/driver/
    ``man``/``diffFilter`` keys select the programs git runs, so an
    agent-authored key lands exactly where the key guard blocks writing it:
    guarding the key set alone is bypassable.
    """
    folded = section.casefold()
    return folded in {
        "core",
        "sequence",
        "filter",
        "merge",
        "diff",
        "man",
        "interactive",
    } or folded.startswith((
        "core.",
        "sequence.",
        "filter.",
        "merge.",
        "diff.",
        "man.",
        "interactive.",
    ))


def _git_config_push_effecting_reason(target: str) -> str:
    """Why a push-effecting config write is a hard guard."""
    return (
        f"git config writes {target!r}, push-effecting configuration: "
        "remote.<name>.url and remote.<name>.pushurl define where a push "
        "to that remote lands, remote.<name>.receivepack and "
        "remote.<name>.uploadpack select the program git runs for the "
        "remote (receivepack is the config twin of the hard-guarded "
        "--receive-pack option; uploadpack is its fetch-side twin, guarded "
        "for symmetry), remote.<name>.proxy selects the proxy program git "
        "runs, and remote.<name>.vcs selects the git-remote-<vcs> helper "
        "that git remote update executes (git remote update and prune stay "
        "routine: with vcs and sshCommand writes barred, the programs they "
        "run are user-configured), url.<base>.insteadOf and "
        "url.<base>.pushInsteadOf rewrite the URL a push uses, "
        "branch.<name>.remote and branch.<name>.pushremote select the "
        "destination of a bare push, remote.<name>.mirror mirrors every "
        "push to that remote, remote.<name>.push installs the refspecs a "
        "refspec-less push uses, and remote.pushDefault selects the remote "
        "a bare push targets; core.sshCommand selects the program git runs "
        "for an ssh push (even a --dry-run), core.gitProxy the program it "
        "runs for a git:// push, core.askPass the program it runs to ask "
        "for a credential, and credential.helper and credential.<url>.helper "
        "install the credential-lookup program a push runs; include.path "
        "and includeIf.<condition>.path author a file git folds into "
        "configuration, which can set any of these keys for a remote the "
        "resolver verified, so an approved push would run what the write "
        "selects; no grant can authorize the write. A push to a remote "
        "whose configuration you set yourself is ordinary publication — "
        "your own config and any file it includes define what push means "
        "there — so run this command manually if you intend it."
    )


def _git_config_program_selection_reason(target: str) -> str:
    """Why a program-selection config write is a hard guard."""
    return (
        f"git config writes {target!r}, program-selection configuration: "
        "core.hooksPath selects the directory git runs hooks from, so a "
        "granted push — or any hook-running git command, including readers "
        "that refresh the index — would execute the hooks there, "
        "core.editor selects the program git runs for commit, tag, and "
        "rebase without -m, sequence.editor selects the program an "
        "interactive rebase runs, core.fsmonitor selects the daemon git "
        "runs for status-bearing subcommands (git add, git commit), "
        "filter.<name>.clean and filter.<name>.process select the programs "
        "git runs for files with that filter (git add, git checkout), "
        "merge.<name>.driver selects the program git runs for files with "
        "that merge driver (git merge), and diff.<name>.command, "
        "diff.<name>.textconv, and diff.<name>.external select the "
        "programs git runs for diffs with that driver (git diff, git log "
        "-p); man.<tool>.path selects the man viewer git runs for "
        "git help once man.viewer selects its tool, and "
        "interactive.diffFilter selects the filter git runs for git add -p "
        "diffs; no grant can authorize the write. A repository whose hooks, "
        "editor, fsmonitor daemon, filters, merge drivers, diff drivers, "
        "man viewer, or interactive diff filter you point at a program of "
        "your choosing is not covered by any git approval — run this "
        "command manually if you intend it"
    )


def _git_config_program_section_reason(target: str, *, destination: bool) -> str:
    """Why a section rename reaching a program-selection section is a hard guard.

    The denial names the destination section when the rename's destination
    operand reaches one, and otherwise the matched operand itself, so the
    reason always names the section the rename actually moves keys into or
    out of.
    """
    if destination:
        return (
            f"git config renames a section to {target!r}, a "
            "program-selection section: the rename moves the keys of "
            f"that section into {target}, whose hooksPath/editor/fsmonitor/"
            "filter/merge/diff/man/interactive keys select the programs "
            "git runs on routine subcommands, so an agent-authored key "
            "lands exactly where the key guard blocks writing it; no grant "
            "can authorize the write. Run this command manually if you "
            "intend it"
        )
    return (
        f"git config renames the program-selection section {target!r}: the "
        "rename moves the keys of that section out of the section whose "
        "program-selection keys the key guard blocks, and an agent-authored "
        "key could take their place; no grant can authorize the write. Run "
        "this command manually if you intend it"
    )


def _git_config_rename_guard_reason(operands: list[str]) -> str | None:
    """Why a section rename is a hard guard, naming the section it lands in.

    A rename to (or from) a program-selection section relocates keys into
    or out of the sections whose program-selection keys the key guard
    blocks, so an agent-authored hooksPath/editor or driver key would take
    effect. The destination operand is named when it reaches a guarded
    section; otherwise the matched operand is named.
    """
    rename_operands = list(operands)
    destination = rename_operands[-1] if rename_operands[1:] else None
    if destination is not None and _git_config_section_is_program_selection(
        destination
    ):
        return _git_config_program_section_reason(destination, destination=True)
    target = next(
        (
            operand
            for operand in rename_operands
            if _git_config_section_is_program_selection(operand)
        ),
        None,
    )
    if target is not None:
        return _git_config_program_section_reason(target, destination=False)
    return None


def _git_config_section_operation_guard_reason(
    operands: list[str], *, rename: bool
) -> str | None:
    """Why a section operation (rename or removal) is a hard guard, or None."""
    target = next(
        (
            operand
            for operand in operands
            if _git_config_section_is_push_effecting(operand)
        ),
        None,
    )
    if target is not None:
        return _git_config_push_effecting_reason(target)
    if rename:
        # Only renames are checked here: a removal cannot move a key into
        # a guarded section.
        return _git_config_rename_guard_reason(operands)
    return None


def _git_config_write_guard_reason(
    operands: list[str], *, section_operation: bool, rename: bool
) -> str | None:
    """Why a mutating ``git config`` call is a hard guard, or None when routine.

    The scan already established that the call mutates; this selects the first
    operand that reaches the guarded surface. Section operations check the
    push-effecting sections first and, for renames, the program-selection
    sections (either operand, so a destination section name is covered); key
    writes check the push-effecting keys and then the program-selection keys.
    """
    if section_operation:
        return _git_config_section_operation_guard_reason(operands, rename=rename)
    target = next(
        (operand for operand in operands if _git_config_key_is_push_effecting(operand)),
        None,
    )
    if target is not None:
        return _git_config_push_effecting_reason(target)
    target = next(
        (
            operand
            for operand in operands
            if _git_config_key_is_program_selection(operand)
        ),
        None,
    )
    if target is not None:
        return _git_config_program_selection_reason(target)
    return None


def _classify_git_config(args: list[str]) -> DangerousCommandClassification:
    """git config: writes to push-effecting or program-selection keys are hard guards.

    ``git config`` is otherwise a permitted routine command, so reads stay
    routine — ``--get``/``--list``/``--get-urlmatch``/... and the ``get`` and
    ``list`` subcommands, including a bare key, which git reads — and so do
    writes to unrelated keys, which cannot change where a granted push
    lands or what program git runs. What must never classify routine is the
    agent redefining the push-effecting surface: a configured
    ``remote.<name>.url``, ``remote.<name>.pushurl``, ``url.<base>.insteadOf``,
    or ``branch.<name>.remote`` points a granted push at a destination of the
    agent's choosing, ``remote.<name>.receivepack`` selects the program the
    push runs, ``remote.<name>.vcs`` selects the helper ``git remote
    update`` executes, ``core.sshCommand``, ``core.gitProxy``,
    ``core.askPass``, and the ``credential`` section select the other
    programs a push runs (the ssh command, the git:// proxy, the ask-pass
    program, and the credential lookup), and ``include.path`` or
    ``includeIf.<condition>.path`` authors a file git folds into
    configuration, which can set any of those keys for a remote the
    resolver verified, while a configured ``remote.<name>.mirror``,
    ``remote.<name>.push`` refspec, or ``remote.pushDefault`` makes an
    ordinary ``git.push`` approval authorize configured forced, deleted, or
    mirrored updates. The program-selection surface is guarded the same way
    in every write scope (local, ``--global``, ``--file``, ``--worktree``,
    and ``-C``/``--git-dir`` relocation): ``core.hooksPath`` selects the
    directory git runs hooks from, ``core.editor`` the program git runs for
    commit/tag/rebase without ``-m``, ``sequence.editor`` the program an
    interactive rebase runs, ``core.fsmonitor`` the daemon git runs for
    status-bearing subcommands (``git add``, ``git commit``),
    ``filter.<name>.clean`` and
    ``filter.<name>.process`` the programs git runs for files with that
    filter (``git add``, ``git checkout``), ``merge.<name>.driver`` the
    program git runs for files with that merge driver (``git merge``), and
    ``diff.<name>.command``, ``diff.<name>.textconv``, and
    ``diff.<name>.external`` the programs git runs for diffs with that
    driver (``git diff``, ``git log -p``), ``man.<tool>.path`` the man
    viewer git runs for ``git help`` once ``man.viewer`` selects its tool,
    and ``interactive.diffFilter`` the filter git runs for ``git add -p``
    diffs — each executes on routine
    subcommands, so every mutation of those keys — set, add, replace-all,
    unset, the ``set``/``unset`` subcommands — is a hard guard with no
    grantable operation. A section rename reaching ``core``, ``sequence``,
    ``filter``, ``merge``, ``diff``, ``man``, or ``interactive`` (either
    operand, so a destination section name is covered) relocates keys into
    the section whose program-selection keys the key guard blocks, and is a
    hard guard too; section removals stay routine for those sections
    because a removal cannot move a key into a guarded section. Editor
    writes (``--edit``/``-e``) are hard guards as well: an editor can set
    any key, so it is never classifiable by key. The boundary is deliberate: a push
    to a remote whose configuration the USER set is ordinary publication,
    because the user's own config — includes included — defines what push
    means there; only the agent's writes are barred.
    """
    call = _scan_git_config_call(args)
    if call.error is not None:
        return _hard_denied(f"git config: {call.error}")
    if call.reads:
        # A read never writes configuration.
        return _ROUTINE
    if call.edits:
        # An editor write can set any key, so it is never classifiable.
        return _hard_denied(
            "git config --edit opens an editor on the config file, which "
            "can set push-effecting configuration (remote.<name>.url, "
            "remote.<name>.pushurl, remote.<name>.receivepack, "
            "remote.<name>.uploadpack, remote.<name>.proxy, "
            "remote.<name>.vcs, url.<base>.insteadOf, "
            "url.<base>.pushInsteadOf, branch.<name>.remote, "
            "branch.<name>.pushremote, remote.<name>.mirror, "
            "remote.<name>.push, remote.pushDefault, core.sshCommand, "
            "core.gitProxy, core.askPass, credential.helper, "
            "credential.<url>.helper, include.path, "
            "includeIf.<condition>.path) or program-selection configuration "
            "(core.hooksPath, core.editor, sequence.editor, "
            "core.fsmonitor, filter.<name>.clean, filter.<name>.process, "
            "merge.<name>.driver, diff.<name>.command, "
            "diff.<name>.textconv, diff.<name>.external, "
            "man.<tool>.path, man.viewer, interactive.diffFilter) that no "
            "grant can authorize; run it manually if you intend it"
        )
    if not call.mutates:
        # A bare key with no action is git's read form; nothing is written.
        return _ROUTINE
    if reason := _git_config_write_guard_reason(
        list(call.operands),
        section_operation=call.section_operation,
        rename=call.renames,
    ):
        return _hard_denied(reason)
    return _ROUTINE


def _classify_git_remote(args: list[str]) -> DangerousCommandClassification:
    """git remote: mutations of the remote definitions are hard guards.

    ``git remote add``/``set-url`` point a bare destination name at an
    arbitrary URL or path, ``rename`` retargets an existing remote, and
    ``remove`` (``rm``) turns a configured bare name back into a URL git
    resolves as a path — each redefines where a granted push lands, so no
    grant can authorize it. Reads (``-v``, ``show``, ``get-url``) and the
    other non-mutating forms stay routine, including ``update`` and
    ``prune``: with ``remote.<name>.vcs`` and ``core.sshCommand`` writes
    barred, the helper and ssh program those forms run are
    user-configured, so they stay in-contract.
    """
    scan = _scan_classified_git_options(
        args, _GIT_REMOTE_TABLE, boolean_shorts=_GIT_REMOTE_BOOLEAN_SHORTS, strict=False
    )
    if scan.error is not None:
        return _hard_denied(f"git remote: {scan.error}")
    if scan.operands and scan.operands[0] in _GIT_REMOTE_MUTATION_SUBCOMMANDS:
        return _hard_denied(
            f"git remote {scan.operands[0]} redefines where a granted push "
            "lands: it defines, repoints, renames, or removes the remote a "
            "push destination resolves through, and a removed remote turns "
            "the bare name into a URL git resolves as a path; no grant can "
            "authorize it — run it manually if you intend it"
        )
    return _ROUTINE


def _dispatch_git_subcommand(
    subcommand: str, args: list[str], globals_scan: _GitGlobalScan
) -> DangerousCommandClassification:
    classification: DangerousCommandClassification
    if subcommand == "push":
        classification = _classify_git_push(args, globals_scan)
    elif subcommand in _GIT_WORKTREE_TABLES:
        classification = _classify_git_worktree_subcommand(
            subcommand,
            f"git.{subcommand}",
            _GIT_WORKTREE_TABLES[subcommand],
            args,
            globals_scan,
        )
    elif subcommand == "switch":
        classification = _classify_git_switch(args, globals_scan)
    elif subcommand in {"stash", "reflog"}:
        classification = _classify_git_stash_or_reflog(subcommand, args, globals_scan)
    elif subcommand in {"reset", "clean"}:
        classification = _classify_git_hard_guard_subcommand(subcommand, args)
    elif subcommand == "config":
        classification = _classify_git_config(args)
    elif subcommand == "remote":
        classification = _classify_git_remote(args)
    else:
        classification = _classify_git_remaining(subcommand)
    return classification


def classify_dangerous_git_command(tokens: list[str]) -> DangerousCommandClassification:
    """Classify one whole expanded call for the dangerous-command flow.

    Pure and typed: the call is ``routine`` (no dangerous-command
    involvement), ``hard_denied`` (denied with no grantable operation), or
    ``grantable`` (denied pending per-operation grants; the classification
    carries the required operations, consequence, modeled operands, and
    repository relocation). Only git invocations are classified; every other
    command is routine and stays governed by the existing policy layer, whose
    enforcement is unchanged. Ambiguous or unmodeled forms fail closed: never
    grantable, never an ordinary git.push. ``git config`` and ``git remote``
    are the routed subcommands that mostly stay routine — they are permitted
    commands — so only writes to push-effecting configuration (remote
    definitions and url rewrites, branch push destinations, mirror/push
    refspecs, push defaults, the programs a push runs, the credential
    section, and the include directives), writes to program-selection
    configuration (``core.hooksPath``, ``core.editor``, ``sequence.editor``,
    ``filter.<name>.clean``, ``filter.<name>.process``,
    ``merge.<name>.driver``, ``diff.<name>.command``,
    ``diff.<name>.textconv``, ``diff.<name>.external``, in every write
    scope, plus section renames reaching ``core``, ``sequence``,
    ``filter``, ``merge``, or ``diff``), and ``git remote`` mutations (add,
    set-url, rename, remove) are hard guards: a push to a remote whose
    configuration the user set is ordinary publication, because the user's
    own config — includes included — defines what push means there, and
    only the agent's writes and remote mutations are barred.
    """
    if not tokens or _command_name(tokens[0]) != "git":
        return _ROUTINE
    globals_scan = _scan_git_globals(tokens[1:])
    if globals_scan.error is not None:
        return _hard_denied(globals_scan.error)
    args = tokens[1:]
    if globals_scan.index >= len(args):
        # No subcommand: bare git prints usage/version and carries nothing to
        # classify.
        return _ROUTINE
    subcommand = args[globals_scan.index]
    return _dispatch_git_subcommand(
        subcommand, args[globals_scan.index + 1 :], globals_scan
    )


# Subcommands the dangerous-command flow routes: malformed or unmodeled forms
# of these fail closed (git config and git remote excepted: they are
# permitted routine commands, so they stay routine unless they mutate the
# push-effecting surface). Every other git invocation — readers, unmodeled
# builtins, unrecognized subcommands, dangling globals without a subcommand —
# carries no dangerous operation and stays governed by the existing policy
# layer, whose enforcement is unchanged.
_DANGEROUS_GIT_SUBCOMMANDS = frozenset({
    "push",
    "checkout",
    "restore",
    "switch",
    "stash",
    "reflog",
    "reset",
    "clean",
    "config",
    "remote",
})


def is_dangerous_git_subcommand(tokens: list[str]) -> bool:
    """Whether one git invocation routes through the dangerous-command flow."""
    canonical = canonical_git_command(tokens)
    return (
        canonical is not None
        and len(canonical) > 1
        and canonical[1] in _DANGEROUS_GIT_SUBCOMMANDS
    )
