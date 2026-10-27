from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from enum import StrEnum
from functools import lru_cache
from io import StringIO
import os
from pathlib import Path
import re
import shlex
from typing import TYPE_CHECKING, Protocol

from tree_sitter import Language, Parser
import tree_sitter_bash as tsbash

from chartreux.core.dangerous_commands import (
    DangerousCommandAuthority,
    DangerousCommandProposal,
    GitRepositoryRelocation,
    RepositoryScope,
    operations_for_ids,
)
from chartreux.core.tools.base import ToolError, ToolPermission
from chartreux.core.tools.builtins._destination import classify_network_egress
from chartreux.core.tools.builtins._shell_argv import (
    PatternGlobCandidate,
    ShellToken,
    _command_name,
    _scan_options,
    split_shell_tokens,
)
from chartreux.core.tools.builtins._shell_command_policy import (
    _COMPILER_OUTPUT_COMMANDS,
    _DESTINATION_POSITION_TABLES,
    _GIT_CHECKOUT_TABLE,
    _GIT_SUBCOMMAND_INDEX,
    _SEVENZIP_ARCHIVE_WRITING_COMMANDS,
    _SEVENZIP_TABLE,
    _WRITE_OPERAND_TABLES,
    ShellCommandPolicy,
    _has_long,
    _has_short,
    _scan_classified_git_options,
    _scan_git_globals,
    analyze_shell_command_policy,
    canonical_git_command,
    classify_dangerous_git_command,
    executor_boundary,
    git_apply_cwd_root_write,
    git_cwd_relocations,
    git_metadata_paths,
    git_push_destination_risk,
    git_repository_config_risk,
    git_work_tree_writing_subcommand,
    inline_interpreter_switch,
    is_dangerous_git_subcommand,
    matches_command_prefix,
    path_candidates,
    path_reaches_bare_repository,
    resolve_git_work_tree,
    resolve_repository_scope,
)
from chartreux.core.tools.builtins._shell_diagnostics import (
    DenialClass,
    Diagnostic,
    aggregate_denial_class,
    render_diagnostic,
)
from chartreux.core.tools.builtins._shell_git_state import (
    RepositoryState,
    config_selects_program,
    in_progress_operation,
    local_branch_exists,
    repositories_intersecting,
    repository_recovery,
    repository_state,
    work_tree_is_clean,
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
        noun = (
            "a dangerous git operation"
            if all(operation.startswith("git.") for operation in proposal.operations)
            else "a dangerous operation"
        )
        return Diagnostic(
            "analysis",
            detail=(
                noun + " requiring approval (" + ", ".join(proposal.operations) + ")"
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
    # Original provenance-carrying tokens for argv-wrapper reassemblies:
    # the synthetic shlex.join text would re-quote unquoted glob
    # metacharacters, so the wrapper visit carries the source tokens and
    # downstream tokenization must not re-derive provenance from that text.
    tokens: tuple[str, ...] | None = None
    # The command's OWN environment assignments: the literal command-prefix
    # assignments (dropped from the part text), the env-wrapper and
    # sudo-wrapper assignment operands (dropped from the reassembled argv),
    # and the statement-level
    # bindings in force where the part runs (an earlier `export NAME=value`
    # or standalone `NAME=value` statement of the same command text). The
    # shell applies all of them to this command's subprocess, so the egress
    # proxy inspection consumes them. A wrapper's child re-visit accumulates
    # the wrapper's assignments into the child part, and a nested shell's
    # payload parts inherit the enclosing statement's, so a wrapped
    # client's part carries the effective environment it runs with.
    assignments: tuple[tuple[str, str], ...] = ()


def _has_unrecognized_nested_shell(tokens: list[str]) -> bool:
    """Detect shell -c behind an unmodeled executable (including script -c).

    The match is token-boundary-aligned: it fires only when a token IS a
    shell name (``sh``, ``/bin/bash``, ...) and a LATER token of the same
    command is a -c-style option (``-c``, ``-xc``, ...). A shell name and
    option appearing as substrings inside one quoted multi-word argument
    (``git commit -m 'fix bash -c handling'``) are data the wrapper
    receives, never a nested shell the wrapper runs, so they never fire.
    """
    return any(
        Path(token).name in _SHELLS
        and any(
            option == "-c" or re.fullmatch(r"-[a-zA-Z]*c", option)
            for option in tokens[index + 1 :]
        )
        for index, token in enumerate(tokens[1:], 1)
    )


def _visit_tokens(part: str, tokens_override: tuple[str, ...] | None) -> list[str]:
    """The tokens a guardrail visit analyzes.

    An argv-wrapper reassembly supplies the ORIGINAL tokens so synthetic
    quoting never re-derives their provenance; real source text is freshly
    tokenized with its own quote provenance.
    """
    if tokens_override is not None:
        return list(tokens_override)
    try:
        return split_shell_tokens(part)
    except ValueError as exc:
        raise _UnsafeShellSyntax(
            "malformed shell quoting",
            denial_class=DenialClass.SYNTAX,
            alternative="write the command with well-formed shell quoting",
        ) from exc


def _expand_guardrail_parts(  # noqa: PLR0915
    command_parts: list[str],
    *,
    scratchpad_roots: frozenset[Path] = frozenset(),
    redirect_operands: tuple[tuple[str, ...], ...] = (),
    workspace_roots: frozenset[Path] | None = None,
    command_assignments: tuple[tuple[tuple[str, str], ...], ...] = (),
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
        if any(character in target for character in _SHELL_GLOB_CHARACTERS) and (
            not isinstance(target, ShellToken) or target.meta_unquoted
        ):
            # Shells pathname-expand redirect destinations with UNQUOTED glob
            # metacharacters (zsh writes every match; bash overwrites a single
            # match or creates the literal name), so the write-target set is
            # not statically knowable: fail closed (POLICY). A metacharacter
            # inside a quoted segment stays literal data (C2b provenance) and
            # a plain str carries no provenance at all, so both deny. Fires
            # before any later redirect check, so the redirect-glob denial
            # wins over the git-metadata guard.
            raise _UnsafeShellSyntax(
                f"redirect target '{target}' contains shell glob characters; "
                "the shell pathname-expands redirect targets, so the write "
                "target set cannot be safely inspected"
            )
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
        tokens_override: tuple[str, ...] | None = None,
        assignments: tuple[tuple[str, str], ...] = (),
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
                tokens=tokens_override,
                assignments=assignments,
            )
        )
        tokens = _visit_tokens(part, tokens_override)
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
                nested_assignments = analysis.command_assignments
                for index, child in enumerate(nested_parts):
                    targets = (
                        nested_operands[index] if index < len(nested_operands) else ()
                    )
                    # The nested shell's subprocess inherits the enclosing
                    # statement's bindings and the wrapper part's own
                    # command-prefix assignments — that is the environment
                    # the payload's commands run with — and the payload's
                    # own bindings, which the nested analysis recorded, win
                    # (POSIX prefix semantics — dict() keeps the last).
                    visit(
                        child,
                        child_scope,
                        depth + 1,
                        redirect_targets=targets,
                        assignments=assignments
                        + (
                            nested_assignments[index]
                            if index < len(nested_assignments)
                            else ()
                        ),
                    )
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
            # env's assignment operands are dropped from the reassembled
            # argv like the shell's own command-prefix assignments, but the
            # env subprocess applies them; recover them from the consumed
            # prefix for the egress proxy inspection. sudo's NAME=value
            # operands are recovered the same way: sudo applies them to the
            # wrapped command's environment (modulo sudoers scrubbing, which
            # the gate must never un-gate — the same fail-closed union
            # semantics the binding walk uses for conditional candidates).
            child_assignments: tuple[tuple[str, str], ...] = ()
            if boundary.inner_command_range is not None and _command_name(
                tokens[0]
            ) in {"env", "sudo"}:
                child_assignments = tuple(
                    (token.partition("=")[0], token.partition("=")[2])
                    for token in tokens[1:start]
                    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", token, re.DOTALL)
                )
            # The wrapped command inherits the wrapper's accumulated
            # assignments — the statement-level bindings in force where the
            # wrapper runs and its own command-prefix assignments are the
            # environment the wrapper's subprocess receives, and the model
            # threads them through every modeled wrapper. That threading is
            # fail-closed: a real sudo scrubs or env -i clears the child's
            # environment, but the gate must never un-gate a known
            # non-loopback proxy. env's operand assignments come last and
            # win: env applies its operands over the inherited environment
            # (POSIX prefix semantics — dict() keeps the last binding).
            # shlex.join is only a lossless argv serialization for the policy
            # helpers. This argv is NEVER parsed as shell source or an AST.
            # The original tokens ride along: re-deriving provenance from the
            # synthetic quoting would classify an unquoted metacharacter as
            # quoted data and bypass the pattern inspection.
            visit(
                shlex.join(child_tokens),
                child_scope,
                depth + 1,
                boundary.cwd_overrides,
                inspect_executable=True,
                tokens_override=tuple(child_tokens),
                assignments=assignments + child_assignments,
            )

    for index, part in enumerate(command_parts):
        targets = redirect_operands[index] if index < len(redirect_operands) else ()
        assignments = (
            command_assignments[index] if index < len(command_assignments) else ()
        )
        visit(part, (), 0, redirect_targets=targets, assignments=assignments)
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

# Work budgets for one pathname expansion. _match_glob_atoms is polynomial
# per enumerated name, but a crafted pattern can still multiply that per-name
# table by the number of directory entries it enumerates (and a long pattern
# makes the table itself large), so the expansion fails closed (None) once
# either budget is exceeded: permission resolution must stay bounded in time.
_MAX_EXPANSION_ENTRIES = 4096
_MAX_EXPANSION_MATCH_STATES = 1 << 19


class ExpansionFailure(StrEnum):
    """Why pathname-expansion inspection failed closed (provenance only).

    The kind selects a denial's detail text, never the verdict: every member
    denies exactly as the bare ``None`` this replaces did. ENTRY_BUDGET,
    MATCH_STATE_BUDGET, and UNSUPPORTED_SYNTAX name the actual inspection
    limit reached while expanding. CALLER_DENIED and NOT_INSPECTED cover
    denials decided outside the expansion inspection — an expansion whose
    results the caller rejects (an option-like word, a match outside the
    authorized workspace) and tokens the inspector never expanded (an
    unknown cwd, a home-relative word, a partially quoted metacharacter) —
    and render the generic path-glob message: attributing a work budget to
    them would be wrong.
    """

    ENTRY_BUDGET = "entry budget"
    MATCH_STATE_BUDGET = "match-state budget"
    UNSUPPORTED_SYNTAX = "unsupported glob syntax"
    CALLER_DENIED = "denied outside the expansion inspection"
    NOT_INSPECTED = "never pathname-expanded"


_EXPANSION_FAILURE_DETAIL = {
    ExpansionFailure.ENTRY_BUDGET: (
        "the pathname expansion exceeded its directory-entry budget"
    ),
    ExpansionFailure.MATCH_STATE_BUDGET: (
        "the pathname expansion exceeded its match-state budget"
    ),
    ExpansionFailure.UNSUPPORTED_SYNTAX: (
        "the pathname expansion uses glob syntax this inspector cannot expand"
    ),
}

# The compliant alternative the glob-inspection denial names: the shell
# would expand the pattern, the inspector cannot, so the reader is told how
# to hand the resolver literal operands instead (P2-e recognition contract).
_PATH_GLOB_ALTERNATIVE = (
    "list the directory with ls to see the matching names, or write the "
    "exact paths instead of the glob"
)


def _expansion_denial(
    token: str, failure: ExpansionFailure, command_part: str
) -> PermissionContext:
    """The path-glob denial carrying the expansion-failure provenance.

    The provenance rides the Diagnostic detail, so it passes through the
    renderer's redaction and bounding like every other diagnostic field. An
    inspection failure (a work budget, unsupported glob syntax, or a token
    the inspector never expanded) is a syntax denial and names the compliant
    alternative; CALLER_DENIED stays a policy denial — the caller rejected
    the expansion's RESULTS (an option-like word, a match outside the
    authorized workspace), so suggesting exact paths would only invite a
    retry that denies again.
    """
    if failure is ExpansionFailure.CALLER_DENIED:
        return _denied(Diagnostic("path_glob", token, command_part))
    return _denied(
        Diagnostic(
            "path_glob",
            token,
            command_part,
            detail=_EXPANSION_FAILURE_DETAIL.get(failure),
            denial_class=DenialClass.SYNTAX,
            alternative=_PATH_GLOB_ALTERNATIVE,
        )
    )


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


def _bracket_range_members(
    low: str, high: str, *, negated: bool
) -> frozenset[str] | None:
    """The characters a shell bracket range conservatively matches.

    Only ASCII digit and same-case ASCII letter endpoints are modeled. For a
    POSITIVE range, letter ranges are broadened to both cases because a
    collation-ordered range (``[a-z]`` under a UTF-8 locale) also matches the
    opposite case; the broadening grows the match set, so it only over-denies.
    A NEGATED range must not be broadened: the model excludes its members, so
    excluding the broadened set would shrink the modeled complement below
    every locale's real ``[!a-z]`` (which matches ``A``, ``Z``, ...). The
    un-broadened range is the narrowest any locale produces, so its
    complement is a superset of every locale's negated match set. A reversed
    range under a negated set fails closed: bash matches nothing for
    ``[c-a]``, so ``[!c-a]`` matches everything and no finite member set
    over-approximates it. Any other endpoint pair returns None: the caller
    must fail closed rather than trust the model's expansion.
    """
    digits = "0" <= low <= "9" and "0" <= high <= "9"
    letters = ("a" <= low <= "z" and "a" <= high <= "z") or (
        "A" <= low <= "Z" and "A" <= high <= "Z"
    )
    if not digits and not letters:
        return None
    if negated and ord(low) > ord(high):
        return None
    first, second = sorted((ord(low), ord(high)))
    members = {chr(point) for point in range(first, second + 1)}
    if letters and not negated:
        members |= {member.upper() for member in members}
        members |= {member.lower() for member in members}
    return frozenset(members)


def _parse_glob_component(component: str) -> tuple | None:
    """Parse one pathname component into conservative match atoms.

    Returns None when the component uses syntax this model cannot faithfully
    expand: POSIX character classes (``[[:alpha:]]``), collating or
    equivalence symbols, brace expansion, a range with unsupported
    endpoints, or a run of two or more consecutive stars. None means the
    caller fails closed.
    """
    atoms: list[tuple] = []
    index = 0
    length = len(component)
    while index < length:
        character = component[index]
        if character == "*":
            if index + 1 < length and component[index + 1] == "*":
                # Two or more consecutive stars are zsh's recursive globstar:
                # the component spans zero or more DIRECTORY levels, which
                # this one-component model cannot expand faithfully (parsing
                # it as two single-level stars would under-model the
                # expansion and authorize reads the shell really makes).
                # Fail closed; the over-denial also covers intra-segment
                # spellings like ``a**b``, matching the brace/POSIX-class
                # precedent.
                return None
            atoms.append(("star",))
            index += 1
            continue
        if character == "?":
            atoms.append(("question",))
            index += 1
            continue
        if character == "{":
            # Bash brace-expands before globbing; the model cannot.
            return None
        if character != "[":
            atoms.append(("literal", character))
            index += 1
            continue
        cursor = index + 1
        negated = cursor < length and component[cursor] in "!^"
        if negated:
            cursor += 1
        members: set[str] = set()
        first_content = True
        closed = False
        while cursor < length:
            member = component[cursor]
            if member == "]" and not first_content:
                closed = True
                cursor += 1
                break
            first_content = False
            if member == "[" and cursor + 1 < length and component[cursor + 1] in ":.=":
                return None
            if (
                cursor + 2 < length
                and component[cursor + 1] == "-"
                and component[cursor + 2] != "]"
            ):
                range_members = _bracket_range_members(
                    member, component[cursor + 2], negated=negated
                )
                if range_members is None:
                    return None
                members |= range_members
                cursor += 3
                continue
            members.add(member)
            cursor += 1
        if not closed:
            # An unclosed '[' is a literal, matching bash and fnmatch.
            atoms.append(("literal", "["))
            index += 1
            continue
        atoms.append(("set", frozenset(members), negated))
        index = cursor
    return tuple(atoms)


def _match_glob_atoms(name: str, atoms: tuple) -> bool:
    """Whether one filename matches a parsed component (classic glob).

    Memoized dynamic programming over (name index, atom index) states: the
    previous recursive backtracking re-explored every split point of every
    star, so a crafted pattern (``*a`` repeated against a long name with an
    impossible suffix) stalled permission resolution exponentially. Every
    state is now computed at most once, so matching is polynomial in the
    name and atom counts.
    """
    name_length = len(name)
    atom_count = len(atoms)
    # matches[atom_index][name_index]: name[name_index:] matches
    # atoms[atom_index:]. Atom rows fill from the last atom backwards, and a
    # star row reads its own next name index, so name indices fill downwards.
    matches = [[False] * (name_length + 1) for _ in range(atom_count + 1)]
    matches[atom_count][name_length] = True
    for atom_index in range(atom_count - 1, -1, -1):
        atom = atoms[atom_index]
        kind = atom[0]
        row = matches[atom_index]
        next_row = matches[atom_index + 1]
        for name_index in range(name_length, -1, -1):
            if kind == "star":
                # A star matches zero characters (advance past it) or one
                # more (consume a character and stay on the star).
                row[name_index] = next_row[name_index] or (
                    name_index < name_length and row[name_index + 1]
                )
            elif name_index >= name_length:
                row[name_index] = False
            elif kind == "question":
                row[name_index] = next_row[name_index + 1]
            elif kind == "literal":
                row[name_index] = (
                    name[name_index] == atom[1] and next_row[name_index + 1]
                )
            else:  # set
                in_set = name[name_index] in atom[1]
                if atom[2]:
                    in_set = not in_set
                row[name_index] = in_set and next_row[name_index + 1]
    return matches[0][0]


def _pattern_expansion(
    cwd: Path, token: str
) -> list[tuple[Path, str]] | ExpansionFailure:
    """A conservative superset of the shell's pathname expansion of *token*.

    Returns ``(path, word)`` pairs: the expanded filesystem path and the
    command-line word the shell substitutes. The cwd is anchored literally
    (never pattern-interpreted), and every component is matched against the
    filesystem with the conservative model above. Returns an
    :class:`ExpansionFailure` when the pattern uses syntax the model cannot
    faithfully expand, when the expansion exceeds its work budget, or when
    the token has no pathname components: the caller must fail closed,
    never authorize the empty expansion. The failure kind is provenance for
    the denial message only — every failure denies identically.
    """
    items: list[tuple[Path, str]] = [(cwd, "")]
    if token.startswith("/"):
        items = [(Path("/"), "/")]
    components = [component for component in token.split("/") if component]
    if not components:
        return ExpansionFailure.NOT_INSPECTED
    enumerated = 0
    match_states = 0
    for index, component in enumerate(components):
        last = index == len(components) - 1
        if component in {".", ".."}:
            items = [
                (
                    directory.parent if component == ".." else directory,
                    f"{word}/{component}"
                    if word not in {"", "/"}
                    else word + component,
                )
                for directory, word in items
            ]
            continue
        atoms = _parse_glob_component(component)
        if atoms is None:
            return ExpansionFailure.UNSUPPORTED_SYNTAX
        has_meta = any(atom[0] != "literal" for atom in atoms)
        nxt: list[tuple[Path, str]] = []
        for directory, word in items:
            if not has_meta:
                child_word = (
                    f"{word}/{component}" if word not in {"", "/"} else word + component
                )
                candidate = directory / component
                if last or candidate.is_dir():
                    nxt.append((candidate, child_word))
                continue
            try:
                names = sorted(os.listdir(directory))
            except OSError:
                continue
            enumerated += len(names)
            if enumerated > _MAX_EXPANSION_ENTRIES:
                # A crafted pattern over a large tree must not turn
                # permission resolution into an unbounded walk: fail closed.
                return ExpansionFailure.ENTRY_BUDGET
            for name in names:
                # A leading dot must be matched by a literal leading dot.
                if name.startswith(".") and not (
                    atoms[0][0] == "literal" and atoms[0][1] == "."
                ):
                    continue
                # The memoized matcher's table is (atoms + 1) x (name + 1)
                # states; count them so a long pattern cannot stall the
                # resolution either.
                match_states += (len(atoms) + 1) * (len(name) + 1)
                if match_states > _MAX_EXPANSION_MATCH_STATES:
                    return ExpansionFailure.MATCH_STATE_BUDGET
                if not _match_glob_atoms(name, atoms):
                    continue
                path = directory / name
                if last or path.is_dir():
                    match_word = (
                        f"{word}/{name}" if word not in {"", "/"} else word + name
                    )
                    nxt.append((path, match_word))
        items = nxt
    return items


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
        if entry.tokens is not None:
            # Provenance-carrying tokens from an argv-wrapper reassembly.
            tokens = list(entry.tokens)
        else:
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
    "egrep",
    "fgrep",
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
    "rg",
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
        # Modeled network clients (v0.5.9 P2-a): curl and wget left this
        # denylist — their DESTINATIONS decide run vs ask under the
        # destination-aware egress rule (see _destination.py), and their
        # local-file operands are path-inspected by their command policy.
        # Every other egress client stays denied fail-closed until modeled:
        # denylist and path policy control known network exfiltration
        # routes (redaction cannot observe pipe/socket transit). These
        # controls harden against known payload shapes, not complete egress
        # containment.
        "wget2",
        "nc",
        "ncat",
        "socat",
        # Probed egress clients and the watch repeater: resolved ALWAYS for
        # arbitrary payloads, so denied fail-closed as the interim until each
        # is modeled (v0.5.9 P1-e2). They stay denied under the
        # destination-aware egress rule until modeled there.
        "ssh",
        "scp",
        "rsync",
        "ftp",
        "telnet",
        "http",
        "httpie",
        "aria2c",
        "lftp",
        "watch",
        # v0.5.9 P1-e3 wrapper-class audit closure: every payload-runner
        # class and named egress client was probed through Bash's permission
        # resolution, and each entry dispositioned. The audit record:
        #
        # modeled transparent (payload re-analyzed; pinned in the corpus
        # permission family): nice, env, stdbuf, flock, setsid, timeout,
        # nohup, ionice, taskset, time, sg, sudo -u (v0.5.9 P2-d).
        #
        # modeled network clients (v0.5.9 P2-a, destination-aware ask-class
        # egress rule): curl, wget — off the denylist, classified by
        # destination in _destination.py, local-file operands path-inspected.
        #
        # fail-closed without a denylist entry (a modeled boundary refuses
        # the form): xargs (unsafe/unmodeled options), find -exec
        # (execution predicates), script -c (unsupported wrapper form).
        # sudo's unmodeled forms (no -u, a shell behind -s/-i, unmodeled
        # options, a missing command) fail closed the same way; the shipped
        # sensitive-prefix rule for sudo retired with the P2-d modeling.
        #
        # run-class by client (exfiltration decision item 4): git clone,
        # git fetch, git pull travel through the modeled git client and
        # are run-class, not denylisted; git push stays classified
        # dangerous per operation.
        #
        # denylisted fail-closed below: every probed entry that resolved
        # ALWAYS without any payload analysis, denied until modeled, one
        # corpus row each in the permission family:
        #   group/namespace wrappers: su, chroot, unshare, nsenter,
        #     parallel
        #   build runners: make, rake, just, ninja, cmake, gradle, mvn,
        #     ant, scons, meson
        #   container runners: docker, podman, kubectl
        #   schedulers: at, batch, systemd-run, crontab
        #   repeaters: watch (above), entr
        #   egress clients: sftp, mosh, gh, curlie, xh, httpx, tftp,
        #     fetch, openssl s_client, openssl s_server
        # atq/atrm only inspect and remove the at queue; they are not
        # payload runners and stay run-class.
        # Group wrappers:
        "su",
        "chroot",
        "unshare",
        "nsenter",
        "parallel",
        # Build runners: a Makefile/Rakefile/build-script target runs
        # arbitrary commands, so the runner cannot be admitted unanalyzed.
        "make",
        "rake",
        "just",
        "ninja",
        "cmake",
        "gradle",
        "mvn",
        "ant",
        "scons",
        "meson",
        # Container runners: the image entrypoint and exec payload run
        # arbitrary commands.
        "docker",
        "podman",
        "kubectl",
        # Schedulers: at/batch submit scripts for later execution;
        # systemd-run and crontab install units and schedules.
        "at",
        "batch",
        "systemd-run",
        "crontab",
        # Repeaters: watch (above), entr runs a command on file change.
        "entr",
        # Egress clients probed by the P1-e3 audit, denied until modeled
        # or until the destination-aware egress rule covers them.
        "sftp",
        "mosh",
        "gh",
        "curlie",
        "xh",
        "httpx",
        "tftp",
        "fetch",
        "openssl s_client",
        "openssl s_server",
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
        return split_shell_tokens(command)
    except ValueError:
        return command.split()


def _part_command(tokens: list[str]) -> str:
    """The command a guardrail part runs, keeping the dot builtin's bare name.

    ``Path(".").name`` is empty, so dot sourcing needs its raw token to join
    the path-inspected surface.
    """
    return Path(tokens[0]).name or tokens[0]


def _find_pattern_operand_uninspectable(token: str) -> bool:
    """Whether a find pattern operand's globs are unquoted, hence injectable.

    The shell pathname-expands an unquoted glob-bearing pattern word before
    find runs, so the expansion products re-enter find's argv: a decoy file
    named `-exec` turns ``find . -name -* ...`` into an execution predicate.
    The inspector cannot expand pattern operands, so the pattern-operand
    carve-out must not assume the word stays inert — an unquoted glob fails
    closed at the path-glob rule, exactly like a redirect target that looks
    like a glob. A metacharacter inside a quoted segment stays literal data
    (C2b provenance) and a plain str carries no provenance at all, so both
    deny.
    """
    return any(character in token for character in _SHELL_GLOB_CHARACTERS) and (
        not isinstance(token, ShellToken) or token.meta_unquoted
    )


# Only commands whose modeled operands cannot be output files receive exact
# instruction-read capabilities. E.g. sort -o, uniq's second operand, sed and
# archivers are deliberately not classified as readers. The script-loading
# builtins read their operand as shell source, so they are readers too: an
# outside-workspace script is denied like any other outside read operand.
_INSTRUCTION_READ_COMMANDS = frozenset({
    ".",
    "cat",
    "egrep",
    "fgrep",
    "grep",
    "rg",
    "head",
    "tail",
    "wc",
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


def _rm_operands(tokens: list[str]) -> list[str]:
    """The positional operands of one rm invocation.

    GNU rm permutes options, so the scan keeps skipping option tokens after
    an operand (``rm dir -rf`` is recursive) until ``--`` ends option
    scanning; a bare ``-`` is an operand (a file named ``-``), not an option.
    """
    operands: list[str] = []
    options_ended = False
    for token in tokens[1:]:
        if token == "--" and not options_ended:
            options_ended = True
            continue
        if not options_ended and token.startswith("-") and token != "-":
            continue
        operands.append(token)
    return operands


def _rm_operand_glob_bearing(token: str) -> bool:
    """Whether the shell pathname-expands this rm operand.

    The same active-metacharacter reading the path layer's glob rule uses: a
    plain ``str`` carries no quote provenance, so its metacharacters read as
    unquoted, and a quoted glob still analyzes as its (superset) expansion.
    """
    return any(character in token for character in "*?") or _has_active_bracket_glob(
        token.encode()
    )


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
        injected_env: Mapping[str, str] | None = None,
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
        # Trust-gated project-injected variables, not the full subprocess
        # environment. The egress rule inspects injected non-loopback proxies;
        # ambient, non-injected environment variables remain uninspected.
        # Direct classifier users may omit this mapping.
        self._injected_env = injected_env

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
        state_cache: dict[tuple[Path, Path | None, bool, frozenset[Path]], bool] = {}
        scoped = list(_scoped_guardrail_cwds(expanded, self.cwd))
        policies = [
            ShellCommandPolicy()
            if entry.access is not None
            else analyze_shell_command_policy(tokens)
            for entry, tokens, _possible_cwds, _cwd_unknown in scoped
        ]
        # Recovery evidence must survive every statement's rm operands, mv
        # sources, and collected writes, regardless of statement order. Earlier
        # writes also invalidate the later rm's clean-tree eligibility scope.
        # Two separate bash calls remain outside one resolution's view: each
        # uses the current on-disk state, with no cross-call state kept.
        call_footprint: frozenset[Path] | None = frozenset()
        prior_writes: list[frozenset[Path] | None] | None = None
        if any(
            policy.requires_approval
            and policy.diagnostic is not None
            and policy.diagnostic.code == "recursive_rm"
            for policy in policies
        ):
            call_footprint = self._call_recovery_destruction_footprint(scoped)
            prior_writes = self._prior_write_surfaces(scoped)
        for position, (entry, tokens, possible_cwds, cwd_unknown) in enumerate(scoped):
            if entry.access is not None:
                continue
            part = entry.text
            if self._is_sensitive(part):
                return _denied(Diagnostic("sensitive_command"))
            policy = policies[position]
            if policy.requires_approval:
                if (
                    policy.diagnostic is not None
                    and policy.diagnostic.code == "recursive_rm"
                    and self._recursive_rm_eligible(
                        tokens,
                        possible_cwds,
                        cwd_unknown,
                        state_cache,
                        call_footprint=call_footprint,
                        prior_writes=(
                            prior_writes[position]
                            if prior_writes is not None
                            else frozenset()
                        ),
                    )
                ):
                    # Run-class under the full P2-c condition: every expanded
                    # operand is in-repository, metadata-excluding, and
                    # recovery-established. The path layer still gates each
                    # operand independently — a run-class resolution dies at
                    # a gate — and every other denial path is unchanged.
                    continue
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

    def _call_recovery_destruction_footprint(
        self, scoped: list[tuple[_GuardrailPart, list[str], set[Path], bool]]
    ) -> frozenset[Path] | None:
        """All rm operands, mv sources, and collected writes across this call.

        A recovery target equal to or below any member cannot survive. All
        statements contribute regardless of order or conditional execution;
        directories conservatively cover their whole subtree. Each operand
        resolves under every possible cwd, with rm/mv source globs expanded.
        Unknown effects fail recursive-rm eligibility closed, never silently
        disappear from the destruction set.
        """
        footprint: set[Path] = set()
        for entry, tokens, possible_cwds, cwd_unknown in scoped:
            surface = self._segment_write_paths(
                entry, tokens, possible_cwds, cwd_unknown
            )
            if surface is None:
                return None
            footprint.update(surface)
            if not tokens or entry.access is not None:
                continue
            command = _part_command(tokens)
            if command not in {"rm", "mv"}:
                continue
            if cwd_unknown or not possible_cwds:
                return None
            if command == "rm":
                operands = _rm_operands(tokens)
            else:
                parsed = _scan_options(tokens[1:], _DESTINATION_POSITION_TABLES["mv"])
                operands = list(
                    parsed.positionals
                    if parsed.values_for("-t", "--target-directory")
                    else parsed.positionals[:-1]
                )
            for token in operands:
                for cwd in possible_cwds:
                    paths = self._rm_operand_paths(token, cwd)
                    if paths is None:
                        return None
                    footprint.update(paths)
        return frozenset(footprint)

    def _prior_write_surfaces(
        self, scoped: list[tuple[_GuardrailPart, list[str], set[Path], bool]]
    ) -> list[frozenset[Path] | None]:
        """Per-position prior write surfaces for a call holding a recursive-rm.

        Indexed by expanded-part position: entry ``i`` holds the union of
        the write surfaces of every statement that ends before the
        statement containing part ``i``. A statement is one command
        together with the synthetic redirect parts attached to it (they are
        synthesized immediately before their command part), so a segment's
        OWN redirects belong to its statement, not to an earlier one. None
        marks a position whose earlier write surface could not be
        established; a later recursive-rm segment fails closed on it.
        """
        priors: list[frozenset[Path] | None] = []
        accumulated: frozenset[Path] | None = frozenset()
        pending: frozenset[Path] | None = frozenset()
        for entry, tokens, possible_cwds, cwd_unknown in scoped:
            surface = self._segment_write_paths(
                entry, tokens, possible_cwds, cwd_unknown
            )
            priors.append(accumulated)
            if entry.access is not None:
                # A synthetic redirect operand joins the statement of the
                # command part that follows it, not a statement of its own.
                pending = _union_write_surfaces(pending, surface)
                continue
            accumulated = _union_write_surfaces(
                _union_write_surfaces(accumulated, pending), surface
            )
            pending = frozenset()
        return priors

    def _segment_write_paths(
        self,
        entry: _GuardrailPart,
        tokens: list[str],
        possible_cwds: set[Path],
        cwd_unknown: bool,
    ) -> frozenset[Path] | None:
        """The resolved write surface of one expanded part, or None (unknown).

        The surface is the existing per-segment write-path collection —
        the same classification and resolution ``shell_command_write_paths``
        applies (write-classified operands, write-path option values, the
        mutating commands' own destinations), resolved against the part's
        tracked cwds — PLUS the mutation scope of a checkout/switch
        invocation: its resolved work-tree root is a bulk work-tree write
        the recovery/destruction analysis accounts exactly like any other
        segment write (an earlier checkout/switch dirties the clean-tree
        state a later recursive-rm segment's eligibility relies on, and its
        root joins the whole call's destruction footprint). None means the
        surface could not be established (an unresolvable write operand,
        eval/exec's opaque effects, or an unresolvable checkout/switch
        root), which callers treat as unknown and fail closed on.

        The mutation root is a PERMISSION-footprint member only: the
        run-phase submission-scan extraction (``shell_command_write_paths``)
        never collects it — the scan walks its collected paths recursively
        with a bounded budget, and a work-tree root would burn it.
        """
        if not tokens:
            return frozenset()
        if _part_command(tokens) in {"eval", "exec"}:
            return None
        part_command = _part_command(tokens)
        git_write_values = frozenset()
        git_write_cwds: frozenset[Path] | None = None
        if part_command == "git":
            git_write_values = _git_write_values(tokens)
            if git_write_values:
                relocated = git_cwd_relocations(tokens, possible_cwds)
                if relocated is not None:
                    git_write_cwds = frozenset(relocated)
        paths: set[Path] = set()
        for token in _write_candidates(entry, tokens, part_command):
            if _write_operand_unresolvable(token, cwd_unknown):
                return None
            for operand_cwd in _operand_resolution_cwds(
                token,
                tokens,
                possible_cwds,
                git_write_values=git_write_values,
                git_write_cwds=git_write_cwds,
            ):
                paths.add(resolve_tool_path(token, operand_cwd))
        if _git_mutation_root_subcommand(tokens) is not None:
            # A checkout/switch rewrites tracked work-tree content in
            # bulk: its mutation scope — the resolved work-tree root — is
            # a segment write, resolved exactly as the mutation-roots
            # footprint resolves it.
            if (
                cwd_unknown
                or not possible_cwds
                or any(name.startswith("GIT_") for name, _value in entry.assignments)
            ):
                # The root the invocation would mutate cannot be
                # established: an unknown cwd, or a GIT_-prefixed assignment
                # (GIT_DIR/GIT_WORK_TREE) relocating the invocation outside
                # the modeled resolution.
                return None
            for possible_cwd in possible_cwds:
                root = resolve_git_work_tree(tokens, possible_cwd)
                if root is None:
                    return None
                paths.add(root)
        return frozenset(paths)

    def _recursive_rm_eligible(  # noqa: PLR0911
        self,
        tokens: list[str],
        possible_cwds: set[Path],
        cwd_unknown: bool,
        state_cache: dict[tuple[Path, Path | None, bool, frozenset[Path]], bool],
        *,
        call_footprint: frozenset[Path] | None,
        prior_writes: frozenset[Path] | None,
    ) -> bool:
        """Whether one recursive rm resolves run-class (v0.5.9 P2-c decision 2).

        Over a git-clean tree, run-class requires EVERY expanded operand to
        lie entirely within the repository whose clean status is assessed,
        to not overlap that repository's Git metadata (neither encompassing
        the Git directory nor lying inside it, including linked-worktree
        common directories), and to leave recovery established for every
        repository contained in or overlapped by the operand (nested
        repositories included). Recovery evidence must survive the ENTIRE
        destruction footprint of the WHOLE CALL: a configured remote's fetch
        target may not lie inside (or equal) any rm operand, mv source, or
        collected write surface in any statement. A write an earlier
        statement of the same call performs
        inside the segment's eligibility scope — the operand footprint, the
        assessed repository's work tree, or a nested repository's work tree
        — fails the segment closed: the write dirties the clean-tree state
        the eligibility relies on, or writes content the deletion destroys.
        Every unestablishable aspect — operand scope, metadata overlap,
        recovery status, expansion, resolution, an earlier statement's write
        surface — and any mixed operand resolves the whole command to
        handoff instead (the shipped blanket denial stays the fallback).
        """
        if cwd_unknown or call_footprint is None:
            return False
        if prior_writes is None:
            # An earlier statement's write surface could not be established;
            # it may land inside this segment's eligibility scope, so the
            # eligibility fails closed.
            return False
        operands = _rm_operands(tokens)
        if not operands:
            # rm without operands removes nothing.
            return True
        for cwd in possible_cwds:
            scope = resolve_repository_scope(cwd, GitRepositoryRelocation())
            assessed = repository_state(scope) if scope is not None else None
            if assessed is None:
                # No repository governs this directory, or its metadata
                # layout cannot be established: fail closed.
                return False
            # Include this segment's expanded operands in the whole-call
            # destruction set; no recovery target may lie beneath a member.
            resolved: list[Path] = []
            for token in operands:
                paths = self._rm_operand_paths(token, cwd)
                if paths is None:
                    return False
                resolved.extend(paths)
            deletion_footprint = call_footprint | frozenset(resolved)
            # The assessed repository's own recovery: a clean tree and no
            # in-progress operation. Its commits are not at risk — the
            # metadata overlap check below protects its Git directory.
            if not self._repository_recovery(
                assessed,
                commits_at_risk=False,
                deletion_footprint=deletion_footprint,
                cache=state_cache,
            ):
                return False
            for operand in resolved:
                if not self._rm_operand_eligible(
                    operand, assessed, deletion_footprint, state_cache, prior_writes
                ):
                    return False
        return True

    def _rm_operand_paths(self, token: str, cwd: Path) -> list[Path] | None:
        """The resolved paths one rm operand deletes, or None (fail closed).

        A glob-bearing operand is pathname-expanded by the shell, so the
        expanded match set is the operand set; an expansion the inspector
        cannot faithfully evaluate fails closed, and an empty match set
        leaves the literal word (bash passes it through unexpanded). A
        home-relative glob is never expanded: the shell expands ``~`` before
        globbing, so the inspector's literal reading would under-model it.
        """
        if not _rm_operand_glob_bearing(token):
            return [resolve_tool_path(token, cwd)]
        if token.startswith("~"):
            return None
        expansion = _pattern_expansion(cwd, token)
        if isinstance(expansion, ExpansionFailure):
            return None
        if expansion:
            return [match.resolve() for match, _word in expansion]
        return [resolve_tool_path(token, cwd)]

    def _rm_operand_eligible(
        self,
        operand: Path,
        assessed: RepositoryState,
        deletion_footprint: frozenset[Path],
        state_cache: dict[tuple[Path, Path | None, bool, frozenset[Path]], bool],
        prior_writes: frozenset[Path] | None,
    ) -> bool:
        """The full P2-c condition for one resolved rm operand."""
        # (a) The operand lies entirely within the assessed repository. The
        # assessed repository always carries a work tree; a state without
        # one fails closed.
        if assessed.work_tree is None or not operand.is_relative_to(assessed.work_tree):
            return False
        # (b) The operand does not overlap the repository's Git metadata —
        # neither encompassing it nor lying inside it.
        for metadata in assessed.metadata_dirs:
            if operand.is_relative_to(metadata) or metadata.is_relative_to(operand):
                return False
        # (c) Recovery is established for every repository the operand
        # contains or overlaps (nested repositories included). A repository
        # whose Git metadata the operand would destroy additionally needs
        # its commits to be safe: no commit may exist that no
        # remote-tracking ref reaches, and the recovery copy behind a
        # configured remote must survive the ENTIRE deletion footprint —
        # this operand and every other operand of the same command.
        nested = repositories_intersecting(operand, assessed)
        if nested is None:
            return False
        if prior_writes:
            # (d) No earlier statement of the same call writes into this
            # segment's eligibility scope: the operand footprint, the
            # assessed repository's work tree, or a nested repository's
            # work tree. Such a write dirties the clean-tree state the
            # eligibility relies on, or writes content the deletion
            # destroys, so the eligibility fails closed.
            protected = {
                operand,
                assessed.work_tree,
                *(repo.work_tree for repo in nested if repo.work_tree is not None),
            }
            if any(
                write.is_relative_to(location)
                for write in prior_writes
                for location in protected
            ):
                return False
        for repo in nested:
            commits_at_risk = any(
                metadata.is_relative_to(operand) for metadata in repo.metadata_dirs
            )
            if not self._repository_recovery(
                repo,
                commits_at_risk=commits_at_risk,
                deletion_footprint=deletion_footprint,
                cache=state_cache,
            ):
                return False
        return True

    def _repository_recovery(
        self,
        repo: RepositoryState,
        *,
        commits_at_risk: bool,
        deletion_footprint: frozenset[Path],
        cache: dict[tuple[Path, Path | None, bool, frozenset[Path]], bool],
    ) -> bool:
        """The cached recovery verdict for one repository under one deletion."""
        key = (repo.git_dir, repo.work_tree, commits_at_risk, deletion_footprint)
        if key not in cache:
            cache[key] = repository_recovery(
                repo,
                commits_at_risk=commits_at_risk,
                deletion_footprint=deletion_footprint,
            )
        return cache[key]

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

    def _ls_listing_glob_expansion_failure(
        self,
        *,
        token: str,
        possible_cwds: set[Path],
        cwd_unknown: bool,
        access: PathAccess,
        authority: PathAuthority,
    ) -> ExpansionFailure | None:
        """Where an ls listing glob would be denied, else None.

        Provenance only: the returned kind never changes the verdict. A
        token the inspector never expanded (an unknown cwd, a home-relative
        or absolute word, a glob-bearing parent) reports NOT_INSPECTED; an
        expansion-inspection failure names its actual limit.
        """
        if cwd_unknown or Path(token).is_absolute() or token.startswith("~"):
            return ExpansionFailure.NOT_INSPECTED
        parent = str(Path(token).parent)
        if any(character in parent for character in _SHELL_GLOB_CHARACTERS):
            return ExpansionFailure.NOT_INSPECTED
        for cwd in possible_cwds:
            if not (
                self.workspace.allows(resolve_tool_path(parent, cwd))
                and authority.allows(resolve_tool_path(parent, cwd), access)
                and authority.allows(resolve_tool_path(token, cwd), access)
            ):
                return ExpansionFailure.CALLER_DENIED
            failure = self._expansion_failure(token, {cwd}, authority, access)
            if failure is not None:
                return failure
        return None

    def _pattern_glob_expansion_failure(
        self,
        token: str,
        possible_cwds: set[Path],
        cwd_unknown: bool,
        authority: PathAuthority,
        access: PathAccess,
    ) -> ExpansionFailure | None:
        """Where a pattern-position glob would be denied, else None.

        A grep-family pattern word whose glob metacharacters are all unquoted
        is pathname-expanded by the shell before the command runs, so the
        expansion's matches join the read-inspected surface. The cwd is
        anchored literally and the expansion is the conservative model in
        :func:`_pattern_expansion`: syntax it cannot faithfully evaluate
        fails closed, never authorizing an empty expansion. An empty match
        set from expandable syntax passes the literal word through as regex
        data, which is safe. PREFLIGHT ONLY: the match set checked here is
        not bound to the shell's subsequent execution, so the model must
        stay a superset of any plausible expansion. Provenance only: the
        returned kind never changes the verdict, and a token the inspector
        never expanded (an unknown cwd or a home-relative word) reports
        NOT_INSPECTED, not a work budget.
        """
        if cwd_unknown or token.startswith("~"):
            return ExpansionFailure.NOT_INSPECTED
        return self._expansion_failure(token, possible_cwds, authority, access)

    def _expansion_failure(
        self,
        token: str,
        possible_cwds: set[Path],
        authority: PathAuthority,
        access: PathAccess,
    ) -> ExpansionFailure | None:
        """Why pathname-expansion inspection would deny *token*, else None.

        Provenance only: the returned kind selects the denial's detail
        text, never the verdict — every failure here denies exactly as the
        boolean this method replaces. A failure inside the expansion
        (unsupported syntax, an exhausted work budget) names its limit; a
        failure of the expansion's RESULTS (an option-like word, a match
        outside the authorized workspace) is this caller's own denial and
        reports CALLER_DENIED, never a budget or syntax attribution.
        """
        for cwd in possible_cwds:
            expansion = _pattern_expansion(cwd, token)
            if isinstance(expansion, ExpansionFailure):
                # Unevaluable syntax or an exhausted work budget: fail
                # closed instead of authorizing an empty expansion the
                # shell could still fill.
                return expansion
            for match, word in expansion:
                # The shell substitutes the word into the command's argv, so
                # the command re-parses it: a spelling beginning with '-'
                # would inject an option (rg --pre, grep -f, ...). Fail
                # closed; a './'-prefixed spelling is not option-like.
                if not token.startswith("/") and word.startswith("-"):
                    return ExpansionFailure.CALLER_DENIED
                resolved = match.resolve()
                if not (
                    self.workspace.allows(resolved)
                    and authority.allows(resolved, access)
                ):
                    return ExpansionFailure.CALLER_DENIED
        return None

    def _command_operand_glob_denial(
        self,
        *,
        token: str,
        entry: _GuardrailPart,
        command: str,
        possible_cwds: set[Path],
        cwd_unknown: bool,
        access: PathAccess,
        authority: PathAuthority,
    ) -> PermissionContext | None:
        """The path-glob denial for one glob-bearing command operand, else None.

        Called only for a read-context operand that is a pattern-position
        candidate or bears glob characters. None means the operand's globs
        are fully accounted for and the caller proceeds past the token: a
        pattern-position candidate whose expansion inspection passed, or an
        ls listing glob within every scoped cwd. Every other such operand
        denies here — with the expansion's provenance when the inspection
        itself failed (a work budget or unsupported syntax), generically
        when the denial was decided outside it.

        Synthetic redirect operands are WRITE-stamped by
        append_redirect_operand, so this heuristic never sees them;
        redirect-target glob denial lives solely at that seam.
        """
        if entry.access is None and isinstance(token, PatternGlobCandidate):
            failure = self._pattern_glob_expansion_failure(
                token, possible_cwds, cwd_unknown, authority, access
            )
            if failure is None:
                return None
            return _expansion_denial(token, failure, entry.text)
        # Python glob and shell bracket expressions differ. Dot-leading
        # patterns may also expand to . or .. in shells.
        supported_glob = "[" not in token and not Path(token).name.startswith(".")
        if command == "ls" and supported_glob:
            failure = self._ls_listing_glob_expansion_failure(
                token=token,
                possible_cwds=possible_cwds,
                cwd_unknown=cwd_unknown,
                access=access,
                authority=authority,
            )
            if failure is None:
                return None
            return _expansion_denial(token, failure, entry.text)
        return _denied(Diagnostic("path_glob", token, entry.text))

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
                # The find pattern-operand carve-out keeps quoted and literal
                # pattern operands (the shell never pathname-expands them),
                # but an unquoted glob-bearing operand is expanded by the
                # shell into find's argv — a decoy file named `-exec` injects
                # an execution predicate — so the inspector, which cannot
                # expand pattern operands, fails closed on it at the
                # path-glob denial below instead of skipping it.
                if (
                    command == "find"
                    and token in tokens[2:]
                    and any(
                        tokens[index] == token
                        and tokens[index - 1] in {"-name", "-iname", "-path", "-ipath"}
                        for index in range(2, len(tokens))
                    )
                    and not _find_pattern_operand_uninspectable(token)
                ):
                    continue
                # A grep-family pattern position whose glob metacharacters
                # are all unquoted is pathname-expanded by the shell before
                # the command runs, so its expansion joins the read-inspected
                # surface: in-root matches pass, any out-of-root match (or a
                # cwd the resolver cannot locate) falls through to the
                # path-glob denial below. Fully quoted metacharacters never
                # arrive here; partially quoted ones arrive unmarked and deny.
                # The expansion check is preflight-only: the checked match
                # set is not bound to the shell's subsequent execution.
                if entry.access is None and (
                    isinstance(token, PatternGlobCandidate)
                    or any(character in token for character in "*?")
                    or _has_active_bracket_glob(token.encode())
                ):
                    denial = self._command_operand_glob_denial(
                        token=token,
                        entry=entry,
                        command=command,
                        possible_cwds=possible_cwds,
                        cwd_unknown=cwd_unknown,
                        access=access,
                        authority=authority,
                    )
                    if denial is not None:
                        return denial
                    continue
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
        to a critical metadata file, which path resolution cannot see. A
        nested bare repository's metadata is gated the same way: the operand's
        own resolved ancestry is probed for git's bare-repository markers, so
        an operand reaching inside a nested bare repository denies exactly
        like a ``.git`` operand, through symlinked components included.
        """
        if ".git" in Path(token).parts or ".git" in resolved.parts:
            return "metadata_operand"
        if any(resolved.is_relative_to(directory) for directory in guard.directories):
            return "metadata_operand"
        if _is_hardlinked_metadata(resolved, guard.inodes):
            return "metadata_hardlink"
        reaches = path_reaches_bare_repository(resolved)
        if reaches is None:
            # The bare-repository probe failed, so the operand's containment
            # cannot be established; fail closed.
            return "metadata_unlocated"
        if reaches:
            return "metadata_operand"
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
                reaches = path_reaches_bare_repository(resolved)
                if reaches is None:
                    # The bare-repository probe failed, so the target's
                    # containment cannot be established; fail closed.
                    hit = "metadata_unlocated"
                    break
                if reaches:
                    # A nested bare repository's metadata is gated like a
                    # .git target: the redirect writes inside it.
                    hit = "metadata_redirect"
                    break
            if hit is not None:
                return _denied(Diagnostic(hit, path))
        return None

    def _checkout_exemption_eligible(
        self,
        exemption: _CheckoutExemption,
        scoped_candidates: list[tuple[Path, RepositoryScope]],
        cache: dict[tuple[Path, Path | None, str, str | None, str | None], bool],
    ) -> bool:
        """Whether every candidate repository scope passes the A2 guardrails.

        The command may run in any statically possible directory, so every
        candidate scope must pass (a fully clean work tree, no in-progress
        operation marker, no program-selecting repository config, and an
        existing local target branch for the plain branch form). The cache
        is per call — one probe set per repository and shape, never across
        calls — following the resolver's recovery-cache pattern.

        The probes assess the EFFECTIVE work tree — the same
        core.worktree-aware resolution an actual checkout honors (and the
        mutation-root path collects) — never the discovery directory alone:
        a configured relocated work tree can be dirty while the discovery
        directory probes clean. An effective work tree that cannot be
        established fails the exemption closed.
        """
        shape = exemption.shape
        entry = exemption.entry
        tokens = (
            list(entry.tokens)
            if entry.tokens is not None
            else _split_command_tokens(entry.text)
        )
        for cwd, scope in scoped_candidates:
            assessed = repository_state(scope)
            if assessed is None or assessed.work_tree is None:
                # No work tree to keep clean (a nominal non-repository
                # scope or a bare repository): the exemption fails closed.
                return False
            effective = resolve_git_work_tree(tokens, cwd)
            if effective is None:
                # The work tree the checkout would rewrite cannot be
                # established: the exemption fails closed.
                return False
            if effective != assessed.work_tree:
                # Probe the work tree the checkout actually rewrites; the
                # metadata directories stay the discovered repository's.
                assessed = replace(assessed, work_tree=effective)
            key = (
                assessed.git_dir,
                assessed.work_tree,
                shape.kind,
                shape.branch,
                shape.start,
            )
            if key not in cache:
                cache[key] = _checkout_guardrails_pass(assessed, shape)
            if not cache[key]:
                return False
        return True

    def _resolve_dangerous_command_permission(  # noqa: PLR0911, PLR0912
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

        The modeled network clients (v0.5.9 P2-a) classify here too: a
        curl/wget invocation whose destination is not literal loopback (or
        whose recognizer cannot parse it — fail-closed, never loopback) or
        whose destination-changing behavior gate fired resolves ask-class as
        the network.egress operation, the same grantable class as git push,
        so the runtime approval flow (dialog in interactive main sessions,
        denial with remediation in headless and child sessions) composes
        unchanged.
        """
        authorities = self._authorities()
        missing: dict[str, RepositoryScope] = {}
        # A2: the whole-call eligibility gate for the clean-tree checkout
        # exemption, evaluated over the full expanded call at the top of the
        # flow. The exemption applies only when the entire composed shell
        # command is exactly one eligible git checkout invocation.
        checkout_exemption = _whole_call_checkout_exemption(args.command, expanded)
        checkout_state_cache: dict[
            tuple[Path, Path | None, str, str | None, str | None], bool
        ] = {}
        for entry, tokens, possible_cwds, cwd_unknown in _scoped_guardrail_cwds(
            expanded, self.cwd
        ):
            if entry.access is not None or not tokens:
                continue  # Synthetic redirect operands are not executable commands.
            egress = classify_network_egress(
                tokens,
                self._injected_env,
                command_env=dict(entry.assignments) if entry.assignments else None,
            )
            if egress.kind == "ask":
                if cwd_unknown:
                    return _denied(
                        Diagnostic(
                            "dangerous_guard",
                            detail=(
                                "the working directory at this network client "
                                "is not statically known, so its repository "
                                f"scope cannot be located ({egress.reason})"
                            ),
                            command_part=entry.text,
                        )
                    )
                egress_scopes: set[RepositoryScope] = set()
                for cwd in possible_cwds:
                    scope = resolve_repository_scope(cwd, GitRepositoryRelocation())
                    if scope is None:
                        return _denied(
                            Diagnostic(
                                "dangerous_guard",
                                detail=(
                                    "the repository scope for this network "
                                    f"client cannot be located ({egress.reason})"
                                ),
                                command_part=entry.text,
                            )
                        )
                    egress_scopes.add(scope)
                covered = all(
                    any(
                        "network.egress" in authority.operations
                        and authority.scope == scope
                        for authority in authorities
                    )
                    for scope in egress_scopes
                )
                if not covered and "network.egress" not in missing:
                    missing["network.egress"] = min(egress_scopes, key=_scope_sort_key)
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
            scoped_candidates: list[tuple[Path, RepositoryScope]] = []
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
                scoped_candidates.append((cwd, scope))
            # A2: the state-aware clean-tree checkout exemption, after
            # candidate-scope resolution and before per-operation coverage.
            # The whole-call gate above pinned the invocation's shape; every
            # candidate repository scope must now pass the guardrails, or
            # the call keeps the existing approval class.
            if (
                checkout_exemption is not None
                and entry is checkout_exemption.entry
                and self._checkout_exemption_eligible(
                    checkout_exemption, scoped_candidates, checkout_state_cache
                )
            ):
                continue
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
                command_assignments=analysis.command_assignments,
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


def _union_write_surfaces(
    left: frozenset[Path] | None, right: frozenset[Path] | None
) -> frozenset[Path] | None:
    """The union of two write surfaces; None (unknown) poisons the union."""
    if left is None or right is None:
        return None
    return left | right


#: The write-root token: a collected value naming the directory tree a
#: command writes into whose members the command names only implicitly (an
#: extraction's member names, a patch's target paths). It is never a literal
#: operand token, so it bypasses the candidates filter in _write_candidates
#: and resolves against the tracked cwds like any relative operand.
_CWD_ROOT_WRITE = "."


def _unzip_extraction_root(tokens: list[str]) -> tuple[str, ...]:
    """The extraction root of one unzip call: -d's value, else the cwd."""
    parsed = _scan_options(tokens[1:], _WRITE_OPERAND_TABLES["unzip"])
    if parsed.values:
        # -d names the extraction destination; its value is collected
        # through the write-operand table.
        return ()
    if _has_short(
        parsed, frozenset("lptvzZh"), _WRITE_OPERAND_TABLES["unzip"]
    ) or _has_long(parsed, "--help", "--version"):
        # The listing, test, pipe, and comment modes write nothing outside
        # -d: the archive is a read and the members never reach the disk.
        return ()
    # The archive operand must be present: with no archive the call prompts
    # for it and, this harness's stdin closed, never extracts.
    return (_CWD_ROOT_WRITE,) if parsed.positionals else ()


def _patch_target_root(tokens: list[str]) -> tuple[str, ...]:
    """The directory tree one patch call writes: -o/-d's value, else the cwd."""
    parsed = _scan_options(tokens[1:], _WRITE_OPERAND_TABLES["patch"])
    if parsed.values_for("-o", "--output") or parsed.values_for("-d", "--directory"):
        # The -o output, -d directory, -r rejects, and -B/-Y backup names
        # are collected through the write-operand table.
        return ()
    if _has_long(parsed, "--help", "--version"):
        return ()
    # Without -o the patched result replaces every target file the diff
    # names in place, under the working directory (the -d value when the
    # call relocates) — the directory root covers them all.
    return (_CWD_ROOT_WRITE,)


def _mutating_destination_writes(  # noqa: PLR0911
    command: str, tokens: list[str], policy: ShellCommandPolicy
) -> tuple[str, ...]:
    """The destination operands one mutating command writes itself.

    cp/mv/install write their final positional — every earlier positional
    is a source — unless -t/--target-directory names the destination
    directory, in which case every positional is a source and the option
    value (already collected through the write-operand table) is the write;
    install's -D only creates the destination's leading components, so the
    final positional stays the write there too. The destination-position
    table scans the command's full value-taking option surface, so a
    value-taking option spelled AFTER the operands (cp's -S .bak,
    install's -m 644) consumes its value and the true final positional
    stays the destination. objcopy's written output is the cp-destination
    analog: the final positional is the written output, and a lone
    positional is the in-place-rewritten input. zip's first positional is
    the archive every invocation creates, replaces, or rewrites (the tar
    -f treatment: write-classified either way). 7z's archive operand
    follows the command letter, and only the archive-rewriting commands
    (a/d/u/rn) name it as a write — the extracting, listing, and testing
    ones leave it unchanged. dd parses no getopt options: its operands
    are native ``name=value`` assignments, and of= names the written
    output file (if= names the read input and stays uncollected).
    sed/gsed rewrite every input file in place under -i/--in-place;
    without it the inputs are reads and stay uncollected (the w/W
    script targets are collected through the policy's write-path
    values). unzip and patch write member/target names no operand
    names: their extraction/patching root is -d/-o's value when given
    (collected through the write-operand table) and the cwd tree root
    otherwise. These operands are MUTATE-classified positionals the
    path layer scopes as a whole (the cwd-root token resolves like any
    relative operand), so collecting them changes no permission
    decision: it is scan-scope coverage only, closing the copy-channel
    gap — a file written through cp/mv/install/dd/sed -i, an archive
    built by zip/7z, an unzip extraction, or a patch application now
    reaches the submission secret scan exactly like a redirect or tee
    target does.
    """
    if command in _DESTINATION_POSITION_TABLES:
        parsed = _scan_options(tokens[1:], _DESTINATION_POSITION_TABLES[command])
        if not parsed.positionals:
            return ()
        if command == "zip":
            return (parsed.positionals[0],)
        if command in {"cp", "mv", "install"} and parsed.values_for(
            "-t", "--target-directory"
        ):
            return ()
        return (parsed.positionals[-1],)
    if command in {"7z", "7za", "7zr"}:
        parsed = _scan_options(tokens[1:], _SEVENZIP_TABLE)
        if (
            parsed.positionals
            and parsed.positionals[0] in _SEVENZIP_ARCHIVE_WRITING_COMMANDS
        ):
            return (parsed.positionals[1],) if len(parsed.positionals) > 1 else ()
        return ()
    if command == "dd":
        return tuple(
            token[3:] for token in tokens[1:] if token.startswith("of=") and token[3:]
        )
    if command == "unzip":
        return _unzip_extraction_root(tokens)
    if command == "patch":
        return _patch_target_root(tokens)
    if command == "git" and git_apply_cwd_root_write(tokens):
        # A plain git apply patches the paths the diff names, relative to
        # the current directory: the cwd tree root is the write root, and
        # the git write-operand resolution below resolves it against the
        # directories git's own -C sequence relocates the tracked cwds to.
        return (_CWD_ROOT_WRITE,)
    if command in {"sed", "gsed"} and policy.mutates_files:
        return policy.positional_values or ()
    return ()


def _write_candidates(
    entry: _GuardrailPart, tokens: list[str], command: str
) -> tuple[str, ...]:
    """The path candidates of one expanded part the policy layer classifies as writes.

    The write surface is exactly the resolver's own classification, never a
    re-derivation from the command text: every operand of a WRITE-classified
    part (a synthetic redirect operand, or a tee/mkdir/touch command), every
    ``write_path_values`` operand (git's raw writes, a modeled network
    client's -o/-O/--rejected-log-style write options), and — for the
    commands whose whole policy option surface is write operands (cp/mv/
    install/ln -t, patch, tar, the compilers' -o/--output, ...) — every
    ``option_path_values`` operand. The remaining MUTATE-classified
    positionals are mixed-source/destination surface the policy layer does
    not decompose, so they stay uncollected — except the destinations the
    commands write themselves (a cp/mv/install/objcopy final positional, a
    zip archive, a 7z archive operand, a dd of= output, a sed -i input
    file) and the cwd-tree roots (an unzip extraction, a patch application,
    a plain git apply), which the submission scan must see; collecting them
    charges the session for no file a command merely read or touched.
    """
    policy = analyze_shell_command_policy(tokens)
    candidates = tuple(
        token
        for token in path_candidates(
            tokens,
            inspect_positional_paths=command in _PATH_COMMANDS,
            inspect_executable=entry.inspect_executable,
        )
        if token != "<redirect>"
    )
    if entry.access == PathAccess.WRITE or _operand_access(command) == PathAccess.WRITE:
        return candidates
    writes = [token for token in candidates if token in policy.write_path_values]
    if command in _WRITE_OPERAND_TABLES or command in _COMPILER_OUTPUT_COMMANDS:
        writes.extend(
            value for value in policy.option_path_values if value in candidates
        )
    writes.extend(
        value
        for value in _mutating_destination_writes(command, tokens, policy)
        if value in candidates or value == _CWD_ROOT_WRITE
    )
    return tuple(dict.fromkeys(writes))


def _git_write_values(tokens: list[str]) -> frozenset[str]:
    """The write-path operands one git invocation's operand resolution covers.

    The policy's write-path values (a writing config --file, an archive
    --output, an apply --directory/--output, an init/worktree/submodule
    destination) plus the derived cwd-tree root of a plain git apply: the
    cwd-root token resolves against the directories git's own -C sequence
    relocates the tracked cwds to, exactly like every git write operand.
    """
    values = frozenset(analyze_shell_command_policy(tokens).write_path_values)
    if git_apply_cwd_root_write(tokens):
        return frozenset({*values, _CWD_ROOT_WRITE})
    return values


# The checkout/switch subcommands whose mutation scope is the work tree they
# run in rather than the individual paths they name: a checkout/switch
# rewrites tracked work-tree content in bulk, so the resolved work-tree root
# is the mutation-scope footprint the permission layer's write-footprint
# collection and the completion audit's touched-path comparisons consume.
# It is a PERMISSION footprint only — never a submission-scan path: the
# submission scan walks collected paths recursively with a bounded budget,
# and a work-tree root would scan the whole repository and burn the budget.
_GIT_MUTATION_ROOT_SUBCOMMANDS = frozenset({"checkout", "switch"})


def _git_mutation_root_subcommand(tokens: list[str]) -> str | None:
    """The subcommand when one git invocation's mutation scope is its work tree.

    Returns the subcommand name for checkout/switch (the canonical form's
    subcommand, so recognized global options and exec-path spellings do not
    hide it), or None for every other invocation — other git subcommands,
    unmodeled or unrecognized forms, and non-git commands.
    """
    canonical = canonical_git_command(tokens)
    if canonical is None or len(canonical) <= _GIT_SUBCOMMAND_INDEX:
        return None
    subcommand = canonical[1]
    return subcommand if subcommand in _GIT_MUTATION_ROOT_SUBCOMMANDS else None


def _write_operand_unresolvable(token: str, cwd_unknown: bool) -> bool:
    """Whether the path layer would deny this write operand instead of resolving it.

    A glob-bearing write operand fails closed at the path-glob rule (only
    read-context globs are modeled), and a relative operand under an unknown
    cwd fails closed at the cwd rule; either way the command never runs, so
    the run-phase extraction reports an unknown footprint rather than a
    partial path set.
    """
    if cwd_unknown and not Path(token).is_absolute() and not token.startswith("~"):
        return True
    return (
        isinstance(token, PatternGlobCandidate)
        or any(character in token for character in "*?")
        or _has_active_bracket_glob(token.encode())
    )


def shell_command_write_paths(
    command: str, *, cwd: Path, workspace: Workspace, authority: PathAuthority
) -> tuple[list[str], bool]:
    """The resolved paths a shell command's write-classified operands name.

    Returns ``(paths, unknown)``: the command's write surface, resolved
    exactly as the permission path layer resolves every operand (occurrence
    position, tracked cwd, git's own -C relocation), and whether that surface
    could not be established. The extraction reuses the resolver's analysis —
    segmentation, expansion, redirect synthesis, and the path layer's write
    classification — never a pattern over the command text.

    A command the resolver denies never runs and produces no result, so its
    surface never reaches a submission; ``unknown`` is the fail-closed
    outcome for a run-phase call whose analysis cannot be reproduced. The
    surface is established exactly when the syntax analysis fully walked the
    command with no findings: any approval reason — an unmodeled construct,
    a non-literal redirection target, a failed analysis — is a hard denial
    the resolver never clears (a grant cannot lift an analysis-level
    reason), so the command never runs, and the extraction refuses to
    certify a partial path set. The submission gate treats an unknown
    footprint as unscannable content.
    """
    try:
        scratchpad_roots = effective_scratchpad_roots(authority)
        workspace_roots = frozenset(workspace.authorized_roots)
        analysis = _analyze_guardrail_source(
            command, scratchpad_roots=scratchpad_roots, workspace_roots=workspace_roots
        )
        if analysis.approval_reasons:
            return [], True
        command_parts = list(analysis.command_parts)
        expanded, _inner_redirects, inner_errors = _expand_guardrail_parts(
            command_parts,
            scratchpad_roots=scratchpad_roots,
            redirect_operands=analysis.redirect_operands,
            workspace_roots=workspace_roots,
            command_assignments=analysis.command_assignments,
        )
        if inner_errors:
            return [], True
        if any(
            _split_command_tokens(part)
            and _split_command_tokens(part)[0] in {"eval", "exec"}
            for part in (entry.text for entry in expanded)
        ):
            return [], True
        collected: dict[str, None] = {}
        for entry, tokens, possible_cwds, cwd_unknown in _scoped_guardrail_cwds(
            expanded, cwd
        ):
            if not tokens:
                continue
            part_command = _part_command(tokens)
            git_write_values = frozenset()
            git_write_cwds: frozenset[Path] | None = None
            if part_command == "git":
                git_write_values = _git_write_values(tokens)
                if git_write_values:
                    relocated = git_cwd_relocations(tokens, possible_cwds)
                    if relocated is not None:
                        git_write_cwds = frozenset(relocated)
            for token in _write_candidates(entry, tokens, part_command):
                if _write_operand_unresolvable(token, cwd_unknown):
                    return [], True
                for operand_cwd in _operand_resolution_cwds(
                    token,
                    tokens,
                    possible_cwds,
                    git_write_values=git_write_values,
                    git_write_cwds=git_write_cwds,
                ):
                    collected[str(resolve_tool_path(token, operand_cwd))] = None
        return list(collected), False
    except Exception:
        # Fail closed: an extraction that cannot complete leaves the write
        # surface unknown rather than guessing a partial path set.
        return [], True


def shell_command_mutation_roots(  # noqa: PLR0911
    command: str, *, cwd: Path, workspace: Workspace, authority: PathAuthority
) -> tuple[list[str], bool]:
    """The resolved work-tree roots one shell command's checkout/switch
    invocations mutate, as a distinct mutation-roots footprint.

    Returns ``(roots, unknown)``: the resolved work-tree root of every
    expanded part that is a checkout/switch invocation, resolved exactly as
    the permission layer resolves the invocation's own scope (per candidate
    cwd, git's own -C relocation, --work-tree, the repository's
    core.worktree), and whether that footprint could not be established.
    This is the mutation-scope footprint the permission write-footprint
    collection and the completion audit's touched-path comparisons consume —
    it never joins the submission secret-scan path list: the scan walks its
    collected paths recursively with a bounded budget, and a work-tree root
    would scan the whole repository and burn the budget, blocking publication.

    The extraction reuses the resolver's analysis — segmentation, expansion,
    and the tracked candidate cwds — never a pattern over the command text.
    A command the resolver denies never runs, so its footprint never reaches
    a submission; ``unknown`` is the fail-closed outcome for a run-phase call
    whose analysis cannot be reproduced (an approval reason is a hard denial
    the resolver never clears), for a checkout/switch whose work-tree root
    cannot be resolved (an unresolvable repository, an unreadable config, a
    GIT_-prefixed environment assignment that relocates the invocation
    outside the modeled resolution), and for a cwd the analysis cannot pin.
    An unresolvable state is an unknown footprint, never a partial path set.
    """
    try:
        scratchpad_roots = effective_scratchpad_roots(authority)
        workspace_roots = frozenset(workspace.authorized_roots)
        analysis = _analyze_guardrail_source(
            command, scratchpad_roots=scratchpad_roots, workspace_roots=workspace_roots
        )
        if analysis.approval_reasons:
            return [], True
        command_parts = list(analysis.command_parts)
        expanded, _inner_redirects, inner_errors = _expand_guardrail_parts(
            command_parts,
            scratchpad_roots=scratchpad_roots,
            redirect_operands=analysis.redirect_operands,
            workspace_roots=workspace_roots,
            command_assignments=analysis.command_assignments,
        )
        if inner_errors:
            return [], True
        if any(
            _split_command_tokens(part)
            and _split_command_tokens(part)[0] in {"eval", "exec"}
            for part in (entry.text for entry in expanded)
        ):
            return [], True
        collected: dict[str, None] = {}
        for entry, tokens, possible_cwds, cwd_unknown in _scoped_guardrail_cwds(
            expanded, cwd
        ):
            if not tokens:
                continue
            if _git_mutation_root_subcommand(tokens) is None:
                continue
            if cwd_unknown or not possible_cwds:
                return [], True
            if any(name.startswith("GIT_") for name, _value in entry.assignments):
                # A GIT_-prefixed assignment relocates the invocation outside
                # the modeled resolution (GIT_DIR/GIT_WORK_TREE): the root the
                # invocation would mutate cannot be established.
                return [], True
            for possible_cwd in possible_cwds:
                root = resolve_git_work_tree(tokens, possible_cwd)
                if root is None:
                    return [], True
                collected[str(root)] = None
        return list(collected), False
    except Exception:
        # Fail closed: an extraction that cannot complete leaves the
        # mutation scope unknown rather than guessing a partial root set.
        return [], True


# --- A2: the fail-closed clean-tree checkout exemption ------------------------


#: The flag injected into the executed string of an exempt checkout: an
#: ignored-file collision fails closed (checkout refuses) instead of
#: silently clobbering local content.
_NO_OVERWRITE_IGNORE = "--no-overwrite-ignore"


@dataclass(frozen=True)
class _CheckoutExemptionShape:
    """One exempt checkout invocation's modeled operands (A2).

    ``switch`` is ``git checkout <branch>``; ``create`` is
    ``git checkout -b <new> [<start>]``.
    """

    kind: str
    branch: str
    start: str | None


@dataclass(frozen=True)
class _CheckoutExemption:
    """One whole-call-eligible exempt checkout invocation (A2)."""

    entry: _GuardrailPart
    shape: _CheckoutExemptionShape
    checkout_token_index: int


def _shell_token_spans(text: str) -> list[tuple[str, int, int]] | None:
    """The shell words of *text* with their source spans, or None.

    Mirrors ``split_shell_tokens`` exactly (POSIX shlex, whitespace
    splitting, no comment handling) while also reporting where each word
    begins and ends in the source, so a caller can rewrite the source string
    without re-deriving any word's quoting. Malformed quoting returns None,
    so callers fail closed.
    """
    stream = StringIO(text)
    lexer = shlex.shlex(stream, posix=True)
    lexer.whitespace_split = True
    lexer.commenters = ""
    spans: list[tuple[str, int, int]] = []
    try:
        while True:
            start = stream.tell()
            token = lexer.get_token()
            if token is None or token == lexer.eof:
                return spans
            spans.append((token, start, stream.tell()))
    except ValueError:
        return None


def _checkout_exemption_shape(  # noqa: PLR0911
    tokens: list[str],
) -> tuple[_CheckoutExemptionShape, int] | None:
    """The exempt checkout shape of one git invocation, with the subcommand's
    token index, or None when the invocation is not an exempt shape.

    Exempt only: ``git checkout <branch>`` — no options, exactly one
    operand — and ``git checkout -b <new> [<start>]`` — the ``-b`` option
    alone, at most one further operand. Every excluded shape stays gated:
    pathspecs (a second operand or an end-of-options marker),
    ``-f``/``--force``/``--discard-changes``/``--merge``,
    ``--orphan``/``--conflict``/``--pathspec-from-file``/
    ``--recurse-submodules``, ``-B``, and any other option. Global options
    are restricted to the recognized grammar that neither overrides
    configuration nor relocates the repository: an inline
    ``-c``/``--config-env`` (which can alter the actual invocation's
    behavior, e.g. core.hooksPath) and repository relocation
    (``-C``/``--git-dir``/``--work-tree``) both fail closed.
    """
    if not tokens or tokens[0] != "git":
        # A bare git executable only: no path, no alias form.
        return None
    globals_scan = _scan_git_globals(tokens[1:])
    if globals_scan.error is not None or globals_scan.end_of_options:
        return None
    if globals_scan.relocation != GitRepositoryRelocation():
        return None
    args = tokens[1:]
    if globals_scan.index >= len(args) or args[globals_scan.index] != "checkout":
        return None
    scan = _scan_classified_git_options(
        args[globals_scan.index + 1 :], _GIT_CHECKOUT_TABLE, strict=False
    )
    if scan.error is not None or scan.end_of_options:
        return None
    subcommand_index = 1 + globals_scan.index
    if not scan.options and not scan.values and len(scan.operands) == 1:
        return (
            _CheckoutExemptionShape("switch", scan.operands[0], None),
            subcommand_index,
        )
    if (
        scan.options == ("-b",)
        and len(scan.values) == 1
        and scan.values[0][0] == "-b"
        and len(scan.operands) <= 1
    ):
        return (
            _CheckoutExemptionShape(
                "create", scan.values[0][1], scan.operands[0] if scan.operands else None
            ),
            subcommand_index,
        )
    return None


def _whole_call_checkout_exemption(
    command: str, expanded: list[_GuardrailPart]
) -> _CheckoutExemption | None:
    """The exempt checkout invocation when the whole call is exactly one.

    The exemption applies only when the entire composed shell command is
    exactly one eligible git invocation: a single expanded part carrying no
    environment assignments and no enclosing scope, whose own word sequence
    is the whole raw command (no wrapper, chained, piped, or redirected
    statement around it), a bare ``git`` executable, recognized global
    options that neither override configuration nor relocate the
    repository, and the checkout subcommand in an exempt shape.
    Recognized global options that neither override configuration nor
    relocate the repository (``--no-pager`` and equivalents) do not
    break eligibility. Any ineligible shape fails closed to the existing
    approval class.
    """
    if len(expanded) != 1:
        return None
    entry = expanded[0]
    if entry.scope or entry.access is not None or entry.assignments:
        # A synthetic operand, a nested scope, or an environment assignment
        # is not a bare single git invocation.
        return None
    tokens = (
        list(entry.tokens)
        if entry.tokens is not None
        else _split_command_tokens(entry.text)
    )
    if not tokens or _split_command_tokens(command) != tokens:
        # The raw command's own words must be exactly this part's words: a
        # wrapper, assignment prefix, subshell, or redirect around the git
        # invocation breaks eligibility for the whole call.
        return None
    shape = _checkout_exemption_shape(tokens)
    if shape is None:
        return None
    return _CheckoutExemption(entry, shape[0], shape[1])


def _checkout_guardrails_pass(
    state: RepositoryState, shape: _CheckoutExemptionShape
) -> bool:
    """The clean-tree checkout guardrails for one repository scope (A2).

    The exemption holds only on a fully clean work tree (no change, no
    staged work, no untracked file), with no in-progress operation marker
    and no repository configuration that selects a program the invocation
    could run; the plain branch form additionally requires the target to be
    an existing LOCAL branch (a sanitized rev-parse probe), so a pathspec
    fallback, a tag, a remote-tracking name, or a commit-ish never rides
    the exemption. Every unestablishable aspect fails closed.

    The configuration check runs FIRST, before any state probe that
    spawns git: it reads the repository's config without invoking
    anything, while ``git status`` can pass work-tree contents through
    repository-configured clean/smudge process filters — a repository
    whose config selects a program is declined without the probes ever
    spawning (the fail-open ordering the deep review repaired).
    """
    if config_selects_program(state) is not False:
        return False
    if work_tree_is_clean(state) is not True:
        return False
    if in_progress_operation(state) is not False:
        return False
    if shape.kind == "switch":
        return local_branch_exists(state, shape.branch)
    return True


class CheckoutInjectionError(ToolError):
    """An exempt checkout's protected executed string could not be derived.

    The whole-call exemption was re-derived at execution time, but the
    ``--no-overwrite-ignore`` insertion could not be performed safely — or
    the re-derivation itself failed after the permission-time exemption.
    The tool call must fail BEFORE execution: running the original string
    would silently drop the ignored-collision protection the exemption
    exists to provide, so the error propagates out of the tool's run
    instead of returning the unprotected command.
    """


def _exempt_checkout_executed_string(
    command: str, exemption: _CheckoutExemption
) -> str:
    """The executed string of one eligible exempt checkout, with the flag.

    The rewrite inserts ``--no-overwrite-ignore`` immediately after the
    ``checkout`` subcommand token, by source span, so every other byte of
    the command — quoting, spacing, comments, expansion — is preserved
    exactly. Every failure mode raises CheckoutInjectionError: an eligible
    checkout whose protected string cannot be derived must fail the call
    before execution, never execute the unprotected string.
    """
    spans = _shell_token_spans(command)
    if spans is None or len(spans) <= exemption.checkout_token_index:
        raise CheckoutInjectionError(
            f"the executed string of {command!r} could not be rewritten safely"
        )
    token, start, end = spans[exemption.checkout_token_index]
    if token != "checkout" or end <= start:
        raise CheckoutInjectionError(
            f"the executed string of {command!r} could not be rewritten safely"
        )
    if command[end - 1].isspace():
        # The lexer read one whitespace terminator past the token and
        # pushed it back; the token's own span ends before it.
        end -= 1
    return f"{command[:end]} {_NO_OVERWRITE_IGNORE}{command[end:]}"


def shell_command_checkout_no_overwrite_ignore(
    command: str, *, workspace: Workspace, authority: PathAuthority
) -> str | None:
    """The command rewritten with ``--no-overwrite-ignore`` for an exempt
    checkout, or None when no injection applies (A2).

    The bash tool executes a command STRING, and ``Bash.run()`` has no access
    to the permission decision, so the exemption predicate is re-derived
    here at execution time through the resolver's whole-call eligibility
    check (the gate and the exempt shape; the permission-time state probes
    are not a transaction, and the flag is the execution-time protection).
    The original command string stays original for display and result
    surfaces; only the executed string carries the flag, so an ignored-file
    collision fails closed (checkout refuses) instead of silently clobbering
    local content.

    The None/raise distinction is the fail-closed contract: None means the
    command is NOT a whole-call-eligible exempt checkout, so the caller
    executes it unmodified; CheckoutInjectionError means the command IS an
    eligible exempt checkout whose protected string could not be safely
    derived — or the re-derivation itself failed after the permission-time
    exemption — so the call fails before execution instead of silently
    dropping the ignored-collision protection.
    """
    try:
        scratchpad_roots = effective_scratchpad_roots(authority)
        workspace_roots = frozenset(workspace.authorized_roots)
        analysis = _analyze_guardrail_source(
            command, scratchpad_roots=scratchpad_roots, workspace_roots=workspace_roots
        )
        if analysis.approval_reasons:
            return None
        command_parts = list(analysis.command_parts)
        expanded, _inner_redirects, inner_errors = _expand_guardrail_parts(
            command_parts,
            scratchpad_roots=scratchpad_roots,
            redirect_operands=analysis.redirect_operands,
            workspace_roots=workspace_roots,
            command_assignments=analysis.command_assignments,
        )
        if inner_errors:
            return None
        exemption = _whole_call_checkout_exemption(command, expanded)
        if exemption is None:
            return None
        return _exempt_checkout_executed_string(command, exemption)
    except CheckoutInjectionError:
        raise
    except Exception as exc:
        # Fail closed: a re-derivation that cannot complete after the
        # permission-time exemption leaves the protected string
        # uncertifiable — the call fails instead of executing the
        # unprotected original.
        raise CheckoutInjectionError(
            f"the executed string of {command!r} could not be rewritten safely"
        ) from exc
