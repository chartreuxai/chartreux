"""Runtime-owned dangerous-command grant stores and approval grant logic.

One agent loop owns the session grants, once permits, and refusal memory of
the dangerous-command approval lifecycle. Session grants are keyed by
operation AND RepositoryScope and live exactly as long as the loop; a once
permit authorizes exactly one invocation of one original call, including
every listed occurrence; refusal memory records the decided call identity so
a denied call returns an honest denial instead of re-prompting. The store is
the ``DangerousCommandAuthorityProvider`` port the tool manager consumes; it
mutates no config, persists nothing, and executes nothing.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
import shlex

from pydantic import BaseModel

from chartreux.core.config.types import ConfigSaveResult
from chartreux.core.dangerous_commands import (
    OPERATION_REGISTRY,
    REMOTE_EXEC_OPERATION_ID,
    DangerousCommandAuthority,
    DangerousCommandProposal,
    GitRepositoryRelocation,
    GrantChoice,
    RemoteDestination,
    RepositoryScope,
    is_grant_choice_valid,
)
from chartreux.core.env import project_env_for_cwd
from chartreux.core.tools.builtins._destination import classify_network_egress
from chartreux.core.tools.builtins._shell_command_policy import (
    classify_dangerous_git_command,
    resolve_repository_scope,
)
from chartreux.core.tools.builtins._shell_permission_resolver import (
    _analyze_guardrail_source,
    _expand_guardrail_parts,
    _scope_sort_key,
    _scoped_guardrail_cwds,
)
from chartreux.questions import (
    QuestionChoice,
    UserQuestion,
    UserQuestionRequest,
    UserQuestionResult,
)

# Frozen display contract: these labels are the parse keys for user answers.
DANGEROUS_COMMAND_DENY_LABEL = "Deny"
DANGEROUS_COMMAND_ONCE_LABEL = "Allow once"
DANGEROUS_COMMAND_SESSION_LABEL = "Allow this session"
DANGEROUS_COMMAND_PROJECT_LABEL = "Always for this project"
DANGEROUS_COMMAND_DECLINED_FEEDBACK = "user declined; do not retry this command"

# Frozen honest-scope statement for a remote-exec approval. A destination
# grant authorizes reaching a remote host; it does not contain or approve the
# payload, because the remote command runs under the remote host's policy.
REMOTE_EXEC_SCOPE_STATEMENT = (
    "Remote commands and their effects are outside chartreux's local guards — "
    "trusting the destination, not approving a contained payload."
)

_LABEL_TO_CHOICE: dict[str, GrantChoice] = {
    DANGEROUS_COMMAND_DENY_LABEL: "deny",
    DANGEROUS_COMMAND_ONCE_LABEL: "allow_once",
    DANGEROUS_COMMAND_SESSION_LABEL: "allow_session",
    DANGEROUS_COMMAND_PROJECT_LABEL: "allow_project",
}


@dataclass(frozen=True)
class DangerousCommandCallIdentity:
    """Repository-bound canonical whole-call identity for refusal memory."""

    scope: RepositoryScope
    command: tuple[str, ...]


@dataclass(frozen=True)
class DangerousCommandOncePermit:
    """Live once authority: the (operation, scope) pairs of one invocation."""

    requirements: frozenset[tuple[str, RepositoryScope]]


@dataclass(frozen=True)
class RemoteExecGrantRequirement:
    """One remote-exec authority requirement: one scope, one exact destination."""

    scope: RepositoryScope
    destination: RemoteDestination


@dataclass(frozen=True)
class RemoteExecOncePermit:
    """Live once authority for remote-exec destinations of one invocation."""

    requirements: frozenset[RemoteExecGrantRequirement]


def _authorities_for(
    requirements: frozenset[tuple[str, RepositoryScope]],
) -> tuple[DangerousCommandAuthority, ...]:
    """Project (operation, scope) pairs into immutable authority snapshots."""
    by_scope: dict[RepositoryScope, set[str]] = {}
    for operation_id, scope in requirements:
        by_scope.setdefault(scope, set()).add(operation_id)
    return tuple(
        DangerousCommandAuthority(
            operations=frozenset(operations), scope=scope, source="runtime_grant"
        )
        for scope, operations in by_scope.items()
    )


class DangerousCommandGrantStore:
    """Session grants, once permits, and refusal memory owned by one loop."""

    def __init__(self) -> None:
        self._session: dict[RepositoryScope, set[str]] = {}
        self._once: list[DangerousCommandOncePermit] = []
        self._remote_exec_session: dict[RepositoryScope, set[RemoteDestination]] = {}
        self._remote_exec_once: list[RemoteExecOncePermit] = []
        self._refusals: set[DangerousCommandCallIdentity] = set()
        self._authority_generation = 0
        self._policy_revision = 0

    @property
    def current_policy_revision(self) -> int:
        return self._policy_revision

    def bump_policy_revision(self) -> None:
        self._policy_revision += 1

    @property
    def authority_generation(self) -> int:
        """Monotonic live-authority generation, exposed read-only.

        Every store mutation — grant, revoke, refusal, once-permit
        install/destroy, invalidation — bumps it, so consumers (revocation
        endpoints, authority cache keying) can detect that the live
        authority or refusal memory changed. It never resets or decreases
        within a session.
        """
        return self._authority_generation

    def _bump_authority_generation(self) -> None:
        self._authority_generation += 1

    def dangerous_command_authorities(self) -> tuple[DangerousCommandAuthority, ...]:
        """The immutable runtime-authority snapshot for one resolution."""
        authorities: list[DangerousCommandAuthority] = []
        for scope, operations in self._session.items():
            if operations:
                authorities.append(
                    DangerousCommandAuthority(
                        operations=frozenset(operations),
                        scope=scope,
                        source="runtime_grant",
                    )
                )
        for permit in self._once:
            authorities.extend(_authorities_for(permit.requirements))
        authorities.extend(self._remote_exec_authorities())
        return tuple(authorities)

    def _remote_exec_authorities(self) -> list[DangerousCommandAuthority]:
        """Destination-keyed authority contributions from this store."""
        authorities = [
            DangerousCommandAuthority(
                operations=frozenset({REMOTE_EXEC_OPERATION_ID}),
                scope=scope,
                source="runtime_grant",
                destination=destination,
            )
            for scope, destinations in self._remote_exec_session.items()
            for destination in destinations
        ]
        authorities.extend(
            DangerousCommandAuthority(
                operations=frozenset({REMOTE_EXEC_OPERATION_ID}),
                scope=requirement.scope,
                source="runtime_grant",
                destination=requirement.destination,
            )
            for permit in self._remote_exec_once
            for requirement in permit.requirements
        )
        return authorities

    def is_refused(self, identity: DangerousCommandCallIdentity) -> bool:
        return identity in self._refusals

    def record_refusal(self, identity: DangerousCommandCallIdentity) -> None:
        if identity not in self._refusals:
            self._refusals.add(identity)
            self._bump_authority_generation()

    def grant_session(
        self, requirements: Iterable[tuple[str, RepositoryScope]]
    ) -> bool:
        """Install session authority per (operation, scope); True when changed."""
        changed = False
        for operation_id, scope in requirements:
            operations = self._session.setdefault(scope, set())
            if operation_id not in operations:
                operations.add(operation_id)
                changed = True
        if changed:
            self._bump_authority_generation()
        return changed

    def revoke_session(
        self, requirements: Iterable[tuple[str, RepositoryScope]]
    ) -> bool:
        """Remove live session authority; True when anything was removed.

        Revoking ``network.remote-exec`` at a scope drops every destination
        grant in that scope: a destination grant is authority for the one
        operation, so there is no finer live unit to revoke by scope alone.
        """
        wanted = tuple(requirements)
        if wanted:
            self.bump_policy_revision()
        changed = False
        for operation_id, scope in wanted:
            if operation_id == REMOTE_EXEC_OPERATION_ID:
                if self._remote_exec_session.pop(scope, None) is not None:
                    changed = True
            operations = self._session.get(scope)
            if operations is not None and operation_id in operations:
                operations.discard(operation_id)
                changed = True
                if not operations:
                    del self._session[scope]
        if changed:
            self._bump_authority_generation()
        return changed

    def grant_remote_exec_session(
        self, requirements: Iterable[RemoteExecGrantRequirement]
    ) -> bool:
        """Install destination authority per (scope, exact destination)."""
        changed = False
        for requirement in requirements:
            destinations = self._remote_exec_session.setdefault(
                requirement.scope, set()
            )
            if requirement.destination not in destinations:
                destinations.add(requirement.destination)
                changed = True
        if changed:
            self._bump_authority_generation()
        return changed

    def revoke_remote_exec_session(
        self, requirements: Iterable[RemoteExecGrantRequirement]
    ) -> bool:
        """Remove live destination authority; True when anything was removed."""
        wanted = tuple(requirements)
        if wanted:
            self.bump_policy_revision()
        changed = False
        for requirement in wanted:
            destinations = self._remote_exec_session.get(requirement.scope)
            if destinations is not None and requirement.destination in destinations:
                destinations.discard(requirement.destination)
                changed = True
                if not destinations:
                    del self._remote_exec_session[requirement.scope]
        if changed:
            self._bump_authority_generation()
        return changed

    def remote_exec_scopes_for_destination(
        self, destination: RemoteDestination
    ) -> tuple[RepositoryScope, ...]:
        """Every scope whose session grants currently hold *destination*."""
        return tuple(
            sorted(
                (
                    scope
                    for scope, destinations in self._remote_exec_session.items()
                    if destination in destinations
                ),
                key=_scope_sort_key,
            )
        )

    def revoke_live_authority(
        self, operation_id: str, destination: RemoteDestination | None
    ) -> bool:
        """Remove one operation's — or one destination's — live authority.

        The single revision-locked revocation entry point: the removal and
        the policy-revision advance are one operation, so live authority is
        never dropped without invalidating the admissions decided under it.
        A ``network.remote-exec`` operation at a scope drops every
        destination grant in that scope; with *destination*, only that
        destination's grants go and every pending once permit containing it
        is invalidated whole. Returns True when anything was removed.
        """
        self.bump_policy_revision()
        if destination is not None:
            changed = self.revoke_remote_exec_session(
                RemoteExecGrantRequirement(scope=scope, destination=destination)
                for scope in self.remote_exec_scopes_for_destination(destination)
            )
            self.invalidate_pending_destinations({destination})
            return changed
        requirements = {
            (operation_id, authority.scope)
            for authority in self.dangerous_command_authorities()
            if operation_id in authority.operations
        }
        changed = self.revoke_session(requirements)
        self.invalidate_pending(requirements)
        return changed

    def install_remote_exec_once(
        self, requirements: frozenset[RemoteExecGrantRequirement]
    ) -> RemoteExecOncePermit:
        """Install a once permit bound to exactly one remote-exec invocation."""
        permit = RemoteExecOncePermit(requirements=requirements)
        self._remote_exec_once.append(permit)
        self._bump_authority_generation()
        return permit

    def destroy_remote_exec_once(self, permit: RemoteExecOncePermit) -> None:
        """Destroy a remote-exec once permit, used or unused."""
        try:
            self._remote_exec_once.remove(permit)
        except ValueError:
            return  # Already destroyed: a no-op, never an error.
        self._bump_authority_generation()

    def install_once(
        self, requirements: frozenset[tuple[str, RepositoryScope]]
    ) -> DangerousCommandOncePermit:
        """Install a once permit bound to exactly one invocation."""
        permit = DangerousCommandOncePermit(requirements=requirements)
        self._once.append(permit)
        self._bump_authority_generation()
        return permit

    def destroy_once(self, permit: DangerousCommandOncePermit) -> None:
        """Destroy a once permit, used or unused; no authority remains."""
        try:
            self._once.remove(permit)
        except ValueError:
            return  # Already destroyed: a no-op, never an error.
        self._bump_authority_generation()

    def invalidate_pending(
        self, requirements: Iterable[tuple[str, RepositoryScope]]
    ) -> int:
        """Invalidate pending once permits matching *requirements*.

        A ``network.remote-exec`` requirement invalidates every pending
        destination permit in that scope, mirroring the scope-level
        revocation rule.
        """
        wanted = set(requirements)
        if wanted:
            self.bump_policy_revision()
        kept = [permit for permit in self._once if not (permit.requirements & wanted)]
        invalidated = len(self._once) - len(kept)
        self._once = kept
        remote_exec_scopes = {
            scope
            for operation_id, scope in wanted
            if operation_id == REMOTE_EXEC_OPERATION_ID
        }
        if remote_exec_scopes:
            kept_permits = [
                permit
                for permit in self._remote_exec_once
                if not any(
                    requirement.scope in remote_exec_scopes
                    for requirement in permit.requirements
                )
            ]
            invalidated += len(self._remote_exec_once) - len(kept_permits)
            self._remote_exec_once = kept_permits
        if invalidated:
            self._bump_authority_generation()
        return invalidated

    def invalidate_pending_destinations(
        self, destinations: Iterable[RemoteDestination]
    ) -> int:
        """Invalidate every pending remote-exec once permit containing a destination.

        A destination-scoped revocation drops each remote-exec once permit
        that contains the revoked destination WHOLE — the permit covers one
        invocation, so it is never trimmed to its remaining destinations —
        mirroring the whole-permit invalidation of the scope-level rule.
        """
        wanted = set(destinations)
        if wanted:
            self.bump_policy_revision()
        kept = [
            permit
            for permit in self._remote_exec_once
            if not any(
                requirement.destination in wanted for requirement in permit.requirements
            )
        ]
        invalidated = len(self._remote_exec_once) - len(kept)
        self._remote_exec_once = kept
        if invalidated:
            self._bump_authority_generation()
        return invalidated

    def clear_session_state(self) -> bool:
        """Drop every session-scoped record: grants, once permits, refusals.

        An in-place conversation or session reset — a resume rebind, a
        history clear, a rewind fork — starts a new session, and session
        grants expire with the old one: they are never auto-restored
        across the boundary. The store object itself survives (the tool
        manager's provider wiring references it); only its records go.
        Returns True when anything was cleared. The authority generation
        always bumps, so consumers keying on it detect the boundary even
        when the store was already empty.
        """
        cleared = bool(
            self._session
            or self._once
            or self._remote_exec_session
            or self._remote_exec_once
            or self._refusals
        )
        self._session = {}
        self._once = []
        self._remote_exec_session = {}
        self._remote_exec_once = []
        self._refusals = set()
        self._bump_authority_generation()
        self.bump_policy_revision()
        return cleared


def _token_needs_quotes(token: str) -> bool:
    return any(character.isspace() or character in "'\"" for character in token)


def _command_source(tokens: tuple[str, ...]) -> str:
    """Rebuild a shell source string from the proposal's command tokens.

    The proposal carries the raw call's shlex tokens, not the raw source;
    re-deriving the approval's coverage needs a string the guardrail
    analysis can parse again. Tokens that would otherwise split apart
    (embedded whitespace or quotes) are quoted; operator tokens stay
    verbatim so the command structure survives the round trip.
    """
    return " ".join(
        shlex.quote(token) if _token_needs_quotes(token) else token for token in tokens
    )


def _proposal_repositories(
    proposal: DangerousCommandProposal,
    *,
    scratchpad_roots: frozenset[Path] = frozenset(),
) -> tuple[Path, ...]:
    """Every repository an approval of *proposal* grants authority in.

    The proposal's own scope is representative only: an approval installs
    grants at every (operation, scope) the original call touches, so the
    question shows every repository involved. The coverage is re-derived
    from the proposal's command; the representative scope always counts,
    and a coverage that can no longer be derived falls back to it alone.
    The caller's live scratch roots are threaded into that re-derivation
    so both derivation call sites mirror the resolver's own expansion.
    """
    scopes = {proposal.scope}
    requirements = derive_call_requirements(
        _command_source(proposal.command),
        proposal.project,
        proposal.operations,
        scratchpad_roots=scratchpad_roots,
    )
    if requirements:
        scopes.update(scope for _, scope in requirements)
    return tuple(
        scope.work_tree or scope.git_dir
        for scope in sorted(scopes, key=_scope_sort_key)
    )


def dangerous_command_request(
    proposal: DangerousCommandProposal,
    *,
    scratchpad_roots: frozenset[Path] = frozenset(),
) -> UserQuestionRequest:
    """Build the mid-turn question asking the user to approve *proposal*.

    *scratchpad_roots* is the caller's live scratch authority, threaded into
    the coverage re-derivation exactly as the approval-time recheck threads
    it.
    """
    operations = tuple(
        OPERATION_REGISTRY[operation_id]
        for operation_id in proposal.operations
        if operation_id in OPERATION_REGISTRY
    )
    # A persistent choice is offered only when every required operation is
    # persistable; a mixed or consequential call never offers it, and a
    # forged "always" answer is rejected at parse time.
    persistable_offer = bool(operations) and all(
        operation.persistable for operation in operations
    )
    repositories = _proposal_repositories(proposal, scratchpad_roots=scratchpad_roots)
    if len(repositories) == 1:
        repository_line = f"Repository: {repositories[0]}\n"
        session_scope = "in this repository"
    else:
        # An approval grants authority at every repository the call touches,
        # so the question names them all.
        repository_line = f"Repositories: {', '.join(str(repository) for repository in repositories)}\n"
        session_scope = "in these repositories"
    if proposal.destination is not None:
        # A remote-exec approval is a destination grant, and the wording must
        # stay honest: the grant authorizes reaching the host, not containing
        # the remote payload.
        destination = proposal.destination
        question = (
            "Allow this remote command?\n\n"
            f"{shlex.join(proposal.command)}\n\n"
            f"Project: {proposal.project}\n"
            f"{repository_line}"
            f"Destination: {destination}\n"
            f"Operations: {', '.join(proposal.operations)}\n"
            f"Consequence: {proposal.consequence}\n\n"
            f"{REMOTE_EXEC_SCOPE_STATEMENT}"
        )
        options = [
            QuestionChoice(
                label=DANGEROUS_COMMAND_DENY_LABEL,
                description="Skip this command; it does not run",
            ),
            QuestionChoice(
                label=DANGEROUS_COMMAND_ONCE_LABEL,
                description="Run this exact command one time",
            ),
            QuestionChoice(
                label=DANGEROUS_COMMAND_SESSION_LABEL,
                description=(
                    f"Allow remote commands to {destination} {session_scope} "
                    "for this session"
                ),
            ),
            QuestionChoice(
                label=DANGEROUS_COMMAND_PROJECT_LABEL,
                description=(
                    "Allow for this session and save the destination to your "
                    "user config for this project"
                ),
            ),
        ]
        return UserQuestionRequest(
            questions=[
                UserQuestion(
                    question=question,
                    header="Remote command",
                    options=options,
                    hide_other=True,
                )
            ],
            footer_note=(
                "Session grants stay in memory; only 'Always for this project' "
                "writes the destination to your user config."
            ),
        )
    # The noun follows the operations: git calls keep the historical
    # wording; a non-git ask-class call (network.egress, v0.5.9 P2-a) is not
    # a git command and is not described as one.
    noun = (
        "dangerous git command"
        if all(operation.startswith("git.") for operation in proposal.operations)
        else "dangerous command"
    )
    question = (
        f"Allow this {noun}?\n\n{shlex.join(proposal.command)}\n\n"
        f"Project: {proposal.project}\n"
        f"{repository_line}"
        f"Operations: {', '.join(proposal.operations)}\n"
        f"Consequence: {proposal.consequence}"
    )
    if persistable_offer:
        question += (
            " This grants permission authority for the operation, not a "
            "standing intent to publish."
        )
    options = [
        QuestionChoice(
            label=DANGEROUS_COMMAND_DENY_LABEL,
            description="Skip this command; it does not run",
        ),
        QuestionChoice(
            label=DANGEROUS_COMMAND_ONCE_LABEL,
            description="Run this exact command one time",
        ),
        QuestionChoice(
            label=DANGEROUS_COMMAND_SESSION_LABEL,
            description=(f"Allow these operations {session_scope} for this session"),
        ),
    ]
    if persistable_offer:
        options.append(
            QuestionChoice(
                label=DANGEROUS_COMMAND_PROJECT_LABEL,
                description=(
                    "Allow for this session and save it to your user config "
                    "for this project"
                ),
            )
        )
    footer_note = (
        "Session grants stay in memory; only 'Always for this project' writes "
        "to your user config."
        if persistable_offer
        else (
            "Session grants stay in memory; consequential operations are never "
            "saved to your config."
        )
    )
    return UserQuestionRequest(
        questions=[
            UserQuestion(
                question=question,
                header="Dangerous command",
                options=options,
                hide_other=True,
            )
        ],
        footer_note=footer_note,
    )


def parse_dangerous_command_result(
    result: BaseModel, operations: tuple[str, ...]
) -> GrantChoice:
    """Map a user answer to a grant choice; anything else fails closed to deny.

    A choice invalid for any required operation (a forged consequential
    "always", a free-text answer, an unknown label) authorizes nothing. The
    parser requires the typed disposition ``answered`` and never inspects
    declared unattended defaults (plan U/OD1): an unanswered expiry result
    can never become a grant.
    """
    if (
        isinstance(result, UserQuestionResult)
        and result.disposition == "answered"
        and not result.cancelled
    ):
        for answer in result.answers:
            choice = _LABEL_TO_CHOICE.get(answer.answer)
            if choice is not None and all(
                is_grant_choice_valid(operation_id, choice)
                for operation_id in operations
            ):
                return choice
    return "deny"


def derive_call_requirements(
    command: str,
    cwd: Path,
    operations: tuple[str, ...],
    *,
    scratchpad_roots: frozenset[Path] = frozenset(),
) -> frozenset[tuple[str, RepositoryScope]] | None:
    """Re-derive every (operation, scope) the original call requires.

    The proposal's scope is representative only: a whole call can span
    several repositories, and an approval must cover every required operation
    at every scope it can operate on. The traversal mirrors the resolver's own
    expansion exactly, on the same raw command and working directory. Returns
    None when the call can no longer be derived, so callers fail closed.

    The scratchpad-roots parameter is the caller's live scratch authority,
    threaded by both call sites so this traversal sees the same roots the
    resolver did. The roots are uniformity-only for now: this re-derivation
    never reads approval_reasons — the only roots-sensitive analysis output
    — so they cannot change the derived requirements. Safety rests on the
    resolver's path layer, which independently gates every literal redirect
    target as a WRITE against the caller's own authority. A future refactor
    reading approval reasons here already has the roots threaded.
    """
    wanted = set(operations)
    try:
        analysis = _analyze_guardrail_source(command, scratchpad_roots=scratchpad_roots)
        expanded, _, _ = _expand_guardrail_parts(
            list(analysis.command_parts),
            scratchpad_roots=scratchpad_roots,
            redirect_operands=analysis.redirect_operands,
            command_assignments=analysis.command_assignments,
        )
        requirements: set[tuple[str, RepositoryScope]] = set()
        for entry, tokens, possible_cwds, cwd_unknown in _scoped_guardrail_cwds(
            expanded, cwd
        ):
            if entry.access is not None or not tokens:
                continue  # Synthetic redirect operands are not executable commands.
            # v0.5.9 P2-a: the modeled network clients re-derive their
            # network.egress requirement exactly as the resolver classified
            # it, so an approved once permit or session grant covers the
            # egress call the proposal named. The injected environment and
            # the command's own assignments are threaded too (v0.5.9 R1/R4):
            # a non-loopback proxy in either is part of what makes the
            # egress call require the operation.
            egress = classify_network_egress(
                tokens,
                project_env_for_cwd(cwd),
                command_env=dict(entry.assignments) if entry.assignments else None,
            )
            if egress.kind == "ask" and "network.egress" in wanted:
                if cwd_unknown:
                    return None
                for possible_cwd in possible_cwds:
                    scope = resolve_repository_scope(
                        possible_cwd, GitRepositoryRelocation()
                    )
                    if scope is None:
                        return None
                    requirements.add(("network.egress", scope))
            classification = classify_dangerous_git_command(tokens)
            if classification.kind != "grantable" or classification.grantable is None:
                continue
            if cwd_unknown:
                return None
            part_operations = tuple(
                operation_id
                for operation_id in classification.grantable.operation_ids
                if operation_id in wanted
            )
            if not part_operations:
                continue
            for possible_cwd in possible_cwds:
                scope = resolve_repository_scope(
                    possible_cwd, classification.grantable.relocation
                )
                if scope is None:
                    return None
                for operation_id in part_operations:
                    requirements.add((operation_id, scope))
        return frozenset(requirements)
    except Exception:
        return None


def dangerous_command_grant_save_note(
    project: Path,
    scope: RepositoryScope,
    outcomes: tuple[tuple[str, ConfigSaveResult | None], ...],
) -> str:
    """Report persistence outcomes apart from the live session grant.

    The live grant is already applied when this note is built; the note
    carries only the persistence outcome, honestly, per operation.
    """
    repository = scope.work_tree or scope.git_dir
    sentences = []
    for operation_id, result in outcomes:
        prefix = (
            f"Dangerous-command grant for {operation_id} in {repository} "
            "applies to this session"
        )
        if result is None or result.error == "write":
            sentences.append(f"{prefix} only; saving it to your user config failed.")
        elif result.error == "cancelled":
            sentences.append(
                f"{prefix}; the save to your user config was cancelled before "
                "it could be confirmed, so treat the grant as session-only."
            )
        elif result.persistence == "durability_uncertain":
            sentences.append(
                f"{prefix}; the save to your user config could not be "
                "confirmed, so treat the grant as session-only."
            )
        elif result.persistence == "saved" and result.error is None:
            sentences.append(
                f"{prefix} and is saved in your user config for project "
                f"{project}. The running session's config is unchanged; "
                "future sessions start with the saved grant."
            )
        elif result.error == "conflict":
            sentences.append(
                f"{prefix} only; saving it to your user config was rejected "
                "because the config changed since it was read. You can retry "
                "'Always for this project' to save it."
            )
        elif result.persistence == "not_saved" and result.error is None:
            sentences.append(
                f"{prefix}; this session does not save dangerous-command "
                "grants to your user config."
            )
        else:
            sentences.append(
                f"{prefix} only; saving it to your user config was rejected."
            )
    return " ".join(sentences)


def remote_exec_grant_save_note(
    project: Path,
    scope: RepositoryScope,
    destination: RemoteDestination,
    result: ConfigSaveResult | None,
) -> str:
    """Report a remote-exec destination persistence outcome honestly.

    The live destination grant is already applied when this note is built;
    the note carries only the persistence outcome.
    """
    repository = scope.work_tree or scope.git_dir
    prefix = (
        f"Remote-exec grant for {destination} in {repository} applies to this session"
    )
    if result is None or result.error == "write":
        note = f"{prefix} only; saving it to your user config failed."
    elif result.error == "cancelled":
        note = (
            f"{prefix}; the save to your user config was cancelled before it "
            "could be confirmed, so treat the grant as session-only."
        )
    elif result.persistence == "durability_uncertain":
        note = (
            f"{prefix}; the save to your user config could not be confirmed, "
            "so treat the grant as session-only."
        )
    elif result.persistence == "saved" and result.error is None:
        note = (
            f"{prefix} and is saved in your user config for project {project}. "
            "The running session's config is unchanged; future sessions start "
            "with the saved grant."
        )
    elif result.error == "conflict":
        note = (
            f"{prefix} only; saving it to your user config was rejected "
            "because the config changed since it was read. You can retry "
            "'Always for this project' to save it."
        )
    elif result.persistence == "not_saved" and result.error is None:
        note = (
            f"{prefix}; this session does not save remote-exec grants to your "
            "user config."
        )
    else:
        note = f"{prefix} only; saving it to your user config was rejected."
    return note
