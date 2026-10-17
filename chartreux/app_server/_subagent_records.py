from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from chartreux.app_server.protocol import (
    DangerousCommandGrantScopeWire,
    DangerousCommandGrantWire,
)
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.dangerous_commands import DangerousCommandAuthority, RepositoryScope
from chartreux.core.llm_models import LLMMessage
from chartreux.core.session_types import ChildSessionLink
from chartreux.core.subagents import (
    AgentAvailability,
    AgentSummary,
    RunStatus,
    RunStopReason,
    TaskResult,
)
from chartreux.core.tools.builtins._shell_command_policy import (
    GitRepositoryRelocation,
    canonical_git_command,
    classify_dangerous_git_command,
    matches_command_prefix,
    resolve_repository_scope,
)
from chartreux.core.tools.builtins._shell_permission_resolver import (
    _split_command_tokens,
)

if TYPE_CHECKING:
    from chartreux.app_server._sessions import SessionRuntime


@dataclass(eq=False, slots=True)
class _ActiveWorkSlot:
    transferred: bool = False


class _AgentState(enum.StrEnum):
    RUNNING = "running"
    FINALIZING = "finalizing"
    IDLE = "idle"
    EVICTING = "evicting"


@dataclass(frozen=True, slots=True)
class StoredRunResult:
    agent_id: str
    run_id: str
    result: TaskResult
    completed_at: float
    root_generation: int
    terminal_identity: tuple[str, str, str] | None = None


@dataclass(slots=True)
class RunRecord:
    run_id: str
    agent_id: str
    profile: str
    status: RunStatus
    completion_task: asyncio.Future[Any]
    task_summary: str | None = None
    start_count: int = 0
    result: TaskResult | None = None
    started_at: float | None = None
    completed_at: float | None = None
    terminal_identity: tuple[str, str, str] | None = None
    progress_summaries: list[str] = field(default_factory=list)
    progress_overflow: bool = False
    stop_reason: RunStopReason | None = None
    requested_stop_reason: RunStopReason | None = None
    initial_turn_id: str | None = None


@dataclass(slots=True)
class AgentRecord:
    agent_id: str
    profile: str
    session_id: str
    runtime: SessionRuntime
    root_generation: int
    idle_ttl_seconds: int | None = None
    retained: bool = False
    profile_display_label: str | None = None
    parent_session_id: str = ""
    initial_task_summary: str | None = None
    state: _AgentState = _AgentState.IDLE
    current_run: RunRecord | None = None
    run_history: list[RunRecord] = field(default_factory=list)
    idle_since: float | None = None
    last_task_summary: str | None = None
    last_run_status: RunStatus | None = None
    latest_run_id: str | None = None
    effective_model: str | None = None
    base_model: str | None = None
    active_provider: str | None = None
    effective_thinking: str | None = None
    context_tokens: int | None = None
    context_window: int | None = None
    compacting: bool = False
    compaction_entry_id: str | None = None
    stop_reason: RunStopReason | None = None
    reserved_for_replacement: bool = False
    replacement_owner: _ReplacementClaim | None = None
    work_slot: _ActiveWorkSlot | None = None
    first_admitted_at: float | None = None

    def effective_idle_ttl(self, global_ttl: int | float) -> int | float:
        return (
            self.idle_ttl_seconds if self.idle_ttl_seconds is not None else global_ttl
        )

    @property
    def availability(self) -> AgentAvailability:
        if self.state is _AgentState.RUNNING:
            return AgentAvailability.RUNNING
        if self.state is _AgentState.FINALIZING:
            return AgentAvailability.FINALIZING
        return AgentAvailability.IDLE

    @availability.setter
    def availability(self, value: AgentAvailability) -> None:
        self.state = (
            _AgentState.RUNNING
            if value is AgentAvailability.RUNNING
            else _AgentState.IDLE
        )


@dataclass(frozen=True, slots=True)
class _ReplacementClaim:
    record: AgentRecord
    run: RunRecord
    parent: SessionRuntime
    parent_identity: tuple[str, int]
    child_generation: int
    authority_revision: int
    slot: _ActiveWorkSlot


@dataclass(frozen=True, slots=True)
class _AgentTombstone:
    """Private terminal metadata kept alongside the public agent summary.

    ``reason`` is ``"closed"`` for a completed-then-closed instance (reuse
    raises :class:`AgentClosedError`) and ``"evicted"`` for a TTL/cap eviction
    (reuse raises :class:`AgentEvictedError`). ``closed_at``/``expires_at`` bound
    the metadata lifetime; ``expires_at`` is always set for a stored tombstone
    (a zero tombstone TTL stores no terminal metadata at all).
    ``profile_display_label`` is captured at admission for later naming work.
    """

    summary: AgentSummary
    child_session_id: str
    parent_identity: tuple[str, int]
    reason: str = "evicted"
    closed_at: float = 0.0
    expires_at: float | None = None
    profile_display_label: str | None = None


@dataclass(frozen=True, slots=True)
class TranscriptReadSnapshot:
    """Authorization captured for a disk-only child transcript read."""

    agent_id: str
    child_session_id: str
    child_dir: Path | None
    parent_dir: Path | None
    parent_identity: tuple[str, int]
    parent_runtime_token: int
    link_identity: tuple[str, str, str, str | None] | None

    @property
    def has_saved_transcript(self) -> bool:
        return self.child_dir is not None


@dataclass(frozen=True, slots=True)
class ResidentTranscriptReadSnapshot:
    """An authorized immutable message snapshot from a resident child runtime."""

    agent_id: str
    child_session_id: str
    parent_identity: tuple[str, int]
    parent_runtime_token: int
    record_token: int
    runtime_token: int
    messages: list[LLMMessage]


@dataclass(frozen=True, slots=True)
class StoredChildSession:
    """A child record whose runtime has not been materialized."""

    parent_runtime: SessionRuntime
    root_session_id: str
    root_generation: int
    child_dir: Path
    link: ChildSessionLink


@dataclass(frozen=True, slots=True)
class DangerousCommandGrantEntry:
    """One (operation, repository scope) authority entry of an inspection."""

    operation_id: str
    scope: RepositoryScope


@dataclass(frozen=True, slots=True)
class DangerousCommandDenylistBlock:
    """An explicit denylist pattern and the registry operations it vetoes."""

    pattern: str
    operations: tuple[str, ...]


def _grant_entry_sort_key(entry: DangerousCommandGrantEntry) -> tuple[str, str, str]:
    scope = entry.scope
    return (entry.operation_id, str(scope.git_dir), str(scope.work_tree))


def _grant_entries(
    authorities: tuple[DangerousCommandAuthority, ...],
) -> tuple[DangerousCommandGrantEntry, ...]:
    """Flatten authority snapshots into sorted (operation, scope) entries."""
    entries = {
        DangerousCommandGrantEntry(operation_id, authority.scope)
        for authority in authorities
        for operation_id in authority.operations
    }
    return tuple(sorted(entries, key=_grant_entry_sort_key))


def _grant_wire(entry: DangerousCommandGrantEntry) -> DangerousCommandGrantWire:
    scope = entry.scope
    return DangerousCommandGrantWire(
        operation_id=entry.operation_id,
        scope=DangerousCommandGrantScopeWire(
            git_dir=str(scope.git_dir),
            work_tree=str(scope.work_tree) if scope.work_tree is not None else None,
        ),
    )


def _accepted_saved_grants(
    loop: AgentLoop, project: Path
) -> tuple[DangerousCommandGrantEntry, ...]:
    """Saved user-config authority the session's accepted sources contribute.

    The union covers the session's inherited restrictions (a child accepts its
    parent's sources) and its own accepted restrictions; the same source can
    appear in both, so entries dedupe by project and operations. Only the
    registered project's grants are relevant to its inspection.
    """
    sources = (
        *loop.runtime_policy.inherited_restrictions,
        *loop.config_orchestrator.restrictions,
    )
    seen: set[tuple[str, tuple[str, ...]]] = set()
    entries: set[DangerousCommandGrantEntry] = set()
    for source in sources:
        for grant in source.dangerous_command_grants:
            key = (str(grant.project), grant.operations)
            if key in seen:
                continue
            seen.add(key)
            if str(grant.project) != str(project):
                continue
            scope = resolve_repository_scope(grant.project, GitRepositoryRelocation())
            if scope is None:
                continue
            entries.update(
                DangerousCommandGrantEntry(operation_id, scope)
                for operation_id in grant.operations
            )
    return tuple(sorted(entries, key=_grant_entry_sort_key))


# One canonical probe form per registry operation. The denylist is matched
# with the resolver's own prefix/canonical-git matching, so a probe doubles as
# the operation's representative command; tests pin each probe to exactly its
# operation through the frozen classifier.
_DANGEROUS_COMMAND_DENYLIST_PROBES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("git.push", ("git", "push")),
    ("git.push.force", ("git", "push", "--force")),
    ("git.push.delete", ("git", "push", "--delete")),
    ("git.push.mirror", ("git", "push", "--mirror")),
    ("git.checkout", ("git", "checkout")),
    ("git.restore", ("git", "restore")),
    ("git.switch.discard", ("git", "switch", "--discard-changes")),
    ("git.stash.drop", ("git", "stash", "drop")),
    ("git.stash.clear", ("git", "stash", "clear")),
    ("git.reflog.expire", ("git", "reflog", "expire")),
    ("git.reflog.delete", ("git", "reflog", "delete")),
)


def _explicit_shell_denylist(loop: AgentLoop) -> tuple[str, ...]:
    """Configured bash denylist entries, without the shipped defaults.

    Mirrors the tool manager's restriction assembly (inherited plus accepted
    sources); the merged tool config also carries built-in defaults, which are
    not explicit contributions and never mention dangerous git operations.
    """
    sources = (
        *loop.runtime_policy.inherited_restrictions,
        *loop.config_orchestrator.restrictions,
    )
    patterns: list[str] = []
    for source in sources:
        for restriction in source.tools:
            if restriction.tool_name != "bash":
                continue
            for pattern in restriction.denylist:
                if pattern not in patterns:
                    patterns.append(pattern)
    return tuple(patterns)


def _denylist_blocked_operations(pattern: str) -> tuple[str, ...]:
    """Registry operations one explicit denylist pattern absolutely vetoes.

    A pattern vetoes an operation when it prefix-matches one of the
    operation's representative probe forms (the resolver's own prefix and
    canonical-git matching), or when the pattern is itself a complete
    grantable form of the operation (``git push origin --delete``). Patterns
    that only block through an unclassifiable extension (an end-of-options
    marker, say) are not reported; the surface is informational and never
    suggests a grant overrides a veto.
    """
    pattern_tokens = _split_command_tokens(pattern)
    if not pattern_tokens:
        return ()
    direct = classify_dangerous_git_command(pattern_tokens)
    direct_operations = (
        direct.grantable.operation_ids if direct.grantable is not None else ()
    )
    blocked: list[str] = []
    for operation_id, probe in _DANGEROUS_COMMAND_DENYLIST_PROBES:
        if operation_id in direct_operations:
            blocked.append(operation_id)
            continue
        tokens = list(probe)
        canonical = canonical_git_command(tokens)
        if matches_command_prefix(tokens, pattern_tokens) or (
            canonical is not None and matches_command_prefix(canonical, pattern_tokens)
        ):
            blocked.append(operation_id)
    return tuple(blocked)


def _blocking_denylist(loop: AgentLoop) -> tuple[DangerousCommandDenylistBlock, ...]:
    """Explicit denylist entries blocking dangerous git operations.

    The denylist is an absolute veto checked before grantability, so these
    entries are reported apart from grant authority: no live or saved grant
    overrides them.
    """
    blocks: list[DangerousCommandDenylistBlock] = []
    for pattern in _explicit_shell_denylist(loop):
        operations = _denylist_blocked_operations(pattern)
        if operations:
            blocks.append(DangerousCommandDenylistBlock(pattern, operations))
    return tuple(blocks)
