"""Typed models for the dangerous-command permission feature.

The operation registry is frozen: every recognized git operation carries a
stable operation ID, a grant-choice class, a consequence summary, and whether
the operation is persistable (only ordinary publication is). The
classification models describe what the pure classifier in the shell-command
policy layer reports for one whole call; the authority and proposal types are
the immutable shapes later packages use for resolver snapshots, runtime-owned
grants, and whole-call proposals. This module owns no grant mutation, no
persistence behavior, and no execution.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal

DangerousCommandClass = Literal["publication", "consequential"]
GrantChoice = Literal["deny", "allow_once", "allow_session", "allow_project"]
AuthoritySource = Literal["protected_config", "runtime_grant"]
ClassificationKind = Literal["routine", "hard_denied", "grantable"]


@dataclass(frozen=True)
class DangerousOperation:
    """One frozen registry entry: an operation a grant can authorize."""

    operation_id: str
    command_class: DangerousCommandClass
    consequence: str
    persistable: bool


_OPERATION_ENTRIES: tuple[DangerousOperation, ...] = (
    DangerousOperation(
        operation_id="git.push",
        command_class="publication",
        consequence=(
            "Publish local commits to the remote (ordinary push: no forced, "
            "deleted, or mirrored refs)."
        ),
        persistable=True,
    ),
    DangerousOperation(
        operation_id="git.push.force",
        command_class="consequential",
        consequence="Overwrite remote ref history (forced push).",
        persistable=False,
    ),
    DangerousOperation(
        operation_id="git.push.delete",
        command_class="consequential",
        consequence="Delete refs on the remote.",
        persistable=False,
    ),
    DangerousOperation(
        operation_id="git.push.mirror",
        command_class="consequential",
        consequence="Overwrite and delete remote refs to mirror the local repository.",
        persistable=False,
    ),
    DangerousOperation(
        operation_id="git.checkout",
        command_class="consequential",
        consequence=(
            "Switch branches or check out paths; overwrites working-tree and "
            "index content and can discard uncommitted changes."
        ),
        persistable=False,
    ),
    DangerousOperation(
        operation_id="git.restore",
        command_class="consequential",
        consequence=(
            "Overwrite working-tree or index files from a tree; discards "
            "uncommitted changes to those files."
        ),
        persistable=False,
    ),
    DangerousOperation(
        operation_id="git.switch.discard",
        command_class="consequential",
        consequence="Switch branches while discarding uncommitted working-tree changes.",
        persistable=False,
    ),
    DangerousOperation(
        operation_id="git.stash.drop",
        command_class="consequential",
        consequence="Drop a stash entry; its changes leave the stash.",
        persistable=False,
    ),
    DangerousOperation(
        operation_id="git.stash.clear",
        command_class="consequential",
        consequence="Drop every stash entry; their changes leave the stash.",
        persistable=False,
    ),
    DangerousOperation(
        operation_id="git.reflog.expire",
        command_class="consequential",
        consequence="Expire reflog entries; destroys the recovery path for lost commits.",
        persistable=False,
    ),
    DangerousOperation(
        operation_id="git.reflog.delete",
        command_class="consequential",
        consequence="Delete a reflog entry; destroys the recovery path for lost commits.",
        persistable=False,
    ),
)

OPERATION_REGISTRY: Mapping[str, DangerousOperation] = MappingProxyType({
    operation.operation_id: operation for operation in _OPERATION_ENTRIES
})
PUBLICATION_OPERATION_IDS: frozenset[str] = frozenset(
    operation.operation_id
    for operation in _OPERATION_ENTRIES
    if operation.command_class == "publication"
)
CONSEQUENTIAL_OPERATION_IDS: frozenset[str] = frozenset(
    operation.operation_id
    for operation in _OPERATION_ENTRIES
    if operation.command_class == "consequential"
)

_GRANT_CHOICES_BY_CLASS: dict[DangerousCommandClass, tuple[GrantChoice, ...]] = {
    # Publication may be remembered for the project; consequential operations
    # never offer a persistent choice, and a forged "always" answer for one is
    # rejected even if a client sends it.
    "publication": ("deny", "allow_once", "allow_session", "allow_project"),
    "consequential": ("deny", "allow_once", "allow_session"),
}


def operations_for_ids(operation_ids: Iterable[str]) -> tuple[DangerousOperation, ...]:
    """Resolve operation IDs to registry entries, in registry order."""
    wanted = set(operation_ids)
    unknown = wanted - set(OPERATION_REGISTRY)
    if unknown:
        raise ValueError(f"unregistered dangerous operation IDs: {sorted(unknown)}")
    return tuple(
        operation
        for operation in _OPERATION_ENTRIES
        if operation.operation_id in wanted
    )


def grant_choices_for(command_class: DangerousCommandClass) -> tuple[GrantChoice, ...]:
    """The grant choices offered for one grant-choice class."""
    return _GRANT_CHOICES_BY_CLASS[command_class]


def is_grant_choice_valid(operation_id: str, choice: GrantChoice) -> bool:
    """Whether one grant choice is valid for one operation.

    Consequential operations reject the persistent choice outright, so a
    forged "always" response never authorizes them; unregistered operations
    accept no choice at all.
    """
    operation = OPERATION_REGISTRY.get(operation_id)
    if operation is None:
        return False
    return choice in _GRANT_CHOICES_BY_CLASS[operation.command_class]


@dataclass(frozen=True)
class GitRepositoryRelocation:
    """Raw repository-relocation operands from git's global options.

    ``-C`` values accumulate in order (git applies each one in turn);
    ``--git-dir``/``--work-tree`` record the last value, which wins.
    """

    cwd_paths: tuple[str, ...] = ()
    git_dir: str | None = None
    work_tree: str | None = None


@dataclass(frozen=True)
class ModeledGitCall:
    """Modeled operands of one classified git invocation.

    ``options`` holds the resolved option tokens (long options by canonical
    name, short options one character each); ``option_values`` holds their
    consumed operands; ``operands`` holds the positional arguments (for push,
    the repository followed by the refspecs).
    """

    subcommand: str
    options: tuple[str, ...]
    option_values: tuple[tuple[str, str], ...]
    operands: tuple[str, ...]


@dataclass(frozen=True)
class GrantableGitCall:
    """A git invocation whose missing authority is grantable per operation."""

    operations: tuple[DangerousOperation, ...]
    consequence: str
    call: ModeledGitCall
    relocation: GitRepositoryRelocation

    @property
    def operation_ids(self) -> tuple[str, ...]:
        """The required operation IDs, in registry order."""
        return tuple(operation.operation_id for operation in self.operations)


@dataclass(frozen=True)
class DangerousCommandClassification:
    """Typed verdict of the pure dangerous-git classifier for one whole call.

    ``routine``: no dangerous-command involvement. ``hard_denied``: denied
    with no grantable operation; ``reason`` says why. ``grantable``: denied
    pending per-operation grants; ``grantable`` carries the required
    operations, the consequence summary, the modeled operands, and the
    repository relocation.
    """

    kind: ClassificationKind
    reason: str | None = None
    grantable: GrantableGitCall | None = None

    def __post_init__(self) -> None:
        if (self.grantable is not None) != (self.kind == "grantable"):
            raise ValueError(
                "grantable payload must be present exactly when kind is 'grantable'"
            )


@dataclass(frozen=True)
class RepositoryScope:
    """Resolved repository/worktree identity scoping dangerous-command authority."""

    git_dir: Path
    work_tree: Path | None


@dataclass(frozen=True)
class DangerousCommandAuthority:
    """Immutable snapshot of the operations one source authorizes in one scope.

    Authority is keyed by operation and repository/worktree scope, never by
    shell prefixes; later packages assemble snapshots from accepted protected
    sources and runtime-owned grants.
    """

    operations: frozenset[str]
    scope: RepositoryScope
    source: AuthoritySource


@dataclass(frozen=True)
class DangerousCommandProposal:
    """Whole-call proposal attached to a grantable dangerous-command denial.

    Runtime-derived by the permission flow: the literal original command, the
    registered project, the repository/worktree identity, the required
    operation IDs, and the consequence summary.
    """

    command: tuple[str, ...]
    project: Path
    scope: RepositoryScope
    operations: tuple[str, ...]
    consequence: str
