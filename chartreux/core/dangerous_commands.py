"""Typed models for the dangerous-command permission feature.

The operation registry is frozen: every recognized operation carries a
stable operation ID, a grant-choice class, a consequence summary, and whether
the operation is persistable (publication and remote-exec are; consequential
operations never are). Remote-exec authority is a destination grant: it
matches one exact canonical (user, host, port) identity and never
cross-authorizes publication or egress. The
classification models describe what the pure classifier in the shell-command
policy layer reports for one whole call; the authority and proposal types are
the immutable shapes later packages use for resolver snapshots, runtime-owned
grants, and whole-call proposals. This module owns no grant mutation, no
persistence behavior, and no execution.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import ipaddress
from pathlib import Path
import re
from types import MappingProxyType
from typing import Literal

DangerousCommandClass = Literal["publication", "consequential", "remote_exec"]
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
    DangerousOperation(
        operation_id="network.egress",
        command_class="publication",
        consequence=(
            "Send data from this machine to a network destination outside "
            "the local loopback (non-localhost egress via a recognized "
            "network client)."
        ),
        persistable=True,
    ),
    DangerousOperation(
        operation_id="network.remote-exec",
        command_class="remote_exec",
        consequence=(
            "Run a command on a remote machine. The remote command and its "
            "effects run under the remote host's accounts and policy, not "
            "under Chartreux's local guards."
        ),
        persistable=True,
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
    # Publication and remote-exec may be remembered for the project;
    # consequential operations never offer a persistent choice, and a forged
    # "always" answer for one is rejected even if a client sends it.
    "publication": ("deny", "allow_once", "allow_session", "allow_project"),
    "consequential": ("deny", "allow_once", "allow_session"),
    "remote_exec": ("deny", "allow_once", "allow_session", "allow_project"),
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


REMOTE_EXEC_OPERATION_ID = "network.remote-exec"

# A destination is one explicit user, one concrete host (DNS name or IP
# literal; IPv6 only in brackets), and an optional port. Wildcards, ranges,
# and comma-separated alias groups never parse, so v1 admits exact
# destinations only.
_REMOTE_DESTINATION_RE = re.compile(
    r"(?P<user>[A-Za-z0-9._+-]+)@"
    r"(?P<host>\[[0-9A-Fa-f:.]+\]|[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?)"
    r"(?::(?P<port>[0-9]{1,5}))?"
)
_MAX_TCP_PORT = 65535


@dataclass(frozen=True)
class RemoteDestination:
    """Canonical remote-exec destination identity.

    Equality is exact over the (user, host, port) tuple, so a near miss —
    another user, another host, another port — never matches a grant.
    """

    user: str
    host: str
    port: int = 22

    @property
    def key(self) -> str:
        """The canonical persisted spelling (``user@host:port``)."""
        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"{self.user}@{host}:{self.port}"

    def __str__(self) -> str:
        return self.key


def canonical_remote_destination(value: str) -> RemoteDestination | None:
    """Parse one destination spelling into its canonical identity.

    The user is mandatory and case-sensitive (POSIX account names are). The
    host is lowercased, an IPv6 literal is compressed, and a missing port
    canonicalizes to 22. Wildcards, alias groups, empty parts, and
    out-of-range ports return None: the caller fails closed, never guesses.
    """
    if not isinstance(value, str):
        return None
    match = _REMOTE_DESTINATION_RE.fullmatch(value.strip())
    if match is None:
        return None
    raw_host = match.group("host")
    if raw_host.startswith("["):
        try:
            host = ipaddress.IPv6Address(raw_host[1:-1]).compressed
        except ValueError:
            return None
    else:
        host = raw_host.lower()
    raw_port = match.group("port")
    port = 22 if raw_port is None else int(raw_port)
    if not 1 <= port <= _MAX_TCP_PORT:
        return None
    return RemoteDestination(user=match.group("user"), host=host, port=port)


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
    sources and runtime-owned grants. A remote-exec authority additionally
    carries the exact destination it was granted for; no other authority
    carries one.
    """

    operations: frozenset[str]
    scope: RepositoryScope
    source: AuthoritySource
    destination: RemoteDestination | None = None


def authority_admits(
    authority: DangerousCommandAuthority,
    operation_id: str,
    scope: RepositoryScope,
    *,
    destination: RemoteDestination | None = None,
) -> bool:
    """Whether one authority snapshot admits one operation at one scope.

    Operation classes never cross-authorize. A destination authority admits
    only ``network.remote-exec``, and only for the exact destination it was
    granted for; a scope-only authority (``git.push``, ``network.egress``)
    never admits a remote command, and a remote-exec destination grant never
    admits non-remote egress or publication.
    """
    if authority.scope != scope or operation_id not in authority.operations:
        return False
    if operation_id == REMOTE_EXEC_OPERATION_ID:
        return (
            destination is not None
            and authority.destination is not None
            and authority.destination == destination
        )
    return authority.destination is None


@dataclass(frozen=True)
class DangerousCommandProposal:
    """Whole-call proposal attached to a grantable dangerous-command denial.

    Runtime-derived by the permission flow: the literal original command, the
    registered project, the repository/worktree identity, the required
    operation IDs, the consequence summary, and — for a remote-exec call —
    the exact destination the grant would cover.
    """

    command: tuple[str, ...]
    project: Path
    scope: RepositoryScope
    operations: tuple[str, ...]
    consequence: str
    destination: RemoteDestination | None = None
