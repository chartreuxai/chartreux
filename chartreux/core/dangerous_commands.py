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

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
import ipaddress
from pathlib import Path
import re
import socket
from types import MappingProxyType
from typing import Literal, cast

from chartreux.utils.remote_exec import (
    REMOTE_EXEC_NETWORK_LOCALITY_CAPABILITY as REMOTE_EXEC_NETWORK_LOCALITY_CAPABILITY,
    REMOTE_EXEC_OPERATION_ID,
    REMOTE_EXEC_SSH_TRUST_CAPABILITY as REMOTE_EXEC_SSH_TRUST_CAPABILITY,
    RemoteExecAttestation as RemoteExecAttestation,
)

DangerousCommandClass = Literal["publication", "consequential", "remote_exec"]
GrantChoice = Literal["deny", "allow_once", "allow_session", "allow_project"]
AuthoritySource = Literal["protected_config", "runtime_grant"]
ClassificationKind = Literal["routine", "hard_denied", "grantable"]
AddressLocality = Literal["local", "non_local"]
# The fail-closed resolve-validate verdict for a hostname destination
# (v0.7.1 S4b2a): only a unique, non-local answer admits; ambiguity (a
# multi-answer resolution), any local answer, and any resolution failure
# deny.
HostnameResolution = Literal["unique_non_local", "ambiguous", "local", "failed"]
# The rebinding verdict for re-observing a hostname's resolution against the
# validated pinned address (v0.7.1 S4b2b): only "unchanged" — the fresh
# answer is still the same unique non-local address — keeps the pin; every
# other verdict denies, so a changed answer is never contacted.
HostnameRebind = Literal["unchanged", "changed", "ambiguous", "local", "failed"]


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


# Fail-closed local-address classification for remote-exec destinations
# (S4b1a). Canonicalization above is NOT address validation: it lowercases
# DNS names and compresses bracketed IPv6 but never proves a literal is
# non-local, so the locality verdict is the classifier's own. The classifier
# is INERT until S4b1b consumes it at the resolver seam — nothing here
# changes any admission on its own.
_IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address

# The whole "this network" block, not just the unspecified 0.0.0.0: Linux
# connects 0.x addresses to loopback, so every 0/8 form denies.
_THIS_NETWORK_V4 = ipaddress.ip_network("0.0.0.0/8")
# IPv4-mapped IPv6 (::ffff:0:0/96) resolves on the wire to the embedded IPv4
# address, which may be local even when the embedding looks public
# (::ffff:8.8.8.8 is globally routed as a literal), so the whole block denies.
_IPV4_MAPPED_V6 = ipaddress.ip_network("::ffff:0:0/96")


def _local_interface_addresses() -> frozenset[_IPAddress]:
    """This host's interface addresses, enumerated through getaddrinfo.

    Any failure raises, and an empty enumeration is equally "absence of
    data": the classifier treats both as locality unknown and denies, so a
    host whose own name does not resolve fails closed instead of skipping
    the interface check.
    """
    addresses: set[_IPAddress] = set()
    for info in socket.getaddrinfo(socket.gethostname(), None):
        addresses.add(ipaddress.ip_address(info[4][0]))
    return frozenset(addresses)


def _parse_ip_literal(host: str) -> _IPAddress | None:
    """Parse one host spelling as an IP literal, brackets tolerated.

    Anything else — DNS names, inet_aton shorthand, malformed octets — is
    not a literal and returns None: a classification failure.
    """
    if not isinstance(host, str):
        return None
    text = host.strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    try:
        return ipaddress.ip_address(text)
    except ValueError:
        return None


def _is_local_form(address: _IPAddress) -> bool:
    """Whether the literal itself is a local or unassignable form."""
    if (
        address.is_unspecified
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
    ):
        return True
    if address.is_private or address.is_reserved or not address.is_global:
        return True
    if isinstance(address, ipaddress.IPv4Address):
        return address in _THIS_NETWORK_V4
    return (
        address in _IPV4_MAPPED_V6
        or address.sixtofour is not None
        or address.teredo is not None
    )


def classify_local_address(
    host: str,
    *,
    interface_addresses: Callable[
        [], frozenset[_IPAddress]
    ] = _local_interface_addresses,
) -> AddressLocality:
    """Fail-closed locality verdict for one remote-exec host spelling.

    ``"local"`` is the deny outcome and is returned for every form that must
    never be contacted: loopback, link-local, unique-local, multicast,
    unspecified (``0.0.0.0``, ``::``, and the whole ``0.0.0.0/8`` "this
    network" block), IPv4-mapped and other IPv4-embedding IPv6, every
    private, reserved or otherwise non-global range, any address owned by a
    local interface, anything that does not parse as an IP literal (DNS
    names are the S4b2 resolution arms and deny until then), and ANY failure
    of the local-interface enumeration — getaddrinfo errors, unparseable
    enumeration entries, and an empty enumeration deny rather than skip
    the interface check. Only a literal that parses, is globally routable,
    sits in no local or unassignable form, and is not owned by a local
    interface classifies ``"non_local"``; a refuse-all classifier is wrong by
    construction.
    """
    address = _parse_ip_literal(host)
    if address is None:
        return "local"
    try:
        interfaces = interface_addresses()
    except Exception:
        return "local"
    if not interfaces or address in interfaces or _is_local_form(address):
        return "local"
    return "non_local"


def _system_hostname_resolver(host: str) -> frozenset[_IPAddress]:
    """Resolve one hostname through the system resolver.

    ``getaddrinfo`` consults the nsswitch chain, so ``/etc/hosts`` is honored
    ahead of DNS exactly as the design's resolve-validate-pin rule requires;
    the answer set is the distinct resolved addresses. Any failure raises
    (getaddrinfo errors, unparseable sockaddrs), and the caller fails closed.
    """
    addresses: set[_IPAddress] = set()
    for info in socket.getaddrinfo(host, None):
        addresses.add(ipaddress.ip_address(info[4][0]))
    return frozenset(addresses)


def classify_hostname_resolution(
    host: str,
    *,
    resolver: Callable[[str], frozenset[_IPAddress]] = _system_hostname_resolver,
    interface_addresses: Callable[
        [], frozenset[_IPAddress]
    ] = _local_interface_addresses,
) -> tuple[HostnameResolution, _IPAddress | None]:
    """Fail-closed resolve-validate verdict for one hostname destination.

    Returns ``("unique_non_local", address)`` only when the system resolver
    (which honors ``/etc/hosts``) yields exactly ONE distinct address and that
    answer classifies ``non_local`` under the S4b1a classifier. Everything
    else denies: a multi-answer resolution is ``("ambiguous", None)`` no
    matter which families its answers are in, a single answer that
    classifies local (loopback, private, mapped, interface-owned, or an
    unestablishable locality) is ``("local", address)``, and any resolution
    failure — resolver error, empty answer set, a non-string host, or an IP
    literal (literals route through ``classify_local_address`` instead) —
    is ``("failed", None)``. A refuse-all classifier is wrong by
    construction: the unique-non-local arm is the only admission.
    """
    if not isinstance(host, str) or _parse_ip_literal(host) is not None:
        return "failed", None
    try:
        answers = resolver(host)
    except Exception:
        return "failed", None
    if not answers:
        return "failed", None
    if len(answers) != 1:
        return "ambiguous", None
    (address,) = tuple(answers)
    if (
        classify_local_address(str(address), interface_addresses=interface_addresses)
        != "non_local"
    ):
        return "local", address
    return "unique_non_local", address


def classify_resolution_rebind(
    host: str,
    pinned_address: str,
    *,
    resolver: Callable[[str], frozenset[_IPAddress]] = _system_hostname_resolver,
    interface_addresses: Callable[
        [], frozenset[_IPAddress]
    ] = _local_interface_addresses,
) -> tuple[HostnameRebind, _IPAddress | None]:
    """Fail-closed rebinding verdict: re-observe a hostname's resolution
    against the validated pinned address (v0.7.1 S4b2b).

    Returns ``("unchanged", address)`` only when the FRESH system-resolver
    answer (honoring ``/etc/hosts``) is still unique, non-local, and exactly
    the pinned address — address equality is by value, so IPv6 spelling
    variants of the same address compare equal. Every other outcome denies:
    another unique non-local address is ``("changed", address)``, while a
    rebind to a local/loopback/unspecified answer, ambiguity, or any
    resolution failure keeps the fresh resolve-validate verdict
    (``"local"``/``"ambiguous"``/``"failed"``) with its address. A pinned
    spelling that does not parse as an IP literal, a non-string host, and a
    literal host (literals never resolve) all fail closed as
    ``("failed", None)``. A refuse-all rebinding model is wrong by
    construction: the unchanged arm is the only admission.
    """
    pinned = _parse_ip_literal(pinned_address)
    if pinned is None:
        return "failed", None
    verdict, resolved = classify_hostname_resolution(
        host, resolver=resolver, interface_addresses=interface_addresses
    )
    if verdict == "unique_non_local" and resolved is not None:
        if resolved != pinned:
            return "changed", resolved
        return "unchanged", resolved
    # Every other fresh verdict denies: "ambiguous"/"failed" carry no usable
    # answer, and a "local" fresh answer is a rebind to a local address —
    # none of them is ever contacted.
    return cast("HostnameRebind", verdict), resolved


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
