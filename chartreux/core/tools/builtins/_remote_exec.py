"""SSH remote-exec grammar, resolve-validate-pin, and per-spawn pinning (v0.7.1 S4b1b + S4b2a).

``ssh`` left the shipped denylist under this module's closed grammar: a
positive destination — an IP literal, or a hostname whose system-resolver
(honoring ``/etc/hosts``) answer is unique and non-local — classifies ``ask``
as the ``network.remote-exec`` operation with the exact destination attached,
and every other form — local or ambiguous literals, ambiguous / local /
failed hostname resolutions, hostile options, composed or wrapped
invocations — hard-denies with no approval path. The fail-closed locality
verdict is the S4b1a classifier (``classify_local_address``) and the
resolve-validate verdict is ``classify_hostname_resolution`` (S4b2a), both
consumed here at the resolver seam.

The execution rewrite pins the admission into the actual argv every spawn
point receives: for a hostname destination the RESOLVED address is the
destination operand of the spawned ``ssh`` argv while the host-key identity
stays bound to the granted hostname (``HostKeyAlias`` is the destination's
canonical key), and for a literal destination the approved address itself
rides — plus the fixed known-hosts file, so the spawn connects to the
validated address under a fixed host-key identity and never re-resolves or
re-reads a user-controlled trust file. S4b2b adds the rebinding guard at the
spawn: the validated pinned address is re-observed once inside the same
plan transaction, and a changed answer — in particular a rebind to a
local/loopback/unspecified address — is never contacted (the spawn denies).
A transport that cannot consume the pin hard-denies instead of falling back
to the original command string.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
import ipaddress
import re
import shlex
from typing import TYPE_CHECKING

from chartreux.core.dangerous_commands import (
    HostnameRebind,
    RemoteDestination,
    _local_interface_addresses,
    _parse_ip_literal,
    _system_hostname_resolver,
    canonical_remote_destination,
    classify_hostname_resolution,
    classify_local_address,
    classify_resolution_rebind,
)
from chartreux.core.tools.builtins._shell_argv import _command_name

if TYPE_CHECKING:
    # The S4b2c transfer grammar's parsed call; imported only for typing to
    # avoid a runtime import cycle (_remote_transfer imports this module).
    from chartreux.core.tools.builtins._remote_transfer import TransferCall

# The fixed host-key identity every pinned spawn carries: the known-hosts file
# is a fixed, never-user-controlled path, and the host-key alias is the
# granted destination's canonical key, so the host key is looked up under
# that identity instead of a user-supplied file or alias.
PINNED_KNOWN_HOSTS_FILE = "/dev/null/chartreux-pinned-known-hosts"

# Closed option grammar. Anything outside these spellings fails closed: an
# option the grammar never saw can redirect execution (ProxyCommand,
# RemoteCommand), open secondary connections (ControlMaster/-L/-R/-D/-W),
# read a user-controlled config or trust file (-F, UserKnownHostsFile,
# GlobalKnownHostsFile, IdentityFile), or otherwise exceed what the grant
# covers.
_SAFE_SSH_FLAGS = frozenset({"-4", "-6", "-C", "-q", "-N", "-T", "-v"})
_SAFE_SSH_O_KEYS = frozenset({
    "BatchMode",
    "ConnectTimeout",
    "ServerAliveInterval",
    "ServerAliveCountMax",
})
# Options this module itself pins into the argv. They are admissible only
# with exactly the pinned values, which is also what makes re-pinning an
# already-pinned argv idempotent.
_PINNED_SSH_O_VALUES = {
    "UserKnownHostsFile": PINNED_KNOWN_HOSTS_FILE,
    "HostKeyAlias": None,  # filled per destination: the canonical key
    "StrictHostKeyChecking": "yes",
    "BatchMode": "yes",
}
_HOSTILE_SSH_O_KEYS = frozenset({
    "ProxyCommand",
    "ProxyJump",
    "GlobalKnownHostsFile",
    "ControlMaster",
    "ControlPath",
    "ControlPersist",
    "IdentityFile",
    "LocalCommand",
    "PermitLocalCommand",
    "RemoteCommand",
    "SendEnv",
    "SetEnv",
})
_SSH_USER_RE = re.compile(r"[A-Za-z0-9._+-]+")
# The DNS-name host spelling the closed grammar admits (the same host
# pattern the canonical destination parser uses, without the port): an
# operand host that is not an IP literal must match this to parse at all.
_SSH_HOSTNAME_RE = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?")
_MAX_TCP_PORT = 65535
_DEFAULT_SSH_PORT = 22


class RemoteExecDenial(Exception):
    """A remote-exec form the closed grammar refuses; no approval path."""


@dataclass(frozen=True)
class SshCall:
    """One parsed ``ssh`` invocation inside the closed grammar.

    ``host`` is the GRANTED host spelling — an IP literal, or a DNS name for
    an S4b2a hostname destination. ``resolved_host`` is the pinned resolved
    address that rides into the actual argv as the destination operand; it
    is set for hostname destinations once classified, and for an already
    pinned argv (whose operand is the resolved address) at parse time.
    """

    option_tokens: tuple[str, ...]
    user: str
    host: str
    port: int
    remote_command: tuple[str, ...]
    resolved_host: str | None = None

    @property
    def destination(self) -> RemoteDestination:
        """The canonical (user, host, port) identity of this call."""
        return RemoteDestination(user=self.user, host=self.host, port=self.port)

    @property
    def destination_operand(self) -> str:
        """The pinned destination spelling for the actual argv."""
        host = self.resolved_host if self.resolved_host is not None else self.host
        bracketed = f"[{host}]" if ":" in host else host
        return f"{self.user}@{bracketed}"

    @property
    def argv_is_pinned(self) -> bool:
        """Whether this call's argv already carries this destination's pin."""
        return _argv_is_pinned(self)


@dataclass(frozen=True)
class RemoteExecClassification:
    """The verdict for one command's argv: ``routine``, ``deny``, or ``ask``.

    ``ask`` carries the exact destination a ``network.remote-exec`` grant
    must cover and the parsed call, so the execution boundary can pin the
    approved address into the spawned argv.
    """

    kind: str
    reason: str | None = None
    destination: RemoteDestination | None = None
    call: SshCall | TransferCall | None = None

    @property
    def is_pinned(self) -> bool:
        """Whether the classified call's argv already carries the pin."""
        return self.call is not None and getattr(self.call, "argv_is_pinned", False)


@dataclass(frozen=True)
class RemoteExecPlan:
    """The pinned execution plan for one standalone remote invocation.

    ``argv`` is the actual spawn argv (approved address + fixed host-key
    identity pinned in); ``command`` is the same argv as shell text for the
    string transports. ``resolved_address`` is the address a hostname
    destination resolved to at validation (None for literal destinations,
    where the host is already the address) — the validated snapshot S4b2b's
    rebinding safety re-observes at the spawn: a changed answer is never
    contacted (the spawn denies). ``client`` names the invoked remote client
    — ``ssh``, or since v0.7.1 S4b2c a bounded transfer client (``scp`` /
    ``rsync``) whose pinned argv carries the same address and host-key
    identity pins. A transport that cannot consume the plan hard-denies
    instead of spawning the original command string.
    """

    destination: RemoteDestination
    argv: tuple[str, ...]
    command: str
    resolved_address: str | None = None
    client: str = "ssh"


def _interface_addresses() -> frozenset:
    """The local-interface enumeration seam (patchable for hermetic tests)."""
    return _local_interface_addresses()


def _hostname_resolver(host: str) -> frozenset:
    """The hostname-resolution seam (patchable for hermetic tests)."""
    return _system_hostname_resolver(host)


def _address_spelling(address: object) -> str:
    """The canonical literal spelling of one resolved address."""
    if isinstance(address, ipaddress.IPv6Address):
        return address.compressed
    return str(address)


def _parse_port(value: str) -> int:
    if not value.isdigit():
        raise RemoteExecDenial(f"the port '{value}' is not a decimal port number")
    port = int(value)
    if not 1 <= port <= _MAX_TCP_PORT:
        raise RemoteExecDenial(f"the port '{value}' is out of range")
    return port


@dataclass
class _SshOptionState:
    """Mutable parse state carried across one ssh argv's option tokens."""

    port: int | None = None
    user: str | None = None


def _check_ssh_o_value(value: str, option_tokens: list[str]) -> None:
    """Validate one ``-o`` value against the closed key grammar and record it."""
    key, sep, _option_value = value.partition("=")
    if not sep:
        raise RemoteExecDenial(f"the -o option '{value}' is not a key=value pair")
    if key in _HOSTILE_SSH_O_KEYS:
        raise RemoteExecDenial(
            f"the hostile ssh option '{key}' is denied: it can redirect "
            "execution or read a user-controlled trust file"
        )
    if key in _PINNED_SSH_O_VALUES or key == "HostKeyAlias":
        # Transport-pinned keys: admissible only with exactly the pinned
        # value, validated once the destination is known (HostKeyAlias must
        # equal the destination's canonical key). This is also what makes
        # re-classifying an already-pinned argv succeed.
        option_tokens.extend(["-o", value])
        return
    if key not in _SAFE_SSH_O_KEYS:
        raise RemoteExecDenial(
            f"the unmodeled ssh option '{key}' is denied fail-closed"
        )
    option_tokens.extend(["-o", value])


def _parse_ssh_option(
    tokens: list[str], index: int, option_tokens: list[str], state: _SshOptionState
) -> int:
    """Parse the option token at ``index``; return the next index.

    Raises :class:`RemoteExecDenial` for every option spelling outside the
    closed grammar.
    """
    token = tokens[index]
    if token in _SAFE_SSH_FLAGS:
        option_tokens.append(token)
        return index + 1
    if token == "-p" or (token.startswith("-p") and token[2:].isdigit()):
        if token == "-p":
            if index + 1 >= len(tokens):
                raise RemoteExecDenial("the -p option is missing its port")
            state.port = _parse_port(tokens[index + 1])
            option_tokens.extend(["-p", tokens[index + 1]])
            return index + 2
        state.port = _parse_port(token[2:])
        option_tokens.append(token)
        return index + 1
    if token == "-l" or (token.startswith("-l") and token[2:]):
        if token == "-l":
            if index + 1 >= len(tokens):
                raise RemoteExecDenial("the -l option is missing its user")
            value = tokens[index + 1]
            option_tokens.extend(["-l", value])
            index += 2
        else:
            value = token[2:]
            option_tokens.append(token)
            index += 1
        if state.user is not None:
            raise RemoteExecDenial("the user is named more than once")
        state.user = value
        return index
    if token == "-o" or token.startswith("-o"):
        if token == "-o":
            if index + 1 >= len(tokens):
                raise RemoteExecDenial("the -o option is missing its value")
            value = tokens[index + 1]
            index += 2
        else:
            value = token[2:]
            index += 1
        _check_ssh_o_value(value, option_tokens)
        return index
    raise RemoteExecDenial(f"the unmodeled ssh option '{token}' is denied fail-closed")


def _resolve_ssh_destination(
    destination_operand: str, state: _SshOptionState
) -> tuple[str, str]:
    """Resolve the destination operand into the (user, host spelling) pair."""
    if "@" in destination_operand:
        operand_user, _, operand_host = destination_operand.rpartition("@")
        if state.user is not None and state.user != operand_user:
            raise RemoteExecDenial("the user is named more than once")
        state.user = operand_user
    else:
        operand_host = destination_operand
    if not state.user or _SSH_USER_RE.fullmatch(state.user) is None:
        raise RemoteExecDenial(
            "the ssh destination must name a user (user@host, or host with -l user)"
        )
    if not operand_host:
        raise RemoteExecDenial("the ssh destination names no host")
    return state.user, operand_host


def _check_pinned_option_values(call: SshCall) -> None:
    """Reject a transport-pinned option carrying any value but the pinned one.

    The known-hosts file is the fixed pinned path, strict host-key checking
    and batch mode stay on, and the host-key alias is this call's canonical
    destination key. Anything else is an override attempt.
    """
    option_map = _option_map(call)
    expected_pin = {
        "UserKnownHostsFile": PINNED_KNOWN_HOSTS_FILE,
        "HostKeyAlias": call.destination.key,
        "StrictHostKeyChecking": "yes",
        "BatchMode": "yes",
    }
    for key, expected in expected_pin.items():
        if key in option_map and option_map[key] != expected:
            raise RemoteExecDenial(
                f"the ssh option '{key}' is pinned by the remote-exec transport "
                f"and cannot be overridden (denied value '{option_map[key]}')"
            )


def _option_map_from_tokens(option_tokens: Iterable[str]) -> dict[str, str]:
    """The ``-o key=value`` pairs of raw option tokens."""
    option_map: dict[str, str] = {}
    tokens = list(option_tokens)
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "-o" and index + 1 < len(tokens):
            key, _, value = tokens[index + 1].partition("=")
            option_map[key] = value
            index += 2
        else:
            index += 1
    return option_map


def _pinned_alias_destination(option_tokens: list[str]) -> RemoteDestination | None:
    """The granted destination a fully pinned argv carries, or None.

    The transport pin is the four pinned ``-o`` options; when all four are
    present with their pinned values, the ``HostKeyAlias`` value is the
    canonical destination key the pin was minted for — the granted hostname
    for an S4b2a hostname destination, the literal itself for an IP-literal
    one. A non-canonical alias returns None and the unpinned path's
    ``_check_pinned_option_values`` denies the override instead.
    """
    option_map = _option_map_from_tokens(option_tokens)
    if (
        option_map.get("UserKnownHostsFile") == PINNED_KNOWN_HOSTS_FILE
        and option_map.get("StrictHostKeyChecking") == "yes"
        and option_map.get("BatchMode") == "yes"
        and "HostKeyAlias" in option_map
    ):
        return canonical_remote_destination(option_map["HostKeyAlias"])
    return None


def _hostname_denial_reason(host: str, verdict: str, address: object = None) -> str:
    """The discriminating denial reason for one hostname resolution verdict."""
    if verdict == "ambiguous":
        return (
            f"the destination host '{host}' resolves to multiple addresses: a "
            "unique non-local answer is required and ambiguity is denied"
        )
    if verdict == "local":
        return (
            f"the destination host '{host}' resolves to the local address "
            f"'{address}': local answers are denied"
        )
    return (
        f"the destination host '{host}' could not be resolved through the "
        "system resolver (which honors /etc/hosts): a resolution failure is denied"
    )


def parse_ssh_call(tokens: Iterable[str]) -> SshCall:
    """Parse one ssh argv inside the closed grammar.

    Raises :class:`RemoteExecDenial` for every form outside the grammar; the
    caller fails closed on any denial. The destination operand must name a
    user (``user@host``) or pair a bare host with ``-l user``; the host must
    be an IP literal (bracketed IPv6) or a DNS name (S4b2a — classification
    resolves it and pins the answer), and an argv already carrying the full
    transport pin parses as that pin's granted destination with the resolved
    address as operand.
    """
    tokens = [str(token) for token in tokens]
    if not tokens or _command_name(tokens[0]) != "ssh":
        raise RemoteExecDenial("not an ssh invocation")
    option_tokens: list[str] = []
    state = _SshOptionState()
    destination_operand: str | None = None
    remote_command: tuple[str, ...] = ()
    index = 1
    while index < len(tokens):
        token = tokens[index]
        if token == "--":
            index += 1
            if index < len(tokens):
                destination_operand = tokens[index]
                remote_command = tuple(tokens[index + 1 :])
            break
        if token.startswith("-") and token != "-":
            index = _parse_ssh_option(tokens, index, option_tokens, state)
            continue
        destination_operand = token
        remote_command = tuple(tokens[index + 1 :])
        break
    if destination_operand is None:
        raise RemoteExecDenial("the ssh invocation names no destination")
    user, operand_host = _resolve_ssh_destination(destination_operand, state)
    port = state.port if state.port is not None else _DEFAULT_SSH_PORT
    alias_destination = _pinned_alias_destination(option_tokens)
    if alias_destination is not None:
        # A fully pinned argv: the operand must be the pinned resolved
        # address (an IP literal) and must name the same user and port as
        # the granted destination the HostKeyAlias pin carries. classify
        # revalidates that the operand is the granted host's current unique
        # non-local resolution, so a hand-written pin cannot point at an
        # unvalidated address.
        operand_address = _parse_ip_literal(operand_host)
        if operand_address is None:
            raise RemoteExecDenial(
                "a pinned ssh argv must carry the resolved address as its "
                f"destination operand, not '{operand_host}'"
            )
        if alias_destination.user != user:
            raise RemoteExecDenial(
                "the pinned HostKeyAlias names another user than the "
                "destination operand"
            )
        if alias_destination.port != port:
            raise RemoteExecDenial(
                "the pinned HostKeyAlias names another port than the invocation"
            )
        call = SshCall(
            option_tokens=tuple(option_tokens),
            user=user,
            host=alias_destination.host,
            port=port,
            remote_command=remote_command,
            resolved_host=_address_spelling(operand_address),
        )
        _check_pinned_option_values(call)
        return call
    address = _parse_ip_literal(operand_host)
    if address is not None:
        host = _address_spelling(address)
    elif _SSH_HOSTNAME_RE.fullmatch(operand_host) is not None:
        # S4b2a: a DNS name admits into the grammar syntactically; only the
        # resolve-validate verdict in classify_ssh_call admits it for real.
        host = operand_host.lower()
    else:
        raise RemoteExecDenial(
            f"the destination host '{operand_host}' is not a valid host spelling"
        )
    call = SshCall(
        option_tokens=tuple(option_tokens),
        user=user,
        host=host,
        port=port,
        remote_command=remote_command,
    )
    _check_pinned_option_values(call)
    return call


def _revalidate_pinned_destination(
    host: str,
    resolved_host: str,
    destination: RemoteDestination,
    enum: Callable[[], frozenset],
    res: Callable[[str], frozenset],
) -> str | None:
    """Revalidate one pinned (host, resolved address) pair at a spawn boundary.

    The pinned operand is checked against the granted destination the pin
    carries: a literal alias host must equal the operand and stay non-local;
    a hostname alias host must still resolve uniquely non-locally TO the
    operand. The spawn connects to the pinned address either way; a changed
    or unmatched answer returns a denial reason (None admits). Shared by the
    ssh grammar and the S4b2c transfer grammar.
    """
    alias_literal = _parse_ip_literal(host)
    if alias_literal is not None:
        if resolved_host != _address_spelling(alias_literal):
            return (
                f"the pinned address '{resolved_host}' does not "
                f"match the granted destination host '{host}' and is denied"
            )
        if classify_local_address(resolved_host, interface_addresses=enum) == "local":
            return (
                f"the pinned address '{resolved_host}' is local or "
                "ambiguous (loopback, private, mapped, interface-owned, "
                "or an unestablishable locality) and is denied"
            )
    else:
        verdict, resolved = classify_hostname_resolution(
            host, resolver=res, interface_addresses=enum
        )
        if verdict != "unique_non_local" or resolved is None:
            return (
                f"the pinned address '{resolved_host}' cannot be "
                f"validated for the granted host '{host}' and is "
                f"denied ({_hostname_denial_reason(host, verdict, resolved)})"
            )
        if _address_spelling(resolved) != resolved_host:
            return (
                f"the pinned address '{resolved_host}' is not the "
                f"current unique non-local resolution "
                f"'{_address_spelling(resolved)}' of the granted host "
                f"'{host}' and is denied"
            )
    return None


def _revalidate_pinned_call(
    call: SshCall, enum: Callable[[], frozenset], res: Callable[[str], frozenset]
) -> RemoteExecClassification:
    """Revalidate an already pinned argv (re-derived at a spawn boundary).

    The pinned operand is checked against the granted destination the
    HostKeyAlias pin carries: a literal alias host must equal the operand and
    stay non-local; a hostname alias host must still resolve uniquely
    non-locally TO the operand. The spawn connects to the pinned address
    either way; a changed or unmatched answer denies.
    """
    assert call.resolved_host is not None  # the caller routes only pinned calls here
    reason = _revalidate_pinned_destination(
        call.host, call.resolved_host, call.destination, enum, res
    )
    if reason is not None:
        return RemoteExecClassification(
            "deny", reason=reason, destination=call.destination, call=call
        )
    return RemoteExecClassification("ask", destination=call.destination, call=call)


def _classify_destination(
    call: SshCall, enum: Callable[[], frozenset], res: Callable[[str], frozenset]
) -> RemoteExecClassification:
    """Classify one parsed ssh call's destination by locality and resolution."""
    if _parse_ip_literal(call.host) is not None:
        if classify_local_address(call.host, interface_addresses=enum) == "local":
            return RemoteExecClassification(
                "deny",
                reason=(
                    f"the destination address '{call.host}' is local or ambiguous "
                    "(loopback, private, mapped, interface-owned, or an "
                    "unestablishable locality) and is denied"
                ),
                destination=call.destination,
                call=call,
            )
        return RemoteExecClassification("ask", destination=call.destination, call=call)
    verdict, resolved = classify_hostname_resolution(
        call.host, resolver=res, interface_addresses=enum
    )
    if verdict != "unique_non_local" or resolved is None:
        return RemoteExecClassification(
            "deny",
            reason=_hostname_denial_reason(call.host, verdict, resolved),
            destination=call.destination,
            call=call,
        )
    pinned = replace(call, resolved_host=_address_spelling(resolved))
    return RemoteExecClassification("ask", destination=pinned.destination, call=pinned)


def classify_ssh_call(
    tokens: Iterable[str],
    *,
    interface_addresses: Callable[[], frozenset] | None = None,
    resolver: Callable[[str], frozenset] | None = None,
) -> RemoteExecClassification:
    """Classify one command's argv for the closed ssh grammar.

    ``routine`` for non-ssh commands; ``deny`` with a reason for every
    refused form; ``ask`` for a positive destination — an IP literal the
    fail-closed locality classifier (S4b1a) proves non-local, or a hostname
    (S4b2a) whose system-resolver answer (honoring ``/etc/hosts``) is unique
    and non-local — carrying the exact granted destination and the parsed
    call with the resolved address pinned in.
    """
    tokens = [str(token) for token in tokens]
    if not tokens or _command_name(tokens[0]) != "ssh":
        return RemoteExecClassification("routine")
    enum = interface_addresses or _interface_addresses
    res = resolver or _hostname_resolver
    try:
        call = parse_ssh_call(tokens)
    except RemoteExecDenial as exc:
        return RemoteExecClassification("deny", reason=str(exc))
    if call.resolved_host is not None:
        return _revalidate_pinned_call(call, enum, res)
    return _classify_destination(call, enum, res)


def _rebind_denial_reason(
    host: str, pinned: str, verdict: HostnameRebind, resolved: object = None
) -> str:
    """The discriminating denial reason for one rebinding verdict."""
    fresh = f"'{_address_spelling(resolved)}'" if resolved is not None else "no answer"
    if verdict == "changed":
        return (
            f"the destination host '{host}' re-resolved at the spawn to "
            f"the changed address {fresh}, not the validated pinned address "
            f"'{pinned}': a changed answer is never contacted and "
            "the spawn is denied"
        )
    if verdict == "local":
        return (
            f"the destination host '{host}' re-resolved at the spawn to "
            f"the local address {fresh}, not the validated pinned address "
            f"'{pinned}': a rebind to a local address is never "
            "contacted and the spawn is denied"
        )
    if verdict == "ambiguous":
        return (
            f"the destination host '{host}' now resolves ambiguously at "
            f"the spawn, not to the validated pinned address "
            f"'{pinned}': a changed answer is never contacted and "
            "the spawn is denied"
        )
    return (
        f"the destination host '{host}' could not be re-resolved at the "
        f"spawn against the validated pinned address '{pinned}': a "
        "changed answer is never contacted and the spawn is denied"
    )


def enforce_pinned_resolution(host: str, resolved_host: str | None) -> None:
    """S4b2b rebinding guard core: the validated pin, never a changed answer.

    A hostname destination's resolution is re-observed once through the same
    system-resolver seam and compared with the validated pinned address
    snapshot. Only an unchanged unique non-local answer passes; ANY changed
    answer — a rebind to a local/loopback/unspecified address, to another
    address, ambiguity, or a resolution failure — raises, so the changed
    answer is never contacted. A literal destination is its own address and
    needs no re-resolution, and a hostname without a pinned address fails
    closed. Shared by the ssh grammar and the S4b2c transfer grammar; the
    caller runs this inside the same plan transaction as the validation,
    before the argv is built, so no changed answer can ride into the spawn.
    """
    if _parse_ip_literal(host) is not None:
        return
    if resolved_host is None:
        raise RemoteExecDenial(
            f"the hostname destination '{host}' has no validated pinned "
            "address to re-observe at the spawn"
        )
    verdict, resolved = classify_resolution_rebind(
        host,
        resolved_host,
        resolver=_hostname_resolver,
        interface_addresses=_interface_addresses,
    )
    if verdict != "unchanged":
        raise RemoteExecDenial(
            _rebind_denial_reason(host, resolved_host, verdict, resolved)
        )


def enforce_pinned_resolution_at_spawn(call: SshCall) -> SshCall:
    """S4b2b rebinding guard at the spawn for one ssh call.

    The validated pinned ``resolved_host`` snapshot is re-observed once
    inside the same plan transaction and any CHANGED answer — a rebind to a
    local/loopback/unspecified address, to another address, ambiguity, or a
    resolution failure — raises, so the changed answer is never contacted.
    A literal destination is its own address and needs no re-resolution, and
    a hostname call without a pinned address fails closed.
    """
    enforce_pinned_resolution(call.host, call.resolved_host)
    return call


def _option_map(call: SshCall) -> dict[str, str]:
    """The ``-o key=value`` pairs of the call's option tokens."""
    return _option_map_from_tokens(call.option_tokens)


def _argv_is_pinned(call: SshCall) -> bool:
    """Whether the call's options already carry this destination's pin."""
    option_map = _option_map(call)
    return (
        option_map.get("UserKnownHostsFile") == PINNED_KNOWN_HOSTS_FILE
        and option_map.get("HostKeyAlias") == call.destination.key
        and option_map.get("StrictHostKeyChecking") == "yes"
    )


def pinned_ssh_argv(call: SshCall) -> tuple[str, ...]:
    """The actual spawn argv: the approved address and fixed host-key identity.

    The pin options lead the argv (ssh takes the first obtained value for an
    option, so the pin cannot be overridden by anything later), the
    destination operand is the canonical approved address — for an S4b2a
    hostname destination the RESOLVED address, while the ``HostKeyAlias`` pin
    stays bound to the granted hostname — and the remote command rides
    unchanged. Re-pinning an already-pinned call is idempotent: the existing
    pin options are dropped and re-added exactly once.
    """
    if call.resolved_host is None and _parse_ip_literal(call.host) is None:
        raise RemoteExecDenial(
            f"the hostname destination '{call.host}' was not resolved, so no "
            "pinned address exists to spawn"
        )
    destination = call.destination
    pin = [
        "-o",
        f"UserKnownHostsFile={PINNED_KNOWN_HOSTS_FILE}",
        "-o",
        f"HostKeyAlias={destination.key}",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        "BatchMode=yes",
    ]
    options = list(call.option_tokens)
    if _argv_is_pinned(call):
        filtered: list[str] = []
        index = 0
        while index < len(options):
            if options[index] == "-o" and index + 1 < len(options):
                key = options[index + 1].partition("=")[0]
                if key in _PINNED_SSH_O_VALUES:
                    index += 2
                    continue
            filtered.append(options[index])
            index += 1
        options = filtered
    argv = ["ssh", *pin, *options]
    if call.port != _DEFAULT_SSH_PORT and not any(
        token == "-p" or (token.startswith("-p") and token[2:].isdigit())
        for token in options
    ):
        argv.extend(["-p", str(call.port)])
    argv.append(call.destination_operand)
    argv.extend(call.remote_command)
    return tuple(argv)


def pinned_ssh_command_text(call: SshCall) -> str:
    """The pinned argv as shell text for string transports (shlex-joined)."""
    return shlex.join(pinned_ssh_argv(call))
