"""Bounded transfer grammar for ``scp``/``rsync`` (v0.7.1 S4b2c).

``scp`` and ``rsync`` left the shipped denylist under this module's CLOSED
grammar. A remote-bearing invocation — exactly two operands, exactly one of
them a ``user@host:path`` remote spec — classifies ``ask`` as the same
``network.remote-exec`` operation with the exact destination attached (the
same destination-keyed grant model as ``ssh``), but only when the closed
option grammar holds and the destination validates exactly like an ssh
destination: an IP literal the S4b1a fail-closed locality classifier proves
non-local, or a hostname (S4b2a) whose system-resolver answer (honoring
``/etc/hosts``) is unique and non-local. Every other remote-bearing spelling
hard-denies with a discriminating reason and no approval path: no user in
the remote spec, remote-to-remote copies, the rsync daemon ``::`` form,
glob-bearing or ``..``-bearing remote paths (server-side expansion and
server-chosen names), empty remote paths, transport overrides (``-S``/``-F``/
``-i``/``-J``/``-c``/``-l`` on scp, ``-e``/``--rsh``/``--rsync-path`` on
rsync), the legacy-protocol and strict-name-check-disabling scp flags
(``-O``/``-T``), symlink-following rsync options (``-L``/``--copy-links``/
``--copy-unsafe-links``), destination-side mutation options (``--delete``,
``--remove-source-files``, ``--inplace``, backup/partial/temp-dir), local or
ambiguous or failed resolutions, hostile or unmodeled options, and composed
or wrapped invocations. A scp/rsync invocation with NO remote operand is not
a transfer: it returns ``routine`` and the ordinary path layer governs it
(the local copy case), exactly as before the denylist flip.

The execution rewrite pins the admission into the actual argv every spawn
point receives, exactly like ssh: the resolved address is the remote
operand's host while the host-key identity (``HostKeyAlias`` + the fixed
unprovisionable known-hosts file) stays bound to the granted destination, the
pin options lead the argv, and for rsync the pinned ``-e`` ssh transport is
injected (a user-supplied ``-e`` is denied as a transport override). S4b2b's
rebinding guard runs inside the same plan transaction: a changed resolution
at the spawn is never contacted (the spawn denies). A transport that cannot
consume the pin hard-denies instead of falling back to the original text.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
import re
import shlex

from chartreux.core.dangerous_commands import (
    RemoteDestination,
    _parse_ip_literal,
    canonical_remote_destination,
    classify_hostname_resolution,
    classify_local_address,
)
from chartreux.core.tools.builtins import _remote_exec as _rx
from chartreux.core.tools.builtins._remote_exec import (
    _DEFAULT_SSH_PORT,
    _SSH_HOSTNAME_RE,
    _SSH_USER_RE,
    PINNED_KNOWN_HOSTS_FILE,
    RemoteExecClassification,
    RemoteExecDenial,
    _address_spelling,
    _check_ssh_o_value,
    _hostname_denial_reason,
    _option_map_from_tokens,
    _parse_port,
    _pinned_alias_destination,
    _revalidate_pinned_destination,
    classify_ssh_call,
    enforce_pinned_resolution,
)
from chartreux.core.tools.builtins._shell_argv import _command_name

#: The transfer clients this grammar models.
TRANSFER_CLIENTS = ("scp", "rsync")

#: Every client the remote-exec path admits as a standalone invocation.
REMOTE_EXEC_CLIENTS = ("ssh", *TRANSFER_CLIENTS)

# A remote path may not carry shell metacharacters, whitespace, or globs:
# the remote side expands a glob-bearing or shell-metacharacter path itself
# (server-chosen names), so only this closed charset is admitted.
_REMOTE_PATH_RE = re.compile(r"[A-Za-z0-9._+/@~=%-]+")
_GLOB_CHARACTERS = ("*", "?", "[")

#: The bounded grammar admits exactly one local and one remote operand.
_TRANSFER_OPERAND_COUNT = 2
#: Offset of an attached short-option value (the ``-eVALUE`` spelling).
_ATTACHED_SHORT_VALUE_OFFSET = 2

# scp's closed short-option grammar. Combined clusters over the safe set are
# admitted; P and o take values and are handled separately.
_SCP_SAFE_SHORT = frozenset("46Cqvpr")
_SCP_DENIED_SHORT_REASONS = {
    "3": (
        "the -3 remote-to-remote copy mode is denied: a bounded transfer "
        "copies between this machine and one granted destination only"
    ),
    "O": (
        "the -O legacy scp protocol is denied: it accepts server-chosen "
        "names and follows symlinks; the pinned plan runs the SFTP protocol"
    ),
    "T": (
        "the -T option disables strict filename checking and is denied: "
        "server-chosen names must stay bounded"
    ),
    "S": (
        "the -S option names a custom ssh transport program and is denied: "
        "the pinned plan injects the pinned ssh transport"
    ),
    "F": "the -F option reads a user-controlled ssh config and is denied",
    "i": (
        "the -i option names an identity file and is denied: the SSH trust "
        "context is pinned, not user-supplied"
    ),
    "J": "the -J proxy-jump destination is denied: no second hop is admitted",
    "c": "the -c cipher option is denied as a transport override",
    "l": "the -l bandwidth-limit option is unmodeled and denied fail-closed",
    "B": "the -B batch-mode option is unmodeled and denied fail-closed",
}

# rsync's closed short-option grammar (combined clusters over the safe set).
_RSYNC_SAFE_SHORT = frozenset("arptzqvn")
_RSYNC_SAFE_LONG = frozenset({
    "--archive",
    "--recursive",
    "--perms",
    "--times",
    "--compress",
    "--quiet",
    "--verbose",
    "--dry-run",
    "--progress",
})
_RSYNC_DENIED_SHORT_REASONS = {
    "e": (
        "the -e/--rsh option supplies the remote transport and is denied: "
        "the pinned plan injects the pinned ssh transport"
    ),
    "L": (
        "the -L/--copy-links option follows symlinks and is denied: the "
        "modeled transfer never follows symlinks"
    ),
    "P": ("the -P/--partial option keeps server-chosen temporary names and is denied"),
    "b": "the -b/--backup option is denied: destination-side mutation beyond a copy",
    "s": "the -s/--protect-args option is unmodeled and denied fail-closed",
    "D": (
        "the -D devices/specials option is denied: a remote server must not "
        "select local device nodes"
    ),
}
_RSYNC_DENIED_LONG_REASONS = {
    "--rsh": _RSYNC_DENIED_SHORT_REASONS["e"],
    "--rsync-path": (
        "the --rsync-path option runs a command on the remote host and is "
        "denied: a bounded transfer copies files, it never runs remote commands"
    ),
    "--copy-links": _RSYNC_DENIED_SHORT_REASONS["L"],
    "--copy-unsafe-links": _RSYNC_DENIED_SHORT_REASONS["L"],
    "--delete": "the --delete option mutates the destination beyond a copy and is denied",
    "--remove-source-files": (
        "the --remove-source-files option mutates the source side and is denied"
    ),
    "--inplace": (
        "the --inplace option writes directly into the target and is "
        "denied: the modeled transfer keeps server-chosen names inside the "
        "bounded local target"
    ),
}


@dataclass(frozen=True)
class TransferCall:
    """One parsed scp/rsync transfer inside the closed grammar.

    ``host`` is the GRANTED host spelling — an IP literal, or a DNS name for
    an S4b2a hostname destination. ``resolved_host`` is the pinned resolved
    address that rides into the actual argv as the remote operand's host; it
    is set for hostname destinations once classified, and for an
    already-pinned argv (whose remote operand host is the resolved address)
    at parse time. ``operands`` keeps the original two operands in order;
    ``remote_index`` says which one is the remote spec.
    """

    client: str
    option_tokens: tuple[str, ...]
    user: str
    host: str
    port: int
    remote_path: str
    operands: tuple[str, str]
    remote_index: int
    resolved_host: str | None = None

    @property
    def destination(self) -> RemoteDestination:
        """The canonical (user, host, port) identity of this transfer."""
        return RemoteDestination(user=self.user, host=self.host, port=self.port)

    @property
    def remote_operand(self) -> str:
        return self.operands[self.remote_index]

    @property
    def local_operand(self) -> str:
        return self.operands[1 - self.remote_index]

    @property
    def destination_operand(self) -> str:
        """The pinned remote operand spelling for the actual argv."""
        host = self.resolved_host if self.resolved_host is not None else self.host
        bracketed = f"[{host}]" if ":" in host else host
        return f"{self.user}@{bracketed}:{self.remote_path}"

    @property
    def argv_is_pinned(self) -> bool:
        """Whether the call's argv already carries this destination's pin."""
        if self.client == "scp":
            option_map = _option_map_from_tokens(self.option_tokens)
            return (
                option_map.get("UserKnownHostsFile") == PINNED_KNOWN_HOSTS_FILE
                and option_map.get("HostKeyAlias") == self.destination.key
                and option_map.get("StrictHostKeyChecking") == "yes"
                and option_map.get("BatchMode") == "yes"
                and self.resolved_host is not None
            )
        transport = _rsync_pinned_transport(self.option_tokens)
        return transport is not None and transport == self.destination.key


def _remote_separator(operand: str) -> int:
    """The index of the remote ``:`` separator in one operand, or -1.

    The separator is the first colon at bracket depth zero that appears
    before any slash (a colon after a slash is an ordinary local path). A
    ``::`` at depth zero is the rsync daemon module form and denies.
    """
    depth = 0
    for index, char in enumerate(operand):
        if char == "[":
            depth += 1
        elif char == "]":
            depth = max(0, depth - 1)
        elif char == "/" and depth == 0:
            return -1
        elif char == ":" and depth == 0:
            if operand[index + 1 : index + 2] == ":":
                raise RemoteExecDenial(
                    "the rsync daemon module form 'host::module' is denied: "
                    "daemon mode bypasses the pinned ssh transport and its "
                    "trust context"
                )
            return index
    return -1


def _parse_remote_operand(operand: str) -> tuple[str, str, str]:
    """Split one remote operand into (user, host, remote path).

    Raises :class:`RemoteExecDenial` for every remote-shaped operand outside
    the closed grammar: no user, an invalid host spelling, an empty remote
    path, glob metacharacters, shell metacharacters, or a ``..`` segment —
    all of which the remote side would expand or choose itself.
    """
    index = _remote_separator(operand)
    if index < 0:
        raise RemoteExecDenial(
            f"the operand '{operand}' is not a remote spec (no ':' before any '/')"
        )
    user_host, remote_path = operand[:index], operand[index + 1 :]
    if "@" not in user_host:
        raise RemoteExecDenial(
            f"the transfer remote operand '{operand}' must name a user "
            "(user@host:path): a bare host:path leaves the account ungranted"
        )
    user, _, host = user_host.rpartition("@")
    if _SSH_USER_RE.fullmatch(user) is None:
        raise RemoteExecDenial(f"the remote user '{user}' is not a valid account name")
    if not host:
        raise RemoteExecDenial("the remote operand names no host")
    address = _parse_ip_literal(host)
    if address is not None:
        host = _address_spelling(address)
    elif _SSH_HOSTNAME_RE.fullmatch(host) is not None:
        host = host.lower()
    else:
        raise RemoteExecDenial(
            f"the remote host '{host}' is not a valid host spelling "
            "(IP literal, bracketed IPv6, or DNS name)"
        )
    if not remote_path:
        raise RemoteExecDenial(
            f"the remote operand '{operand}' names an empty remote path: "
            "server-chosen names in the default directory are not bounded"
        )
    if _REMOTE_PATH_RE.fullmatch(remote_path) is None:
        raise RemoteExecDenial(
            f"the remote path '{remote_path}' carries characters outside the "
            "closed transfer path grammar (globs, whitespace, and shell "
            "metacharacters are server-side expansion and are denied)"
        )
    if ".." in remote_path.split("/"):
        raise RemoteExecDenial(
            f"the remote path '{remote_path}' contains a '..' segment and is denied"
        )
    return user, host, remote_path


def _check_local_operand(operand: str) -> None:
    """Bound one local operand: no glob metacharacters (the pinned argv runs
    without a shell expansion, and scp/rsync must not glob server-side).
    """
    if not operand:
        raise RemoteExecDenial("a transfer operand is empty")
    if any(char in operand for char in _GLOB_CHARACTERS):
        raise RemoteExecDenial(
            f"the transfer operand '{operand}' carries glob metacharacters "
            "and is denied: the pinned argv never expands names"
        )


def _parse_scp_options(  # noqa: PLR0912, PLR0915
    tokens: list[str], index: int, option_tokens: list[str], state: dict
) -> tuple[int, list[str]]:
    """Parse scp option tokens from ``index``.

    Returns the operand-start index and the DEFERRED denial reasons (value
    consumption still happens, so operands are correct): a denial fires only
    for a remote-bearing invocation, because a local scp copy is governed by
    the path layer, not the transfer grammar.
    """
    deferred: list[str] = []
    while index < len(tokens):
        token = tokens[index]
        if token == "--":
            return index + 1, deferred
        if not token.startswith("-") or token == "-":
            return index, deferred
        if token.startswith("--"):
            deferred.append(
                f"the unmodeled scp long option '{token}' is denied fail-closed"
            )
            index += 1
            continue
        chars = token[1:]
        consumed_value = False
        for offset, char in enumerate(chars):
            if char in _SCP_SAFE_SHORT:
                continue
            if char == "P":
                attached = chars[offset + 1 :]
                if attached:
                    try:
                        state["port"] = _parse_port(attached)
                        option_tokens.append(token)
                    except RemoteExecDenial as exc:
                        deferred.append(str(exc))
                elif index + 1 >= len(tokens):
                    deferred.append("the -P option is missing its port")
                else:
                    try:
                        state["port"] = _parse_port(tokens[index + 1])
                        option_tokens.extend(["-P", tokens[index + 1]])
                    except RemoteExecDenial as exc:
                        deferred.append(str(exc))
                    index += 1
                consumed_value = True
                break
            if char == "o":
                attached = chars[offset + 1 :]
                if attached:
                    value = attached
                else:
                    if index + 1 >= len(tokens):
                        deferred.append("the -o option is missing its value")
                        consumed_value = True
                        break
                    value = tokens[index + 1]
                    index += 1
                try:
                    _check_ssh_o_value(value, option_tokens)
                except RemoteExecDenial as exc:
                    deferred.append(str(exc))
                consumed_value = True
                break
            reason = _SCP_DENIED_SHORT_REASONS.get(char)
            if reason is not None:
                deferred.append(f"the scp option '-{char}' is denied: {reason}")
            else:
                deferred.append(
                    f"the unmodeled scp option '{token}' is denied fail-closed"
                )
            consumed_value = True
            break
        if not consumed_value:
            option_tokens.append(token)
        index += 1
    return index, deferred


def _parse_rsync_options(  # noqa: PLR0912
    tokens: list[str], index: int, option_tokens: list[str]
) -> tuple[int, list[str]]:
    """Parse rsync option tokens from ``index``.

    Returns the operand-start index and the DEFERRED denial reasons (value
    consumption still happens, so operands are correct): a denial fires only
    for a remote-bearing invocation, because a local rsync sync is governed
    by the path layer, not the transfer grammar.
    """
    deferred: list[str] = []
    while index < len(tokens):
        token = tokens[index]
        if token == "--":
            return index + 1, deferred
        if not token.startswith("-") or token == "-":
            return index, deferred
        if token.startswith("--"):
            name, sep, _attached = token.partition("=")
            if name in _RSYNC_SAFE_LONG and not sep:
                option_tokens.append(token)
                index += 1
                continue
            reason = _RSYNC_DENIED_LONG_REASONS.get(name)
            if reason is None and (
                name in {"--delete", "--delete-excluded", "--delete-missing-args"}
                or name.startswith("--delete-")
                or name.startswith("--backup")
                or name in {"--partial", "--partial-dir", "--temp-dir", "--suffix"}
            ):
                reason = (
                    f"the rsync option '{name}' mutates the destination beyond "
                    "a copy or keeps server-chosen names and is denied"
                )
            if reason is None:
                reason = (
                    f"the unmodeled rsync long option '{token}' is denied fail-closed"
                )
            deferred.append(f"the rsync option '{name}' is denied: {reason}")
            index += 1
            continue
        chars = token[1:]
        denied = False
        for offset, char in enumerate(chars):
            if char in _RSYNC_SAFE_SHORT:
                continue
            if char == "e":
                # The transport override consumes its value so it never
                # surfaces as an operand. A value that is exactly a pinned
                # ssh transport is the plan's own pin (re-parse), not an
                # override.
                attached = chars[offset + 1 :]
                if attached:
                    value = attached
                elif index + 1 >= len(tokens):
                    value = ""
                else:
                    value = tokens[index + 1]
                    index += 1
                try:
                    parts = shlex.split(value)
                except ValueError:
                    parts = []
                if (
                    parts
                    and _command_name(parts[0]) == "ssh"
                    and _pinned_alias_key(parts[1:]) is not None
                ):
                    option_tokens.extend(["-e", value])
                else:
                    deferred.append(
                        f"the rsync option '-e' is denied: {_RSYNC_DENIED_SHORT_REASONS['e']}"
                    )
            elif char in _RSYNC_DENIED_SHORT_REASONS:
                deferred.append(
                    f"the rsync option '-{char}' is denied: "
                    f"{_RSYNC_DENIED_SHORT_REASONS[char]}"
                )
            else:
                deferred.append(
                    f"the unmodeled rsync option '{token}' is denied fail-closed"
                )
            denied = True
            break
        if not denied:
            option_tokens.append(token)
        index += 1
    return index, deferred


def _rsync_pinned_transport(option_tokens: Iterable[str]) -> str | None:
    """The destination key a pinned rsync ``-e`` ssh transport carries.

    Returns the canonical destination key when the ``-e`` value is exactly
    the pinned ssh transport (the four pinned ``-o`` options), else None.
    """
    tokens = [str(token) for token in option_tokens]
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "-e" and index + 1 < len(tokens):
            value = tokens[index + 1]
            index += 2
        elif token.startswith("-e") and len(token) > _ATTACHED_SHORT_VALUE_OFFSET:
            value = token[_ATTACHED_SHORT_VALUE_OFFSET:]
            index += 1
        else:
            index += 1
            continue
        try:
            parts = shlex.split(value)
        except ValueError:
            return None
        if not parts or _command_name(parts[0]) != "ssh":
            return None
        return _pinned_alias_key(parts[1:])
    return None


def _pinned_alias_key(ssh_option_tokens: list[str]) -> str | None:
    """The canonical destination key the pinned ssh options carry, or None."""
    destination = _pinned_alias_destination(ssh_option_tokens)
    return destination.key if destination is not None else None


def _check_pinned_option_values(option_tokens: Iterable[str], key: str) -> None:
    """Reject a pinned argv carrying any pinned option with a foreign value."""
    option_map = _option_map_from_tokens(option_tokens)
    expected = {
        "UserKnownHostsFile": PINNED_KNOWN_HOSTS_FILE,
        "HostKeyAlias": key,
        "StrictHostKeyChecking": "yes",
        "BatchMode": "yes",
    }
    for option_key, expected_value in expected.items():
        if option_key in option_map and option_map[option_key] != expected_value:
            raise RemoteExecDenial(
                f"the ssh option '{option_key}' is pinned by the remote-exec "
                f"transport and cannot be overridden (denied value "
                f"'{option_map[option_key]}')"
            )


def parse_transfer_call(  # noqa: PLR0914
    tokens: Iterable[str],
) -> TransferCall | None:
    """Parse one scp/rsync argv inside the closed grammar.

    Returns None when the invocation names no remote operand (a local copy,
    governed by the ordinary path layer — not a transfer). Raises
    :class:`RemoteExecDenial` for every remote-bearing form outside the
    grammar. An argv already carrying the full transport pin parses as that
    pin's granted destination with the resolved address as the remote
    operand's host.
    """
    tokens = [str(token) for token in tokens]
    client = _command_name(tokens[0]) if tokens else ""
    if client not in TRANSFER_CLIENTS:
        raise RemoteExecDenial(f"not a {TRANSFER_CLIENTS} invocation")
    option_tokens: list[str] = []
    state: dict = {"port": None}
    if client == "scp":
        index, deferred = _parse_scp_options(tokens, 1, option_tokens, state)
    else:
        index, deferred = _parse_rsync_options(tokens, 1, option_tokens)
    operands = tokens[index:]
    remote_indices = [
        position
        for position, operand in enumerate(operands)
        if _remote_separator(operand) >= 0
    ]
    if not remote_indices:
        # No remote operand: a local copy, not a transfer — the ordinary path
        # layer governs it, and the option denials above do not fire.
        return None
    if deferred:
        raise RemoteExecDenial(deferred[0])
    if any(token.startswith("-") for token in operands):
        raise RemoteExecDenial(
            "a transfer operand starting with '-' is denied: it would be "
            "parsed as an unmodeled option"
        )
    if len(remote_indices) > 1:
        raise RemoteExecDenial(
            "a remote-to-remote transfer (two remote operands) is denied: a "
            "bounded transfer copies between this machine and one granted "
            "destination only"
        )
    if len(operands) != _TRANSFER_OPERAND_COUNT:
        raise RemoteExecDenial(
            f"the {client} invocation admits exactly one local and one remote "
            f"operand (got {len(operands)} operands): the transfer footprint "
            "must stay bounded"
        )
    remote_index = remote_indices[0]
    user, host, remote_path = _parse_remote_operand(operands[remote_index])
    local_operand = operands[1 - remote_index]
    _check_local_operand(local_operand)
    port = state["port"] if state["port"] is not None else 22
    alias_destination = None
    if client == "scp":
        alias_destination = _pinned_alias_destination(option_tokens)
    else:
        key = _rsync_pinned_transport(option_tokens)
        alias_destination = (
            canonical_remote_destination(key) if key is not None else None
        )
    if alias_destination is not None:
        # A fully pinned argv: the remote operand's host must be the pinned
        # resolved address (an IP literal) and must name the same user and
        # port as the granted destination the pin carries. classify
        # revalidates that the literal is the granted host's current unique
        # non-local resolution, so a hand-written pin cannot point at an
        # unvalidated address.
        if _parse_ip_literal(host) is None:
            raise RemoteExecDenial(
                "a pinned transfer argv must carry the resolved address as "
                f"its remote operand host, not '{host}'"
            )
        if alias_destination.user != user:
            raise RemoteExecDenial(
                "the pinned HostKeyAlias names another user than the remote operand"
            )
        if alias_destination.port != port:
            raise RemoteExecDenial(
                "the pinned HostKeyAlias names another port than the invocation"
            )
        if client == "scp":
            _check_pinned_option_values(option_tokens, alias_destination.key)
        return TransferCall(
            client=client,
            option_tokens=tuple(option_tokens),
            user=user,
            host=alias_destination.host,
            port=port,
            remote_path=remote_path,
            operands=(operands[0], operands[1]),
            remote_index=remote_index,
            resolved_host=host,
        )
    call = TransferCall(
        client=client,
        option_tokens=tuple(option_tokens),
        user=user,
        host=host,
        port=port,
        remote_path=remote_path,
        operands=(operands[0], operands[1]),
        remote_index=remote_index,
    )
    # Transport-pinned options are admissible only with exactly the pinned
    # values (this is also what makes re-classifying an already-pinned argv
    # succeed); any other value is an override attempt.
    _check_pinned_option_values(option_tokens, call.destination.key)
    return call


def _classify_transfer_destination(
    call: TransferCall, enum: Callable[[], frozenset], res: Callable[[str], frozenset]
) -> RemoteExecClassification:
    """Classify one parsed transfer's destination by locality and resolution."""
    if _parse_ip_literal(call.host) is not None:
        if classify_local_address(call.host, interface_addresses=enum) == "local":
            return RemoteExecClassification(
                "deny",
                reason=(
                    f"the transfer destination address '{call.host}' is local "
                    "or ambiguous (loopback, private, mapped, interface-owned, "
                    "or an unestablishable locality) and is denied"
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


def classify_transfer_call(
    tokens: Iterable[str],
    *,
    interface_addresses: Callable[[], frozenset] | None = None,
    resolver: Callable[[str], frozenset] | None = None,
) -> RemoteExecClassification:
    """Classify one command's argv for the closed scp/rsync transfer grammar.

    ``routine`` for non-transfer commands AND for scp/rsync invocations with
    no remote operand (local copies — the path layer governs them);
    ``deny`` with a reason for every refused remote-bearing form; ``ask``
    for a positive destination — an IP literal the fail-closed locality
    classifier proves non-local, or a hostname whose system-resolver answer
    is unique and non-local — carrying the exact granted destination and the
    parsed call with the resolved address pinned in.
    """
    tokens = [str(token) for token in tokens]
    if not tokens or _command_name(tokens[0]) not in TRANSFER_CLIENTS:
        return RemoteExecClassification("routine")
    enum = interface_addresses or _rx._interface_addresses
    res = resolver or _rx._hostname_resolver
    try:
        call = parse_transfer_call(tokens)
    except RemoteExecDenial as exc:
        return RemoteExecClassification("deny", reason=str(exc))
    if call is None:
        return RemoteExecClassification("routine")
    if call.resolved_host is not None:
        reason = _revalidate_pinned_destination(
            call.host, call.resolved_host, call.destination, enum, res
        )
        if reason is not None:
            return RemoteExecClassification(
                "deny", reason=reason, destination=call.destination, call=call
            )
        return RemoteExecClassification("ask", destination=call.destination, call=call)
    return _classify_transfer_destination(call, enum, res)


def classify_remote_exec_call(
    tokens: Iterable[str],
    *,
    interface_addresses: Callable[[], frozenset] | None = None,
    resolver: Callable[[str], frozenset] | None = None,
) -> RemoteExecClassification:
    """Dispatch one command's argv to the closed remote-exec grammars.

    ``ssh`` routes to the ssh grammar (S4b1b/S4b2a/S4b2b); ``scp``/``rsync``
    route to the bounded transfer grammar (S4b2c); anything else is
    ``routine``.
    """
    tokens = [str(token) for token in tokens]
    if not tokens:
        return RemoteExecClassification("routine")
    name = _command_name(tokens[0])
    if name == "ssh":
        return classify_ssh_call(
            tokens, interface_addresses=interface_addresses, resolver=resolver
        )
    if name in TRANSFER_CLIENTS:
        return classify_transfer_call(
            tokens, interface_addresses=interface_addresses, resolver=resolver
        )
    return RemoteExecClassification("routine")


def _pin_option_tokens(destination: RemoteDestination) -> list[str]:
    """The leading pinned ``-o`` options every pinned argv carries."""
    return [
        "-o",
        f"UserKnownHostsFile={PINNED_KNOWN_HOSTS_FILE}",
        "-o",
        f"HostKeyAlias={destination.key}",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        "BatchMode=yes",
    ]


def _pinned_ssh_transport_text(destination: RemoteDestination, port: int) -> str:
    """The pinned ssh transport rsync's ``-e`` value carries."""
    parts = ["ssh", *_pin_option_tokens(destination)]
    if port != _DEFAULT_SSH_PORT:
        parts.extend(["-p", str(port)])
    return shlex.join(parts)


def pinned_transfer_argv(call: TransferCall) -> tuple[str, ...]:
    """The actual spawn argv for one admitted transfer.

    The pin leads the argv (first obtained value wins, so the pin cannot be
    overridden), the remote operand is the pinned destination operand — for
    a hostname destination the RESOLVED address while the ``HostKeyAlias``
    pin stays bound to the granted hostname — and the local operand rides
    unchanged (the path layer already gated it). scp carries the pin as
    leading ``-o`` options; rsync carries it as the injected ``-e`` pinned
    ssh transport (a user ``-e`` is denied at parse, so the pin is the only
    one). Re-pinning an already-pinned call is idempotent.
    """
    if call.resolved_host is None and _parse_ip_literal(call.host) is None:
        raise RemoteExecDenial(
            f"the {call.client} destination '{call.host}' was not resolved, so "
            "no pinned address exists to spawn"
        )
    destination = call.destination
    options = list(call.option_tokens)
    if call.client == "scp":
        if call.argv_is_pinned:
            filtered: list[str] = []
            index = 0
            while index < len(options):
                if options[index] == "-o" and index + 1 < len(options):
                    key = options[index + 1].partition("=")[0]
                    if key in {
                        "UserKnownHostsFile",
                        "HostKeyAlias",
                        "StrictHostKeyChecking",
                        "BatchMode",
                    }:
                        index += 2
                        continue
                filtered.append(options[index])
                index += 1
            options = filtered
        argv = [call.client, *_pin_option_tokens(destination), *options]
    else:
        argv = [
            call.client,
            "-e",
            _pinned_ssh_transport_text(destination, call.port),
            *options,
        ]
    pinned_operands = list(call.operands)
    pinned_operands[call.remote_index] = call.destination_operand
    argv.extend(pinned_operands)
    return tuple(argv)


def pinned_transfer_command_text(call: TransferCall) -> str:
    """The pinned argv as shell text for string transports (shlex-joined)."""
    return shlex.join(pinned_transfer_argv(call))


def enforce_pinned_transfer_resolution_at_spawn(call: TransferCall) -> TransferCall:
    """S4b2b rebinding guard at the spawn for one transfer call.

    The validated pinned address is re-observed once inside the same plan
    transaction and any CHANGED answer — a rebind to a local/loopback/
    unspecified address, to another address, ambiguity, or a resolution
    failure — raises, so the changed answer is never contacted.
    """
    enforce_pinned_resolution(call.host, call.resolved_host)
    return call
