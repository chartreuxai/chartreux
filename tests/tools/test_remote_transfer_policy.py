"""Bounded transfer policy for scp/rsync (v0.7.1 S4b2c).

scp/rsync left the shipped denylist WITH the closed transfer grammar in
_remote_transfer.py, classified at the same resolver seam as ssh
(classify_remote_exec_call). The corpus below pins both sides of the flip:

- POSITIVE bounded transfers classify ``ask`` as ``network.remote-exec`` with
  the exact destination on the proposal (the same destination-keyed grant
  model as ssh): supported options, remote path forms, upload and download
  directions, port and IPv6 forms, and hostname destinations resolved
  uniquely non-locally (pinned).
- NEGATIVE remote-bearing spellings hard-deny with a discriminating reason
  and NO proposal: no user in the remote spec, remote-to-remote, the rsync
  daemon form, server-chosen names (globs, whitespace, ``..``, empty remote
  path), transport overrides, legacy-protocol/strict-name flags,
  symlink-following, destination-side mutation, local/ambiguous/failed
  resolutions, hostile/unmodeled options, operand-count overflow, and
  composed or wrapped calls. A scp/rsync call with NO remote operand is
  routine (the path layer governs the local copy).
- Outside-workspace writes are gated by the path layer BEFORE any transfer
  authority is considered.
- The execution plan pins the approved address + fixed host-key identity
  into the actual argv (scp: leading -o pins; rsync: the injected -e pinned
  ssh transport), and the S4b2b rebinding guard denies a changed resolution
  inside the same plan transaction (HHH: deny-on-change or
  execute-the-original-pinned-address are both conforming; the changed
  answer never rides).

Every arm runs against fixed hermetic seams (no real DNS, no host
interfaces), mirroring the corpus observer's fixed-answer table.
"""

from __future__ import annotations

import ipaddress
from pathlib import Path
import socket
import subprocess

import pytest

from chartreux.core.dangerous_commands import (
    REMOTE_EXEC_OPERATION_ID,
    DangerousCommandAuthority,
    RemoteExecAttestation,
    authority_admits,
    canonical_remote_destination,
)
from chartreux.core.tools.base import BaseToolState, ToolPermission
from chartreux.core.tools.builtins import _remote_exec
from chartreux.core.tools.builtins._remote_exec import RemoteExecDenial
from chartreux.core.tools.builtins._remote_transfer import (
    TransferCall,
    classify_remote_exec_call,
    classify_transfer_call,
    parse_transfer_call,
    pinned_transfer_argv,
    pinned_transfer_command_text,
)
from chartreux.core.tools.builtins._shell_command_policy import (
    GitRepositoryRelocation,
    resolve_repository_scope,
)
from chartreux.core.tools.builtins._shell_permission_resolver import (
    PermissionContext,
    remote_exec_plan_for_command,
)
from chartreux.core.tools.builtins.bash import Bash, BashArgs, BashToolConfig
from chartreux.core.tools.utils import effective_scratchpad_roots

#: The fixed hermetic resolution table (the corpus observer's table): a
#: unique non-local answer, an ambiguous answer, a local answer, and failure.
FIXED_HOSTNAME_RESOLUTIONS: dict[str, tuple[str, ...]] = {
    "example.com": ("93.184.216.34",),
    "ambiguous.example": ("93.184.216.34", "1.1.1.1"),
    "local.example": ("127.0.0.1",),
}

FIXED_INTERFACES = frozenset({
    ipaddress.ip_address("192.168.10.20"),
    ipaddress.ip_address("::1"),
})


def fixed_hostname_resolver(host: str) -> frozenset:
    answers = FIXED_HOSTNAME_RESOLUTIONS.get(host.lower())
    if answers is None:
        raise socket.gaierror(-2, "Name or service not known")
    return frozenset(ipaddress.ip_address(text) for text in answers)


@pytest.fixture
def hermetic_seams(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the locality enumeration and the hostname resolver to fixed sets."""

    def fake_getaddrinfo(host, port):
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.168.10.20", 0)),
            (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("::1", 0, 0, 0)),
        ]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    monkeypatch.setattr(_remote_exec, "_interface_addresses", lambda: FIXED_INTERFACES)
    monkeypatch.setattr(_remote_exec, "_hostname_resolver", fixed_hostname_resolver)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    path = tmp_path / "repo"
    path.mkdir()
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    return path


def _tool(repo: Path, *authorities: DangerousCommandAuthority) -> Bash:
    tool = Bash(config_getter=lambda: BashToolConfig(), state=BaseToolState(), cwd=repo)
    tool.dangerous_command_authorities_getter = lambda: tuple(authorities)
    return tool


def _grant(repo: Path, destination: str) -> DangerousCommandAuthority:
    scope = resolve_repository_scope(repo, GitRepositoryRelocation())
    assert scope is not None
    parsed = canonical_remote_destination(destination)
    assert parsed is not None
    return DangerousCommandAuthority(
        operations=frozenset({REMOTE_EXEC_OPERATION_ID}),
        scope=scope,
        source="runtime_grant",
        destination=parsed,
    )


def _plan(tool: Bash, command: str):
    return remote_exec_plan_for_command(
        command,
        cwd=tool.cwd,
        workspace=tool.workspace,
        authority=tool.path_authority,
        scratchpad_roots=effective_scratchpad_roots(tool.path_authority),
        workspace_roots=frozenset(tool.workspace.authorized_roots),
    )


# ---------------------------------------------------------------------------
# Grammar classification corpus (hermetic seams)
# ---------------------------------------------------------------------------

_POSITIVE_TRANSFERS: list[tuple[list[str], str]] = [
    # (argv, expected destination key)
    (["scp", "file", "deploy@8.8.8.8:backup"], "deploy@8.8.8.8:22"),
    (["scp", "-r", "-p", "dir/", "deploy@8.8.8.8:bak/"], "deploy@8.8.8.8:22"),
    (["scp", "-rp", "dir/", "deploy@8.8.8.8:bak/"], "deploy@8.8.8.8:22"),
    (["scp", "-C", "-q", "f", "deploy@1.1.1.1:x"], "deploy@1.1.1.1:22"),
    (["scp", "-P", "2222", "f", "deploy@8.8.8.8:x"], "deploy@8.8.8.8:2222"),
    (["scp", "-P2222", "f", "deploy@8.8.8.8:x"], "deploy@8.8.8.8:2222"),
    (["scp", "-o", "ConnectTimeout=5", "f", "deploy@8.8.8.8:x"], "deploy@8.8.8.8:22"),
    (["scp", "deploy@8.8.8.8:bak", "local"], "deploy@8.8.8.8:22"),
    (
        ["scp", "deploy@[2606:4700:4700::1111]:x", "l"],
        "deploy@[2606:4700:4700::1111]:22",
    ),
    (["scp", "--", "f", "deploy@8.8.8.8:x"], "deploy@8.8.8.8:22"),
    (["rsync", "-a", "src/", "deploy@8.8.8.8:dst/"], "deploy@8.8.8.8:22"),
    (["rsync", "-z", "deploy@8.8.8.8:src/", "dst/"], "deploy@8.8.8.8:22"),
    (["rsync", "--archive", "--dry-run", "s", "deploy@8.8.8.8:d"], "deploy@8.8.8.8:22"),
    (["rsync", "-a", "--progress", "s", "deploy@8.8.8.8:d"], "deploy@8.8.8.8:22"),
    # Hostname destinations resolve uniquely non-locally and pin the answer.
    (["scp", "f", "deploy@example.com:x"], "deploy@example.com:22"),
    (["rsync", "-a", "s", "deploy@example.com:d"], "deploy@example.com:22"),
]


@pytest.mark.parametrize(("argv", "destination_key"), _POSITIVE_TRANSFERS)
def test_positive_bounded_transfers_ask_with_the_destination(
    hermetic_seams, argv, destination_key
):
    classification = classify_transfer_call(argv)
    assert classification.kind == "ask", classification.reason
    assert classification.destination is not None
    assert classification.destination.key == destination_key
    call = classification.call
    assert isinstance(call, TransferCall)
    assert call.client in ("scp", "rsync")
    # The pinned argv carries the approved address and the fixed host-key
    # identity; a hostname destination pins the RESOLVED address as the
    # remote operand's host while HostKeyAlias stays on the granted hostname.
    pinned = pinned_transfer_argv(call)
    assert pinned[0] == call.client
    if call.host == "example.com":
        assert f"deploy@{FIXED_HOSTNAME_RESOLUTIONS['example.com'][0]}:" in pinned[
            -2
        ] or (f"deploy@{FIXED_HOSTNAME_RESOLUTIONS['example.com'][0]}:" in pinned[-1])
    text = pinned_transfer_command_text(call)
    assert "UserKnownHostsFile=/dev/null/chartreux-pinned-known-hosts" in text
    assert f"HostKeyAlias={destination_key}" in text
    assert "StrictHostKeyChecking=yes" in text
    assert "BatchMode=yes" in text
    if call.client == "rsync":
        assert pinned[1] == "-e"
        assert pinned[2].startswith("ssh -o UserKnownHostsFile=")


_NEGATIVE_TRANSFERS: list[tuple[list[str], str]] = [
    # No user in the remote spec: the account is ungranted.
    (["scp", "file", "host:backup"], "must name a user"),
    (["rsync", "-a", "src/", "host:dst/"], "must name a user"),
    # Remote-to-remote.
    (["scp", "a@8.8.8.8:x", "b@9.9.9.9:y"], "remote-to-remote"),
    (["scp", "-3", "a@8.8.8.8:x", "b@9.9.9.9:y"], "remote-to-remote"),
    (["rsync", "a@8.8.8.8:x", "b@9.9.9.9:y"], "remote-to-remote"),
    # rsync daemon module form bypasses the pinned ssh transport.
    (["rsync", "a", "deploy@8.8.8.8::mod"], "daemon module form"),
    # Server-selected names / remote-side expansion.
    (["scp", "f", "deploy@8.8.8.8:"], "empty remote path"),
    (["scp", "f", "deploy@8.8.8.8:*.py"], "closed transfer path grammar"),
    (["scp", "f", "deploy@8.8.8.8:../etc"], "'..' segment"),
    (["scp", "f", "deploy@8.8.8.8:a b"], "closed transfer path grammar"),
    (["scp", "*.py", "deploy@8.8.8.8:x"], "glob metacharacters"),
    # Transport overrides and trust-context options.
    (["scp", "-S", "evil", "f", "deploy@8.8.8.8:x"], "custom ssh transport"),
    (["scp", "-F", "/tmp/c", "f", "deploy@8.8.8.8:x"], "user-controlled ssh config"),
    (["scp", "-i", "/tmp/id", "f", "deploy@8.8.8.8:x"], "identity file"),
    (["scp", "-J", "jump", "f", "deploy@8.8.8.8:x"], "proxy-jump"),
    (["scp", "-c", "aes128-ctr", "f", "deploy@8.8.8.8:x"], "transport override"),
    (["rsync", "-e", "ssh -p 2222", "a", "deploy@8.8.8.8:b"], "remote transport"),
    (["rsync", "--rsh=evil", "a", "deploy@8.8.8.8:b"], "remote transport"),
    (
        ["rsync", "--rsync-path=evil", "a", "deploy@8.8.8.8:b"],
        "runs a command on the remote",
    ),
    # Legacy protocol / strict-name-check disable: server-chosen names.
    (["scp", "-O", "f", "deploy@8.8.8.8:x"], "legacy scp protocol"),
    (["scp", "-T", "f", "deploy@8.8.8.8:x"], "strict filename checking"),
    # Symlink following.
    (["rsync", "-L", "a", "deploy@8.8.8.8:b"], "follows symlinks"),
    (["rsync", "--copy-links", "a", "deploy@8.8.8.8:b"], "follows symlinks"),
    (["rsync", "--copy-unsafe-links", "a", "deploy@8.8.8.8:b"], "follows symlinks"),
    # Destination-side mutation beyond a copy.
    (["rsync", "--delete", "a", "deploy@8.8.8.8:b"], "mutates the destination"),
    (["rsync", "--remove-source-files", "a", "deploy@8.8.8.8:b"], "mutates the source"),
    (["rsync", "--inplace", "a", "deploy@8.8.8.8:b"], "--inplace"),
    (["rsync", "-P", "a", "deploy@8.8.8.8:b"], "server-chosen temporary names"),
    (["rsync", "--backup", "a", "deploy@8.8.8.8:b"], "mutates the destination"),
    # Hostile / pinned-override / unmodeled options.
    (
        ["scp", "-o", "ProxyCommand=nc x 1", "f", "deploy@8.8.8.8:x"],
        "hostile ssh option",
    ),
    (
        ["scp", "-o", "UserKnownHostsFile=/tmp/evil", "f", "deploy@8.8.8.8:x"],
        "pinned by the remote-exec transport",
    ),
    (["scp", "-Z", "f", "deploy@8.8.8.8:x"], "unmodeled scp option"),
    (["rsync", "--bwlimit=1", "a", "deploy@8.8.8.8:b"], "unmodeled rsync long option"),
    (["rsync", "-W", "a", "deploy@8.8.8.8:b"], "unmodeled rsync option"),
    # Footprint overflow.
    (["scp", "a", "b", "deploy@8.8.8.8:x"], "exactly one local and one remote"),
    (["rsync", "a", "b", "deploy@8.8.8.8:d"], "exactly one local and one remote"),
    # Local / ambiguous / failed resolutions.
    (["scp", "deploy@127.0.0.1:x", "l"], "is local or ambiguous"),
    (["scp", "f", "deploy@0.0.0.0:x"], "is local or ambiguous"),
    (["scp", "f", "deploy@10.0.0.1:x"], "is local or ambiguous"),
    (["scp", "f", "deploy@[::ffff:8.8.8.8]:x"], "is local or ambiguous"),
    (["scp", "f", "deploy@192.168.10.20:x"], "is local or ambiguous"),
    (["scp", "f", "deploy@ambiguous.example:x"], "multiple addresses"),
    (["scp", "f", "deploy@local.example:x"], "local address"),
    (["scp", "f", "deploy@unresolved.example:x"], "could not be resolved"),
]


@pytest.mark.parametrize(("argv", "reason_fragment"), _NEGATIVE_TRANSFERS)
def test_negative_remote_spellings_hard_deny_with_a_reason(
    hermetic_seams, argv, reason_fragment
):
    classification = classify_transfer_call(argv)
    assert classification.kind == "deny"
    assert classification.reason is not None
    assert reason_fragment in classification.reason
    assert classification.destination is None or classification.kind == "deny"


@pytest.mark.parametrize(
    "argv",
    [
        ["scp", "a", "b"],
        ["rsync", "-a", "src/", "dst/"],
        ["rsync", "--delete", "a", "b"],
        ["scp", "-r", ".", "copy/"],
    ],
)
def test_local_scp_rsync_without_a_remote_operand_is_routine(hermetic_seams, argv):
    # A local copy is not a transfer: the ordinary path layer governs it.
    assert classify_transfer_call(argv).kind == "routine"
    assert classify_remote_exec_call(argv).kind == "routine"


def test_dispatcher_routes_ssh_and_returns_routine_otherwise(hermetic_seams):
    assert classify_remote_exec_call(["ssh", "deploy@8.8.8.8", "uptime"]).kind == "ask"
    assert classify_remote_exec_call(["scp", "f", "deploy@8.8.8.8:x"]).kind == "ask"
    assert classify_remote_exec_call(["ls", "-la"]).kind == "routine"


# ---------------------------------------------------------------------------
# Permission-level flow (resolver seam)
# ---------------------------------------------------------------------------


class TestTransferPermissionFlow:
    def test_positive_transfer_asks_with_network_remote_exec_and_destination(
        self, hermetic_seams, repo
    ):
        tool = _tool(repo)
        permission = tool.resolve_permission(
            BashArgs(command="scp file deploy@8.8.8.8:backup")
        )
        assert isinstance(permission, PermissionContext)
        assert permission.permission is ToolPermission.NEVER
        assert permission.denial_kind == "dangerous_command"
        assert "network.remote-exec" in (permission.reason or "")
        proposal = permission.dangerous_command_proposal
        assert proposal is not None
        assert proposal.operations == ("network.remote-exec",)
        assert proposal.destination is not None
        assert proposal.destination.key == "deploy@8.8.8.8:22"

    def test_rsync_upload_asks_with_the_destination(self, hermetic_seams, repo):
        tool = _tool(repo)
        permission = tool.resolve_permission(
            BashArgs(command="rsync -a src/ deploy@8.8.8.8:dst/")
        )
        assert isinstance(permission, PermissionContext)
        assert permission.denial_kind == "dangerous_command"
        proposal = permission.dangerous_command_proposal
        assert proposal is not None
        assert proposal.destination is not None
        assert proposal.destination.key == "deploy@8.8.8.8:22"

    def test_port_form_attaches_the_full_destination_key(self, hermetic_seams, repo):
        tool = _tool(repo)
        permission = tool.resolve_permission(
            BashArgs(command="scp -P 2222 f deploy@8.8.8.8:x")
        )
        assert isinstance(permission, PermissionContext)
        proposal = permission.dangerous_command_proposal
        assert proposal is not None and proposal.destination is not None
        assert proposal.destination.key == "deploy@8.8.8.8:2222"

    def test_hostname_transfer_asks_with_the_hostname_destination(
        self, hermetic_seams, repo
    ):
        tool = _tool(repo)
        permission = tool.resolve_permission(
            BashArgs(command="rsync -a s deploy@example.com:d")
        )
        assert isinstance(permission, PermissionContext)
        proposal = permission.dangerous_command_proposal
        assert proposal is not None and proposal.destination is not None
        assert proposal.destination.key == "deploy@example.com:22"

    def test_exact_destination_grant_admits_the_transfer(self, hermetic_seams, repo):
        tool = _tool(repo, _grant(repo, "deploy@8.8.8.8"))
        result = tool.resolve_permission(
            BashArgs(command="scp file deploy@8.8.8.8:backup")
        )
        assert result is None or result.permission is ToolPermission.ALWAYS

    def test_near_miss_destination_grant_denies_the_transfer(
        self, hermetic_seams, repo
    ):
        for other in ("deploy@1.1.1.1", "deploy@8.8.8.8:2222", "root@8.8.8.8"):
            tool = _tool(repo, _grant(repo, other))
            permission = tool.resolve_permission(
                BashArgs(command="scp file deploy@8.8.8.8:backup")
            )
            assert isinstance(permission, PermissionContext)
            assert permission.permission is ToolPermission.NEVER
            assert permission.denial_kind == "dangerous_command"

    @pytest.mark.parametrize(
        "command",
        [
            "scp file host:backup",
            "rsync -a src/ host:dst/",
            "scp a@8.8.8.8:x b@9.9.9.9:y",
            "rsync -e ssh a deploy@8.8.8.8:b",
            "rsync --rsync-path=evil a deploy@8.8.8.8:b",
            "rsync -L a deploy@8.8.8.8:b",
            "scp -O f deploy@8.8.8.8:x",
            "scp -T f deploy@8.8.8.8:x",
            "scp -S evil f deploy@8.8.8.8:x",
            "scp f deploy@8.8.8.8:*.py",
            "scp f deploy@8.8.8.8:",
            "scp deploy@127.0.0.1:x l",
            "scp f deploy@ambiguous.example:x",
            "scp f deploy@unresolved.example:x",
            "scp f deploy@8.8.8.8:x && echo done",
            "nice scp f deploy@8.8.8.8:x",
            "scp a b deploy@8.8.8.8:x",
        ],
    )
    def test_denied_forms_hard_deny_without_proposal_even_with_a_grant(
        self, hermetic_seams, repo, command
    ):
        tool = _tool(repo, _grant(repo, "deploy@8.8.8.8"))
        permission = tool.resolve_permission(BashArgs(command=command))
        assert isinstance(permission, PermissionContext)
        assert permission.permission is ToolPermission.NEVER
        assert permission.denial_kind is None
        assert permission.dangerous_command_proposal is None
        assert "matches denylist pattern" not in (permission.reason or "")

    def test_outside_workspace_write_is_denied_by_the_path_layer_first(
        self, hermetic_seams, repo
    ):
        # The download target outside the workspace is a path denial, gated
        # before any transfer authority is considered.
        tool = _tool(repo, _grant(repo, "deploy@8.8.8.8"))
        permission = tool.resolve_permission(
            BashArgs(command="scp deploy@8.8.8.8:backup /etc/passwd")
        )
        assert isinstance(permission, PermissionContext)
        assert permission.permission is ToolPermission.NEVER
        assert permission.denial_kind == "out_of_root"
        assert permission.dangerous_command_proposal is None
        assert "outside the authorized workspace" in (permission.reason or "")

    def test_local_copy_is_not_a_transfer_at_the_permission_seam(
        self, hermetic_seams, repo
    ):
        tool = _tool(repo)
        result = tool.resolve_permission(BashArgs(command="rsync -a src/ dst/"))
        assert result is None or result.permission is not ToolPermission.NEVER


class TestTransferAuthoritySemantics:
    def test_exact_destination_admitted_near_miss_denied(self, repo):
        scope = resolve_repository_scope(repo, GitRepositoryRelocation())
        assert scope is not None
        destination = canonical_remote_destination("deploy@8.8.8.8:22")
        assert destination is not None
        grant = DangerousCommandAuthority(
            operations=frozenset({REMOTE_EXEC_OPERATION_ID}),
            scope=scope,
            source="runtime_grant",
            destination=destination,
        )
        assert authority_admits(
            grant, REMOTE_EXEC_OPERATION_ID, scope, destination=destination
        )
        other = canonical_remote_destination("deploy@8.8.8.8:2222")
        assert other is not None
        assert not authority_admits(
            grant, REMOTE_EXEC_OPERATION_ID, scope, destination=other
        )

    def test_egress_grant_never_admits_a_transfer_and_vice_versa(self, repo):
        scope = resolve_repository_scope(repo, GitRepositoryRelocation())
        assert scope is not None
        destination = canonical_remote_destination("deploy@8.8.8.8:22")
        assert destination is not None
        egress = DangerousCommandAuthority(
            operations=frozenset({"network.egress"}),
            scope=scope,
            source="runtime_grant",
        )
        assert not authority_admits(
            egress, REMOTE_EXEC_OPERATION_ID, scope, destination=destination
        )
        grant = DangerousCommandAuthority(
            operations=frozenset({REMOTE_EXEC_OPERATION_ID}),
            scope=scope,
            source="runtime_grant",
            destination=destination,
        )
        assert not authority_admits(grant, "network.egress", scope)


# ---------------------------------------------------------------------------
# Execution plan (pinned argv) and the S4b2b rebinding guard at the plan
# ---------------------------------------------------------------------------


class TestTransferPlan:
    def test_scp_literal_plan_pins_address_and_host_key_identity(
        self, hermetic_seams, repo
    ):
        tool = _tool(repo)
        plan = _plan(tool, "scp -r file deploy@8.8.8.8:backup/")
        assert plan is not None
        assert plan.client == "scp"
        assert plan.destination.key == "deploy@8.8.8.8:22"
        assert plan.resolved_address is None
        assert plan.argv[0] == "scp"
        assert plan.argv[1:9] == (
            "-o",
            "UserKnownHostsFile=/dev/null/chartreux-pinned-known-hosts",
            "-o",
            "HostKeyAlias=deploy@8.8.8.8:22",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            "BatchMode=yes",
        )
        assert plan.argv[-2:] == ("file", "deploy@8.8.8.8:backup/")
        assert (
            plan.command
            == pinned_transfer_command_text(
                parse_transfer_call(plan.argv)  # type: ignore[arg-type]
            )
            or "deploy@8.8.8.8:backup/" in plan.command
        )

    def test_rsync_hostname_plan_carries_the_resolved_address_and_pinned_transport(
        self, hermetic_seams, repo
    ):
        tool = _tool(repo)
        plan = _plan(tool, "rsync -a src/ deploy@example.com:dst/")
        assert plan is not None
        assert plan.client == "rsync"
        assert plan.resolved_address == "93.184.216.34"
        assert plan.destination.key == "deploy@example.com:22"
        argv = plan.argv
        assert argv[0] == "rsync"
        assert argv[1] == "-e"
        assert argv[2].startswith("ssh ")
        assert "HostKeyAlias=deploy@example.com:22" in argv[2]
        assert "UserKnownHostsFile=/dev/null/chartreux-pinned-known-hosts" in argv[2]
        # The resolved address rides as the remote operand's host.
        assert argv[-1] == "deploy@93.184.216.34:dst/"
        assert "deploy@example.com" not in argv[-1]

    def test_scp_hostname_plan_pins_the_resolved_address_as_operand(
        self, hermetic_seams, repo
    ):
        tool = _tool(repo)
        plan = _plan(tool, "scp f deploy@example.com:x")
        assert plan is not None
        assert plan.resolved_address == "93.184.216.34"
        assert plan.argv[-1] == "deploy@93.184.216.34:x"
        assert "HostKeyAlias=deploy@example.com:22" in plan.argv

    @pytest.mark.parametrize(
        "command",
        [
            "scp file host:backup",
            "rsync -a src/ host:dst/",
            "rsync -e ssh a deploy@8.8.8.8:b",
            "scp -O f deploy@8.8.8.8:x",
            "scp f deploy@8.8.8.8:*.py",
            "scp deploy@127.0.0.1:x l",
            "scp f deploy@8.8.8.8:x && echo done",
            "nice scp f deploy@8.8.8.8:x",
        ],
    )
    def test_denied_form_raises_at_the_plan_and_never_returns_a_plan(
        self, hermetic_seams, repo, command
    ):
        tool = _tool(repo, _grant(repo, "deploy@8.8.8.8"))
        with pytest.raises(RemoteExecDenial):
            _plan(tool, command)

    def test_no_remote_invocation_returns_none(self, hermetic_seams, repo):
        tool = _tool(repo)
        assert _plan(tool, "rsync -a src/ dst/") is None
        assert _plan(tool, "echo hi") is None

    def test_pinned_argv_revalidates_as_pinned_ask(self, hermetic_seams, repo):
        # The background spawn re-parses the pinned argv: it must come back
        # ask + pinned (the approved address as operand, HostKeyAlias bound
        # to the granted hostname).
        tool = _tool(repo)
        plan = _plan(tool, "scp f deploy@example.com:x")
        assert plan is not None
        classification = classify_transfer_call(list(plan.argv))
        assert classification.kind == "ask"
        assert classification.is_pinned
        assert classification.destination is not None
        assert classification.destination.key == "deploy@example.com:22"
        plan_r = _plan(tool, "rsync -a s deploy@example.com:d")
        assert plan_r is not None
        classification_r = classify_transfer_call(list(plan_r.argv))
        assert classification_r.kind == "ask"
        assert classification_r.is_pinned

    def test_hand_written_pin_pointing_at_an_unvalidated_address_denies(
        self, hermetic_seams, repo
    ):
        # A pin whose operand address is not the granted host's current
        # unique non-local resolution denies at revalidation.
        argv = [
            "scp",
            "-o",
            "UserKnownHostsFile=/dev/null/chartreux-pinned-known-hosts",
            "-o",
            "HostKeyAlias=deploy@example.com:22",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            "BatchMode=yes",
            "f",
            "deploy@1.1.1.1:x",
        ]
        classification = classify_transfer_call(argv)
        assert classification.kind == "deny"
        assert "not the current unique non-local resolution" in (
            classification.reason or ""
        )


class TestTransferRebindAtThePlan:
    """S4b2c rebinding arm at the plan: a CHANGED resolution is never planned.

    HHH permits either conforming half — the spawn is denied, or the ORIGINAL
    validated pinned address rides — the changed answer never does.
    """

    @pytest.fixture
    def rebind_seams(self, monkeypatch: pytest.MonkeyPatch):
        state = {"calls": 0}

        def resolver(host: str) -> frozenset:
            if host != "example.com":
                raise socket.gaierror(-2, "Name or service not known")
            state["calls"] += 1
            answer = "93.184.216.34" if state["calls"] == 1 else "127.0.0.1"
            return frozenset({ipaddress.ip_address(answer)})

        monkeypatch.setattr(
            _remote_exec, "_interface_addresses", lambda: FIXED_INTERFACES
        )
        monkeypatch.setattr(_remote_exec, "_hostname_resolver", resolver)
        return state

    @pytest.mark.parametrize("client", ["scp", "rsync"])
    def test_changed_answer_never_reaches_the_transfer_plan(
        self, hermetic_seams, rebind_seams, repo, client
    ):
        command = (
            "scp f deploy@example.com:x"
            if client == "scp"
            else "rsync -a s deploy@example.com:d"
        )
        tool = _tool(repo, _grant(repo, "deploy@example.com"))
        plan = None
        message = ""
        try:
            plan = _plan(tool, command)
        except RemoteExecDenial as exc:
            message = str(exc)
        if plan is None:
            # Deny half (HHH outcome a): the changed answer is denied with a
            # discriminating reason naming the rebind.
            assert "never contacted" in message
            assert "127.0.0.1" in message
            assert "93.184.216.34" in message
        else:
            # Execute-original-pinned half (HHH outcome b): the plan carries
            # the ORIGINAL validated pin — the rebound answer is nowhere.
            assert plan.client == client
            assert plan.resolved_address == "93.184.216.34"
            assert "deploy@93.184.216.34:" in plan.argv[
                -1
            ] or "deploy@93.184.216.34:" in " ".join(plan.argv)
            assert "127.0.0.1" not in plan.command
            assert "127.0.0.1" not in " ".join(plan.argv)

    def test_unchanged_answer_still_builds_the_pinned_transfer_plan(
        self, hermetic_seams, repo
    ):
        tool = _tool(repo, _grant(repo, "deploy@example.com"))
        plan = _plan(tool, "rsync -a s deploy@example.com:d")
        assert plan is not None
        assert plan.resolved_address == "93.184.216.34"
        assert plan.argv[-1] == "deploy@93.184.216.34:d"
        assert "HostKeyAlias=deploy@example.com:22" in plan.argv[2]


# ---------------------------------------------------------------------------
# X's attestation-content arm at the plan seam (the spawn-observing arms
# live in tests/tools/test_bash_execution.py)
# ---------------------------------------------------------------------------


class TestLocalityOnlyAttestationSemantics:
    def test_locality_only_entry_hard_denies_transfers(self):
        # A locality-only attestation entry (network_locality without
        # ssh_trust_context) is not execution-bound: transfers are denied on
        # a client-owned transport even though the destination is same-network
        # non-local. (The spawn-observing assertion lives at S4b2c's carrier
        # in test_bash_execution.py.)
        attestation = RemoteExecAttestation(
            network_locality=True, ssh_trust_context=False
        )
        assert attestation.locality_only
        assert not attestation.execution_bound
        full = RemoteExecAttestation(network_locality=True, ssh_trust_context=True)
        assert full.execution_bound
