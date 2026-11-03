"""Dangerous-command authority inspection and revocation over the wire.

The settings surfaces (policy/dangerous_commands/read and
policy/dangerous_commands/revoke) inspect and revoke one registered session's
dangerous-command authority: live session grants from the session's own store,
accepted saved authority from the restrictions the session enforces, the fresh
disk read with its revision, and the blocking explicit denylist entries.
Revocation removes live and saved authority with separate outcomes. The
in-process backend and the wire surface behave identically.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tomllib
from typing import Any

import pytest

from chartreux.app_server._session_backend_impl import SessionBackendImpl
from chartreux.app_server.client import AppServerClient
from chartreux.app_server.protocol import (
    AppServerResponseError,
    ClientCapabilities,
    ClientInfo,
    DangerousCommandAuthorityReadParams,
    DangerousCommandAuthorityReadResponse,
    DangerousCommandRevokeParams,
    DangerousCommandRevokeResponse,
    ProtocolErrorCode,
)
from chartreux.app_server.server import AppServer
from chartreux.app_server.session import AppServerSession
from chartreux.app_server.transport import memory_transport_pair
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.agent_loop._dangerous_command_grants import (
    RemoteExecGrantRequirement,
)
from chartreux.core.config import ChartreuxConfigSchema
from chartreux.core.config._dangerous_command_authority import (
    GRANTS_FIELD,
    REMOTE_EXEC_GRANTS_FIELD,
)
from chartreux.core.config.layers.default import DefaultConfigLayer
from chartreux.core.config.layers.overrides import OverridesLayer
from chartreux.core.config.layers.user import UserConfigLayer
from chartreux.core.config.orchestrator import ConfigOrchestrator
from chartreux.core.dangerous_commands import (
    OPERATION_REGISTRY,
    REMOTE_EXEC_OPERATION_ID,
    RemoteDestination,
    canonical_remote_destination,
)
from chartreux.core.tools.base import ToolPermission
from chartreux.core.tools.builtins._destination import classify_network_egress
from chartreux.core.tools.builtins._shell_command_policy import (
    GitRepositoryRelocation,
    classify_dangerous_git_command,
    resolve_repository_scope,
)
from tests.stubs.app_server import build_test_app_server
from tests.stubs.fake_backend import FakeBackend
from tests.stubs.fake_mcp_registry import FakeMCPRegistry

DENY_PROJECTION_OPERATIONS = [
    "git.push",
    "git.push.force",
    "git.push.delete",
    "git.push.mirror",
]


def make_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    return path


def _scope_of(path: Path):
    scope = resolve_repository_scope(path, GitRepositoryRelocation())
    assert scope is not None
    return scope


async def _make_loop(
    project: Path, config_path: Path, backend: FakeBackend | None = None
) -> AgentLoop:
    """Build a root loop over a real user layer so disk IO reaches the file."""
    user = UserConfigLayer(path=config_path)
    overlay = OverridesLayer(data={})
    orchestrator = await ConfigOrchestrator.create(
        schema=ChartreuxConfigSchema,
        layers=[DefaultConfigLayer(schema=ChartreuxConfigSchema), user, overlay],
        default_layer_resolver=lambda: overlay,
    )
    return AgentLoop(
        config_orchestrator=orchestrator,
        cwd=project,
        backend=backend or FakeBackend(),
        mcp_registry=FakeMCPRegistry(),
        user_input_capability=True,
    )


async def _open_session(
    loop: AgentLoop,
) -> tuple[AppServer, AppServerClient, AppServerSession]:
    client_transport, server_transport = memory_transport_pair()
    server = build_test_app_server(loop, server_transport)
    client = AppServerClient(client_transport, run_peer=server.serve)
    session = await AppServerSession.start(
        client,
        client_info=ClientInfo(name="dangerous-permissions-test", version="0"),
        capabilities=ClientCapabilities(callback_kinds=["user_input"]),
    )
    return server, client, session


def _backend(server: AppServer) -> SessionBackendImpl:
    root = server._require_root()
    assert isinstance(root, SessionBackendImpl)
    return root


def _saved_grants(config_path: Path) -> dict[str, list[str]]:
    if not config_path.exists():
        return {}
    data: dict[str, Any] = tomllib.loads(config_path.read_text())
    grants = data.get(GRANTS_FIELD, {})
    return {str(key): [str(item) for item in value] for key, value in grants.items()}


def _saved_remote_exec_grants(config_path: Path) -> dict[str, list[str]]:
    """Saved remote-exec destinations per project, in canonical key spelling."""
    if not config_path.exists():
        return {}
    data: dict[str, Any] = tomllib.loads(config_path.read_text())
    grants = data.get(REMOTE_EXEC_GRANTS_FIELD, {})
    return {
        str(key): sorted(
            parsed.key
            for item in value
            if (parsed := canonical_remote_destination(str(item))) is not None
        )
        for key, value in grants.items()
    }


def _destination(spelling: str) -> RemoteDestination:
    parsed = canonical_remote_destination(spelling)
    assert parsed is not None
    return parsed


# Two distinct exact destinations: a default-port one and a custom-port one.
DESTINATION_A = _destination("deploy@203.0.113.10")
DESTINATION_B = _destination("deploy@203.0.113.20:2222")


def _remote_exec_requirement(
    project: Path, destination: RemoteDestination
) -> RemoteExecGrantRequirement:
    return RemoteExecGrantRequirement(scope=_scope_of(project), destination=destination)


def _remote_exec_authorities(loop: AgentLoop) -> list[tuple[str, str]]:
    """(destination key, git dir) of every live remote-exec authority entry."""
    return sorted(
        (authority.destination.key, str(authority.scope.git_dir))
        for authority in loop._dangerous_command_grants.dangerous_command_authorities()
        if authority.destination is not None
    )


def _authorities(loop: AgentLoop) -> list[tuple[frozenset[str], object]]:
    return [
        (authority.operations, authority.scope)
        for authority in loop._dangerous_command_grants.dangerous_command_authorities()
    ]


async def _wire_read(
    client: AppServerClient, session_id: str
) -> DangerousCommandAuthorityReadResponse:
    return DangerousCommandAuthorityReadResponse.model_validate(
        await client.request(
            "policy/dangerous_commands/read",
            DangerousCommandAuthorityReadParams(session_id=session_id),
        )
    )


async def _wire_revoke(
    client: AppServerClient,
    session_id: str,
    operation_id: str,
    expected_live_generation: int,
    expected_disk_revision: str | None,
    destination: str | None = None,
) -> DangerousCommandRevokeResponse:
    return DangerousCommandRevokeResponse.model_validate(
        await client.request(
            "policy/dangerous_commands/revoke",
            DangerousCommandRevokeParams(
                session_id=session_id,
                operation_id=operation_id,
                expected_live_generation=expected_live_generation,
                expected_disk_revision=expected_disk_revision,
                destination=destination,
                user_initiated=True,
            ),
        )
    )


@pytest.mark.asyncio
async def test_inspect_projects_live_accepted_saved_disk_and_denylist(
    tmp_path: Path, config_dir: Path
) -> None:
    project = make_repo(tmp_path / "project")
    config_path = config_dir / "config.toml"
    # Accepted at orchestrator creation: a saved grant and an explicit
    # denylist entry that absolutely vetoes push operations.
    config_path.write_text(
        config_path.read_text()
        + f'\n[{GRANTS_FIELD}]\n"{project.resolve()}" = ["git.push"]\n'
        + '\n[tools.bash]\ndenylist = ["git push"]\n',
        encoding="utf-8",
    )
    loop = await _make_loop(project, config_path)
    server, client, session = await _open_session(loop)
    try:
        # A live session grant on the session's own store.
        loop._dangerous_command_grants.grant_session({("git.push", _scope_of(project))})
        wire = await _wire_read(client, loop.session_id)
        assert wire.project == str(project)
        assert wire.live_authority_generation == 1
        assert [
            (grant.operation_id, grant.scope.git_dir) for grant in wire.live_grants
        ] == [("git.push", str(_scope_of(project).git_dir))]
        assert [
            (grant.operation_id, grant.scope.git_dir)
            for grant in wire.accepted_saved_grants
        ] == [("git.push", str(_scope_of(project).git_dir))]
        assert wire.saved_operations == ["git.push"]
        assert wire.disk_revision
        # The explicit denylist entry is reported with the operations it
        # vetoes, apart from grant authority: no grant overrides it.
        assert [
            (block.pattern, block.operations) for block in wire.blocking_denylist
        ] == [("git push", DENY_PROJECTION_OPERATIONS)]

        # The in-process backend and registry surfaces agree with the wire.
        backend = _backend(server)
        in_process = await backend.read_dangerous_command_authority(
            DangerousCommandAuthorityReadParams(session_id=loop.session_id)
        )
        assert in_process == wire
        registry_view = await backend.children.read_dangerous_command_authority(
            loop.session_id
        )
        assert registry_view == wire
        # The client resource surface is the same wire call.
        resource = await session.resources.config.read_dangerous_command_authority()
        assert resource == wire
    finally:
        await session.close()
        await loop.aclose()


@pytest.mark.asyncio
async def test_inspect_disk_read_tracks_the_file_without_accepting_it(
    tmp_path: Path, config_dir: Path
) -> None:
    project = make_repo(tmp_path / "project")
    other = make_repo(tmp_path / "other")
    config_path = config_dir / "config.toml"
    loop = await _make_loop(project, config_path)
    server, client, session = await _open_session(loop)
    try:
        first = await _wire_read(client, loop.session_id)
        assert first.live_grants == []
        assert first.accepted_saved_grants == []
        assert first.saved_operations == []
        assert first.disk_revision
        # An intervening writer adds a grant for another project on disk.
        config_path.write_text(
            config_path.read_text()
            + f'\n[{GRANTS_FIELD}]\n"{other.resolve()}" = ["git.push"]\n',
            encoding="utf-8",
        )
        second = await _wire_read(client, loop.session_id)
        # The disk read is fresh, but the running session's accepted
        # authority is unchanged until a reload.
        assert second.saved_operations == []
        assert second.accepted_saved_grants == []
        assert second.disk_revision != first.disk_revision
    finally:
        await session.close()
        await loop.aclose()


@pytest.mark.asyncio
async def test_revoke_removes_live_and_saved_with_separate_outcomes(
    tmp_path: Path, config_dir: Path
) -> None:
    project = make_repo(tmp_path / "project")
    config_path = config_dir / "config.toml"
    config_path.write_text(
        config_path.read_text()
        + f'\n[{GRANTS_FIELD}]\n"{project.resolve()}" = ["git.push"]\n',
        encoding="utf-8",
    )
    loop = await _make_loop(project, config_path)
    server, client, session = await _open_session(loop)
    try:
        loop._dangerous_command_grants.grant_session({("git.push", _scope_of(project))})
        read = await _wire_read(client, loop.session_id)
        response = await _wire_revoke(
            client,
            loop.session_id,
            "git.push",
            read.live_authority_generation,
            read.disk_revision,
        )
        assert response.live.status == "removed"
        assert response.live.generation == read.live_authority_generation + 1
        assert response.saved.status == "removed"
        assert response.saved.revision
        # The runtime store no longer carries the operation, but the
        # session's accepted restrictions still authorize it until a
        # reload, so the response warns instead of reporting a full removal.
        assert response.warning is not None
        assert "keeps git.push authority from accepted config" in response.warning
        assert response.live_authority_generation == read.live_authority_generation + 1
        # Live authority is gone immediately; the disk grant is removed.
        assert _authorities(loop) == []
        assert _saved_grants(config_path) == {}
        after = await _wire_read(client, loop.session_id)
        assert after.live_grants == []
        assert after.saved_operations == []
        # The running session's accepted config is unchanged until a reload,
        # exactly as a save never applies to the running session.
        assert [grant.operation_id for grant in after.accepted_saved_grants] == [
            "git.push"
        ]

        # The in-process backend surface behaves identically.
        loop._dangerous_command_grants.grant_session({("git.push", _scope_of(project))})
        config_path.write_text(
            config_path.read_text()
            + f'\n[{GRANTS_FIELD}]\n"{project.resolve()}" = ["git.push"]\n',
            encoding="utf-8",
        )
        reread = await _wire_read(client, loop.session_id)
        backend = _backend(server)
        in_process = await backend.revoke_dangerous_command_grant(
            DangerousCommandRevokeParams(
                session_id=loop.session_id,
                operation_id="git.push",
                expected_live_generation=reread.live_authority_generation,
                expected_disk_revision=reread.disk_revision,
                user_initiated=True,
            )
        )
        assert in_process.live.status == "removed"
        assert in_process.saved.status == "removed"
        assert _authorities(loop) == []
        assert _saved_grants(config_path) == {}
        # The client resource surface is the same wire call.
        loop._dangerous_command_grants.grant_session({("git.push", _scope_of(project))})
        config_path.write_text(
            config_path.read_text()
            + f'\n[{GRANTS_FIELD}]\n"{project.resolve()}" = ["git.push"]\n',
            encoding="utf-8",
        )
        reread = await _wire_read(client, loop.session_id)
        resource = await session.resources.config.revoke_dangerous_command_grant(
            operation_id="git.push",
            expected_live_generation=reread.live_authority_generation,
            expected_disk_revision=reread.disk_revision,
            user_initiated=True,
        )
        assert resource.live.status == "removed"
        assert resource.saved.status == "removed"
        assert _authorities(loop) == []
        assert _saved_grants(config_path) == {}
    finally:
        await session.close()
        await loop.aclose()


@pytest.mark.asyncio
async def test_revoke_with_stale_live_generation_conflicts_and_leaves_saved_untouched(
    tmp_path: Path, config_dir: Path
) -> None:
    project = make_repo(tmp_path / "project")
    config_path = config_dir / "config.toml"
    config_path.write_text(
        config_path.read_text()
        + f'\n[{GRANTS_FIELD}]\n"{project.resolve()}" = ["git.push"]\n',
        encoding="utf-8",
    )
    loop = await _make_loop(project, config_path)
    server, client, session = await _open_session(loop)
    try:
        loop._dangerous_command_grants.grant_session({("git.push", _scope_of(project))})
        read = await _wire_read(client, loop.session_id)
        # The live authority changes between the read and the revocation.
        loop._dangerous_command_grants.grant_session({
            ("git.checkout", _scope_of(project))
        })
        response = await _wire_revoke(
            client,
            loop.session_id,
            "git.push",
            read.live_authority_generation,
            read.disk_revision,
        )
        # A stale live generation is a conflict/re-read outcome: nothing is
        # removed and the saved side is untouched.
        assert response.live.status == "conflict"
        assert response.live.generation == read.live_authority_generation + 1
        assert response.saved.status == "skipped"
        assert response.warning is None
        assert response.live_authority_generation == read.live_authority_generation + 1
        # Both grants survive, merged per repository scope.
        assert _authorities(loop) == [
            (frozenset({"git.push", "git.checkout"}), _scope_of(project))
        ]
        assert _saved_grants(config_path) == {str(project.resolve()): ["git.push"]}
    finally:
        await session.close()
        await loop.aclose()


@pytest.mark.asyncio
async def test_revoke_with_stale_disk_revision_reports_saved_conflict_but_removes_live(
    tmp_path: Path, config_dir: Path
) -> None:
    project = make_repo(tmp_path / "project")
    config_path = config_dir / "config.toml"
    config_path.write_text(
        config_path.read_text()
        + f'\n[{GRANTS_FIELD}]\n"{project.resolve()}" = ["git.push"]\n',
        encoding="utf-8",
    )
    loop = await _make_loop(project, config_path)
    server, client, session = await _open_session(loop)
    try:
        loop._dangerous_command_grants.grant_session({("git.push", _scope_of(project))})
        read = await _wire_read(client, loop.session_id)
        # An intervening writer changes the user config after the read.
        config_path.write_text(
            config_path.read_text() + "\n# intervening writer\n", encoding="utf-8"
        )
        response = await _wire_revoke(
            client,
            loop.session_id,
            "git.push",
            read.live_authority_generation,
            read.disk_revision,
        )
        # Live removal is applied; the saved removal reports its own conflict.
        assert response.live.status == "removed"
        assert response.saved.status == "conflict"
        assert response.warning is not None
        assert "can return on the next config reload or session start" in (
            response.warning
        )
        assert _authorities(loop) == []
        assert _saved_grants(config_path) == {str(project.resolve()): ["git.push"]}
    finally:
        await session.close()
        await loop.aclose()


@pytest.mark.asyncio
async def test_revoke_requires_user_initiation_and_a_registered_session(
    tmp_path: Path, config_dir: Path
) -> None:
    project = make_repo(tmp_path / "project")
    config_path = config_dir / "config.toml"
    loop = await _make_loop(project, config_path)
    server, client, session = await _open_session(loop)
    try:
        loop._dangerous_command_grants.grant_session({("git.push", _scope_of(project))})
        read = await _wire_read(client, loop.session_id)
        base: dict[str, Any] = {
            "sessionId": loop.session_id,
            "operationId": "git.push",
            "expectedLiveGeneration": read.live_authority_generation,
            "expectedDiskRevision": read.disk_revision,
            "userInitiated": True,
        }
        # Revocation requires the literal user initiation: only the literal
        # true passes, and lax coercion of truthy values is rejected like any
        # other non-literal.
        for rejected in (False, 1, "true", None):
            with pytest.raises(AppServerResponseError) as excinfo:
                await client.request(
                    "policy/dangerous_commands/revoke",
                    base | {"userInitiated": rejected},
                )
            assert excinfo.value.error.code is ProtocolErrorCode.INVALID_PARAMS
        with pytest.raises(AppServerResponseError) as excinfo:
            await client.request(
                "policy/dangerous_commands/revoke",
                {key: value for key, value in base.items() if key != "userInitiated"},
            )
        assert excinfo.value.error.code is ProtocolErrorCode.INVALID_PARAMS
        # Unregistered sessions cannot revoke.
        with pytest.raises(AppServerResponseError) as excinfo:
            await client.request(
                "policy/dangerous_commands/revoke",
                base | {"sessionId": "unregistered-session"},
            )
        assert excinfo.value.error.code is ProtocolErrorCode.NOT_FOUND
        # Unknown operations are rejected before any authority is touched.
        with pytest.raises(AppServerResponseError) as excinfo:
            await client.request(
                "policy/dangerous_commands/revoke", base | {"operationId": "git.rebase"}
            )
        assert excinfo.value.error.code is ProtocolErrorCode.INVALID_PARAMS
        # No caller-supplied project key is accepted on any new surface.
        with pytest.raises(AppServerResponseError):
            await client.request(
                "policy/dangerous_commands/read",
                {"sessionId": loop.session_id, "project": str(project)},
            )
        with pytest.raises(AppServerResponseError):
            await client.request(
                "policy/dangerous_commands/revoke", base | {"project": str(project)}
            )
        # Nothing was revoked.
        assert _authorities(loop) == [(frozenset({"git.push"}), _scope_of(project))]
    finally:
        await session.close()
        await loop.aclose()


@pytest.mark.asyncio
async def test_inspect_rejects_unregistered_and_stale_sessions(
    tmp_path: Path, config_dir: Path
) -> None:
    project = make_repo(tmp_path / "project")
    config_path = config_dir / "config.toml"
    loop = await _make_loop(project, config_path)
    server, client, session = await _open_session(loop)
    try:
        with pytest.raises(AppServerResponseError) as excinfo:
            await _wire_read(client, "unregistered-session")
        assert excinfo.value.error.code is ProtocolErrorCode.NOT_FOUND
        with pytest.raises(AppServerResponseError) as excinfo:
            await client.request(
                "policy/dangerous_commands/revoke",
                {
                    "sessionId": "unregistered-session",
                    "operationId": "git.push",
                    "expectedLiveGeneration": 0,
                    "expectedDiskRevision": None,
                    "userInitiated": True,
                },
            )
        assert excinfo.value.error.code is ProtocolErrorCode.NOT_FOUND

        # A stale (closed) child session is rejected too.
        backend = _backend(server)
        registry = backend.children
        policy = loop.child_runtime_policy
        child = AgentLoop(
            config_orchestrator=loop.config_orchestrator.copy(),
            cwd=project,
            backend=FakeBackend(),
            is_subagent=True,
            inherited_workspace=policy.inherited_workspace,
            inherited_restrictions=policy.inherited_restrictions,
            inherited_mode_restrictions=policy.inherited_mode_restrictions,
            inherited_plan_write_scopes=policy.inherited_plan_write_scopes,
            inherited_root_grants=policy.inherited_root_grants,
            user_input_capability=True,
            parent_authority_getter=policy.parent_authority_getter,
            parent_authority_revision_getter=policy.parent_authority_revision_getter,
        )
        runtime = registry._build_child_runtime(child)
        registry._children[child.session_id] = runtime
        try:
            stale_read = await _wire_read(client, child.session_id)
            assert stale_read.project == str(project)
            await runtime.close()
            with pytest.raises(AppServerResponseError) as excinfo:
                await _wire_read(client, child.session_id)
            assert excinfo.value.error.code is ProtocolErrorCode.NOT_FOUND
            with pytest.raises(AppServerResponseError) as excinfo:
                await client.request(
                    "policy/dangerous_commands/revoke",
                    {
                        "sessionId": child.session_id,
                        "operationId": "git.push",
                        "expectedLiveGeneration": 0,
                        "expectedDiskRevision": None,
                        "userInitiated": True,
                    },
                )
            assert excinfo.value.error.code is ProtocolErrorCode.NOT_FOUND
        finally:
            registry._children.pop(child.session_id, None)
            await child.aclose()
    finally:
        await session.close()
        await loop.aclose()


@pytest.mark.asyncio
async def test_revoke_invalidates_pending_approvals(
    tmp_path: Path, config_dir: Path
) -> None:
    """Pending once approvals cannot resurrect revoked authority."""
    project = make_repo(tmp_path / "project")
    config_path = config_dir / "config.toml"
    loop = await _make_loop(project, config_path)
    server, client, session = await _open_session(loop)
    try:
        store = loop._dangerous_command_grants
        store.grant_session({("git.push", _scope_of(project))})
        store.install_once(frozenset({("git.push", _scope_of(project))}))
        assert len(_authorities(loop)) == 2
        read = await _wire_read(client, loop.session_id)
        response = await _wire_revoke(
            client,
            loop.session_id,
            "git.push",
            read.live_authority_generation,
            read.disk_revision,
        )
        assert response.live.status == "removed"
        # The session grant and the pending once permit are both gone.
        assert _authorities(loop) == []
        # Cached authority was dropped: a fresh resolution sees no authority.
        tool = loop.tool_manager.get("bash")
        args = tool.validate_arguments({"command": "git push"})
        resolved = tool.resolve_permission(args)
        assert resolved is not None
        assert resolved.denial_kind == "dangerous_command"
    finally:
        await session.close()
        await loop.aclose()


@pytest.mark.asyncio
async def test_revoke_consequential_operation_reports_saved_not_applicable(
    tmp_path: Path, config_dir: Path
) -> None:
    project = make_repo(tmp_path / "project")
    config_path = config_dir / "config.toml"
    loop = await _make_loop(project, config_path)
    server, client, session = await _open_session(loop)
    try:
        loop._dangerous_command_grants.grant_session({
            ("git.stash.drop", _scope_of(project))
        })
        read = await _wire_read(client, loop.session_id)
        response = await _wire_revoke(
            client,
            loop.session_id,
            "git.stash.drop",
            read.live_authority_generation,
            read.disk_revision,
        )
        assert response.live.status == "removed"
        # Consequential operations are never saved, so no saved authority can
        # exist and no disk removal is attempted.
        assert response.saved.status == "not_applicable"
        assert response.warning is None
        assert _authorities(loop) == []
    finally:
        await session.close()
        await loop.aclose()


@pytest.mark.asyncio
async def test_revoke_without_a_disk_revision_skips_the_saved_side(
    tmp_path: Path, config_dir: Path
) -> None:
    project = make_repo(tmp_path / "project")
    config_path = config_dir / "config.toml"
    config_path.write_text(
        config_path.read_text()
        + f'\n[{GRANTS_FIELD}]\n"{project.resolve()}" = ["git.push"]\n',
        encoding="utf-8",
    )
    loop = await _make_loop(project, config_path)
    server, client, session = await _open_session(loop)
    try:
        loop._dangerous_command_grants.grant_session({("git.push", _scope_of(project))})
        read = await _wire_read(client, loop.session_id)
        response = await _wire_revoke(
            client, loop.session_id, "git.push", read.live_authority_generation, None
        )
        assert response.live.status == "removed"
        assert response.saved.status == "skipped"
        # The saved side was not attempted, but the accepted config still
        # authorizes the operation in this session until a reload.
        assert response.warning is not None
        assert "keeps git.push authority from accepted config" in response.warning
        assert _authorities(loop) == []
        assert _saved_grants(config_path) == {str(project.resolve()): ["git.push"]}
    finally:
        await session.close()
        await loop.aclose()


@pytest.mark.asyncio
async def test_revoke_warns_and_keeps_accepted_authority_until_reload(
    tmp_path: Path, config_dir: Path
) -> None:
    """Revoking a live grant does not clear authority the accepted config keeps."""
    project = make_repo(tmp_path / "project")
    config_path = config_dir / "config.toml"
    config_path.write_text(
        config_path.read_text()
        + f'\n[{GRANTS_FIELD}]\n"{project.resolve()}" = ["git.push"]\n',
        encoding="utf-8",
    )
    loop = await _make_loop(project, config_path)
    server, client, session = await _open_session(loop)
    try:
        loop._dangerous_command_grants.grant_session({("git.push", _scope_of(project))})
        read = await _wire_read(client, loop.session_id)
        assert [grant.operation_id for grant in read.accepted_saved_grants] == [
            "git.push"
        ]
        response = await _wire_revoke(
            client,
            loop.session_id,
            "git.push",
            read.live_authority_generation,
            read.disk_revision,
        )
        # The runtime store no longer carries the operation...
        assert response.live.status == "removed"
        assert _authorities(loop) == []
        # ...but the response warns that the accepted config still
        # authorizes it in this session until a reload or restart.
        assert response.warning is not None
        assert "keeps git.push authority from accepted config" in response.warning
        assert "reload" in response.warning
        # The session's live authority persists: the resolver still admits
        # the operation through the accepted saved grant.
        tool = loop.tool_manager.get("bash")
        args = tool.validate_arguments({"command": "git push"})
        admitted = tool.resolve_permission(args)
        assert admitted is not None
        assert admitted.permission is ToolPermission.ALWAYS
        assert admitted.denial_kind is None
        # A reload drops the accepted authority, so the next resolution
        # denies the operation again. The reload retires the bound tool
        # instance, so the check re-acquires a current one.
        await session.resources.config.reload()
        current_tool = loop.tool_manager.get("bash")
        denied = current_tool.resolve_permission(args)
        assert denied is not None
        assert denied.denial_kind == "dangerous_command"
    finally:
        await session.close()
        await loop.aclose()


@pytest.mark.asyncio
async def test_revoke_with_only_accepted_config_authority_warns_and_keeps_it(
    tmp_path: Path, config_dir: Path
) -> None:
    """The panel's Saved row: an empty runtime store still warns on revoke."""
    project = make_repo(tmp_path / "project")
    config_path = config_dir / "config.toml"
    config_path.write_text(
        config_path.read_text()
        + f'\n[{GRANTS_FIELD}]\n"{project.resolve()}" = ["git.push"]\n',
        encoding="utf-8",
    )
    loop = await _make_loop(project, config_path)
    server, client, session = await _open_session(loop)
    try:
        # The runtime store is empty: the session's authority comes only
        # from the accepted saved grant (the panel's Saved row).
        assert _authorities(loop) == []
        read = await _wire_read(client, loop.session_id)
        assert [grant.operation_id for grant in read.accepted_saved_grants] == [
            "git.push"
        ]
        response = await _wire_revoke(
            client,
            loop.session_id,
            "git.push",
            read.live_authority_generation,
            read.disk_revision,
        )
        # The removal is idempotent on the live side and removes the saved
        # grant, but the warning covers the truth: the session still admits
        # the operation through the accepted config until a reload.
        assert response.live.status == "removed"
        assert response.saved.status == "removed"
        assert _authorities(loop) == []
        assert _saved_grants(config_path) == {}
        assert response.warning is not None
        assert "keeps git.push authority from accepted config" in response.warning
        assert "reload" in response.warning
        # The session still authorizes the operation until a reload.
        tool = loop.tool_manager.get("bash")
        args = tool.validate_arguments({"command": "git push"})
        admitted = tool.resolve_permission(args)
        assert admitted is not None
        assert admitted.permission is ToolPermission.ALWAYS
        assert admitted.denial_kind is None
        # A reload drops the accepted authority, so the next resolution
        # denies the operation again. The reload retires the bound tool
        # instance, so the check re-acquires a current one.
        await session.resources.config.reload()
        current_tool = loop.tool_manager.get("bash")
        denied = current_tool.resolve_permission(args)
        assert denied is not None
        assert denied.denial_kind == "dangerous_command"
    finally:
        await session.close()
        await loop.aclose()


@pytest.mark.asyncio
async def test_reload_after_revoke_clears_accepted_saved_authority(
    tmp_path: Path, config_dir: Path
) -> None:
    project = make_repo(tmp_path / "project")
    config_path = config_dir / "config.toml"
    config_path.write_text(
        config_path.read_text()
        + f'\n[{GRANTS_FIELD}]\n"{project.resolve()}" = ["git.push"]\n',
        encoding="utf-8",
    )
    loop = await _make_loop(project, config_path)
    server, client, session = await _open_session(loop)
    try:
        read = await _wire_read(client, loop.session_id)
        assert [grant.operation_id for grant in read.accepted_saved_grants] == [
            "git.push"
        ]
        response = await _wire_revoke(
            client,
            loop.session_id,
            "git.push",
            read.live_authority_generation,
            read.disk_revision,
        )
        assert response.saved.status == "removed"
        # The running session keeps its accepted view until a reload.
        before_reload = await _wire_read(client, loop.session_id)
        assert [
            grant.operation_id for grant in before_reload.accepted_saved_grants
        ] == ["git.push"]
        await session.resources.config.reload()
        after_reload = await _wire_read(client, loop.session_id)
        assert after_reload.accepted_saved_grants == []
        assert after_reload.saved_operations == []
    finally:
        await session.close()
        await loop.aclose()


@pytest.mark.asyncio
async def test_child_surfaces_use_the_children_own_project(
    tmp_path: Path, config_dir: Path
) -> None:
    """Inspection and revocation key off the registered session's cwd."""
    project = make_repo(tmp_path / "project")
    nested = project / "nested"
    nested.mkdir()
    config_path = config_dir / "config.toml"
    loop = await _make_loop(project, config_path)
    server, client, session = await _open_session(loop)
    try:
        backend = _backend(server)
        registry = backend.children
        policy = loop.child_runtime_policy
        child = AgentLoop(
            config_orchestrator=loop.config_orchestrator.copy(),
            cwd=nested,
            backend=FakeBackend(),
            is_subagent=True,
            inherited_workspace=policy.inherited_workspace,
            inherited_restrictions=policy.inherited_restrictions,
            inherited_mode_restrictions=policy.inherited_mode_restrictions,
            inherited_plan_write_scopes=policy.inherited_plan_write_scopes,
            inherited_root_grants=policy.inherited_root_grants,
            user_input_capability=True,
            parent_authority_getter=policy.parent_authority_getter,
            parent_authority_revision_getter=policy.parent_authority_revision_getter,
        )
        registry._children[child.session_id] = registry._build_child_runtime(child)
        try:
            # The child's own live grant and saved grant (keyed by its cwd).
            child._dangerous_command_grants.grant_session({
                ("git.push", _scope_of(nested))
            })
            config_path.write_text(
                config_path.read_text()
                + f'\n[{GRANTS_FIELD}]\n"{nested.resolve()}" = ["git.push"]\n',
                encoding="utf-8",
            )
            read = await _wire_read(client, child.session_id)
            assert read.project == str(nested)
            assert [
                (grant.operation_id, grant.scope.git_dir) for grant in read.live_grants
            ] == [("git.push", str(_scope_of(nested).git_dir))]
            assert read.saved_operations == ["git.push"]
            # The root session's inspection is keyed by its own cwd.
            root_read = await _wire_read(client, loop.session_id)
            assert root_read.project == str(project)
            assert root_read.live_grants == []
            assert root_read.saved_operations == []
            # Revoking through the child removes the child's live authority
            # and the saved grant keyed by the child's cwd.
            response = await _wire_revoke(
                client,
                child.session_id,
                "git.push",
                read.live_authority_generation,
                read.disk_revision,
            )
            assert response.live.status == "removed"
            assert response.saved.status == "removed"
            assert _authorities(child) == []
            assert _authorities(loop) == []
            assert _saved_grants(config_path) == {}
        finally:
            await child.aclose()
    finally:
        await session.close()
        await loop.aclose()


def _write_saved_destinations(
    config_path: Path, *entries: tuple[Path, list[str]]
) -> None:
    """Append one remote_exec_grants_by_project table with the given entries."""
    lines = [f"\n[{REMOTE_EXEC_GRANTS_FIELD}]"]
    for project, destinations in entries:
        lines.append(f'"{project.resolve()}" = {json.dumps(destinations)}')
    config_path.write_text(
        config_path.read_text() + "\n".join(lines) + "\n", encoding="utf-8"
    )


@pytest.mark.asyncio
async def test_read_includes_destination_scoped_grants(
    tmp_path: Path, config_dir: Path
) -> None:
    """The read projection carries the exact destination on remote-exec entries."""
    project = make_repo(tmp_path / "project")
    config_path = config_dir / "config.toml"
    _write_saved_destinations(
        config_path, (project, ["deploy@203.0.113.10", "deploy@203.0.113.20:2222"])
    )
    loop = await _make_loop(project, config_path)
    server, client, session = await _open_session(loop)
    try:
        store = loop._dangerous_command_grants
        store.grant_remote_exec_session({
            _remote_exec_requirement(project, DESTINATION_A)
        })
        store.install_remote_exec_once(
            frozenset({_remote_exec_requirement(project, DESTINATION_B)})
        )
        wire = await _wire_read(client, loop.session_id)
        # Live entries: the session grant for A and the pending once permit
        # for B, each carrying its exact destination and the scope.
        assert [
            (grant.operation_id, grant.scope.git_dir, grant.destination)
            for grant in wire.live_grants
        ] == [
            (
                REMOTE_EXEC_OPERATION_ID,
                str(_scope_of(project).git_dir),
                DESTINATION_A.key,
            ),
            (
                REMOTE_EXEC_OPERATION_ID,
                str(_scope_of(project).git_dir),
                DESTINATION_B.key,
            ),
        ]
        # The accepted saved destinations project the same way.
        assert [
            (grant.operation_id, grant.destination)
            for grant in wire.accepted_saved_grants
        ] == [
            (REMOTE_EXEC_OPERATION_ID, DESTINATION_A.key),
            (REMOTE_EXEC_OPERATION_ID, DESTINATION_B.key),
        ]
        # The disk read reports operations and destinations separately.
        assert wire.saved_operations == []
        assert wire.saved_destinations == [DESTINATION_A.key, DESTINATION_B.key]
        assert wire.disk_revision
        # Non-remote-exec entries never carry a destination.
        store.grant_session({("git.push", _scope_of(project))})
        reread = await _wire_read(client, loop.session_id)
        push_entry = next(
            grant for grant in reread.live_grants if grant.operation_id == "git.push"
        )
        assert push_entry.destination is None
    finally:
        await session.close()
        await loop.aclose()


@pytest.mark.asyncio
async def test_destination_revoke_removes_only_that_destination_live_and_saved(
    tmp_path: Path, config_dir: Path
) -> None:
    """Selective removal: one destination goes live and on disk, the other stays."""
    project = make_repo(tmp_path / "project")
    config_path = config_dir / "config.toml"
    _write_saved_destinations(
        config_path, (project, ["deploy@203.0.113.10", "deploy@203.0.113.20:2222"])
    )
    loop = await _make_loop(project, config_path)
    server, client, session = await _open_session(loop)
    try:
        store = loop._dangerous_command_grants
        store.grant_remote_exec_session({
            _remote_exec_requirement(project, DESTINATION_A),
            _remote_exec_requirement(project, DESTINATION_B),
        })
        read = await _wire_read(client, loop.session_id)
        # The non-canonical spelling is accepted and canonicalized to the
        # exact destination identity.
        response = await _wire_revoke(
            client,
            loop.session_id,
            REMOTE_EXEC_OPERATION_ID,
            read.live_authority_generation,
            read.disk_revision,
            destination="deploy@203.0.113.10",
        )
        assert response.live.status == "removed"
        assert response.saved.status == "removed"
        assert response.saved.revision
        # Only destination A is gone live; B's session grant survives.
        assert _remote_exec_authorities(loop) == [
            (DESTINATION_B.key, str(_scope_of(project).git_dir))
        ]
        # Only A is removed from disk; B's saved destination survives.
        assert _saved_remote_exec_grants(config_path) == {
            str(project.resolve()): [DESTINATION_B.key]
        }
        # The accepted config still authorizes A in this session until a
        # reload, so the response warns for the exact destination.
        assert response.warning is not None
        assert (
            f"keeps {REMOTE_EXEC_OPERATION_ID} authority for {DESTINATION_A.key} "
            "from accepted config" in response.warning
        )
        after = await _wire_read(client, loop.session_id)
        assert [(grant.destination) for grant in after.live_grants] == [
            DESTINATION_B.key
        ]
        assert after.saved_destinations == [DESTINATION_B.key]
        assert sorted(
            grant.destination or "" for grant in after.accepted_saved_grants
        ) == [DESTINATION_A.key, DESTINATION_B.key]
    finally:
        await session.close()
        await loop.aclose()


@pytest.mark.asyncio
async def test_destination_revoke_invalidates_whole_once_permits_and_cached_authority(
    tmp_path: Path, config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A permit containing the revoked destination is invalidated whole; caches drop."""
    project = make_repo(tmp_path / "project")
    config_path = config_dir / "config.toml"
    loop = await _make_loop(project, config_path)
    server, client, session = await _open_session(loop)
    try:
        store = loop._dangerous_command_grants
        req_a = _remote_exec_requirement(project, DESTINATION_A)
        req_b = _remote_exec_requirement(project, DESTINATION_B)
        store.grant_remote_exec_session({req_a, req_b})
        # One pending once permit covering BOTH destinations, one covering B only.
        store.install_remote_exec_once(frozenset({req_a, req_b}))
        store.install_remote_exec_once(frozenset({req_b}))
        assert len(_remote_exec_authorities(loop)) == 5
        invalidations = 0
        original_invalidate = loop.tool_manager._invalidate_dangerous_command_authority

        def _counting_invalidate() -> None:
            nonlocal invalidations
            invalidations += 1
            original_invalidate()

        monkeypatch.setattr(
            loop.tool_manager,
            "_invalidate_dangerous_command_authority",
            _counting_invalidate,
        )
        read = await _wire_read(client, loop.session_id)
        response = await _wire_revoke(
            client,
            loop.session_id,
            REMOTE_EXEC_OPERATION_ID,
            read.live_authority_generation,
            read.disk_revision,
            destination=DESTINATION_A.key,
        )
        assert response.live.status == "removed"
        # The permit containing A is dropped WHOLE — its B coverage goes with
        # it — while the B-only permit survives, as does B's session grant.
        assert len(store._remote_exec_once) == 1
        assert [key for key, _ in _remote_exec_authorities(loop)] == [
            DESTINATION_B.key,
            DESTINATION_B.key,
        ]
        # Cached authority was invalidated (the revoke path drops the caches).
        assert invalidations >= 1
        assert response.live_authority_generation == read.live_authority_generation + 2
    finally:
        await session.close()
        await loop.aclose()


@pytest.mark.asyncio
async def test_destination_revoke_with_stale_live_generation_conflicts_and_leaves_saved_untouched(
    tmp_path: Path, config_dir: Path
) -> None:
    """A stale expected generation conflicts a destination-scoped revoke; disk untouched."""
    project = make_repo(tmp_path / "project")
    config_path = config_dir / "config.toml"
    _write_saved_destinations(config_path, (project, ["deploy@203.0.113.10"]))
    loop = await _make_loop(project, config_path)
    server, client, session = await _open_session(loop)
    try:
        store = loop._dangerous_command_grants
        store.grant_remote_exec_session({
            _remote_exec_requirement(project, DESTINATION_A)
        })
        read = await _wire_read(client, loop.session_id)
        # The live authority changes between the read and the revocation.
        store.grant_remote_exec_session({
            _remote_exec_requirement(project, DESTINATION_B)
        })
        response = await _wire_revoke(
            client,
            loop.session_id,
            REMOTE_EXEC_OPERATION_ID,
            read.live_authority_generation,
            read.disk_revision,
            destination=DESTINATION_A.key,
        )
        # A stale live generation is a conflict/re-read outcome: the
        # destination's live grant and its saved disk entry both survive.
        assert response.live.status == "conflict"
        assert response.live.generation == read.live_authority_generation + 1
        assert response.saved.status == "skipped"
        assert response.warning is None
        assert response.live_authority_generation == read.live_authority_generation + 1
        assert sorted(key for key, _ in _remote_exec_authorities(loop)) == [
            DESTINATION_A.key,
            DESTINATION_B.key,
        ]
        assert _saved_remote_exec_grants(config_path) == {
            str(project.resolve()): [DESTINATION_A.key]
        }
    finally:
        await session.close()
        await loop.aclose()


@pytest.mark.asyncio
async def test_destination_revoke_with_stale_disk_revision_reports_saved_conflict_but_removes_live(
    tmp_path: Path, config_dir: Path
) -> None:
    """A stale disk revision conflicts only the saved side of a destination revoke."""
    project = make_repo(tmp_path / "project")
    config_path = config_dir / "config.toml"
    _write_saved_destinations(
        config_path, (project, ["deploy@203.0.113.10", "deploy@203.0.113.20:2222"])
    )
    loop = await _make_loop(project, config_path)
    server, client, session = await _open_session(loop)
    try:
        store = loop._dangerous_command_grants
        store.grant_remote_exec_session({
            _remote_exec_requirement(project, DESTINATION_A)
        })
        read = await _wire_read(client, loop.session_id)
        # An intervening writer changes the user config after the read.
        config_path.write_text(
            config_path.read_text() + "\n# intervening writer\n", encoding="utf-8"
        )
        response = await _wire_revoke(
            client,
            loop.session_id,
            REMOTE_EXEC_OPERATION_ID,
            read.live_authority_generation,
            read.disk_revision,
            destination=DESTINATION_A.key,
        )
        # Live removal is applied; the saved removal reports its own conflict.
        assert response.live.status == "removed"
        assert response.saved.status == "conflict"
        assert response.warning is not None
        assert "can return on the next config reload or session start" in (
            response.warning
        )
        assert _remote_exec_authorities(loop) == []
        # Both saved destinations survive the conflicted disk write.
        assert _saved_remote_exec_grants(config_path) == {
            str(project.resolve()): [DESTINATION_A.key, DESTINATION_B.key]
        }
    finally:
        await session.close()
        await loop.aclose()


@pytest.mark.asyncio
async def test_destination_revoke_without_a_disk_revision_reports_separate_live_and_saved_outcomes(
    tmp_path: Path, config_dir: Path
) -> None:
    """Live removed, saved skipped: the two sides report apart under a destination."""
    project = make_repo(tmp_path / "project")
    config_path = config_dir / "config.toml"
    _write_saved_destinations(config_path, (project, ["deploy@203.0.113.10"]))
    loop = await _make_loop(project, config_path)
    server, client, session = await _open_session(loop)
    try:
        loop._dangerous_command_grants.grant_remote_exec_session({
            _remote_exec_requirement(project, DESTINATION_A)
        })
        read = await _wire_read(client, loop.session_id)
        response = await _wire_revoke(
            client,
            loop.session_id,
            REMOTE_EXEC_OPERATION_ID,
            read.live_authority_generation,
            None,
            destination=DESTINATION_A.key,
        )
        assert response.live.status == "removed"
        assert response.saved.status == "skipped"
        assert _remote_exec_authorities(loop) == []
        # The saved side was not attempted, but the accepted config still
        # authorizes the exact destination until a reload, so it warns.
        assert response.warning is not None
        assert (
            f"keeps {REMOTE_EXEC_OPERATION_ID} authority for {DESTINATION_A.key} "
            "from accepted config" in response.warning
        )
        assert _saved_remote_exec_grants(config_path) == {
            str(project.resolve()): [DESTINATION_A.key]
        }
    finally:
        await session.close()
        await loop.aclose()


@pytest.mark.asyncio
async def test_destination_revoke_child_scope(tmp_path: Path, config_dir: Path) -> None:
    """A destination revoke through a child keys off the child's own store and cwd."""
    project = make_repo(tmp_path / "project")
    nested = project / "nested"
    nested.mkdir()
    config_path = config_dir / "config.toml"
    _write_saved_destinations(
        config_path,
        (project, ["deploy@203.0.113.10"]),
        (nested, ["deploy@203.0.113.10"]),
    )
    loop = await _make_loop(project, config_path)
    server, client, session = await _open_session(loop)
    try:
        backend = _backend(server)
        registry = backend.children
        policy = loop.child_runtime_policy
        child = AgentLoop(
            config_orchestrator=loop.config_orchestrator.copy(),
            cwd=nested,
            backend=FakeBackend(),
            is_subagent=True,
            inherited_workspace=policy.inherited_workspace,
            inherited_restrictions=policy.inherited_restrictions,
            inherited_mode_restrictions=policy.inherited_mode_restrictions,
            inherited_plan_write_scopes=policy.inherited_plan_write_scopes,
            inherited_root_grants=policy.inherited_root_grants,
            user_input_capability=True,
            parent_authority_getter=policy.parent_authority_getter,
            parent_authority_revision_getter=policy.parent_authority_revision_getter,
        )
        registry._children[child.session_id] = registry._build_child_runtime(child)
        try:
            # The root holds destination A at the project scope; the child
            # holds A and B at its own (nested-cwd-keyed) scope.
            loop._dangerous_command_grants.grant_remote_exec_session({
                _remote_exec_requirement(project, DESTINATION_A)
            })
            child._dangerous_command_grants.grant_remote_exec_session({
                _remote_exec_requirement(nested, DESTINATION_A),
                _remote_exec_requirement(nested, DESTINATION_B),
            })
            read = await _wire_read(client, child.session_id)
            assert read.project == str(nested)
            assert sorted(grant.destination or "" for grant in read.live_grants) == [
                DESTINATION_A.key,
                DESTINATION_B.key,
            ]
            response = await _wire_revoke(
                client,
                child.session_id,
                REMOTE_EXEC_OPERATION_ID,
                read.live_authority_generation,
                read.disk_revision,
                destination=DESTINATION_A.key,
            )
            assert response.live.status == "removed"
            assert response.saved.status == "removed"
            # The child's live grant for A is gone; its B grant survives.
            assert [key for key, _ in _remote_exec_authorities(child)] == [
                DESTINATION_B.key
            ]
            # The root's own live grant for A is untouched.
            assert [key for key, _ in _remote_exec_authorities(loop)] == [
                DESTINATION_A.key
            ]
            # The disk removal is keyed by the child's cwd: the nested
            # entry is gone, the project entry survives.
            assert _saved_remote_exec_grants(config_path) == {
                str(project.resolve()): [DESTINATION_A.key]
            }
        finally:
            await child.aclose()
    finally:
        await session.close()
        await loop.aclose()


@pytest.mark.asyncio
async def test_destination_revoke_regrant_race_rechecks_and_warns(
    tmp_path: Path, config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pending approval answered during the revoke re-grants; the response warns."""
    project = make_repo(tmp_path / "project")
    config_path = config_dir / "config.toml"
    _write_saved_destinations(config_path, (project, ["deploy@203.0.113.10"]))
    loop = await _make_loop(project, config_path)
    server, client, session = await _open_session(loop)
    try:
        store = loop._dangerous_command_grants
        store.grant_remote_exec_session({
            _remote_exec_requirement(project, DESTINATION_A)
        })
        read = await _wire_read(client, loop.session_id)
        assert read.live_authority_generation == 1
        # Simulate a pending approval answered concurrently with the
        # revocation: the regrant lands while the saved-side disk write is
        # in flight.
        orchestrator = loop.config_orchestrator
        original_remove = orchestrator.remove_remote_exec_grant

        async def _racing_remove(**kwargs: Any) -> Any:
            store.grant_remote_exec_session({
                _remote_exec_requirement(project, DESTINATION_B)
            })
            return await original_remove(**kwargs)

        monkeypatch.setattr(orchestrator, "remove_remote_exec_grant", _racing_remove)
        response = await _wire_revoke(
            client,
            loop.session_id,
            REMOTE_EXEC_OPERATION_ID,
            read.live_authority_generation,
            read.disk_revision,
            destination=DESTINATION_A.key,
        )
        # The revoke itself reports removed on both sides...
        assert response.live.status == "removed"
        assert response.saved.status == "removed"
        # ...but the recheck sees the concurrent regrant: the generation moved
        # past the post-live-removal value and the response warns honestly.
        assert response.live_authority_generation == read.live_authority_generation + 2
        assert response.live.generation == response.live_authority_generation
        assert response.warning is not None
        assert "re-granted fresh authority" in response.warning
        # The re-granted destination is live again; the revoked one is gone
        # live and on disk.
        assert [key for key, _ in _remote_exec_authorities(loop)] == [DESTINATION_B.key]
        assert _saved_remote_exec_grants(config_path) == {}
    finally:
        await session.close()
        await loop.aclose()


@pytest.mark.asyncio
async def test_whole_scope_remote_exec_revoke_removes_every_saved_destination(
    tmp_path: Path, config_dir: Path
) -> None:
    """No destination parameter keeps the whole-scope revoke, disk included."""
    project = make_repo(tmp_path / "project")
    config_path = config_dir / "config.toml"
    _write_saved_destinations(
        config_path, (project, ["deploy@203.0.113.10", "deploy@203.0.113.20:2222"])
    )
    loop = await _make_loop(project, config_path)
    server, client, session = await _open_session(loop)
    try:
        store = loop._dangerous_command_grants
        req_a = _remote_exec_requirement(project, DESTINATION_A)
        req_b = _remote_exec_requirement(project, DESTINATION_B)
        store.grant_remote_exec_session({req_a, req_b})
        store.install_remote_exec_once(frozenset({req_a, req_b}))
        read = await _wire_read(client, loop.session_id)
        response = await _wire_revoke(
            client,
            loop.session_id,
            REMOTE_EXEC_OPERATION_ID,
            read.live_authority_generation,
            read.disk_revision,
        )
        assert response.live.status == "removed"
        assert response.saved.status == "removed"
        # Every live destination and every pending remote-exec permit is gone.
        assert _remote_exec_authorities(loop) == []
        assert store._remote_exec_once == []
        # Every saved destination of the project is removed from disk.
        assert _saved_remote_exec_grants(config_path) == {}
    finally:
        await session.close()
        await loop.aclose()


@pytest.mark.asyncio
async def test_destination_revoke_validates_the_destination_parameter(
    tmp_path: Path, config_dir: Path
) -> None:
    """Only network.remote-exec accepts a destination, and only a canonical one."""
    project = make_repo(tmp_path / "project")
    config_path = config_dir / "config.toml"
    loop = await _make_loop(project, config_path)
    server, client, session = await _open_session(loop)
    try:
        loop._dangerous_command_grants.grant_session({("git.push", _scope_of(project))})
        loop._dangerous_command_grants.grant_remote_exec_session({
            _remote_exec_requirement(project, DESTINATION_A)
        })
        read = await _wire_read(client, loop.session_id)
        base: dict[str, Any] = {
            "sessionId": loop.session_id,
            "expectedLiveGeneration": read.live_authority_generation,
            "expectedDiskRevision": read.disk_revision,
            "userInitiated": True,
        }
        # A destination on a scope-only operation is rejected.
        with pytest.raises(AppServerResponseError) as excinfo:
            await client.request(
                "policy/dangerous_commands/revoke",
                base | {"operationId": "git.push", "destination": DESTINATION_A.key},
            )
        assert excinfo.value.error.code is ProtocolErrorCode.INVALID_PARAMS
        # A non-canonical destination spelling fails closed.
        for bad in ("deploy@*", "not-a-destination", "deploy@203.0.113.10@x"):
            with pytest.raises(AppServerResponseError) as excinfo:
                await client.request(
                    "policy/dangerous_commands/revoke",
                    base
                    | {"operationId": REMOTE_EXEC_OPERATION_ID, "destination": bad},
                )
            assert excinfo.value.error.code is ProtocolErrorCode.INVALID_PARAMS
        # Nothing was revoked.
        assert sorted(key for key, _ in _remote_exec_authorities(loop)) == [
            DESTINATION_A.key
        ]
        assert _authorities(loop) == [
            (frozenset({"git.push"}), _scope_of(project)),
            (frozenset({REMOTE_EXEC_OPERATION_ID}), _scope_of(project)),
        ]
    finally:
        await session.close()
        await loop.aclose()


def test_denylist_probes_match_the_frozen_classifier() -> None:
    """Each denylist probe is exactly one registry operation's grantable form."""
    from chartreux.app_server._sessions import _DANGEROUS_COMMAND_DENYLIST_PROBES

    # Every registry operation has a probe; the git operations have exactly
    # one each, and the modeled network clients each get a network.egress
    # probe (v0.5.9 P2-a).
    assert {operation_id for operation_id, _ in _DANGEROUS_COMMAND_DENYLIST_PROBES} == {
        operation.operation_id for operation in OPERATION_REGISTRY.values()
    }
    for operation_id, probe in _DANGEROUS_COMMAND_DENYLIST_PROBES:
        if operation_id == "network.egress":
            # The egress probe is the operation's ask-class form, classified
            # by the destination recognizer, not the git classifier.
            assert classify_network_egress(list(probe)).kind == "ask"
            continue
        if operation_id == "network.remote-exec":
            # The remote-exec classifier lands with the SSH admission
            # (v0.7.0 S4b1); the probe is the representative SSH form.
            assert probe[0] == "ssh"
            continue
        classification = classify_dangerous_git_command(list(probe))
        assert classification.kind == "grantable"
        assert classification.grantable is not None
        assert classification.grantable.operation_ids == (operation_id,)


_ALL_OPERATIONS = [
    "git.push",
    "git.push.force",
    "git.push.delete",
    "git.push.mirror",
    "git.checkout",
    "git.restore",
    "git.switch.discard",
    "git.stash.drop",
    "git.stash.clear",
    "git.reflog.expire",
    "git.reflog.delete",
]


@pytest.mark.parametrize(
    ("pattern", "operations"),
    [
        ("git", _ALL_OPERATIONS),
        ("git push", DENY_PROJECTION_OPERATIONS),
        ("git push --force", ["git.push.force"]),
        ("git push origin --delete", ["git.push.delete"]),
        ("git stash", ["git.stash.drop", "git.stash.clear"]),
        ("git reflog", ["git.reflog.expire", "git.reflog.delete"]),
        ("git checkout -", ["git.checkout"]),
        # v0.5.9 P2-a: a denylist pattern naming a modeled network client
        # vetoes the egress operation through its probe.
        ("curl", ["network.egress"]),
        ("wget", ["network.egress"]),
        ("curl https://example.invalid/", ["network.egress"]),
        # v0.7.0 S4a: an SSH-client pattern vetoes the remote-exec operation.
        ("ssh", ["network.remote-exec"]),
    ],
)
def test_denylist_blocking_operations(pattern: str, operations: list[str]) -> None:
    from chartreux.app_server._sessions import _denylist_blocked_operations

    assert list(_denylist_blocked_operations(pattern)) == operations
