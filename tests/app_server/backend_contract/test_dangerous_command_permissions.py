"""Backend-contract wire tests for dangerous-command permissions.

The full harness (in-process transport, real runtime construction) binds the
loop's DangerousCommandGrantPersistencePort at session registration, so a
mid-turn dangerous-command question flows through the user-input callback,
"Always for this project" persists through the root orchestrator, and the
settings surfaces inspect and revoke the authority over the same wire the
in-process surfaces use.
"""

from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
import subprocess
import tomllib
from typing import Any

import httpx
import pytest
import respx

from chartreux.app_server._runtime import create_harness_server
from chartreux.app_server.client import AppServerClient
from chartreux.app_server.events import CallbackRequested
from chartreux.app_server.host import AppServerHost
from chartreux.app_server.models import (
    UserAnswer,
    UserInputCallbackDetail,
    UserInputCallbackOutput,
    UserQuestionResult,
)
from chartreux.app_server.protocol import (
    SERVER_METHODS,
    AppServerResponseError,
    ClientCapabilities,
    ClientInfo,
    ProtocolErrorCode,
    SessionOptions,
)
from chartreux.app_server.transport import memory_transport_pair
from chartreux.core.config._dangerous_command_authority import GRANTS_FIELD
from chartreux.utils import AgentEntrypoint
from tests.app_server.backend_contract.conftest import (
    BackendContractConnection,
    connect_backend_contract_host,
)


def _make_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    return path


@pytest.fixture
def dangerous_project(tmp_path: Path) -> Path:
    return _make_repo(tmp_path / "project")


@pytest.fixture
def backend_contract_session_options(dangerous_project: Path) -> SessionOptions:
    return SessionOptions(cwd=str(dangerous_project))


@pytest.fixture
def backend_contract_capabilities() -> ClientCapabilities:
    return ClientCapabilities(callback_kinds=["user_input"])


def _bash_turn(
    build_response: Callable[..., httpx.Response], command: str, call_id: str
) -> httpx.Response:
    return build_response(
        "",
        tool_calls=[
            {
                "id": call_id,
                "index": 0,
                "function": {
                    "name": "bash",
                    "arguments": json.dumps({"command": command}),
                },
            }
        ],
    )


def _saved_grants(config_path: Path) -> dict[str, list[str]]:
    if not config_path.exists():
        return {}
    data: dict[str, Any] = tomllib.loads(config_path.read_text())
    grants = data.get(GRANTS_FIELD, {})
    return {str(key): [str(item) for item in value] for key, value in grants.items()}


async def _connect(entrypoint: AgentEntrypoint, project: Path) -> AppServerHost:
    """Connect a CLI-capable client so the session may prompt for input."""
    client_transport, server_transport = memory_transport_pair()
    harness = await create_harness_server(server_transport, transport_kind="in_process")
    client = AppServerClient(client_transport, run_peer=harness.serve)
    return await AppServerHost.connect(
        client,
        client_info=ClientInfo(
            name="dangerous-backend-contract-test", version="0", entrypoint=entrypoint
        ),
        capabilities=ClientCapabilities(callback_kinds=["user_input"]),
        session_options=SessionOptions(cwd=str(project)),
        client_factory=harness.connect_client,
    )


@pytest.mark.asyncio
async def test_prompt_persist_inspect_and_revoke_round_trip_over_the_wire(
    config_dir: Path,
    dangerous_project: Path,
    backend_contract_mistral_api: respx.Route,
    backend_contract_mistral_response: Callable[..., httpx.Response],
) -> None:
    marker = dangerous_project / "executed.txt"
    config_path = config_dir / "config.toml"
    backend_contract_mistral_api.mock(
        side_effect=[
            _bash_turn(
                backend_contract_mistral_response,
                f"git push; touch {marker.name}",
                "push-1",
            ),
            backend_contract_mistral_response("done"),
        ]
    )
    host = await _connect("cli", dangerous_project)
    session = await host.open_session()
    try:
        callbacks: list[str] = []
        async for event in session.act("publish"):
            if isinstance(event, CallbackRequested):
                callbacks.append(event.callback.callback_id)
                detail = event.callback.detail
                assert isinstance(detail, UserInputCallbackDetail)
                question = detail.request.questions[0]
                assert question.header == "Dangerous command"
                assert "git.push" in question.question
                await session.respond_to_callback(
                    event.callback.callback_id,
                    UserInputCallbackOutput(
                        result=UserQuestionResult(
                            answers=[
                                UserAnswer(
                                    question="grant", answer="Always for this project"
                                )
                            ]
                        )
                    ),
                )

        # The prompt reached the client and the answer resolved the request:
        # the original call resumed and executed.
        assert len(callbacks) == 1
        assert marker.exists()

        # The approval persisted through the root orchestrator, keyed by the
        # registered session's cwd.
        assert _saved_grants(config_path) == {
            str(dangerous_project.resolve()): ["git.push"]
        }

        # Inspection reports the live grant and the disk-saved grant with its
        # revision; the mid-turn save never applies to the running session's
        # accepted config, exactly as the persistence contract promises.
        read = await session.resources.config.read_dangerous_command_authority()
        assert read.project == str(dangerous_project)
        assert [grant.operation_id for grant in read.live_grants] == ["git.push"]
        assert read.accepted_saved_grants == []
        assert read.saved_operations == ["git.push"]
        assert read.disk_revision

        # Revocation removes live and saved authority with separate outcomes.
        revoked = await session.resources.config.revoke_dangerous_command_grant(
            operation_id="git.push",
            expected_live_generation=read.live_authority_generation,
            expected_disk_revision=read.disk_revision,
            user_initiated=True,
        )
        assert revoked.live.status == "removed"
        assert revoked.saved.status == "removed"
        assert revoked.warning is None
        assert _saved_grants(config_path) == {}

        after = await session.resources.config.read_dangerous_command_authority()
        assert after.live_grants == []
        assert after.saved_operations == []
    finally:
        await session.close()
        await host.close()


@pytest.mark.asyncio
async def test_dangerous_command_surfaces_reject_unregistered_sessions(
    backend_contract_connection: BackendContractConnection,
) -> None:
    session = await backend_contract_connection.host.open_session()
    try:
        client = backend_contract_connection.client
        with pytest.raises(AppServerResponseError) as excinfo:
            await client.request(
                "policy/dangerous_commands/read", {"sessionId": "unregistered-session"}
            )
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
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_revocation_requires_the_literal_user_initiation(
    dangerous_project: Path, backend_contract_connection: BackendContractConnection
) -> None:
    session = await backend_contract_connection.host.open_session()
    try:
        # Only the literal true passes: coerced truthy values are rejected
        # like any other non-literal.
        for rejected in (False, 1, "true"):
            with pytest.raises(AppServerResponseError) as excinfo:
                await backend_contract_connection.client.request(
                    "policy/dangerous_commands/revoke",
                    {
                        "sessionId": session.session_id,
                        "operationId": "git.push",
                        "expectedLiveGeneration": 0,
                        "expectedDiskRevision": None,
                        "userInitiated": rejected,
                    },
                )
            assert excinfo.value.error.code is ProtocolErrorCode.INVALID_PARAMS
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_no_public_grant_endpoint_exists(
    backend_contract_connection: BackendContractConnection,
) -> None:
    """Grants are created by runtime callback completion only."""
    assert not [
        method
        for method in SERVER_METHODS
        if method.startswith("policy/dangerous_commands/")
        and method
        not in {"policy/dangerous_commands/read", "policy/dangerous_commands/revoke"}
    ]
    session = await backend_contract_connection.host.open_session()
    try:
        with pytest.raises(AppServerResponseError) as excinfo:
            await backend_contract_connection.client.request(
                "policy/dangerous_commands/save",
                {
                    "sessionId": session.session_id,
                    "operationId": "git.push",
                    "expectedDiskRevision": "rev",
                    "userInitiated": True,
                },
            )
        assert excinfo.value.error.code is ProtocolErrorCode.METHOD_NOT_FOUND
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_a_fresh_session_reads_the_saved_grant_from_disk(
    config_dir: Path,
    dangerous_project: Path,
    backend_contract_mistral_api: respx.Route,
    backend_contract_mistral_response: Callable[..., httpx.Response],
) -> None:
    config_path = config_dir / "config.toml"
    config_path.write_text(
        config_path.read_text()
        + f'\n[{GRANTS_FIELD}]\n"{dangerous_project.resolve()}" = ["git.push"]\n',
        encoding="utf-8",
    )
    backend_contract_mistral_api.mock(
        return_value=backend_contract_mistral_response("done")
    )
    connection = await connect_backend_contract_host(
        session_options=SessionOptions(cwd=str(dangerous_project)),
        capabilities=ClientCapabilities(callback_kinds=["user_input"]),
    )
    session = await connection.host.open_session()
    try:
        read = await session.resources.config.read_dangerous_command_authority()
        assert read.saved_operations == ["git.push"]
        assert [grant.operation_id for grant in read.accepted_saved_grants] == [
            "git.push"
        ]
        assert read.live_grants == []
        # Revoking the saved grant removes it from disk; the running session's
        # accepted view persists until a reload, per the persistence contract.
        revoked = await session.resources.config.revoke_dangerous_command_grant(
            operation_id="git.push",
            expected_live_generation=read.live_authority_generation,
            expected_disk_revision=read.disk_revision,
            user_initiated=True,
        )
        assert revoked.live.status == "removed"
        assert revoked.saved.status == "removed"
        assert _saved_grants(config_path) == {}
        after = await session.resources.config.read_dangerous_command_authority()
        assert after.saved_operations == []
        assert [grant.operation_id for grant in after.accepted_saved_grants] == [
            "git.push"
        ]
    finally:
        await session.close()
        await connection.host.close()
