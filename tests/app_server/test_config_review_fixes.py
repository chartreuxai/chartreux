from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
import traceback
from unittest.mock import AsyncMock, patch

import pytest

from chartreux.app_server._runtime_resources import ConfigResource
from chartreux.app_server.client_state import ClientBootstrap, ClientSessionState
from chartreux.app_server.connection import AppServerResourceConnection
from chartreux.app_server.protocol import (
    ConfigFieldsReadParams,
    ConfigFieldsReadResponse,
    ConfigReloadParams,
    ConfigWriteOpWire,
    RuntimeReadParams,
    RuntimeReadResponse,
    SessionReadParams,
    SessionReadResponse,
)
from chartreux.core.config.layers.project import ProjectConfigLayer
from chartreux.core.config.types import ConfigChangeEvent
from chartreux.core.trusted_folders import trusted_folders_manager
from tests.app_server.test_explicit_config import opened, revision, write


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["preparation", "application"])
async def test_reload_exception_has_no_raw_cause_or_context(
    tmp_path: Path, stage: str
) -> None:
    from chartreux.app_server._dispatch import RequestFailure
    from chartreux.app_server._resources import ResourceRequestHandler
    from tests.app_server.test_tree_policy import ignore, tree

    async with tree(tmp_path) as (registry, runtimes):
        loop = runtimes[0].agent_loop
        resources = ResourceRequestHandler(
            loop, runtimes[0].execution, ignore, reserve_config=registry.reserve_config
        )
        before = loop.config
        manager = loop.tool_manager
        method = "_prepare_reload" if stage == "preparation" else "_commit_reload"
        with (
            patch.object(
                loop, method, side_effect=ValueError("reload-secret-sentinel")
            ),
            pytest.raises(RequestFailure) as caught,
        ):
            await resources._config_reload(
                ConfigReloadParams(session_id=loop.session_id)
            )
        assert caught.value.__cause__ is None
        assert caught.value.__context__ is None
        assert "reload-secret-sentinel" not in "".join(
            traceback.format_exception(caught.value)
        )
        assert loop.config is before
        assert loop.tool_manager is manager
        assert not registry._policy_reserved


@pytest.mark.asyncio
async def test_session_edit_notifies_real_orchestrator_and_keeps_wire_success(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.toml"
    async with opened(path) as (loop, client):
        seen: list[str] = []

        def observe(event: ConfigChangeEvent) -> None:
            assert loop.config.theme == "light"
            seen.append(event.reason)
            raise RuntimeError("subscriber-sentinel")

        loop.config_orchestrator.subscribe(observe)
        response = await write(loop, client)
        assert response.application == "applied", response.failures
        assert not response.rejected
        assert not response.failures
        assert len(seen) == 1
        assert not path.exists()


@pytest.mark.asyncio
async def test_session_edit_keeps_external_sources_unaccepted(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('theme = "accepted"\n')
    async with opened(path) as (loop, client):
        user = loop.config_orchestrator.get_layer("user-toml")
        accepted_revision = user.fingerprint
        path.write_text('theme = "external"\n')
        result = await write(loop, client, path="/auto_compact_threshold", value="50")
        assert result.application == "applied", result.failures
        assert loop.config.theme == "accepted"
        assert user.fingerprint == accepted_revision
        await client.request(
            "config/reload", ConfigReloadParams(session_id=loop.session_id)
        )
        assert loop.config.theme == "external"
        assert user.fingerprint != accepted_revision


@pytest.mark.asyncio
@pytest.mark.parametrize("unavailable", ["untrusted", "invalid", "storage"])
async def test_fields_read_preserves_user_ui_when_project_unavailable(
    tmp_path: Path, unavailable: str
) -> None:
    project = tmp_path / "project"
    directory = project / ".chartreux"
    directory.mkdir(parents=True)
    project_file = directory / "config.toml"
    project_file.write_text('theme = "project"\n')
    if unavailable != "untrusted":
        trusted_folders_manager.trust_for_session(directory)
    async with opened(tmp_path / "user.toml", project) as (loop, client):
        if unavailable == "invalid":
            project_file.write_text("invalid = [\n")
        with (
            patch.object(
                ProjectConfigLayer,
                "_build_config_snapshot",
                side_effect=OSError("unavailable"),
            )
            if unavailable == "storage"
            else nullcontext()
        ):
            response = ConfigFieldsReadResponse.model_validate(
                await client.request(
                    "config/fields/read",
                    ConfigFieldsReadParams(session_id=loop.session_id),
                )
            )
        assert response.fields
        assert "user-toml" in response.revisions
        assert "project-toml" not in response.revisions
        result = await write(
            loop, client, target="user", expected=response.revisions["user-toml"]
        )
        assert result.persistence == "saved", result.failures


@pytest.mark.asyncio
async def test_public_subscriber_failure_does_not_hide_save(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    async with opened(path) as (loop, client):
        session = SessionReadResponse.model_validate(
            await client.request(
                "session/read", SessionReadParams(session_id=loop.session_id)
            )
        )
        runtime = RuntimeReadResponse.model_validate(
            await client.request(
                "runtime/read", RuntimeReadParams(session_id=loop.session_id)
            )
        )
        state = ClientSessionState(
            ClientBootstrap(state=session.state, runtime=runtime)
        )
        connection = AsyncMock(spec=AppServerResourceConnection)
        connection.connect.return_value = client
        resource = ConfigResource(connection, state)
        seen: list[str] = []

        def fail(_config: object) -> None:
            raise RuntimeError("subscriber-sentinel")

        resource.subscribe(fail)
        resource.subscribe(lambda config: seen.append(config.theme))
        response = await resource.write(
            [ConfigWriteOpWire(op="set", path="/theme", value="light")],
            reason="test",
            target="user",
            expected_revision=await revision(loop, client),
        )
        assert (response.persistence, response.application) == ("saved", "applied")
        assert resource.current.theme == "light"
        assert seen == ["light"]
        assert 'theme = "light"' in path.read_text()
