from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from unittest.mock import patch

import pytest

from chartreux.app_server.client import AppServerClient
from chartreux.app_server.protocol import (
    ClientInfo,
    ConfigFieldsReadParams,
    ConfigFieldsReadResponse,
    ConfigWriteOpWire,
    ConfigWriteParams,
    ConfigWriteResponse,
    SessionStartParams,
)
from chartreux.app_server.transport import memory_transport_pair
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
from chartreux.core.config.layers.default import DefaultConfigLayer
from chartreux.core.config.layers.overrides import OverridesLayer
from chartreux.core.config.layers.project import ProjectConfigLayer
from chartreux.core.config.layers.user import UserConfigLayer
from chartreux.core.config.orchestrator import ConfigOrchestrator
from tests.stubs.app_server import build_test_app_server
from tests.stubs.fake_backend import FakeBackend
from tests.stubs.fake_mcp_registry import FakeMCPRegistry


@asynccontextmanager
async def opened(
    path: Path, project: Path | None = None
) -> AsyncIterator[tuple[AgentLoop, AppServerClient]]:
    session = OverridesLayer(data={"session_logging": {"enabled": False}})
    orchestrator = await ConfigOrchestrator.create(
        schema=ChartreuxConfigSchema,
        layers=[
            DefaultConfigLayer(schema=ChartreuxConfigSchema),
            UserConfigLayer(path=path),
            *([ProjectConfigLayer(path=project)] if project is not None else []),
            session,
        ],
        default_layer_resolver=lambda: session,
    )
    loop = AgentLoop(
        config_orchestrator=orchestrator,
        backend=FakeBackend(),
        mcp_registry=FakeMCPRegistry(),
    )
    client_transport, server_transport = memory_transport_pair()
    server = build_test_app_server(loop, server_transport)
    client = AppServerClient(client_transport, run_peer=server.serve)
    try:
        await client.initialize(ClientInfo(name="explicit-config", version="1"))
        await client.notify("initialized")
        await client.request("session/start", SessionStartParams())
        yield loop, client
    finally:
        await client_transport.close()
        await server_transport.close()
        await loop.aclose()


async def revision(loop: AgentLoop, client: AppServerClient) -> str:
    result = ConfigFieldsReadResponse.model_validate(
        await client.request(
            "config/fields/read", ConfigFieldsReadParams(session_id=loop.session_id)
        )
    )
    return result.revisions["user-toml"]


async def write(
    loop: AgentLoop,
    client: AppServerClient,
    *,
    target: Literal["session", "user", "project"] = "session",
    expected: str | None = None,
    path: str = "/theme",
    value: str = "light",
) -> ConfigWriteResponse:
    return ConfigWriteResponse.model_validate(
        await client.request(
            "config/write",
            ConfigWriteParams(
                session_id=loop.session_id,
                ops=[ConfigWriteOpWire(op="set", path=path, value=value)],
                target=target,
                expected_revision=expected,
            ),
        )
    )


@pytest.mark.asyncio
async def test_session_default_never_creates_file(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    async with opened(path) as (loop, client):
        result = await write(loop, client)
        assert result.application == "applied", result.failures
        assert result.persistence == "not_saved"
        assert loop.config.theme == "light"
        assert not path.exists()


@pytest.mark.asyncio
async def test_project_save_changes_only_selected_source(tmp_path: Path) -> None:
    from chartreux.core.trusted_folders import trusted_folders_manager

    project = tmp_path / "project"
    config_dir = project / ".chartreux"
    config_dir.mkdir(parents=True)
    project_file = config_dir / "config.toml"
    project_file.write_text('theme = "project-theme"\n')
    trusted_folders_manager.add_trusted(config_dir)
    user = tmp_path / "user.toml"
    user.write_text('theme = "user-theme"\n')
    async with opened(user, project) as (loop, client):
        fields = ConfigFieldsReadResponse.model_validate(
            await client.request(
                "config/fields/read", ConfigFieldsReadParams(session_id=loop.session_id)
            )
        )
        result = await write(
            loop, client, target="project", expected=fields.revisions["project-toml"]
        )
        assert result.persistence == "saved", result.failures
        assert result.application == "applied"
        assert result.saved_values == {"theme": "light"}
        assert result.fields[0].value == "light"
        assert result.fields[0].origin == "project-toml"
        assert user.read_text() == 'theme = "user-theme"\n'
        assert project_file.read_text() == 'theme = "light"\n'


@pytest.mark.asyncio
async def test_explicit_save_revision_shadow_and_conflict(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    async with opened(path) as (loop, client):
        rev = await revision(loop, client)
        assert (
            await write(loop, client, value="session-theme")
        ).application == "applied"
        result = await write(loop, client, target="user", expected=rev)
        assert (result.persistence, result.application) == ("saved", "applied"), (
            result.failures
        )
        assert loop.config.theme == "session-theme"
        assert 'theme = "light"' in path.read_text()
        before = path.read_bytes()
        result = await write(
            loop, client, target="user", expected=rev, value="conflict"
        )
        assert result.persistence == "not_saved"
        assert result.failures == ["conflict"]
        assert path.read_bytes() == before
        assert loop.config.theme == "session-theme"


@pytest.mark.asyncio
@pytest.mark.parametrize("target", ["session", "user"])
async def test_prompt_preparation_failure_preserves_active_state(
    tmp_path: Path, target: Literal["session", "user"]
) -> None:
    path = tmp_path / "config.toml"
    async with opened(path) as (loop, client):
        config, manager, backend = loop.config, loop.tool_manager, loop.backend
        rev = await revision(loop, client) if target == "user" else None
        result = await write(
            loop,
            client,
            target=target,
            expected=rev,
            path="/compaction_prompt_id",
            value="missing-prompt-SENSITIVE",
        )
        assert result.rejected
        assert "SENSITIVE" not in str(result.failures)
        assert loop.config is config
        assert loop.tool_manager is manager
        assert loop.backend is backend
        assert not path.exists()


@pytest.mark.asyncio
async def test_saved_but_not_applied_is_not_rolled_back(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    async with opened(path) as (loop, client):
        config, manager = loop.config, loop.tool_manager
        rev = await revision(loop, client)
        with patch.object(
            loop, "_commit_reload", side_effect=RuntimeError("SENSITIVE")
        ):
            result = await write(loop, client, target="user", expected=rev)
        assert (result.persistence, result.application) == ("saved", "failed")
        assert result.failures == ["application"]
        assert loop.config is config
        assert loop.tool_manager is manager
        assert 'theme = "light"' in path.read_text()


@pytest.mark.asyncio
async def test_tree_reservation_rejects_queued_descendant(tmp_path: Path) -> None:
    from tests.app_server.test_tree_policy import queued, tree

    async with tree(tmp_path) as (registry, runtimes):
        runtimes[-1].turns.enqueue(queued(runtimes[-1].agent_loop.session_id))
        with pytest.raises(RuntimeError), registry.reserve_config():
            pytest.fail("Queued descendants must prevent admission")
        assert not registry._policy_reserved


@pytest.mark.asyncio
async def test_cancelled_writer_drains_under_tree_reservation(tmp_path: Path) -> None:
    from chartreux.app_server._resources import ResourceRequestHandler
    from tests.app_server.test_tree_policy import ignore, queued, tree

    async with tree(tmp_path) as (registry, runtimes):
        loop = runtimes[0].agent_loop
        resources = ResourceRequestHandler(
            loop, runtimes[0].execution, ignore, reserve_config=registry.reserve_config
        )
        user = loop.config_orchestrator.get_layer("user-toml")
        assert isinstance(user, UserConfigLayer)
        entered, release = asyncio.Event(), asyncio.Event()
        original = user.save_checked

        async def slow_save(*args, **kwargs):
            entered.set()
            await release.wait()
            return await original(*args, **kwargs)

        before = loop.config
        with patch.object(user, "save_checked", side_effect=slow_save):
            task = asyncio.create_task(
                resources._config_write(
                    ConfigWriteParams(
                        session_id=loop.session_id,
                        target="user",
                        expected_revision=user.fingerprint,
                        ops=[ConfigWriteOpWire(op="set", path="/theme", value="light")],
                    )
                )
            )
            await entered.wait()
            task.cancel()
            await asyncio.sleep(0)
            assert registry._policy_reserved
            assert not task.done()
            with pytest.raises(RuntimeError):
                runtimes[-1].turns.enqueue(queued(runtimes[-1].agent_loop.session_id))
            release.set()
            result = await task
        assert result.persistence == "saved"
        assert result.application == "unchanged"
        assert result.failures == ["cancelled"]
        assert loop.config is before
        assert not registry._policy_reserved


@pytest.mark.asyncio
async def test_runtime_tls_precedes_backend_and_failed_preparation_restores_policy(
    tmp_path: Path,
) -> None:
    async with opened(tmp_path / "config.toml") as (loop, _):
        previous = loop.config.enable_system_trust_store
        candidate = loop.config.model_copy(
            update={"enable_system_trust_store": not previous}
        )

        def fail_backend(config):
            assert config.enable_system_trust_store is (not previous)
            raise RuntimeError("backend preparation failed")

        with patch.object(loop, "backend_factory", side_effect=fail_backend):
            with pytest.raises(RuntimeError, match="backend preparation failed"):
                loop._prepare_reload(candidate, False)
        assert loop.config.enable_system_trust_store is previous


@pytest.mark.asyncio
async def test_startup_applies_tls_before_backend_construction(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("enable_system_trust_store = true\n")
    original = AgentLoop.backend_factory
    seen = []

    def backend_factory(loop, *args, **kwargs):
        seen.append((args[0] if args else loop.config).enable_system_trust_store)
        return original(loop, *args, **kwargs)

    with patch.object(AgentLoop, "backend_factory", backend_factory):
        async with opened(path):
            assert seen and all(seen)


@pytest.mark.asyncio
async def test_late_prompt_failure_precedes_runtime_swap(tmp_path: Path) -> None:
    async with opened(tmp_path / "config.toml") as (loop, _):
        config, manager, backend = loop.config, loop.tool_manager, loop.backend
        prepared = loop._prepare_reload(config, False)
        loop._skills_adopted += 1
        with patch.object(
            loop, "_render_system_prompt", side_effect=ValueError("prompt failed")
        ):
            with pytest.raises(ValueError, match="prompt failed"):
                loop._commit_reload(prepared, False)
        assert loop.config is config
        assert loop.tool_manager is manager
        assert loop.backend is backend
