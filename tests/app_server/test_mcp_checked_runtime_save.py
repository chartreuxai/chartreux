from __future__ import annotations

import asyncio
from contextlib import nullcontext
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import pytest

from chartreux.app_server._dispatch import RequestFailure
from chartreux.app_server._mcp_auth import MCPAuthenticationService
from chartreux.app_server._resources import ResourceRequestHandler
from chartreux.app_server._session_backend_impl import SessionBackendImpl
from chartreux.app_server._session_backend_port import SessionBackendError
from chartreux.app_server.mcp_catalog import MCPCatalogService
from chartreux.app_server.models import MCPSourceStatus
from chartreux.app_server.protocol import (
    AppServerResponseError,
    MCPAddParams,
    MCPCatalogMutationResponse,
    MCPLoginParams,
    MCPRemoveParams,
    MCPToggleParams,
    ProtocolErrorCode,
)
from chartreux.core.config.layers.user import UserConfigLayer
from chartreux.core.config.types import ConfigDurabilityError
from tests.app_server.test_explicit_config import opened
from tests.stubs.fake_mcp_registry import FakeMCPRegistry

_INITIAL = (
    '[[mcp_servers]]\nname = "local"\ntransport = "stdio"\ncommand = "fake-mcp"\n'
)


@pytest.mark.asyncio
async def test_configured_http_server_toggle_persists_and_projects_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "chartreux.app_server._session_backend_impl.MCPRegistry", FakeMCPRegistry
    )
    path = tmp_path / "config.toml"
    path.write_text(
        '[[mcp_servers]]\nname = "figma"\ntransport = "streamable-http"\n'
        'url = "https://figma.example/mcp"\n'
    )
    async with opened(path) as (loop, client):
        response = MCPCatalogMutationResponse.model_validate(
            await client.request(
                "mcp_catalog/toggle",
                MCPToggleParams(
                    session_id=loop.session_id, name="figma", disabled=True
                ),
            )
        )
        assert loop.config.mcp_servers[0].disabled is True
        assert "disabled = true" in path.read_text()
        assert response.runtime is not None
        source = next(s for s in response.runtime.mcp.sources if s.name == "figma")
        assert source.status is MCPSourceStatus.DISABLED


@pytest.mark.asyncio
async def test_configured_http_server_login_uses_configured_url_and_updates_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "chartreux.app_server._session_backend_impl.MCPRegistry", FakeMCPRegistry
    )
    path = tmp_path / "config.toml"
    initial = (
        '[[mcp_servers]]\nname = "figma"\ntransport = "streamable-http"\n'
        'url = "https://figma.example/mcp"\n'
        '[mcp_servers.auth]\ntype = "oauth"\nscopes = []\n'
    )
    path.write_text(initial)
    async with opened(path) as (loop, client):
        with (
            patch(
                "chartreux.app_server._mcp_auth.perform_oauth_login", new=AsyncMock()
            ) as login,
            patch.object(
                SessionBackendImpl,
                "authorization_changed",
                autospec=True,
                side_effect=SessionBackendImpl.authorization_changed,
            ) as changed,
        ):
            response = MCPCatalogMutationResponse.model_validate(
                await client.request(
                    "mcp_catalog/login",
                    MCPLoginParams(session_id=loop.session_id, name="figma"),
                )
            )
        assert login.await_args is not None
        assert login.await_args.args[0].url == "https://figma.example/mcp"
        assert changed.await_args is not None
        assert changed.await_args.kwargs["name"] == "figma"
        assert response.runtime is not None
        assert [source.name for source in response.runtime.mcp.sources] == ["figma"]
        assert path.read_text() == initial


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["add", "remove", "toggle", "tool"])
@pytest.mark.parametrize(
    "failure", [None, "preflight", "race", "uncertain", "apply", "convergence"]
)
async def test_catalog_checked_runtime_save(
    tmp_path: Path, operation: str, failure: str | None
) -> None:
    path = tmp_path / "config.toml"
    path.write_text(_INITIAL)
    async with opened(path) as (loop, client):
        config, manager, registry = loop.config, loop.tool_manager, loop.mcp_registry
        user = loop.config_orchestrator.get_layer("user-toml")
        assert isinstance(user, UserConfigLayer)
        prepare = loop._prepare_reload
        save = user.save_checked
        prepared = []
        writes = []
        external = _INITIAL + '\ntheme = "external"\n'

        def prepare_runtime(*args, **kwargs):
            assert path.read_text() == _INITIAL
            assert loop.config is config
            assert loop.tool_manager is manager
            assert loop.mcp_registry is registry
            if failure == "preflight":
                raise ValueError("PRIVATE preparation detail")
            result = prepare(*args, **kwargs)
            prepared.append(result)
            assert isinstance(kwargs["mcp_registry"], FakeMCPRegistry)
            assert kwargs["mcp_registry"] is not registry
            if failure == "race":
                path.write_text(external)
            return result

        async def write_source(*args, **kwargs):
            writes.append(True)
            assert prepared, "Production runtime preflight must precede source IO"
            revision = await save(*args, **kwargs)
            if failure == "uncertain":
                raise ConfigDurabilityError(revision)
            return revision

        params = (
            MCPAddParams(
                session_id=loop.session_id,
                name="remote",
                url="https://mcp.example.test/mcp",
            )
            if operation == "add"
            else MCPRemoveParams(session_id=loop.session_id, name="local")
            if operation == "remove"
            else MCPToggleParams(
                session_id=loop.session_id,
                name="local",
                disabled=True,
                tool_name="echo" if operation == "tool" else None,
            )
        )
        method = f"mcp_catalog/{'toggle' if operation == 'tool' else operation}"
        commit = loop._commit_reload

        def apply_runtime(*args, **kwargs):
            assert path.read_text() != _INITIAL
            if failure == "apply":
                raise ValueError("PRIVATE application detail")
            return commit(*args, **kwargs)

        with (
            patch(
                "chartreux.app_server._session_backend_impl.MCPRegistry",
                FakeMCPRegistry,
            ),
            patch.object(loop, "_prepare_reload", side_effect=prepare_runtime) as hook,
            patch.object(loop, "_commit_reload", side_effect=apply_runtime),
            patch.object(user, "save_checked", side_effect=write_source),
            patch(
                "chartreux.core.tools.manager.ToolManager.reconfigure_mcp_async",
                side_effect=ValueError("PRIVATE convergence detail"),
            )
            if failure == "convergence"
            else nullcontext(),
        ):
            if failure is None:
                await client.request(method, params)
            else:
                with pytest.raises(AppServerResponseError) as caught:
                    await client.request(method, params)
                message = caught.value.error.message
                assert "PRIVATE" not in message
                expected = {
                    "preflight": ("not_saved", "unchanged", "validation"),
                    "race": ("not_saved", "unchanged", "conflict"),
                    "uncertain": ("durability_uncertain", "applied", None),
                    "apply": ("saved", "failed", "application"),
                    "convergence": ("saved", "failed", "convergence"),
                }[failure]
                assert caught.value.error.data == dict(
                    zip(("persistence", "application", "error"), expected, strict=True)
                )
                if failure == "race":
                    assert caught.value.error.code == ProtocolErrorCode.CONFLICT
                elif failure == "uncertain":
                    assert "durability_uncertain" in message
                    assert "application=applied" in message
                elif failure in {"apply", "convergence"}:
                    assert "persistence=saved" in message
                    assert "application=failed" in message
            hook.assert_called_once()
        if failure in {"preflight", "race", "apply"}:
            assert loop.config is config
            assert loop.tool_manager is manager
            assert loop.mcp_registry is registry
        else:
            assert loop.config is not config
            assert loop.tool_manager is prepared[0].tool_manager
            assert loop.mcp_registry is not registry
        if failure == "preflight":
            assert path.read_text() == _INITIAL
            assert not writes
        elif failure == "race":
            assert path.read_text() == external
        else:
            assert path.read_text() != _INITIAL


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["queued", "writer", "convergence"])
async def test_catalog_holds_existing_tree_admission(
    tmp_path: Path, boundary: str
) -> None:
    from tests.app_server.test_tree_policy import ignore, queued, tree

    async with tree(tmp_path) as (registry, runtimes):
        runtime = runtimes[0]
        loop = runtime.agent_loop
        resources = ResourceRequestHandler(loop, runtime.execution, ignore)
        backend = SessionBackendImpl(
            session=runtime,
            resources=resources,
            coordinator=cast(Any, None),
            handler=cast(Any, None),
            children=registry,
            record_last_session=lambda *args: None,
        )
        service = MCPCatalogService(MCPAuthenticationService())
        user = loop.config_orchestrator.get_layer("user-toml")
        assert isinstance(user, UserConfigLayer)
        original_save = user.save_checked
        entered, release = asyncio.Event(), asyncio.Event()
        before = loop.config, loop.tool_manager, loop.mcp_registry

        async def save(*args, **kwargs):
            assert registry._policy_reserved
            if boundary == "writer":
                entered.set()
                await release.wait()
            return await original_save(*args, **kwargs)

        async def converge(*args, **kwargs):
            assert registry._policy_reserved
            entered.set()
            await release.wait()

        async def mutate():
            return await service.dispatch(
                "mcp_catalog/add",
                MCPAddParams(
                    session_id=loop.session_id,
                    name="remote",
                    url="https://mcp.example.test/mcp",
                ).model_dump(by_alias=True),
                root=backend,
                notify=ignore,
            )

        if boundary == "queued":
            runtimes[-1].turns.enqueue(queued(runtimes[-1].agent_loop.session_id))
        with (
            patch(
                "chartreux.app_server._session_backend_impl.MCPRegistry",
                FakeMCPRegistry,
            ),
            patch.object(user, "save_checked", side_effect=save) as writer,
            patch.object(
                loop, "_prepare_reload", wraps=loop._prepare_reload
            ) as prepare,
            patch.object(
                SessionBackendImpl, "suspend_mcp", new_callable=AsyncMock
            ) as suspend,
        ):
            if boundary == "queued":
                with pytest.raises(SessionBackendError) as caught:
                    await mutate()
                assert caught.value.code == ProtocolErrorCode.CONFLICT
                writer.assert_not_called()
                prepare.assert_not_called()
                suspend.assert_not_called()
                assert (loop.config, loop.tool_manager, loop.mcp_registry) == before
            else:
                with patch(
                    "chartreux.core.tools.manager.ToolManager.reconfigure_mcp_async",
                    side_effect=converge,
                ):
                    task = asyncio.create_task(mutate())
                    await asyncio.wait_for(entered.wait(), 2)
                    task.cancel()
                    await asyncio.sleep(0)
                    assert registry._policy_reserved
                    assert not task.done()
                    with pytest.raises(RuntimeError):
                        runtimes[-1].turns.enqueue(
                            queued(runtimes[-1].agent_loop.session_id)
                        )
                    release.set()
                    if boundary == "writer":
                        with pytest.raises(
                            RequestFailure,
                            match="persistence=saved; application=unchanged; error=cancelled",
                        ):
                            await task
                        assert (
                            loop.config,
                            loop.tool_manager,
                            loop.mcp_registry,
                        ) == before
                    else:
                        result = await task
                        assert result.runtime_updated
                        assert loop.tool_manager is not before[1]
        assert not registry._policy_reserved
