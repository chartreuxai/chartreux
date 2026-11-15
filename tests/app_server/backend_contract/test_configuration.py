from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
import json
from pathlib import Path
import tomllib

import httpx
import pytest
import pytest_asyncio
import respx
import tomli_w

from chartreux.app_server.protocol import (
    AppServerResponseError,
    ClientCapabilities,
    ConfigFieldsReadParams,
    ConfigFieldsReadResponse,
    ConfigProxyWriteParams,
    ConfigWriteOpWire,
    ProtocolErrorCode,
    SessionOptions,
    SettingsReadParams,
    SettingsReadResponse,
)
from chartreux.app_server.session import AppServerSession
from chartreux.app_server.transport import STDIO_TRANSPORT
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.config import build_default_orchestrator
from chartreux.core.config.harness_files import HarnessFilesManager
from tests.app_server.backend_contract.conftest import (
    BackendContractConnection,
    connect_backend_contract_host,
    connect_backend_contract_stdio_host,
    install_memory_root_agent,
)
from tests.stubs.fake_backend import FakeBackend
from tests.stubs.fake_mcp_registry import FakeMCPRegistry


async def _layered_agent_loop() -> AgentLoop:
    """A loop whose orchestrator is the real layered stack over this test's config.

    The configuration reads project the agent's own orchestrator and its
    layered sources, so the stdio leg's memory root runs the shipped
    orchestrator rather than an in-memory double: a read answered on either
    route is then comparable leaf for leaf.
    """
    manager = HarnessFilesManager(sources=("user", "project"))
    return AgentLoop(
        config_orchestrator=await build_default_orchestrator(
            harness_files=manager.for_session(Path.cwd())
        ),
        backend=FakeBackend(),
        mcp_registry=FakeMCPRegistry(),
    )


@pytest_asyncio.fixture
async def configuration_connection(
    monkeypatch: pytest.MonkeyPatch,
    stdio_after_autouse: str,
    backend_contract_session_options: SessionOptions,
    backend_contract_capabilities: ClientCapabilities,
) -> AsyncIterator[BackendContractConnection]:
    """One connection on the node's transport, its agent owning the real config."""
    loop: AgentLoop | None = None
    if stdio_after_autouse == STDIO_TRANSPORT:
        loop = await _layered_agent_loop()
        install_memory_root_agent(monkeypatch, loop)
        connection = await connect_backend_contract_stdio_host(
            session_options=backend_contract_session_options,
            capabilities=backend_contract_capabilities,
        )
    else:
        connection = await connect_backend_contract_host(
            session_options=backend_contract_session_options,
            capabilities=backend_contract_capabilities,
        )
    try:
        yield connection
    finally:
        await connection.host.close()
        if loop is not None:
            await loop.aclose()


@pytest_asyncio.fixture
async def configuration_session(
    configuration_connection: BackendContractConnection,
    backend_contract_mistral_api: respx.Route,
) -> AsyncIterator[AppServerSession]:
    """One session on the node's transport with an agent-owned layered config."""
    session = await configuration_connection.host.open_session()
    try:
        yield session
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_session_configuration_updates_and_reloads_through_the_backend(
    backend_contract_session: AppServerSession,
) -> None:
    await backend_contract_session.resources.sessions.update_settings(
        max_turns=4, max_tokens=1024
    )
    await backend_contract_session.resources.config.update({"theme": "dark"})
    reload_result = await backend_contract_session.resources.config.reload(
        reload_runtime=False
    )

    assert backend_contract_session.resources.config.current.theme == "dark"
    assert reload_result.stripped_history_images == 0
    assert reload_result.launch_metadata_persisted is True


@pytest.mark.asyncio
async def test_config_updates_refresh_the_public_runtime_tool_catalog(
    backend_contract_session: AppServerSession,
) -> None:
    assert backend_contract_session.resources.runtime.has_tool("grep")

    await backend_contract_session.resources.config.update(
        {"disabled_tools": ["grep"]}, reload_runtime=True
    )

    assert not backend_contract_session.resources.runtime.has_tool("grep")


@pytest.mark.asyncio
async def test_config_subscribers_observe_updates_until_unsubscribed(
    backend_contract_session: AppServerSession,
) -> None:
    themes: list[str] = []
    unsubscribe = backend_contract_session.resources.config.subscribe(
        lambda config: themes.append(config.theme)
    )

    await backend_contract_session.resources.config.update({"theme": "light"})
    unsubscribe()
    await backend_contract_session.resources.config.update({"theme": "dark"})

    assert themes == ["light"]


@pytest.mark.asyncio
async def test_first_turn_does_not_pin_default_and_resume_keeps_committed_default(
    config_dir: Path,
    tmp_path: Path,
    backend_contract_mistral_api: respx.Route,
    backend_contract_mistral_response: Callable[[str], httpx.Response],
) -> None:
    config_file = config_dir / "config.toml"
    config = tomllib.loads(config_file.read_text(encoding="utf-8"))
    _write_catalog(config_dir, {"alternate": "alternate-model"})
    session_root = tmp_path / "sessions"
    config["session_logging"] = {"enabled": True, "save_dir": str(session_root)}
    config_file.write_text(tomli_w.dumps(config), encoding="utf-8")
    backend_contract_mistral_api.mock(
        return_value=backend_contract_mistral_response("stored")
    )

    connection = await connect_backend_contract_host(
        session_options=SessionOptions(), capabilities=ClientCapabilities()
    )
    session = await connection.host.open_session()
    try:
        assert session.resources.config.current.active_model_pinned is False
        default_model = session.resources.config.current.active_model.alias
        await session.resources.config.update(
            {"active_model": "alternate"}, reload_runtime=True
        )
        assert session.resources.config.current.active_model.alias == "alternate"
        assert _session_model_is_persisted(session_root, session.session_id) is False
        await session.resources.config.update({"active_model": ""}, reload_runtime=True)
        _ = [event async for event in session.act("use the default without pinning")]
        session_id = session.session_id
        assert session.resources.config.current.active_model_pinned is False
        assert _stored_active_model(session_root, session_id) == ""

        await session.resources.config.update(
            {"active_model": "alternate"}, reload_runtime=True
        )
        assert session.resources.config.current.active_model.alias == "alternate"
        assert _stored_active_model(session_root, session_id) == ""
        _ = [event async for event in session.act("persist the alternate model")]
        assert _stored_active_model(session_root, session_id) == ("alternate")

        await session.resources.config.update({"active_model": ""}, reload_runtime=True)
        assert session.resources.config.current.active_model.alias == default_model
        assert _stored_active_model(session_root, session_id) == "alternate"
        _ = [event async for event in session.act("persist the unpinned selection")]
        assert _stored_active_model(session_root, session_id) == ""
    finally:
        await session.close()

    config = tomllib.loads(config_file.read_text(encoding="utf-8"))
    assert "active_model" not in config

    resumed_connection = await connect_backend_contract_host(
        session_options=SessionOptions(), capabilities=ClientCapabilities()
    )
    resumed = await resumed_connection.host.resume_session(session_id)
    try:
        assert resumed.resources.config.current.active_model.alias == default_model
        assert resumed.resources.config.current.active_model_pinned is False
        await resumed.clear_history()
        assert resumed.session_id != session_id
        assert resumed.resources.config.current.active_model.alias == default_model
    finally:
        await resumed.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("target_layer", "updates_session_pin"), [("user-toml", False), ("overrides", True)]
)
async def test_explicit_active_model_target_controls_session_pin_update(
    config_dir: Path,
    tmp_path: Path,
    backend_contract_mistral_api: respx.Route,
    backend_contract_mistral_response: Callable[[str], httpx.Response],
    target_layer: str,
    updates_session_pin: bool,
) -> None:
    config_file = config_dir / "config.toml"
    config = tomllib.loads(config_file.read_text(encoding="utf-8"))
    _write_catalog(config_dir, {"alternate": "alternate-model"})
    session_root = tmp_path / "sessions"
    config["session_logging"] = {"enabled": True, "save_dir": str(session_root)}
    config_file.write_text(tomli_w.dumps(config), encoding="utf-8")
    backend_contract_mistral_api.mock(
        return_value=backend_contract_mistral_response("stored")
    )

    connection = await connect_backend_contract_host(
        session_options=SessionOptions(), capabilities=ClientCapabilities()
    )
    session = await connection.host.open_session()
    try:
        pinned_model = session.resources.config.current.active_model.alias
        await session.resources.config.update(
            {"active_model": pinned_model}, target_layer="overrides"
        )
        _ = [event async for event in session.act("record the explicit selection")]
        assert _stored_active_model(session_root, session.session_id) == pinned_model

        if target_layer == "user-toml":
            fields = await session.resources.config.read_fields()
            result = await session.resources.config.write(
                [ConfigWriteOpWire(op="set", path="/active_model", value="alternate")],
                target="user",
                expected_revision=fields.revisions["user-toml"],
                reason="explicit saved model selection",
            )
            assert result.rejected is True
        else:
            await session.resources.config.update(
                {"active_model": "alternate"},
                target_layer=target_layer,
                reload_runtime=True,
            )

        active_model = "alternate" if updates_session_pin else pinned_model
        assert session.resources.config.current.active_model.alias == active_model
        assert _stored_active_model(session_root, session.session_id) == pinned_model

        _ = [event async for event in session.act("persist the selected model")]
        assert _stored_active_model(session_root, session.session_id) == active_model
    finally:
        await session.close()

    persisted_config = tomllib.loads(config_file.read_text(encoding="utf-8"))
    assert "active_model" not in persisted_config


@pytest.mark.asyncio
async def test_removed_session_active_model_falls_back_to_default(
    config_dir: Path,
    tmp_path: Path,
    backend_contract_mistral_api: respx.Route,
    backend_contract_mistral_response: Callable[[str], httpx.Response],
) -> None:
    config_file = config_dir / "config.toml"
    config = tomllib.loads(config_file.read_text(encoding="utf-8"))
    _write_catalog(
        config_dir,
        {"removed-model": "removed-model", "lower-layer-model": "lower-layer-model"},
    )
    session_root = tmp_path / "sessions"
    config["session_logging"] = {"enabled": True, "save_dir": str(session_root)}
    config_file.write_text(tomli_w.dumps(config), encoding="utf-8")
    backend_contract_mistral_api.mock(
        return_value=backend_contract_mistral_response("stored")
    )

    connection = await connect_backend_contract_host(
        session_options=SessionOptions(), capabilities=ClientCapabilities()
    )
    session = await connection.host.open_session()
    try:
        await session.resources.config.update(
            {"active_model": "removed-model"}, reload_runtime=True
        )
        _ = [event async for event in session.act("pin the removed model")]
        session_id = session.session_id
        assert _stored_active_model(session_root, session_id) == "removed-model"
    finally:
        await session.close()

    config = tomllib.loads(config_file.read_text(encoding="utf-8"))
    _write_catalog(config_dir, {"lower-layer-model": "lower-layer-model"})
    config_file.write_text(tomli_w.dumps(config), encoding="utf-8")

    resumed_connection = await connect_backend_contract_host(
        session_options=SessionOptions(), capabilities=ClientCapabilities()
    )
    # Blueprint resume (fresh session open) fails fast: the removed committed
    # model is a configuration error, not a recoverable state. Only the
    # root-session resume RPC keeps the recovery flow.
    with pytest.raises(AppServerResponseError):
        await resumed_connection.host.resume_session(session_id)


def _write_catalog(config_dir: Path, models: dict[str, str]) -> None:
    (config_dir / "models.toml").write_text(
        tomli_w.dumps({
            "models": {
                alias: {"deployments": [{"provider": "mistral", "name": name}]}
                for alias, name in models.items()
            }
        }),
        encoding="utf-8",
    )


def _stored_active_model(session_root: Path, session_id: str) -> str | None:
    metadata_path = next(session_root.glob(f"*_{session_id[:8]}/meta.json"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return metadata["config"].get("active_model")


def _session_model_is_persisted(session_root: Path, session_id: str) -> bool:
    return any(session_root.glob(f"*_{session_id[:8]}/meta.json"))


@pytest.mark.asyncio
async def test_runtime_mutations_leave_the_connection_usable_with_the_updated_runtime(
    backend_contract_session: AppServerSession,
) -> None:
    await backend_contract_session.resources.config.update(
        {"disabled_tools": ["grep"]}, reload_runtime=True
    )
    injected = await backend_contract_session.inject_user_context(
        "The runtime update is complete", as_message=True, client_message_id="runtime-1"
    )

    assert not backend_contract_session.resources.runtime.has_tool("grep")
    assert injected[0].entry.id == "runtime-1"


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["write", "reload"])
async def test_config_mutations_conflict_while_a_turn_is_running(
    operation: str,
    backend_contract_gated_mistral_response: Callable[..., httpx.Response],
    backend_contract_mistral_api: respx.Route,
    backend_contract_session: AppServerSession,
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    backend_contract_mistral_api.mock(
        return_value=backend_contract_gated_mistral_response(
            "finished", started=started, release=release
        )
    )

    async def consume_turn() -> None:
        _ = [event async for event in backend_contract_session.act("wait")]

    turn = asyncio.create_task(consume_turn())
    try:
        await asyncio.wait_for(started.wait(), timeout=1)
        with pytest.raises(AppServerResponseError) as exc_info:
            if operation == "write":
                await backend_contract_session.resources.config.update({
                    "theme": "dark"
                })
            else:
                await backend_contract_session.resources.config.reload()
    finally:
        release.set()
        await turn

    assert exc_info.value.error.code is ProtocolErrorCode.CONFLICT


@pytest.mark.asyncio
async def test_config_fields_read_is_answered_by_the_owning_agent(
    configuration_connection: BackendContractConnection,
    configuration_session: AppServerSession,
) -> None:
    result = await configuration_connection.client.request(
        "config/fields/read",
        ConfigFieldsReadParams(session_id=configuration_session.session_id),
    )

    response = ConfigFieldsReadResponse.model_validate(result)
    wires = {wire.name: wire for wire in response.fields}
    assert wires
    assert wires["theme"].value == configuration_session.resources.config.current.theme
    assert response.targets
    # The revision of the source a save would be checked against travels with
    # the read, so a stale-revision save is refused on either route.
    assert isinstance(response.revisions["user-toml"], str)


@pytest.mark.asyncio
async def test_settings_read_pairs_leaves_with_the_user_revision(
    configuration_connection: BackendContractConnection,
    configuration_session: AppServerSession,
) -> None:
    result = await configuration_connection.client.request(
        "config/settings/read",
        SettingsReadParams(session_id=configuration_session.session_id),
    )

    response = SettingsReadResponse.model_validate(result)
    leaves = {field.path: field for field in response.fields}
    assert leaves
    assert (
        leaves["ask_confirmation_on_exit"].effective_value
        is configuration_session.resources.config.current.ask_confirmation_on_exit
    )
    assert response.catalog
    assert response.web_search is not None
    assert response.user_layer == "user-toml"
    assert isinstance(response.user_revision, str)


@pytest.mark.asyncio
async def test_proxy_settings_round_trip_where_the_env_file_lives(
    configuration_session: AppServerSession,
) -> None:
    proxy = "http://proxy.backend-contract.test:3128"
    before = await configuration_session.resources.config.read_proxy()
    assert before.values["HTTPS_PROXY"] is None

    await configuration_session.resources.config.update_proxy({"HTTPS_PROXY": proxy})
    written = await configuration_session.resources.config.read_proxy()
    assert written.values["HTTPS_PROXY"] == proxy
    assert "HTTPS_PROXY" in written.descriptions

    await configuration_session.resources.config.update_proxy({"HTTPS_PROXY": None})
    cleared = await configuration_session.resources.config.read_proxy()
    assert cleared.values["HTTPS_PROXY"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("changes", "named"),
    [
        ({"NOT_A_PROXY_VAR": "http://proxy.test:3128"}, "NOT_A_PROXY_VAR"),
        ({"HTTPS_PROXY": "proxy.test:3128"}, "HTTPS_PROXY"),
    ],
)
async def test_rejected_proxy_write_keeps_the_in_process_param_refusal(
    configuration_connection: BackendContractConnection,
    configuration_session: AppServerSession,
    changes: dict[str, str | None],
    named: str,
) -> None:
    with pytest.raises(AppServerResponseError) as exc_info:
        await configuration_connection.client.request(
            "config/proxy/write",
            ConfigProxyWriteParams(
                session_id=configuration_session.session_id, changes=changes
            ),
        )

    assert exc_info.value.error.code is ProtocolErrorCode.INVALID_PARAMS
    assert named in exc_info.value.error.message


@pytest.mark.asyncio
async def test_config_write_applies_in_the_owning_agent_and_reports_its_runtime(
    configuration_session: AppServerSession,
) -> None:
    await configuration_session.resources.config.update(
        {"theme": "dark"}, reload_runtime=True
    )

    assert configuration_session.resources.config.current.theme == "dark"

    # The read that follows a write answers from the same owner, so a route
    # that wrote somewhere else would show the stale value here.
    fields = await configuration_session.resources.config.read_fields()
    assert {wire.name: wire for wire in fields.fields}["theme"].value == "dark"
