from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import tomli_w

from chartreux.app_server._config_write import config_write_targets
from chartreux.app_server.agent_config_write import AgentConfigWriteService
from chartreux.app_server.agent_wire import (
    RECONFIGURE_MODE_CONFIG_WRITE,
    ReconfigureReply,
    UnsupportedAgentRequestError,
    decode_reconfigure_reply,
    decode_reconfigure_request,
    encode_reconfigure_reply,
    encode_reconfigure_request,
)
from chartreux.app_server.client import AppServerClient
from chartreux.app_server.protocol import (
    ClientInfo,
    ConfigWriteOpWire,
    ConfigWriteParams,
    SessionStartParams,
)
from chartreux.app_server.transport import memory_transport_pair
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.config import build_default_orchestrator
from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
from chartreux.core.config.layers.default import DefaultConfigLayer
from chartreux.core.config.layers.overrides import OverridesLayer
from chartreux.core.config.layers.user import UserConfigLayer
from chartreux.core.config.orchestrator import ConfigOrchestrator
from chartreux.core.config.types import ConfigDurabilityError
from chartreux.core.trusted_folders import trusted_folders_manager
from tests.conftest import build_test_vibe_config
from tests.stubs.app_server import build_test_app_server
from tests.stubs.fake_backend import FakeBackend
from tests.stubs.fake_mcp_registry import FakeMCPRegistry


async def _write_model_ops(
    *, config: ChartreuxConfigSchema, ops: list[ConfigWriteOpWire]
) -> dict[str, Any]:
    client_transport, server_transport = memory_transport_pair()
    Path(UserConfigLayer().source_locator).write_text(
        tomli_w.dumps(
            config.model_dump(
                mode="json",
                exclude_none=True,
                exclude={"active_model", "thinking_overrides"},
            )
        )
    )
    orchestrator = await build_default_orchestrator()
    agent_loop = AgentLoop(
        config_orchestrator=orchestrator,
        backend=FakeBackend(),
        mcp_registry=FakeMCPRegistry(),
    )
    server = build_test_app_server(agent_loop, server_transport)
    client = AppServerClient(client_transport, run_peer=server.serve)

    try:
        await client.initialize(ClientInfo(name="config-write-test", version="1"))
        await client.notify("initialized")
        await client.request("session/start", SessionStartParams())
        response = await client.request(
            "config/write", ConfigWriteParams(session_id=agent_loop.session_id, ops=ops)
        )
    finally:
        await client_transport.close()
        await server_transport.close()
        await agent_loop.aclose()

    assert response["rejected"] is True
    assert response["persistence"] == "not_saved"
    assert agent_loop.config.available_models() == config.available_models()
    return response


async def _write_model_field(
    *, config: ChartreuxConfigSchema, alias: str, field: str, value: Any
) -> dict[str, Any]:
    models = await _write_model_ops(
        config=config,
        ops=[ConfigWriteOpWire(op="set", path=f"/models/{alias}/{field}", value=value)],
    )
    return models


def test_config_write_translation_preserves_order_paths_and_targets() -> None:
    from chartreux.app_server._config_write import config_write_ops_to_patches

    ops = [
        ConfigWriteOpWire(
            op="set",
            path="/models/a~1b~0c/thinking",
            value="low",
            target_layer="user-toml",
        ),
        ConfigWriteOpWire(
            op="remove", path="/models/a~1b~0c/thinking", target_layer="user-toml"
        ),
        ConfigWriteOpWire(
            op="set",
            path="/auto_compact_threshold",
            value=123,
            target_layer="overrides",
        ),
        ConfigWriteOpWire(op="set", path="", value={"theme": "dark"}),
    ]
    patches = config_write_ops_to_patches(ops)
    assert [p.path for p in patches] == [o.path for o in ops]
    assert [p.target_layer_name for p in patches] == [o.target_layer for o in ops]
    assert [p.to_json_patch() for p in patches] == [
        {"op": "add", "path": ops[0].path, "value": "low"},
        {"op": "remove", "path": ops[1].path},
        {"op": "add", "path": ops[2].path, "value": 123},
        {"op": "add", "path": "", "value": {"theme": "dark"}},
    ]


@pytest.mark.asyncio
async def test_session_catalog_batch_requires_explicit_user_save() -> None:
    models = await _write_model_ops(
        config=build_test_vibe_config(active_model="glm-5-3"),
        ops=[
            ConfigWriteOpWire(op="set", path="/models/glm-5-3/thinking", value="low"),
            ConfigWriteOpWire(op="set", path="/models/glm-5-3/temperature", value=0.7),
        ],
    )
    assert models["rejected"] is True


@pytest.mark.asyncio
async def test_session_catalog_field_requires_explicit_user_save() -> None:
    # Definitions belong to the user source, including changes to existing models.
    config = build_test_vibe_config(active_model="glm-5-3")
    persisted = await _write_model_field(
        config=config, alias="glm-5-3", field="thinking", value="low"
    )

    assert persisted["persistence"] == "not_saved"


@pytest.mark.asyncio
@pytest.mark.parametrize("with_user", [False, True])
async def test_real_wire_targets_and_catalog_write_enforcement(
    tmp_path: Path, with_user: bool
) -> None:
    from chartreux.app_server.protocol import (
        ConfigFieldsReadParams,
        ConfigFieldsReadResponse,
    )
    from chartreux.core.agent_loop import AgentLoop
    from chartreux.core.config._catalog import CATALOG_DEFINITION_FIELDS
    from chartreux.core.config.layers.default import DefaultConfigLayer
    from chartreux.core.config.layers.overrides import OverridesLayer
    from chartreux.core.config.layers.user import UserConfigLayer
    from chartreux.core.config.orchestrator import ConfigOrchestrator
    from tests.stubs.fake_backend import FakeBackend
    from tests.stubs.fake_mcp_registry import FakeMCPRegistry

    user = UserConfigLayer(path=tmp_path / "config.toml", name="renamed-user")
    session = OverridesLayer(
        data={"session_logging": {"enabled": False, "generate_titles": False}}
    )
    orch = await ConfigOrchestrator.create(
        schema=ChartreuxConfigSchema,
        layers=[
            DefaultConfigLayer(schema=ChartreuxConfigSchema),
            *([user] if with_user else []),
            session,
        ],
        default_layer_resolver=lambda: session,
    )
    loop = AgentLoop(
        config_orchestrator=orch, backend=FakeBackend(), mcp_registry=FakeMCPRegistry()
    )
    client_transport, server_transport = memory_transport_pair()
    server = build_test_app_server(loop, server_transport)
    client = AppServerClient(client_transport, run_peer=server.serve)
    try:
        await client.initialize(ClientInfo(name="catalog-wire", version="1"))
        await client.notify("initialized")
        await client.request("session/start", SessionStartParams())
        raw = await client.request(
            "config/fields/read", ConfigFieldsReadParams(session_id=loop.session_id)
        )
        response = ConfigFieldsReadResponse.model_validate(raw)
        for field in response.fields:
            expected = (
                (["renamed-user"] if with_user else [])
                if field.name in CATALOG_DEFINITION_FIELDS
                else ["overrides", *(["renamed-user"] if with_user else [])]
            )
            assert field.writable_targets == expected
        before = orch.config
        raw = await client.request(
            "config/write",
            ConfigWriteParams(
                session_id=loop.session_id,
                ops=[
                    ConfigWriteOpWire(
                        op="set", path="/models", value={}, target_layer="overrides"
                    )
                ],
            ),
        )
        assert raw["rejected"] is True
        assert orch.config is before
        assert not (tmp_path / "config.toml").exists()
    finally:
        await client_transport.close()
        await server_transport.close()
        await loop.aclose()


@pytest.mark.asyncio
async def test_config_write_targets_offer_user_project_and_session(
    config_dir: Path, tmp_working_directory: Path
) -> None:
    project_vibe_dir = tmp_working_directory / ".chartreux"
    project_vibe_dir.mkdir(parents=True, exist_ok=True)
    (project_vibe_dir / "config.toml").write_text('theme = "project-theme"\n')
    trusted_folders_manager.add_trusted(project_vibe_dir)

    orchestrator = await build_default_orchestrator()

    assert config_write_targets(orchestrator) == [
        "overrides",
        "user-toml",
        "project-toml",
    ]


@pytest.mark.asyncio
async def test_config_write_targets_skip_untrusted_project(
    config_dir: Path, tmp_working_directory: Path
) -> None:
    project_vibe_dir = tmp_working_directory / ".chartreux"
    project_vibe_dir.mkdir(parents=True, exist_ok=True)
    (project_vibe_dir / "config.toml").write_text('theme = "project-theme"\n')

    orchestrator = await build_default_orchestrator()

    assert config_write_targets(orchestrator) == ["overrides", "user-toml"]


@pytest.mark.asyncio
async def test_config_write_targets_skip_undiscovered_project(
    config_dir: Path, tmp_working_directory: Path
) -> None:
    orchestrator = await build_default_orchestrator()

    assert config_write_targets(orchestrator) == ["overrides", "user-toml"]


# ---------------------------------------------------------------------------
# Item 1 op 10 mode (d): the config-write reconfigure over the process wire.
#
# The request carries the config-write candidate and the agent runs the shipped
# transaction -- prepare on a candidate backend, the atomic commit, the shipped
# compensation that retires a prepared backend the commit did not adopt, and the
# changed-field projection. The receipt is the SHIPPED response, so every
# persistence x application composition survives the wire verbatim: an
# applied-but-undurable write stays that composition, and "nothing applied" is
# only ever claimed about RUNTIME application, never about durable disk state.
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _wire_loop(
    tmp_path: Path,
) -> AsyncIterator[tuple[AgentLoop, UserConfigLayer]]:
    """A loop on a real orchestrator with a user source and a session layer."""
    session = OverridesLayer(data={"session_logging": {"enabled": False}})
    user = UserConfigLayer(path=tmp_path / "config.toml")
    orchestrator = await ConfigOrchestrator.create(
        schema=ChartreuxConfigSchema,
        layers=[DefaultConfigLayer(schema=ChartreuxConfigSchema), user, session],
        default_layer_resolver=lambda: session,
    )
    loop = AgentLoop(
        config_orchestrator=orchestrator,
        backend=FakeBackend(),
        mcp_registry=FakeMCPRegistry(),
    )
    try:
        yield loop, user
    finally:
        await loop.aclose()


def _theme_params(
    loop: AgentLoop,
    *,
    target: str = "session",
    expected_revision: str | None = None,
    value: str = "dark",
) -> ConfigWriteParams:
    return ConfigWriteParams(
        session_id=loop.session_id,
        ops=[ConfigWriteOpWire(op="set", path="/theme", value=value)],
        target=target,  # type: ignore[arg-type]
        expected_revision=expected_revision,
    )


def test_a_config_write_request_carries_the_candidate_not_the_reload_flag() -> None:
    request = encode_reconfigure_request(
        mode=RECONFIGURE_MODE_CONFIG_WRITE,
        session_id="session-1",
        deadline=1234.5,
        config_write=ConfigWriteParams(
            session_id="session-1",
            ops=[ConfigWriteOpWire(op="set", path="/theme", value="dark")],
            reason="model change",
            target="user",
            expected_revision="rev-1",
            reload_runtime=True,
        ),
    )

    # The mode IS the reload leg, so the flag is not carried on this path.
    assert "reload_runtime" not in request["config_write"]

    decoded = decode_reconfigure_request(request)

    assert decoded.mode == RECONFIGURE_MODE_CONFIG_WRITE
    assert decoded.deadline == 1234.5
    assert decoded.config_write is not None
    assert decoded.config_write.target == "user"
    assert decoded.config_write.expected_revision == "rev-1"
    assert decoded.config_write.reason == "model change"
    assert decoded.config_write.ops[0].path == "/theme"


def test_a_config_write_request_without_its_deadline_or_candidate_fails_typed() -> None:
    candidate = ConfigWriteParams(
        session_id="session-1",
        ops=[ConfigWriteOpWire(op="set", path="/theme", value="dark")],
    ).model_dump(mode="json")

    with pytest.raises(UnsupportedAgentRequestError, match="no config-write candidate"):
        decode_reconfigure_request({
            "mode": RECONFIGURE_MODE_CONFIG_WRITE,
            "session_id": "session-1",
            "deadline": 10.0,
        })
    with pytest.raises(UnsupportedAgentRequestError, match="absolute deadline"):
        decode_reconfigure_request({
            "mode": RECONFIGURE_MODE_CONFIG_WRITE,
            "session_id": "session-1",
            "config_write": candidate,
        })


@pytest.mark.asyncio
async def test_a_saved_write_reports_saved_and_applied_over_the_wire(
    tmp_path: Path,
) -> None:
    async with _wire_loop(tmp_path) as (loop, user):
        request = decode_reconfigure_request(
            encode_reconfigure_request(
                mode=RECONFIGURE_MODE_CONFIG_WRITE,
                session_id=loop.session_id,
                deadline=100.0,
                config_write=_theme_params(
                    loop,
                    target="user",
                    expected_revision=user.fingerprint,
                    value="light",
                ),
            )
        )
        assert request.config_write is not None
        response = await AgentConfigWriteService(loop).write(request.config_write)
        reply = decode_reconfigure_reply(
            encode_reconfigure_reply(
                ReconfigureReply(
                    mode=request.mode,
                    outcome=response.application,
                    config_write=response,
                )
            )
        )

    receipt = reply.config_write
    assert receipt is not None
    assert (receipt.persistence, receipt.application) == ("saved", "applied")
    assert receipt.rejected is False
    assert receipt.failures == []
    assert receipt.revision is not None
    assert receipt.saved_values == {"theme": "light"}
    assert receipt.fields[0].value == "light"
    assert receipt.launch_metadata_persisted is True
    # The shipped response travels whole: the runtime projection is part of it.
    assert receipt.runtime.config.theme == "light"
    assert reply.outcome == "applied"
    assert 'theme = "light"' in (tmp_path / "config.toml").read_text()


@pytest.mark.asyncio
async def test_an_applied_but_undurable_write_reports_both_dimensions(
    tmp_path: Path,
) -> None:
    async with _wire_loop(tmp_path) as (loop, user):

        async def undurable(*args: Any, **kwargs: Any) -> Any:
            raise ConfigDurabilityError("rev-uncertain")

        with patch.object(user, "save_checked", side_effect=undurable):
            response = await AgentConfigWriteService(loop).write(
                _theme_params(loop, target="user", expected_revision=user.fingerprint)
            )

        live_theme = loop.config.theme

    assert response.persistence == "durability_uncertain"
    assert response.application == "applied"
    assert response.rejected is False
    assert response.revision == "rev-uncertain"
    # APPLIED-BUT-UNDURABLE is the composition it is: the runtime did apply it.
    assert live_theme == "dark"
    assert response.saved_values == {"theme": "dark"}


@pytest.mark.asyncio
async def test_a_cancelled_save_keeps_the_runtime_and_retires_the_prepared_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with _wire_loop(tmp_path) as (loop, user):
        retired: list[Any] = []
        prepared_backends: list[Any] = []
        monkeypatch.setattr(
            loop._backend_lifetime,
            "retire",
            lambda backend, *, whole_turn_active: retired.append(backend),
        )

        def factory(config: Any = None) -> Any:
            # A reload prepares its own backend; an injected one would BE the
            # live backend and leave the compensation nothing to retire.
            backend = FakeBackend()
            prepared_backends.append(backend)
            return backend

        monkeypatch.setattr(loop, "backend_factory", factory)
        entered, release = asyncio.Event(), asyncio.Event()
        original = user.save_checked

        async def slow_save(*args: Any, **kwargs: Any) -> Any:
            entered.set()
            await release.wait()
            return await original(*args, **kwargs)

        before = loop.config
        with patch.object(user, "save_checked", side_effect=slow_save):
            task = asyncio.create_task(
                AgentConfigWriteService(loop).write(
                    _theme_params(
                        loop, target="user", expected_revision=user.fingerprint
                    )
                )
            )
            await entered.wait()
            task.cancel()
            await asyncio.sleep(0)
            release.set()
            response = await task

        live_config = loop.config

    # The replacement landed, but the runtime was never published.
    assert (response.persistence, response.application) == ("saved", "unchanged")
    assert response.rejected is False
    assert response.failures == ["cancelled"]
    assert live_config is before
    # The prepared backend the commit never adopted was retired by the shipped
    # compensation, not leaked behind the transaction.
    assert retired == prepared_backends


@pytest.mark.asyncio
async def test_a_rejected_write_retires_the_prepared_backend_and_keeps_the_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with _wire_loop(tmp_path) as (loop, _user):
        retired: list[Any] = []
        prepared_backends: list[Any] = []
        monkeypatch.setattr(
            loop._backend_lifetime,
            "retire",
            lambda backend, *, whole_turn_active: retired.append(backend),
        )

        def factory(config: Any = None) -> Any:
            backend = FakeBackend()
            prepared_backends.append(backend)
            return backend

        monkeypatch.setattr(loop, "backend_factory", factory)

        def refuse_commit(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("the candidate runtime was refused")

        monkeypatch.setattr(loop, "_commit_reload", refuse_commit)
        before = loop.config

        response = await AgentConfigWriteService(loop).write(_theme_params(loop))

        live_config = loop.config

    assert response.rejected is True
    assert (response.persistence, response.application) == ("not_saved", "unchanged")
    assert response.failures == ["Configuration preparation or application failed"]
    assert live_config is before
    assert retired == prepared_backends


@pytest.mark.asyncio
async def test_every_shipped_dimension_composition_survives_the_wire(
    tmp_path: Path,
) -> None:
    async with _wire_loop(tmp_path) as (loop, _user):
        response = await AgentConfigWriteService(loop).write(_theme_params(loop))

    for persistence in ("not_saved", "saved", "durability_uncertain"):
        for application in ("unchanged", "applied", "failed"):
            composed = response.model_copy(
                update={"persistence": persistence, "application": application}
            )
            reply = decode_reconfigure_reply(
                encode_reconfigure_reply(
                    ReconfigureReply(
                        mode=RECONFIGURE_MODE_CONFIG_WRITE,
                        outcome=application,
                        config_write=composed,
                    )
                )
            )
            receipt = reply.config_write
            assert receipt is not None
            assert (receipt.persistence, receipt.application) == (
                persistence,
                application,
            )
            assert receipt.runtime.config.theme == response.runtime.config.theme
            assert reply.outcome == application


@pytest.mark.asyncio
async def test_a_config_write_reply_whose_outcome_hides_its_result_is_refused(
    tmp_path: Path,
) -> None:
    async with _wire_loop(tmp_path) as (loop, _user):
        response = await AgentConfigWriteService(loop).write(_theme_params(loop))

    encoded = encode_reconfigure_reply(
        ReconfigureReply(
            mode=RECONFIGURE_MODE_CONFIG_WRITE,
            outcome=response.application,
            config_write=response,
        )
    )
    encoded["outcome"] = "accepted"

    with pytest.raises(UnsupportedAgentRequestError, match="application result"):
        decode_reconfigure_reply(encoded)
