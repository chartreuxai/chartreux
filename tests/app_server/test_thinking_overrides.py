from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from chartreux.app_server._runtime import AgentRuntimeFactory
from chartreux.app_server._session_model import (
    config_thinking_overrides,
    restore_session_thinking_overrides,
)
from chartreux.app_server.config import ThinkingLevel
from chartreux.app_server.protocol import AppServerResponseError, ConfigWriteOpWire
from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
from chartreux.core.config.layers.overrides import OverridesLayer
from chartreux.core.config.layers.user import UserConfigLayer
from chartreux.core.config.models import ModelConfig, SessionLoggingConfig
from chartreux.core.config.orchestrator import (
    ConfigOrchestrator,
    ConfigPatchValidationError,
)
from chartreux.core.model_catalog.resolver import ModelResolutionError
from tests.conftest import FakeBackend, build_test_agent_loop, build_test_vibe_config
from tests.stubs.app_server import (
    attach_test_app_server_session,
    create_test_app_server_session,
    start_test_app_server,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("historical", "parsed", "effective"),
    [
        ({}, None, {"alpha": "low"}),
        ({"thinking_overrides": {}}, {}, {"alpha": "low"}),
        (
            {"thinking_overrides": {"beta": "high"}},
            {"beta": "high"},
            {"alpha": "low", "beta": "high"},
        ),
        ({"thinking_overrides": None}, {}, {"alpha": "low"}),
        ({"thinking_overrides": ["high"]}, {}, {"alpha": "low"}),
        (
            {"thinking_overrides": {"beta": "medium", "alpha": 3, 4: "high"}},
            {"beta": "medium"},
            {"alpha": "low", "beta": "medium"},
        ),
    ],
)
async def test_historical_thinking_restores_effective_layered_configuration(
    historical: dict[str, Any],
    parsed: dict[str, str] | None,
    effective: dict[str, str],
    tmp_path: Path,
) -> None:
    config = build_test_vibe_config(
        active_model="alpha",
        models=[
            ModelConfig(name="alpha", provider="mistral", alias="alpha"),
            ModelConfig(name="beta", provider="mistral", alias="beta"),
        ],
    )
    path = tmp_path / "config.toml"
    path.write_text('theme = "dark"\n')
    user = UserConfigLayer(path=path)
    orchestrator = await ConfigOrchestrator.create(
        schema=ChartreuxConfigSchema,
        layers=[
            user,
            OverridesLayer(
                name="launch",
                data={"active_model": "alpha", "thinking_overrides": {"alpha": "low"}},
            ),
            OverridesLayer(
                data={"thinking_overrides": {"alpha": "max", "beta": "max"}}
            ),
        ],
        default_layer_resolver=lambda: user,
        catalog_snapshot=config.catalog_snapshot,
    )
    restored = config_thinking_overrides({"config": historical})
    assert restored == parsed
    assert (
        await restore_session_thinking_overrides(
            orchestrator, restored, reason="historical fixture"
        )
        == []
    )
    assert orchestrator.config.thinking_overrides == effective
    assert (
        orchestrator.config.available_models()["alpha"].thinking == effective["alpha"]
    )
    assert (
        getattr(
            orchestrator.get_layer("overrides").cached_data, "thinking_overrides", None
        )
        == parsed
    )
    # Reload must not resurrect the prior attached session's map.
    await orchestrator.reload()
    assert orchestrator.config.thinking_overrides == effective


async def _use_layered_session_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async def load(
        config: ChartreuxConfigSchema,
    ) -> ConfigOrchestrator[ChartreuxConfigSchema]:
        path = tmp_path / "launch-config.toml"
        path.write_text('theme = "dark"\n')
        user = UserConfigLayer(path=path)
        return await ConfigOrchestrator.create(
            schema=ChartreuxConfigSchema,
            layers=[
                user,
                OverridesLayer(
                    name="launch",
                    data={
                        "active_model": config.active_model,
                        "thinking_overrides": config.thinking_overrides,
                        "session_logging": config.session_logging.model_dump(
                            mode="json"
                        ),
                    },
                ),
                OverridesLayer(data={}),
            ],
            default_layer_resolver=lambda: user,
            catalog_snapshot=config.catalog_snapshot,
        )

    monkeypatch.setattr("tests.conftest._load_orchestrator", load)


@pytest.mark.asyncio
@pytest.mark.parametrize("overrides", [{"alpha": "invalid"}, {"missing": "high"}])
async def test_historical_thinking_semantic_errors_reject_without_mutation(
    overrides: dict[str, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _use_layered_session_config(monkeypatch, tmp_path)
    loop = build_test_agent_loop(
        config=build_test_vibe_config(
            active_model="alpha",
            models=[ModelConfig(name="alpha", provider="mistral", alias="alpha")],
        )
    )
    try:
        before = loop.config
        parsed = config_thinking_overrides({
            "config": {"thinking_overrides": overrides}
        })
        with pytest.raises((ConfigPatchValidationError, ModelResolutionError)):
            await restore_session_thinking_overrides(
                loop.config_orchestrator, parsed, reason="invalid history"
            )
        assert loop.config is before
    finally:
        await loop.aclose()


@pytest.mark.asyncio
async def test_switch_to_historical_session_does_not_leak_thinking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _use_layered_session_config(monkeypatch, tmp_path)
    config = build_test_vibe_config(
        active_model="alpha",
        models=[
            ModelConfig(name="alpha", provider="mistral", alias="alpha", thinking="low")
        ],
        thinking_overrides={"alpha": "medium"},
        session_logging=SessionLoggingConfig(
            enabled=True, save_dir=str(tmp_path / "sessions")
        ),
    )
    older = build_test_agent_loop(config=config, cwd=tmp_path)
    await older.persist_empty_session()
    older_id = older.session_id
    assert older.session_logger.session_dir is not None
    metadata_path = older.session_logger.session_dir / "meta.json"
    # The actual historical file has no thinking field (not an explicit {}).
    metadata = json.loads(metadata_path.read_text())
    metadata["config"].pop("thinking_overrides")
    metadata_path.write_text(json.dumps(metadata))
    await older.aclose()

    source = build_test_agent_loop(config=config, cwd=tmp_path)
    session = await create_test_app_server_session(source)
    try:
        await session.resources.config.set_thinking("max")
        assert source.config.get_active_model().thinking == "max"
        await session.resume(older_id)
        await session.resources.runtime.refresh()
        assert source.config.thinking_overrides == {"alpha": "medium"}
        assert session.resources.config.current.active_model.thinking == "medium"
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_picker_model_then_thinking_save_reopen_and_switch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _use_layered_session_config(monkeypatch, tmp_path)
    canonical = "beta/x~y"
    config = build_test_vibe_config(
        active_model="alpha",
        models=[
            ModelConfig(
                name="alpha-wire", provider="mistral", alias="alpha", thinking="low"
            ),
            ModelConfig(
                name="beta-wire", provider="mistral", alias=canonical, thinking="medium"
            ),
        ],
        session_logging=SessionLoggingConfig(
            enabled=True, save_dir=str(tmp_path / "sessions")
        ),
    )
    loop = build_test_agent_loop(config=config, backend=FakeBackend(), cwd=tmp_path)
    session = await create_test_app_server_session(loop)
    saved_id = loop.session_id
    try:
        # These are the separate writes made by the model and thinking pickers.
        await session.resources.config.update(
            {"active_model": canonical}, target_layer="overrides", reload_runtime=True
        )
        assert session.resources.config.current.active_model.alias == canonical
        await session.resources.config.set_thinking("high")
        await session.resources.runtime.refresh()
        assert session.resources.config.current.active_model.thinking == "high"
        assert loop.config.thinking_overrides == {canonical: "high"}
        assert loop.committed_model is not None
        assert loop.committed_model.base_model == canonical
        _ = [event async for event in session.act("save picker state")]
    finally:
        await session.close()

    reopened = build_test_agent_loop(config=config, backend=FakeBackend(), cwd=tmp_path)
    session = await attach_test_app_server_session(
        start_test_app_server(reopened), resume_session_id=saved_id
    )
    try:
        await session.resources.runtime.refresh()
        assert session.resources.config.current.active_model.alias == canonical
        assert session.resources.config.current.active_model.thinking == "high"
        assert reopened.config.thinking_overrides == {canonical: "high"}
        other = build_test_agent_loop(config=config, cwd=tmp_path)
        await other.persist_empty_session()
        other_id = other.session_id
        await other.aclose()
        await session.resume(other_id)
        await session.resources.runtime.refresh()
        assert session.resources.config.current.active_model.alias == "alpha"
        assert reopened.config.thinking_overrides == {}
        await session.resume(saved_id)
        await session.resources.runtime.refresh()
        assert session.resources.config.current.active_model.alias == canonical
        assert session.resources.config.current.active_model.thinking == "high"
        assert reopened.config.thinking_overrides == {canonical: "high"}
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_set_thinking_targets_session_override_and_projects_resolved_active_model() -> (
    None
):
    models = [
        ModelConfig(
            name="alpha-model", provider="mistral", alias="alpha", thinking="low"
        ),
        ModelConfig(
            name="beta-model", provider="mistral", alias="beta", thinking="medium"
        ),
    ]
    agent_loop = build_test_agent_loop(
        config=build_test_vibe_config(models=models, active_model="alpha")
    )
    session = await create_test_app_server_session(agent_loop)
    try:
        persisted_before = (
            await agent_loop.config_orchestrator.load_persistence_layer()
        ).model_dump(mode="json")

        await session.resources.config.set_thinking("high")

        current = session.resources.config.current
        assert current.active_model.thinking == "high"
        assert (
            next(model.thinking for model in current.models if model.alias == "alpha")
            == "high"
        )
        assert agent_loop.config.thinking_overrides == {"alpha": "high"}
        persisted_after = (
            await agent_loop.config_orchestrator.load_persistence_layer()
        ).model_dump(mode="json")
        assert persisted_after == persisted_before
        assert agent_loop.config.available_models()["alpha"].thinking == "high"
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_thinking_choices_survive_model_switch_and_session_remove_falls_back() -> (
    None
):
    models = [
        ModelConfig(
            name="alpha-model", provider="mistral", alias="alpha", thinking="low"
        ),
        ModelConfig(
            name="beta-model", provider="mistral", alias="beta", thinking="medium"
        ),
    ]
    agent_loop = build_test_agent_loop(
        config=build_test_vibe_config(models=models, active_model="alpha")
    )
    session = await create_test_app_server_session(agent_loop)
    try:
        await session.resources.config.set_thinking("high")
        await session.resources.config.update(
            {"active_model": "beta"}, target_layer="overrides", reload_runtime=True
        )
        await session.resources.config.set_thinking("max")
        await session.resources.config.update(
            {"active_model": "alpha"}, target_layer="overrides", reload_runtime=True
        )

        assert session.resources.config.current.active_model.thinking == "high"
        assert agent_loop.config.thinking_overrides == {"alpha": "high", "beta": "max"}

        response = await session.resources.config.write(
            [
                ConfigWriteOpWire(
                    op="remove",
                    path="/thinking_overrides/alpha",
                    target_layer="overrides",
                )
            ],
            reason="test remove thinking override",
        )
        assert response.rejected is False
        assert session.resources.config.current.active_model.thinking == "low"
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_thinking_choice_survives_session_resume_without_changing_preset(
    tmp_path: Path,
) -> None:
    logging = SessionLoggingConfig(enabled=True, save_dir=str(tmp_path / "sessions"))
    config = build_test_vibe_config(
        active_model="alpha",
        models=[
            ModelConfig(
                name="alpha-model", provider="mistral", alias="alpha", thinking="low"
            )
        ],
        session_logging=logging,
    )
    saved = build_test_agent_loop(config=config, backend=FakeBackend(), cwd=tmp_path)
    default_preset = saved.config.catalog_snapshot.catalog.roles["orchestrator"]
    await saved.persist_empty_session()
    session_id = saved.session_id
    session = await create_test_app_server_session(saved)
    try:
        await session.resources.config.set_thinking("high")
        _ = [event async for event in session.act("remember the thinking choice")]
        assert saved.config.get_active_model().thinking == "high"
        assert (
            saved.config.catalog_snapshot.catalog.roles["orchestrator"]
            == default_preset
        )
    finally:
        await session.close()

    resumed = build_test_agent_loop(
        config=build_test_vibe_config(
            active_model="alpha",
            models=[
                ModelConfig(
                    name="alpha-model",
                    provider="mistral",
                    alias="alpha",
                    thinking="low",
                )
            ],
            session_logging=logging,
        ),
        backend=FakeBackend(),
        cwd=tmp_path,
    )
    try:
        await AgentRuntimeFactory().resume_root(resumed, session_id)
        assert resumed.config.get_active_model().thinking == "high"
        assert resumed.config.thinking_overrides == {"alpha": "high"}
    finally:
        await resumed.aclose()


@pytest.mark.asyncio
async def test_invalid_thinking_level_or_alias_is_rejected_without_state_change() -> (
    None
):
    config = build_test_vibe_config(
        active_model="alpha",
        models=[
            ModelConfig(
                name="alpha-model", provider="mistral", alias="alpha", thinking="low"
            )
        ],
    )
    agent_loop = build_test_agent_loop(config=config)
    session = await create_test_app_server_session(agent_loop)
    try:
        before = agent_loop.config.model_dump(mode="json")
        before_restrictions = agent_loop.config_orchestrator.restrictions
        with pytest.raises(AppServerResponseError, match="Invalid thinking level"):
            await session.resources.config.set_thinking(cast(ThinkingLevel, "invalid"))
        assert agent_loop.config.model_dump(mode="json") == before

        response = await session.resources.config.write(
            [
                ConfigWriteOpWire(
                    op="set",
                    path="/thinking_overrides/secret-thinking-alias",
                    value="high",
                    target_layer="overrides",
                )
            ],
            reason="test invalid thinking alias",
        )
        assert response.rejected is True
        assert "secret-thinking-alias" not in response.model_dump_json()
        assert agent_loop.config.model_dump(mode="json") == before
        assert agent_loop.config_orchestrator.restrictions == before_restrictions
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_set_thinking_escapes_model_alias_pointer() -> None:
    alias = "alpha/x~y"
    agent_loop = build_test_agent_loop(
        config=build_test_vibe_config(
            active_model=alias,
            models=[
                ModelConfig(
                    name="alpha-model", provider="mistral", alias=alias, thinking="off"
                )
            ],
        )
    )
    session = await create_test_app_server_session(agent_loop)
    try:
        await session.resources.config.set_thinking("medium")
        assert agent_loop.config.thinking_overrides == {alias: "medium"}
        assert session.resources.config.current.active_model.thinking == "medium"
    finally:
        await session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["system_prompt_id", "compaction_prompt_id"])
@pytest.mark.parametrize("value", ["missing-synthetic-prompt", "../synthetic-prompt"])
async def test_prompt_preflight_rejects_without_replacing_runtime(
    field: str, value: str
) -> None:
    loop = build_test_agent_loop(config=build_test_vibe_config())
    session = await create_test_app_server_session(loop)
    try:
        before = loop.config
        backend = loop.backend
        tools = loop.tool_manager
        response = await session.resources.config.write(
            [ConfigWriteOpWire(op="set", path=f"/{field}", value=value)],
            reason="synthetic prompt validation",
        )
        assert response.rejected is True
        assert response.failures
        assert value not in response.model_dump_json()
        assert loop.config is before
        assert loop.backend is backend
        assert loop.tool_manager is tools
    finally:
        await session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case", ["malformed", "mixed", "json", "pointer", "write", "returned"]
)
async def test_public_thinking_failure_redaction(
    case: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "config.toml"
    path.write_text('[tools.bash]\ndeny = ["blocked"]\n')
    file_before = path.read_bytes()

    async def load(
        config: ChartreuxConfigSchema,
    ) -> ConfigOrchestrator[ChartreuxConfigSchema]:
        definition_path = tmp_path / "definition.toml"
        definition_path.write_text('theme = "dark"\n')
        user = UserConfigLayer(path=definition_path, name="definition")
        return await ConfigOrchestrator.create(
            schema=ChartreuxConfigSchema,
            layers=[user, OverridesLayer(data={"active_model": config.active_model})],
            default_layer_resolver=lambda: user,
            catalog_snapshot=config.catalog_snapshot,
        )

    monkeypatch.setattr("tests.conftest._load_orchestrator", load)
    agent_loop = build_test_agent_loop(
        config=build_test_vibe_config(
            active_model="alpha",
            models=[ModelConfig(name="alpha", provider="mistral", alias="alpha")],
        )
    )
    session = await create_test_app_server_session(agent_loop)
    try:
        orchestrator = agent_loop.config_orchestrator
        config = orchestrator.config
        restrictions = orchestrator.restrictions
        token = orchestrator.accepted_token
        persisted = (await orchestrator.load_persistence_layer()).model_dump()
        alias, value = "private-alias", "private-value"
        ops = [
            ConfigWriteOpWire(
                op="set",
                path=f"/thinking_overrides/{alias}",
                value=value,
                target_layer="overrides",
            )
        ]
        if case == "mixed":
            ops[0] = ops[0].model_copy(update={"value": "high"})
            ops.append(
                ConfigWriteOpWire(
                    op="set",
                    path="/auto_compact_threshold",
                    value=value,
                    target_layer="overrides",
                )
            )
        elif case == "json":
            ops = [
                ConfigWriteOpWire(
                    op="remove",
                    path=f"/thinking_overrides/{alias}",
                    target_layer="overrides",
                )
            ]
        elif case == "pointer":
            ops[0] = ops[0].model_copy(
                update={"path": f"/thinking_overrides/{alias}~invalid"}
            )
        elif case in {"write", "returned"}:
            ops = [
                ConfigWriteOpWire(
                    op="set",
                    path="/thinking_overrides/alpha",
                    value="high",
                    target_layer="overrides",
                )
            ]
            failure = OSError(value)
            if case == "write":
                monkeypatch.setattr(
                    orchestrator.get_layer("overrides"),
                    "_save_to_store",
                    AsyncMock(side_effect=failure),
                )
            else:
                monkeypatch.setattr(
                    orchestrator,
                    "apply_session_patch",
                    AsyncMock(return_value=[failure]),
                )
        response = await session.resources.config.write(ops, reason="test")
        assert response.failures
        assert response.rejected is (case not in {"write", "returned"})
        rendered = response.model_dump_json()
        assert alias not in rendered and value not in rendered
        assert "thinking_overrides" in " ".join(response.failures)
        assert "overrides" in " ".join(response.failures)
        assert orchestrator.config is config
        assert orchestrator.restrictions == restrictions
        assert orchestrator.accepted_token is token
        assert path.read_bytes() == file_before
        assert (await orchestrator.load_persistence_layer()).model_dump() == persisted
    finally:
        await session.close()
