from __future__ import annotations

from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock

import pytest
import tomli_w

from chartreux.app_server.config import ThinkingLevel
from chartreux.app_server.protocol import AppServerResponseError, ConfigWriteOpWire
from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
from chartreux.core.config.layers.overrides import OverridesLayer
from chartreux.core.config.layers.user import UserConfigLayer
from chartreux.core.config.models import ModelConfig
from chartreux.core.config.orchestrator import ConfigOrchestrator
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.stubs.app_server import create_test_app_server_session


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
        definition_path.write_text(
            tomli_w.dumps(config.model_dump(mode="json", exclude_none=True))
        )
        user = UserConfigLayer(path=definition_path, name="definition")
        return await ConfigOrchestrator.create(
            schema=ChartreuxConfigSchema,
            layers=[user, OverridesLayer(data={})],
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
