"""Production source enforcement and paired snapshot publication (not full P6)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from chartreux.core.agent_loop import AgentLoop
from chartreux.core.config import ChartreuxConfigSchema
from chartreux.core.config.layer import RawConfig
from chartreux.core.config.layers.default import DefaultConfigLayer
from chartreux.core.config.layers.overrides import OverridesLayer
from chartreux.core.config.layers.user import UserConfigLayer
from chartreux.core.config.orchestrator import ConfigOrchestrator
import chartreux.core.events as event_module
from chartreux.core.events import ToolResultEvent
from tests.core.agent_loop.test_chartreux_policy_denials import _call, _collect
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend

pytestmark = pytest.mark.asyncio


async def make_orchestrator(
    path: Path, *, reverse: bool = False
) -> ConfigOrchestrator[ChartreuxConfigSchema]:
    user = UserConfigLayer(path=path)
    session = OverridesLayer(
        data={
            "tools": {
                "read_file": {
                    "permission": "always",
                    "denylist": [],
                    "sensitive_patterns": [],
                }
            }
        }
    )
    layers = [session, user] if reverse else [user, session]
    return await ConfigOrchestrator.create(
        schema=ChartreuxConfigSchema,
        layers=[DefaultConfigLayer(schema=ChartreuxConfigSchema), *layers],
        default_layer_resolver=lambda: session,
    )


@pytest.mark.parametrize("restriction", ["never", "denylist"])
@pytest.mark.parametrize("reverse", [False, True])
async def test_production_loop_denies_hidden_source_under_policy(
    tmp_path: Path, restriction: str, reverse: bool
) -> None:
    path = tmp_path / "settings.toml"
    path.write_text(
        "[tools.read_file]\n"
        + (
            'permission = "never"\n'
            if restriction == "never"
            else 'denylist = ["*fixture.txt"]\n'
        )
    )
    target = tmp_path / "fixture.txt"
    target.write_text("harmless fixture")
    orchestrator = await make_orchestrator(path, reverse=reverse)
    backend = FakeBackend([
        [
            mock_llm_chunk(
                tool_calls=[_call("read_file", {"file_path": str(target)}, "denied")]
            )
        ],
        [mock_llm_chunk(tool_calls=[_call("todo", {"action": "read"}, "continued")])],
        [mock_llm_chunk(content="Continued")],
    ])
    agent = AgentLoop(config_orchestrator=orchestrator, backend=backend, cwd=tmp_path)
    try:
        if not reverse:
            assert agent.config.tools["read_file"]["permission"] == "always"
            assert agent.config.tools["read_file"]["denylist"] == []
        # Also exercises the replacement production manager, not just startup.
        await agent.reload_with_initial_messages(reload_config=True)
        events = await _collect(agent)
        assert not hasattr(event_module, "ApprovalRequestEvent")
        assert not any("approval" in type(event).__name__.lower() for event in events)
        results = [e for e in events if isinstance(e, ToolResultEvent)]
        assert len(results) == 2
        assert results[0].skipped and results[0].skip_reason
        assert results[0].result is None
        assert results[1].result is not None and not results[1].skipped
    finally:
        await agent.aclose()


@pytest.mark.parametrize(
    "field,value",
    [("permission", '"invalid"'), ("denylist", "[23]"), ("sensitive_patterns", "[23]")],
)
async def test_shadowed_invalid_source_rejected_on_create(
    tmp_path: Path, field: str, value: str
) -> None:
    path = tmp_path / "settings.toml"
    path.write_text(f"[tools.read_file]\n{field} = {value}\n")
    with pytest.raises(
        ValueError, match=rf"user.*settings.toml.*tools.read_file.{field}"
    ):
        await make_orchestrator(path)


async def test_preview_failure_and_copy_keep_accepted_authority(tmp_path: Path) -> None:
    path = tmp_path / "settings.toml"
    path.write_text('[tools.read_file]\npermission = "never"\n')
    orchestrator = await make_orchestrator(path)
    accepted = orchestrator.config
    restrictions = orchestrator.restrictions
    layer = orchestrator.get_layer("user-toml")
    cached = layer.cached_data
    fingerprint = layer.fingerprint
    path.write_text('[tools.read_file]\npermission = "always"\n')
    preview = await orchestrator.preview_candidate(force_load=True)
    assert preview.restrictions != restrictions
    preview.config.tools.clear()
    assert orchestrator.config is accepted
    assert orchestrator.restrictions is restrictions
    assert layer.cached_data is cached and layer.fingerprint == fingerprint

    async def reject(_: ChartreuxConfigSchema) -> None:
        raise RuntimeError("preparation failed")

    with pytest.raises(RuntimeError, match="preparation failed"):
        await orchestrator.reload(preflight=reject)
    assert orchestrator.config is accepted
    assert orchestrator.restrictions is restrictions
    assert layer.cached_data is cached and layer.fingerprint == fingerprint
    path.write_text('[tools.read_file]\npermission = "invalid"\n')
    with pytest.raises(ValueError, match="tools.read_file.permission"):
        await orchestrator.reload()
    assert orchestrator.restrictions is restrictions
    assert layer.cached_data is cached
    clone = orchestrator.copy()
    clone.config.tools.clear()
    assert orchestrator.config.tools
    assert clone.restrictions == restrictions
    # The clone starts at accepted state, not a fresh read of rejected bytes.
    assert clone.get_layer("user-toml").fingerprint == fingerprint
    path.write_text('[tools.read_file]\npermission = "always"\n')
    await clone.reload()
    assert orchestrator.restrictions is restrictions
    assert clone.restrictions != restrictions
    await orchestrator.reload()
    assert orchestrator.restrictions == clone.restrictions
    assert layer.fingerprint != fingerprint


@pytest.mark.parametrize("failure", ["validation", "preparation"])
async def test_failed_reload_keeps_live_cached_tool_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    path = tmp_path / "settings.toml"
    path.write_text('[tools.read_file]\npermission = "never"\n')
    orchestrator = await make_orchestrator(path)
    target = tmp_path / "fixture.txt"
    target.write_text("fixture")
    agent = AgentLoop(
        config_orchestrator=orchestrator,
        cwd=tmp_path,
        backend=FakeBackend([
            [
                mock_llm_chunk(
                    tool_calls=[
                        _call("read_file", {"file_path": str(target)}, "denied")
                    ]
                )
            ],
            [mock_llm_chunk(content="Done")],
        ]),
    )
    cached = agent.tool_manager.get("read_file")
    accepted = orchestrator.config
    restrictions = orchestrator.restrictions
    try:
        permission = "invalid" if failure == "validation" else "always"
        path.write_text(f'[tools.read_file]\npermission = "{permission}"\n')

        def reject_backend(_: ChartreuxConfigSchema) -> FakeBackend:
            raise RuntimeError("synthetic preparation failure")

        with monkeypatch.context() as patch:
            if failure == "preparation":
                patch.setattr(agent, "backend_factory", reject_backend)
            with pytest.raises(ValueError if failure == "validation" else RuntimeError):
                await agent.reload_with_initial_messages(reload_config=True)
        assert orchestrator.config is accepted
        assert orchestrator.restrictions is restrictions
        assert agent.tool_manager.get("read_file") is cached
        events = await _collect(agent)
        assert not hasattr(event_module, "ApprovalRequestEvent")
        assert not any("approval" in type(event).__name__.lower() for event in events)
        result = next(e for e in events if isinstance(e, ToolResultEvent))
        assert result.skipped and result.result is None
    finally:
        await agent.aclose()


async def test_sensitive_contributions_survive_ordinary_clear(tmp_path: Path) -> None:
    path = tmp_path / "settings.toml"
    path.write_text('[tools.read_file]\nsensitive_patterns = ["*.private"]\n')
    orchestrator = await make_orchestrator(path)
    agent = AgentLoop(
        config_orchestrator=orchestrator, cwd=tmp_path, backend=FakeBackend()
    )
    try:
        assert agent.config.tools["read_file"]["sensitive_patterns"] == []
        assert (
            "*.private"
            in agent.tool_manager.get_tool_config("read_file").sensitive_patterns
        )
        before = orchestrator.restrictions
        invalid: dict[str, Any] = {"tools": {"read_file": {"denylist": [23]}}}
        with pytest.raises(ValueError):
            await orchestrator.preview_candidate(
                layer_overrides={"overrides": RawConfig.model_validate(invalid)}
            )
        assert orchestrator.restrictions is before
    finally:
        await agent.aclose()
