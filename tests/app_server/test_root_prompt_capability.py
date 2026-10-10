from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

from chartreux.app_server import _runtime
from chartreux.app_server.protocol import ClientCapabilities, ClientInfo, SessionOptions
from chartreux.core.agent_loop import AgentRuntimePolicy
from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
from chartreux.core.config.orchestrator import ConfigOrchestrator


def _policy(**changes: object) -> AgentRuntimePolicy:
    values: dict[str, object] = {
        "max_turns": None,
        "max_price": None,
        "max_tokens": None,
        "max_session_tokens": None,
        "enable_streaming": False,
        "launch_context": None,
        "headless": False,
        "hook_config_result": None,
        "cache_store": None,  # type: ignore[arg-type]
    }
    values.update(changes)
    return AgentRuntimePolicy(**values)  # type: ignore[arg-type]


def _blueprint(
    entrypoint: str, *, headless: bool = False
) -> _runtime._RootRuntimeBlueprint:
    return _runtime._RootRuntimeBlueprint(
        config_orchestrator=SimpleNamespace(
            config=SimpleNamespace(
                session_logging=SimpleNamespace(generate_titles=False)
            ),
            copy=lambda: SimpleNamespace(),
        ),  # type: ignore[arg-type]
        harness_files=None,  # type: ignore[arg-type]
        options=SessionOptions(cwd=str(Path.cwd()), headless=headless),
        client_info=ClientInfo(
            name="test",
            title="Test",
            version="1",
            entrypoint=entrypoint,  # type: ignore[arg-type]
        ),
        client_capabilities=ClientCapabilities(callback_kinds=["user_input"]),
        hook_config_result=None,  # type: ignore[arg-type]
        cache_store=None,  # type: ignore[arg-type]
    )


def test_policy_capability_defaults_false() -> None:
    assert _policy().user_input_capability is False


def test_root_blueprint_sets_capability_truth_table(monkeypatch) -> None:
    captured: list[AgentRuntimePolicy] = []

    def build(self):
        captured.append(self.policy)
        return SimpleNamespace()

    monkeypatch.setattr(_runtime._AgentLoopBlueprint, "build", build)
    for entrypoint, headless, expected in (
        ("cli", False, True),
        ("programmatic", True, False),
        ("acp", False, False),
    ):
        _blueprint(entrypoint, headless=headless).build()
        assert captured[-1].user_input_capability is expected


def test_runtime_policy_copy_and_child_policy_inherit_capability() -> None:
    policy = _policy(user_input_capability=True)
    assert replace(policy).user_input_capability is True

    loop = object.__new__(_runtime.AgentLoop)
    loop._user_input_capability = True
    loop._max_turns = None
    loop._max_price = None
    loop._max_tokens = None
    loop._max_session_tokens = None
    loop.enable_streaming = False
    loop.launch_context = None
    loop._headless = False
    loop._hook_config_result = None
    loop.cache_store = None  # type: ignore[assignment]
    loop._auto_title_enabled = False
    loop._inherited_restrictions = ()
    loop._inherited_mode_restrictions = ()
    loop._inherited_workspace = None
    loop._inherited_plan_write_scopes = ()
    loop._parent_authority_getter = None
    loop._parent_authority_revision_getter = None
    loop._config_orchestrator = cast(
        ConfigOrchestrator[ChartreuxConfigSchema], SimpleNamespace(restrictions=())
    )
    loop.agent_manager = SimpleNamespace(config=SimpleNamespace())  # type: ignore[assignment]
    loop.tool_manager = SimpleNamespace(workspace=None)  # type: ignore[assignment]
    loop._authority_revision = 0
    assert loop.runtime_policy.user_input_capability is True
    assert loop.child_runtime_policy.user_input_capability is True
