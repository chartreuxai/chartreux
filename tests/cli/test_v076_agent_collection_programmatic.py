"""v0.7.6 P2 chunk 2c: the harness-owned continuation lifecycle in
programmatic mode (``chartreux/cli/programmatic.py``).

``run_programmatic`` consumes one ``act()`` stream and then runs the
continuation: while owned dependency runs are active the session stays open
and consumes notification-driven turns through the continuation event
subscription; termination happens only at the atomic quiescence check; at the
wall-clock cap (or a mid-continuation STOP) active dependency runs are
cancelled via the registry cancel path and every completed-but-unfetched
result is force-surfaced in the terminal outcome before finalization.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import pytest

from chartreux.app_server.local import ClientDescriptor, LocalHarnessOptions
from chartreux.app_server.protocol import ClientCapabilities, ClientInfo, SessionOptions
from chartreux.cli.programmatic import (
    OutputFormat,
    ProgrammaticLimitError,
    run_programmatic,
)
from chartreux.core.config import ChartreuxConfigSchema
from chartreux.core.config.harness_files import HarnessFilesManager
from chartreux.core.config.layers.default import DefaultConfigLayer
from chartreux.core.config.layers.overrides import OverridesLayer
from chartreux.core.config.layers.user import UserConfigLayer
from chartreux.core.config.orchestrator import ConfigOrchestrator
from chartreux.core.llm_models import FunctionCall, LLMChunk, ToolCall
from tests.app_server._subagent_test_support import _task_call
from tests.conftest import build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend

_CATALOG_FIELDS = {"models", "providers"}
_ORDINARY_OVERRIDE_EXCLUDED_FIELDS = {
    *_CATALOG_FIELDS,
    "authorized_roots_by_project",
    "dangerous_command_grants_by_project",
    "remote_exec_grants_by_project",
}


def _options(*, ask_user: bool = False) -> LocalHarnessOptions:
    disabled = [] if ask_user else ["ask_user_question"]
    return LocalHarnessOptions(
        client=ClientDescriptor(
            info=ClientInfo(
                name="vibe_programmatic", version="test", entrypoint="programmatic"
            ),
            capabilities=ClientCapabilities(callback_kinds=["user_input"]),
        ),
        session_options=SessionOptions(disabled_tools=disabled, headless=True),
    )


def _use_runtime_config(
    monkeypatch: pytest.MonkeyPatch, config: ChartreuxConfigSchema, tmp_path: Path
) -> None:
    async def build_orchestrator(
        data: dict[str, object] | None = None, *, harness_files: HarnessFilesManager
    ) -> ConfigOrchestrator[ChartreuxConfigSchema]:
        del harness_files
        user_path = tmp_path / "user.toml"
        user_path.write_text(
            "credential_env_passthrough = "
            f"{json.dumps(config.credential_env_passthrough)}\n",
            encoding="utf-8",
        )
        user = UserConfigLayer(path=user_path)
        base = OverridesLayer(
            data=config.model_dump(
                mode="json",
                exclude={
                    *_ORDINARY_OVERRIDE_EXCLUDED_FIELDS,
                    "credential_env_passthrough",
                },
            ),
            name="base",
        )
        default = DefaultConfigLayer(schema=ChartreuxConfigSchema)
        session = OverridesLayer(data=data or {})
        return await ConfigOrchestrator.create(
            schema=ChartreuxConfigSchema,
            layers=[default, user, base, session],
            default_layer_resolver=lambda: base,
        )

    monkeypatch.setattr(
        "chartreux.app_server._runtime.build_default_orchestrator", build_orchestrator
    )


def _install_backends(
    monkeypatch: pytest.MonkeyPatch, parent_backend: Any, child_backends: list[Any]
) -> None:
    """Root-loop attempts use ``parent_backend``; each child loop gets its own
    injected backend (launch order).
    """
    from chartreux.app_server._runtime import AgentRuntimeFactory
    from chartreux.core.agent_loop import AgentLoop

    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend",
        lambda **kwargs: parent_backend,
    )
    remaining = list(child_backends)
    original_create_child = AgentRuntimeFactory.create_child

    async def create_child_with_backend(
        factory: AgentRuntimeFactory, loop: AgentLoop, agent_name: str, **kwargs: Any
    ) -> AgentLoop:
        child = await original_create_child(factory, loop, agent_name, **kwargs)
        if remaining:
            child._injected_backend = remaining.pop(0)
        return child

    monkeypatch.setattr(AgentRuntimeFactory, "create_child", create_child_with_backend)


class _DelayedChildBackend(FakeBackend):
    """A child that completes after a short delay (no cross-thread signal)."""

    def __init__(self, content: str = "Child result body", delay: float = 0.2) -> None:
        super().__init__()
        self._content = content
        self._delay = delay

    async def complete(self, **kwargs: Any) -> Any:
        await asyncio.sleep(self._delay)
        return mock_llm_chunk(content=self._content)

    async def complete_streaming(self, **kwargs: Any) -> AsyncGenerator[LLMChunk, None]:
        yield await self.complete(**kwargs)


class _NeverCompletingChildBackend(FakeBackend):
    """A child whose run never terminates on its own (budget/cap must stop it)."""

    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self._release = asyncio.Event()

    async def complete(self, **kwargs: Any) -> Any:
        self.started.set()
        await self._release.wait()
        return mock_llm_chunk(content="unreachable")

    async def complete_streaming(self, **kwargs: Any) -> AsyncGenerator[LLMChunk, None]:
        yield await self.complete(**kwargs)


def _fetch_call(agent_id: str = "agent-1", call_id: str = "call-fetch") -> ToolCall:
    return ToolCall(
        id=call_id,
        index=0,
        function=FunctionCall(
            name="get_agent_result", arguments=json.dumps({"agent_id": agent_id})
        ),
    )


def _quiescence_spy(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    """Record every atomic quiescence check's post-declaration state (2a)."""
    from chartreux.app_server._sessions import SessionRuntimeRegistry

    declared_after: list[bool] = []
    original = SessionRuntimeRegistry.check_quiescence_and_declare_termination

    async def spy(self: SessionRuntimeRegistry) -> Any:
        snapshot = await original(self)
        declared_after.append(self.termination_declared)
        return snapshot

    monkeypatch.setattr(
        SessionRuntimeRegistry, "check_quiescence_and_declare_termination", spy
    )
    return declared_after


@pytest.mark.timeout(30)
def test_continuation_stays_open_until_atomic_quiescence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The session stays open while the dependency run is active, consumes the
    notification-driven turn across the act() boundary, and terminates only at
    the atomic quiescence check (2a's declaration under the held acquisition).
    """
    config = build_test_vibe_config(
        include_model_info=False, include_commit_signature=False
    )
    _use_runtime_config(monkeypatch, config, tmp_path)
    declared_after = _quiescence_spy(monkeypatch)
    parent_backend = FakeBackend([
        [
            mock_llm_chunk(
                content="", tool_calls=[_task_call(background=True, retain=True)]
            )
        ],
        [mock_llm_chunk(content="Parent delegated.")],
        [mock_llm_chunk(content="", tool_calls=[_fetch_call("agent-1")])],
        [mock_llm_chunk(content="Fetched the result.")],
    ])
    _install_backends(monkeypatch, parent_backend, [_DelayedChildBackend()])
    result = run_programmatic(
        harness_options=_options(),
        prompt="Delegate this",
        output_format=OutputFormat.TEXT,
        continuation_cap_seconds=15.0,
    )
    assert result == "Fetched the result."
    # Termination was declared by the atomic check (2a), not by a re-check.
    assert True in declared_after


@pytest.mark.timeout(30)
def test_wall_clock_cap_force_surfaces_unfetched_and_names_cancelled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """At the cap: the never-completing dependency run is cancelled via the
    registry cancel path (named separately), and the completed-but-unfetched
    result is force-surfaced with its full body in the terminal outcome before
    finalization.
    """
    config = build_test_vibe_config(
        include_model_info=False, include_commit_signature=False
    )
    _use_runtime_config(monkeypatch, config, tmp_path)
    parent_backend = FakeBackend([
        [
            mock_llm_chunk(
                content="",
                tool_calls=[
                    _task_call(background=True, retain=True, tool_call_id="task-1"),
                    _task_call(background=True, retain=True, tool_call_id="task-2"),
                ],
            )
        ],
        [mock_llm_chunk(content="Delegated twice.")],
        [mock_llm_chunk(content="Ignoring the notification.")],
    ])
    _install_backends(
        monkeypatch,
        parent_backend,
        [
            _DelayedChildBackend(content="Child one body", delay=0.2),
            _NeverCompletingChildBackend(),
        ],
    )
    result = run_programmatic(
        harness_options=_options(),
        prompt="Delegate twice",
        output_format=OutputFormat.TEXT,
        continuation_cap_seconds=0.8,
    )
    assert result is not None
    assert "wall-clock-cap" in result
    # The completed-but-unfetched result is force-surfaced with its full body.
    assert "Force-surfaced completed-but-unfetched results:" in result
    assert "agent-1" in result and "Child one body" in result
    # The cancelled dependency run is named separately.
    assert "Cancelled dependency runs:" in result
    assert "agent-2" in result
    assert "orchestrator_cancelled" in result


@pytest.mark.timeout(30)
def test_cap_notice_lands_in_json_output_as_terminal_notice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The force-surfaced terminal outcome reaches the JSON route too (as a
    terminal_notice element) before finalization.
    """
    config = build_test_vibe_config(
        include_model_info=False, include_commit_signature=False
    )
    _use_runtime_config(monkeypatch, config, tmp_path)
    parent_backend = FakeBackend([
        [
            mock_llm_chunk(
                content="", tool_calls=[_task_call(background=True, retain=True)]
            )
        ],
        [mock_llm_chunk(content="Delegated.")],
        [mock_llm_chunk(content="Ignoring the notification.")],
    ])
    _install_backends(monkeypatch, parent_backend, [_DelayedChildBackend(delay=0.2)])
    result = run_programmatic(
        harness_options=_options(),
        prompt="Delegate",
        output_format=OutputFormat.JSON,
        continuation_cap_seconds=0.8,
    )
    assert result is None
    # The JSON route dumps one indented array; the terminal notice rides it.
    payload = json.loads(capsys.readouterr().out)
    notices = [
        entry
        for entry in payload
        if isinstance(entry, dict) and entry.get("type") == "terminal_notice"
    ]
    assert len(notices) == 1
    text = notices[0]["text"]
    assert "wall-clock-cap" in text
    assert "Force-surfaced completed-but-unfetched results:" in text
    assert "Child result body" in text


@pytest.mark.timeout(30)
def test_mid_continuation_limit_stop_cancels_and_surfaces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mid-continuation STOP (token/price/turn LIMIT on a notification-driven
    turn) cancels the active dependency run and force-surfaces the terminal
    outcome before the ProgrammaticLimitError is raised.
    """
    config = build_test_vibe_config(
        include_model_info=False, include_commit_signature=False
    )
    _use_runtime_config(monkeypatch, config, tmp_path)
    parent_backend = FakeBackend([
        [
            mock_llm_chunk(
                content="", tool_calls=[_task_call(background=True, retain=True)]
            )
        ],
        [mock_llm_chunk(content="Delegated.")],
    ])
    _install_backends(monkeypatch, parent_backend, [_DelayedChildBackend(delay=0.2)])
    options = _options()
    options = replace(
        options,
        session_options=SessionOptions(
            disabled_tools=["ask_user_question"], headless=True, max_turns=1
        ),
    )
    with pytest.raises(ProgrammaticLimitError) as exc_info:
        run_programmatic(
            harness_options=options,
            prompt="Delegate",
            output_format=OutputFormat.TEXT,
            continuation_cap_seconds=15.0,
        )
    message = str(exc_info.value)
    # The STOP (turn limit) landed inside the continuation lifecycle: the
    # continuation recorded the mid-continuation STOP, cancelled the still
    # active dependency run via the registry cancel path, and force-surfaced
    # the terminal outcome before the ProgrammaticLimitError was raised.
    assert "limit-stop" in message
    assert "Force-surfaced completed-but-unfetched results:" in message
    assert "Cancelled dependency runs:" in message
    assert "agent-1" in message
    assert "orchestrator_cancelled" in message
