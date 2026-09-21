"""Tests for deferred initialization: _complete_init, _wait_for_init, integrate_mcp idempotency."""

from __future__ import annotations

import asyncio
from pathlib import Path
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chartreux.core.agent_loop import AgentLoop
from chartreux.core.config import MCPStdio, SessionLoggingConfig
from chartreux.core.tools.manager import ToolManager
from chartreux.core.tools.mcp import AuthStatus
from tests.conftest import (
    build_test_agent_loop,
    build_test_vibe_config,
    set_agent_config,
)
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend
from tests.stubs.fake_mcp_registry import FakeMCPRegistry


@pytest.mark.asyncio
@pytest.mark.parametrize("defer_heavy_init", [False, True])
async def test_construction_leaves_single_file_sessions_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, defer_heavy_init: bool
) -> None:
    session_dir = tmp_path / "sessions"
    session_dir.mkdir()
    sentinel = session_dir / "session_old.json"
    original = (
        b'{ "metadata": {"session_id": "old"},\n'
        b'  "messages": [{"role": "user", "content": "keep me"}] }\n'
    )
    sentinel.write_bytes(original)
    config = build_test_vibe_config(
        session_logging=SessionLoggingConfig(
            enabled=True, save_dir=str(session_dir), session_prefix="session"
        )
    )
    started: list[threading.Thread] = []
    start = threading.Thread.start

    def record_start(thread: threading.Thread) -> None:
        started.append(thread)
        start(thread)

    with monkeypatch.context() as context:
        context.setattr(threading.Thread, "start", record_start)
        loop = build_test_agent_loop(config=config, defer_heavy_init=defer_heavy_init)
    try:
        await loop.wait_until_ready()
        # Join construction threads so the assertion cannot race an old migrator.
        for thread in started:
            await asyncio.to_thread(thread.join, 5)
            assert not thread.is_alive()
        assert "migrate_sessions" not in [thread.name for thread in started]
        assert sentinel.read_bytes() == original
        assert not sentinel.with_suffix("").exists()
    finally:
        await loop.aclose()


def _build_uninitiated_loop(**kwargs):
    """Build a test loop with defer_heavy_init=True but without auto-starting the init thread."""
    with patch.object(AgentLoop, "_start_deferred_init"):
        return build_test_agent_loop(defer_heavy_init=True, **kwargs)


# ---------------------------------------------------------------------------
# _complete_init
# ---------------------------------------------------------------------------


def _run_init(loop: AgentLoop) -> None:
    """Run _complete_init in a thread (matching production behavior) and wait."""
    thread = threading.Thread(target=loop._complete_init, daemon=True)
    loop._deferred_init_thread = thread
    thread.start()
    thread.join()


class TestCompleteInit:
    def test_success_sets_init_complete(self) -> None:
        loop = _build_uninitiated_loop()
        assert not loop.is_initialized

        _run_init(loop)

        assert loop.is_initialized
        assert loop._init_error is None

    def test_failure_sets_init_complete_and_stores_error(self) -> None:
        loop = _build_uninitiated_loop()
        error = RuntimeError("mcp boom")

        with patch.object(loop.tool_manager, "integrate_all", side_effect=error):
            _run_init(loop)

        assert loop.is_initialized
        assert loop._init_error is error

    def test_mcp_failure_sets_init_error(self) -> None:
        mcp_server = MCPStdio(name="test-server", transport="stdio", command="echo")
        config = build_test_vibe_config(mcp_servers=[mcp_server])
        loop = _build_uninitiated_loop(config=config)

        with patch.object(
            loop.tool_manager,
            "integrate_all",
            side_effect=RuntimeError("mcp discovery boom"),
        ):
            _run_init(loop)

        assert loop.is_initialized
        assert isinstance(loop._init_error, RuntimeError)
        assert str(loop._init_error) == "mcp discovery boom"


# ---------------------------------------------------------------------------
# wait_until_ready
# ---------------------------------------------------------------------------


class TestWaitForInit:
    @pytest.mark.asyncio
    async def test_returns_immediately_when_already_complete(self) -> None:
        loop = build_test_agent_loop(defer_heavy_init=True)

        await loop.wait_until_ready()  # should not block

        assert loop.is_initialized

    @pytest.mark.asyncio
    async def test_waits_for_background_thread(self) -> None:
        loop = build_test_agent_loop(defer_heavy_init=True)

        await loop.wait_until_ready()

        assert loop.is_initialized

    @pytest.mark.asyncio
    async def test_raises_stored_error(self) -> None:
        loop = _build_uninitiated_loop()
        error = RuntimeError("init failed")

        with patch.object(loop.tool_manager, "integrate_all", side_effect=error):
            loop._complete_init()

        with pytest.raises(RuntimeError, match="init failed"):
            await loop.wait_until_ready()

    @pytest.mark.asyncio
    async def test_raises_error_for_every_caller(self) -> None:
        loop = _build_uninitiated_loop()
        error = RuntimeError("once only")

        with patch.object(loop.tool_manager, "integrate_all", side_effect=error):
            loop._complete_init()

        with pytest.raises(RuntimeError):
            await loop.wait_until_ready()

        with pytest.raises(RuntimeError):
            await loop.wait_until_ready()


# ---------------------------------------------------------------------------
# integrate_mcp idempotency
# ---------------------------------------------------------------------------


class TestIntegrateMcpIdempotency:
    def test_second_call_is_noop(self) -> None:
        mcp_server = MCPStdio(name="test-server", transport="stdio", command="echo")
        config = build_test_vibe_config(mcp_servers=[mcp_server])
        registry = FakeMCPRegistry()
        manager = ToolManager(lambda: config, mcp_registry=registry, defer_mcp=True)

        manager.integrate_mcp()
        tools_after_first = dict(manager.registered_tools)

        # Spy on the registry to ensure get_tools is not called again.
        registry.get_tools = MagicMock(wraps=registry.get_tools)
        manager.integrate_mcp()

        registry.get_tools.assert_not_called()
        assert manager.registered_tools == tools_after_first

    def test_flag_not_set_when_no_servers(self) -> None:
        config = build_test_vibe_config(mcp_servers=[])
        manager = ToolManager(lambda: config, defer_mcp=True)

        manager.integrate_mcp()

        # No servers means the method returns early without setting the flag,
        # so a future call with servers would still run discovery.
        assert not manager._mcp_integrated

    def test_no_servers_syncs_shared_registry_status(self) -> None:
        config = build_test_vibe_config(
            mcp_servers=[MCPStdio(name="srv", transport="stdio", command="echo")]
        )
        registry = FakeMCPRegistry()
        manager = ToolManager(lambda: config, mcp_registry=registry, defer_mcp=True)

        manager.integrate_mcp()
        assert registry.status() == {"srv": AuthStatus.STDIO}

        config = build_test_vibe_config(mcp_servers=[])
        manager = ToolManager(lambda: config, mcp_registry=registry, defer_mcp=True)
        manager.integrate_mcp()

        assert registry.status() == {}
        assert not manager._mcp_integrated


class TestRefreshRemoteTools:
    @pytest.mark.asyncio
    async def test_refresh_rediscovers_mcp_tools(self) -> None:
        mcp_server = MCPStdio(name="srv", transport="stdio", command="echo")
        config = build_test_vibe_config(mcp_servers=[mcp_server])
        registry = FakeMCPRegistry()
        registry.get_tools_async = AsyncMock(wraps=registry.get_tools_async)
        manager = ToolManager(lambda: config, mcp_registry=registry, defer_mcp=True)

        await manager.refresh_remote_tools_async()

        assert "srv_fake_tool" in manager.registered_tools

        await manager.refresh_remote_tools_async()

        assert registry.get_tools_async.await_count == 2
        assert "srv_fake_tool" in manager.registered_tools


class TestDeferredInitPublicMethods:
    @pytest.mark.asyncio
    async def test_act_waits_for_deferred_init(self) -> None:
        loop = build_test_agent_loop(
            defer_heavy_init=True, backend=FakeBackend(mock_llm_chunk(content="hello"))
        )

        events = [event async for event in loop.act("Hello")]

        assert loop.is_initialized
        assert [event.content for event in events if hasattr(event, "content")][
            -1
        ] == "hello"

    @pytest.mark.asyncio
    async def test_reload_with_initial_messages_waits_for_deferred_init(self) -> None:
        loop = build_test_agent_loop(defer_heavy_init=True)

        await loop.reload_with_initial_messages()

        assert loop.is_initialized

    @pytest.mark.asyncio
    async def test_reload_creates_shared_mcp_registry_after_servers_are_added(
        self,
    ) -> None:
        loop = build_test_agent_loop(defer_heavy_init=True, mcp_registry=None)
        await loop.wait_until_ready()
        assert loop.mcp_registry is None

        mcp_server = MCPStdio(name="srv", transport="stdio", command="echo")
        config = build_test_vibe_config(mcp_servers=[mcp_server])
        registry = FakeMCPRegistry()
        set_agent_config(loop, config)

        with (
            patch.object(AgentLoop, "_create_mcp_registry", return_value=registry),
            patch.object(ToolManager, "integrate_all"),
        ):
            await loop.reload_with_initial_messages()

        assert loop.mcp_registry is registry
        assert loop.tool_manager._mcp_registry is registry

    @pytest.mark.asyncio
    async def test_clear_history_waits_for_deferred_init(self) -> None:
        loop = build_test_agent_loop(
            defer_heavy_init=True, backend=FakeBackend(mock_llm_chunk(content="hello"))
        )
        [_ async for _ in loop.act("Hello")]

        await loop.clear_history()

        assert loop.is_initialized
        assert len(loop.messages) == 1

    @pytest.mark.asyncio
    async def test_compact_waits_for_deferred_init(self) -> None:
        loop = build_test_agent_loop(
            defer_heavy_init=True,
            backend=FakeBackend([
                [mock_llm_chunk(content="hello")],
                [mock_llm_chunk(content="<summary>summary</summary>")],
            ]),
        )
        [_ async for _ in loop.act("Hello")]

        summary = await loop.compact()

        assert loop.is_initialized
        assert summary == "summary"

    @pytest.mark.asyncio
    async def test_inject_user_context_waits_for_deferred_init(self) -> None:
        loop = build_test_agent_loop(defer_heavy_init=True)

        await loop.inject_user_context("context")

        assert loop.is_initialized
        assert loop.messages[-1].content == "context"


class TestInitDurationMsProperty:
    @pytest.mark.asyncio
    async def test_is_none_before_wait_until_ready_on_deferred_path(self) -> None:
        loop = _build_uninitiated_loop()

        assert loop.init_duration_ms is None

    @pytest.mark.asyncio
    async def test_is_populated_after_wait_until_ready_on_deferred_path(self) -> None:
        loop = build_test_agent_loop(defer_heavy_init=True)

        assert loop.init_duration_ms is None
        await loop.wait_until_ready()

        duration = loop.init_duration_ms
        assert duration is not None
        assert isinstance(duration, int)
        assert duration >= 0

    @pytest.mark.asyncio
    async def test_stays_none_on_non_deferred_path(self) -> None:
        loop = build_test_agent_loop(defer_heavy_init=False)

        await loop.wait_until_ready()

        assert loop.init_duration_ms is None
