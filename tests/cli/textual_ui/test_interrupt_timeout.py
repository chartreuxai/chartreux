from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock

import pytest

from chartreux.app_server.session import AppServerSession
from chartreux.cli.textual_ui import app as app_module
from chartreux.cli.textual_ui.app import ChartreuxApp
from chartreux.cli.textual_ui.widgets.messages import InterruptMessage
from tests.conftest import build_test_chartreux_app


async def _wait_until(predicate, timeout: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0)
    return False


@pytest.mark.asyncio
async def test_cancellation_resistant_client_task_does_not_hang_ui(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = build_test_chartreux_app()
    release = asyncio.Event()
    warnings: list[str] = []
    monkeypatch.setattr(app_module, "_INTERRUPT_WAIT_TIMEOUT", 0.01)

    async with app.run_test():

        async def resistant_task() -> None:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await release.wait()
                raise

        app._agent_task = asyncio.create_task(resistant_task())
        await asyncio.sleep(0)
        monkeypatch.setattr(
            app, "notify", lambda message, **_kwargs: warnings.append(str(message))
        )
        app._begin_interrupt_settle()

        await asyncio.wait_for(app._interrupt_turn(), timeout=0.2)

        assert warnings == [app_module._INTERRUPT_STILL_STOPPING_WARNING]
        assert app._interrupt_requested
        assert app._interrupt_operation is not None
        assert app._agent_job_active()
        assert not list(app.query(InterruptMessage))

        release.set()
        assert await _wait_until(lambda: app._interrupt_operation is None)
        assert len(list(app.query(InterruptMessage))) == 1


@pytest.mark.asyncio
async def test_server_interrupt_timeout_blocks_submit_past_settle_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app: ChartreuxApp = build_test_chartreux_app()
    release = asyncio.Event()
    active = {"value": True}
    dispatched: list[str] = []
    interrupt_calls = 0
    monkeypatch.setattr(app_module, "_INTERRUPT_WAIT_TIMEOUT", 0.01)
    monkeypatch.setattr(app_module, "_INTERRUPT_SETTLE_TIMEOUT", 0.01)
    monkeypatch.setattr(
        AppServerSession, "turn_active", property(lambda _self: active["value"])
    )

    async with app.run_test():

        async def gated_interrupt() -> None:
            nonlocal interrupt_calls
            interrupt_calls += 1
            await release.wait()

        async def record_dispatch(value: str) -> None:
            dispatched.append(value)

        monkeypatch.setattr(app.app_server, "interrupt", gated_interrupt)
        monkeypatch.setattr(app, "_dispatch_submitted_value", record_dispatch)
        monkeypatch.setattr(app, "notify", lambda *_args, **_kwargs: None)
        app._begin_interrupt_settle()
        await app._interrupt_turn()

        # The turn can look idle before the interrupt request settles. Neither
        # that ordering nor expiry of the ordinary settle window admits a turn.
        active["value"] = False
        await app._submit_or_defer("replacement")
        await asyncio.sleep(0.05)
        assert dispatched == []
        assert app._interrupt_pending
        assert app._agent_job_active()

        # Repeated Escape owns the same request rather than issuing another one,
        # even if turn finalization has already cleared its presentation flag.
        app._interrupt_requested = False
        await app._interrupt_turn()
        assert interrupt_calls == 1

        release.set()
        assert await _wait_until(lambda: dispatched == ["replacement"])
        assert len(list(app.query(InterruptMessage))) == 1


@pytest.mark.asyncio
async def test_late_interrupt_completion_cannot_cancel_replacement_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = build_test_chartreux_app()
    release = asyncio.Event()
    active = {"value": True}
    replacement_started = asyncio.Event()
    monkeypatch.setattr(app_module, "_INTERRUPT_WAIT_TIMEOUT", 0.01)
    monkeypatch.setattr(
        AppServerSession, "turn_active", property(lambda _self: active["value"])
    )

    async with app.run_test():

        async def gated_interrupt() -> None:
            await release.wait()

        async def replacement(_value: str) -> None:
            replacement_started.set()

        monkeypatch.setattr(app.app_server, "interrupt", gated_interrupt)
        monkeypatch.setattr(app, "_dispatch_submitted_value", replacement)
        monkeypatch.setattr(app, "notify", lambda *_args, **_kwargs: None)
        app._begin_interrupt_settle()
        await app._interrupt_turn()
        active["value"] = False
        await app._submit_or_defer("replacement")
        await asyncio.sleep(0)
        assert not replacement_started.is_set()

        release.set()
        await asyncio.wait_for(replacement_started.wait(), timeout=1.0)
        assert app._interrupt_operation is None


@pytest.mark.asyncio
async def test_interrupt_eventual_failure_is_consumed_and_torn_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = build_test_chartreux_app()
    release = asyncio.Event()
    active = {"value": True}
    warnings: list[str] = []
    monkeypatch.setattr(app_module, "_INTERRUPT_WAIT_TIMEOUT", 0.01)
    monkeypatch.setattr(
        AppServerSession, "turn_active", property(lambda _self: active["value"])
    )

    async with app.run_test():

        async def failing_interrupt() -> None:
            await release.wait()
            raise RuntimeError("interrupt transport failed")

        monkeypatch.setattr(app.app_server, "interrupt", failing_interrupt)
        monkeypatch.setattr(
            app, "notify", lambda message, **_kwargs: warnings.append(str(message))
        )
        app._begin_interrupt_settle()
        await app._interrupt_turn()
        release.set()

        assert await _wait_until(lambda: app._interrupt_operation is None)
        assert not app._interrupt_requested
        assert not list(app.query(InterruptMessage))
        assert any("interrupt transport failed" in warning for warning in warnings)


@pytest.mark.asyncio
async def test_finalization_failure_after_timeout_is_consumed_and_settled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = build_test_chartreux_app()
    release_interrupt = asyncio.Event()
    finalization_started = asyncio.Event()
    release_finalization = asyncio.Event()
    active = {"value": True}
    monkeypatch.setattr(app_module, "_INTERRUPT_WAIT_TIMEOUT", 0.01)
    monkeypatch.setattr(
        AppServerSession, "turn_active", property(lambda _self: active["value"])
    )

    async with app.run_test():

        async def gated_interrupt() -> None:
            await release_interrupt.wait()

        async def failing_finalization() -> None:
            finalization_started.set()
            await release_finalization.wait()
            raise RuntimeError("late DOM failure")

        monkeypatch.setattr(app.app_server, "interrupt", gated_interrupt)
        monkeypatch.setattr(
            app.event_handler, "finalize_streaming", failing_finalization
        )
        monkeypatch.setattr(app, "notify", lambda *_args, **_kwargs: None)
        app._begin_interrupt_settle()
        await app._interrupt_turn()
        operation = app._interrupt_operation
        assert operation is not None

        active["value"] = False
        release_interrupt.set()
        await asyncio.wait_for(finalization_started.wait(), timeout=1.0)
        assert app._interrupt_operation is operation
        assert app._interrupt_pending

        release_finalization.set()
        await asyncio.wait_for(operation, timeout=1.0)
        assert app._interrupt_operation is None
        assert not app._interrupt_pending
        assert not app._interrupt_requested


@pytest.mark.asyncio
async def test_shutdown_cancels_pending_interrupt_supervisor() -> None:
    app = build_test_chartreux_app()
    interrupt_started = asyncio.Event()

    async with app.run_test():

        async def pending_interrupt() -> None:
            interrupt_started.set()
            await asyncio.Event().wait()

        app._begin_interrupt_settle()
        operation = asyncio.create_task(app._finish_interrupt(pending_interrupt()))
        app._interrupt_operation = operation
        await asyncio.wait_for(interrupt_started.wait(), timeout=1.0)

        await asyncio.wait_for(app.shutdown_cleanup(), timeout=1.0)

        assert operation.cancelled()
        assert app._interrupt_operation is None
        assert not app._interrupt_pending


@pytest.mark.asyncio
async def test_incomplete_stream_retry_proceeds_when_interrupt_transport_settles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = build_test_chartreux_app()
    release_interrupt = asyncio.Event()
    dispatched: list[int] = []
    monkeypatch.setattr(app_module, "_INTERRUPT_SETTLE_TIMEOUT", 0.01)

    async with app.run_test():
        assert app.event_handler is not None

        async def record_retry(
            _prompt: str, *, incomplete_stream_retries: int, **_kwargs: object
        ) -> None:
            dispatched.append(incomplete_stream_retries)

        async def interrupt_transport() -> None:
            await release_interrupt.wait()

        monkeypatch.setattr(app.event_handler, "begin_retry", lambda: True)
        monkeypatch.setattr(app.event_handler, "finalize_streaming", AsyncMock())
        monkeypatch.setattr(app, "_handle_turn", record_retry)
        app._begin_interrupt_settle()
        operation = asyncio.create_task(app._finish_interrupt(interrupt_transport()))
        app._interrupt_operation = operation
        retry = asyncio.create_task(app._auto_retry_incomplete_stream(1))

        # Expiring the ordinary settle deadline must not strand the queued retry
        # while the interrupt transport still owns the gate.
        await asyncio.sleep(0.05)
        assert dispatched == []
        assert app._interrupt_pending

        release_interrupt.set()
        await asyncio.wait_for(operation, timeout=1.0)
        await asyncio.wait_for(retry, timeout=1.0)
        assert not app._interrupt_pending
        assert dispatched == [1]
