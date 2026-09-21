from __future__ import annotations

import pytest

from chartreux.core.utils.concurrency import run_sync


async def _value() -> str:
    return "completed"


def test_run_sync_succeeds_without_active_loop() -> None:
    assert run_sync(_value()) == "completed"


@pytest.mark.asyncio
async def test_run_sync_succeeds_from_active_loop() -> None:
    assert run_sync(_value()) == "completed"


@pytest.mark.asyncio
async def test_run_sync_preserves_runtime_error_from_thread_future() -> None:
    original = RuntimeError("builder failed")

    async def fail() -> None:
        raise original

    with pytest.raises(RuntimeError) as raised:
        run_sync(fail())

    assert raised.value is original
    assert str(raised.value) == "builder failed"
