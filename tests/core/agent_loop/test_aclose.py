from __future__ import annotations

import asyncio
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.conftest import build_test_agent_loop


@pytest.mark.asyncio
async def test_aclose_preserves_lease_after_mcp_failure_and_retries() -> None:
    loop = build_test_agent_loop()
    lease = MagicMock()
    pool = MagicMock()
    pool.cleanup_complete = False
    pool.aclose = AsyncMock(side_effect=[RuntimeError("MCP still closing"), None])
    loop._session_lease = lease
    loop._mcp_pool = pool

    with pytest.raises(BaseExceptionGroup, match="MCP shutdown cleanup incomplete"):
        await loop.aclose()

    lease.release.assert_not_called()
    assert loop._session_lease is lease

    pool.cleanup_complete = True
    await loop.aclose()

    assert pool.aclose.await_count == 2
    lease.release.assert_called_once_with()
    assert loop._session_lease is None


@pytest.mark.asyncio
async def test_cancelled_aclose_waiter_does_not_strand_owned_cleanup(
    monkeypatch,
) -> None:
    loop = build_test_agent_loop()
    started = asyncio.Event()
    release = asyncio.Event()

    async def close_owned() -> None:
        started.set()
        await release.wait()

    monkeypatch.setattr(loop, "_aclose_owned", close_owned)
    waiter = asyncio.create_task(loop.aclose())
    await asyncio.wait_for(started.wait(), timeout=1)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter

    release.set()
    await asyncio.wait_for(loop.aclose(), timeout=1)
    assert loop._close_task is not None and loop._close_task.done()


@pytest.mark.asyncio
async def test_aclose_deferred_init_timeout_keeps_lease_for_retry(monkeypatch) -> None:
    loop = build_test_agent_loop()
    lease = MagicMock()

    class StuckInit:
        alive = True

        def join(self, _timeout: float) -> None:
            return None

        def is_alive(self) -> bool:
            return self.alive

    stuck = StuckInit()
    loop._session_lease = lease
    loop._deferred_init_thread = cast(Any, stuck)

    async def immediate_to_thread(function, *args):
        return function(*args)

    monkeypatch.setattr(asyncio, "to_thread", immediate_to_thread)
    with pytest.raises(TimeoutError, match="Deferred initialization did not stop"):
        await loop.aclose()
    lease.release.assert_not_called()
    assert loop._session_lease is lease

    stuck.alive = False
    await loop.aclose()
    lease.release.assert_called_once_with()
    assert loop._session_lease is None
