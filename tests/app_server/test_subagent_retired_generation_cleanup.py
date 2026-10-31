"""Retired-generation cleanup of the readable logging-disabled children (P1c).

A logging-disabled foreground child leaves no on-disk transcript, so after
completion the registry keeps its closed runtime as a read-only snapshot in
``_readable_children``. Generation retirement now drops those snapshots
(design-accepted degradation): the three fallback readers that consult
``_readable_children`` — ``require_child``, the transcript ``resolve_child``
read, and ``_resolve_session_loop`` (the resolver both grant ports are wired
with) — all degrade for the retired sessions, while stored results stay
readable.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from tests.app_server._subagent_test_support import (
    _foreground_registry,
    _foreground_result,
    _stored_result,
)
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend


@pytest.mark.asyncio
async def test_generation_retirement_clears_readable_children_and_degrades_fallback_readers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    registry, parent = await _foreground_registry(
        monkeypatch, FakeBackend([mock_llm_chunk(content="done")])
    )
    try:
        result = await _foreground_result(registry, parent)
        assert result.completed
        await asyncio.gather(*tuple(registry._teardown_tasks))

        # A logging-disabled foreground child completed into a read-only
        # snapshot readable in its own generation.
        assert set(registry._readable_children) == set(
            registry._readable_children_identities
        )
        child_session_id = next(iter(registry._readable_children))
        assert registry.resolve_child(child_session_id) is not None
        assert registry._resolve_session_loop(child_session_id) is not None

        stored = _stored_result("agent-stored", "run-1")
        registry._result_store[(stored.agent_id, stored.run_id)] = stored

        parent._session_generation = 1
        registry.begin_root_generation()

        # The retired generation's snapshots are gone and the identities with
        # them.
        assert registry._readable_children == {}
        assert registry._readable_children_identities == {}

        # Stored results survive retirement.
        preserved = await registry.get_agent_result(stored.agent_id, stored.run_id)
        assert preserved is not None and preserved == stored.result

        # Degradation pin 1: require_child falls back to _readable_children
        # (chartreux/app_server/_child_runtime_lifecycle.py:260-266) and now
        # raises.
        with pytest.raises(KeyError):
            registry._require_child(child_session_id)

        # Degradation pin 2: the transcript read's _readable_children
        # fallback (chartreux/app_server/_subagent_transcript.py:71) now
        # resolves nothing — the child has no on-disk transcript, so the
        # read falls to "no transcript".
        assert registry.resolve_child(child_session_id) is None

        # Degradation pin 3: the grant ports' resolver
        # (chartreux/app_server/_sessions.py:766) now returns None for the
        # retired session; the root-grant port surfaces the failed
        # resolution as its registered-session error.
        assert registry._resolve_session_loop(child_session_id) is None
        port = parent._root_grant_port
        assert port is not None
        with pytest.raises(RuntimeError, match="requires a registered session"):
            await port.grant_root(child_session_id, tmp_path)
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_drain_children_clears_readable_children_identities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The drain path retires the snapshots and their identity tuples together.

    ``_readable_children`` and ``_readable_children_identities`` are keyed by
    the same session ids; the generation-retirement path pops both. The drain
    path (``_drain_remaining_runs``) clears the snapshots wholesale, so it must
    clear the identity map wholesale too — a stale identity tuple outlives the
    session it belongs to otherwise.
    """
    registry, parent = await _foreground_registry(
        monkeypatch, FakeBackend([mock_llm_chunk(content="done")])
    )
    drained = False
    try:
        result = await _foreground_result(registry, parent)
        assert result.completed
        await asyncio.gather(*tuple(registry._teardown_tasks))

        assert registry._readable_children
        assert set(registry._readable_children) == set(
            registry._readable_children_identities
        )

        await registry.drain_children()
        drained = True

        assert registry._readable_children == {}
        assert registry._readable_children_identities == {}
    finally:
        if not drained:
            await registry.drain_children()
        await parent.aclose()
