from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from chartreux.app_server.protocol import RootsReadResponse
from chartreux.cli.textual_ui.app import ChartreuxApp
from tests.conftest import build_test_chartreux_app


@pytest.mark.asyncio
@pytest.mark.parametrize("saved", [["/saved/root"], None])
async def test_status_lists_effective_and_saved_roots(
    saved: list[str] | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = build_test_chartreux_app()
    response = RootsReadResponse(
        revision="r1",
        project="/project/current",
        roots=[],
        effective_roots=["/effective/root"],
        saved_roots=saved,
        user_revision="u1" if saved is not None else None,
    )
    app_server = SimpleNamespace(
        resources=SimpleNamespace(
            runtime=SimpleNamespace(
                stats=SimpleNamespace(
                    session_cached_tokens=0,
                    last_turn_cached_tokens=0,
                    steps=1,
                    session_prompt_tokens=2,
                    session_completion_tokens=3,
                    session_total_llm_tokens=5,
                    last_turn_total_tokens=5,
                    session_cost=0.0,
                )
            ),
            config=SimpleNamespace(read_roots=AsyncMock(return_value=response)),
        )
    )
    monkeypatch.setattr(ChartreuxApp, "app_server", property(lambda _app: app_server))
    app._mount_and_scroll = AsyncMock()
    await app._show_status()
    message = cast(Any, app._mount_and_scroll).await_args.args[0]
    text = message._content
    assert "`/project/current`" in text
    assert "Effective roots" in text and "`/effective/root`" in text
    assert "Saved roots" in text
    if saved is None:
        assert "unavailable" in text
        assert "no saved roots" not in text.lower()
    else:
        assert "`/saved/root`" in text
