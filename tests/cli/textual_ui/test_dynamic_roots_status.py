from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from chartreux.app_server.protocol import AgentSummaryModel, RootsReadResponse
from chartreux.cli.textual_ui.app import ChartreuxApp, _markdown_literal
from tests.conftest import build_test_chartreux_app, build_test_vibe_config
from tests.stubs.app_config import build_test_app_config


def _stub_status_app_server(monkeypatch: pytest.MonkeyPatch) -> None:
    """Back ``_show_status`` with fixed runtime/config stubs (no app server)."""
    response = RootsReadResponse(
        revision="r1",
        project="/project/current",
        roots=[],
        effective_roots=["/effective/root"],
        saved_roots=None,
        user_revision=None,
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
                ),
                refresh=AsyncMock(),
            ),
            config=SimpleNamespace(
                read_roots=AsyncMock(return_value=response),
                current=build_test_app_config(),
            ),
        )
    )
    monkeypatch.setattr(ChartreuxApp, "app_server", property(lambda _app: app_server))


def _agent(
    agent_id: str, *, name: str, uncollected: tuple[str, ...] = ()
) -> AgentSummaryModel:
    return AgentSummaryModel(
        agent_id=agent_id,
        name=name,
        availability="idle",
        uncollected_run_ids=list(uncollected),
    )


async def _render_status(app: ChartreuxApp) -> str:
    app._mount_and_scroll = AsyncMock()
    await app._show_status()
    message = cast(Any, app._mount_and_scroll).await_args.args[0]
    return cast(str, message._content)


def _background_results_section(text: str) -> str:
    assert "## Background Agent Results" in text
    return text.split("## Background Agent Results", 1)[1]


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
                ),
                refresh=AsyncMock(),
            ),
            config=SimpleNamespace(
                read_roots=AsyncMock(return_value=response),
                current=build_test_app_config(),
            ),
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


@pytest.mark.asyncio
async def test_status_shows_requested_semantic_and_api_effort() -> None:
    """The three-value row comes from the server-owned next-request estimate."""
    from chartreux.core.config import ModelConfig

    config = build_test_vibe_config(
        active_model="large",
        models=[
            ModelConfig(
                name="mistral-large-4",
                provider="mistral",
                alias="large",
                thinking="low",
            )
        ],
    )
    app = build_test_chartreux_app(config=config)
    async with app.run_test() as pilot:
        await app._session_ready.wait()
        await pilot.pause()
        app._mount_and_scroll = AsyncMock()
        await app._show_status()
        message = cast(Any, app._mount_and_scroll).await_args.args[0]
        text = message._content

        assert "- **Requested**: low" in text
        assert "- **Semantic**: low" in text
        # Mistral Large 4 declares low/medium/high reasoning efforts.
        assert "- **API effort**: low" in text
        assert "next request" in text


@pytest.mark.asyncio
async def test_status_lists_uncollected_results_from_the_last_agents_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_status_app_server(monkeypatch)
    app = build_test_chartreux_app()
    app._agent_summaries = [
        _agent("agent-2", name="researcher-glm-5-3-2", uncollected=("run-b", "run-a")),
        _agent("agent-1", name="coder-mistral-4-1"),
    ]

    text = await _render_status(app)

    section = _background_results_section(text)
    # Freshness is scoped to the snapshot; check_agents as-of-now is never claimed.
    assert "as of the last agents update" in section.lower()
    assert "visible agent inventory" in section.lower()
    assert "check_agents" not in section
    assert "- **researcher-glm-5-3-2** (agent-2): run-b, run-a" in section
    # An agent whose results were all collected is not part of the list.
    assert "coder-mistral-4-1" not in section
    assert "No visible uncollected results." not in section
    # The statistics, thinking and authorized-roots output is unchanged.
    assert "## Agent Statistics" in text
    assert "## Thinking (next request)" in text
    assert "## Authorized Roots" in text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "branch",
    [
        pytest.param("empty", id="empty-snapshot"),
        pytest.param("no-run-ids", id="snapshot-without-uncollected-run-ids"),
    ],
)
async def test_status_reports_no_visible_uncollected_results(
    branch: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_status_app_server(monkeypatch)
    app = build_test_chartreux_app()
    app._agent_summaries = (
        [] if branch == "empty" else [_agent("agent-1", name="coder-mistral-4-1")]
    )

    text = await _render_status(app)

    section = _background_results_section(text)
    content = [line for line in section.splitlines() if line.strip()]
    # The lead sentence is the only italic line; the body is the exact
    # empty-state literal and nothing else.
    body = [line for line in content if not line.startswith("_")]
    assert body == ["No visible uncollected results."]


@pytest.mark.asyncio
async def test_status_clears_uncollected_results_after_a_later_agents_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_status_app_server(monkeypatch)
    app = build_test_chartreux_app()
    app._agent_summaries = [
        _agent("agent-1", name="coder-mistral-4-1", uncollected=("run-a",))
    ]

    before = await _render_status(app)
    assert "**coder-mistral-4-1** (agent-1): run-a" in _background_results_section(
        before
    )
    assert "No visible uncollected results." not in before

    # A later agents update after collection (or expiry) drops the IDs; the
    # badge is gone because /status reads the current snapshot, not history.
    app._agent_summaries = [_agent("agent-1", name="coder-mistral-4-1", uncollected=())]

    after = await _render_status(app)
    section = _background_results_section(after)
    assert "No visible uncollected results." in section
    assert "run-a" not in section
    assert "coder-mistral-4-1" not in section


@pytest.mark.asyncio
async def test_status_renders_agent_names_handles_and_run_ids_literally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wire-supplied names and IDs never become live inline Markdown."""
    _stub_status_app_server(monkeypatch)
    name = "coder[*bold*](x)<b>`run`"
    agent_id = "agent`1_*x*"
    run_id = "run`a_b*"
    app = build_test_chartreux_app()
    app._agent_summaries = [_agent(agent_id, name=name, uncollected=(run_id,))]

    text = await _render_status(app)

    section = _background_results_section(text)
    assert _markdown_literal(name) in section
    assert f"({_markdown_literal(agent_id)})" in section
    assert _markdown_literal(run_id) in section
    assert "[*bold*]" not in section
    assert "`run`" not in section
