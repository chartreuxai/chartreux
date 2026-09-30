from __future__ import annotations

import pytest
from textual.app import App, ComposeResult

from chartreux.app_server.protocol import AgentSummaryModel
from chartreux.cli.textual_ui.widgets.agent_bar import AgentBar
from chartreux.model_display import format_model_display_name


def test_shared_display_name_helper_formats_provider_and_missing_values() -> None:
    assert format_model_display_name("mistral", "large") == "mistral/large"
    assert format_model_display_name(None, None) == "unknown"


@pytest.mark.asyncio
async def test_agent_browser_row_uses_effective_provider_model() -> None:
    agent = AgentSummaryModel(
        agent_id="agent-1",
        profile="worker",
        availability="idle",
        base_model="base",
        active_provider="provider",
    )

    class Host(App[None]):
        def compose(self) -> ComposeResult:
            yield AgentBar()

    async with Host().run_test() as pilot:
        bar = pilot.app.query_one(AgentBar)
        bar.update_agents((agent,))
        bar.open_browser()
        assert "provider/base" not in str(bar.render())
        assert "provider/base" in bar._agent_details(agent)
