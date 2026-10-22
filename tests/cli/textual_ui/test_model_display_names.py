from __future__ import annotations

import pytest
from textual.app import App, ComposeResult

from chartreux.app_server.protocol import AgentSummaryModel
from chartreux.cli.textual_ui.widgets.agent_bar import AgentBar
from chartreux.core.subagents import derive_agent_name
from chartreux.model_display import format_model_display_name


def test_shared_display_name_helper_formats_provider_and_missing_values() -> None:
    assert format_model_display_name("mistral", "large") == "mistral/large"
    assert format_model_display_name(None, None) == "unknown"


@pytest.mark.asyncio
async def test_agent_browser_row_uses_effective_provider_model() -> None:
    agent = AgentSummaryModel(
        agent_id="agent-1",
        identity="coder",
        availability="idle",
        base_model="base",
        active_provider="provider",
        name=derive_agent_name(
            agent_id="agent-1",
            identity="coder",
            identity_display_label=None,
            base_model="base",
            effective_model=None,
        ),
    )

    class Host(App[None]):
        def compose(self) -> ComposeResult:
            yield AgentBar()

    async with Host().run_test() as pilot:
        bar = pilot.app.query_one(AgentBar)
        bar.update_agents((agent,))
        bar.open_browser()
        # Let the first layout pass settle: _agent_details clips to the settled
        # content width, which is zero until the browser is arranged.
        await pilot.pause()
        assert "provider/base" not in str(bar.render())
        assert "provider/base" in bar._agent_details(agent)
