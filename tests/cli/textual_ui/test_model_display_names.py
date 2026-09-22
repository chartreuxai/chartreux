from __future__ import annotations

from chartreux.app_server.protocol import AgentSummaryModel
from chartreux.cli.textual_ui.widgets.agent_bar import AgentBar
from chartreux.model_display import format_model_display_name


def test_shared_display_name_helper_formats_provider_and_missing_values() -> None:
    assert format_model_display_name("mistral", "large") == "mistral/large"
    assert format_model_display_name(None, None) == "unknown"


def test_agent_browser_row_uses_effective_provider_model() -> None:
    agent = AgentSummaryModel(
        agent_id="agent-1",
        profile="worker",
        availability="idle",
        base_model="base",
        active_provider="provider",
    )
    bar = AgentBar()
    bar.update_agents((agent,))
    bar.open_browser()
    assert "provider/base" in str(bar.render())
