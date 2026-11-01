from __future__ import annotations

import pytest

from chartreux.app_server.events import HistoryEntryAdded
from chartreux.app_server.models import (
    PublicEntryGenerationStatus,
    PublicReasoningEntry,
)
from chartreux.cli.textual_ui.widgets.thinking_picker import ThinkingPickerApp
from chartreux.core.config import ChartreuxConfigSchema, ModelConfig, ProviderConfig
from chartreux.core.llm_models import Backend, LLMMessage, Role
from chartreux.ui.widgets.no_markup_static import NoMarkupStatic
from tests.conftest import (
    build_test_agent_loop,
    build_test_chartreux_app,
    build_test_vibe_config,
    wait_until,
)


def _anthropic_config() -> ChartreuxConfigSchema:
    provider = ProviderConfig(
        name="claude",
        api_base="https://claude.test/v1",
        api_key_env_var="ANTHROPIC_API_KEY",
        backend=Backend.GENERIC,
        api_style="anthropic",
    )
    return build_test_vibe_config(
        active_model="claude",
        providers=[provider],
        models=[
            ModelConfig(
                name="claude-wire", provider="claude", alias="claude", thinking="off"
            )
        ],
    )


def _reasoning_entry(session_id: str) -> PublicReasoningEntry:
    return PublicReasoningEntry(
        id="reasoning-1",
        session_id=session_id,
        created_at=0,
        updated_at=0,
        generation_status=PublicEntryGenerationStatus.COMPLETED,
        text="considering the question",
    )


@pytest.mark.asyncio
async def test_appended_thinking_history_refreshes_the_estimate() -> None:
    """A completed reasoning entry re-projects the next-request annotation."""
    config = _anthropic_config()
    agent_loop = build_test_agent_loop(config=config)
    app = build_test_chartreux_app(config=config, agent_loop=agent_loop)
    async with app.run_test() as pilot:
        await app._session_ready.wait()
        await pilot.pause()
        banner = app._banner
        assert banner is not None
        before = str(banner.query_one("#banner-model", NoMarkupStatic).content)
        assert before.endswith("[off]")
        assert "API:" not in before

        agent_loop.messages.append(
            LLMMessage(
                role=Role.assistant,
                content="Answered",
                reasoning_payloads=[{"type": "thinking", "thinking": "considering"}],
            )
        )
        await app._handle_turn_event(
            HistoryEntryAdded(entry=_reasoning_entry(app.app_server.session_id))
        )
        await pilot.pause()

        after = str(banner.query_one("#banner-model", NoMarkupStatic).content)
        assert after.endswith("[off; API: medium]")

        # The picker reads the same refreshed projection.
        await app._show_thinking()
        picker = app.query_one(ThinkingPickerApp)
        await wait_until(pilot, lambda: bool(picker.query("#thinkingpicker-effect")))
        explanation = str(
            picker.query_one("#thinkingpicker-effect", NoMarkupStatic).content
        )
        assert "retained thinking history keeps this model at medium" in explanation
