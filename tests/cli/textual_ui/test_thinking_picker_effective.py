from __future__ import annotations

import pytest
from textual.css.query import NoMatches
from textual.pilot import Pilot
from textual.widgets import OptionList

from chartreux.app_server.config import EffectiveThinkingView
from chartreux.cli.textual_ui.app import ChartreuxApp
from chartreux.cli.textual_ui.widgets.thinking_picker import ThinkingPickerApp
from chartreux.ui.widgets.no_markup_static import NoMarkupStatic
from tests.conftest import build_test_chartreux_app, wait_until


def _effective(
    *,
    requested: str = "medium",
    semantic: str | None = "medium",
    api_effort: str | None = "high",
    known: bool = True,
    matches: bool | None = False,
) -> EffectiveThinkingView:
    return EffectiveThinkingView.model_validate({
        "requested": requested,
        "semantic": semantic,
        "api_effort": api_effort,
        "known": known,
        "matches_requested": matches,
    })


async def _open_picker(
    app: ChartreuxApp, pilot: Pilot[None], effective: EffectiveThinkingView | None
) -> ThinkingPickerApp:
    app.app_server._state.config = app.config.model_copy(
        update={"effective_thinking": effective}
    )
    await app._show_thinking()
    picker = app.query_one(ThinkingPickerApp)
    await wait_until(pilot, lambda: picker.query_one(OptionList).has_focus)
    return picker


def _explanation(picker: ThinkingPickerApp) -> str | None:
    try:
        effect = picker.query_one("#thinkingpicker-effect", NoMarkupStatic)
    except NoMatches:
        return None
    return str(effect.content)


@pytest.mark.asyncio
async def test_picker_omits_the_explanation_when_effort_matches() -> None:
    app = build_test_chartreux_app()
    async with app.run_test() as pilot:
        await app._session_ready.wait()
        picker = await _open_picker(
            app, pilot, _effective(api_effort="medium", matches=True)
        )

        assert _explanation(picker) is None
        # The picker still marks the session's current level.
        prompts = [
            str(option.prompt) for option in picker.query_one(OptionList).options
        ]
        assert any("This session" in prompt for prompt in prompts)


@pytest.mark.asyncio
async def test_picker_explains_a_known_mismatch_from_the_projection() -> None:
    app = build_test_chartreux_app()
    async with app.run_test() as pilot:
        await app._session_ready.wait()
        picker = await _open_picker(app, pilot, _effective())

        explanation = _explanation(picker)
        assert explanation is not None
        assert "Selected medium" in explanation
        assert "API effort is high" in explanation
        assert "next request" in explanation


@pytest.mark.asyncio
async def test_picker_explains_an_unknown_estimate_as_unknown() -> None:
    app = build_test_chartreux_app()
    async with app.run_test() as pilot:
        await app._session_ready.wait()
        picker = await _open_picker(
            app,
            pilot,
            _effective(semantic=None, api_effort=None, known=False, matches=None),
        )

        explanation = _explanation(picker)
        assert explanation is not None
        assert "unknown" in explanation
        assert "API effort is high" not in explanation


@pytest.mark.asyncio
async def test_picker_explains_history_forced_thinking() -> None:
    app = build_test_chartreux_app()
    async with app.run_test() as pilot:
        await app._session_ready.wait()
        picker = await _open_picker(
            app,
            pilot,
            _effective(requested="off", semantic="medium", api_effort="medium"),
        )

        explanation = _explanation(picker)
        assert explanation is not None
        assert "Selected off" in explanation
        assert "retained thinking history keeps this model at medium" in explanation
