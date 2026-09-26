from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static

from chartreux.app_server.models import (
    CompletedEffectState,
    EffectResultDisplay,
    FailedEffectState,
    SkippedEffectState,
)
from chartreux.cli.textual_ui.widgets.status_message import IndicatorState
from chartreux.cli.textual_ui.widgets.tool_grouping import (
    ToolGroupExpansionState,
    ToolGroupKey,
)
from chartreux.cli.textual_ui.widgets.tools import ToolGroup
from chartreux.utils.tool_presentation import ToolEffectKind


class _ToolGroupApp(App[None]):
    CSS_PATH = Path(__file__).parents[3] / "chartreux/cli/textual_ui/app.tcss"

    def __init__(self, group: ToolGroup) -> None:
        super().__init__()
        self.group = group

    def compose(self) -> ComposeResult:
        yield self.group


@pytest.mark.asyncio
async def test_group_header_category_and_collapsed_body_persist() -> None:
    state = ToolGroupExpansionState()
    key = ToolGroupKey("first")
    group = ToolGroup(key=key, expansion_state=state)
    group.add_call_kind(ToolEffectKind.FILE_READ)
    group.add_content_child(Static("tool call"))
    app = _ToolGroupApp(group)

    async with app.run_test() as pilot:
        await pilot.pause()
        assert group.header.get_content() == "Reading files"
        assert group.is_collapsed
        assert group.content_container.display is False

        await pilot.click(".tool-group-header")
        await pilot.pause()
        assert not group.is_collapsed
        assert group.content_container.display is True
        assert not state.is_collapsed(key)

    restored = ToolGroup(key=key, expansion_state=state)
    assert not restored.is_collapsed


@pytest.mark.asyncio
async def test_group_header_skips_unchanged_summary_updates() -> None:
    group = ToolGroup()
    group.add_call_kind(ToolEffectKind.FILE_READ)
    app = _ToolGroupApp(group)

    async with app.run_test() as pilot:
        await pilot.pause()
        text_widget = group.header._text_widget
        assert text_widget is not None

        with patch.object(text_widget, "update", wraps=text_widget.update) as update:
            group.add_call_kind(ToolEffectKind.FILE_READ)
            group.mark_reasoning()
            group.mark_reasoning()
            group.resume()
            assert update.call_count == 1
            assert update.call_args.args == ("Reading files, thinking",)

            group.add_call_kind(ToolEffectKind.FILE_WRITE)

        assert update.call_count == 2
        assert update.call_args.args == ("Reading files, writing files, thinking",)


@pytest.mark.asyncio
async def test_running_indicator_settles_to_colored_disclosure() -> None:
    group = ToolGroup()
    group.add_call_kind(ToolEffectKind.SHELL)
    app = _ToolGroupApp(group)

    async with app.run_test() as pilot:
        await pilot.pause()
        indicator = group.header._indicator_widget
        assert indicator is not None
        assert str(indicator.render()) in {"■", "□"}
        assert group.header._spinner_timer is not None

        group.settle_indicator(IndicatorState.ERROR)
        group.finalize()
        await pilot.pause()
        assert str(indicator.render()) == "⏵"
        assert "error" in indicator.classes
        assert group.header._spinner_timer is None


@pytest.mark.asyncio
async def test_latest_timeline_call_controls_group_indicator() -> None:
    group = ToolGroup()
    app = _ToolGroupApp(group)

    async with app.run_test() as pilot:
        await pilot.pause()
        group.record_effect(
            1,
            FailedEffectState(
                error={"message": "failed"},  # type: ignore[arg-type]
                display=EffectResultDisplay(success=False, message="failed"),
            ),
        )
        group.record_effect(
            2,
            CompletedEffectState(
                display=EffectResultDisplay(success=True, message="done")
            ),
        )
        group.finalize()
        await pilot.pause()
        indicator = group.header._indicator_widget
        assert indicator is not None
        assert "success" in indicator.classes
        assert "error" not in indicator.classes


def test_effect_states_use_policy_indicator_mapping() -> None:
    group = ToolGroup()
    group.settle_effect(
        FailedEffectState(
            error={"message": "failed"},  # type: ignore[arg-type]
            display=EffectResultDisplay(success=False, message="failed"),
        )
    )
    assert group.header._last_state is IndicatorState.ERROR

    group.settle_effect(
        SkippedEffectState(
            reason="not needed",
            display=EffectResultDisplay(success=True, message="skipped"),
        )
    )
    assert group.header._last_state is IndicatorState.MUTED

    group.settle_effect(
        CompletedEffectState(display=EffectResultDisplay(success=True, message="done"))
    )
    assert group.header._last_state is IndicatorState.SUCCESS


@pytest.mark.asyncio
async def test_spinner_timer_stops_when_group_is_removed() -> None:
    group = ToolGroup()
    app = _ToolGroupApp(group)

    async with app.run_test() as pilot:
        await pilot.pause()
        assert group.header._spinner_timer is not None
        await group.remove()
        await pilot.pause()
        assert group.header._spinner_timer is None


@pytest.mark.asyncio
async def test_spinner_timer_stops_when_app_shuts_down() -> None:
    group = ToolGroup()
    app = _ToolGroupApp(group)

    async with app.run_test() as pilot:
        await pilot.pause()
        assert group.header._spinner_timer is not None

    assert group.header._spinner_timer is None
