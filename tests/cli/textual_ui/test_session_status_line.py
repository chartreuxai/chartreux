from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from rich.cells import cell_len
from textual.app import App, ComposeResult

from chartreux.app_server.config import StatusLineConfigView
from chartreux.app_server.models import UsageTotals, UsageWindowSummary
from chartreux.app_server.protocol import SettingLeafWire, SettingsReadResponse
from chartreux.cli.textual_ui.screens.status_line_settings import (
    DESCRIPTIONS,
    StatusLineSettingsScreen,
)
from chartreux.cli.textual_ui.widgets.session_status_line import (
    SessionStatusLine,
    SessionStatusState,
    format_segment,
    format_status_line,
)
from chartreux.ui.context_display import format_context
from chartreux.ui.settings_service import SettingsService
from chartreux.ui.usage_display import format_usage_cost


@pytest.fixture
def state() -> SessionStatusState:
    return SessionStatusState(
        cwd="/home/test/workspace",
        home_directory=Path("/home/test"),
        pid=123,
        model_identity="Provider/model",
        context_tokens=135_000,
        auto_compact_threshold=400_000,
        branch="feature/topic",
        branch_status="branch",
    )


@pytest.mark.parametrize(
    ("segment", "expected"),
    [
        ("directory", "workspace"),
        ("pid", "pid 123"),
        ("model", "Provider/model"),
        ("context", "135k/400k (34%)"),
        ("git-branch", "feature/topic"),
        ("spend-today", "Today —"),
        ("spend-week", "Week —"),
        ("background-jobs", "Jobs 0"),
        ("spend-month", "Month —"),
    ],
)
def test_segments(state, segment, expected):
    assert format_segment(segment, state, StatusLineConfigView()) == expected
    assert "$0.00" not in expected


@pytest.mark.parametrize(
    ("segment", "expected"),
    [
        ("pid", "pid -"),
        ("model", "Model -"),
        ("context", "-/-"),
        ("git-branch", "Git -"),
        ("spend-today", "Today -"),
        ("spend-week", "Week -"),
        ("spend-month", "Month -"),
    ],
)
def test_ascii_unknown_segments(segment: str, expected: str) -> None:
    state = SessionStatusState(cwd="/work", ascii_chrome=True)
    assert format_segment(segment, state, StatusLineConfigView()) == expected


@pytest.mark.parametrize(
    ("cwd", "home", "expected"),
    [
        ("/home/test/workspace", Path("/home/test"), "~/workspace"),
        ("/home/test", Path("/home/test"), "~"),
        ("/home/testing/project", Path("/home/test"), "/home/testing/project"),
        ("/work/project", None, "/work/project"),
        ("/", Path("/home/test"), "/"),
    ],
)
def test_path_style(state, cwd, home, expected):
    config = StatusLineConfigView(directory_style="path")
    assert (
        format_segment(
            "directory", replace(state, cwd=cwd, home_directory=home), config
        )
        == expected
    )
    assert (
        format_segment("directory", replace(state, cwd="/"), StatusLineConfigView())
        == "/"
    )


@pytest.mark.parametrize(
    ("status", "branch", "expected"),
    [
        ("branch", "main", "main"),
        ("branch", None, "Git —"),
        ("detached", "stale", "Git detached"),
        ("not_repository", "stale", "Git not a repository"),
        ("unknown", "stale", "Git —"),
    ],
)
def test_branch_states(state, status, branch, expected):
    assert (
        format_segment(
            "git-branch",
            replace(state, branch_status=status, branch=branch),
            StatusLineConfigView(),
        )
        == expected
    )


@pytest.mark.parametrize("style", ["tokens", "tokens-percent"])
@pytest.mark.parametrize("tokens", [None, -1, 0, 135_000, 800_000])
@pytest.mark.parametrize("threshold", [None, 0, 400_000])
@pytest.mark.parametrize("compacting", [False, True])
def test_context_matches_shared_formatter(state, style, tokens, threshold, compacting):
    snapshot = replace(
        state,
        context_tokens=tokens,
        auto_compact_threshold=threshold,
        compacting=compacting,
    )
    assert format_segment(
        "context", snapshot, StatusLineConfigView(context_style=style)
    ) == format_context(tokens, threshold, style=style, compacting=compacting)


def test_last_recorded_and_unavailable_identities(state):
    snapshot = replace(state, last_recorded=True, pid=None, model_identity=None)
    config = StatusLineConfigView()
    assert format_segment("context", snapshot, config).endswith(" (last recorded)")
    assert format_segment("pid", snapshot, config) == "pid —"
    assert format_segment("model", snapshot, config) == "Model —"
    with pytest.raises(ValueError, match="Unknown status line segment"):
        format_segment("unsupported", state, config)


@pytest.mark.parametrize("separator", ["pipe", "space"])
def test_order_and_pid_first_then_optional_tail(state, separator):
    names = [
        "model",
        "context",
        "spend-today",
        "pid",
        "background-jobs",
        "git-branch",
        "directory",
    ]
    config = StatusLineConfigView(segments=names, separator=separator)
    sep = " | " if separator == "pipe" else " "

    def row(remaining):
        return sep.join(format_segment(name, state, config) for name in remaining)

    assert format_status_line(state, config, 200) == row(names)
    names.remove("pid")
    assert format_status_line(state, config, cell_len(row(names))) == row(names)
    for name in ["git-branch", "background-jobs", "spend-today", "model"]:
        names.remove(name)
        assert format_status_line(state, config, cell_len(row(names))) == row(names)


@pytest.mark.parametrize("ascii_chrome", [False, True])
@pytest.mark.parametrize("separator", ["space", "pipe"])
def test_cell_fitting_extreme_widths_and_controls(state, ascii_chrome, separator):
    snapshot = replace(
        state,
        cwd="/" + "界e\u0301" * 100,
        model_identity="モデル" * 100,
        branch="枝" * 100,
        ascii_chrome=ascii_chrome,
    )
    config = StatusLineConfigView(
        segments=["model", "directory", "git-branch", "context", "pid"],
        separator=separator,
    )
    for width in range(121):
        row = format_status_line(snapshot, config, width)
        assert cell_len(row) <= width
        assert "\n" not in row
        if 2 <= width <= 4:
            assert row == ("." * min(width, 3) if ascii_chrome else "…")
    clean = replace(state, model_identity="[red]model\nname\t\x1b", branch="x\ry")
    assert format_segment("model", clean, config) == "[red]model name  "
    assert format_segment("git-branch", clean, config) == "x y"


def test_directory_shortens_before_context_and_order_survives(state):
    config = StatusLineConfigView(segments=["context", "directory"])
    row = format_status_line(state, config, 22)
    assert row == "135k/400k (34%) | wor…"
    assert format_status_line(state, config, 0) == ""
    assert format_status_line(state, config, 1) == "…"


def test_formatting_does_not_fetch_home(state):
    with patch.object(Path, "home", side_effect=AssertionError("render fetched home")):
        assert format_status_line(
            state, StatusLineConfigView(directory_style="path"), 100
        ).startswith("~/workspace")


def _usage_summary(totals: UsageTotals) -> UsageWindowSummary:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2024, 1, 2, tzinfo=UTC)
    return UsageWindowSummary(
        **totals.model_dump(),
        start_local=start,
        end_local=end,
        start_utc=start,
        end_utc=end,
        timezone="UTC",
    )


@pytest.mark.parametrize(
    ("totals", "expected"),
    [
        (None, "—"),
        (UsageTotals(state="loading"), "—"),
        (UsageTotals(state="unavailable"), "—"),
        (UsageTotals(), "$0.00"),
        (UsageTotals(requests=1, has_known_cost=True), "$0.00"),
        (UsageTotals(requests=1, has_unknown_cost=True), "Unknown"),
        (UsageTotals(requests=1), "Unknown"),
        (UsageTotals(requests=1, known_cost_usd=12.34, has_known_cost=True), "$12.34"),
        (
            UsageTotals(
                requests=2,
                known_cost_usd=12.34,
                has_known_cost=True,
                has_unknown_cost=True,
            ),
            "$12.34+",
        ),
        (UsageTotals(requests=2, has_known_cost=True, has_unknown_cost=True), "$0.00+"),
        (
            UsageTotals(requests=1, has_known_cost=True, has_unknown_tokens=True),
            "$0.00",
        ),
    ],
)
@pytest.mark.parametrize("ascii_chrome", [False, True])
def test_spend_snapshot_states(state, totals, expected, ascii_chrome):
    summary = _usage_summary(totals) if totals is not None else None
    before = summary.model_dump() if summary is not None else None
    snapshot = replace(
        state,
        usage_day=summary,
        usage_week=summary,
        usage_month=summary,
        ascii_chrome=ascii_chrome,
    )
    assert format_usage_cost(summary) == expected
    displayed = expected.replace("—", "-") if ascii_chrome else expected
    for segment, label in (
        ("spend-today", "Today"),
        ("spend-week", "Week"),
        ("spend-month", "Month"),
    ):
        assert format_segment(segment, snapshot, StatusLineConfigView()) == (
            f"{label} {displayed}"
        )
    assert (summary.model_dump() if summary is not None else None) == before


@pytest.mark.parametrize("ascii_chrome", [False, True])
@pytest.mark.parametrize("cost", [0.0, 12.34])
def test_degraded_spend_is_not_presented_as_complete(state, ascii_chrome, cost):
    summary = _usage_summary(
        UsageTotals(known_cost_usd=cost, has_known_cost=cost > 0)
    ).model_copy(update={"degraded": True})
    snapshot = replace(
        state,
        usage_day=summary,
        usage_week=summary,
        usage_month=summary,
        ascii_chrome=ascii_chrome,
    )
    assert format_usage_cost(summary) == "—"
    for segment, label in (
        ("spend-today", "Today"),
        ("spend-week", "Week"),
        ("spend-month", "Month"),
    ):
        assert format_segment(segment, snapshot, StatusLineConfigView()) == (
            f"{label} {'-' if ascii_chrome else '—'}"
        )


def test_supplied_spend_snapshots_render_without_io_or_accumulation(state):
    day = _usage_summary(
        UsageTotals(requests=1, known_cost_usd=1.23, has_known_cost=True)
    )
    week = day.model_copy(update={"known_cost_usd": 12.34, "has_unknown_cost": True})
    month = day.model_copy(update={"has_known_cost": False, "has_unknown_cost": True})
    snapshot = replace(state, usage_day=day, usage_week=week, usage_month=month)
    config = StatusLineConfigView(
        segments=["directory", "context", "spend-today", "spend-week", "spend-month"]
    )
    expected = (
        "workspace | 135k/400k (34%) | Today $1.23 | Week $12.34+ | Month Unknown"
    )
    for _ in range(3):
        assert format_status_line(snapshot, config, 200) == expected
    # Optional costs disappear whole; workspace and context keep their space.
    required = "workspace | 135k/400k (34%)"
    assert format_status_line(snapshot, config, cell_len(required)) == required
    for width in range(80):
        assert cell_len(format_status_line(snapshot, config, width)) <= width


@pytest.mark.parametrize("segment", ["spend-today", "spend-week", "spend-month"])
def test_spend_settings_help_describes_live_global_cost(segment):
    description = DESCRIPTIONS[segment]
    assert "recorded USD spend across all projects" in description
    assert "Current project filter never changes this scope" in description
    assert "calendar" in description
    assert (
        "+" in description and "Unknown" in description and "unavailable" in description
    )
    assert "Requires" not in description and "currently" not in description


@pytest.mark.parametrize("count", [0, 2])
@pytest.mark.parametrize("ascii_chrome", [False, True])
def test_jobs_opt_in(state, count, ascii_chrome):
    snapshot = replace(
        state, active_background_job_count=count, ascii_chrome=ascii_chrome
    )
    config = StatusLineConfigView(segments=["directory", "background-jobs", "context"])
    assert f"Jobs {count}" in format_status_line(snapshot, config, 100)
    assert "Jobs" not in format_status_line(snapshot, StatusLineConfigView(), 100)
    assert "root session" in DESCRIPTIONS["background-jobs"]
    assert "children" in DESCRIPTIONS["background-jobs"]


def test_settings_preview_uses_deterministic_supplied_costs():
    config = StatusLineConfigView(
        segments=["directory", "context", "spend-today", "spend-week", "spend-month"]
    )
    settings = SettingsReadResponse(
        fields=[
            SettingLeafWire(
                path=f"status_line.{key}",
                effective_value=value,
                origin="default",
                saved_explicit=False,
            )
            for key, value in config.model_dump(mode="json", by_alias=False).items()
        ]
    )
    service = Mock(spec=SettingsService)
    first = StatusLineSettingsScreen(service, settings)
    second = StatusLineSettingsScreen(service, settings)
    assert first._example == second._example
    assert first._example.home_directory == Path("/home/example")
    rendered = format_status_line(first._example, config, 200)
    assert "Today $1.23" in rendered
    assert "Week $12.34" in rendered
    assert "Month $45.67" in rendered
    for width in (20, 40, 80, 120):
        assert cell_len(format_status_line(first._example, config, width)) <= width
    assert service.mock_calls == []


class _StatusApp(App[None]):
    def __init__(self, state: SessionStatusState):
        super().__init__()
        self.line = SessionStatusLine(state, id="status")

    def compose(self) -> ComposeResult:
        yield self.line


@pytest.mark.asyncio
@pytest.mark.parametrize("theme", ["textual-light", "textual-dark"])
async def test_widget_feedback_resize_and_local_css(state, theme):
    app = _StatusApp(state)
    app.theme = theme
    async with app.run_test(size=(80, 24)) as pilot:
        line = app.line
        original = line.render().plain
        assert line.size.height == 1
        assert line.styles.text_wrap == "nowrap"
        assert line.render().no_wrap
        line.set_feedback("Press Ctrl+C again to quit")
        line.set_state(replace(state, cwd="/new/work"))
        line.set_config(StatusLineConfigView(separator="space"))
        await pilot.pause()
        assert line.render().plain == "Press Ctrl+C again to quit"
        await pilot.resize_terminal(8, 24)
        await pilot.pause()
        assert line.size.height == 1
        assert cell_len(line.render().plain) <= 8
        assert line.render().plain.endswith("…")
        line.clear_feedback()
        await pilot.resize_terminal(80, 24)
        await pilot.pause()
        assert line.render().plain == "work pid 123 135k/400k (34%)"
        assert line.render().plain != original
        summary = _usage_summary(
            UsageTotals(requests=1, known_cost_usd=12.34, has_known_cost=True)
        )
        line.set_config(
            StatusLineConfigView(segments=["directory", "context", "spend-today"])
        )
        line.set_feedback("Waiting for confirmation")
        line.set_state(replace(state, cwd="/new/work", usage_day=summary))
        await pilot.pause()
        assert line.render().plain == "Waiting for confirmation"
        line.clear_feedback()
        await pilot.pause()
        assert line.render().plain == "work | 135k/400k (34%) | Today $12.34"
        line.set_config(StatusLineConfigView(separator="space"))
        await pilot.resize_terminal(2, 24)
        await pilot.pause()
        assert line.size.height == 1
        assert cell_len(line.render_line(0).text) <= 2
        await pilot.resize_terminal(80, 24)
        await pilot.pause()
        assert line.render().plain == "work pid 123 135k/400k (34%)"
