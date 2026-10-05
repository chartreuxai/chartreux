from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from datetime import UTC, datetime
import re
import time

import pytest
import pytest_textual_snapshot

from chartreux.cli.textual_ui.widgets import message_header, session_status_line
from chartreux.core.llm_models import use_posting_clock
from chartreux.core.timing import use_elapsed_clock

SNAPSHOT_NOW = datetime(2026, 6, 15, 12, 34, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _normalize_svg_whitespace(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep generated SVG snapshots free of trailing whitespace."""
    normalize = pytest_textual_snapshot.normalize_svg
    monkeypatch.setattr(
        pytest_textual_snapshot,
        "normalize_svg",
        lambda svg: re.sub(r"[ \t]+(?=\r?$)", "", normalize(svg), flags=re.MULTILINE),
    )


@pytest.fixture(autouse=True)
def _pin_timezone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TZ", "UTC")
    time.tzset()


@pytest.fixture(autouse=True)
def _pin_clocks(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Pin canonical posting and renderer today before any app tasks are created."""
    monkeypatch.setattr(message_header, "local_now", lambda: SNAPSHOT_NOW)
    # A fixed elapsed helper makes backend-generated totals measured zero;
    # timing showcases supply explicit nonzero durations. Never pin perf_counter.
    with use_posting_clock(lambda: SNAPSHOT_NOW), use_elapsed_clock(lambda: 0.0):
        yield


@pytest.fixture(autouse=True)
def _pin_snapshot_colors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)


@pytest.fixture(autouse=True)
def _pin_banner_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "chartreux.cli.textual_ui.widgets.banner.banner.__version__", "0.0.0"
    )


@pytest.fixture(autouse=True)
def _pin_process_title(monkeypatch: pytest.MonkeyPatch) -> None:
    # PID now belongs to the supplied status state, not an app label widget.
    format_segment = session_status_line.format_segment
    monkeypatch.setattr(
        session_status_line,
        "format_segment",
        lambda segment, state, config: format_segment(
            segment, replace(state, pid=0), config
        ),
    )


@pytest.fixture(autouse=True)
def _pin_spinner_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop spinners ticking, so a captured frame does not depend on timing.

    Every SpinnerMixin widget advances its frame on a 0.1s interval, so how many
    ticks land before the screenshot varies with machine load. Widgets still
    render their first frame; they just stop moving.
    """
    monkeypatch.setattr(
        "chartreux.cli.textual_ui.widgets.spinner.SpinnerMixin.start_spinner_timer",
        lambda self: None,
    )
    monkeypatch.setattr(
        "chartreux.cli.textual_ui.widgets.spinner_text.SpinnerText._advance",
        lambda self: None,
    )
