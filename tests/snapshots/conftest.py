from __future__ import annotations

import re
import time

import pytest
import pytest_textual_snapshot


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
def _pin_snapshot_colors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)


@pytest.fixture(autouse=True)
def _pin_banner_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "chartreux.cli.textual_ui.widgets.banner.banner.__version__", "0.0.0"
    )


@pytest.fixture(autouse=True)
def _pin_process_title(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "chartreux.cli.textual_ui.app.process_id_label", lambda: "[PID 00000]"
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
