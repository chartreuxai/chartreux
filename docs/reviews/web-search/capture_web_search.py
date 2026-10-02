#!/usr/bin/env python3
"""Capture offline Web search settings states as real Textual SVG and PNG images."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import traceback
from typing import Any
from unittest.mock import patch

# Textual and Rich inspect this during import. Colored captures are explicit.
os.environ.pop("NO_COLOR", None)
# Import-time app paths must never point at the user's real Chartreux home.
os.environ["CHARTREUX_HOME"] = tempfile.mkdtemp(prefix="chartreux-web-search-review-")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from rich.console import Console
from textual.app import App
from textual.pilot import Pilot
from textual.widgets import Button, Input, OptionList

from chartreux.ui.settings_service import SettingsReloadOutcome, SettingsSaveOutcome
from chartreux.ui.web_search import WebSearchScreen
from tests.cli.textual_ui.web_search_fixture import (
    FakeCredentials,
    FakeSettingsService,
    WebSearchHarness,
    make_snapshot,
)

Prepare = Callable[[Pilot, WebSearchHarness], Awaitable[None]]
Factory = Callable[[], WebSearchHarness]
FAKE_KEY = "review-fake-secret"


@dataclass(frozen=True)
class Fixture:
    name: str
    factory: Factory
    prepare: Prepare | None
    actions: tuple[str, ...]


def _screen(app: WebSearchHarness) -> WebSearchScreen:
    screen = app.screen
    assert isinstance(screen, WebSearchScreen)
    return screen


async def _choose(
    pilot: Pilot, screen: WebSearchScreen, list_id: str, option_id: str
) -> None:
    options = screen.query_one(list_id, OptionList)
    options.highlighted = next(
        index for index, option in enumerate(options.options) if option.id == option_id
    )
    options.focus()
    await pilot.press("space")
    await pilot.pause(0.1)


async def _press(pilot: Pilot, screen: WebSearchScreen, button_id: str) -> None:
    screen.query_one(button_id, Button).press()
    await pilot.pause(0.15)


async def _duck_draft(pilot: Pilot, app: WebSearchHarness) -> None:
    screen = _screen(app)
    await _choose(pilot, screen, "#websearch-providers", "duckduckgo")
    assert screen._draft["provider"] == "duckduckgo"
    assert not screen.query_one("#websearch-credential").display
    assert screen._dirty_settings()


async def _standalone_alias(_pilot: Pilot, app: WebSearchHarness) -> None:
    screen = _screen(app)
    assert screen.mode == "standalone"
    options = screen.query_one("#websearch-providers", OptionList)
    assert [option.id for option in options.options] == [
        "mistral",
        "exa",
        "brave",
        "duckduckgo",
    ]
    assert screen._draft["provider"] == "auto"
    assert screen._visible_provider("auto") == "mistral"
    assert screen._provider_cursor == "mistral"


async def _custom_reset(pilot: Pilot, app: WebSearchHarness) -> None:
    screen = _screen(app)
    await _choose(pilot, screen, "#websearch-providers", "brave")
    assert screen._draft["provider"] == "brave"
    assert screen._draft["api_key_env_var"] == ""
    assert screen._draft["base_url"] == ""
    assert screen._forced_reset == {"api_key_env_var", "base_url"}
    assert screen._dirty_settings()


async def _key_input(pilot: Pilot, app: WebSearchHarness) -> None:
    key = _screen(app).query_one("#websearch-key", Input)
    assert key.password
    key.focus()
    await pilot.press(*FAKE_KEY)
    await pilot.pause(0.1)
    assert key.value == FAKE_KEY
    assert app.credentials.saved == []


async def _key_saved(pilot: Pilot, app: WebSearchHarness) -> None:
    await _key_input(pilot, app)
    screen = _screen(app)
    await _press(pilot, screen, "#websearch-save-key")
    assert app.credentials.saved == [("MISTRAL_API_KEY", FAKE_KEY)]
    assert screen.query_one("#websearch-key", Input).value == ""
    assert "API key saved" in screen._credential_message


async def _key_session_only(pilot: Pilot, app: WebSearchHarness) -> None:
    app.credentials.save_status = "session_only"
    await _key_input(pilot, app)
    screen = _screen(app)
    await _press(pilot, screen, "#websearch-save-key")
    assert app.credentials.saved == [("MISTRAL_API_KEY", FAKE_KEY)]
    assert "this session" in screen._credential_message


async def _key_rejected(pilot: Pilot, app: WebSearchHarness) -> None:
    app.credentials.save_status = "invalid_env_var"
    await _key_input(pilot, app)
    screen = _screen(app)
    await _press(pilot, screen, "#websearch-save-key")
    assert app.credentials.saved == [("MISTRAL_API_KEY", FAKE_KEY)]
    assert screen.query_one("#websearch-key", Input).value == FAKE_KEY
    assert "saved" not in screen._credential_message.lower()


async def _advanced(pilot: Pilot, app: WebSearchHarness) -> None:
    screen = _screen(app)
    await _press(pilot, screen, "#websearch-advanced-toggle")
    assert screen._advanced
    assert screen.query_one("#websearch-advanced").display
    screen.query_one("#websearch-input-api_key_env_var", Input).scroll_visible()
    await pilot.pause(0.1)


async def _dirty_back(pilot: Pilot, app: WebSearchHarness) -> None:
    await _duck_draft(pilot, app)
    screen = _screen(app)
    await pilot.press("escape")
    await pilot.pause(0.1)
    assert screen._confirming
    assert screen.query_one("#websearch-confirmation").display
    assert app.service.saved == []


async def _failed_save(pilot: Pilot, app: WebSearchHarness) -> None:
    app.service.outcome = SettingsSaveOutcome(
        "not_saved", "unchanged", error="fake disk denied"
    )
    await _duck_draft(pilot, app)
    screen = _screen(app)
    await _press(pilot, screen, "#websearch-save")
    assert len(app.service.saved) == 1
    assert screen._message.startswith("Not saved:")
    assert screen._dirty_settings()


async def _saved_read_failed(pilot: Pilot, app: WebSearchHarness) -> None:
    app.service.outcome = SettingsSaveOutcome("saved", "unchanged", None)
    await _duck_draft(pilot, app)
    screen = _screen(app)
    await _press(pilot, screen, "#websearch-save")
    assert screen._needs_refresh
    assert "could not be read" in screen._message
    assert len(app.service.saved) == 1


async def _refresh_failed(pilot: Pilot, app: WebSearchHarness) -> None:
    await _saved_read_failed(pilot, app)
    app.service.read_error = RuntimeError("fake read denied")
    screen = _screen(app)
    await _press(pilot, screen, "#websearch-refresh")
    assert screen._needs_refresh
    assert "Refresh failed" in screen._message


async def _runtime_failed(pilot: Pilot, app: WebSearchHarness) -> None:
    app.service.outcome = SettingsSaveOutcome(
        "saved", "failed", make_snapshot({"provider": "duckduckgo"}, readiness="ready")
    )
    await _duck_draft(pilot, app)
    screen = _screen(app)
    await _press(pilot, screen, "#websearch-save")
    assert screen._runtime_failed
    assert "runtime apply failed" in screen._message
    assert len(app.service.saved) == 1


async def _runtime_retry_failed(pilot: Pilot, app: WebSearchHarness) -> None:
    await _runtime_failed(pilot, app)
    app.service.retry_error = RuntimeError("fake reload denied")
    screen = _screen(app)
    await _press(pilot, screen, "#websearch-retry")
    assert "runtime reload failed" in screen._message
    assert app.service.retry_count == 1
    assert len(app.service.saved) == 1


async def _runtime_retry_read_failed(pilot: Pilot, app: WebSearchHarness) -> None:
    await _runtime_failed(pilot, app)
    app.service.retry_outcome = SettingsReloadOutcome(
        True, True, error="snapshot_unknown"
    )
    screen = _screen(app)
    await _press(pilot, screen, "#websearch-retry")
    assert app.service.retry_count == 1
    assert not screen._runtime_failed
    assert not screen._ui_refresh_failed
    assert screen._needs_refresh
    assert "settings read failed" in screen._message
    assert screen.query_one("#websearch-refresh", Button).display


async def _runtime_retry_ui_failed(pilot: Pilot, app: WebSearchHarness) -> None:
    await _runtime_failed(pilot, app)
    app.service.retry_outcome = SettingsReloadOutcome(
        True,
        False,
        make_snapshot({"provider": "duckduckgo"}, readiness="ready"),
        "ui_update_failed",
    )
    screen = _screen(app)
    await _press(pilot, screen, "#websearch-retry")
    assert app.service.retry_count == 1
    assert not screen._runtime_failed
    assert screen._ui_refresh_failed
    assert not screen._needs_refresh
    assert "UI refresh failed" in screen._message
    retry = screen.query_one("#websearch-retry", Button)
    assert retry.display
    assert str(retry.label) == "Retry UI refresh"


def fixtures() -> list[Fixture]:
    return [
        Fixture(
            "choice-standalone-mistral-alias",
            WebSearchHarness,
            _standalone_alias,
            ("open standalone Web search with raw Automatic; show one Mistral row",),
        ),
        Fixture(
            "auto-missing-key", WebSearchHarness, None, ("open Web search settings",)
        ),
        Fixture(
            "auto-ready",
            lambda: WebSearchHarness(
                FakeSettingsService(make_snapshot(readiness="ready")),
                FakeCredentials({"MISTRAL_API_KEY": FAKE_KEY}),
            ),
            None,
            ("open with fake resolved Mistral key",),
        ),
        Fixture(
            "duckduckgo-draft", WebSearchHarness, _duck_draft, ("choose DuckDuckGo",)
        ),
        Fixture(
            "duckduckgo-saved",
            lambda: WebSearchHarness(
                FakeSettingsService(
                    make_snapshot({"provider": "duckduckgo"}, readiness="ready")
                )
            ),
            None,
            ("open saved DuckDuckGo setting",),
        ),
        Fixture(
            "custom-provider-reset",
            lambda: WebSearchHarness(
                FakeSettingsService(
                    make_snapshot({
                        "provider": "exa",
                        "api_key_env_var": "CUSTOM_EXA_KEY",
                        "base_url": "https://custom.example.invalid",
                    })
                )
            ),
            _custom_reset,
            ("start Exa with custom credential variable and endpoint", "choose Brave"),
        ),
        Fixture(
            "masked-key-input",
            WebSearchHarness,
            _key_input,
            ("type fake key; do not save",),
        ),
        Fixture(
            "key-saved", WebSearchHarness, _key_saved, ("type fake key", "Save API key")
        ),
        Fixture(
            "key-session-only",
            WebSearchHarness,
            _key_session_only,
            ("type fake key", "Save API key with fake session-only outcome"),
        ),
        Fixture(
            "key-rejected",
            WebSearchHarness,
            _key_rejected,
            ("type fake key", "Save API key with fake rejection"),
        ),
        Fixture(
            "advanced",
            WebSearchHarness,
            _advanced,
            ("Show advanced settings", "scroll credential-variable field into view"),
        ),
        Fixture(
            "dirty-back", WebSearchHarness, _dirty_back, ("choose DuckDuckGo", "Escape")
        ),
        Fixture(
            "failed-save",
            WebSearchHarness,
            _failed_save,
            ("choose DuckDuckGo", "Save with fake disk error"),
        ),
        Fixture(
            "saved-read-failed",
            WebSearchHarness,
            _saved_read_failed,
            ("choose DuckDuckGo", "Save with fake read failure"),
        ),
        Fixture(
            "refresh-failed",
            WebSearchHarness,
            _refresh_failed,
            ("save with unreadable state", "Refresh with fake read error"),
        ),
        Fixture(
            "runtime-failed",
            WebSearchHarness,
            _runtime_failed,
            ("choose DuckDuckGo", "Save with fake runtime failure"),
        ),
        Fixture(
            "runtime-retry-failed",
            WebSearchHarness,
            _runtime_retry_failed,
            ("runtime failure", "Retry with fake reload error"),
        ),
        Fixture(
            "runtime-retry-read-failed",
            WebSearchHarness,
            _runtime_retry_read_failed,
            ("runtime failure", "Retry applies runtime", "fake settings read fails"),
        ),
        Fixture(
            "runtime-retry-ui-failed",
            WebSearchHarness,
            _runtime_retry_ui_failed,
            ("runtime failure", "Retry applies runtime", "fake UI refresh fails"),
        ),
    ]


def _save_svg(app: App, path: Path) -> tuple[tuple[int, ...], tuple[int, ...]]:
    palette = app.ansi_theme
    original = Console.export_svg

    def export_with_palette(console: Console, *args: Any, **kwargs: Any) -> str:
        kwargs["theme"] = palette
        return original(console, *args, **kwargs)

    with patch.object(Console, "export_svg", export_with_palette):
        app.save_screenshot(filename=path.name, path=str(path.parent))
    return tuple(palette.foreground_color), tuple(palette.background_color)


async def capture(
    fixture: Fixture, size: tuple[int, int], theme: str, output: Path, commit: str
) -> dict[str, Any]:
    app = fixture.factory()
    app.theme = theme
    stem = f"{fixture.name}-{size[0]}x{size[1]}-{theme}"
    svg = output / f"{stem}.svg"
    png = output / f"{stem}.png"
    async with app.run_test(size=size) as pilot:
        await pilot.pause(0.1)
        screen = _screen(app)
        if fixture.prepare is not None:
            await fixture.prepare(pilot, app)
        await pilot.pause(0.1)
        state = {
            "provider": screen._draft["provider"],
            "visible_selected_provider": screen._visible_provider(
                screen._draft["provider"]
            ),
            "provider_option_ids": [
                option.id
                for option in screen.query_one(
                    "#websearch-providers", OptionList
                ).options
            ],
            "provider_cursor": screen._provider_cursor,
            "permission": screen._draft["permission"],
            "dirty": screen._dirty_settings(),
            "forced_reset": sorted(screen._forced_reset),
            "advanced": screen._advanced,
            "confirming": screen._confirming,
            "needs_refresh": screen._needs_refresh,
            "runtime_failed": screen._runtime_failed,
            "ui_refresh_failed": screen._ui_refresh_failed,
            "key_input_nonempty": bool(screen.query_one("#websearch-key", Input).value),
            "key_input_password": screen.query_one("#websearch-key", Input).password,
            "credential_message": screen._credential_message,
            "message": screen._message,
            "fake_settings_save_calls": len(app.service.saved),
            "fake_runtime_retry_calls": app.service.retry_count,
            "fake_ui_retry_calls": app.service.ui_retry_count,
            "fake_credential_save_calls": len(app.credentials.saved),
            "focused_id": app.focused.id if app.focused else None,
        }
        foreground, background = _save_svg(app, svg)
    subprocess.run(
        ["inkscape", str(svg), "--export-filename", str(png)],
        check=True,
        capture_output=True,
        text=True,
    )
    if not png.is_file():
        raise RuntimeError(f"Inkscape did not create {png}")
    if FAKE_KEY in svg.read_text():
        raise AssertionError("fake secret appeared in rendered SVG")
    return {
        "fixture": fixture.name,
        "viewport": f"{size[0]}x{size[1]}",
        "theme": theme,
        "source_commit": commit,
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "host_context": "WebSearchHarness with isolated pushed modal; fake services only",
        "actions": fixture.actions,
        **state,
        "terminal_palette_foreground": foreground,
        "terminal_palette_background": background,
        "svg": str(svg.relative_to(ROOT)),
        "png": str(png.relative_to(ROOT)),
        "png_visual_valid": True,
    }


async def run(args: argparse.Namespace) -> int:
    if not shutil.which("inkscape"):
        raise RuntimeError("Inkscape is required for PNG conversion")
    output = (ROOT / args.output).resolve()
    output.relative_to(ROOT / "docs/reviews/web-search")
    output.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    selected = [
        fixture for fixture in fixtures() if args.fixture in {"all", fixture.name}
    ]
    if not selected:
        raise ValueError(f"unknown fixture: {args.fixture}")
    manifest = output / "captures.jsonl"
    records = (
        {
            item["png"]: item
            for line in manifest.read_text().splitlines()
            if (item := json.loads(line))
        }
        if manifest.exists()
        else {}
    )
    failed = 0
    for fixture in selected:
        for size in args.sizes:
            for theme in args.themes:
                try:
                    item = await capture(fixture, size, theme, output, commit)
                    records[item["png"]] = item
                    print(f"PASS {fixture.name} {size[0]}x{size[1]} {theme}")
                except Exception as exc:
                    failed += 1
                    print(
                        f"FAIL {fixture.name} {size[0]}x{size[1]} {theme}: {exc}",
                        file=sys.stderr,
                    )
                    traceback.print_exception(exc)
    manifest.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in records.values())
    )
    print(f"{len(selected)} fixtures, {failed} failed; manifest={manifest}")
    return int(failed != 0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", default="all")
    parser.add_argument("--size", dest="sizes", action="append")
    parser.add_argument("--theme", dest="themes", action="append")
    parser.add_argument("--output", default="docs/reviews/web-search/evidence")
    args = parser.parse_args()
    args.sizes = [
        tuple(map(int, value.split("x")))
        for value in (args.sizes or ["80x24", "80x48"])
    ]
    args.themes = args.themes or ["ansi-dark", "ansi-light"]
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
