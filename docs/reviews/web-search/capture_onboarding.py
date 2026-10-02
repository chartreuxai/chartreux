#!/usr/bin/env python3
"""Capture offline onboarding Web search decisions in the production modal."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
from typing import Any, cast

from capture_web_search import (
    FAKE_KEY,
    ROOT,
    _advanced,
    _choose,
    _key_session_only,
    _press,
    _save_svg,
)
from textual.pilot import Pilot
from textual.widgets import Button, Input, OptionList

from chartreux.ui.settings_service import SettingsSaveOutcome
from chartreux.ui.web_search import WebSearchExit, WebSearchScreen
from tests.cli.textual_ui.web_search_fixture import (
    FakeCredentials,
    FakeSettingsService,
    WebSearchHarness,
    make_snapshot,
)


class OnboardingWebSearchHarness(WebSearchHarness):
    """In-memory host for the onboarding variant and its exit result."""

    def __init__(
        self,
        service: FakeSettingsService | None = None,
        credentials: FakeCredentials | None = None,
    ) -> None:
        super().__init__(service, credentials, mode="onboarding")

    @property
    def exit_result(self) -> WebSearchExit | None:
        return cast(WebSearchExit | None, self.result)


Prepare = Callable[[Pilot, OnboardingWebSearchHarness], Awaitable[None]]
Finish = Callable[[Pilot, OnboardingWebSearchHarness], Awaitable[WebSearchExit]]
Factory = Callable[[], OnboardingWebSearchHarness]


@dataclass(frozen=True)
class Fixture:
    name: str
    factory: Factory
    prepare: Prepare | None
    actions: tuple[str, ...]
    finish: Finish | None = None


def _screen(app: OnboardingWebSearchHarness) -> WebSearchScreen:
    screen = app.screen
    assert isinstance(screen, WebSearchScreen)
    assert screen.mode == "onboarding"
    return screen


async def _missing_finish_guidance(
    pilot: Pilot, app: OnboardingWebSearchHarness
) -> None:
    screen = _screen(app)
    await _press(pilot, screen, "#websearch-save")
    assert "Save a web search API key" in screen._message
    assert app.exit_result is None


async def _no_mistral_fallback(_pilot: Pilot, app: OnboardingWebSearchHarness) -> None:
    screen = _screen(app)
    options = screen.query_one("#websearch-providers", OptionList)
    assert [option.id for option in options.options] == ["exa", "brave", "duckduckgo"]
    assert screen._draft["provider"] == "auto"
    assert screen._visible_provider("auto") is None
    assert screen.query_one("#websearch-save", Button).disabled
    assert not screen.query_one("#websearch-credential").display


async def _exa_chosen(_pilot: Pilot, app: OnboardingWebSearchHarness) -> None:
    screen = _screen(app)
    options = screen.query_one("#websearch-providers", OptionList)
    assert [option.id for option in options.options] == ["exa", "brave", "duckduckgo"]
    assert screen._visible_provider(screen._draft["provider"]) == "exa"
    assert not screen.query_one("#websearch-save", Button).disabled
    assert screen.query_one("#websearch-credential").display


def _exa_factory() -> OnboardingWebSearchHarness:
    snapshot = make_snapshot({"provider": "exa"}, readiness="missing_key")
    web = snapshot.web_search
    assert web is not None
    snapshot = snapshot.model_copy(
        update={
            "web_search": web.model_copy(
                update={
                    "readiness_message": "EXA_API_KEY is missing",
                    "credential_env_var": "EXA_API_KEY",
                }
            )
        }
    )
    return OnboardingWebSearchHarness(FakeSettingsService(snapshot))


async def _duck_draft(pilot: Pilot, app: OnboardingWebSearchHarness) -> None:
    screen = _screen(app)
    await _choose(pilot, screen, "#websearch-providers", "duckduckgo")
    assert screen._draft["provider"] == "duckduckgo"
    assert screen._dirty_settings()
    assert not screen.query_one("#websearch-credential").display
    assert str(screen.query_one("#websearch-save", Button).label) == "Save and finish"


async def _duck_draft_ready_save(pilot: Pilot, app: OnboardingWebSearchHarness) -> None:
    await _duck_draft(pilot, app)
    ready_snapshot = make_snapshot({"provider": "duckduckgo"}, readiness="ready")
    app.service.snapshot = ready_snapshot
    app.service.outcome = SettingsSaveOutcome("saved", "applied", ready_snapshot)


async def _session_key(pilot: Pilot, app: OnboardingWebSearchHarness) -> None:
    await _key_session_only(pilot, app)
    assert app.credentials.resolve_key("MISTRAL_API_KEY") == FAKE_KEY
    screen = _screen(app)
    assert screen._selected_env() == "MISTRAL_API_KEY"
    assert screen._has_key("MISTRAL_API_KEY")
    # The fake read must reflect the credential the fake service just accepted.
    app.service.snapshot = make_snapshot(readiness="ready")
    assert app.exit_result is None


async def _advanced_open(pilot: Pilot, app: OnboardingWebSearchHarness) -> None:
    await _advanced(pilot, app)
    assert _screen(app)._advanced


async def _dirty_skip(pilot: Pilot, app: OnboardingWebSearchHarness) -> None:
    await _duck_draft(pilot, app)
    screen = _screen(app)
    await _press(pilot, screen, "#websearch-skip")
    assert screen._confirming
    assert app.exit_result is None


async def _save_error(pilot: Pilot, app: OnboardingWebSearchHarness) -> None:
    app.service.outcome = SettingsSaveOutcome(
        "not_saved", "unchanged", error="fake disk denied"
    )
    await _duck_draft(pilot, app)
    screen = _screen(app)
    await _press(pilot, screen, "#websearch-save")
    assert screen._message.startswith("Not saved:")
    assert screen._dirty_settings()
    assert app.exit_result is None


async def _finish(pilot: Pilot, app: OnboardingWebSearchHarness) -> WebSearchExit:
    screen = _screen(app)
    await _press(pilot, screen, "#websearch-save")
    assert app.exit_result == "finish", (
        app.exit_result,
        screen._message,
        screen._dirty_settings(),
    )
    return "finish"


async def _skip_discard(pilot: Pilot, app: OnboardingWebSearchHarness) -> WebSearchExit:
    await _press(pilot, _screen(app), "#websearch-discard")
    assert app.exit_result == "skip"
    return "skip"


async def _back(pilot: Pilot, app: OnboardingWebSearchHarness) -> WebSearchExit:
    await _press(pilot, _screen(app), "#websearch-back")
    assert app.exit_result == "back"
    return "back"


def fixtures() -> list[Fixture]:
    return [
        Fixture(
            "choice-no-mistral-fallback",
            OnboardingWebSearchHarness,
            _no_mistral_fallback,
            ("open fallback onboarding with raw Automatic and no Mistral key",),
        ),
        Fixture(
            "choice-exa-chosen",
            _exa_factory,
            _exa_chosen,
            ("open fallback onboarding with saved Exa selection",),
        ),
        Fixture(
            "onboarding-missing-key",
            OnboardingWebSearchHarness,
            None,
            ("open Web search onboarding with missing Mistral key",),
        ),
        Fixture(
            "onboarding-missing-finish-guidance",
            OnboardingWebSearchHarness,
            _missing_finish_guidance,
            ("open missing-key onboarding", "press Finish setup"),
        ),
        Fixture(
            "onboarding-duckduckgo-ready",
            lambda: OnboardingWebSearchHarness(
                FakeSettingsService(
                    make_snapshot({"provider": "duckduckgo"}, readiness="ready")
                )
            ),
            None,
            (
                "open saved keyless DuckDuckGo onboarding",
                "press Finish setup after capture",
            ),
            _finish,
        ),
        Fixture(
            "onboarding-duckduckgo-draft",
            OnboardingWebSearchHarness,
            _duck_draft_ready_save,
            ("choose DuckDuckGo with Space", "press Save and finish after capture"),
            _finish,
        ),
        Fixture(
            "onboarding-session-key",
            OnboardingWebSearchHarness,
            _session_key,
            (
                "type synthetic key",
                "save for this session",
                "fake authoritative read becomes ready",
                "finish after capture",
            ),
            _finish,
        ),
        Fixture(
            "onboarding-advanced",
            OnboardingWebSearchHarness,
            _advanced_open,
            ("show Advanced", "scroll credential field into view"),
        ),
        Fixture(
            "onboarding-dirty-skip",
            OnboardingWebSearchHarness,
            _dirty_skip,
            ("choose DuckDuckGo", "press Skip for now", "Discard after capture"),
            _skip_discard,
        ),
        Fixture(
            "onboarding-back-presets",
            OnboardingWebSearchHarness,
            None,
            ("open Web search onboarding", "press Back to presets after capture"),
            _back,
        ),
        Fixture(
            "onboarding-save-error",
            OnboardingWebSearchHarness,
            _save_error,
            ("choose DuckDuckGo", "press Save and finish with fake disk error"),
        ),
    ]


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
            "mode": screen.mode,
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
            "dirty": screen._dirty_settings(),
            "confirming": screen._confirming,
            "needs_refresh": screen._needs_refresh,
            "runtime_failed": screen._runtime_failed,
            "ui_refresh_failed": screen._ui_refresh_failed,
            "message": screen._message,
            "credential_message": screen._credential_message,
            "key_input_nonempty": bool(screen.query_one("#websearch-key", Input).value),
            "key_input_password": screen.query_one("#websearch-key", Input).password,
            "finish_label": str(screen.query_one("#websearch-save", Button).label),
            "skip_visible": screen.query_one("#websearch-skip", Button).display,
            "back_label": str(screen.query_one("#websearch-back", Button).label),
            "focused_id": app.focused.id if app.focused else None,
            "fake_settings_save_calls": len(app.service.saved),
            "fake_credential_save_calls": len(app.credentials.saved),
        }
        foreground, background = _save_svg(app, svg)
        if fixture.finish is not None:
            state["post_capture_result"] = await fixture.finish(pilot, app)
    subprocess.run(
        ["inkscape", str(svg), "--export-filename", str(png)],
        check=True,
        capture_output=True,
        text=True,
    )
    if not png.is_file():
        raise RuntimeError(f"Inkscape did not create {png}")
    if FAKE_KEY in svg.read_text():
        raise AssertionError("synthetic key appeared in rendered SVG")
    return {
        "fixture": fixture.name,
        "viewport": f"{size[0]}x{size[1]}",
        "theme": theme,
        "source_commit": commit,
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "host_context": "isolated onboarding WebSearchScreen with fake services",
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
                        f"FAIL {fixture.name} {size[0]}x{size[1]} {theme}: {exc!r}",
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
    parser.add_argument(
        "--output", default="docs/reviews/web-search/evidence/onboarding"
    )
    args = parser.parse_args()
    args.sizes = [
        tuple(map(int, value.split("x")))
        for value in (args.sizes or ["80x24", "80x48"])
    ]
    args.themes = args.themes or ["ansi-dark", "ansi-light"]
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
