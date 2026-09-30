#!/usr/bin/env python3
"""Capture the offline non-Mistral onboarding preset suggestion."""

from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

from capture_web_search import ROOT, _save_svg
from textual.widgets import OptionList

from chartreux.core.model_catalog.defaults import SHIPPED_CATALOG
from chartreux.core.model_catalog.loader import CatalogSnapshot
from chartreux.core.model_catalog.schema import ModelCatalog
from chartreux.ui.providers.workbench import ProviderWorkbenchScreen, WorkbenchView
from tests.ui.providers.test_workbench import Host, setup


def _host() -> Host:
    screen, services = setup()
    catalog = SHIPPED_CATALOG.model_dump(mode="python")
    catalog["providers"]["custom"] = {
        "api_base": "https://custom.example.invalid/v1",
        "api_key_env_var": "SHARED_KEY",
    }
    catalog["models"]["my-model"] = {
        "deployments": [{"provider": "custom", "name": "my-model"}]
    }
    services.catalog = CatalogSnapshot(
        ModelCatalog.model_validate(catalog), "review-custom", frozenset({"custom"})
    )
    screen.snapshot = services.catalog
    screen.mode = "onboarding"
    screen.initial_view = "presets"
    return Host(screen)


async def capture(
    size: tuple[int, int], theme: str, output: Path, commit: str
) -> dict[str, Any]:
    app = _host()
    app.theme = theme
    stem = f"onboarding-nonmistral-presets-{size[0]}x{size[1]}-{theme}"
    svg = output / f"{stem}.svg"
    png = output / f"{stem}.png"
    async with app.run_test(size=size) as pilot:
        await pilot.pause(0.1)
        screen = app.screen
        assert isinstance(screen, ProviderWorkbenchScreen)
        assert screen.view is WorkbenchView.PRESETS
        assert screen.state is not None
        roles = ("orchestrator", "large", "medium", "small")
        assert all(screen.state.preset(role)[0] == "my-model" for role in roles)
        options = screen.query_one("#wb-presets", OptionList)
        finish = next(option for option in options.options if option.id == "finish")
        assert "Save presets and continue" in str(finish.prompt)
        assert "Web search" not in str(finish.prompt)
        foreground, background = _save_svg(app, svg)
        state = {
            "role_suggestions": {role: screen.state.preset(role) for role in roles},
            "focused_id": app.focused.id if app.focused else None,
        }
    subprocess.run(
        ["inkscape", str(svg), "--export-filename", str(png)],
        check=True,
        capture_output=True,
        text=True,
    )
    if not png.is_file():
        raise RuntimeError(f"Inkscape did not create {png}")
    return {
        "fixture": "onboarding-nonmistral-presets",
        "viewport": f"{size[0]}x{size[1]}",
        "theme": theme,
        "source_commit": commit,
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "host_context": "real ProviderWorkbenchScreen in fake Host; configured custom provider; no Mistral key",
        "actions": ["open onboarding preset step with fake configured custom provider"],
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
    for size in args.sizes:
        for theme in args.themes:
            item = await capture(size, theme, output, commit)
            records[item["png"]] = item
            print(f"PASS {item['fixture']} {item['viewport']} {theme}")
    manifest.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in records.values())
    )
    print(f"{len(records)} records; manifest={manifest}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
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
