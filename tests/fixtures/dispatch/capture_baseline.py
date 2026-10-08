"""Capture the shipped legacy routing text through Chartreux prompt readers.

Run from the repository root with ``uv run python tests/fixtures/dispatch/capture_baseline.py``.
The CLI template is loaded through ``load_system_prompt`` (including its normal
strip behavior), interpolated through the system-prompt renderer with the date
pinned to 2000-01-01, and sliced by its source line anchors. Task routing is
captured from ``Task.get_full_description()`` so its unstripped served bytes are
preserved.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from chartreux.core.config.harness_files import init_harness_files_manager
from chartreux.core.prompts import load_system_prompt
from chartreux.core.system_prompt import _interpolate_prompt
from chartreux.core.tools.builtins.task import Task

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent / "legacy-orchestrated"
PINNED_DATE = "2000-01-01"


def main() -> None:
    init_harness_files_manager()
    # load_system_prompt resolves builtins and applies the normal .strip() read
    # path; fixed date avoids a wall-clock substitution changing the capture.
    with patch("chartreux.core.system_prompt.date") as date_mock:
        date_mock.today.return_value.isoformat.return_value = PINNED_DATE
        cli = _interpolate_prompt(load_system_prompt("cli"))
    lines = cli.splitlines(keepends=True)
    # Prompt.read/load_prompt strips the complete document, so source line 127
    # remains index 126 in the loaded prompt. Keep exactly the routing block.
    cli_block = "".join(lines[126:139])
    task_description = Task.get_full_description()
    task_lines = task_description.splitlines(keepends=True)
    task_block = task_lines[14]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "cli-routing.md").write_bytes(cli_block.encode("utf-8"))
    (OUTPUT / "task-routing.md").write_bytes(task_block.encode("utf-8"))


if __name__ == "__main__":
    main()
