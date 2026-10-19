from __future__ import annotations

from pathlib import Path
import subprocess

from chartreux.core.prompts import load_system_prompt

ROOT = Path(__file__).resolve().parents[2]


def test_fresh_worker_prompt_includes_operating_rules() -> None:
    rendered = load_system_prompt("worker")

    assert "<!-- worker-operating-rules -->" in rendered
    assert (
        "Shell grammar: The shell has no variable expansion, command substitution, or loops; "
        "commands must be literal. Environment assignments prefixing commands must use "
        "uppercase names. No leading `=` in command names. Run scripts through an explicit "
        "interpreter, not implicit `uv`."
    ) in rendered
    assert "bash_read: max_bytes >= 4096, wait_seconds <= 30." in rendered
    assert (
        "Interpreter default: system Python lacks project dependencies; use the project's "
        "uv-managed interpreter by default."
    ) in rendered


def test_bash_read_description_exposes_schema_limits() -> None:
    description = (
        ROOT / "chartreux/core/tools/builtins/prompts/bash_read.md"
    ).read_text(encoding="utf-8")

    assert (
        "`max_bytes` must be 4096-64000 UTF-8 output bytes (minimum 4096)."
        in description
    )
    assert "`wait_seconds` must not exceed 30." in description


def test_existing_worker_denial_and_authority_rules_remain() -> None:
    rendered = load_system_prompt("worker")
    baseline = subprocess.run(
        [
            "git",
            "show",
            "545e2719086f32368fb4ea051e1a2e547d37828f:chartreux/core/prompts/worker.md",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    def section(prompt: str, start: str, end: str) -> str:
        return prompt.split(start, maxsplit=1)[1].split(end, maxsplit=1)[0]

    current_contract = section(
        rendered, "## Subagent contract", "## Worker operating rules"
    )
    assert current_contract == section(
        baseline, "## Subagent contract", "## Operational discipline"
    )
    assert section(
        rendered, "## Operational discipline", "## Role guidance"
    ) == section(baseline, "## Operational discipline", "## Role guidance")
