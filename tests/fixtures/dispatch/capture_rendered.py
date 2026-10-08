"""Capture the rendered dispatch goldens for both shipped presets.

Run from the repository root with ``uv run python tests/fixtures/dispatch/capture_rendered.py``.
Each golden is the full normalized rendered output for one preset: the CLI
prompt document (skeleton plus rendered dispatch regions, date pinned to
2000-01-01, shipped prompt pinned) and the task tool description (the raw
served bytes, unstripped). The orchestrated multi-model fixture reproduces the
WP0 legacy baseline bytes in its routing regions.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import patch

from chartreux.core.config.harness_files import init_harness_files_manager
from chartreux.core.dispatch.presets import ORCHESTRATED_PRESET, STANDALONE_PRESET
from chartreux.core.dispatch.renderer import (
    render_cli_prompt,
    render_task_description,
    roster_for,
    task_skeleton,
)
from chartreux.core.dispatch.schema import DispatchPolicy
from chartreux.core.model_catalog.defaults import SHIPPED_CATALOG
from chartreux.core.model_catalog.schema import ModelCatalog
from chartreux.core.prompts import load_system_prompt
from chartreux.core.system_prompt import _interpolate_prompt

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent
PINNED_DATE = date(2000, 1, 1)

# A deterministic two-canonical-model catalog: the tier roles resolve across
# two base models, so the roster is multi-model.
MULTI_MODEL_CATALOG = ModelCatalog.model_validate({
    "providers": {
        "alpha": {"api_base": "https://alpha.test"},
        "beta": {"api_base": "https://beta.test"},
    },
    "models": {
        "alpha-model": {
            "thinking": "medium",
            "deployments": [{"provider": "alpha", "name": "alpha-wire"}],
        },
        "beta-model": {
            "thinking": "high",
            "deployments": [{"provider": "beta", "name": "beta-wire"}],
        },
    },
    "roles": {
        "orchestrator": {
            "model": "beta-model",
            "thinking": "high",
            "description": "main assistant preset",
        },
        "small": {
            "model": "alpha-model",
            "thinking": "low",
            "description": "small preset",
        },
        "medium": {
            "model": "alpha-model",
            "thinking": "medium",
            "description": "medium preset",
        },
        "large": {
            "model": "beta-model",
            "thinking": "high",
            "description": "large preset",
        },
    },
})


def _render_cli(policy: DispatchPolicy, catalog: ModelCatalog) -> str:
    with patch("chartreux.core.system_prompt.date") as date_mock:
        date_mock.today.return_value = PINNED_DATE
        return render_cli_prompt(
            policy,
            roster_for(catalog, policy),
            _interpolate_prompt(load_system_prompt("cli")),
        )


def _render_task(policy: DispatchPolicy, catalog: ModelCatalog) -> str:
    return render_task_description(policy, roster_for(catalog, policy), task_skeleton())


def main() -> None:
    init_harness_files_manager()
    fixtures = (
        ("standalone", STANDALONE_PRESET, MULTI_MODEL_CATALOG),
        ("orchestrated", ORCHESTRATED_PRESET, MULTI_MODEL_CATALOG),
        ("orchestrated-singlemodel", ORCHESTRATED_PRESET, SHIPPED_CATALOG),
    )
    for name, policy, catalog in fixtures:
        directory = OUTPUT / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "cli.md").write_bytes(
            (_render_cli(policy, catalog).rstrip("\n") + "\n").encode("utf-8")
        )
        (directory / "task.md").write_bytes(
            (_render_task(policy, catalog).rstrip("\n") + "\n").encode("utf-8")
        )
        print(f"captured {name}/cli.md and {name}/task.md")


if __name__ == "__main__":
    main()
