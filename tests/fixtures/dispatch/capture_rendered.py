"""Capture the rendered dispatch goldens for both shipped presets.

Run from the repository root with ``uv run python tests/fixtures/dispatch/capture_rendered.py``.
Each golden is the full normalized rendered output for one preset: the CLI
prompt document (skeleton plus rendered dispatch regions, date pinned to
2000-01-01, shipped prompt pinned) and the task tool description (the raw
served bytes, unstripped). The orchestrated multi-model fixture renders the
current roster prose; the WP0 legacy baseline bytes it replaced are preserved
under ``legacy-orchestrated/`` (ADR 0018-G.1 superseded, see the ADR
amendment).
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
from chartreux.core.model_catalog.schema import ModelCatalog
from chartreux.core.prompts import load_system_prompt
from chartreux.core.system_prompt import _interpolate_prompt

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent
PINNED_DATE = date(2000, 1, 1)

# A deterministic two-canonical-model catalog: the roster roles resolve across
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
        "scout": {
            "model": "alpha-model",
            "thinking": "low",
            "description": "scout preset",
        },
        "worker": {
            "model": "alpha-model",
            "thinking": "medium",
            "description": "worker preset",
        },
        "heavy": {
            "model": "beta-model",
            "thinking": "high",
            "description": "heavy preset",
        },
    },
})

# A deterministic one-canonical-model catalog: every role the shipped policies
# reference binds the same base model, so the roster is single-model. It is an
# explicit fixture, not SHIPPED_CATALOG, so the single-model golden stays stable
# when the shipped roster changes.
SINGLE_MODEL_CATALOG = ModelCatalog.model_validate({
    "providers": {"solo": {"api_base": "https://solo.test"}},
    "models": {
        "solo-model": {
            "thinking": "medium",
            "deployments": [{"provider": "solo", "name": "solo-wire"}],
        }
    },
    "roles": {
        "orchestrator": {
            "model": "solo-model",
            "thinking": "high",
            "description": "main assistant preset",
        },
        "worker": {
            "model": "solo-model",
            "thinking": "medium",
            "description": "worker preset",
        },
        "scout": {
            "model": "solo-model",
            "thinking": "low",
            "description": "scout preset",
        },
        "heavy": {
            "model": "solo-model",
            "thinking": "high",
            "description": "heavy preset",
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
        ("orchestrated-singlemodel", ORCHESTRATED_PRESET, SINGLE_MODEL_CATALOG),
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
