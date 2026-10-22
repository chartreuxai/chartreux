"""Dispatch lint surface: the composed-artifact lint (S8) and the loud report.

Package 16b retired the legacy dispatch lint surface (``lint_catalog``,
``lint_activation``, ``lint_mode``, ``parse_references``, ``roster_shape``)
along with the dispatch policy itself. What survives is the S8 composed-artifact
lint, and the model-catalog load path now reports a legacy ``[dispatch]`` table
loudly with successor guidance instead of linting, merging, or repairing it.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import re

import pytest

from chartreux.core.agents import AgentManager
from chartreux.core.agents.identities import get_builtin_identity
from chartreux.core.config.agent_selection import AgentSelectionConfig
from chartreux.core.dispatch import lint as lint_module
from chartreux.core.dispatch.lint import Diagnostic, lint_composed
from chartreux.core.model_catalog.defaults import SHIPPED_CATALOG
from chartreux.core.model_catalog.loader import load_catalog
from chartreux.core.skills.manager import SkillManager
from chartreux.core.system_prompt import get_universal_system_prompt
from tests.conftest import build_test_vibe_config
from tests.stubs.fake_config_orchestrator import FakeConfigOrchestrator


def ids(diagnostics: Iterable[Diagnostic]) -> set[str]:
    return {item.id for item in diagnostics}


# --- Load-path pins: a legacy [dispatch] table is reported, never linted -----
#
# The retired table used to flow through resolve_dispatch_overlay and the S1/
# S9/S10 lint boundaries. The successor behavior: the loader reports it loudly
# with successor guidance, never merges it, never lints it, and never rewrites
# the user's file.


def test_retired_dispatch_table_is_reported_and_never_linted(tmp_path: Path) -> None:
    path = tmp_path / "models.toml"
    original = b'[dispatch.slots.mechanical]\nrole = "@missing"\n'
    path.write_bytes(original)

    snapshot = load_catalog(path)

    # The retired table is reported with successor guidance and carries no
    # dispatch state on the snapshot at all.
    assert not hasattr(snapshot, "dispatch")
    assert len(snapshot.dispatch_diagnostics) == 1
    report = snapshot.dispatch_diagnostics[0]
    assert "[dispatch]" in report
    assert "retired" in report
    assert "[orchestration.routes.<purpose>]" in report
    assert str(path) in report
    # The load succeeds and serves the shipped catalog: the retired table is
    # not linted, not merged, and not fatal.
    assert snapshot.catalog == SHIPPED_CATALOG
    # The user's file is never repaired or rewritten.
    assert path.read_bytes() == original


def test_retired_dispatch_load_is_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "models.toml"
    path.write_text('[dispatch]\nmode = "standalone"\n')

    first = load_catalog(path)
    second = load_catalog(path)

    assert first.dispatch_diagnostics == second.dispatch_diagnostics
    assert first.revision == second.revision
    assert first.catalog == second.catalog == SHIPPED_CATALOG


def test_lint_surface_retains_only_the_composed_lint() -> None:
    # The legacy lint entry points are gone; the composed-artifact lint and its
    # diagnostic plumbing survive.
    for retired in (
        "lint_catalog",
        "lint_activation",
        "lint_mode",
        "parse_references",
        "roster_shape",
        "RosterShape",
        "References",
    ):
        assert not hasattr(lint_module, retired), retired
    for kept in ("lint_composed", "Diagnostic", "DispatchLintError", "reject_errors"):
        assert hasattr(lint_module, kept), kept


# --- S8: the composed-artifact lint (design §5) -----------------------------
#
# S8 is re-anchored to the composed artifact and the identity/workflow axes:
# universal critical-instruction anchors are required in every composed
# prompt; the delegation-obligation anchor is required for identities whose
# typed delegation is required or optional; delegated-verification anchors
# belong to the ceremony workflow only. The DispatchMode axis is gone — the
# lint covers composed output, not raw skeletons.

CODER_FACTS = get_builtin_identity("coder")
ORCHESTRATOR_FACTS = get_builtin_identity("orchestrator")
ORCHESTRATED_SELECTION = AgentSelectionConfig(
    identity="orchestrator",
    workflow="ceremony",
    orchestration_strategy="builtin:default",
)


def _composed_prompt(selection: AgentSelectionConfig | None = None) -> str:
    # The composed artifact with every optional detail section off: the
    # served text is exactly the S8-linted policy text.
    config = build_test_vibe_config(
        agent=selection or AgentSelectionConfig(),
        include_model_info=False,
        include_prompt_detail=False,
        include_project_context=False,
        include_commit_signature=False,
    )
    return str(
        get_universal_system_prompt(
            config,
            SkillManager(lambda: config),
            AgentManager(FakeConfigOrchestrator(config)),
        )
    )


def test_s8_composed_anchor_tripwire() -> None:
    prompt = _composed_prompt()
    assert lint_composed(prompt, identity_facts=CODER_FACTS, workflow="minimal") == ()
    # Removing the truthful-reporting anchor fails the lint.
    mutated = prompt.replace("Never claim a check you did not run.", "Be honest.")
    assert "never claim a check you did not run" not in mutated.lower()
    assert "S8" in ids(
        lint_composed(mutated, identity_facts=CODER_FACTS, workflow="minimal")
    )
    # A coder composition missing its delegation obligation fails the lint.
    mutated = re.sub(r"delegation is optional", "redacted", prompt, flags=re.IGNORECASE)
    assert "delegation is optional" not in mutated.lower()
    assert "S8" in ids(
        lint_composed(mutated, identity_facts=CODER_FACTS, workflow="minimal")
    )
    # Injecting residual gate language into a minimal composition fails.
    assert "S8" in ids(
        lint_composed(
            f"{prompt}\n\nNever edit repo files yourself.",
            identity_facts=CODER_FACTS,
            workflow="minimal",
        )
    )
    # The ceremony workflow requires delegated verification.
    ceremony = _composed_prompt(ORCHESTRATED_SELECTION)
    assert (
        lint_composed(ceremony, identity_facts=ORCHESTRATOR_FACTS, workflow="ceremony")
        == ()
    )
    mutated = ceremony.replace(
        "never run tests, builds, or other verification yourself", "run checks"
    )
    assert (
        "never run tests, builds, or other verification yourself" not in mutated.lower()
    )
    assert "S8" in ids(
        lint_composed(mutated, identity_facts=ORCHESTRATOR_FACTS, workflow="ceremony")
    )


@pytest.mark.parametrize(
    "identity", ["coder", "orchestrator", "researcher", "reviewer", "advisor"]
)
@pytest.mark.parametrize("workflow", ["minimal", "ceremony"])
def test_s8_composed_prompt_anchors_shipped_selections(
    identity: str, workflow: str
) -> None:
    # Every shipped selection composes to a lint-clean prompt; the lint is
    # keyed on the typed identity facts and the workflow, not DispatchMode.
    prompt = _composed_prompt(
        AgentSelectionConfig(identity=identity, workflow=workflow)
    )
    assert (
        lint_composed(
            prompt, identity_facts=get_builtin_identity(identity), workflow=workflow
        )
        == ()
    )
    assert "$dispatch" not in prompt


def test_s8_custom_identity_needs_no_delegation_obligation(config_dir: Path) -> None:
    # A custom identity has no harness-known posture: the lint requires no
    # delegation-obligation anchor for it, and none is composed.
    user_prompts = config_dir / "prompts"
    user_prompts.mkdir()
    (user_prompts / "custom.md").write_text(
        "You are the custom specialist.\n", encoding="utf-8"
    )
    prompt = _composed_prompt(AgentSelectionConfig(identity="user:custom"))
    assert "**Delegation posture.**" not in prompt
    assert lint_composed(prompt, identity_facts=None, workflow="minimal") == ()
