from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path

import pytest

from chartreux.core.agents.models import BUILTIN_SUBAGENTS
from chartreux.core.dispatch.lint import (
    Diagnostic,
    DispatchLintError,
    lint_activation,
    lint_catalog,
    lint_mode,
    lint_rendered,
    parse_references,
    reject_errors,
    roster_shape,
)
from chartreux.core.dispatch.presets import SHIPPED_PRESETS, STANDALONE_PRESET
from chartreux.core.dispatch.renderer import render_cli_prompt, roster_for
from chartreux.core.dispatch.schema import DispatchPolicy
from chartreux.core.model_catalog.contracts import (
    CatalogChanges,
    CatalogValidationError,
)
from chartreux.core.model_catalog.defaults import SHIPPED_CATALOG
from chartreux.core.model_catalog.loader import (
    CatalogSnapshot,
    CatalogStore,
    fallback_dispatch_snapshot,
    merge_catalog_overlay,
    resolve_dispatch_overlay,
)
from chartreux.core.model_catalog.resolver import ModelResolver
from chartreux.core.prompts import SystemPrompt
from tests.conftest import multi_model_catalog


def candidate(**changes: object) -> DispatchPolicy:
    # model_copy intentionally permits invalid candidates for negative lint tests.
    return STANDALONE_PRESET.model_copy(update=changes)


def ids(diagnostics: Iterable[Diagnostic]) -> set[str]:
    return {item.id for item in diagnostics}


def slot_candidate(name: str, **changes: object) -> DispatchPolicy:
    slots = dict(STANDALONE_PRESET.slots)
    slots[name] = slots[name].model_copy(update=changes)
    return candidate(slots=slots)


@pytest.mark.parametrize("policy", SHIPPED_PRESETS.values())
def test_shipped_presets_pass_both_boundaries(policy: DispatchPolicy) -> None:
    assert lint_catalog(policy, SHIPPED_CATALOG) == ()
    assert lint_activation(policy, BUILTIN_SUBAGENTS) == ()


def test_s1_unknown_and_unavailable_active_roles() -> None:
    assert "S1" in ids(
        lint_catalog(slot_candidate("mechanical", role="@missing"), SHIPPED_CATALOG)
    )
    catalog = merge_catalog_overlay(
        SHIPPED_CATALOG, {"roles": {"draft": {"model": "missing", "thinking": "high"}}}
    )
    assert lint_catalog(STANDALONE_PRESET, catalog) == ()
    assert "S1" in ids(
        lint_catalog(slot_candidate("mechanical", role="@draft"), catalog)
    )


def test_missing_roles_recover_without_seeding_or_repair() -> None:
    catalog = SHIPPED_CATALOG.model_copy(update={"roles": {}})
    policy, diagnostics = resolve_dispatch_overlay({}, catalog=catalog)
    assert policy is STANDALONE_PRESET
    assert "S1 slots.mechanical.role" in diagnostics[0]
    assert catalog.roles == {}
    assert lint_catalog(policy, catalog)  # Recovery does not invent bindings.


def test_activation_recovery_rejects_whole_candidate_and_preserves_diagnostics() -> (
    None
):
    policy = slot_candidate("mechanical", profile="missing")
    snapshot = CatalogSnapshot(
        SHIPPED_CATALOG, "candidate", dispatch=policy, dispatch_diagnostics=("prior",)
    )
    diagnostics = lint_activation(policy, BUILTIN_SUBAGENTS)
    recovered = fallback_dispatch_snapshot(snapshot, "; ".join(map(str, diagnostics)))
    assert recovered.dispatch is STANDALONE_PRESET
    assert recovered.catalog is snapshot.catalog
    assert recovered.dispatch_diagnostics[0] == "prior"
    assert "R1 slots.mechanical.profile" in recovered.dispatch_diagnostics[-1]
    assert "dispatch config was bypassed" in recovered.dispatch_diagnostics[-1]
    assert snapshot.dispatch.slots["mechanical"].profile == "missing"


def test_s2_builtin_shadowing_only_at_activation() -> None:
    assert lint_catalog(STANDALONE_PRESET, SHIPPED_CATALOG) == ()
    assert "S2" in ids(
        lint_activation(
            STANDALONE_PRESET, BUILTIN_SUBAGENTS, shadowed_profiles=["worker"]
        )
    )
    assert (
        lint_activation(
            STANDALONE_PRESET, BUILTIN_SUBAGENTS, shadowed_profiles=["custom"]
        )
        == ()
    )


def test_s3_shipped_vocabulary_cannot_be_removed_but_can_be_extended() -> None:
    vocabulary = dict(STANDALONE_PRESET.vocabulary)
    vocabulary.pop("search")
    assert "S3" in ids(lint_catalog(candidate(vocabulary=vocabulary), SHIPPED_CATALOG))
    vocabulary = dict(STANDALONE_PRESET.vocabulary)
    vocabulary["custom"] = vocabulary["search"]
    assert lint_catalog(candidate(vocabulary=vocabulary), SHIPPED_CATALOG) == ()


@pytest.mark.parametrize(
    "purpose", ["implementation", "implementation-demanding-settled", "mechanical-edit"]
)
def test_s4_read_only_slot_cannot_implement(purpose: str) -> None:
    policy = slot_candidate("advisor", purposes=(purpose,))
    assert "S4" in ids(lint_catalog(policy, SHIPPED_CATALOG))


def test_s5_all_escalation_routes_require_reason() -> None:
    policy = slot_candidate("escalation-implementor", purposes=("implementation",))
    assert "S5" not in ids(lint_catalog(policy, SHIPPED_CATALOG))
    assert "S5" in ids(lint_activation(policy, BUILTIN_SUBAGENTS))
    policy = candidate(failure_routing="execution: use slot `escalation-implementor`.")
    assert "S5" in ids(lint_activation(policy, BUILTIN_SUBAGENTS))
    policy = candidate(
        failure_routing="execution: use slot `escalation-implementor` and state the reason."
    )
    assert lint_activation(policy, BUILTIN_SUBAGENTS) == ()


@pytest.mark.parametrize(
    "compositions",
    [
        "",
        "review.deep: use no seats.",
        "review.deep: use slots `reviewer`, `reviewer`.",
        "review.deep: use slot `missing`.",
        "review.deep: use slot `implementor`.",
        "review.deep: use slot `reviewer`. If `reviewer` authored the work, substitute slot `reviewer`.",
        "review.deep: use slots `reviewer`, `execution-reviewer`. If `reviewer` authored the work, substitute slot `execution-reviewer`.",
    ],
)
def test_s6_rejects_degenerate_compositions(compositions: str) -> None:
    assert "S6" in ids(
        lint_activation(candidate(compositions=compositions), BUILTIN_SUBAGENTS)
    )


def test_s6_valid_independent_substitution() -> None:
    policy = candidate(
        compositions="review.deep: use slot `reviewer`. If `reviewer` authored the work, substitute slot `execution-reviewer`."
    )
    assert lint_activation(policy, BUILTIN_SUBAGENTS) == ()


@pytest.mark.parametrize(
    "routing",
    [
        "execution: use slot `missing`.",
        "execution: use slot `implementor`.\nexecution: use slot `advisor`.",
        "execution: use no target.",
        "execution: use @missing.",
    ],
)
def test_s7_failure_table_rejects_unknown_targets_and_duplicates(routing: str) -> None:
    assert "S7" in ids(
        lint_catalog(candidate(failure_routing=routing), SHIPPED_CATALOG)
    )


def test_s7_markdown_table_and_unique_classes() -> None:
    routing = "| Class | Target |\n| --- | --- |\n| execution | slot `implementor` |\n| scope | user |"
    assert lint_catalog(candidate(failure_routing=routing), SHIPPED_CATALOG) == ()


@pytest.mark.parametrize("mode", ["standalone", "orchestrated"])
def test_s8_rendered_anchor_tripwire(mode: str) -> None:
    text = "Never run tests, builds, or other verification yourself; never claim a check you did not run."
    if mode == "orchestrated":
        text += " Never edit repo files yourself."
    assert lint_rendered(text, mode) == ()
    assert "S8" in ids(
        lint_rendered(
            text.replace("never claim a check you did not run", "be honest"), mode
        )
    )
    assert "S8" in ids(
        lint_rendered(
            text.replace(
                "Never run tests, builds, or other verification yourself",
                "Run checks yourself",
            ),
            mode,
        )
    )
    wrong = (
        text.replace(" Never edit repo files yourself.", "")
        if mode == "orchestrated"
        else text + " Never edit repo files yourself."
    )
    assert "S8" in ids(lint_rendered(wrong, mode))


def test_s8_rendered_prompt_anchors_both_modes() -> None:
    # The reversion tripwire operates on rendered output, not the skeleton:
    # the mode-variable regions (including the edit prohibition) render per
    # mode at assembly.
    for policy in SHIPPED_PRESETS.values():
        for catalog in (SHIPPED_CATALOG, multi_model_catalog()):
            rendered = render_cli_prompt(
                policy, roster_for(catalog, policy), SystemPrompt.CLI.read()
            )
            assert lint_rendered(rendered, policy.mode) == ()
            assert "$dispatch" not in rendered


def test_s9_only_shipped_mode_names() -> None:
    assert lint_mode("standalone") == ()
    assert lint_mode("orchestrated") == ()
    assert "S9" in ids(lint_mode("please delegate"))
    _, diagnostics = resolve_dispatch_overlay({"dispatch": {"mode": "unknown"}})
    assert "S9" in diagnostics[0]


def test_s10_no_silent_repair_and_persistence_is_atomic(tmp_path: Path) -> None:
    path = tmp_path / "models.toml"
    path.write_text('[dispatch.slots.mechanical]\nrole = "@missing"\n')
    original = path.read_bytes()
    _, diagnostics = resolve_dispatch_overlay({
        "dispatch": {"slots": {"mechanical": {"role": "@missing"}}}
    })
    assert "S10" in diagnostics[0] and "S1" in diagnostics[0]
    result = CatalogStore(path).apply_changes(
        CatalogChanges("", {}, dispatch={"slots": {"mechanical": {"role": "@missing"}}})
    )
    assert isinstance(result, CatalogValidationError)
    assert "S1" in result.message
    assert path.read_bytes() == original
    policy = slot_candidate("mechanical", role="@missing")
    with pytest.raises(DispatchLintError):
        reject_errors(lint_catalog(policy, SHIPPED_CATALOG))
    assert policy.slots["mechanical"].role == "@missing"


@pytest.mark.parametrize("block", ["failure_routing", "compositions", "contrasts"])
@pytest.mark.parametrize(
    "reference", ["slot `missing`", "purpose `missing`", "`@missing`"]
)
def test_r1_curated_references_resolve(block: str, reference: str) -> None:
    policy = candidate(**{block: reference})
    assert "R1" in ids(lint_catalog(policy, SHIPPED_CATALOG))


def test_r1_profiles_are_post_discovery_only() -> None:
    policy = slot_candidate("mechanical", profile="custom")
    policy = policy.model_copy(update={"instructions": "Use profile `custom`."})
    assert lint_catalog(policy, SHIPPED_CATALOG) == ()
    assert "R1" in ids(lint_activation(policy, BUILTIN_SUBAGENTS))
    profiles = {
        **BUILTIN_SUBAGENTS,
        "custom": replace(BUILTIN_SUBAGENTS["worker"], name="custom"),
    }
    assert lint_activation(policy, profiles) == ()


def test_r2_contrasts_need_distinct_slots() -> None:
    policy = candidate(contrasts="Use slot `mechanical` instead of slot `mechanical`.")
    assert "R2" in ids(lint_activation(policy, BUILTIN_SUBAGENTS))
    assert (
        lint_activation(
            candidate(contrasts="Use slot `mechanical` instead of slot `implementor`."),
            BUILTIN_SUBAGENTS,
        )
        == ()
    )


def test_r3_warns_only_for_purpose_mismatch() -> None:
    policy = candidate(
        contrasts="Use slot `mechanical` for purpose `implementation`; use slot `implementor` for purpose `implementation`."
    )
    diagnostics = lint_activation(policy, BUILTIN_SUBAGENTS)
    assert ids(diagnostics) == {"R3"}
    assert all(item.severity == "warning" for item in diagnostics)
    reject_errors(diagnostics)
    policy = candidate(
        contrasts="Use slot `mechanical` for purpose `search`; use slot `implementor` for purpose `implementation`."
    )
    assert lint_activation(policy, BUILTIN_SUBAGENTS) == ()


def test_reference_parser_preserves_duplicate_seats_and_substitutions() -> None:
    refs = parse_references(
        "Use slots `a`, `b`, and `a`, profile `worker`, purpose `review.deep`, `@medium`. If `a` authored the work, substitute slot `b`."
    )
    assert refs.slots == ("a", "b", "a", "b", "a")
    assert refs.profiles == ("worker",)
    assert set(refs.purposes) == {"review.deep"}
    assert refs.roles == ("medium",)


def test_roster_shape_preserves_thinking_not_provider_diversity() -> None:
    resolver = ModelResolver(CatalogSnapshot(SHIPPED_CATALOG, "test"))
    scout = resolver.resolve("@scout")
    worker = resolver.resolve("@worker")
    shape = roster_shape([scout, worker, scout])
    assert shape.single_model
    assert len(shape.bindings) == 2
    other_provider = replace(
        scout, deployment=scout.deployment.model_copy(update={"provider": "elsewhere"})
    )
    assert roster_shape([scout, other_provider]).single_model
    assert len(roster_shape([scout, other_provider]).bindings) == 1
    assert not roster_shape([
        scout,
        replace(worker, base_model="different"),
    ]).single_model
    assert not roster_shape([]).single_model
