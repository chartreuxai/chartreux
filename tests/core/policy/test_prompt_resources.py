from __future__ import annotations

from pathlib import Path
import re

import pytest

from chartreux.core.agents.identities import Delegation, get_builtin_identity
from chartreux.core.policy.models import ResourceKind
from chartreux.core.policy.resources import ResourceLoader, ResourceResolutionError
from chartreux.core.prompts import PROMPTS_DIR, UtilityPrompt

IDENTITIES_DIR = PROMPTS_DIR / "identities"
FRAGMENTS_DIR = PROMPTS_DIR / "fragments"
WORKFLOWS_DIR = PROMPTS_DIR / "workflows"
STRATEGIES_DIR = PROMPTS_DIR / "orchestration_strategies"

IDENTITY_IDS: tuple[str, ...] = (
    "coder",
    "orchestrator",
    "researcher",
    "reviewer",
    "advisor",
)
FRAGMENT_IDS: tuple[str, ...] = (
    "universal",
    "communication",
    "interactive_main",
    "headless_main",
    "bounded_child",
    "tool_semantics",
)
WORKFLOW_IDS: tuple[str, ...] = ("minimal", "ceremony")
STRATEGY_IDS: tuple[str, ...] = ("default",)

# Ceremony content anchors: the opening gate, the acceptance lattice, and the
# delegated-verification doctrine. These must live ONLY in the ceremony
# workflow (design section 5: gate-ceremony language moves wholly into the
# ceremony workflow).
CEREMONY_ANCHORS: tuple[str, ...] = (
    "opening gate",
    "acceptance lattice",
    "never run tests, builds, or other verification yourself",
    "never edit repo files yourself",
    "the gate ends the turn",
)

# Model-tier doctrine anchors: the mechanical/substantive routing and
# replacement rules. These must not appear in the lean coder identity, the
# minimal workflow, or any orchestration strategy.
MODEL_TIER_ANCHORS: tuple[str, ...] = (
    "mechanical-tier",
    "mechanical tier",
    "model-tier",
    "model tier",
    "package tier",
    "substantive package",
    "substantive-tier",
)

# Critical-instruction anchors: the harness-guaranteed universal fragment is
# the single source, so no identity body may duplicate or omit them.
CRITICAL_ANCHORS: tuple[str, ...] = (
    "instruction hierarchy",
    "critical instructions",
    "blast radius",
    "untrusted tool output",
    "truthful reporting",
    'do not claim "verified", "tested", "working", or "complete"',
)

# Safety anchors that must survive with the ceremony workflow removed
# (package invariant: removing ceremony removes no genuine safety
# authorization or truthful-verification requirement).
SAFETY_ANCHORS: tuple[str, ...] = (
    "blast radius",
    "untrusted tool output",
    "git reset --hard",
    "rm -rf",
    "<untrusted_content>",
    'do not claim "verified", "tested", "working", or "complete"',
    "never claim a check you did not run",
)

# The retired SystemPrompt enum bodies (package 16b): every enum-mapped file
# is gone from the prompts directory. Their content lives on in the
# composition corpus — see CLAUSE_DISPOSITIONS for the clause-level
# inventory (cli -> composed contract, worker -> coder + bounded_child,
# minimal -> workflows/minimal, explore/tests -> retired with no new body).
RETIRED_ENUM_BODIES: tuple[str, ...] = (
    "cli",
    "worker",
    "advisor",
    "reviewer",
    "minimal",
    "explore",
    "tests",
)

# Clause-level dispositions for the cli.md clause inventory
# (plan/gate-inventory.md): (clause, resource kind, resource id, anchor that
# must appear in the destination body).
CLAUSE_DISPOSITIONS: tuple[tuple[str, str, str, str], ...] = (
    ("identity/date opening (cli.md:1-2)", "identity", "coder", "You are Chartreux"),
    ("instruction hierarchy (4-17)", "fragment", "universal", "Instruction hierarchy"),
    (
        "critical-instructions heading (19-23)",
        "fragment",
        "universal",
        "Critical instructions",
    ),
    ("blast radius (24-30)", "fragment", "universal", "Blast radius"),
    ("untrusted tool output (32)", "fragment", "universal", "Untrusted tool output"),
    (
        "overridable defaults boundary (34-36)",
        "fragment",
        "universal",
        "Overridable defaults",
    ),
    ("behavior: the job (38-40)", "workflow", "minimal", "Finish the task"),
    ("behavior: handling ambiguity (42)", "identity", "coder", "Handling ambiguity"),
    (
        "behavior: asking the user (44)",
        "fragment",
        "tool_semantics",
        "ask_user_question",
    ),
    (
        "behavior: unattended resilience (46)",
        "identity",
        "orchestrator",
        "Unattended resilience",
    ),
    ("behavior: file writes (48-54)", "identity", "coder", "File writes"),
    ("behavior: non-code requests (56)", "identity", "coder", "Non-code requests"),
    ("read before you act (58-66)", "identity", "coder", "Read before you act"),
    (
        "read-before-act dispatch reading (66)",
        "identity",
        "orchestrator",
        "$dispatch_reading",
    ),
    ("change minimally (68-79)", "identity", "coder", "Change minimally"),
    (
        "prove it worked: truthful core (81-87)",
        "fragment",
        "universal",
        "Truthful reporting",
    ),
    (
        "prove it worked: delegated verification (81-87)",
        "workflow",
        "ceremony",
        "never run tests, builds, or other verification yourself",
    ),
    (
        "prove it worked: direct verification (81-87)",
        "workflow",
        "minimal",
        "run the relevant checks yourself",
    ),
    ("stop when stuck (89-93)", "identity", "coder", "Stop when stuck"),
    ("workflow gates: opening (95-99)", "workflow", "ceremony", "Opening gate"),
    (
        "workflow gates: acceptance lattice (101-105)",
        "workflow",
        "ceremony",
        "Acceptance lattice",
    ),
    ("numbered workflow (107-112)", "workflow", "ceremony", "Classify"),
    (
        "numbered workflow: implement step (111)",
        "workflow",
        "ceremony",
        "$dispatch_implement",
    ),
    ("delegated lifecycle tools (114)", "identity", "orchestrator", "check_agents"),
    (
        "delegation protocol identity placeholder (116-120)",
        "identity",
        "orchestrator",
        "$dispatch_identity",
    ),
    (
        "delegation tool policy placeholder (120)",
        "identity",
        "orchestrator",
        "$dispatch_tools",
    ),
    (
        "failure-time replacement (124)",
        "identity",
        "orchestrator",
        "Failure-time replacement",
    ),
    (
        "disclosure placeholder (126)",
        "identity",
        "orchestrator",
        "$dispatch_disclosure",
    ),
    ("routing placeholder (128)", "identity", "orchestrator", "$dispatch_routing"),
    ("shell (130-132)", "fragment", "tool_semantics", "Shell"),
    ("communication voice (134-137)", "fragment", "communication", "Voice"),
    ("communication length (139)", "fragment", "communication", "Length"),
    (
        "open/during/close (141-145)",
        "fragment",
        "communication",
        "state intent before acting",
    ),
    ("response format (147-152)", "fragment", "communication", "Structure first"),
    ("what not to do (154-163)", "fragment", "communication", "What not to do"),
    (
        "truthfulness sentence (161)",
        "fragment",
        "universal",
        'Do not claim "verified", "tested", "working", or "complete"',
    ),
    ("no emoji (164)", "fragment", "communication", "No emoji"),
    ("worker: subagent contract", "fragment", "bounded_child", "bounded child"),
    ("worker: operating rules", "fragment", "bounded_child", "Child operating rules"),
    ("worker: output format", "fragment", "bounded_child", "files_touched"),
    ("worker: role guidance", "identity", "coder", "Role guidance"),
    ("advisor: role + output format", "identity", "advisor", "Denials encountered"),
    ("reviewer: role + output format", "identity", "reviewer", "Review Report"),
    (
        "reviewer: read-only posture, no shell verification",
        "identity",
        "reviewer",
        "do not hide failed checks",
    ),
    ("researcher: read-only investigation role", "identity", "researcher", "read-only"),
    ("minimal.md: goal/how/if-stuck", "workflow", "minimal", "If stuck"),
    (
        "explore: structure-first conventions",
        "fragment",
        "communication",
        "Structure first",
    ),
    ("strategy: packaging preferences", "strategy", "default", "Brief discipline"),
)


def _builtin_map(directory: Path, ids: tuple[str, ...]) -> dict[str, Path]:
    return {name: directory / f"{name}.md" for name in ids}


@pytest.fixture
def user_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "user-resources"
    directory.mkdir()
    return directory


@pytest.fixture
def identity_loader(user_dir: Path) -> ResourceLoader:
    return ResourceLoader(
        {ResourceKind.IDENTITY: user_dir},
        {ResourceKind.IDENTITY: _builtin_map(IDENTITIES_DIR, IDENTITY_IDS)},
    )


@pytest.fixture
def workflow_loader(user_dir: Path) -> ResourceLoader:
    return ResourceLoader(
        {ResourceKind.WORKFLOW: user_dir},
        {ResourceKind.WORKFLOW: _builtin_map(WORKFLOWS_DIR, WORKFLOW_IDS)},
    )


@pytest.fixture
def strategy_loader(user_dir: Path) -> ResourceLoader:
    return ResourceLoader(
        {ResourceKind.STRATEGY: user_dir},
        {ResourceKind.STRATEGY: _builtin_map(STRATEGIES_DIR, STRATEGY_IDS)},
    )


@pytest.fixture
def fragment_loader() -> ResourceLoader:
    # Fragments are harness-owned composition inputs, not selectable
    # resources: they are registered as loader builtins so they resolve with
    # builtin semantics and cannot be shadowed by user resources.
    return ResourceLoader(
        {}, {ResourceKind.IDENTITY: _builtin_map(FRAGMENTS_DIR, FRAGMENT_IDS)}
    )


@pytest.fixture
def bodies(
    identity_loader: ResourceLoader,
    workflow_loader: ResourceLoader,
    strategy_loader: ResourceLoader,
    fragment_loader: ResourceLoader,
) -> dict[str, str]:
    """Every shipped resource body, keyed ``<kind>/<id>``."""
    collected: dict[str, str] = {}
    for name in IDENTITY_IDS:
        resolved = identity_loader.resolve(ResourceKind.IDENTITY, name)
        assert resolved.content is not None
        collected[f"identity/{name}"] = resolved.content
    for name in WORKFLOW_IDS:
        resolved = workflow_loader.resolve(ResourceKind.WORKFLOW, name)
        assert resolved.content is not None
        collected[f"workflow/{name}"] = resolved.content
    for name in STRATEGY_IDS:
        resolved = strategy_loader.resolve(ResourceKind.STRATEGY, name)
        assert resolved.content is not None
        collected[f"strategy/{name}"] = resolved.content
    for name in FRAGMENT_IDS:
        resolved = fragment_loader.resolve(ResourceKind.IDENTITY, name)
        assert resolved.content is not None
        collected[f"fragment/{name}"] = resolved.content
    return collected


def _assert_contains(body: str, anchor: str) -> None:
    assert anchor.lower() in body.lower(), f"missing anchor: {anchor!r}"


def _assert_lacks(body: str, anchor: str) -> None:
    assert anchor.lower() not in body.lower(), f"unexpected anchor: {anchor!r}"


# --- Loader contract -------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "directory", "ids"),
    [
        (ResourceKind.IDENTITY, IDENTITIES_DIR, IDENTITY_IDS),
        (ResourceKind.WORKFLOW, WORKFLOWS_DIR, WORKFLOW_IDS),
        (ResourceKind.STRATEGY, STRATEGIES_DIR, STRATEGY_IDS),
    ],
)
def test_shipped_resources_resolve_as_builtins(
    kind: ResourceKind, directory: Path, ids: tuple[str, ...], user_dir: Path
) -> None:
    loader = ResourceLoader({kind: user_dir}, {kind: _builtin_map(directory, ids)})
    for name in ids:
        bare = loader.resolve(kind, name)
        qualified = loader.resolve(kind, f"builtin:{name}")
        assert bare.source == "builtin"
        assert qualified.source == "builtin"
        assert bare.content is not None
        assert bare.content == qualified.content
        assert bare.path is not None
        assert bare.path.parent == directory
        assert len(bare.content.encode("utf-8")) <= ResourceLoader.MAX_BYTES


@pytest.mark.parametrize(
    ("kind", "directory", "ids"),
    [
        (ResourceKind.IDENTITY, IDENTITIES_DIR, IDENTITY_IDS),
        (ResourceKind.WORKFLOW, WORKFLOWS_DIR, WORKFLOW_IDS),
        (ResourceKind.STRATEGY, STRATEGIES_DIR, STRATEGY_IDS),
    ],
)
def test_shipped_resource_names_are_reserved_from_user_namespace(
    kind: ResourceKind, directory: Path, ids: tuple[str, ...], user_dir: Path
) -> None:
    loader = ResourceLoader({kind: user_dir}, {kind: _builtin_map(directory, ids)})
    for name in ids:
        (user_dir / f"{name}.md").write_text("shadow", encoding="utf-8")
        with pytest.raises(ResourceResolutionError, match="reserved built-in"):
            loader.resolve(kind, f"user:{name}")


def test_fragments_resolve_as_unshadowable_builtins(
    fragment_loader: ResourceLoader, user_dir: Path
) -> None:
    for name in FRAGMENT_IDS:
        resolved = fragment_loader.resolve(ResourceKind.IDENTITY, name)
        assert resolved.source == "builtin"
        assert resolved.content is not None
        assert resolved.path is not None
        assert resolved.path.parent == FRAGMENTS_DIR
    with pytest.raises(ResourceResolutionError, match="reserved built-in"):
        fragment_loader.resolve(ResourceKind.IDENTITY, "user:universal")


def test_strategy_none_resolves_concretely_without_loading_a_body(
    strategy_loader: ResourceLoader,
) -> None:
    resolved = strategy_loader.resolve(ResourceKind.STRATEGY, "none")
    assert resolved.id == "none"
    assert resolved.source == "builtin"
    assert resolved.content is None
    assert resolved.path is None


def test_strategy_validation_rejects_none_delegation_with_a_strategy(
    strategy_loader: ResourceLoader,
) -> None:
    # A future identity with delegation=none plus an explicit strategy is a
    # configuration error (design section 3), not a silent no-op.
    with pytest.raises(ResourceResolutionError, match="delegation=none"):
        strategy_loader.validate_strategy("none", "builtin:default")
    # The orchestrator's required delegation is compatible with strategy=none.
    strategy_loader.validate_strategy("required", "none")


# --- Enum retirement -------------------------------------------------------


def test_system_prompt_enum_is_retired_from_the_loader_surface() -> None:
    from chartreux.core import prompts

    assert not hasattr(prompts, "SystemPrompt")
    assert not hasattr(prompts, "load_system_prompt")


def test_retired_enum_bodies_are_gone_from_the_prompts_directory() -> None:
    for retired in RETIRED_ENUM_BODIES:
        assert not (PROMPTS_DIR / f"{retired}.md").exists(), retired


def test_retired_enum_entries_have_no_new_resource_body(
    identity_loader: ResourceLoader,
    workflow_loader: ResourceLoader,
    strategy_loader: ResourceLoader,
) -> None:
    # explore/tests retired with no successor body: they resolve to nothing
    # in any loader. (The other five enum entries' content lives on in the
    # composition corpus — see CLAUSE_DISPOSITIONS.)
    for retired in ("explore", "tests"):
        with pytest.raises(ResourceResolutionError):
            identity_loader.resolve(ResourceKind.IDENTITY, f"builtin:{retired}")
        with pytest.raises(ResourceResolutionError):
            workflow_loader.resolve(ResourceKind.WORKFLOW, f"builtin:{retired}")
        with pytest.raises(ResourceResolutionError):
            strategy_loader.resolve(ResourceKind.STRATEGY, f"builtin:{retired}")


def test_utility_prompts_remain_a_separate_untouched_mechanism() -> None:
    # Utility prompts (compact, session_title, ...) are not part of the
    # identity/workflow/strategy corpus; they load through the strict
    # loader branch and survive the enum retirement untouched.
    assert len(list(UtilityPrompt)) == 8
    for prompt in UtilityPrompt:
        assert prompt.read()


# --- Clause dispositions ---------------------------------------------------


def test_every_cli_clause_has_a_disposition_in_the_corpus(
    identity_loader: ResourceLoader,
    workflow_loader: ResourceLoader,
    strategy_loader: ResourceLoader,
    fragment_loader: ResourceLoader,
) -> None:
    loaders = {
        "identity": (identity_loader, ResourceKind.IDENTITY),
        "workflow": (workflow_loader, ResourceKind.WORKFLOW),
        "strategy": (strategy_loader, ResourceKind.STRATEGY),
        "fragment": (fragment_loader, ResourceKind.IDENTITY),
    }
    for clause, kind, resource_id, anchor in CLAUSE_DISPOSITIONS:
        loader, resource_kind = loaders[kind]
        resolved = loader.resolve(resource_kind, resource_id)
        assert resolved.content is not None, clause
        _assert_contains(resolved.content, anchor)


def test_identity_openings_are_identity_first_at_character_zero(
    bodies: dict[str, str],
) -> None:
    role_words = {
        "coder": "coding agent",
        "orchestrator": "orchestrator",
        "researcher": "researcher",
        "reviewer": "reviewer",
        "advisor": "advisor",
    }
    for name, role_word in role_words.items():
        body = bodies[f"identity/{name}"]
        first_line = next(line for line in body.splitlines() if line.strip())
        assert role_word in first_line.lower(), name
        # No identity body may start with another identity's opening.
        assert body.lstrip().startswith(first_line)


# --- Universal fragment mandate --------------------------------------------


def test_universal_fragment_is_mandatory_for_every_identity(
    identity_loader: ResourceLoader, fragment_loader: ResourceLoader, user_dir: Path
) -> None:
    universal = fragment_loader.resolve(ResourceKind.IDENTITY, "universal")
    assert universal.content is not None
    _assert_contains(universal.content, "harness-guaranteed")
    _assert_contains(universal.content, "can omit it")
    for anchor in CRITICAL_ANCHORS:
        _assert_contains(universal.content, anchor)

    # Resolving any built-in identity composes the critical instructions in.
    for name in IDENTITY_IDS:
        resolved = identity_loader.resolve(ResourceKind.IDENTITY, name)
        assert resolved.content is not None
        composed = f"{universal.content}\n\n{resolved.content}"
        for anchor in CRITICAL_ANCHORS:
            _assert_contains(composed, anchor)
        # The critical block has a single source: no identity body
        # duplicates it, so a custom identity cannot drop it either.
        _assert_lacks(resolved.content, "blast radius")
        _assert_lacks(resolved.content, "## critical instructions")

    # A custom identity cannot omit the universal fragment either: the
    # composition contract adds it regardless of the identity body.
    (user_dir / "custom.md").write_text(
        "You are a custom identity. Do the thing.", encoding="utf-8"
    )
    custom = identity_loader.resolve(ResourceKind.IDENTITY, "user:custom")
    assert custom.content is not None
    for anchor in CRITICAL_ANCHORS:
        _assert_lacks(custom.content, anchor)
        _assert_contains(f"{universal.content}\n\n{custom.content}", anchor)


# --- Ceremony scoping ------------------------------------------------------


def test_ceremony_workflow_carries_all_ceremony(bodies: dict[str, str]) -> None:
    ceremony = bodies["workflow/ceremony"]
    for anchor in (
        "opening gate",
        "acceptance lattice",
        "never run tests, builds, or other verification yourself",
        "$dispatch_implement",
        "classify",
        "the gate ends the turn",
        "verify",
    ):
        _assert_contains(ceremony, anchor)


def test_ceremony_content_is_absent_from_every_other_shipped_body(
    bodies: dict[str, str],
) -> None:
    for key, body in bodies.items():
        if key == "workflow/ceremony":
            continue
        for anchor in CEREMONY_ANCHORS:
            _assert_lacks(body, anchor)


def test_coder_and_minimal_carry_no_ceremony_or_model_tier_doctrine(
    bodies: dict[str, str],
) -> None:
    for key in ("identity/coder", "workflow/minimal"):
        body = bodies[key]
        for anchor in CEREMONY_ANCHORS + MODEL_TIER_ANCHORS:
            _assert_lacks(body, anchor)
        for word in ("tier", "mechanical", "substantive", "acceptance", "ceremony"):
            _assert_lacks(body, word)
        assert not re.search(r"\bgates?\b", body, re.IGNORECASE), key


def test_model_tier_doctrine_is_absent_outside_the_orchestrator_identity(
    bodies: dict[str, str],
) -> None:
    # The orchestrator identity legitimately carries the condensed
    # failure-replacement pointer (validated f4cb7085 delta); every other
    # shipped body is tier-doctrine free.
    for key, body in bodies.items():
        if key == "identity/orchestrator":
            continue
        for anchor in MODEL_TIER_ANCHORS:
            _assert_lacks(body, anchor)


# --- Orchestrator identity -------------------------------------------------


def test_orchestrator_identity_carries_the_validated_deltas(
    bodies: dict[str, str],
) -> None:
    orchestrator = bodies["identity/orchestrator"]
    # Identity-first opening (validated delta): the orchestrator identity
    # statement is at character zero, not a doer opening.
    first_line = next(line for line in orchestrator.splitlines() if line.strip())
    assert "orchestrator" in first_line.lower()
    # Delegation obligation lives in the identity, never in a strategy.
    _assert_contains(orchestrator, "delegation is required")
    # Narrowed gate exemption clarification (validated delta).
    _assert_contains(orchestrator, "who performs")
    # Skills ride in the brief (validated delta).
    _assert_contains(orchestrator, "ride in the brief")
    # Closing re-anchor (validated delta).
    _assert_contains(orchestrator, "when in doubt, dispatch")
    # Unattended resilience and the condensed failure-replacement pointer.
    _assert_contains(orchestrator, "unattended resilience")
    _assert_contains(orchestrator, "failure-time replacement")
    _assert_contains(orchestrator, "check_agents")
    # The renderer-injected dispatch regions are carried by the identity.
    for placeholder in (
        "$dispatch_reading",
        "$dispatch_identity",
        "$dispatch_tools",
        "$dispatch_disclosure",
        "$dispatch_routing",
    ):
        _assert_contains(orchestrator, placeholder)


def test_orchestrator_minimal_none_resolves_with_no_missing_required_pieces(
    identity_loader: ResourceLoader,
    workflow_loader: ResourceLoader,
    strategy_loader: ResourceLoader,
    fragment_loader: ResourceLoader,
) -> None:
    identity = get_builtin_identity("orchestrator")
    assert identity is not None
    assert identity.delegation == Delegation.REQUIRED

    # Every required piece of the orchestrator + minimal + strategy=none
    # composition resolves through the strict loader.
    resolved_identity = identity_loader.resolve(ResourceKind.IDENTITY, "orchestrator")
    resolved_workflow = workflow_loader.resolve(ResourceKind.WORKFLOW, "minimal")
    resolved_strategy = strategy_loader.resolve(ResourceKind.STRATEGY, "none")
    assert resolved_identity.content is not None
    assert resolved_workflow.content is not None
    assert resolved_strategy.content is None  # none resolves concretely
    for fragment in (
        "universal",
        "interactive_main",
        "headless_main",
        "tool_semantics",
        "communication",
    ):
        resolved = fragment_loader.resolve(ResourceKind.IDENTITY, fragment)
        assert resolved.content is not None, fragment
    # The child side of the same session resolves too.
    child = fragment_loader.resolve(ResourceKind.IDENTITY, "bounded_child")
    assert child.content is not None
    strategy_loader.validate_strategy(identity.delegation, "none")


# --- Reviewer / bounded child / strategy -----------------------------------


def test_reviewer_identity_makes_no_shell_verification_promise(
    bodies: dict[str, str],
) -> None:
    reviewer = bodies["identity/reviewer"]
    # The old reviewer.md:28 promise ("run declared check-only project
    # verification commands") is removed by the honest read-only reviewer
    # posture (design section 9).
    for anchor in ("run declared check-only", "check-only", "bash"):
        _assert_lacks(reviewer, anchor)
    # The truthfulness requirement stays.
    for anchor in ("not run", "do not hide failed checks", "never modify"):
        _assert_contains(reviewer, anchor)
    _assert_contains(reviewer, "read-only")


def test_bounded_child_returns_blockers_and_never_restarts_the_parent_workflow(
    bodies: dict[str, str],
) -> None:
    child = bodies["fragment/bounded_child"]
    for anchor in (
        "structured blocker",
        "never restart the parent",
        "re-invoke its ceremony",
        "do not ask the user questions",
        "do not spawn",
        "denial record",
        "touched-path",
        "non-interactive",
    ):
        _assert_contains(child, anchor)
    for anchor in CEREMONY_ANCHORS:
        _assert_lacks(child, anchor)


def test_default_strategy_is_packaging_and_routing_preferences_only(
    bodies: dict[str, str],
) -> None:
    strategy = bodies["strategy/default"]
    for anchor in ("brief", "parallel", "reuse"):
        _assert_contains(strategy, anchor)
    # No ceremony, no model-tier doctrine, no authority claims, and no
    # delegation obligation (that lives in the orchestrator identity).
    for anchor in CEREMONY_ANCHORS + MODEL_TIER_ANCHORS:
        _assert_lacks(strategy, anchor)
    for anchor in (
        "you may",
        "you must",
        "you are authorized",
        "never edit repo files yourself",
        "delegation is required",
        "you are an orchestrator",
        "authority",
    ):
        _assert_lacks(strategy, anchor)
    assert not re.search(r"\byou\b", strategy, re.IGNORECASE)


# --- Invariant: safety survives without ceremony ---------------------------


def test_safety_clauses_do_not_live_only_in_a_ceremony_scoped_body(
    bodies: dict[str, str],
) -> None:
    universal = bodies["fragment/universal"]
    # Every safety clause is retained by the universal fragment, which is
    # composed independently of the selected workflow.
    for anchor in SAFETY_ANCHORS:
        _assert_contains(universal, anchor)

    # The no-ceremony default composition (coder + minimal + shared
    # references, no ceremony workflow) still carries every safety clause.
    default_composition = "\n\n".join(
        bodies[key]
        for key in (
            "identity/coder",
            "fragment/universal",
            "fragment/interactive_main",
            "workflow/minimal",
            "fragment/tool_semantics",
            "fragment/communication",
        )
    )
    for anchor in SAFETY_ANCHORS:
        _assert_contains(default_composition, anchor)
    # And the ceremony workflow adds no safety clause of its own that the
    # default would lack: nothing safety-critical is ceremony-scoped.
    for anchor in SAFETY_ANCHORS:
        assert anchor.lower() in universal.lower()
