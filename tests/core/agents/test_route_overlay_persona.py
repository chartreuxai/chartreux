"""Cross-cutting amendment — the execution-seat overlay persona application.

Design §9 (round-2 decision): the analytical and peer review seats bind the
read-only ``reviewer`` identity with NO overlay, while the execution seat
binds ``coder`` with the ``review-execution`` instruction overlay — a shipped
overlay body resolved through the strict policy resource contract, carried by
the canonical resolver into the resolved launch, and threaded by the manager
bridge into the launch persona instructions. The overlay rides the persona:
it never mutates the identity binding, its revision, or any authority axis.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chartreux.core.agents.identities import ToolPosture, get_builtin_identity
from chartreux.core.agents.identity_launch import (
    OVERLAY_IDS,
    OVERLAYS_DIR,
    InvalidLaunchPurposeError,
    TaskIdentityArgs,
    _overlay_resource_loader,
    resolve_identity_launch,
)
from chartreux.core.agents.launch import LaunchCandidate
from chartreux.core.agents.manager import resolve_identity_candidate
from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
from chartreux.core.config.harness_files import get_harness_files_manager
from chartreux.core.config.models import ModelConfig, ProviderConfig
from chartreux.core.dispatch.lint import lint_composed, reject_errors
from chartreux.core.dispatch.renderer import (
    render_composition_regions,
    render_launch_target_region,
)
from chartreux.core.dispatch.route_models import PackageTier, PurposeRoute, RouteTarget
from chartreux.core.dispatch.routes import shipped_routes
from chartreux.core.launch_types import ResolvedIdentityLaunch
from chartreux.core.llm_models import Backend
from chartreux.core.policy.composition import (
    CompositionInputs,
    ExecutionContext,
    compose_policy,
    fragment_loader,
    policy_resource_loader,
    resolve_identity_facts,
    user_resource_directories,
)
from chartreux.core.policy.manifest import resource_revision
from chartreux.core.policy.models import ResourceKind
from chartreux.core.policy.resources import ResourceLoader, ResourceResolutionError
from chartreux.core.prompts import PROMPTS_DIR
from chartreux.core.subagents import InvalidLaunchToolError
from tests.conftest import build_test_vibe_config
from tests.stubs.fake_config_orchestrator import FakeConfigOrchestrator


def _deep_routes(overlay: str | None = "review-execution") -> dict[str, PurposeRoute]:
    return {
        "review.deep": PurposeRoute(
            targets=(
                RouteTarget(
                    model="small",
                    identity="reviewer",
                    tier=PackageTier.SUBSTANTIVE,
                    seat="analytical",
                ),
                RouteTarget(
                    model="large",
                    identity="reviewer",
                    tier=PackageTier.SUBSTANTIVE,
                    seat="peer",
                ),
                RouteTarget(
                    model="small",
                    identity="coder",
                    tier=PackageTier.SUBSTANTIVE,
                    seat="execution",
                    overlay=overlay,
                ),
            ),
            requires_distinct_models=2,
        )
    }


def _config(routes: dict[str, PurposeRoute] | None = None) -> ChartreuxConfigSchema:
    provider = ProviderConfig(
        name="provider", api_base="https://example.test", backend=Backend.GENERIC
    )
    return build_test_vibe_config(
        providers=[provider],
        models=[
            ModelConfig(
                name="small",
                alias="small",
                provider="provider",
                thinking="off",
                supported_thinking_levels=["off", "low", "high"],
            ),
            ModelConfig(
                name="large",
                alias="large",
                provider="provider",
                thinking="high",
                supported_thinking_levels=["off", "high"],
            ),
        ],
        orchestration={"routes": routes if routes is not None else _deep_routes()},
    )


def _overlay_body() -> str:
    resolved = _overlay_resource_loader().resolve(
        ResourceKind.OVERLAY, "review-execution"
    )
    assert resolved.content is not None
    return resolved.content


def _expected_identity_revision(name: str) -> str:
    """The manifest-compatible identity revision, computed independently."""
    loader = ResourceLoader(
        {ResourceKind.IDENTITY: Path("/nonexistent-route-overlay-check")},
        {ResourceKind.IDENTITY: {name: PROMPTS_DIR / "identities" / f"{name}.md"}},
    )
    content = loader.resolve(ResourceKind.IDENTITY, name).content
    assert content is not None
    return resource_revision(content)


def _resolve(
    args: dict[str, object], config: ChartreuxConfigSchema
) -> ResolvedIdentityLaunch:
    return resolve_identity_launch(
        TaskIdentityArgs.model_validate(args),
        config=config,
        authorized_tool_names=frozenset({"bash", "read_file"}),
    )


def _candidate(
    args: dict[str, object], config: ChartreuxConfigSchema
) -> LaunchCandidate:
    return resolve_identity_candidate(
        TaskIdentityArgs.model_validate(args),
        parent_orchestrator=FakeConfigOrchestrator(config),
        tool_inventory={"bash": object(), "read_file": object()},
        authorized_tool_names=frozenset({"bash", "read_file"}),
    )


def _staged_identity_posture(candidate: LaunchCandidate) -> ToolPosture:
    identity = get_builtin_identity(
        candidate.orchestrator.config.agent.identity.removeprefix("builtin:")
    )
    assert identity is not None
    return identity.personal_tools


# ---------------------------------------------------------------------------
# The shipped overlay body (policy resource layer)
# ---------------------------------------------------------------------------


def test_shipped_overlay_body_resolves_through_the_strict_loader() -> None:
    loader = _overlay_resource_loader()
    assert OVERLAY_IDS == ("review-execution",)
    for reference in ("review-execution", "builtin:review-execution"):
        resolved = loader.resolve(ResourceKind.OVERLAY, reference)
        assert resolved.source == "builtin"
        assert resolved.content is not None
        assert resolved.path is not None
        assert resolved.path.parent == OVERLAYS_DIR
        assert len(resolved.content.encode("utf-8")) <= ResourceLoader.MAX_BYTES
    # A shipped overlay name cannot be shadowed from the user namespace.
    with pytest.raises(ResourceResolutionError, match="reserved built-in"):
        loader.resolve(ResourceKind.OVERLAY, "user:review-execution")


def test_shipped_overlay_body_is_reviewer_derived_and_grants_nothing() -> None:
    body = _overlay_body()
    for anchor in (
        "execution seat",
        "check-only verification commands",
        "not run, never implied",
        "do not hide failed checks",
        "`file:line` citation",
        "read it in full rather than only its diff",
        "grants no new permissions",
        "widens no tool authority",
    ):
        assert anchor.lower() in body.lower(), anchor


# ---------------------------------------------------------------------------
# The canonical resolver carries the resolved overlay (identity_launch)
# ---------------------------------------------------------------------------


def test_execution_seat_resolves_coder_with_the_overlay_body() -> None:
    resolved = _resolve(
        {"task": "review it", "purpose": "review.deep", "seat": "execution"}, _config()
    )
    assert resolved.identity == "coder"
    assert resolved.overlay == "review-execution"
    assert resolved.overlay_body == _overlay_body()
    # The identity binding and its revision are untouched by the overlay.
    assert resolved.identity_revision == _expected_identity_revision("coder")
    # The posture comes from the resolved identity only: the execution seat is
    # the executing coder (all granted) — the overlay text widens nothing.
    assert resolved.authority.personal_posture is ToolPosture.ALL_GRANTED
    assert resolved.authority.delegation_posture is ToolPosture.ALL_GRANTED


@pytest.mark.parametrize("seat", ["analytical", "peer"])
def test_read_only_seats_resolve_reviewer_with_no_overlay(seat: str) -> None:
    resolved = _resolve(
        {"task": "review it", "purpose": "review.deep", "seat": seat}, _config()
    )
    assert resolved.identity == "reviewer"
    assert resolved.overlay is None
    assert resolved.overlay_body is None
    assert resolved.authority.personal_posture is ToolPosture.NON_MUTATING
    assert resolved.authority.delegation_posture is ToolPosture.NON_MUTATING


def test_direct_launches_carry_no_overlay() -> None:
    resolved = _resolve({"task": "do work"}, _config())
    assert resolved.purpose is None
    assert resolved.overlay is None
    assert resolved.overlay_body is None


def test_unknown_route_overlay_fails_loudly() -> None:
    with pytest.raises(InvalidLaunchPurposeError) as excinfo:
        _resolve(
            {"task": "review it", "purpose": "review.deep", "seat": "execution"},
            _config(routes=_deep_routes(overlay="ghost-overlay")),
        )
    assert "ghost-overlay" in str(excinfo.value)
    assert excinfo.value.field == "purpose"


# ---------------------------------------------------------------------------
# The manager bridge threads the overlay into the persona (design §9)
# ---------------------------------------------------------------------------


def test_execution_seat_launch_carries_overlay_instructions() -> None:
    candidate = _candidate(
        {"task": "review it", "purpose": "review.deep", "seat": "execution"}, _config()
    )
    # The overlay rides the persona instructions (the frozen persona/bundle
    # content), not the identity binding.
    assert candidate.persona.instructions == _overlay_body()
    assert candidate.orchestrator.config.agent.identity == "coder"
    assert _staged_identity_posture(candidate) is ToolPosture.ALL_GRANTED


def test_execution_seat_keeps_additive_instructions_ahead_of_the_overlay() -> None:
    candidate = _candidate(
        {
            "task": "review it",
            "purpose": "review.deep",
            "seat": "execution",
            "config": {"instructions": "Additive per-launch instructions."},
        },
        _config(),
    )
    assert candidate.persona.instructions == (
        "Additive per-launch instructions.\n\n" + _overlay_body()
    )


@pytest.mark.parametrize("seat", ["analytical", "peer"])
def test_read_only_seat_launches_carry_reviewer_and_no_overlay_instructions(
    seat: str,
) -> None:
    candidate = _candidate(
        {"task": "review it", "purpose": "review.deep", "seat": seat}, _config()
    )
    assert candidate.orchestrator.config.agent.identity == "reviewer"
    assert candidate.persona.instructions is None
    assert _staged_identity_posture(candidate) is ToolPosture.NON_MUTATING


def test_direct_launch_persona_carries_no_overlay_instructions() -> None:
    candidate = _candidate({"task": "do work"}, _config())
    assert candidate.persona.instructions is None
    assert candidate.orchestrator.config.agent.identity == "coder"


# ---------------------------------------------------------------------------
# The overlay does not widen authority (posture from the identity only)
# ---------------------------------------------------------------------------


def test_overlay_does_not_widen_launch_authority() -> None:
    # The parent's delegation ceiling still binds the execution seat: the
    # overlay's check-running instructions cannot admit a tool the parent
    # never delegated.
    args = {
        "task": "review it",
        "purpose": "review.deep",
        "seat": "execution",
        "config": {"enabled_tools": ["edit"]},
    }
    with pytest.raises(InvalidLaunchToolError):
        resolve_identity_candidate(
            TaskIdentityArgs.model_validate(args),
            parent_orchestrator=FakeConfigOrchestrator(_config()),
            tool_inventory={"bash": object(), "read_file": object(), "edit": object()},
            authorized_tool_names=frozenset({"bash", "read_file"}),
        )


# ---------------------------------------------------------------------------
# The overlay joins the composed child policy (the frozen bundle contract)
# ---------------------------------------------------------------------------


def _fragment_body(name: str) -> str:
    resolved = fragment_loader().resolve(ResourceKind.IDENTITY, name)
    assert resolved.content is not None
    return resolved.content


def test_overlay_joins_the_composed_child_policy_and_passes_lint() -> None:
    """The overlay rides the persona into the composition: it renders inside
    the identity-behavior step as harness-owned policy content, and the
    composed coder + bounded-child policy passes the S8 lint with the overlay
    present (no residual-ceremony or authority leak).
    """
    loader = policy_resource_loader(
        user_resource_directories(get_harness_files_manager())
    )
    facts = resolve_identity_facts("coder")
    inputs = CompositionInputs(
        identity=loader.resolve(ResourceKind.IDENTITY, "coder"),
        workflow=loader.resolve(ResourceKind.WORKFLOW, "minimal"),
        strategy=loader.resolve(ResourceKind.STRATEGY, "none"),
        identity_facts=facts,
        execution_context=ExecutionContext.BOUNDED_CHILD,
        universal=_fragment_body("universal"),
        context_fragment=_fragment_body("bounded_child"),
        tool_semantics=_fragment_body("tool_semantics"),
        communication=_fragment_body("communication"),
        launch_targets=render_launch_target_region(
            dict(shipped_routes()), "test-model"
        ),
        regions=render_composition_regions(
            delegation=facts.delegation if facts is not None else None
        ),
        model_info=None,
        skills=None,
        scratchpad=None,
        environment=None,
        project_context=None,
        commit_signature=None,
        instruction_overlay=_overlay_body(),
        agents_documents=(),
        current_date="2000-01-01 (Saturday)",
    )
    result = compose_policy(inputs)
    behavior = next(step for step in result.steps if step.name == "identity_behavior")
    overlay_text = _overlay_body()
    assert overlay_text in behavior.content
    assert behavior.content.index("Delegation posture") < behavior.content.index(
        overlay_text
    )
    assert overlay_text in result.policy_text
    reject_errors(
        lint_composed(
            result.policy_text,
            identity_facts=result.identity_facts,
            workflow=result.workflow.id,
        )
    )
