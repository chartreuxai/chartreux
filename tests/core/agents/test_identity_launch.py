"""Package 8 — resource-free identity launch resolution (design §7, §9).

Covers the resolved-launches contract (interfaces.md): the canonical grammar
resolves model/deployment, thinking, identity revision, authority, and the
effective package tier BEFORE any backend, MCP client, child log, handle, or
reservation is constructed; the default child is coder with the bounded-child
context and inherits no workflow/strategy; required-delegation child identities
are rejected before resource construction; an explicit model override retains
the route target's tier; a direct unrouted launch uses the identity/purpose
default; and the replacement-capability marking stays a separate axis from the
package tier.
"""

from __future__ import annotations

from dataclasses import is_dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError
import pytest

from chartreux.core.agents.identities import Delegation, ToolPosture
from chartreux.core.agents.identity_launch import (
    DEFAULT_CHILD_IDENTITY,
    DEFAULT_LAUNCH_TIER,
    InvalidChildIdentityError,
    InvalidLaunchModelSelectionError,
    InvalidLaunchPurposeError,
    TaskIdentityArgs,
    resolve_identity_launch,
)
from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
from chartreux.core.dispatch.route_models import (
    PackageTier,
    PurposeRoute,
    ReplacementCapability,
    RouteTarget,
)
from chartreux.core.model_catalog.loader import CatalogSnapshot
from chartreux.core.model_catalog.schema import ModelCatalog
from chartreux.core.policy.composition import ExecutionContext
from chartreux.core.policy.manifest import resource_revision
from chartreux.core.policy.models import ResourceKind
from chartreux.core.policy.resources import ResourceLoader
from chartreux.core.prompts import PROMPTS_DIR
from chartreux.core.recovery_models import PackageTier as RecoveryPackageTier
from tests.conftest import build_test_vibe_config


def _catalog() -> ModelCatalog:
    """A deterministic three-model catalog with two role presets.

    ``alpha-model`` and ``beta-model`` carry price data so the shipped
    cost-tier route defaults resolve (the cheapest capable model is
    ``alpha-model``); ``limited-model`` stays unpriced and is never the
    cost-tier default.
    """
    return ModelCatalog.model_validate({
        "providers": {
            "alpha": {"api_base": "https://alpha.test"},
            "beta": {"api_base": "https://beta.test"},
        },
        "models": {
            "alpha-model": {
                "thinking": "medium",
                "deployments": [
                    {
                        "provider": "alpha",
                        "name": "alpha-wire",
                        "prices": {"input": 1.0, "output": 2.0},
                    }
                ],
            },
            "beta-model": {
                "thinking": "high",
                "deployments": [
                    {
                        "provider": "beta",
                        "name": "beta-wire",
                        "prices": {"input": 3.0, "output": 8.0},
                    }
                ],
            },
            "limited-model": {
                "thinking": "off",
                "deployments": [
                    {
                        "provider": "beta",
                        "name": "limited-wire",
                        "supported_thinking_levels": ["off"],
                    }
                ],
            },
        },
        "roles": {
            "main": {
                "model": "beta-model",
                "thinking": "high",
                "description": "main preset",
            },
            "fast": {
                "model": "alpha-model",
                "thinking": "low",
                "description": "fast preset",
            },
        },
    })


def _config(active_model: str = "@main", **kwargs: Any) -> ChartreuxConfigSchema:
    base = build_test_vibe_config(active_model=active_model, **kwargs)
    return base.attach_catalog_snapshot(
        CatalogSnapshot(_catalog(), "identity-launch-test")
    )


def _resolve(
    args: TaskIdentityArgs | dict[str, Any],
    config: ChartreuxConfigSchema,
    *,
    routes: dict[str, PurposeRoute] | None = None,
    user_prompts_dirs: tuple[Path, ...] = (),
    session_model: str | None = None,
    authorized: frozenset[str] = frozenset({"bash", "read_file"}),
):
    return resolve_identity_launch(
        args,
        config=config,
        authorized_tool_names=authorized,
        session_model=session_model,
        user_prompts_dirs=user_prompts_dirs,
        routes=routes,
    )


def _expected_builtin_revision(name: str) -> str:
    """The manifest-compatible identity revision, computed independently."""
    loader = ResourceLoader(
        {ResourceKind.IDENTITY: Path("/nonexistent-identity-launch-check")},
        {ResourceKind.IDENTITY: {name: PROMPTS_DIR / "identities" / f"{name}.md"}},
    )
    content = loader.resolve(ResourceKind.IDENTITY, name).content
    assert content is not None
    return resource_revision(content)


def _deep_routes() -> dict[str, PurposeRoute]:
    return {
        "review.deep": PurposeRoute(
            targets=(
                RouteTarget(
                    model="alpha-model",
                    identity="reviewer",
                    tier=PackageTier.SUBSTANTIVE,
                    seat="analytical",
                ),
                RouteTarget(
                    model="beta-model",
                    identity="reviewer",
                    tier=PackageTier.SUBSTANTIVE,
                    seat="peer",
                ),
                RouteTarget(
                    model="beta-model",
                    identity="coder",
                    tier=PackageTier.SUBSTANTIVE,
                    seat="execution",
                    overlay="review-execution",
                ),
            ),
            requires_distinct_models=2,
        )
    }


# ---------------------------------------------------------------------------
# The default child: coder, bounded-child context, no inherited bodies
# ---------------------------------------------------------------------------


def test_default_child_is_coder_with_bounded_child_context() -> None:
    resolved = _resolve({"task": "do work"}, _config())
    assert resolved.identity == DEFAULT_CHILD_IDENTITY == "coder"
    assert resolved.execution_context is ExecutionContext.BOUNDED_CHILD
    assert resolved.identity_facts is not None
    assert resolved.identity_facts.id == "coder"
    assert resolved.identity_revision == _expected_builtin_revision("coder")
    assert resolved.identity_revision.startswith("blake2b-128:")


def test_children_never_inherit_the_parent_workflow_or_strategy() -> None:
    config = _config(
        agent={
            "identity": "orchestrator",
            "workflow": "ceremony",
            "orchestration_strategy": "builtin:default",
        }
    )
    resolved = _resolve({"task": "do work"}, config)
    # The child default — not the parent's orchestrator selection.
    assert resolved.identity == "coder"
    assert resolved.workflow is None
    assert resolved.strategy is None
    assert resolved.execution_context is ExecutionContext.BOUNDED_CHILD
    # An explicit child identity inherits no bodies either.
    explicit = _resolve({"task": "do work", "identity": "reviewer"}, config)
    assert explicit.identity == "reviewer"
    assert explicit.workflow is None
    assert explicit.strategy is None
    assert explicit.execution_context is ExecutionContext.BOUNDED_CHILD


# ---------------------------------------------------------------------------
# Child-identity resolution and the required-delegation rejection
# ---------------------------------------------------------------------------


def test_explicit_builtin_identity_is_honored() -> None:
    resolved = _resolve({"task": "x", "identity": "researcher"}, _config())
    assert resolved.identity == "researcher"
    assert resolved.identity_facts is not None
    assert resolved.identity_facts.delegation is Delegation.OPTIONAL
    assert resolved.identity_revision == _expected_builtin_revision("researcher")
    qualified = _resolve({"task": "x", "identity": "builtin:reviewer"}, _config())
    assert qualified.identity == "builtin:reviewer"
    assert qualified.identity_facts is not None
    assert qualified.identity_facts.id == "reviewer"


def test_custom_user_identity_resolves_with_no_harness_known_posture(
    tmp_path: Path,
) -> None:
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "field-notes.md").write_text(
        "# Field notes\n\nYou are a field-notes agent.\n"
    )
    config = _config()
    resolved = _resolve(
        {"task": "x", "identity": "user:field-notes"},
        config,
        user_prompts_dirs=(prompts,),
    )
    assert resolved.identity == "user:field-notes"
    assert resolved.identity_facts is None
    # Decision 8: a custom identity carries no harness-known posture.
    assert resolved.authority.personal_posture is None
    assert resolved.authority.delegation_posture is None
    expected = resource_revision("# Field notes\n\nYou are a field-notes agent.")
    assert resolved.identity_revision == expected
    # A bare custom id resolves through the same strict contract.
    bare = _resolve(
        {"task": "x", "identity": "field-notes"}, config, user_prompts_dirs=(prompts,)
    )
    assert bare.identity == "field-notes"
    assert bare.identity_revision == expected


def test_unknown_custom_identity_fails_with_an_actionable_error(tmp_path: Path) -> None:
    with pytest.raises(InvalidChildIdentityError) as excinfo:
        _resolve(
            {"task": "x", "identity": "user:missing"},
            _config(),
            user_prompts_dirs=(tmp_path,),
        )
    assert "user:missing" in str(excinfo.value)
    assert excinfo.value.field == "identity"


def test_required_delegation_identities_cannot_be_children() -> None:
    config = _config()
    # The structured field.
    with pytest.raises(InvalidChildIdentityError) as structured:
        _resolve({"task": "x", "identity": "orchestrator"}, config)
    message = str(structured.value)
    assert "requires delegation" in message
    assert "coder" in message and "researcher" in message
    assert structured.value.field == "identity"
    # The shorthand identity component.
    with pytest.raises(InvalidChildIdentityError):
        _resolve({"task": "x", "model": "@main:orchestrator"}, config)
    # A route bound to a required-delegation identity.
    routes = {
        "orchestrate": PurposeRoute(
            targets=(
                RouteTarget(
                    model="@main", identity="orchestrator", tier=PackageTier.SUBSTANTIVE
                ),
            )
        )
    }
    with pytest.raises(InvalidChildIdentityError):
        _resolve({"task": "x", "purpose": "orchestrate"}, config, routes=routes)
    # The rejection precedes model resolution: an invalid model does not
    # change the error.
    with pytest.raises(InvalidChildIdentityError):
        _resolve(
            {"task": "x", "identity": "orchestrator", "model": "no-such-model"}, config
        )


def test_required_delegation_rejection_precedes_the_identity_resource_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The rejection happens BEFORE resource construction: the identity
    revision read (the one pure resource read of the resolution) never runs
    for a rejected launch.
    """

    def _forbidden(*_args: Any, **_kwargs: Any) -> str:
        raise AssertionError("identity resource read during a rejected launch")

    monkeypatch.setattr(
        "chartreux.core.agents.identity_launch.resource_revision", _forbidden
    )
    with pytest.raises(InvalidChildIdentityError):
        _resolve({"task": "x", "identity": "orchestrator"}, _config())


# ---------------------------------------------------------------------------
# Route/purpose/seat resolution
# ---------------------------------------------------------------------------


def test_purpose_route_resolution_and_explicit_overrides() -> None:
    config = _config()
    resolved = _resolve({"task": "research", "purpose": "research"}, config)
    assert resolved.identity == "researcher"
    assert resolved.purpose == "research"
    assert resolved.model_expression == "@main"
    assert resolved.base_model == "beta-model"
    # Explicit fields override the route's corresponding defaults.
    overridden = _resolve(
        {
            "task": "research",
            "purpose": "research",
            "model": "@fast",
            "identity": "advisor",
        },
        config,
    )
    assert overridden.identity == "advisor"
    assert overridden.base_model == "alpha-model"


def test_shorthand_components_override_the_route_defaults() -> None:
    resolved = _resolve(
        {"task": "research", "purpose": "research", "model": "@fast:reviewer"},
        _config(),
    )
    assert resolved.identity == "reviewer"
    assert resolved.base_model == "alpha-model"
    assert resolved.purpose == "research"


def test_multi_seat_purpose_requires_and_resolves_a_seat() -> None:
    config = _config()
    routes = _deep_routes()
    with pytest.raises(InvalidLaunchPurposeError) as no_seat:
        _resolve({"task": "x", "purpose": "review.deep"}, config, routes=routes)
    assert "explicit seat" in str(no_seat.value)
    analytical = _resolve(
        {"task": "x", "purpose": "review.deep", "seat": "analytical"},
        config,
        routes=routes,
    )
    assert (analytical.identity, analytical.seat, analytical.base_model) == (
        "reviewer",
        "analytical",
        "alpha-model",
    )
    execution = _resolve(
        {"task": "x", "purpose": "review.deep", "seat": "execution"},
        config,
        routes=routes,
    )
    assert (execution.identity, execution.seat, execution.overlay) == (
        "coder",
        "execution",
        "review-execution",
    )
    with pytest.raises(InvalidLaunchPurposeError):
        _resolve(
            {"task": "x", "purpose": "review.deep", "seat": "chair"},
            config,
            routes=routes,
        )
    with pytest.raises(InvalidLaunchPurposeError):
        _resolve({"task": "x", "purpose": "no-such-purpose"}, config, routes=routes)


def test_shipped_deep_review_reports_missing_model_diversity() -> None:
    with pytest.raises(InvalidLaunchPurposeError) as excinfo:
        _resolve(
            {"task": "x", "purpose": "review.deep", "seat": "analytical"}, _config()
        )
    assert "requires 3 distinct canonical models" in str(excinfo.value)


def test_unavailable_route_model_binding_is_rejected() -> None:
    routes = {
        "broken": PurposeRoute(
            targets=(
                RouteTarget(
                    model="ghost-model", identity="coder", tier=PackageTier.SUBSTANTIVE
                ),
            )
        )
    }
    with pytest.raises(InvalidLaunchPurposeError):
        _resolve({"task": "x", "purpose": "broken"}, _config(), routes=routes)


# ---------------------------------------------------------------------------
# The effective package tier
# ---------------------------------------------------------------------------


def test_explicit_model_override_retains_the_route_target_tier() -> None:
    config = _config()
    # mechanical-edit binds the session model at the MECHANICAL tier; an
    # explicit model override does not silently change the tier.
    resolved = _resolve(
        {"task": "x", "purpose": "mechanical-edit", "model": "beta-model"}, config
    )
    assert resolved.effective_tier is RecoveryPackageTier.MECHANICAL
    assert resolved.base_model == "beta-model"
    substantive = _resolve(
        {"task": "x", "purpose": "implementation", "model": "alpha-model"}, config
    )
    assert substantive.effective_tier is RecoveryPackageTier.SUBSTANTIVE
    assert substantive.base_model == "alpha-model"


def test_direct_unrouted_launch_uses_the_identity_purpose_default() -> None:
    config = _config()
    resolved = _resolve({"task": "do work"}, config)
    assert resolved.identity == "coder"
    assert resolved.effective_tier is DEFAULT_LAUNCH_TIER
    assert resolved.effective_tier is RecoveryPackageTier.SUBSTANTIVE
    assert resolved.model_expression == "@main"
    assert resolved.base_model == "beta-model"
    assert resolved.purpose is None
    assert resolved.seat is None
    assert resolved.overlay is None
    # An explicit identity on a direct launch keeps the default tier.
    researcher = _resolve({"task": "research stuff", "identity": "researcher"}, config)
    assert researcher.identity == "researcher"
    assert researcher.effective_tier is RecoveryPackageTier.SUBSTANTIVE


def test_replacement_capability_is_a_separate_axis_from_the_package_tier() -> None:
    config = _config()
    routes = {
        "dual": PurposeRoute(
            targets=(
                RouteTarget(
                    model="beta-model",
                    identity="coder",
                    # The package-tier axis ...
                    tier=PackageTier.MECHANICAL,
                    # ... and the capability axis are distinct fields.
                    replacement_capability=ReplacementCapability.SUBSTANTIVE,
                ),
            )
        )
    }
    resolved = _resolve({"task": "x", "purpose": "dual"}, config, routes=routes)
    assert resolved.effective_tier is RecoveryPackageTier.MECHANICAL
    assert resolved.replacement_capability is ReplacementCapability.SUBSTANTIVE
    # The shipped routes carry no capability marking.
    shipped = _resolve({"task": "x", "purpose": "implementation"}, config)
    assert shipped.replacement_capability is ReplacementCapability.NONE
    assert shipped.effective_tier is RecoveryPackageTier.SUBSTANTIVE


# ---------------------------------------------------------------------------
# Model/deployment, thinking, and authority resolution
# ---------------------------------------------------------------------------


def test_model_deployment_and_identity_revision_resolve_before_resources() -> None:
    resolved = _resolve(
        {"task": "x", "model": "@fast", "identity": "researcher"}, _config()
    )
    assert resolved.model_expression == "@fast"
    assert resolved.committed_model.base_model == "alpha-model"
    assert resolved.committed_model.provider == "alpha"
    assert resolved.committed_model.wire_name == "alpha-wire"
    assert resolved.committed_model.catalog_revision == "identity-launch-test"
    assert resolved.committed_model.thinking == "low"
    assert (resolved.base_model, resolved.provider, resolved.wire_name) == (
        "alpha-model",
        "alpha",
        "alpha-wire",
    )
    assert resolved.identity_revision == _expected_builtin_revision("researcher")


def test_effective_thinking_precedence() -> None:
    config = _config()
    # The session role's preset thinking rides the launch.
    assert _resolve({"task": "x"}, config).effective_thinking == "high"
    assert _resolve({"task": "x", "model": "@fast"}, config).effective_thinking == (
        "low"
    )
    # A canonical model takes its definition default.
    assert _resolve(
        {"task": "x", "model": "alpha-model"}, config
    ).effective_thinking == ("medium")
    # An explicit per-launch thinking beats the preset.
    assert (
        _resolve(
            {"task": "x", "model": "@fast", "config": {"thinking": "high"}}, config
        ).effective_thinking
        == "high"
    )
    # A configured override beats the model default for canonical models.
    overridden = _config(thinking_overrides={"beta-model": "low"})
    assert (
        _resolve({"task": "x", "model": "beta-model"}, overridden).effective_thinking
        == "low"
    )


def test_unknown_model_and_unsupported_thinking_are_typed() -> None:
    config = _config()
    with pytest.raises(InvalidLaunchModelSelectionError) as model_error:
        _resolve({"task": "x", "model": "no-such-model"}, config)
    assert model_error.value.field == "model"
    with pytest.raises(InvalidLaunchModelSelectionError) as thinking_error:
        _resolve(
            {"task": "x", "model": "limited-model", "config": {"thinking": "high"}},
            config,
        )
    assert thinking_error.value.field == "config.thinking"


def test_resolution_without_an_attached_catalog_is_typed() -> None:
    config = build_test_vibe_config(active_model="@main")
    object.__setattr__(config, "_catalog_snapshot", None)
    with pytest.raises(InvalidLaunchModelSelectionError) as excinfo:
        _resolve({"task": "x"}, config)
    assert excinfo.value.field == "model"
    assert "catalog" in str(excinfo.value).lower()


def test_authority_axes_resolve_from_the_child_identity_and_ceiling() -> None:
    ceiling = frozenset({"bash", "read_file", "grep"})
    config = _config()
    coder = _resolve({"task": "x"}, config, authorized=ceiling)
    assert coder.authority.personal_posture is ToolPosture.ALL_GRANTED
    assert coder.authority.delegation_posture is ToolPosture.ALL_GRANTED
    assert coder.authority.authorized_tool_names == ceiling
    researcher = _resolve({"task": "x", "identity": "researcher"}, config)
    assert researcher.authority.personal_posture is ToolPosture.NON_MUTATING
    assert researcher.authority.delegation_posture is ToolPosture.NON_MUTATING


def test_resolution_accepts_prevalidated_args_and_a_session_model_override() -> None:
    config = _config()
    args = TaskIdentityArgs.model_validate({"task": "x"})
    resolved = _resolve(args, config)
    assert resolved.identity == "coder"
    # The session model binds the $session route targets.
    other = _resolve(
        {"task": "x", "purpose": "research"}, config, session_model="@fast"
    )
    assert other.base_model == "alpha-model"


# ---------------------------------------------------------------------------
# Preservation of additive instructions and lineage/lifecycle fields
# ---------------------------------------------------------------------------


def test_additive_instructions_and_lineage_lifecycle_fields_are_preserved() -> None:
    resolved = _resolve(
        {
            "task": "do work",
            "task_summary": "  summarize   the work  ",
            "background": False,
            "retain": True,
            "agent_id": "agent-7",
            "replace_run": True,
            "model": "@fast",
            "identity": "researcher",
            "config": {
                "instructions": "Additive per-launch instructions.",
                "thinking": "high",
                "work_package_id": "wp-1",
                "predecessor_agent_id": "agent-3",
                "package_tier": "mechanical",
                "enabled_tools": ["bash"],
            },
        },
        _config(),
    )
    assert resolved.instructions == "Additive per-launch instructions."
    assert resolved.work_package_id == "wp-1"
    assert resolved.predecessor_agent_id == "agent-3"
    assert resolved.package_tier_claim is RecoveryPackageTier.MECHANICAL
    assert resolved.semantic_overrides.thinking == "high"
    assert resolved.semantic_overrides.enabled_tools == ["bash"]
    assert resolved.background is False
    assert resolved.retain is True
    assert resolved.agent_id == "agent-7"
    assert resolved.replace_run is True
    assert resolved.task_summary == "summarize the work"


def test_resolution_does_not_mutate_its_inputs() -> None:
    config = _config()
    snapshot = config.catalog_snapshot
    args = TaskIdentityArgs.model_validate({
        "task": "do work",
        "config": {"instructions": "extra"},
    })
    resolved = _resolve(args, config)
    assert resolved.instructions == "extra"
    assert args.config is not None
    assert args.config.instructions == "extra"
    assert config.catalog_snapshot is snapshot
    assert config.active_model == "@main"
    # The preserved semantic overrides are a deep copy.
    args.config.instructions = "changed"
    assert resolved.instructions == "extra"


# ---------------------------------------------------------------------------
# The resource-free invariant: invalid launches have zero side effects
# ---------------------------------------------------------------------------


def test_resolution_is_resource_free_and_invalid_launches_have_zero_side_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """INVARIANT (package 8): invalid launches create NO backend, MCP client,
    child log, handle, or reservation — validation failure leaves zero side
    effects. The full resolution completes with no backend/MCP/resource
    construction: every spy below stays untripped by valid AND invalid
    launches alike.
    """

    def _forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("resource constructed during launch resolution")

    monkeypatch.setattr("chartreux.core.llm.backend.factory.create_backend", _forbidden)
    monkeypatch.setattr("chartreux.core.tools.mcp.registry.MCPRegistry", _forbidden)
    monkeypatch.setattr("chartreux.core.agents.manager.AgentManager", _forbidden)

    config = _config()
    # Valid resolutions complete with every spy armed.
    resolved = _resolve({"task": "do work"}, config)
    assert resolved.identity == "coder"
    _resolve({"task": "research", "purpose": "research"}, config)
    _resolve({"task": "r", "model": "@fast:researcher"}, config)
    # The resolved launch is a pure value: no handle, log, or reservation.
    assert is_dataclass(resolved)
    assert resolved.agent_id is None
    # Invalid launches raise the typed grammar/resolution errors — before any
    # resource construction.
    with pytest.raises(InvalidChildIdentityError):
        _resolve({"task": "x", "identity": "orchestrator"}, config)
    with pytest.raises(InvalidLaunchPurposeError):
        _resolve({"task": "x", "purpose": "no-such-purpose"}, config)
    with pytest.raises(InvalidLaunchModelSelectionError):
        _resolve({"task": "x", "model": "no-such-model"}, config)
    with pytest.raises(ValidationError):
        _resolve({"task": "x", "model": "@fast", "config": {"model": "fast"}}, config)
    with pytest.raises(ValidationError):
        _resolve({"task": "x", "model": "@fast:reviewer", "identity": "coder"}, config)
