"""WP5 — failure-time replacement policy + replacement handoff.

Covers the WP5 acceptance: the orchestrator chooses a replacement AT FAILURE
TIME from validated eligible sets (no dispatch-time substitution table, no
automatic cross-model fallback); endpoint independence is enforced against
the ATTEMPTED failed endpoint (unknown identity cannot prove independence;
same-endpoint aliases and same-endpoint different-model substitutions are
rejected); luna-class (mechanical-tier) models are rejected for substantive
packages; every substitution emits a milestone notice with an honest cost
implication; the replacement gets a NEW agent_id with predecessor/package
lineage; the old recovery is disabled before the write-ownership transfer,
which happens only after reconciliation + the explicit handoff; budgets and
counters span replacements and are never reset; a HEAD/branch preflight stop
stays a stop; exhausted packages park and independent work continues; and the
unattended-resilience rule (the v0.5.5 production incident): infrastructure/
harness failures NEVER produce ``ask_user_question`` — the controlled
repeated-infrastructure-failure scenario produces ZERO ask invocations,
continues independent work where available, and otherwise ends cleanly with
the consolidated resumable report carrying U's deferred records.

Harness: policy unit tests build envelopes directly (endpoint-carrying
provider faults); the end-to-end tests drive the real registry
(``registry.run``) with a three-endpoint catalog (luna-model@alpha is the
mechanical tier via the ``@scout`` slot, main-model@beta is the failing
package's model, alt-model@gamma is the endpoint-independent replacement),
behind per-loop scripted backends patched at the ``create_backend`` facade.
"""

from __future__ import annotations

from typing import Any, cast, get_args
from unittest.mock import AsyncMock, MagicMock

import httpx
from pydantic import ValidationError
import pytest

from chartreux.app_server._sessions import (
    AgentRecord,
    SessionRuntimeRegistry,
    _AgentState,
)
from chartreux.app_server._subagent_results import ResultsStateView, RunResults
from chartreux.core.agent_loop.errors import EmptyLLMResponseError
from chartreux.core.agents.manager import AgentManager
from chartreux.core.config.chartreux_schema import OrchestrationConfig
from chartreux.core.config.models import RecoveryConfig, RecoverySectionConfig
from chartreux.core.deferred_questions import DeferredQuestionStore
from chartreux.core.dispatch.route_models import (
    PackageTier as RoutePackageTier,
    PurposeRoute,
    ReplacementCapability,
    RouteTarget,
)
from chartreux.core.dispatch.routes import (
    COST_TIER_DEFAULT,
    SESSION_MODEL,
    shipped_routes,
)
from chartreux.core.llm.exceptions import (
    BackendError,
    IncompleteStreamError,
    PayloadSummary,
)
from chartreux.core.llm_models import FunctionCall, LLMChunk, ToolCall
from chartreux.core.model_catalog.defaults import SHIPPED_CATALOG
from chartreux.core.model_catalog.loader import CatalogSnapshot
from chartreux.core.model_catalog.resolver import ModelResolver
from chartreux.core.model_catalog.schema import ModelCatalog
from chartreux.core.recovery_models import (
    FailureClass,
    FailureEnvelope,
    PackageTier,
    build_failure_envelope,
)
from chartreux.core.recovery_observation import observe_exception
from chartreux.core.recovery_policy import (
    INFRASTRUCTURE_FAILURE_CLASSES,
    UNATTENDED_RESILIENCE_MARKER,
    ParkedPackage,
    ReplacementCandidate,
    ReplacementDecision,
    ReplacementPolicyError,
    build_consolidated_blocker_report,
    candidates_from_catalog,
    eligible_replacement_candidates,
    is_infrastructure_failure,
    mechanical_base_models_from_routes,
    parked_decision,
    route_recovery_facts,
    validate_replacement_selection,
)
from chartreux.core.subagents import LaunchConfig, ResumeOutcome, TaskArgs, TaskResult
from chartreux.core.tools.base import BaseToolState, InvokeContext, ToolError
from chartreux.core.tools.builtins.task import (
    Task,
    TaskToolConfig,
    _validate_replacement_claim_args,
)
from chartreux.questions import (
    QuestionKind,
    UnansweredQuestion,
    UnansweredQuestions,
    UnattendedDefault,
    UnattendedDefaultAction,
)
from tests.app_server._subagent_test_support import (
    ImmediateFailureBackend,
    _background_result,
    _config,
    _GatedWakeup,
    _ManualClock,
    _wait_result,
)
from tests.conftest import ConfigBuilder, OrchestratorLoader, build_test_agent_loop
from tests.mock.utils import collect_result, mock_llm_chunk
from tests.recovery.test_explicit_resume import _until
from tests.stubs.fake_backend import FakeBackend

# ---------------------------------------------------------------------------
# Policy unit fixtures: endpoint-carrying provider faults and envelopes
# ---------------------------------------------------------------------------


def _payload_summary() -> PayloadSummary:
    return PayloadSummary(
        model="main-wire",
        message_count=1,
        approx_chars=2,
        temperature=0.0,
        has_tools=False,
        tool_choice=None,
    )


def _stall(endpoint: str, *, provider: str = "beta") -> BackendError:
    """A provider-stall fault (TIMEOUT category -> PROVIDER_STALL) carrying the
    attempted endpoint identity.
    """
    error = BackendError(
        provider=provider,
        endpoint=endpoint,
        status=None,
        reason="read timed out",
        headers=None,
        body_text=None,
        parsed_error="Network error",
        model="main-wire",
        payload_summary=_payload_summary(),
    )
    error.__cause__ = httpx.ReadTimeout("read timed out")
    return error


def _envelope(
    *,
    agent_id: str = "agent-1",
    run_id: str = "run-1",
    endpoint: str = "https://beta.test/v1",
    base_model: str = "main-model",
    work_package_id: str | None = "wp-1",
    predecessor_agent_id: str | None = None,
    tool_calls_present: bool = True,
    endpoint_correlated: bool = False,
) -> FailureEnvelope:
    """One ELIGIBLE provider-stall envelope (a settled tool effect, so the
    reconciliation barrier does not block a replacement claim).
    """
    observation = observe_exception(_stall(endpoint), phase="turn", turn_id="turn-1")
    return build_failure_envelope(
        observation=observation,
        agent_id=agent_id,
        run_id=run_id,
        attempted_deployment=(base_model, "beta", "main-wire"),
        configured_model=base_model,
        tool_calls_present=tool_calls_present,
        endpoint_correlated=endpoint_correlated,
        work_package_id=work_package_id,
        predecessor_agent_id=predecessor_agent_id,
    )


def _with_class(
    envelope: FailureEnvelope, failure_class: FailureClass
) -> FailureEnvelope:
    return envelope.model_copy(
        update={
            "failure": envelope.failure.model_copy(
                update={"failure_class": failure_class}
            )
        }
    )


def _candidate(
    base_model: str,
    provider: str,
    endpoint: str | None,
    *,
    tier: PackageTier = PackageTier.SUBSTANTIVE,
    wire: str | None = None,
    price: float | None = None,
) -> ReplacementCandidate:
    return ReplacementCandidate(
        base_model=base_model,
        provider=provider,
        wire_name=wire or f"{base_model}-wire",
        endpoint=endpoint,
        tier=tier,
        output_price_per_million=price,
    )


_ALT = _candidate("alt-model", "gamma", "https://gamma.test/v1", price=5.0)
_FAILED = _candidate("main-model", "beta", "https://beta.test/v1", price=3.0)
_LUNA = _candidate(
    "luna-model",
    "alpha",
    "https://alpha.test/v1",
    tier=PackageTier.MECHANICAL,
    price=0.5,
)


# ---------------------------------------------------------------------------
# Failure-time selection and the validated eligible set
# ---------------------------------------------------------------------------


def test_replacement_is_chosen_at_failure_time_from_the_validated_eligible_set() -> (
    None
):
    """The orchestrator chooses at failure time from the validated eligible
    set: only endpoint-independent, tier-capable candidates survive, and an
    empty set parks (never improvises). There is no dispatch-time substitution
    table — the decision consumes the failure envelope.
    """
    envelope = _envelope()
    candidates = [_ALT, _FAILED, _LUNA]
    eligible = eligible_replacement_candidates(
        envelope=envelope,
        package_tier=PackageTier.SUBSTANTIVE,
        candidates=candidates,
        config=RecoveryConfig(),
    )
    # Same-endpoint (failed) and mechanical-tier (luna) candidates are
    # rejected; only the endpoint-independent substantive one survives.
    assert [candidate.base_model for candidate in eligible] == ["alt-model"]
    # An eligible set of only same-endpoint candidates is empty -> park.
    assert (
        eligible_replacement_candidates(
            envelope=envelope,
            package_tier=PackageTier.SUBSTANTIVE,
            candidates=[_FAILED],
            config=RecoveryConfig(),
        )
        == []
    )


def test_endpoint_independence_enforced_against_the_attempted_failed_endpoint() -> None:
    """Endpoint independence is measured against the ATTEMPTED failed
    endpoint carried by the envelope: a same-endpoint alias (same model,
    different wire name) and a same-endpoint different-model substitution are
    both rejected, including userinfo/case/path variants of the same host.
    """
    envelope = _envelope(endpoint="https://beta.test/v1")
    config = RecoveryConfig()

    alias = _candidate("main-model", "beta", "https://beta.test/v1", wire="alias-wire")
    decision = validate_replacement_selection(
        envelope=envelope,
        package_tier=PackageTier.SUBSTANTIVE,
        claimed=alias,
        candidates=[alias],
        config=config,
    )
    assert decision.disposition == "rejected"
    assert "same_endpoint_substitution_rejected" in decision.reason_codes

    other_model = _candidate("alt-model", "beta", "https://beta.test/v1")
    decision = validate_replacement_selection(
        envelope=envelope,
        package_tier=PackageTier.SUBSTANTIVE,
        claimed=other_model,
        candidates=[other_model],
        config=config,
    )
    assert decision.disposition == "rejected"
    assert "same_endpoint_substitution_rejected" in decision.reason_codes

    variant = _candidate("alt-model", "gamma", "https://user:pw@BETA.test/v1/")
    decision = validate_replacement_selection(
        envelope=envelope,
        package_tier=PackageTier.SUBSTANTIVE,
        claimed=variant,
        candidates=[variant],
        config=config,
    )
    assert decision.disposition == "rejected"
    assert "same_endpoint_substitution_rejected" in decision.reason_codes

    # A genuinely independent endpoint is selected.
    decision = validate_replacement_selection(
        envelope=envelope,
        package_tier=PackageTier.SUBSTANTIVE,
        claimed=_ALT,
        candidates=[_ALT],
        config=config,
    )
    assert decision.disposition == "selected"


def test_unknown_endpoint_identity_cannot_prove_independence() -> None:
    """An unknown identity (failed side or candidate side) cannot prove
    endpoint independence, so no substitution is admitted.
    """
    envelope = _envelope(endpoint="https://beta.test/v1")
    unknown_failed = envelope.model_copy(
        update={"deployment": envelope.deployment.model_copy(update={"endpoint": None})}
    )
    decision = validate_replacement_selection(
        envelope=unknown_failed,
        package_tier=PackageTier.SUBSTANTIVE,
        claimed=_ALT,
        candidates=[_ALT],
        config=RecoveryConfig(),
    )
    assert decision.disposition == "rejected"
    assert "failed_endpoint_unknown_cannot_prove_independence" in decision.reason_codes

    unknown_candidate = _candidate("alt-model", "gamma", None)
    decision = validate_replacement_selection(
        envelope=envelope,
        package_tier=PackageTier.SUBSTANTIVE,
        claimed=unknown_candidate,
        candidates=[unknown_candidate],
        config=RecoveryConfig(),
    )
    assert decision.disposition == "rejected"
    assert (
        "candidate_endpoint_unknown_cannot_prove_independence" in decision.reason_codes
    )


def test_luna_rejected_for_substantive_packages() -> None:
    """A mechanical-tier (luna-class) model is never an eligible replacement
    for a substantive package — and IS eligible for a mechanical one.
    """
    envelope = _envelope()
    decision = validate_replacement_selection(
        envelope=envelope,
        package_tier=PackageTier.SUBSTANTIVE,
        claimed=_LUNA,
        candidates=[_LUNA],
        config=RecoveryConfig(),
    )
    assert decision.disposition == "rejected"
    assert "mechanical_tier_rejected_for_substantive_package" in decision.reason_codes

    decision = validate_replacement_selection(
        envelope=envelope,
        package_tier=PackageTier.MECHANICAL,
        claimed=_LUNA,
        candidates=[_LUNA],
        config=RecoveryConfig(),
    )
    assert decision.disposition == "selected"


# ---------------------------------------------------------------------------
# Package 8: the replacement-capability axis is SEPARATE from the package
# tier (interfaces.md settled decision; identity_launch resolution)
# ---------------------------------------------------------------------------


def _capability_candidate(
    base_model: str,
    provider: str,
    endpoint: str | None,
    *,
    tier: PackageTier = PackageTier.SUBSTANTIVE,
    capability: ReplacementCapability | None = None,
    wire: str | None = None,
    price: float | None = None,
) -> ReplacementCandidate:
    return ReplacementCandidate(
        base_model=base_model,
        provider=provider,
        wire_name=wire or f"{base_model}-wire",
        endpoint=endpoint,
        tier=tier,
        capability=capability,
        output_price_per_million=price,
    )


def test_capability_is_a_separate_axis_from_the_package_tier() -> None:
    """A substantive-tier-LABELED candidate explicitly marked
    mechanical-capable is rejected for a substantive package: eligibility is
    the intrinsic model-level capability, evaluated separately from the
    package tier — the tier label alone never confers eligibility.
    """
    envelope = _envelope()
    marked = _capability_candidate(
        "alt-model",
        "gamma",
        "https://gamma.test/v1",
        tier=PackageTier.SUBSTANTIVE,
        capability=ReplacementCapability.MECHANICAL,
        price=5.0,
    )
    decision = validate_replacement_selection(
        envelope=envelope,
        package_tier=PackageTier.SUBSTANTIVE,
        claimed=marked,
        candidates=[marked],
        config=RecoveryConfig(),
    )
    assert decision.disposition == "rejected"
    assert "candidate_capability_below_package_tier" in decision.reason_codes

    # The package tier remains a separate constraint: a mechanical-tier-
    # labeled candidate is rejected for a substantive package even when
    # marked substantive-capable.
    labeled = _capability_candidate(
        "luna-model",
        "alpha",
        "https://alpha.test/v1",
        tier=PackageTier.MECHANICAL,
        capability=ReplacementCapability.SUBSTANTIVE,
        price=0.5,
    )
    decision = validate_replacement_selection(
        envelope=envelope,
        package_tier=PackageTier.SUBSTANTIVE,
        claimed=labeled,
        candidates=[labeled],
        config=RecoveryConfig(),
    )
    assert decision.disposition == "rejected"
    assert "mechanical_tier_rejected_for_substantive_package" in decision.reason_codes


def test_unmarked_candidate_is_not_execution_capable() -> None:
    """Eligibility requires an explicit model-level capability marking."""
    envelope = _envelope()
    unmarked = _capability_candidate(
        "alt-model",
        "gamma",
        "https://gamma.test/v1",
        tier=PackageTier.SUBSTANTIVE,
        capability=ReplacementCapability.NONE,
        price=5.0,
    )
    decision = validate_replacement_selection(
        envelope=envelope,
        package_tier=PackageTier.MECHANICAL,
        claimed=unmarked,
        candidates=[unmarked],
        config=RecoveryConfig(),
    )
    assert decision.disposition == "rejected"
    assert "candidate_capability_not_execution_marked" in decision.reason_codes

    decision = validate_replacement_selection(
        envelope=envelope,
        package_tier=PackageTier.SUBSTANTIVE,
        claimed=unmarked,
        candidates=[unmarked],
        config=RecoveryConfig(),
    )
    assert decision.disposition == "rejected"
    assert "candidate_capability_below_package_tier" in decision.reason_codes


def test_capabilities_mapping_marks_one_canonical_capability_per_model() -> None:
    """The capability marking is model-keyed: a model bound on both mechanical
    and substantive route targets carries ONE canonical capability, and route
    duplication never upgrades a model.
    """
    candidates = candidates_from_catalog(
        SHIPPED_CATALOG,
        mechanical_base_models=frozenset({"glm-5-3"}),
        capabilities={
            "glm-5-3": ReplacementCapability.SUBSTANTIVE,
            "mistral-large-4": ReplacementCapability.MECHANICAL,
        },
    )
    by_model = {candidate.base_model: candidate for candidate in candidates}
    # glm-5-3 keeps its mechanical tier LABEL but carries the marked
    # capability: one canonical capability, not two target-derived identities.
    assert by_model["glm-5-3"].tier is PackageTier.MECHANICAL
    assert by_model["glm-5-3"].capability is ReplacementCapability.SUBSTANTIVE
    assert by_model["glm-5-3"].effective_capability is ReplacementCapability.SUBSTANTIVE
    # mistral-large-4 is marked mechanical-capable: not eligible for a
    # substantive package even though its tier label is substantive.
    assert by_model["mistral-large-4"].tier is PackageTier.SUBSTANTIVE
    assert (
        by_model["mistral-large-4"].effective_capability
        is ReplacementCapability.MECHANICAL
    )
    envelope = _envelope(endpoint="https://beta.test/v1")
    # For a substantive package BOTH are rejected — each on its own axis:
    # glm-5-3 on the tier label, mistral-large-4 on the capability marking.
    assert (
        eligible_replacement_candidates(
            envelope=envelope,
            package_tier=PackageTier.SUBSTANTIVE,
            candidates=candidates,
            config=RecoveryConfig(),
        )
        == []
    )
    # For a mechanical package both are eligible (execution-capable markings,
    # endpoint-independent of the failed call).
    mechanical_eligible = eligible_replacement_candidates(
        envelope=envelope,
        package_tier=PackageTier.MECHANICAL,
        candidates=candidates,
        config=RecoveryConfig(),
    )
    assert {candidate.base_model for candidate in mechanical_eligible} == {
        "glm-5-3",
        "mistral-large-4",
    }
    # A model absent from the marking map is marked NONE (an explicit marking
    # is required): it is not execution-capable for either tier.
    unmarked = candidates_from_catalog(
        SHIPPED_CATALOG, capabilities={"glm-5-3": ReplacementCapability.SUBSTANTIVE}
    )
    by_unmarked = {candidate.base_model: candidate for candidate in unmarked}
    assert by_unmarked["mistral-large-4"].capability is ReplacementCapability.NONE
    assert (
        by_unmarked["mistral-large-4"].effective_capability
        is ReplacementCapability.NONE
    )


def test_legacy_tier_label_derivation_is_unchanged_without_markings() -> None:
    """Without a capabilities mapping the capability derives from the tier
    label (the legacy slot-purpose inference path): existing behavior —
    budgets, endpoint independence, and the tier guard — is unchanged.
    """
    candidates = candidates_from_catalog(
        SHIPPED_CATALOG, mechanical_base_models=frozenset({"glm-5-3"})
    )
    by_model = {candidate.base_model: candidate for candidate in candidates}
    assert by_model["glm-5-3"].capability is None
    assert by_model["glm-5-3"].effective_capability is ReplacementCapability.MECHANICAL
    assert by_model["mistral-large-4"].capability is None
    assert (
        by_model["mistral-large-4"].effective_capability
        is ReplacementCapability.SUBSTANTIVE
    )
    envelope = _envelope()
    decision = validate_replacement_selection(
        envelope=envelope,
        package_tier=PackageTier.SUBSTANTIVE,
        claimed=by_model["glm-5-3"],
        candidates=candidates,
        config=RecoveryConfig(),
    )
    assert decision.disposition == "rejected"
    assert "mechanical_tier_rejected_for_substantive_package" in decision.reason_codes
    decision = validate_replacement_selection(
        envelope=envelope,
        package_tier=PackageTier.MECHANICAL,
        claimed=by_model["glm-5-3"],
        candidates=candidates,
        config=RecoveryConfig(),
    )
    assert decision.disposition == "selected"


def test_route_recovery_facts_canonicalize_presets_and_markings() -> None:
    """The recovery facts canonicalize preset references through the
    catalog mapping and carry the explicit capability markings keyed by
    canonical base model; targets without a marking contribute no marking
    (with no marking anywhere the legacy tier-label derivation applies).
    """
    routes = {
        "mechanical-edit": PurposeRoute(
            targets=(
                RouteTarget(
                    model="@scout", identity="coder", tier=RoutePackageTier.MECHANICAL
                ),
            )
        ),
        "session-bound": PurposeRoute(
            targets=(
                RouteTarget(
                    model=SESSION_MODEL,
                    identity="coder",
                    tier=RoutePackageTier.MECHANICAL,
                ),
            )
        ),
        "heavy-work": PurposeRoute(
            targets=(
                RouteTarget(
                    model="alt-model",
                    identity="coder",
                    tier=RoutePackageTier.SUBSTANTIVE,
                    replacement_capability=ReplacementCapability.MECHANICAL,
                ),
            )
        ),
        "cost-tier": PurposeRoute(
            targets=(
                RouteTarget(
                    model=COST_TIER_DEFAULT,
                    identity="coder",
                    tier=RoutePackageTier.MECHANICAL,
                ),
            )
        ),
    }
    mechanical, capabilities = route_recovery_facts(
        routes,
        session_model="main-model",
        canonical_models={
            "luna-model": "luna-model",
            "main-model": "main-model",
            "alt-model": "alt-model",
            "@scout": "luna-model",
            "@main": "main-model",
        },
    )
    # The preset reference and the $session binding both canonicalize to
    # their base models; the explicit marking is keyed by canonical base.
    # The cost-tier default contributes nothing without price data (a
    # price-less catalog cannot have resolved it at launch).
    assert mechanical == frozenset({"luna-model", "main-model"})
    assert capabilities == {"alt-model": ReplacementCapability.MECHANICAL}
    # With price data the cost-tier default resolves to the cheapest
    # capable model and joins the mechanical set.
    mechanical, capabilities = route_recovery_facts(
        routes,
        session_model="main-model",
        canonical_models={
            "luna-model": "luna-model",
            "main-model": "main-model",
            "alt-model": "alt-model",
            "budget-model": "budget-model",
            "@scout": "luna-model",
            "@main": "main-model",
        },
        catalog_prices={"budget-model": 1.5, "alt-model": 9.0},
    )
    assert mechanical == frozenset({"luna-model", "main-model", "budget-model"})
    assert capabilities == {"alt-model": ReplacementCapability.MECHANICAL}
    # No explicit marking anywhere: no markings are reported (the legacy
    # tier-label derivation applies unchanged).
    assert route_recovery_facts(
        _MECHANICAL_LUNA_ROUTES.routes,
        session_model="main-model",
        canonical_models={"luna-model": "luna-model", "main-model": "main-model"},
    ) == (frozenset({"luna-model"}), {})


def test_capability_marking_never_overrides_endpoint_independence() -> None:
    """The capability axis composes with — never replaces — the endpoint-
    independence and budget checks.
    """
    envelope = _envelope(endpoint="https://beta.test/v1")
    same_endpoint = _capability_candidate(
        "alt-model",
        "beta",
        "https://beta.test/v1",
        tier=PackageTier.SUBSTANTIVE,
        capability=ReplacementCapability.SUBSTANTIVE,
    )
    decision = validate_replacement_selection(
        envelope=envelope,
        package_tier=PackageTier.SUBSTANTIVE,
        claimed=same_endpoint,
        candidates=[same_endpoint],
        config=RecoveryConfig(),
    )
    assert decision.disposition == "rejected"
    assert "same_endpoint_substitution_rejected" in decision.reason_codes


def test_every_substitution_emits_a_milestone_notice_with_honest_cost() -> None:
    """Every selected substitution carries a milestone notice whose cost
    implication is stated honestly — including when pricing is unavailable.
    """
    envelope = _envelope(base_model="main-model")
    decision = validate_replacement_selection(
        envelope=envelope,
        package_tier=PackageTier.SUBSTANTIVE,
        claimed=_ALT,
        candidates=[_ALT, _FAILED],
        config=RecoveryConfig(),
    )
    assert decision.disposition == "selected"
    assert decision.candidate is _ALT
    assert "$5.00/M" in decision.cost_implication
    assert "$3.00/M" in decision.cost_implication
    assert "costs $2.00/M more per output token" in decision.cost_implication
    assert "Replacement milestone" in decision.notice
    assert "alt-model" in decision.notice
    assert "endpoint-independent" in decision.notice
    assert "wp-1" in decision.notice

    unpriced = _candidate("alt-model", "gamma", "https://gamma.test/v1", price=None)
    decision = validate_replacement_selection(
        envelope=envelope,
        package_tier=PackageTier.SUBSTANTIVE,
        claimed=unpriced,
        candidates=[unpriced, _FAILED],
        config=RecoveryConfig(),
    )
    assert decision.disposition == "selected"
    assert "pricing unavailable" in decision.cost_implication
    assert "the spend delta is unknown" in decision.cost_implication


def test_allowlist_and_price_cap_rejections() -> None:
    """The configured fallback allowlist and output-price cap are enforced
    as code; an unverifiable cap (unknown price) rejects, never guesses.
    """
    envelope = _envelope()
    config = RecoveryConfig(fallback_allowed_base_models=["other-model"])
    decision = validate_replacement_selection(
        envelope=envelope,
        package_tier=PackageTier.SUBSTANTIVE,
        claimed=_ALT,
        candidates=[_ALT],
        config=config,
    )
    assert decision.disposition == "rejected"
    assert "base_model_not_in_fallback_allowlist" in decision.reason_codes

    capped = RecoveryConfig(fallback_max_output_price_per_million=2.0)
    decision = validate_replacement_selection(
        envelope=envelope,
        package_tier=PackageTier.SUBSTANTIVE,
        claimed=_ALT,
        candidates=[_ALT],
        config=capped,
    )
    assert decision.disposition == "rejected"
    assert "fallback_price_cap_exceeded" in decision.reason_codes

    unpriced = _candidate("alt-model", "gamma", "https://gamma.test/v1", price=None)
    decision = validate_replacement_selection(
        envelope=envelope,
        package_tier=PackageTier.SUBSTANTIVE,
        claimed=unpriced,
        candidates=[unpriced],
        config=capped,
    )
    assert decision.disposition == "rejected"
    assert "fallback_price_unavailable_cap_unverifiable" in decision.reason_codes


def test_preflight_stop_preserved_and_protected_classes_rejected() -> None:
    """A HEAD/branch/workspace preflight stop stays a stop (never bypassed);
    protected classes (cancellation, authorization, permanent config, budget
    stop, task) never substitute; unknown/internal causes report-and-park.
    """
    envelope = _envelope()
    decision = validate_replacement_selection(
        envelope=_with_class(envelope, FailureClass.PREFLIGHT),
        package_tier=PackageTier.SUBSTANTIVE,
        claimed=_ALT,
        candidates=[_ALT],
        config=RecoveryConfig(),
    )
    assert decision.disposition == "rejected"
    assert "preflight_stop_preserved" in decision.reason_codes

    for failure_class, code in (
        (FailureClass.CANCELLATION, "protected_class_no_replacement"),
        (FailureClass.AUTHORIZATION, "protected_class_no_replacement"),
        (FailureClass.PERMANENT_CONFIG, "protected_class_no_replacement"),
        (FailureClass.BUDGET_STOP, "protected_class_no_replacement"),
        (FailureClass.TASK, "protected_class_no_replacement"),
        (FailureClass.UNKNOWN, "unknown_cause_report_and_park"),
        (FailureClass.INTERNAL, "harness_internal_report_and_park"),
    ):
        decision = validate_replacement_selection(
            envelope=_with_class(envelope, failure_class),
            package_tier=PackageTier.SUBSTANTIVE,
            claimed=_ALT,
            candidates=[_ALT],
            config=RecoveryConfig(),
        )
        assert decision.disposition == "rejected"
        assert code in decision.reason_codes


def test_decision_vocabulary_never_contains_a_user_question_action() -> None:
    """The disposition vocabulary is selected | rejected | parked — never a
    user-question action. Every infrastructure failure class resolves to one
    of those three, never an ask.
    """
    annotation = ReplacementDecision.model_fields["disposition"].annotation
    dispositions = set(get_args(annotation))
    assert dispositions == {"selected", "rejected", "parked"}
    assert "ask" not in dispositions

    envelope = _envelope()
    for failure_class in sorted(INFRASTRUCTURE_FAILURE_CLASSES, key=str):
        decision = validate_replacement_selection(
            envelope=_with_class(envelope, failure_class),
            package_tier=PackageTier.SUBSTANTIVE,
            claimed=_ALT,
            candidates=[_ALT],
            config=RecoveryConfig(),
        )
        assert decision.disposition in {"selected", "rejected", "parked"}


def test_parked_decision_carries_context_and_never_asks() -> None:
    envelope = _envelope(work_package_id="wp-9", predecessor_agent_id="agent-8")
    decision = parked_decision(
        envelope=envelope,
        reason_codes=["no_endpoint_independent_replacement"],
        detail="every catalog model shares the failed endpoint",
    )
    assert decision.disposition == "parked"
    assert decision.work_package_id == "wp-9"
    # The decision names the FAILED agent as the replacement's predecessor
    # (the claim value); the envelope's own predecessor_agent_id ("agent-8")
    # is the failed agent's ancestry, carried by the envelope, not the decision.
    assert decision.predecessor_agent_id == "agent-1"
    assert decision.failed_endpoint == "beta.test"
    assert "no user question is asked" in decision.notice


# ---------------------------------------------------------------------------
# The consolidated resumable blocker report + U's deferred records
# ---------------------------------------------------------------------------


def _echoed_question(
    question_id: str,
    *,
    kind: QuestionKind = QuestionKind.PREFERENCE,
    action: UnattendedDefaultAction = UnattendedDefaultAction.DEFER_AFFECTED_WORK,
) -> UnansweredQuestion:
    # U's contract: preserve_existing_choice must name the existing choice.
    existing_choice = (
        "Keep the current plan"
        if action is UnattendedDefaultAction.PRESERVE_EXISTING_CHOICE
        else None
    )
    return UnansweredQuestion(
        question_id=question_id,
        question="Ship it?",
        kind=kind,
        unattended_default=UnattendedDefault(
            action=action,
            description=f"Defer {question_id}.",
            existing_choice=existing_choice,
        ),
    )


def _record_deferred(
    store: DeferredQuestionStore, request_id: str, *questions: UnansweredQuestion
) -> None:
    store.record_unanswered(
        session_id="session-1",
        request_id=request_id,
        unanswered=UnansweredQuestions(
            questions=list(questions),
            elapsed_seconds=61.0,
            timeout_seconds=60.0,
            continuation_guidance="Continue without the user's answer.",
        ),
    )


def test_consolidated_blocker_report_carries_resumable_state_and_deferred_records() -> (
    None
):
    """The consolidated report (emitted only when routes/budgets are
    exhausted) carries the resumable state, surfaces U's deferred records
    distinguishing a deferred preference from an actual blocker from a
    still-open authorization decision, notes correlated parent failures
    without autonomously replacing the root, and ends with a one-instruction
    resume path.
    """
    envelope = _envelope(endpoint_correlated=True)
    parked = [
        ParkedPackage.from_envelope(
            envelope,
            reason_codes=["no_endpoint_independent_replacement"],
            detail="every catalog model shares the failed endpoint",
            proposed_classification="provider-side stall",
            missing_evidence=["response headers"],
        )
    ]
    store = DeferredQuestionStore()
    _record_deferred(
        store,
        "req-preference",
        _echoed_question("q1", action=UnattendedDefaultAction.PRESERVE_EXISTING_CHOICE),
    )
    _record_deferred(
        store,
        "req-informational",
        _echoed_question("q2", kind=QuestionKind.INFORMATIONAL),
    )
    _record_deferred(
        store,
        "req-blocker",
        _echoed_question("q3"),
        _echoed_question("q4", kind=QuestionKind.INFORMATIONAL),
    )

    report = build_consolidated_blocker_report(
        parked=parked, deferred_questions=store, session_id="session-1"
    )

    assert UNATTENDED_RESILIENCE_MARKER in report
    assert "No open interactive question is pending" in report
    assert "wp-1" in report
    assert "provider_stall" in report
    assert "beta.test" in report
    assert "proposed classification: provider-side stall" in report
    assert "missing evidence: response headers" in report
    # Correlated parent/child failures are visible; the root orchestrator is
    # NEVER autonomously replaced.
    assert "Correlated parent/child failures are visible" in report
    assert "NEVER autonomously replaced" in report
    # Deferred records distinguish the three kinds.
    assert "deferred preference (not a blocker)" in report
    assert "deferred informational note (not a blocker)" in report
    assert "deferred work (possible blocker)" in report
    # Still-open authorization decisions are called out as never expiring.
    empty = build_consolidated_blocker_report(
        parked=[], deferred_questions=DeferredQuestionStore()
    )
    assert (
        "still-open authorization decisions never expire and are not listed here"
        in empty
    )
    # The one-instruction resume path.
    assert "Resume with one instruction" in report
    assert "work_package_id" in report


# ---------------------------------------------------------------------------
# Task-tool claim-shape guards
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_task_tool_claim_shape_guards(
    build_config: ConfigBuilder, load_orchestrator: OrchestratorLoader[Any]
) -> None:
    """The task tool rejects malformed replacement claims before dispatch: a
    claim launches a NEW agent (agent_id omitted), in background mode, naming
    its model and work package explicitly.
    """
    task = Task(config_getter=lambda: TaskToolConfig(), state=BaseToolState())
    ctx = InvokeContext(
        tool_call_id="test-call-id",
        agent_manager=AgentManager(load_orchestrator(build_config())),
    )

    with pytest.raises(ToolError, match="omit agent_id"):
        await collect_result(
            task.run(
                TaskArgs(
                    task="work",
                    identity="coder",
                    background=True,
                    agent_id="agent-1",
                    config=LaunchConfig(
                        model="alt-model",
                        work_package_id="wp-1",
                        predecessor_agent_id="agent-0",
                    ),
                ),
                ctx,
            )
        )
    with pytest.raises(ToolError, match="requires background mode"):
        await collect_result(
            task.run(
                TaskArgs(
                    task="work",
                    identity="coder",
                    background=False,
                    config=LaunchConfig(
                        model="alt-model",
                        work_package_id="wp-1",
                        predecessor_agent_id="agent-0",
                    ),
                ),
                ctx,
            )
        )
    with pytest.raises(ToolError, match="requires the replacement model"):
        await collect_result(
            task.run(
                TaskArgs(
                    task="work",
                    identity="coder",
                    background=True,
                    config=LaunchConfig(
                        work_package_id="wp-1", predecessor_agent_id="agent-0"
                    ),
                ),
                ctx,
            )
        )
    with pytest.raises(ToolError, match="requires config.work_package_id"):
        await collect_result(
            task.run(
                TaskArgs(
                    task="work",
                    identity="coder",
                    background=True,
                    config=LaunchConfig(
                        model="alt-model", predecessor_agent_id="agent-0"
                    ),
                ),
                ctx,
            )
        )


def test_task_tool_claim_guard_accepts_the_top_level_model() -> None:
    """The task tool's claim-shape guard accepts the canonical top-level
    model field (the legacy nested config.model is the promoted
    equivalent); a dual supply stays a schema validation error.
    """
    _validate_replacement_claim_args(
        TaskArgs(
            task="work",
            model="alt-model",
            background=True,
            config=LaunchConfig(work_package_id="wp-1", predecessor_agent_id="agent-0"),
        )
    )
    with pytest.raises(ValidationError, match="supplied both"):
        TaskArgs(
            task="work",
            model="alt-model",
            background=True,
            config=LaunchConfig(
                model="main-model",
                work_package_id="wp-1",
                predecessor_agent_id="agent-0",
            ),
        )


# ---------------------------------------------------------------------------
# End-to-end: the claim through the real registry
# ---------------------------------------------------------------------------


def _replacement_catalog() -> ModelCatalog:
    """Three endpoints: luna-model@alpha (mechanical tier via @scout),
    main-model@beta (the failing package's model via @worker), alt-model@gamma
    (the endpoint-independent replacement via @heavy).
    """
    return ModelCatalog.model_validate({
        "providers": {
            "alpha": {"api_base": "https://alpha.test"},
            "beta": {"api_base": "https://beta.test"},
            "gamma": {"api_base": "https://gamma.test"},
        },
        "models": {
            "luna-model": {
                "thinking": "low",
                "deployments": [
                    {
                        "provider": "alpha",
                        "name": "luna-wire",
                        "prices": {"input": 0.1, "output": 0.5, "cached_input": 0.01},
                    }
                ],
            },
            "main-model": {
                "thinking": "medium",
                "deployments": [
                    {
                        "provider": "beta",
                        "name": "main-wire",
                        "prices": {"input": 1.0, "output": 3.0, "cached_input": 0.1},
                    }
                ],
            },
            "alt-model": {
                "thinking": "high",
                "deployments": [
                    {
                        "provider": "gamma",
                        "name": "alt-wire",
                        "prices": {"input": 2.0, "output": 5.0, "cached_input": 0.2},
                    }
                ],
            },
        },
        "roles": {
            "main": {
                "model": "main-model",
                "thinking": "medium",
                "description": "main assistant preset",
            },
            "scout": {
                "model": "luna-model",
                "thinking": "low",
                "description": "scout preset",
            },
            "worker": {
                "model": "main-model",
                "thinking": "medium",
                "description": "worker preset",
            },
            "heavy": {
                "model": "alt-model",
                "thinking": "high",
                "description": "heavy preset",
            },
        },
    })


def _replacement_config(
    recovery: RecoveryConfig, routes: OrchestrationConfig | None = None
):
    config = _config(enabled_tools=["task", "todo"])
    update: dict[str, Any] = {
        "recovery": RecoverySectionConfig(subagents=recovery),
        "subagents": config.subagents.model_copy(
            update={"idle_ttl_seconds": 0, "max_idle_agents": 0}
        ),
    }
    if routes is not None:
        update["orchestration"] = routes
    config = config.model_copy(update=update)
    return config.attach_catalog_snapshot(
        CatalogSnapshot(_replacement_catalog(), "wp5-test")
    )


# A mechanical-tier purpose route binding the luna-class model explicitly:
# under the successor semantics the mechanical set is the set of models the
# configured mechanical-tier routes bind, not a slot roster.
_MECHANICAL_LUNA_ROUTES = OrchestrationConfig(
    routes={
        "mechanical-edit": PurposeRoute(
            targets=(
                RouteTarget(
                    model="luna-model",
                    identity="coder",
                    tier=RoutePackageTier.MECHANICAL,
                ),
            )
        )
    }
)

# Two mechanical-tier routes on distinct endpoints: the package launches
# through "mechanical-edit" (luna-model@alpha) and the claim names the
# mechanical "search" model (alt-model@gamma) — endpoint-independent.
_TIER_ROUTES = OrchestrationConfig(
    routes={
        "mechanical-edit": PurposeRoute(
            targets=(
                RouteTarget(
                    model="luna-model",
                    identity="coder",
                    tier=RoutePackageTier.MECHANICAL,
                ),
            )
        ),
        "search": PurposeRoute(
            targets=(
                RouteTarget(
                    model="alt-model",
                    identity="researcher",
                    tier=RoutePackageTier.MECHANICAL,
                ),
            )
        ),
    }
)

# A substantive-tier route target explicitly marking alt-model
# mechanical-CAPABLE at the model level (a separate axis from the tier).
_CAPABILITY_ROUTES = OrchestrationConfig(
    routes={
        "heavy-work": PurposeRoute(
            targets=(
                RouteTarget(
                    model="alt-model",
                    identity="coder",
                    tier=RoutePackageTier.SUBSTANTIVE,
                    replacement_capability=ReplacementCapability.MECHANICAL,
                ),
            )
        )
    }
)

# A mechanical-tier route bound via the @scout PRESET: the recovery facts
# canonicalize the preset reference to its canonical base (luna-model).
_PRESET_MECHANICAL_ROUTES = OrchestrationConfig(
    routes={
        "mechanical-edit": PurposeRoute(
            targets=(
                RouteTarget(
                    model="@scout", identity="coder", tier=RoutePackageTier.MECHANICAL
                ),
            )
        )
    }
)


class _EndpointTimeout(TimeoutError):
    """A read-timeout fault carrying the attempted endpoint identity."""

    def __init__(self, endpoint: str) -> None:
        super().__init__("read timed out")
        self.endpoint = endpoint


def _todo_write_call(call_id: str, content: str) -> ToolCall:
    return ToolCall(
        id=call_id,
        index=0,
        function=FunctionCall(
            name="todo",
            arguments=(
                '{"action": "write", "todos": ['
                f'{{"id": "1", "content": "{content}", "status": "pending"}}'
                "]}"
            ),
        ),
    )


class _WriteThenFaultBackend(FakeBackend):
    """The first logical call answers with a settled todo write (so the
    failure envelope's tool effects are SETTLED -> ELIGIBLE); every later call
    raises the endpoint-carrying provider fault (retries included).
    """

    def __init__(self, fault: Exception) -> None:
        super().__init__([])
        self._fault = fault
        self.calls = 0

    async def complete(self, **kwargs: Any) -> LLMChunk:
        self.calls += 1
        if self.calls == 1:
            return mock_llm_chunk(
                content="", tool_calls=[_todo_write_call("call-1", "package work")]
            )
        raise self._fault

    async def complete_streaming(self, **kwargs: Any):
        self.calls += 1
        if self.calls == 1:
            yield mock_llm_chunk(
                content="", tool_calls=[_todo_write_call("call-1", "package work")]
            )
            return
        raise self._fault


class _LoopBackends:
    """Per-child-loop backend instances behind the patched create_backend
    facade, bound in launch order (one shared instance per loop, so a
    write-then-fault script accumulates across that loop's logical calls).
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch, loops: list[Any]) -> None:
        self._pending = list(loops)
        self._bound: dict[int, Any] = {}
        monkeypatch.setattr(
            "chartreux.core.agent_loop._loop.create_backend", self._create
        )

    def _create(self, **kwargs: Any) -> Any:
        on_retry = kwargs.get("on_retry")
        loop = getattr(on_retry, "__self__", None)
        key = id(loop) if loop is not None else -1
        if key not in self._bound:
            assert self._pending, "create_backend called by an unscripted loop"
            self._bound[key] = self._pending.pop(0)
        return self._bound[key]


async def _replacement_registry(
    monkeypatch: pytest.MonkeyPatch,
    recovery: RecoveryConfig,
    backends: list[Any],
    routes: OrchestrationConfig | None = None,
) -> tuple[SessionRuntimeRegistry, Any]:
    _LoopBackends(monkeypatch, backends)
    parent = build_test_agent_loop(
        config=_replacement_config(recovery, routes), backend=FakeBackend()
    )
    registry = SessionRuntimeRegistry(
        AsyncMock(),
        AsyncMock(),
        lambda _session_id: 0,
        notify_agents=AsyncMock(),
        clock=_ManualClock(),
        wakeup=_GatedWakeup(),
    )
    root = registry._build_child_runtime(parent)
    root.turns._projector = MagicMock()
    root.turns.link_subagent = AsyncMock()
    registry.bind_root(root)
    return registry, parent


async def _launch(
    registry: SessionRuntimeRegistry,
    parent: Any,
    *,
    task: str = "package work",
    tool_call_id: str = "launch",
    config: dict[str, Any] | None = None,
    model: str | None = None,
) -> TaskResult:
    arguments: dict[str, Any] = {
        "task": task,
        "identity": "coder",
        "background": True,
        "retain": True,
    }
    if config is not None:
        arguments["config"] = config
    if model is not None:
        arguments["model"] = model
    return await _background_result(
        registry,
        TaskArgs(**arguments),
        InvokeContext(tool_call_id=tool_call_id, session_id=parent.session_id),
    )


async def _launch_failed_package(
    registry: SessionRuntimeRegistry,
    parent: Any,
    *,
    endpoint: str = "https://beta.test/v1",
    tool_call_id: str = "launch",
) -> tuple[str, str, TaskResult]:
    """Launch a retained child that settles one tool effect and then fails
    with an endpoint-carrying provider stall (ELIGIBLE envelope).
    """
    launch = await _launch(registry, parent, tool_call_id=tool_call_id)
    assert launch.agent_id is not None and launch.run_id is not None
    failed = await _wait_result(registry, launch.agent_id, launch.run_id)
    assert not failed.completed
    return launch.agent_id, launch.run_id, failed


def _claim_config(
    *, model: str, work_package_id: str, predecessor_agent_id: str
) -> dict[str, Any]:
    return {
        "model": model,
        "work_package_id": work_package_id,
        "predecessor_agent_id": predecessor_agent_id,
        "package_tier": "substantive",
    }


class _EndpointEmptyResponse(EmptyLLMResponseError):
    """The WP1 typed EMPTY_RESPONSE transient carrying the attempted
    endpoint identity (failover-ineligible: no availability cooldown armed).
    """

    def __init__(self, endpoint: str) -> None:
        super().__init__("beta", "main-wire")
        self.endpoint = endpoint


class _EndpointIncompleteStream(IncompleteStreamError):
    """The WP1 typed INCOMPLETE_STREAM transient carrying the attempted
    endpoint identity. Failover-ineligible (no availability cooldown armed)
    and never replayed by the in-call gateway retry, so a resumed successor
    can launch on the same deployment.
    """

    def __init__(self, endpoint: str) -> None:
        super().__init__("beta", "main-wire")
        self.endpoint = endpoint


class _WriteThenEmptyBackend(FakeBackend):
    """The first logical call answers with a settled todo write (so the
    failure envelope's tool effects are SETTLED -> ELIGIBLE); every later
    call raises the endpoint-carrying EMPTY_RESPONSE transient.
    """

    def __init__(self, fault: Exception) -> None:
        super().__init__([])
        self._fault = fault
        self.calls = 0

    async def complete(self, **kwargs: Any) -> LLMChunk:
        self.calls += 1
        if self.calls == 1:
            return mock_llm_chunk(
                content="", tool_calls=[_todo_write_call("call-1", "package work")]
            )
        raise self._fault

    async def complete_streaming(self, **kwargs: Any):
        self.calls += 1
        if self.calls == 1:
            yield mock_llm_chunk(
                content="", tool_calls=[_todo_write_call("call-1", "package work")]
            )
            return
        raise self._fault


class _WriteFaultThenCompleteBackend(FakeBackend):
    """call 1: a settled todo write; call 2: the endpoint-carrying fault (the
    first run fails); call 3: terminal content (a resumed successor run
    completes).
    """

    def __init__(self, fault: Exception) -> None:
        super().__init__([])
        self._fault = fault
        self.calls = 0

    async def complete(self, **kwargs: Any) -> LLMChunk:
        self.calls += 1
        if self.calls == 1:
            return mock_llm_chunk(
                content="", tool_calls=[_todo_write_call("call-1", "package work")]
            )
        if self.calls == 2:
            raise self._fault
        return mock_llm_chunk(content="resumed done")

    async def complete_streaming(self, **kwargs: Any):
        self.calls += 1
        if self.calls == 1:
            yield mock_llm_chunk(
                content="", tool_calls=[_todo_write_call("call-1", "package work")]
            )
            return
        if self.calls == 2:
            raise self._fault
        yield mock_llm_chunk(content="resumed done")


def _release_wakeup(registry: SessionRuntimeRegistry, delay: float) -> None:
    """Release the first gated wakeup arm for ``delay`` (the controller
    harness's wakeup driver, inlined for the registry-level harness).
    """
    wakeup = cast(_GatedWakeup, registry._wakeup)
    for index, call in enumerate(wakeup.calls):
        if call == delay:
            wakeup.gates[index].set()
            return
    raise AssertionError(f"no pending wakeup arm for delay {delay!r}")


def _pending_wakeup(registry: SessionRuntimeRegistry, delay: float) -> bool:
    """Whether a wakeup arm for ``delay`` is still waiting on its gate."""
    wakeup = cast(_GatedWakeup, registry._wakeup)
    return any(
        call == delay and not gate.is_set()
        for call, gate in zip(wakeup.calls, wakeup.gates, strict=True)
    )


def _release_pending_wakeup(registry: SessionRuntimeRegistry, delay: float) -> None:
    """Release the next STILL-PENDING wakeup arm for ``delay``.

    The shared ``_release_wakeup`` re-sets the FIRST matching gate, which is
    already released when a second sequential admission needs the next one.
    """
    wakeup = cast(_GatedWakeup, registry._wakeup)
    for index, call in enumerate(wakeup.calls):
        if call == delay and not wakeup.gates[index].is_set():
            wakeup.gates[index].set()
            return
    raise AssertionError(f"no pending wakeup arm for delay {delay!r}")


class _WriterTokenView:
    """Minimal result-state stub for the package-exclusivity pin (review R6)."""

    def __init__(self, records: dict[str, AgentRecord]) -> None:
        self._agent_records = records
        self._result_write_tokens: dict[tuple[str, str], object] = {}


def _package_record(agent_id: str, package_id: str | None) -> AgentRecord:
    return AgentRecord(
        agent_id=agent_id,
        identity="coder",
        session_id=f"session-{agent_id}",
        runtime=cast(Any, MagicMock()),
        root_generation=0,
        work_package_id=package_id,
    )


@pytest.mark.asyncio
async def test_claim_admits_new_agent_with_lineage_milestone_notice_and_handoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: a valid replacement claim is admitted at failure time —
    the replacement gets a NEW agent_id carrying the predecessor/package
    lineage, the acknowledgment carries the validated decision and its
    milestone notice (with the honest cost implication), the old recovery is
    disabled (recovery_superseded), the per-package budget is consumed, and
    check_agents shows the lineage pair.
    """
    registry, parent = await _replacement_registry(
        monkeypatch,
        RecoveryConfig(),
        [
            _WriteThenFaultBackend(_EndpointTimeout("https://beta.test/v1")),
            FakeBackend(mock_llm_chunk(content="replacement done")),
        ],
    )
    try:
        agent_id, _run_id, failed = await _launch_failed_package(registry, parent)
        envelope = failed.failure
        assert envelope is not None
        assert envelope.failure.failure_class is FailureClass.PROVIDER_STALL
        assert envelope.deployment.endpoint == "beta.test"
        assert envelope.recovery.eligibility.value == "eligible"

        ack = await _launch(
            registry,
            parent,
            tool_call_id="claim",
            config=_claim_config(
                model="alt-model", work_package_id="wp-1", predecessor_agent_id=agent_id
            ),
        )

        # A NEW agent_id with the predecessor/package lineage.
        assert ack.agent_id is not None and ack.agent_id != agent_id
        assert ack.work_package_id == "wp-1"
        assert ack.predecessor_agent_id == agent_id
        # The validated decision + milestone notice on the acknowledgment.
        assert ack.replacement is not None
        assert ack.replacement.disposition == "selected"
        assert ack.replacement.candidate is not None
        assert ack.replacement.candidate.base_model == "alt-model"
        assert "Replacement milestone" in ack.response
        assert "alt-model" in ack.response
        assert "endpoint-independent" in ack.response
        assert "$5.00/M" in ack.response and "$3.00/M" in ack.response
        # The old recovery is disabled by the handoff.
        predecessor = registry._agent_records[agent_id]
        assert predecessor.recovery_superseded is True
        # The per-package budget is consumed (and spans replacements).
        assert registry._package_replacements["wp-1"] == 1
        # check_agents shows the lineage pair on the new agent's summary.
        summaries = await registry.check_agents()
        replacement = next(item for item in summaries if item.agent_id == ack.agent_id)
        assert replacement.work_package_id == "wp-1"
        assert replacement.predecessor_agent_id == agent_id
        # The failure envelope the claim validated against is the
        # predecessor's latest failed run's envelope.
        assert envelope.work_package_id is None  # the predecessor was unclaimed
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_claim_rejected_for_same_endpoint_substitution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A claim naming a model on the ATTEMPTED failed endpoint is rejected
    (same-endpoint different-model substitution) and admits no new agent.
    """
    registry, parent = await _replacement_registry(
        monkeypatch,
        RecoveryConfig(),
        [_WriteThenFaultBackend(_EndpointTimeout("https://beta.test/v1"))],
    )
    try:
        agent_id, _run_id, _failed = await _launch_failed_package(registry, parent)
        before = set(registry._agent_records)
        with pytest.raises(ReplacementPolicyError) as excinfo:
            await _launch(
                registry,
                parent,
                tool_call_id="claim",
                config=_claim_config(
                    model="main-model",
                    work_package_id="wp-1",
                    predecessor_agent_id=agent_id,
                ),
            )
        assert "same_endpoint_substitution_rejected" in excinfo.value.reason_codes
        assert set(registry._agent_records) == before  # no new agent admitted
        assert registry._package_replacements == {}  # no budget consumed
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_claim_rejected_for_luna_on_substantive_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A claim naming the mechanical-tier (luna-class) model for a
    substantive package is rejected; the mechanical set comes from the
    configured purpose routes' mechanical-tier targets.
    """
    registry, parent = await _replacement_registry(
        monkeypatch,
        RecoveryConfig(),
        [_WriteThenFaultBackend(_EndpointTimeout("https://beta.test/v1"))],
        routes=_MECHANICAL_LUNA_ROUTES,
    )
    try:
        agent_id, _run_id, _failed = await _launch_failed_package(registry, parent)
        with pytest.raises(ReplacementPolicyError) as excinfo:
            await _launch(
                registry,
                parent,
                tool_call_id="claim",
                config=_claim_config(
                    model="luna-model",
                    work_package_id="wp-1",
                    predecessor_agent_id=agent_id,
                ),
            )
        assert (
            "mechanical_tier_rejected_for_substantive_package"
            in excinfo.value.reason_codes
        )
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_claim_observes_the_launch_resolved_package_tier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The launch-resolved effective tier reaches the recovery boundary: a
    package launched through a mechanical-tier route is recorded mechanical,
    so a claim naming no package_tier is judged against the TRUE tier and a
    mechanical replacement is selectable (before the tier reached the
    boundary the package defaulted to substantive and the claim was
    rejected as a mechanical-for-substantive substitution).
    """
    registry, parent = await _replacement_registry(
        monkeypatch,
        RecoveryConfig(),
        [
            _WriteThenFaultBackend(_EndpointTimeout("https://alpha.test/v1")),
            FakeBackend(mock_llm_chunk(content="replacement done")),
        ],
        routes=_TIER_ROUTES,
    )
    try:
        # The package launches through the mechanical-tier route: the child
        # commits the route's luna-model and the record carries the tier.
        launch = await _background_result(
            registry,
            TaskArgs(
                task="package work",
                identity="coder",
                purpose="mechanical-edit",
                background=True,
                retain=True,
            ),
            InvokeContext(tool_call_id="launch", session_id=parent.session_id),
        )
        assert launch.agent_id is not None and launch.run_id is not None
        failed = await _wait_result(registry, launch.agent_id, launch.run_id)
        assert not failed.completed
        envelope = failed.failure
        assert envelope is not None
        assert envelope.deployment.base_model == "luna-model"
        assert envelope.deployment.endpoint == "alpha.test"
        assert envelope.recovery.eligibility.value == "eligible"
        record = registry._agent_records[launch.agent_id]
        assert record.package_tier is PackageTier.MECHANICAL
        # A claim naming no package_tier: the boundary reads the record's
        # launch-resolved tier, so the mechanical alt-model is selectable.
        ack = await _launch(
            registry,
            parent,
            tool_call_id="claim",
            config={
                "model": "alt-model",
                "work_package_id": "wp-1",
                "predecessor_agent_id": launch.agent_id,
            },
        )
        assert ack.agent_id is not None and ack.agent_id != launch.agent_id
        assert ack.replacement is not None
        assert ack.replacement.disposition == "selected"
        assert ack.replacement.candidate is not None
        assert ack.replacement.candidate.base_model == "alt-model"
        assert ack.replacement.candidate.tier is PackageTier.MECHANICAL
        assert "package tier mechanical" in ack.response
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_claim_without_a_tier_carries_it_into_the_successor_record_and_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An omitted-tier claim is validated against the predecessor's
    established tier, and the successor carries that tier end to end: the
    agent record AND the child loop's launch metadata both observe the
    package's true tier (before the inheritance the successor was recorded
    with the direct-launch substantive default, so later lineage checks
    observed the wrong tier).
    """
    registry, parent = await _replacement_registry(
        monkeypatch,
        RecoveryConfig(),
        [
            _WriteThenFaultBackend(_EndpointTimeout("https://alpha.test/v1")),
            FakeBackend(mock_llm_chunk(content="replacement done")),
        ],
        routes=_TIER_ROUTES,
    )
    try:
        # The mechanical package fails on the alpha endpoint.
        launch = await _background_result(
            registry,
            TaskArgs(
                task="package work",
                identity="coder",
                purpose="mechanical-edit",
                background=True,
                retain=True,
            ),
            InvokeContext(tool_call_id="launch", session_id=parent.session_id),
        )
        assert launch.agent_id is not None and launch.run_id is not None
        failed = await _wait_result(registry, launch.agent_id, launch.run_id)
        assert not failed.completed
        assert (
            registry._agent_records[launch.agent_id].package_tier
            is PackageTier.MECHANICAL
        )
        # The claim names no package_tier: admission validates it against
        # the predecessor's established mechanical tier.
        ack = await _launch(
            registry,
            parent,
            tool_call_id="claim",
            config={
                "model": "alt-model",
                "work_package_id": "wp-1",
                "predecessor_agent_id": launch.agent_id,
            },
        )
        assert ack.agent_id is not None and ack.agent_id != launch.agent_id
        assert ack.replacement is not None
        assert ack.replacement.disposition == "selected"
        # The successor's record and loop carry that same tier.
        successor = registry._agent_records[ack.agent_id]
        assert successor.predecessor_agent_id == launch.agent_id
        assert successor.package_tier is PackageTier.MECHANICAL
        assert (
            successor.runtime.agent_loop.launch_package_tier is PackageTier.MECHANICAL
        )
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_claim_observes_the_route_replacement_capability_marking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The model-level replacement-capability marking reaches the recovery
    boundary: a route target explicitly marking alt-model mechanical-capable
    makes a claim for it on a substantive package fail the capability guard
    (before the marking reached the boundary the tier label alone decided
    and the claim was selected).
    """
    registry, parent = await _replacement_registry(
        monkeypatch,
        RecoveryConfig(),
        [_WriteThenFaultBackend(_EndpointTimeout("https://beta.test/v1"))],
        routes=_CAPABILITY_ROUTES,
    )
    try:
        agent_id, _run_id, failed = await _launch_failed_package(registry, parent)
        assert failed.failure is not None
        assert failed.failure.deployment.endpoint == "beta.test"
        record = registry._agent_records[agent_id]
        assert record.package_tier is PackageTier.SUBSTANTIVE
        with pytest.raises(ReplacementPolicyError) as excinfo:
            await _launch(
                registry,
                parent,
                tool_call_id="claim",
                config=_claim_config(
                    model="alt-model",
                    work_package_id="wp-1",
                    predecessor_agent_id=agent_id,
                ),
            )
        assert "candidate_capability_below_package_tier" in excinfo.value.reason_codes
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_claim_canonicalizes_preset_bound_mechanical_routes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preset references are canonicalized in the candidates: a
    mechanical-tier route target bound via the ``@scout`` preset labels its
    canonical base (luna-model) mechanical, so a claim for it on a
    substantive package is rejected (before canonicalization the preset
    string never matched a candidate and the claim was selected).
    """
    registry, parent = await _replacement_registry(
        monkeypatch,
        RecoveryConfig(),
        [_WriteThenFaultBackend(_EndpointTimeout("https://beta.test/v1"))],
        routes=_PRESET_MECHANICAL_ROUTES,
    )
    try:
        agent_id, _run_id, _failed = await _launch_failed_package(registry, parent)
        with pytest.raises(ReplacementPolicyError) as excinfo:
            await _launch(
                registry,
                parent,
                tool_call_id="claim",
                config=_claim_config(
                    model="luna-model",
                    work_package_id="wp-1",
                    predecessor_agent_id=agent_id,
                ),
            )
        assert (
            "mechanical_tier_rejected_for_substantive_package"
            in excinfo.value.reason_codes
        )
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_claim_with_the_top_level_model_is_admitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A replacement claim names the replacement model at the canonical
    top-level field (the legacy nested config.model is the promoted
    equivalent): the claim is admitted and the new agent commits the
    claimed model.
    """
    registry, parent = await _replacement_registry(
        monkeypatch,
        RecoveryConfig(),
        [
            _WriteThenFaultBackend(_EndpointTimeout("https://beta.test/v1")),
            FakeBackend(mock_llm_chunk(content="replacement done")),
        ],
    )
    try:
        agent_id, _run_id, _failed = await _launch_failed_package(registry, parent)
        ack = await _launch(
            registry,
            parent,
            tool_call_id="claim",
            model="alt-model",
            config={"work_package_id": "wp-1", "predecessor_agent_id": agent_id},
        )
        assert ack.agent_id is not None and ack.agent_id != agent_id
        assert ack.replacement is not None
        assert ack.replacement.disposition == "selected"
        assert ack.replacement.candidate is not None
        assert ack.replacement.candidate.base_model == "alt-model"
        replacement = registry._agent_records[ack.agent_id]
        child = replacement.runtime.agent_loop
        assert child.committed_model is not None
        assert child.committed_model.base_model == "alt-model"
        assert child.committed_model.provider == "gamma"
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_claim_blocked_by_reconcile_required_barrier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The write-ownership transfer happens only after reconciliation: a
    predecessor suspended with a reconcile-required barrier (unknown tool
    effects — no tool calls) blocks the claim until reconcile_result.
    """
    registry, parent = await _replacement_registry(
        monkeypatch,
        RecoveryConfig(),
        [ImmediateFailureBackend(_EndpointTimeout("https://beta.test/v1"))],
    )
    try:
        agent_id, _run_id, failed = await _launch_failed_package(registry, parent)
        envelope = failed.failure
        assert envelope is not None
        assert envelope.recovery.eligibility.value == "reconcile_required"
        with pytest.raises(ReplacementPolicyError) as excinfo:
            await _launch(
                registry,
                parent,
                tool_call_id="claim",
                config=_claim_config(
                    model="alt-model",
                    work_package_id="wp-1",
                    predecessor_agent_id=agent_id,
                ),
            )
        assert "reconcile_required_before_replacement" in excinfo.value.reason_codes
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_package_replacement_budget_exhaustion_parks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The per-package replacement budget spans the package lineage: once
    consumed, further claims for the same package are rejected with
    package_replacements_exhausted (the orchestrator parks the package).
    """
    registry, parent = await _replacement_registry(
        monkeypatch,
        RecoveryConfig(fallback_replacements_max_per_package=1),
        [
            _WriteThenFaultBackend(_EndpointTimeout("https://beta.test/v1")),
            FakeBackend(mock_llm_chunk(content="replacement done")),
        ],
    )
    try:
        agent_id, _run_id, _failed = await _launch_failed_package(registry, parent)
        ack = await _launch(
            registry,
            parent,
            tool_call_id="claim-1",
            config=_claim_config(
                model="alt-model", work_package_id="wp-1", predecessor_agent_id=agent_id
            ),
        )
        assert ack.replacement is not None and ack.replacement.disposition == "selected"
        assert registry._package_replacements["wp-1"] == 1
        with pytest.raises(ReplacementPolicyError) as excinfo:
            await _launch(
                registry,
                parent,
                tool_call_id="claim-2",
                config=_claim_config(
                    model="alt-model",
                    work_package_id="wp-1",
                    predecessor_agent_id=agent_id,
                ),
            )
        assert "package_replacements_exhausted" in excinfo.value.reason_codes
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_handoff_disables_old_recovery_before_write_token_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Handoff ordering: the old recovery is invalidated BEFORE the new run's
    write-ownership token is registered at acceptance.
    """
    registry, parent = await _replacement_registry(
        monkeypatch,
        RecoveryConfig(),
        [
            _WriteThenFaultBackend(_EndpointTimeout("https://beta.test/v1")),
            FakeBackend(mock_llm_chunk(content="replacement done")),
        ],
    )
    order: list[str] = []
    original_invalidate = registry._invalidate_recovery_locked
    original_register = registry._register_writer_locked

    def spy_invalidate(record: Any) -> None:
        order.append("invalidate_recovery")
        original_invalidate(record)

    def spy_register(*args: Any, **kwargs: Any) -> Any:
        order.append("register_writer")
        return original_register(*args, **kwargs)

    try:
        agent_id, _run_id, _failed = await _launch_failed_package(registry, parent)
        # Spy window: the claim admission only. The failed run's own writer
        # registration (before this window) is unrelated to the handoff; the
        # ordering under test is invalidate-before-register WITHIN the claim.
        monkeypatch.setattr(registry, "_invalidate_recovery_locked", spy_invalidate)
        monkeypatch.setattr(registry, "_register_writer_locked", spy_register)
        ack = await _launch(
            registry,
            parent,
            tool_call_id="claim",
            config=_claim_config(
                model="alt-model", work_package_id="wp-1", predecessor_agent_id=agent_id
            ),
        )
        assert ack.replacement is not None
        assert order == ["invalidate_recovery", "register_writer"]
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_budgets_and_counters_span_replacements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Budgets/counters are NOT reset across the replacement: the
    replacement's own failure envelope carries the package lineage (the WP4
    correlation key is the envelope's work_package_id), and the package
    counter still counts the earlier replacement.
    """
    registry, parent = await _replacement_registry(
        monkeypatch,
        RecoveryConfig(fallback_replacements_max_per_package=2),
        [
            _WriteThenFaultBackend(_EndpointTimeout("https://beta.test/v1")),
            _WriteThenFaultBackend(_EndpointTimeout("https://gamma.test/v1")),
        ],
    )
    try:
        agent_id, _run_id, _failed = await _launch_failed_package(registry, parent)
        ack = await _launch(
            registry,
            parent,
            tool_call_id="claim",
            config=_claim_config(
                model="alt-model", work_package_id="wp-1", predecessor_agent_id=agent_id
            ),
        )
        assert ack.agent_id is not None and ack.agent_id != agent_id
        assert ack.run_id is not None
        replacement_failed = await _wait_result(registry, ack.agent_id, ack.run_id)
        assert not replacement_failed.completed
        envelope = replacement_failed.failure
        assert envelope is not None
        # The lineage pair rides the replacement's failure envelope.
        assert envelope.work_package_id == "wp-1"
        assert envelope.predecessor_agent_id == agent_id
        assert envelope.deployment.endpoint == "gamma.test"
        # The budget was not reset: the package counter spans replacements.
        assert registry._package_replacements["wp-1"] == 1
        # The replacement record carries the lineage too.
        record = registry._agent_records[ack.agent_id]
        assert record.work_package_id == "wp-1"
        assert record.predecessor_agent_id == agent_id
        assert record.package_tier is PackageTier.SUBSTANTIVE
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_claim_rejected_when_predecessor_has_a_newer_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review R6: a claim may replace a predecessor only while its failed
    run is the LATEST terminal state — after a resumed successor completes,
    the historical failure no longer grounds a replacement claim.
    """
    registry, parent = await _replacement_registry(
        monkeypatch,
        RecoveryConfig(),
        [
            # INCOMPLETE_STREAM is failover-ineligible and never replayed
            # in-call, so the failure arms no availability cooldown and the
            # resumed successor can launch on the same deployment.
            _WriteFaultThenCompleteBackend(
                _EndpointIncompleteStream("https://beta.test/v1")
            ),
            FakeBackend(mock_llm_chunk(content="replacement done")),
        ],
    )
    try:
        agent_id, run_id, failed = await _launch_failed_package(registry, parent)
        assert failed.failure is not None
        # Resume the suspended predecessor: the successor completes, so the
        # failed run is no longer the latest run.
        resumed = await registry.resume_agent(
            agent_id, run_id, "k", requester_session_id=parent.session_id
        )
        assert resumed.outcome is ResumeOutcome.SCHEDULED
        assert resumed.successor_run_id is not None
        successor = await _wait_result(registry, agent_id, resumed.successor_run_id)
        assert successor.completed
        record = registry._agent_records[agent_id]
        assert record.latest_run_id == resumed.successor_run_id

        # The claim on the historical failure is rejected: the predecessor's
        # latest run is not the failed one.
        before = set(registry._agent_records)
        with pytest.raises(ReplacementPolicyError) as excinfo:
            await _launch(
                registry,
                parent,
                tool_call_id="claim",
                config=_claim_config(
                    model="alt-model",
                    work_package_id="wp-1",
                    predecessor_agent_id=agent_id,
                ),
            )
        assert "predecessor_run_not_terminal" in excinfo.value.reason_codes
        assert set(registry._agent_records) == before  # no new agent admitted
        assert registry._package_replacements == {}  # no budget consumed
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_second_claim_on_superseded_predecessor_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review R6: exclusive package ownership — once a claim is admitted the
    predecessor is superseded and can never ground a SECOND replacement
    (repeated claims must not create multiple replacements).
    """
    registry, parent = await _replacement_registry(
        monkeypatch,
        RecoveryConfig(),
        [
            _WriteThenFaultBackend(_EndpointTimeout("https://beta.test/v1")),
            FakeBackend(mock_llm_chunk(content="replacement done")),
            FakeBackend(mock_llm_chunk(content="second replacement")),
        ],
    )
    try:
        agent_id, _run_id, _failed = await _launch_failed_package(registry, parent)
        ack = await _launch(
            registry,
            parent,
            tool_call_id="claim-1",
            config=_claim_config(
                model="alt-model", work_package_id="wp-1", predecessor_agent_id=agent_id
            ),
        )
        assert ack.agent_id is not None and ack.agent_id != agent_id
        assert registry._package_replacements["wp-1"] == 1
        predecessor = registry._agent_records[agent_id]
        assert predecessor.recovery_superseded is True

        # A second claim on the same (now superseded) predecessor is
        # rejected — the package already has its replacement.
        before = set(registry._agent_records)
        with pytest.raises(ReplacementPolicyError) as excinfo:
            await _launch(
                registry,
                parent,
                tool_call_id="claim-2",
                config=_claim_config(
                    model="alt-model",
                    work_package_id="wp-1",
                    predecessor_agent_id=agent_id,
                ),
            )
        assert "predecessor_already_superseded" in excinfo.value.reason_codes
        assert set(registry._agent_records) == before  # no second replacement
        assert registry._package_replacements["wp-1"] == 1  # no extra budget
    finally:
        await registry.drain_children()
        await parent.aclose()


def test_writer_tokens_are_package_exclusive() -> None:
    """Review R6: writer tokens are package-exclusive, not agent-keyed —
    registering a replacement's token revokes every OTHER live token whose
    record carries the same work package, so a superseded predecessor or an
    earlier replacement still in flight can never publish over the
    package's current owner. Unrelated packages and unclaimed agents keep
    their tokens.
    """
    view = _WriterTokenView({
        "agent-a": _package_record("agent-a", "wp-1"),
        "agent-b": _package_record("agent-b", "wp-1"),
        "agent-c": _package_record("agent-c", "wp-2"),
        "agent-d": _package_record("agent-d", None),
    })
    results = RunResults(cast(ResultsStateView, view))
    results.register_writer_locked("agent-a", "run-a1", object())
    results.register_writer_locked("agent-c", "run-c1", object())
    results.register_writer_locked("agent-d", "run-d1", object())
    assert len(view._result_write_tokens) == 3
    # The package's new owner registers: the same-package predecessor token
    # is revoked; the unrelated package and the unclaimed agent keep theirs.
    token = object()
    results.register_writer_locked("agent-b", "run-b2", token)
    assert view._result_write_tokens[("agent-b", "run-b2")] is token
    assert ("agent-a", "run-a1") not in view._result_write_tokens
    assert ("agent-c", "run-c1") in view._result_write_tokens
    assert ("agent-d", "run-d1") in view._result_write_tokens


@pytest.mark.asyncio
async def test_replacement_inherits_predecessor_cumulative_stats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review R7: a replacement creates a fresh loop whose AgentStats start
    at zero while the middleware limits evaluate those LOCAL stats — the
    predecessor's cumulative consumption must transfer, or replacement
    renews every budget the predecessor had already consumed. Per-turn
    measurements stay local to the fresh loop.
    """
    registry, parent = await _replacement_registry(
        monkeypatch,
        RecoveryConfig(),
        [
            _WriteThenFaultBackend(_EndpointTimeout("https://beta.test/v1")),
            FakeBackend(mock_llm_chunk(content="replacement done")),
        ],
    )
    try:
        agent_id, _run_id, failed = await _launch_failed_package(registry, parent)
        assert failed.failure is not None
        predecessor_stats = registry._agent_records[agent_id].runtime.agent_loop.stats
        # The failed run consumed real budget: one settled tool call and at
        # least one accounted model call.
        assert predecessor_stats.tool_calls_succeeded >= 1
        assert predecessor_stats.session_prompt_tokens > 0
        assert predecessor_stats.session_completion_tokens > 0
        assert predecessor_stats.known_cost_total > 0
        assert predecessor_stats.steps >= 1
        ack = await _launch(
            registry,
            parent,
            tool_call_id="claim",
            config=_claim_config(
                model="alt-model", work_package_id="wp-1", predecessor_agent_id=agent_id
            ),
        )
        assert ack.agent_id is not None and ack.agent_id != agent_id
        replacement_stats = registry._agent_records[
            ack.agent_id
        ].runtime.agent_loop.stats
        # The cumulative counters carry the predecessor's consumption.
        assert (
            replacement_stats.tool_calls_succeeded
            >= predecessor_stats.tool_calls_succeeded
        )
        assert (
            replacement_stats.session_prompt_tokens
            >= predecessor_stats.session_prompt_tokens
        )
        assert (
            replacement_stats.session_completion_tokens
            >= predecessor_stats.session_completion_tokens
        )
        assert (
            replacement_stats.session_cached_tokens
            >= predecessor_stats.session_cached_tokens
        )
        assert replacement_stats.known_cost_total >= predecessor_stats.known_cost_total
        assert replacement_stats.steps >= predecessor_stats.steps
        # Per-turn measurements stay local to the fresh loop.
        assert replacement_stats.context_tokens == 0
        assert replacement_stats.last_turn_prompt_tokens == 0
        assert replacement_stats.last_turn_completion_tokens == 0
        assert replacement_stats.tokens_per_second == 0.0
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_recurrence_counter_migrates_to_the_package_key_on_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review R7: an initially-unclaimed package's recurrence key changes
    ``brief:...`` -> ``pkg:...`` when a replacement claim establishes the
    package identity — the earlier automatic-admission count MIGRATES with
    it (never resets). With ``auto_retries_max=1``, agent A's one admitted
    automatic resume is charged to the brief key; the claimed replacement
    B's first failure is then refused at the lineage gate (no schedule)
    because the migrated count already exhausts the package budget.
    """
    registry, parent = await _replacement_registry(
        monkeypatch,
        RecoveryConfig(
            enabled=True,
            auto_retries_max=1,
            cooldowns_seconds=[5.0],
            jitter_fraction=0.0,
        ),
        [
            _WriteThenEmptyBackend(_EndpointEmptyResponse("https://beta.test/v1")),
            _WriteThenEmptyBackend(_EndpointEmptyResponse("https://beta.test/v1")),
        ],
    )
    try:
        controller = registry._recovery_controller
        # A fails (endpoint-carrying EMPTY_RESPONSE after a settled todo
        # write -> ELIGIBLE) and suspends; the controller schedules.
        agent_a, run_a, failed_a = await _launch_failed_package(registry, parent)
        envelope_a = failed_a.failure
        assert envelope_a is not None
        assert envelope_a.recovery.eligibility.value == "eligible"
        assert envelope_a.deployment.endpoint == "beta.test"
        record_a = registry._agent_records[agent_a]
        await _until(lambda: record_a.state is _AgentState.SUSPENDED)
        wakeup = cast(_GatedWakeup, registry._wakeup)
        await _until(lambda: bool(wakeup.calls))
        # The auto-resume is admitted: the brief-keyed recurrence count is 1.
        _release_wakeup(registry, 5.0)
        await _until(lambda: record_a.latest_run_id not in (None, run_a))
        successor_run = record_a.latest_run_id
        assert successor_run is not None
        successor = await _wait_result(registry, agent_a, successor_run)
        assert not successor.completed
        # A's successor fails too: the episode budget (1) is exhausted.
        await _until(lambda: controller._episodes == {})
        await _until(lambda: record_a.state is _AgentState.SUSPENDED)
        assert controller._lineage == {"brief:coder:package work": 1}

        # The claim: a NEW agent B carrying the package identity wp-1.
        ack = await _launch(
            registry,
            parent,
            tool_call_id="claim",
            config=_claim_config(
                model="alt-model", work_package_id="wp-1", predecessor_agent_id=agent_a
            ),
        )
        agent_b = ack.agent_id
        assert agent_b is not None and agent_b != agent_a
        assert ack.run_id is not None
        assert ack.work_package_id == "wp-1"
        # B fails (same fault shape) and suspends.
        failed_b = await _wait_result(registry, agent_b, ack.run_id)
        assert not failed_b.completed
        record_b = registry._agent_records[agent_b]
        await _until(lambda: record_b.state is _AgentState.SUSPENDED)
        # Refused at the lineage gate: the migrated count exhausts the
        # package budget — no episode, no successor for B, and the count
        # lives under the PACKAGE key now (the brief key is gone).
        assert controller._episodes == {}
        runs_b = [
            run
            for run in [record_b.current_run, *record_b.run_history]
            if run is not None
        ]
        assert [run.run_id for run in runs_b] == [ack.run_id]
        assert controller._lineage == {"pkg:wp-1": 1}
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_replacement_migration_never_steals_a_same_brief_lineage_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Round-2 V5: the R7 lineage-key migration moves only the count the
    PREDECESSOR lineage itself contributed. A brief key
    (``brief:{identity}:{summary}``) can be shared by an UNRELATED lineage
    with the same brief; the migration must never pop that other lineage's
    count into the package key.

    With ``auto_retries_max=2``: the UNRELATED agent C (same brief) is
    auto-admitted once (brief count 1; its successor completes). Agent A
    (the same brief) is then auto-admitted once (brief count 2) and A's own
    successor is refused at the lineage gate (the shared brief budget is
    exhausted) — A suspends on its failed LATEST run. The claimed
    replacement B (work_package_id wp-1, predecessor A) fails: the
    migration folds A's OWN count (1) into the package key and leaves C's
    count under the brief key.

    FAILS pre-fix: the migration popped the whole shared brief-keyed
    count (2) — C's count was stolen into the package key
    (``{"pkg:wp-1": 2}``, with B refused at the gate, instead of
    ``{"brief:coder:package work": 1, "pkg:wp-1": 1}`` with B scheduled).
    """
    registry, parent = await _replacement_registry(
        monkeypatch,
        RecoveryConfig(
            enabled=True,
            auto_retries_max=2,
            cooldowns_seconds=[5.0, 5.0],
            jitter_fraction=0.0,
        ),
        [
            # INCOMPLETE_STREAM is failover-ineligible and never replayed
            # in-call, so C's first run fails deterministically (ELIGIBLE:
            # its settled todo write) and the resumed successor completes.
            _WriteFaultThenCompleteBackend(
                _EndpointIncompleteStream("https://beta.test/v1")
            ),
            _WriteThenEmptyBackend(_EndpointEmptyResponse("https://beta.test/v1")),
            _WriteThenEmptyBackend(_EndpointEmptyResponse("https://beta.test/v1")),
        ],
    )
    try:
        controller = registry._recovery_controller

        # The UNRELATED same-brief lineage C: its run fails (ELIGIBLE) and
        # suspends; the auto-resume is admitted (brief count 1) and the
        # successor COMPLETES (C stays resident — the launch retains it).
        agent_c, run_c, failed_c = await _launch_failed_package(
            registry, parent, tool_call_id="launch-c"
        )
        envelope_c = failed_c.failure
        assert envelope_c is not None
        assert envelope_c.recovery.eligibility.value == "eligible"
        record_c = registry._agent_records[agent_c]
        await _until(lambda: record_c.state is _AgentState.SUSPENDED)
        await _until(lambda: _pending_wakeup(registry, 5.0))
        _release_pending_wakeup(registry, 5.0)
        await _until(lambda: record_c.latest_run_id not in (None, run_c))
        successor_c = record_c.latest_run_id
        assert successor_c is not None
        done_c = await _wait_result(registry, agent_c, successor_c)
        assert done_c.completed
        await _until(lambda: controller._lineage == {"brief:coder:package work": 1})

        # Agent A (the SAME brief — the shared brief key): admitted once
        # (brief count 2); A's successor fails and is refused at the
        # lineage gate, so A suspends on its failed LATEST run (the claim
        # below needs the failed run to be the latest terminal state).
        agent_a, run_a, failed_a = await _launch_failed_package(
            registry, parent, tool_call_id="launch-a"
        )
        envelope_a = failed_a.failure
        assert envelope_a is not None
        assert envelope_a.recovery.eligibility.value == "eligible"
        record_a = registry._agent_records[agent_a]
        assert agent_a != agent_c
        await _until(lambda: record_a.state is _AgentState.SUSPENDED)
        await _until(lambda: _pending_wakeup(registry, 5.0))
        _release_pending_wakeup(registry, 5.0)
        await _until(lambda: record_a.latest_run_id not in (None, run_a))
        successor_a = record_a.latest_run_id
        assert successor_a is not None
        failed_successor = await _wait_result(registry, agent_a, successor_a)
        assert not failed_successor.completed
        await _until(lambda: controller._episodes == {})
        await _until(lambda: record_a.state is _AgentState.SUSPENDED)
        assert controller._lineage == {"brief:coder:package work": 2}

        # The claim: agent B carries the package identity wp-1 with
        # predecessor A. B fails (same fault shape) and suspends.
        ack = await _launch(
            registry,
            parent,
            tool_call_id="claim",
            config=_claim_config(
                model="alt-model", work_package_id="wp-1", predecessor_agent_id=agent_a
            ),
        )
        agent_b = ack.agent_id
        assert agent_b is not None and agent_b not in (agent_a, agent_c)
        assert ack.run_id is not None
        assert ack.work_package_id == "wp-1"
        failed_b = await _wait_result(registry, agent_b, ack.run_id)
        assert not failed_b.completed
        record_b = registry._agent_records[agent_b]
        await _until(lambda: record_b.state is _AgentState.SUSPENDED)

        # The migration folded ONLY A's own count into the package key:
        # C's same-brief count survives under the brief key (1), and the
        # package key carries exactly A's migrated count (1).
        assert controller._lineage == {"brief:coder:package work": 1, "pkg:wp-1": 1}
        assert controller._lineage_agent_counts == {
            (agent_c, "brief:coder:package work"): 1
        }
        # B's gate sees the honest package count (1 < 2): B is scheduled,
        # not refused — the package budget was not inflated by the theft.
        await _until(lambda: agent_b in controller._episodes)
        runs_b = [
            run
            for run in [record_b.current_run, *record_b.run_history]
            if run is not None
        ]
        assert [run.run_id for run in runs_b] == [ack.run_id]
    finally:
        await registry.drain_children()
        await parent.aclose()


# ---------------------------------------------------------------------------
# The unattended-resilience rule (the v0.5.5 production incident)
# ---------------------------------------------------------------------------


def _shipped_mechanical_models() -> frozenset[str]:
    # The shipped mechanical-tier route targets bind the cost-tier default;
    # with the shipped catalog's price data the cheapest capable model is
    # mistral-large-4, so the mechanical set is that model.
    resolver = ModelResolver(CatalogSnapshot(SHIPPED_CATALOG, "shipped"))
    return mechanical_base_models_from_routes(
        shipped_routes(),
        session_model="glm-5-3",
        catalog_prices=resolver.catalog_prices(),
    )


def test_unattended_resilience_zero_ask_under_repeated_infrastructure_failure() -> None:
    """THE acceptance test: a controlled repeated-infrastructure-failure
    scenario produces ZERO ask_user_question invocations, continues
    independent work where available, and otherwise ends cleanly with the
    consolidated resumable report carrying U's deferred records.

    The driver follows the cli.md rule using the REAL policy functions. Two
    packages fail with repeated provider stalls: wp-1 fails on a gateway
    endpoint (an endpoint-independent replacement exists in the shipped
    catalog, so independent work continues) and wp-2 fails on
    api.mistral.ai — where BOTH shipped models sit (the A.1b case), so its
    eligible set is empty and it parks.
    """
    asks: list[str] = []

    def ask_user_question(*_args: Any, **_kwargs: Any) -> None:
        asks.append("asked")
        raise AssertionError(
            "ask_user_question must never be called for infrastructure failures"
        )

    mechanical = _shipped_mechanical_models()
    # The shipped mechanical-tier routes default to the cheapest capable
    # model under the shipped catalog's prices (mistral-large-4).
    assert mechanical == frozenset({"mistral-large-4"})
    candidates = candidates_from_catalog(
        SHIPPED_CATALOG, mechanical_base_models=mechanical
    )
    config = RecoveryConfig()

    failures = [
        _envelope(
            agent_id="agent-1",
            run_id="run-1",
            endpoint="https://gateway.test/v1",
            base_model="glm-5-3",
            work_package_id="wp-1",
        ),
        _envelope(
            agent_id="agent-2",
            run_id="run-2",
            endpoint="https://api.mistral.ai/v1",
            base_model="mistral-large-4",
            work_package_id="wp-2",
        ),
    ]
    for envelope in failures:
        assert is_infrastructure_failure(envelope)

    selected: list[ReplacementDecision] = []
    parked: list[ParkedPackage] = []
    for envelope in failures:
        assert is_infrastructure_failure(envelope)
        eligible = eligible_replacement_candidates(
            envelope=envelope,
            package_tier=PackageTier.SUBSTANTIVE,
            candidates=candidates,
            config=config,
        )
        if eligible:
            # The orchestrator chooses AT FAILURE TIME from the validated set.
            decision = validate_replacement_selection(
                envelope=envelope,
                package_tier=PackageTier.SUBSTANTIVE,
                claimed=eligible[0],
                candidates=candidates,
                config=config,
            )
            assert decision.disposition == "selected"
            assert decision.notice  # the milestone notice with the cost
            selected.append(decision)
            continue  # independent work continues
        # Park-and-continue: no eligible route; never a user question.
        decision = parked_decision(
            envelope=envelope,
            reason_codes=["no_endpoint_independent_replacement"],
            detail="every catalog model shares the failed endpoint",
        )
        assert decision.disposition == "parked"
        parked.append(
            ParkedPackage.from_envelope(
                envelope, reason_codes=decision.reason_codes, detail=decision.notice
            )
        )

    store = DeferredQuestionStore()
    _record_deferred(store, "req-1", _echoed_question("q1"))
    report = build_consolidated_blocker_report(
        parked=parked, deferred_questions=store, session_id="session-1"
    )

    # ZERO ask_user_question invocations across the whole scenario.
    assert asks == []
    # Independent work continued: wp-1 was replaced on an independent
    # endpoint. Under the shipped cost-tier defaults glm-5-3 is
    # substantive-tier (the mechanical set is the cheapest capable model,
    # mistral-large-4), so the failed model's own catalog deployment on
    # api.mistral.ai is the one endpoint-independent eligible candidate.
    assert len(selected) == 1
    assert selected[0].candidate is not None
    assert selected[0].candidate.base_model == "glm-5-3"
    assert selected[0].candidate.tier is PackageTier.SUBSTANTIVE
    assert selected[0].failed_endpoint == "gateway.test"
    # wp-2 parked (the shipped catalog has no endpoint-independent route).
    assert len(parked) == 1
    assert parked[0].work_package_id == "wp-2"
    # The consolidated report ends the turn cleanly with resumable state.
    assert UNATTENDED_RESILIENCE_MARKER in report
    assert "wp-2" in report
    assert "deferred preference (not a blocker)" in report
    assert "No open interactive question is pending" in report
    assert "Resume with one instruction" in report


def test_unattended_resilience_still_asks_for_authorization_decisions() -> None:
    """The rule never overrides genuine user questions: an authorization-class
    failure is NOT infrastructure — it remains a genuine user question, and
    the policy never substitutes onto it.
    """
    asks: list[str] = []

    def ask_user_question(*_args: Any, **_kwargs: Any) -> None:
        asks.append("asked")

    envelope = _with_class(_envelope(), FailureClass.AUTHORIZATION)
    assert not is_infrastructure_failure(envelope)
    # The driver's rule: only infrastructure failures park-and-continue;
    # authorization/scope/safety decisions remain genuine user questions.
    if not is_infrastructure_failure(envelope):
        ask_user_question()
    assert asks == ["asked"]
    # And the policy rejects any substitution onto a protected class.
    decision = validate_replacement_selection(
        envelope=envelope,
        package_tier=PackageTier.SUBSTANTIVE,
        claimed=_ALT,
        candidates=[_ALT],
        config=RecoveryConfig(),
    )
    assert decision.disposition == "rejected"
    assert "protected_class_no_replacement" in decision.reason_codes
