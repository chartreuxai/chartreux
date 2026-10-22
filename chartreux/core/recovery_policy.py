"""Failure-time replacement policy for subagents (WP5; handoff §2.2, resolved Q8).

Fallback bindings are ORCHESTRATION POLICY, not a static pre-authorized table:
the orchestrator chooses a replacement deployment WHEN a failure is surfaced,
from the validated eligible sets this module builds, subject to the hard
constraints the release evidence established:

- (a) endpoint independence — a replacement must not share the failed call's
  API endpoint (A.1b: models sharing one endpoint fail together under
  endpoint-level throttling, so a same-endpoint substitution — an alias or a
  different model on the same endpoint — is not a real fallback). Endpoint
  identity is enforced against the ATTEMPTED failed endpoint carried by the
  failure envelope; an unknown identity cannot prove independence.
- (b) tier capability — a mechanical-tier model (luna-class) is never an
  eligible replacement for a substantive package.
- (b2) candidate capability — replacement ELIGIBILITY is an intrinsic
  model-level capability, explicitly marked execution-capable for the
  requested work, and is evaluated SEPARATELY from the package tier
  (interfaces.md settled decision): the package tier remains a separate
  constraint/label and never changes the candidate's capability, and a model
  bound on both mechanical and substantive route targets carries ONE canonical
  capability, not two target-derived identities.
- (c) cost awareness — every substitution emits a milestone notice stating the
  cost implication honestly (including when pricing is unavailable).

There is NO dispatch-time substitution table and NO automatic cross-model
fallback: the policy validates claims at launch admission and provides the
validated eligible sets; the choice is the orchestrator's, at failure time.

This module also owns the unattended-resilience rule's harness side: the
consolidated resumable blocker report (park-and-continue) and the greppable
marker shared with the orchestrator prompt
(``core/prompts/identities/orchestrator.md``).
Infrastructure/harness failures NEVER produce a user question — the decision
vocabulary is ``selected`` | ``rejected`` | ``parked``, never ``ask``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field

from chartreux.core.config.models import RecoveryConfig
from chartreux.core.deferred_questions import DeferredQuestionStore
from chartreux.core.dispatch.route_models import PurposeRoute, ReplacementCapability
from chartreux.core.dispatch.routes import SESSION_MODEL
from chartreux.core.recovery_models import (
    FailureClass,
    FailureEnvelope,
    PackageTier,
    canonical_endpoint_identity,
)

if TYPE_CHECKING:
    from chartreux.core.model_catalog.schema import ModelCatalog

__all__ = [
    "INFRASTRUCTURE_FAILURE_CLASSES",
    "UNATTENDED_RESILIENCE_MARKER",
    "UNATTENDED_RESILIENCE_RULE",
    "ParkedPackage",
    "ReplacementCandidate",
    "ReplacementDecision",
    "ReplacementPolicyError",
    "build_consolidated_blocker_report",
    "candidates_from_catalog",
    "eligible_replacement_candidates",
    "is_infrastructure_failure",
    "mechanical_base_models_from_routes",
    "parked_decision",
    "route_recovery_facts",
    "validate_replacement_selection",
]

#: Greppable marker for the unattended-resilience rule. The same marker appears
#: in the orchestrator prompt (``core/prompts/identities/orchestrator.md``) and
#: in the consolidated blocker report, so the rule is findable from either side.
UNATTENDED_RESILIENCE_MARKER = "UNATTENDED-RESILIENCE"

#: The unattended-resilience rule in one sentence (the prompt carries the full
#: paragraph). Infrastructure/harness failures never produce a user question.
UNATTENDED_RESILIENCE_RULE = (
    "Infrastructure and harness failures never produce ask_user_question: park "
    "the failed package and continue independent work; only when eligible "
    "routes and budgets are exhausted, end the turn with one consolidated "
    "blocker report carrying resumable state. Authorization, scope, safety, "
    "and destructive-action decisions remain genuine user questions."
)

#: Failure classes the unattended-resilience rule covers: provider/harness
#: failures, never the protected classes (those are genuine stops/decisions).
INFRASTRUCTURE_FAILURE_CLASSES = frozenset({
    FailureClass.PROVIDER_STALL,
    FailureClass.PROVIDER_OVERLOADED,
    FailureClass.PROVIDER_RATE_LIMIT,
    FailureClass.LAUNCH_FAILURE,
    FailureClass.INCOMPLETE_STREAM,
    FailureClass.EMPTY_RESPONSE,
    FailureClass.UNKNOWN,
    FailureClass.INTERNAL,
})

_PROTECTED_NO_REPLACEMENT_CLASSES = frozenset({
    FailureClass.CANCELLATION,
    FailureClass.AUTHORIZATION,
    FailureClass.PERMANENT_CONFIG,
    FailureClass.BUDGET_STOP,
    FailureClass.TASK,
})


class ReplacementCandidate(BaseModel):
    """One validated eligible-set member (a concrete catalog deployment)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    base_model: str
    provider: str
    wire_name: str | None = None
    # Canonical redacted endpoint identity (recovery_models); None = unknown.
    endpoint: str | None = None
    tier: PackageTier = PackageTier.SUBSTANTIVE
    # The model-level replacement capability — an EXPLICIT, intrinsic marking,
    # independent of the package tier (interfaces.md settled decision).
    # ``None`` derives the capability from the tier label (the legacy
    # slot-purpose inference path); an explicit marking — including NONE — is
    # honored as-is, so eligibility never depends on a tier guess.
    capability: ReplacementCapability | None = None
    # Output price per million tokens; None = pricing unavailable (never 0.0).
    output_price_per_million: float | None = None

    @property
    def effective_capability(self) -> ReplacementCapability:
        """The capability axis: the explicit marking, else the tier label."""
        if self.capability is not None:
            return self.capability
        if self.tier is PackageTier.MECHANICAL:
            return ReplacementCapability.MECHANICAL
        return ReplacementCapability.SUBSTANTIVE


class ReplacementDecision(BaseModel):
    """The typed outcome of the failure-time replacement policy.

    ``disposition`` is ``selected`` | ``rejected`` | ``parked`` — the vocabulary
    never contains a user-question action: infrastructure/harness failures are
    never escalated to ``ask_user_question`` (the unattended-resilience rule).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    disposition: Literal["selected", "rejected", "parked"]
    reason_codes: list[str] = Field(default_factory=list)
    candidate: ReplacementCandidate | None = None
    # Honest cost text; empty unless a substitution was selected.
    cost_implication: str = ""
    # The milestone notice (every substitution emits one with its cost
    # implication; rejections/parkings carry the reason instead).
    notice: str = ""
    failed_base_model: str | None = None
    failed_endpoint: str | None = None
    predecessor_agent_id: str | None = None
    work_package_id: str | None = None


class ReplacementPolicyError(ValueError):
    """A claimed replacement violates the failure-time replacement policy."""

    def __init__(self, reason_codes: Sequence[str], message: str) -> None:
        self.reason_codes = list(reason_codes)
        super().__init__(f"{message} [{', '.join(self.reason_codes)}]")


class ParkedPackage(BaseModel):
    """One parked work package for the consolidated blocker report."""

    model_config = ConfigDict(extra="forbid")

    work_package_id: str | None
    agent_id: str
    run_id: str | None = None
    failure_class: str
    attempted_base_model: str | None = None
    attempted_endpoint: str | None = None
    # Effect uncertainty (handoff §1.7 tool_effects disposition).
    tool_effects: str = "unknown"
    same_agent_resumes_used: int = 0
    reason_codes: list[str] = Field(default_factory=list)
    # Why automation stopped.
    detail: str = ""
    # A proposed classification with supporting and missing evidence; proposals
    # never modify policy or grant recovery authority.
    proposed_classification: str = ""
    evidence_codes: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    artifact_refs: list[str] = Field(default_factory=list)
    endpoint_correlated: bool = False

    @classmethod
    def from_envelope(
        cls,
        envelope: FailureEnvelope,
        *,
        reason_codes: Sequence[str],
        detail: str,
        proposed_classification: str = "",
        missing_evidence: Sequence[str] = (),
    ) -> ParkedPackage:
        return cls(
            work_package_id=envelope.work_package_id,
            agent_id=envelope.agent_id,
            run_id=envelope.run_id,
            failure_class=envelope.failure.failure_class.value,
            attempted_base_model=envelope.deployment.base_model,
            attempted_endpoint=envelope.deployment.endpoint,
            tool_effects=envelope.progress.tool_effects.value,
            same_agent_resumes_used=envelope.recovery.same_agent_resumes_used,
            reason_codes=list(reason_codes),
            detail=detail,
            proposed_classification=proposed_classification,
            evidence_codes=list(envelope.failure.evidence_codes),
            missing_evidence=list(missing_evidence),
            artifact_refs=list(envelope.progress.artifact_refs),
            endpoint_correlated="endpoint_correlated"
            in envelope.failure.evidence_codes,
        )


def is_infrastructure_failure(envelope: FailureEnvelope) -> bool:
    """Whether the envelope's failure class is infrastructure/harness.

    Infrastructure/harness failures are never a user question under the
    unattended-resilience rule.
    """
    return envelope.failure.failure_class in INFRASTRUCTURE_FAILURE_CLASSES


def mechanical_base_models_from_routes(
    routes: Mapping[str, PurposeRoute], *, session_model: str
) -> frozenset[str]:
    """The mechanical-tier base models the configured purpose routes bind.

    The successor of the retired slot-purpose inference (package 16b): a route
    target is mechanical-tier when its explicit ``tier`` is ``mechanical``,
    and its bound model is the target's model — ``$session`` resolves to the
    session model. Unbound targets contribute nothing.
    """
    models: set[str] = set()
    for route in routes.values():
        for target in route.targets:
            if target.tier != PackageTier.MECHANICAL:
                continue
            model = session_model if target.model == SESSION_MODEL else target.model
            if model:
                models.add(model)
    return frozenset(models)


def route_recovery_facts(
    routes: Mapping[str, PurposeRoute],
    *,
    session_model: str,
    canonical_models: Mapping[str, str],
) -> tuple[frozenset[str], dict[str, ReplacementCapability]]:
    """The canonical mechanical-tier set and capability markings of the routes.

    The recovery boundary labels candidates by canonical base model, but a
    route target names its model by preset reference (``@role``), canonical
    base, or the ``$session`` binding — so every reference is canonicalized
    through the catalog mapping: a preset-bound mechanical target labels its
    base model mechanical, and a target's explicit ``replacement_capability``
    marking reaches the candidates keyed by the same canonical base (one
    canonical capability per model, never two target-derived identities).
    Targets without an explicit marking contribute no marking: the
    explicit-marking regime is opt-in, and with no explicit marking anywhere
    the legacy tier-label derivation applies unchanged.
    """
    mechanical: set[str] = set()
    capabilities: dict[str, ReplacementCapability] = {}
    for route in routes.values():
        for target in route.targets:
            model = (
                session_model
                if target.model == SESSION_MODEL
                else canonical_models.get(target.model, target.model)
            )
            if not model:
                continue
            if target.tier == PackageTier.MECHANICAL:
                mechanical.add(model)
            if target.replacement_capability is not ReplacementCapability.NONE:
                capabilities[model] = target.replacement_capability
    return frozenset(mechanical), capabilities


def candidates_from_catalog(
    catalog: ModelCatalog,
    *,
    mechanical_base_models: frozenset[str] = frozenset(),
    capabilities: Mapping[str, ReplacementCapability] | None = None,
) -> list[ReplacementCandidate]:
    """The catalog-derived candidate list (deterministic catalog order).

    Only enabled models with enabled deployments on enabled providers are
    included; each candidate carries its canonical redacted endpoint identity
    and output price (``None`` = pricing unavailable).

    ``capabilities`` is the model-level capability marking (route/model
    capability explicitly marked execution-capable): it is keyed by canonical
    base model, so a model bound on both mechanical and substantive route
    targets carries ONE canonical capability — route duplication never upgrades
    a model. When the mapping is supplied, an unmarked model is marked NONE
    (eligibility requires an explicit marking); when it is omitted, the legacy
    tier-label derivation applies unchanged.
    """
    candidates: list[ReplacementCandidate] = []
    for base_model, definition in catalog.models.items():
        if definition.disabled:
            continue
        tier = (
            PackageTier.MECHANICAL
            if base_model in mechanical_base_models
            else PackageTier.SUBSTANTIVE
        )
        capability = (
            capabilities.get(base_model, ReplacementCapability.NONE)
            if capabilities is not None
            else None
        )
        for deployment in definition.deployments:
            if deployment.disabled:
                continue
            provider = catalog.providers.get(deployment.provider)
            if provider is None or provider.disabled:
                continue
            candidates.append(
                ReplacementCandidate(
                    base_model=base_model,
                    provider=deployment.provider,
                    wire_name=deployment.name,
                    endpoint=canonical_endpoint_identity(provider.api_base),
                    tier=tier,
                    capability=capability,
                    output_price_per_million=deployment.prices.output,
                )
            )
    return candidates


class _DecisionContext(TypedDict):
    failed_base_model: str | None
    failed_endpoint: str | None
    predecessor_agent_id: str | None
    work_package_id: str | None


def _context(envelope: FailureEnvelope) -> _DecisionContext:
    return {
        "failed_base_model": envelope.deployment.base_model,
        "failed_endpoint": envelope.deployment.endpoint,
        "predecessor_agent_id": envelope.agent_id,
        "work_package_id": envelope.work_package_id,
    }


def _reject(
    envelope: FailureEnvelope, reason_codes: Sequence[str], detail: str
) -> ReplacementDecision:
    codes = list(reason_codes)
    return ReplacementDecision(
        disposition="rejected",
        reason_codes=codes,
        notice=(
            f"Replacement rejected: package {envelope.work_package_id or 'unnamed'} "
            f"(agent {envelope.agent_id}) — {detail}"
        ),
        **_context(envelope),
    )


def _format_price(value: float | None) -> str:
    return "pricing unavailable" if value is None else f"${value:.2f}/M output"


def _cost_implication(
    claimed: ReplacementCandidate, failed: ReplacementCandidate | None
) -> str:
    """The honest cost implication of a substitution (never guessed)."""
    claimed_price = claimed.output_price_per_million
    failed_price = failed.output_price_per_million if failed is not None else None
    parts = [f"replacement {claimed.base_model} {_format_price(claimed_price)}"]
    if failed is None:
        parts.append("; failed model not in the catalog, its pricing unavailable")
    else:
        parts.append(f"; failed {failed.base_model} {_format_price(failed_price)}")
    if claimed_price is not None and failed_price is not None:
        delta = claimed_price - failed_price
        if delta > 0:
            parts.append(f"; substitution costs ${delta:.2f}/M more per output token")
        elif delta < 0:
            parts.append(f"; substitution costs ${-delta:.2f}/M less per output token")
        else:
            parts.append("; substitution costs the same per output token")
    else:
        parts.append("; the spend delta is unknown — stated honestly, not guessed")
    return "".join(parts)


def _milestone_notice(
    *,
    envelope: FailureEnvelope,
    package_tier: PackageTier,
    claimed: ReplacementCandidate,
    cost_implication: str,
) -> str:
    failure_class = envelope.failure.failure_class.value
    failed_model = envelope.deployment.base_model or "unknown model"
    failed_endpoint = envelope.deployment.endpoint or "unknown endpoint"
    wire = claimed.wire_name or claimed.base_model
    return (
        f"Replacement milestone: package {envelope.work_package_id or 'unnamed'} — "
        f"agent {envelope.agent_id} run {envelope.run_id} failed ({failure_class}) "
        f"on {failed_model}@{failed_endpoint}; replacement {claimed.base_model} "
        f"({claimed.provider}/{wire}) selected on endpoint {claimed.endpoint} "
        f"(endpoint-independent of the failed call); package tier "
        f"{package_tier.value}; {cost_implication}; the new agent carries the "
        "predecessor/package lineage and budgets and recurrence counters span "
        "the replacement."
    )


def _endpoint_identity(endpoint: str | None) -> str | None:
    """Idempotent endpoint identity for independence comparison.

    ``canonical_endpoint_identity`` (recovery_models) maps a raw URL to its
    redacted host identity; it is not idempotent on a bare host (no netloc),
    so an already-canonical identity falls through lowercased. ``None`` means
    the identity is unavailable or unparsable — never a guess.
    """
    if endpoint is None:
        return None
    canonical = canonical_endpoint_identity(endpoint)
    if canonical is not None:
        return canonical
    return endpoint.strip().lower() or None


def validate_replacement_selection(
    *,
    envelope: FailureEnvelope,
    package_tier: PackageTier,
    claimed: ReplacementCandidate,
    candidates: Sequence[ReplacementCandidate],
    config: RecoveryConfig,
) -> ReplacementDecision:
    """The Q8 decision procedure: validate one claimed substitution.

    The invariants are CODE, never config escape hatches: endpoint
    independence against the ATTEMPTED failed endpoint (unknown identity
    cannot prove independence), tier capability, the SEPARATE candidate
    capability axis (an intrinsic model-level marking, never a tier guess),
    and the configured allowlist/price-cap constraints. Protected failure
    classes (preflight, cancellation, authorization, permanent config,
    budget stop, task) and unknown/internal causes never substitute — a
    HEAD/branch/workspace preflight stop is preserved, never bypassed.
    """
    failure_class = envelope.failure.failure_class
    # Endpoint independence is enforced against the ATTEMPTED failed
    # endpoint on the canonical redacted identity: a raw or variant URL
    # (userinfo, case, path) of the same host is the SAME endpoint — an
    # alias, never an independent fallback. An unknown identity on either
    # side cannot prove independence.
    failed_identity = _endpoint_identity(envelope.deployment.endpoint)
    claimed_identity = _endpoint_identity(claimed.endpoint)
    guards: list[tuple[bool, list[str], str]] = [
        (
            failure_class is FailureClass.PREFLIGHT,
            ["preflight_stop_preserved"],
            "a HEAD/branch/workspace preflight stop is preserved; verify the "
            "assignment facts — replacement never bypasses it",
        ),
        (
            failure_class in _PROTECTED_NO_REPLACEMENT_CLASSES,
            ["protected_class_no_replacement"],
            f"a {failure_class.value} failure is a protected stop, not an "
            "infrastructure failure; no substitution is permitted",
        ),
        (
            failure_class is FailureClass.UNKNOWN,
            ["unknown_cause_report_and_park"],
            "an unknown cause reports and parks; it is never confidently "
            "relabeled as a provider failure to justify a substitution",
        ),
        (
            failure_class is FailureClass.INTERNAL,
            ["harness_internal_report_and_park"],
            "a harness-internal failure reports and parks; it is not a provider "
            "failure and no substitution is permitted",
        ),
        (
            failed_identity is None,
            ["failed_endpoint_unknown_cannot_prove_independence"],
            "the failed call's endpoint identity is unknown, so no replacement "
            "can prove endpoint independence",
        ),
        (
            claimed_identity is None,
            ["candidate_endpoint_unknown_cannot_prove_independence"],
            f"replacement {claimed.base_model} has no known endpoint identity, "
            "so it cannot prove endpoint independence",
        ),
        (
            claimed_identity == failed_identity,
            ["same_endpoint_substitution_rejected"],
            f"replacement {claimed.base_model} shares the failed call's "
            f"endpoint {failed_identity}; a same-endpoint substitution — an "
            "alias or a different model on the same endpoint — is not a real "
            "fallback under endpoint throttling",
        ),
        (
            package_tier is PackageTier.SUBSTANTIVE
            and claimed.tier is PackageTier.MECHANICAL,
            ["mechanical_tier_rejected_for_substantive_package"],
            f"replacement {claimed.base_model} is a mechanical-tier model and "
            "the package is substantive; luna-class models are mechanical-only",
        ),
        (
            package_tier is PackageTier.SUBSTANTIVE
            and claimed.effective_capability is not ReplacementCapability.SUBSTANTIVE,
            ["candidate_capability_below_package_tier"],
            f"replacement {claimed.base_model} is marked "
            f"{claimed.effective_capability.value}-capable at the model level "
            "and the package is substantive; replacement eligibility is an "
            "intrinsic model capability, evaluated separately from the package "
            "tier",
        ),
        (
            package_tier is PackageTier.MECHANICAL
            and claimed.effective_capability is ReplacementCapability.NONE,
            ["candidate_capability_not_execution_marked"],
            f"replacement {claimed.base_model} carries no explicit "
            "execution-capability marking; a candidate is eligible only if its "
            "model-level capability is explicitly marked execution-capable for "
            "the requested work",
        ),
        (
            bool(config.fallback_allowed_base_models)
            and claimed.base_model not in config.fallback_allowed_base_models,
            ["base_model_not_in_fallback_allowlist"],
            f"replacement {claimed.base_model} is not in the configured "
            "fallback_allowed_base_models allowlist",
        ),
    ]
    for fired, reason_codes, detail in guards:
        if fired:
            return _reject(envelope, reason_codes, detail)
    cap = config.fallback_max_output_price_per_million
    if cap is not None:
        price = claimed.output_price_per_million
        if price is None:
            return _reject(
                envelope,
                ["fallback_price_unavailable_cap_unverifiable"],
                f"replacement {claimed.base_model} has no known output price, so "
                "the configured price cap cannot be verified",
            )
        if price > cap:
            return _reject(
                envelope,
                ["fallback_price_cap_exceeded"],
                f"replacement {claimed.base_model} output price ${price:.2f}/M "
                f"exceeds the configured cap ${cap:.2f}/M",
            )
    failed_candidate = next(
        (
            candidate
            for candidate in candidates
            if candidate.base_model == envelope.deployment.base_model
        ),
        None,
    )
    cost_implication = _cost_implication(claimed, failed_candidate)
    return ReplacementDecision(
        disposition="selected",
        reason_codes=["endpoint_independent", "tier_capable", "constraints_satisfied"],
        candidate=claimed,
        cost_implication=cost_implication,
        notice=_milestone_notice(
            envelope=envelope,
            package_tier=package_tier,
            claimed=claimed,
            cost_implication=cost_implication,
        ),
        **_context(envelope),
    )


def eligible_replacement_candidates(
    *,
    envelope: FailureEnvelope,
    package_tier: PackageTier,
    candidates: Sequence[ReplacementCandidate],
    config: RecoveryConfig,
) -> list[ReplacementCandidate]:
    """The validated eligible set the orchestrator chooses from at failure time.

    Every candidate is run through the full decision procedure; only selected
    substitutions survive. An empty result means the package parks
    (park-and-continue), never that the policy improvises.
    """
    eligible: list[ReplacementCandidate] = []
    for candidate in candidates:
        decision = validate_replacement_selection(
            envelope=envelope,
            package_tier=package_tier,
            claimed=candidate,
            candidates=candidates,
            config=config,
        )
        if decision.disposition == "selected":
            eligible.append(candidate)
    return eligible


def parked_decision(
    *, envelope: FailureEnvelope, reason_codes: Sequence[str], detail: str
) -> ReplacementDecision:
    """The park-and-continue disposition for an exhausted package."""
    return ReplacementDecision(
        disposition="parked",
        reason_codes=list(reason_codes),
        notice=(
            f"Park-and-continue: package {envelope.work_package_id or 'unnamed'} "
            f"(agent {envelope.agent_id}) is parked — {detail}. Continue "
            "independent work; no eligible replacement route remains and no "
            "user question is asked."
        ),
        **_context(envelope),
    )


def build_consolidated_blocker_report(
    *,
    parked: Sequence[ParkedPackage],
    deferred_questions: DeferredQuestionStore | None = None,
    session_id: str | None = None,
) -> str:
    """The consolidated resumable blocker report (unattended-resilience rule).

    Emitted only when eligible routes and budgets are exhausted: it carries the
    resumable state (parked packages with first-seen/unclassified failures,
    observed evidence, attempted endpoint/deployment or null, effect
    uncertainty, recovery actions already attempted, why automation stopped,
    and a proposed classification with supporting and missing evidence) plus
    U's deferred-question records, distinguishing a deferred preference from an
    actual blocker from a still-open authorization decision (which never
    expires and never appears in the deferred store). The user resumes with
    one instruction. Proposals never modify policy or grant recovery
    authority.
    """
    lines = [
        f"Consolidated blocker report ({UNATTENDED_RESILIENCE_MARKER}): eligible "
        "routes and budgets are exhausted. No open interactive question is "
        "pending; this report carries the resumable state."
    ]
    if session_id is not None:
        lines.append(f"Session: {session_id}")
    lines.append("")
    lines.append("Parked packages:")
    if not parked:
        lines.append("- none")
    for package in parked:
        attempted = package.attempted_base_model or "unknown model"
        endpoint = package.attempted_endpoint or "unknown endpoint"
        lines.append(
            f"- package {package.work_package_id or 'unnamed'} (agent "
            f"{package.agent_id}, run {package.run_id or 'n/a'}): failure class "
            f"{package.failure_class}; attempted {attempted}@{endpoint}; tool "
            f"effects {package.tool_effects} (uncertainty stated, never assumed "
            f"settled); same-agent resumes used {package.same_agent_resumes_used}; "
            f"stopped because: {package.detail or ', '.join(package.reason_codes)}"
        )
        if package.proposed_classification:
            lines.append(
                f"  proposed classification: {package.proposed_classification}"
            )
        if package.evidence_codes:
            lines.append(f"  supporting evidence: {', '.join(package.evidence_codes)}")
        if package.missing_evidence:
            lines.append(f"  missing evidence: {', '.join(package.missing_evidence)}")
        if package.artifact_refs:
            lines.append(f"  artifact refs: {', '.join(package.artifact_refs)}")
    correlated = [package for package in parked if package.endpoint_correlated]
    if correlated:
        endpoints = sorted({
            package.attempted_endpoint
            for package in correlated
            if package.attempted_endpoint
        })
        lines.append("")
        lines.append(
            "Correlated parent/child failures are visible (endpoint "
            f"{', '.join(endpoints)} throttled across models): the report shows "
            "the correlation; the root orchestrator is NEVER autonomously "
            "replaced by a cross-model substitution in this release."
        )
    lines.append("")
    lines.append("Deferred questions (U's records):")
    if deferred_questions is None:
        lines.append("- no deferred-question store available")
    else:
        lines.append(deferred_questions.render_summary())
    lines.append("")
    lines.append(
        "Resume with one instruction: re-dispatch a parked package (task with "
        "the same work_package_id and predecessor_agent_id lineage) once its "
        'endpoint recovers, or say "resume <package>" to continue it; '
        "still-open authorization decisions are answered as ordinary questions "
        "when you return."
    )
    return "\n".join(lines)
