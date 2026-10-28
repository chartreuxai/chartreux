"""Public typed failure envelope and validated recovery policy models (WP1).

This module is the PUBLIC failure envelope of the v0.5.6 recovery plan
(handoff §1.7, schema v1). WP0 landed the PRIVATE observation carrier
(``chartreux/core/recovery_observation.py``); this module builds the bounded,
redacted PUBLIC projection on top of it — never raw exceptions. Every field is
bounded; ``null`` means *unavailable*, never zero. Progress hints are
observable artifacts, not complete filesystem accounting, and turn counts or
checkpoint presence never establish replay safety.

The envelope is *representational* in v0.5.6: it classifies and carries typed
evidence so the WP4/R2 controllers can make bounded decisions later. It grants
no recovery authority by itself — automatic recovery stays disabled until WP6.

Also home to the measured per-logical-call evidence carrier
(:class:`CallEvidence`) the LLM gateway attaches to raised errors, and the
canonical redacted endpoint-identity convention used for deployment
recording and endpoint correlation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import enum
import time
from typing import Any, Literal, cast
from urllib.parse import urlsplit
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from chartreux.core.completion_contract import CompletionRejectionReason
from chartreux.core.llm.failures import FailureCategory

__all__ = [
    "CLASSIFICATION_VERSION",
    "CallEvidence",
    "ExhaustionReason",
    "FailureClass",
    "FailureConfidence",
    "FailureEnvelope",
    "FailurePhase",
    "PackageTier",
    "ProgressState",
    "RecoveryEligibility",
    "RecoveryReference",
    "RootTurnFailure",
    "ToolEffectsDisposition",
    "attach_attempted_deployment",
    "attach_call_evidence",
    "backend_transport_attempts",
    "build_completion_rejection_envelope",
    "build_failure_envelope",
    "build_root_turn_failure",
    "canonical_endpoint_identity",
]

#: Classification provenance/version stamp carried by every envelope.
CLASSIFICATION_VERSION = "v0.5.6-wp1"

_MAX_MESSAGE_CHARS = 500
_MAX_EVIDENCE_CODES = 32
#: Arity of an attempted-deployment identity tuple (base_model, provider, wire_name).
_ATTEMPTED_DEPLOYMENT_ARITY = 3
#: Public-carriage bound for the provider-controlled request id (review R4).
_MAX_REQUEST_ID_CHARS = 128
#: Public-carriage bound for the provider-controlled fingerprint error code.
_MAX_FINGERPRINT_CODE_CHARS = 64
#: Identifier charset for provider-controlled fingerprint fields (review R4).
_FINGERPRINT_CODE_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.:-"
)


class FailureClass(enum.StrEnum):
    """Subagent-level failure class (handoff §1.6 operational signatures).

    These are *operational signatures, not proven causes*: the class records
    what was observed, never a guessed root cause.
    """

    PROVIDER_STALL = "provider_stall"
    PROVIDER_OVERLOADED = "provider_overloaded"
    PROVIDER_RATE_LIMIT = "provider_rate_limit"
    LAUNCH_FAILURE = "launch_failure"
    PREFLIGHT = "preflight"
    TASK = "task"
    CANCELLATION = "cancellation"
    AUTHORIZATION = "authorization"
    PERMANENT_CONFIG = "permanent_config"
    BUDGET_STOP = "budget_stop"
    # P11: a typed completion-audit rejection — a worker-side disposition
    # (P1's CompletionRejectionReason vocabulary), never derived from an LLM
    # observation. Replacement-authorized by the failure-time policy; never
    # auto-resumed.
    COMPLETION_REJECTION = "completion_rejection"
    INCOMPLETE_STREAM = "incomplete_stream"
    EMPTY_RESPONSE = "empty_response"
    UNKNOWN = "unknown"
    INTERNAL = "internal"


class FailurePhase(enum.StrEnum):
    """Execution phase where the failure was observed (handoff §1.6).

    ``monitor`` and ``unknown`` extend the design's five execution phases to
    represent harness-side observation failures and missing evidence
    honestly instead of relabeling them.
    """

    ADMISSION = "admission"
    INIT = "init"
    LLM_CALL = "llm_call"
    TOOL = "tool"
    FINALIZE = "finalize"
    MONITOR = "monitor"
    UNKNOWN = "unknown"


class FailureConfidence(enum.StrEnum):
    """How the class was established."""

    MEASURED = "measured"
    """Directly observed (a measured phase, a provider's explicit code, a
    typed disposition measured at the response boundary)."""

    SIGNATURE = "signature"
    """An operational signature inferred from measured attempt history
    (handoff §1.6: repeated timeouts / no progress / exhausted budget)."""


class ProgressState(enum.StrEnum):
    """Progress state of the failed run (handoff §1.7)."""

    NONE = "none"
    CHECKPOINTED = "checkpointed"
    UNCHECKPOINTED = "uncheckpointed"
    UNCERTAIN = "uncertain"


class ToolEffectsDisposition(enum.StrEnum):
    """Tool-effect reconciliation state (F4 matrix; handoff §1.7)."""

    SETTLED = "settled"
    UNRESOLVED = "unresolved"
    UNKNOWN = "unknown"


class RecoveryEligibility(enum.StrEnum):
    """Recovery disposition carried by the envelope (handoff §1.7/§1.8).

    WP1 represents the disposition; it never acts on it.
    """

    ELIGIBLE = "eligible"
    RECONCILE_REQUIRED = "reconcile_required"
    PROHIBITED = "prohibited"


class PackageTier(enum.StrEnum):
    """The work-package tier for the failure-time replacement policy (WP5/Q8(b)).

    A package is ``mechanical`` (bounded searches, verification, mechanical
    single-file edits — the mechanical dispatch tier, luna-class) or
    ``substantive`` (all implementation work). The tier is orchestrator-known
    package metadata, declared with the replacement claim: a mechanical-tier
    model is never an eligible replacement for a substantive package.
    """

    MECHANICAL = "mechanical"
    SUBSTANTIVE = "substantive"


class ExhaustionReason(enum.StrEnum):
    """Why the exhausted logical call gave up (handoff §1.6)."""

    DEADLINE = "deadline"
    ATTEMPTS = "attempts"
    NONRETRYABLE = "nonretryable"
    UNKNOWN = "unknown"


class MeasuredTimeoutReason(enum.StrEnum):
    """Specific MEASURED timeout reasons (C5a) — a reason vocabulary, never a
    failure class.

    Every member classifies INSIDE an existing failure class (the timeout
    operational signature): the value rides the envelope's failure evidence
    codes, so consumers recognize the specific harness-measured timeout
    without a new ``FailureClass`` member (P11's wire mirror stays
    untouched). The producer contract (C5a -> C5b): a harness-raised
    deadline failure carries the reason value as its ``error_code``, which
    the WP0 observation carrier already extracts from the exception chain.
    """

    RESUME_PROGRESS_DEADLINE = "resume_progress_deadline"


#: Default ports dropped from the canonical endpoint identity (review R9):
#: an explicit default port never splits identity.
_DEFAULT_PORTS = {"https": 443, "http": 80}


def canonical_endpoint_identity(endpoint: str | None) -> str | None:
    """The one redacted canonical endpoint identity convention (WP1).

    Endpoint identity is the URL's network location (``api.mistral.ai``),
    lowercased and stripped of userinfo/credentials — never the raw URL, so
    the identity is safe to carry in envelopes, cooldown keys, and
    exclusions. The literal port is normalized (review R9): an explicit
    DEFAULT port (443 for https, 80 for http) is dropped, so
    ``https://example.test`` and ``https://example.test:443`` share ONE
    identity — same-endpoint substitution cannot pass independence on a
    port spelling, and correlation state cannot split on it. A NON-default
    port is part of the identity. Returns ``None`` when unavailable or
    unparsable (null = unavailable, never a guess).
    """
    if not endpoint or not isinstance(endpoint, str):
        return None
    try:
        parts = urlsplit(endpoint.strip())
    except ValueError:
        return None
    netloc = parts.netloc
    if not netloc:
        return None
    # Strip userinfo (credentials); urlsplit's hostname is already
    # lowercased and IPv6-bracket-stripped. An invalid port makes the URL
    # unparsable as an endpoint identity: unavailable, never a guess.
    try:
        port = parts.port
    except ValueError:
        return None
    host = parts.hostname
    if not host:
        return None
    host = host.lower()
    if ":" in host:
        # IPv6 hosts are re-bracketed so the identity stays a valid netloc.
        host = f"[{host}]"
    default_port = port is not None and port == _DEFAULT_PORTS.get(parts.scheme)
    if port is not None and not default_port:
        host = f"{host}:{port}"
    return host or None


def backend_transport_attempts(backend: object) -> int | None:
    """Read a backend's transport-start counter, or ``None`` (unavailable).

    Backends that record transport starts expose ``transport_attempts``
    (the generic and Mistral wrappers count every attempt through the retry
    wrappers' ``on_attempt`` channel). Backends without the counter report
    ``None`` — unavailable is never zero.
    """
    value = getattr(backend, "transport_attempts", None)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


@dataclass(slots=True)
class CallEvidence:
    """Measured evidence for one logical completion call (WP1).

    Minted once per logical call by the gateway facade — shared across the
    bounded in-call empty replay, so the exhausted-logical-call event stays
    ONE event, not one per retry — and attached to the raised error so the
    WP0 observation captures it before the message-only public projection.
    Null means unavailable, never zero.
    """

    logical_call_id: str
    started_at: float
    budget_seconds: float | None = None
    response_started: bool | None = None
    semantic_output_published: bool | None = None
    transport_attempts: int | None = None
    elapsed_seconds: float | None = None
    budget_exhausted: bool | None = None
    _transport_before: int | None = field(default=None, repr=False)
    _backend: object | None = field(default=None, repr=False)

    @classmethod
    def start(cls, backend: object, budget_seconds: float | None) -> CallEvidence:
        """Mint evidence at the start of one logical call."""
        return cls(
            logical_call_id=uuid4().hex,
            started_at=time.perf_counter(),
            budget_seconds=budget_seconds,
            _transport_before=backend_transport_attempts(backend),
            _backend=backend,
        )

    @classmethod
    def admission_phase(cls) -> CallEvidence:
        """Evidence for a call that failed before any transport start.

        No backend was created or borrowed for the logical call, so zero
        transport starts is a MEASURED fact (launch_failure requires
        admission/init evidence), not a guess.
        """
        return cls(
            logical_call_id=uuid4().hex,
            started_at=time.perf_counter(),
            transport_attempts=0,
        )

    def finish(self, *, budget_exhausted: bool | None) -> None:
        """Settle the measured facts at call exit (success or failure)."""
        self.elapsed_seconds = max(0.0, time.perf_counter() - self.started_at)
        if self._backend is not None and self._transport_before is not None:
            after = backend_transport_attempts(self._backend)
            if after is not None:
                self.transport_attempts = max(0, after - self._transport_before)
        self.budget_exhausted = budget_exhausted


def attach_call_evidence(error: BaseException, evidence: CallEvidence) -> None:
    """Attach measured call evidence to a raised error (total, idempotent).

    Never masks the primary failure: attachment failures are swallowed, and
    an error that already carries evidence keeps the first measurement.
    """
    if getattr(error, "call_evidence", None) is not None:
        return
    try:
        error.call_evidence = evidence  # type: ignore[attr-defined]
    except Exception:
        pass


def attach_attempted_deployment(
    error: BaseException, identity: tuple[str, str, str]
) -> None:
    """Attach the attempted deployment identity to a raised error.

    Total, idempotent, first-wins — the same contract as
    :func:`attach_call_evidence`. The completion coordinator attaches the
    failover identity BEFORE rolling the committed model back, so a
    failed-attempt envelope can report the deployment that actually served
    the failed attempt instead of the restored one (review R11).
    """
    if getattr(error, "attempted_deployment", None) is not None:
        return
    try:
        error.attempted_deployment = identity  # type: ignore[attr-defined]
    except Exception:
        pass


class EnvelopeFailure(BaseModel):
    """The failure block (handoff §1.7): class, cause, phase, bounded message."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    failure_class: FailureClass = Field(
        # The wire spelling is "class" (§1.7); the field keeps its Python
        # name for construction, with the alias on both wire directions.
        serialization_alias="class",
        validation_alias="class",
        description="Subagent-level operational signature (not a proven cause).",
    )
    cause_category: FailureCategory | None = Field(
        default=None,
        description="The low-level classifier's category for the observed cause; null = unavailable.",
    )
    phase: FailurePhase
    exception_type: str | None = None
    message: str | None = Field(
        default=None,
        description="Bounded (<=500 chars) and redacted failure message; never raw exception internals.",
    )
    confidence: FailureConfidence
    evidence_codes: list[str] = Field(
        default_factory=list,
        description="Open-ended bounded evidence codes (no log parsing).",
    )
    fingerprint: str | None = Field(
        default=None,
        description=(
            "Sanitized fingerprint (origin, exception type, status, structured "
            "code, adapter, layer) — never raw message text."
        ),
    )
    classification_version: str = Field(
        default=CLASSIFICATION_VERSION,
        description="Version of the classification that produced this envelope.",
    )
    classification_provenance: str = Field(
        default="",
        description="Bounded description of where/how the classification was measured.",
    )
    diagnostics_state: Literal["captured", "unavailable"] = Field(
        default="unavailable",
        description=(
            "Private full-diagnostics state: 'captured' when the envelope was "
            "built from a preserved typed observation, else 'unavailable'."
        ),
    )


class EnvelopeDeployment(BaseModel):
    """The ATTEMPTED deployment (handoff §1.7) — not just the committed model."""

    model_config = ConfigDict(extra="forbid")

    configured_model: str | None = Field(
        default=None, description="Configured model selection; null = unavailable."
    )
    base_model: str | None = None
    provider: str | None = None
    wire_name: str | None = None
    endpoint: str | None = Field(
        default=None,
        description=(
            "Canonical redacted identity of the endpoint actually serving "
            "the call; null = no call was served (or unavailable)."
        ),
    )


class EnvelopeRequest(BaseModel):
    """The request block (handoff §1.7): one logical call, measured."""

    model_config = ConfigDict(extra="forbid")

    logical_call_id: str | None = None
    transport_attempts: int | None = Field(
        default=None,
        description="Measured transport starts for the logical call; null = unavailable, 0 = measured zero.",
    )
    elapsed_seconds: float | None = None
    retry_budget_seconds: float | None = None
    exhaustion: ExhaustionReason | None = None
    request_id: str | None = None
    request_state: str | None = Field(
        default=None,
        description="Open-ended observed request lifecycle state; null = unavailable.",
    )
    retry_after_seconds: float | None = None
    response_started: bool | None = None
    semantic_output_published: bool | None = None


class EnvelopeProgress(BaseModel):
    """The progress block (handoff §1.7) — observable artifacts, not accounting."""

    model_config = ConfigDict(extra="forbid")

    turns_used: int | None = None
    state: ProgressState = ProgressState.UNCERTAIN
    checkpoint_ref: str | None = None
    transcript_boundary: str | None = Field(
        default=None,
        description="The transcript evidence boundary (the failed turn); null = unavailable.",
    )
    touched_paths_hint: list[str] = Field(
        default_factory=list,
        description="Bounded observable touched-paths hint; hints are not complete filesystem accounting.",
    )
    artifact_refs: list[str] = Field(
        default_factory=list,
        description="Bounded observable artifact references; empty = no observable refs.",
    )
    tool_effects: ToolEffectsDisposition = ToolEffectsDisposition.UNKNOWN
    last_progress_at: float | None = None


class EnvelopeRecovery(BaseModel):
    """The recovery block (handoff §1.7): represented disposition, never acted on."""

    model_config = ConfigDict(extra="forbid")

    eligibility: RecoveryEligibility
    reason_codes: list[str] = Field(default_factory=list)
    same_agent_resumes_used: int = Field(
        default=0,
        description="Same-agent resumes already used in this chain (WP2 owns resume counting).",
    )
    expires_at: float | None = Field(
        default=None,
        description="Recovery-chain expiry (epoch seconds); null = no expiry set (WP3 owns suspension TTL).",
    )


class FailureEnvelope(BaseModel):
    """The public typed failure envelope (handoff §1.7, schema v1).

    One envelope per exhausted logical call — deduplicatable by ``event_id``
    — never one per transport retry. Bounded and redacted throughout; full
    diagnostics stay private (the WP0 observation carrier).
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    event_id: str = Field(
        description="Deduplicatable event identity: one per exhausted logical call of one run."
    )
    agent_id: str
    run_id: str
    # WP5 lineage pair: the orchestrator work-package identity and the agent
    # this run's agent replaced (a cross-model replacement gets a NEW agent_id
    # with predecessor/package lineage). WP5 populates the pair through the
    # contract-surface launch fields; null = unavailable.
    work_package_id: str | None = Field(
        default=None,
        description="Orchestrator work-package lineage; null = unavailable.",
    )
    predecessor_agent_id: str | None = Field(
        default=None,
        description="The failed agent this run's agent replaced; null = not a replacement.",
    )
    recovery_chain_id: str = Field(
        description="Recovery-chain seed for this run; successors extend the chain (WP2/WP5)."
    )
    observed_at: float = Field(
        description="Epoch seconds when the failure was observed."
    )
    failure: EnvelopeFailure
    deployment: EnvelopeDeployment
    request: EnvelopeRequest
    progress: EnvelopeProgress
    recovery: EnvelopeRecovery


class RecoveryReference(BaseModel):
    """Compact recovery-chain linkage carried by results and summaries.

    A bounded projection of the envelope's identity + recovery blocks, so
    wait/notification consumers can correlate a failed run without carrying
    the full envelope.
    """

    model_config = ConfigDict(extra="forbid")

    recovery_chain_id: str
    event_id: str
    work_package_id: str | None = None
    # WP5: the lineage pair rides the reference too, so a failed replacement
    # run's linkage stays correlatable without the full envelope.
    predecessor_agent_id: str | None = None
    eligibility: RecoveryEligibility
    same_agent_resumes_used: int = 0
    expires_at: float | None = None

    @classmethod
    def from_envelope(cls, envelope: FailureEnvelope) -> RecoveryReference:
        return cls(
            recovery_chain_id=envelope.recovery_chain_id,
            event_id=envelope.event_id,
            work_package_id=envelope.work_package_id,
            predecessor_agent_id=envelope.predecessor_agent_id,
            eligibility=envelope.recovery.eligibility,
            same_agent_resumes_used=envelope.recovery.same_agent_resumes_used,
            expires_at=envelope.recovery.expires_at,
        )


# ---------------------------------------------------------------------------
# Derivation (subagent-level classification per handoff §1.6)
# ---------------------------------------------------------------------------

_PROTECTED_CLASSES = frozenset({
    FailureClass.PREFLIGHT,
    FailureClass.CANCELLATION,
    FailureClass.AUTHORIZATION,
    FailureClass.PERMANENT_CONFIG,
    FailureClass.BUDGET_STOP,
    FailureClass.TASK,
})

_TRANSIENT_CLASSES = frozenset({
    FailureClass.PROVIDER_STALL,
    FailureClass.PROVIDER_OVERLOADED,
    FailureClass.PROVIDER_RATE_LIMIT,
    FailureClass.LAUNCH_FAILURE,
    FailureClass.INCOMPLETE_STREAM,
    FailureClass.EMPTY_RESPONSE,
})

_BUDGET_STOP_REASONS = frozenset({"budget_exceeded", "budget_unverifiable"})

# C7: cooldown and endpoint-cooldown are both time-based transient launch
# blocks; every other exclusion reason is a permanent admission restriction.
_LAUNCH_TRANSIENT_EXCLUSION_REASONS = frozenset({"cooldown", "endpoint-cooldown"})


def _derive_phase(
    observation_phase: str | None, transport_attempts: int | None
) -> FailurePhase:
    """Map the WP0 capture phase to the execution phase, honestly.

    A measured zero transport starts inside the turn-phase runner is an
    admission-phase failure (the call never reached a backend). Without the
    measurement the phase is not relabeled.
    """
    if observation_phase is None:
        return FailurePhase.UNKNOWN
    if observation_phase == "monitor":
        return FailurePhase.MONITOR
    if observation_phase == "turn_terminal":
        return FailurePhase.FINALIZE
    if observation_phase == "turn":
        if transport_attempts == 0:
            return FailurePhase.ADMISSION
        return FailurePhase.LLM_CALL
    return FailurePhase.UNKNOWN


def _derive_exhaustion(
    *,
    budget_exhausted: bool | None,
    cause_category: FailureCategory | None,
    transport_attempts: int | None,
) -> ExhaustionReason | None:
    """Derive the exhaustion reason from measured facts (handoff §1.6)."""
    if budget_exhausted is True:
        return ExhaustionReason.DEADLINE
    if transport_attempts is not None and transport_attempts > 1:
        return ExhaustionReason.ATTEMPTS
    if cause_category is not None and cause_category is not FailureCategory.UNKNOWN:
        from chartreux.core.llm.failures import FailureInfo

        if not FailureInfo(cause_category).retry_eligible:
            return ExhaustionReason.NONRETRYABLE
    return ExhaustionReason.UNKNOWN


def _derive_tool_effects(
    *, phase: FailurePhase, tool_calls_present: bool, tool_calls_pending: bool
) -> tuple[ToolEffectsDisposition, list[str]]:
    """Derive the tool-effects disposition from transcript evidence only.

    F4: zero tool results are NOT proof of no effects — hooks, jobs, and
    child launches are not transcript-visible, so a run with no tool calls
    reports ``unknown``. An admission-phase launch failure never began
    execution, which is positive structural evidence of no effects.
    """
    if phase is FailurePhase.ADMISSION:
        return ToolEffectsDisposition.SETTLED, ["admission_phase_no_execution"]
    if tool_calls_pending:
        return ToolEffectsDisposition.UNRESOLVED, ["tool_calls_pending_results"]
    if tool_calls_present:
        return ToolEffectsDisposition.SETTLED, ["tool_results_settled_in_transcript"]
    return ToolEffectsDisposition.UNKNOWN, ["tool_effects_not_transcript_visible"]


def _measured_timeout_reason(observation: Any | None) -> MeasuredTimeoutReason | None:
    """The specific measured timeout reason an observation carries, if any.

    The C5a producer contract: a harness-raised measured deadline failure
    carries the reason value as its ``error_code`` (the WP0 carrier extracts
    ``error_code`` from the exception chain). Recognition errs toward the
    prohibition direction: a colliding code can only suppress the automatic
    continuation, never enable one.
    """
    if observation is None:
        return None
    code = getattr(observation, "error_code", None)
    if not isinstance(code, str):
        return None
    try:
        return MeasuredTimeoutReason(code)
    except ValueError:
        return None


def measured_timeout_reason_of(
    envelope: FailureEnvelope,
) -> MeasuredTimeoutReason | None:
    """The envelope's specific measured timeout reason (C5a), if any.

    The reason rides the failure block's evidence codes (the reason value IS
    the evidence code); consumers — the automatic-recovery controller's
    prohibition gate, the failure-time replacement policy — recognize it
    through this one helper so the evidence-code contract stays in one place.
    """
    for reason in MeasuredTimeoutReason:
        if reason.value in envelope.failure.evidence_codes:
            return reason
    return None


def build_failure_envelope(  # noqa: PLR0913, PLR0912, PLR0914, PLR0915
    # One deliberate derivation entry point: the §1.7 envelope is built in a
    # single place so every field's provenance stays reviewable together.
    *,
    observation: Any | None,
    agent_id: str,
    run_id: str,
    turn_id: str | None = None,
    turns_used: int | None = None,
    attempted_deployment: tuple[str, str, str] | None = None,
    configured_model: str | None = None,
    stop_reason: str | None = None,
    exclusion_reasons: list[str] | None = None,
    tool_calls_present: bool = False,
    tool_calls_pending: bool = False,
    endpoint_correlated: bool = False,
    same_agent_resumes_used: int = 0,
    recovery_chain_id: str | None = None,
    work_package_id: str | None = None,
    predecessor_agent_id: str | None = None,
) -> FailureEnvelope:
    """Build the public bounded failure envelope for one failed run.

    The subagent-level classification combines the low-level cause, the
    execution phase, the measured attempt history, the exhaustion reason,
    and the progress state (handoff §1.6). Every OBSERVED terminal failure —
    including UNKNOWN and harness-local (monitor) failures — receives an
    envelope; unknown stays UNKNOWN and reports-and-parks rather than being
    relabeled.
    """
    call_evidence: CallEvidence | None = None
    if observation is not None:
        call_evidence = getattr(observation, "call_evidence", None)
        if not isinstance(call_evidence, CallEvidence):
            call_evidence = None

    transport_attempts: int | None = None
    if call_evidence is not None:
        transport_attempts = call_evidence.transport_attempts

    phase = _derive_phase(
        getattr(observation, "phase", None) if observation is not None else None,
        transport_attempts,
    )

    cause_category: FailureCategory | None = None
    classification: Any = getattr(observation, "classification", None)
    if classification is not None:
        cause_category = getattr(classification, "category", None)
        if not isinstance(cause_category, FailureCategory):
            cause_category = None

    measured_timeout = _measured_timeout_reason(observation)

    exception_type: str | None = (
        getattr(observation, "exception_type", None)
        if observation is not None
        else None
    )
    is_admission_error = exception_type == "AllDeploymentsUnavailableError"

    # --- class derivation (protected classes dominate) ---
    evidence_codes: list[str] = []
    if measured_timeout is not None:
        # C5a: a harness-measured progress deadline is a MEASURED timeout —
        # the specific reason rides the evidence codes, the class stays the
        # existing timeout operational signature (never a new FailureClass),
        # and the confidence is MEASURED (the harness measured the deadline
        # directly, not a signature inference from attempt history).
        cause_category = FailureCategory.TIMEOUT
        failure_class = FailureClass.PROVIDER_STALL
        confidence = FailureConfidence.MEASURED
        evidence_codes.append(measured_timeout.value)
    elif stop_reason in _BUDGET_STOP_REASONS:
        failure_class = FailureClass.BUDGET_STOP
        confidence = FailureConfidence.MEASURED
        evidence_codes.append("budget_stop")
    elif cause_category is FailureCategory.CANCELLATION:
        failure_class = FailureClass.CANCELLATION
        confidence = FailureConfidence.MEASURED
        evidence_codes.append("protected_cancellation")
    elif phase is FailurePhase.MONITOR:
        failure_class = FailureClass.INTERNAL
        confidence = FailureConfidence.MEASURED
        evidence_codes.append("harness_monitor_failure")
    elif cause_category is FailureCategory.AUTH:
        failure_class = FailureClass.AUTHORIZATION
        confidence = FailureConfidence.MEASURED
        evidence_codes.append("protected_authorization")
    elif cause_category is not None and cause_category in {
        FailureCategory.BILLING,
        FailureCategory.QUOTA,
        FailureCategory.INVALID_REQUEST,
        FailureCategory.CONTEXT_SIZE,
    }:
        failure_class = FailureClass.PERMANENT_CONFIG
        confidence = FailureConfidence.MEASURED
        evidence_codes.append(f"protected_{cause_category.value}")
    elif (
        phase is FailurePhase.ADMISSION
        or (is_admission_error and transport_attempts == 0)
    ) and (transport_attempts == 0 or is_admission_error):
        # launch_failure requires admission/init evidence: a measured zero
        # transport starts (or the admission-phase exhaustion error itself).
        failure_class = FailureClass.LAUNCH_FAILURE
        confidence = FailureConfidence.MEASURED
        evidence_codes.append("admission_phase")
        evidence_codes.append("zero_transport_attempts")
    elif cause_category is FailureCategory.RATE_LIMIT:
        failure_class = FailureClass.PROVIDER_RATE_LIMIT
        confidence = FailureConfidence.MEASURED
        evidence_codes.append("provider_rate_limit_code")
    elif cause_category is FailureCategory.OVERLOAD:
        failure_class = FailureClass.PROVIDER_OVERLOADED
        confidence = FailureConfidence.MEASURED
        evidence_codes.append("provider_overload_code")
    elif cause_category is not None and cause_category in {
        FailureCategory.TIMEOUT,
        FailureCategory.CONNECTION,
        FailureCategory.SERVER_ERROR,
    }:
        # Operational signature, not a proven cause (§1.6).
        failure_class = FailureClass.PROVIDER_STALL
        confidence = FailureConfidence.SIGNATURE
        evidence_codes.append(f"signature_{cause_category.value}")
    elif cause_category is FailureCategory.INCOMPLETE_STREAM:
        failure_class = FailureClass.INCOMPLETE_STREAM
        confidence = FailureConfidence.MEASURED
        evidence_codes.append("typed_transient_incomplete_stream")
    elif cause_category is FailureCategory.EMPTY_RESPONSE:
        failure_class = FailureClass.EMPTY_RESPONSE
        confidence = FailureConfidence.MEASURED
        evidence_codes.append("typed_transient_empty_response")
    else:
        failure_class = FailureClass.UNKNOWN
        confidence = FailureConfidence.MEASURED
        evidence_codes.append("unknown_cause_report_and_park")

    if endpoint_correlated:
        evidence_codes.append("endpoint_correlated")

    semantic_output_published: bool | None = None
    if call_evidence is not None:
        semantic_output_published = call_evidence.semantic_output_published
        if semantic_output_published:
            evidence_codes.append("semantic_output_published_no_replay")

    # --- exhaustion ---
    exhaustion: ExhaustionReason | None = None
    if measured_timeout is not None:
        # C5a: the measured deadline IS the exhaustion fact — derived, not
        # inferred from the attempt history or the retry budget.
        exhaustion = ExhaustionReason.DEADLINE
    elif call_evidence is not None or cause_category is not None:
        exhaustion = _derive_exhaustion(
            budget_exhausted=(
                call_evidence.budget_exhausted if call_evidence is not None else None
            ),
            cause_category=cause_category,
            transport_attempts=transport_attempts,
        )
    if exhaustion is not None:
        evidence_codes.append(f"exhaustion_{exhaustion.value}")

    # --- progress / tool effects ---
    tool_effects, tool_effect_codes = _derive_tool_effects(
        phase=phase,
        tool_calls_present=tool_calls_present,
        tool_calls_pending=tool_calls_pending,
    )
    evidence_codes.extend(tool_effect_codes)

    if turns_used is None:
        progress_state = ProgressState.UNCERTAIN
    elif turns_used == 0:
        progress_state = ProgressState.NONE
    else:
        progress_state = ProgressState.UNCHECKPOINTED

    # --- recovery eligibility (represented, never acted on) ---
    if failure_class in _PROTECTED_CLASSES:
        eligibility = RecoveryEligibility.PROHIBITED
        reason_codes = ["protected_class_no_automatic_recovery"]
    elif failure_class is FailureClass.INTERNAL:
        eligibility = RecoveryEligibility.PROHIBITED
        reason_codes = ["harness_internal_no_automatic_recovery"]
    elif failure_class is FailureClass.UNKNOWN:
        eligibility = RecoveryEligibility.PROHIBITED
        reason_codes = ["unknown_cause_report_and_park"]
    elif failure_class is FailureClass.COMPLETION_REJECTION:
        # P11: a typed audit rejection is a worker-side disposition the
        # failure-time policy authorizes a cross-agent replacement for (the
        # escalation joins the package's lineage); it never authorizes a
        # warm same-agent auto-resume (P12's suspension gates stay shut).
        eligibility = RecoveryEligibility.PROHIBITED
        reason_codes = ["completion_rejection_no_auto_recovery"]
    elif failure_class is FailureClass.LAUNCH_FAILURE:
        reasons = exclusion_reasons or []
        if reasons and all(
            reason in _LAUNCH_TRANSIENT_EXCLUSION_REASONS for reason in reasons
        ):
            eligibility = RecoveryEligibility.ELIGIBLE
            reason_codes = ["transient_launch_failure", "no_execution_progress"]
            evidence_codes.append(
                "endpoint_cooldown_exclusion"
                if "endpoint-cooldown" in reasons
                else "cooldown_exclusion"
            )
        elif not reasons:
            # No exclusion detail observed: never confidently relabel.
            eligibility = RecoveryEligibility.PROHIBITED
            reason_codes = ["launch_failure_exclusions_unavailable"]
        else:
            eligibility = RecoveryEligibility.PROHIBITED
            reason_codes = ["permanent_launch_exclusion"]
            evidence_codes.append("permanent_exclusion")
    elif failure_class in _TRANSIENT_CLASSES:
        # General error policy: recovery eligibility requires trusted origin
        # AND measured evidence. A transient-classed cause without measured
        # logical-call evidence in a model-call phase is NOT a confirmed
        # provider transient — a tool/hook/config/resource/projection/
        # finalization timeout is not a provider timeout — so it can never
        # authorize another model call (review R2; the root path's gate is
        # build_root_turn_failure's transient_cause_confirmed). The gate
        # covers the RECONCILE_REQUIRED barrier too: a barrier that a
        # reconcile submission can clear must only exist for a measured
        # transient, or the reconcile path would bypass this gate.
        if call_evidence is None or phase is not FailurePhase.LLM_CALL:
            eligibility = RecoveryEligibility.PROHIBITED
            reason_codes = ["transient_cause_unmeasured_no_recovery"]
            evidence_codes.append("transient_cause_unmeasured")
        elif tool_effects is ToolEffectsDisposition.UNRESOLVED:
            eligibility = RecoveryEligibility.RECONCILE_REQUIRED
            reason_codes = ["tool_effects_unresolved_reconcile_first"]
        elif tool_effects is ToolEffectsDisposition.UNKNOWN:
            eligibility = RecoveryEligibility.RECONCILE_REQUIRED
            reason_codes = ["tool_effects_unknown_reconcile_first"]
        else:
            eligibility = RecoveryEligibility.ELIGIBLE
            reason_codes = ["transient_cause", "tool_effects_settled"]
    else:
        eligibility = RecoveryEligibility.PROHIBITED
        reason_codes = ["unrecoverable_class"]

    # --- bounded/redacted message + sanitized fingerprint ---
    message: str | None = None
    fingerprint: str | None = None
    if observation is not None:
        raw_message = getattr(observation, "message", None)
        if isinstance(raw_message, str) and raw_message:
            message = _bound_and_redact(raw_message)
        status = getattr(observation, "status", None)
        error_code = getattr(observation, "error_code", None)
        provider = attempted_deployment[1] if attempted_deployment else None
        fingerprint = (
            f"origin={phase.value};type={exception_type or 'unknown'};"
            f"status={status if status is not None else 'none'};"
            f"code={_sanitize_fingerprint_code(error_code)};"
            f"adapter={provider or 'unknown'};"
            f"layer={getattr(observation, 'phase', 'unknown')}"
        )

    event_turn = f":{turn_id}" if turn_id else ""
    provenance = (
        "chartreux.core.llm.failures.classify_failure over the WP0 "
        f"observation (phase={getattr(observation, 'phase', None)}); "
        f"subagent-level derivation chartreux.core.recovery_models {CLASSIFICATION_VERSION}"
    )
    if measured_timeout is not None:
        provenance += f"; measured harness deadline {measured_timeout.value}"

    return FailureEnvelope(
        event_id=f"failure-event:{agent_id}:{run_id}{event_turn}",
        agent_id=agent_id,
        run_id=run_id,
        work_package_id=work_package_id,
        predecessor_agent_id=predecessor_agent_id,
        # Review R10: an explicit chain id keeps a resumed run's envelope on
        # the record's episode chain; the per-run derivation is only the
        # fallback for runs that never published one.
        recovery_chain_id=(
            recovery_chain_id
            if recovery_chain_id is not None
            else f"recovery-chain:{agent_id}:{run_id}"
        ),
        observed_at=(
            observation.observed_at
            if observation is not None and getattr(observation, "observed_at", None)
            else time.time()
        ),
        failure=EnvelopeFailure(
            failure_class=failure_class,
            cause_category=cause_category,
            phase=phase,
            exception_type=exception_type,
            message=message,
            confidence=confidence,
            evidence_codes=evidence_codes[:_MAX_EVIDENCE_CODES],
            fingerprint=fingerprint,
            classification_version=CLASSIFICATION_VERSION,
            classification_provenance=provenance,
            diagnostics_state=(
                "captured" if observation is not None else "unavailable"
            ),
        ),
        deployment=EnvelopeDeployment(
            configured_model=configured_model,
            base_model=attempted_deployment[0] if attempted_deployment else None,
            provider=attempted_deployment[1] if attempted_deployment else None,
            wire_name=attempted_deployment[2] if attempted_deployment else None,
            endpoint=canonical_endpoint_identity(
                getattr(observation, "endpoint", None)
                if observation is not None
                else None
            ),
        ),
        request=EnvelopeRequest(
            logical_call_id=(
                call_evidence.logical_call_id if call_evidence is not None else None
            ),
            transport_attempts=transport_attempts,
            elapsed_seconds=(
                call_evidence.elapsed_seconds if call_evidence is not None else None
            ),
            retry_budget_seconds=(
                call_evidence.budget_seconds if call_evidence is not None else None
            ),
            exhaustion=exhaustion,
            request_id=_bound_request_id(
                getattr(observation, "request_id", None)
                if observation is not None
                else None
            ),
            request_state=(
                "no_transport_start"
                if transport_attempts == 0
                else ("failed" if observation is not None else None)
            ),
            retry_after_seconds=(
                classification.retry_after if classification is not None else None
            ),
            response_started=(
                call_evidence.response_started if call_evidence is not None else None
            ),
            semantic_output_published=semantic_output_published,
        ),
        progress=EnvelopeProgress(
            turns_used=turns_used,
            state=progress_state,
            checkpoint_ref=None,
            transcript_boundary=turn_id,
            touched_paths_hint=[],
            artifact_refs=[],
            tool_effects=tool_effects,
            last_progress_at=(
                observation.observed_at
                if observation is not None and getattr(observation, "observed_at", None)
                else None
            ),
        ),
        recovery=EnvelopeRecovery(
            eligibility=eligibility,
            reason_codes=reason_codes,
            same_agent_resumes_used=same_agent_resumes_used,
            expires_at=None,
        ),
    )


def build_completion_rejection_envelope(
    *,
    agent_id: str,
    run_id: str,
    rejection: CompletionRejectionReason,
    turn_id: str | None = None,
    work_package_id: str | None = None,
    predecessor_agent_id: str | None = None,
    recovery_chain_id: str | None = None,
    message: str | None = None,
    clock: Any = time.time,
) -> FailureEnvelope:
    """Build the bounded public envelope for one typed completion rejection.

    The P11 failure class is never derived from an LLM-call observation —
    a completion rejection is the audit's verdict on a worker's claimed
    completion — so this bounded constructor classifies it directly from the
    P1 rejection vocabulary instead of running the §1.6 derivation. Bounded
    and redacted throughout: the free-text message is redacted then bounded,
    the typed rejection value rides the evidence codes as an identifier, the
    deployment/request blocks carry no transport attempt, and the recovery
    block is PROHIBITED with its own distinct reason code — a typed audit
    rejection authorizes a cross-agent replacement (recovery_policy), never
    a warm same-agent auto-resume (P12's suspension gates stay shut).
    """
    event_turn = f":{turn_id}" if turn_id else ""
    return FailureEnvelope(
        event_id=f"failure-event:{agent_id}:{run_id}{event_turn}",
        agent_id=agent_id,
        run_id=run_id,
        work_package_id=work_package_id,
        predecessor_agent_id=predecessor_agent_id,
        # The rejection is never a resume: the per-run derivation fallback
        # keeps the record on its own episode chain (review R10's contract).
        recovery_chain_id=(
            recovery_chain_id
            if recovery_chain_id is not None
            else f"recovery-chain:{agent_id}:{run_id}"
        ),
        observed_at=clock(),
        failure=EnvelopeFailure(
            failure_class=FailureClass.COMPLETION_REJECTION,
            cause_category=None,
            phase=FailurePhase.FINALIZE,
            exception_type=None,
            message=_bound_and_redact(message) if message else None,
            confidence=FailureConfidence.MEASURED,
            evidence_codes=[
                f"completion_rejection_{rejection.value}",
                "completion_audit_rejection",
            ][:_MAX_EVIDENCE_CODES],
            fingerprint=(
                f"origin={FailurePhase.FINALIZE.value};type=none;"
                f"status=none;code={_sanitize_fingerprint_code(rejection.value)};"
                f"adapter=unknown;layer={FailurePhase.FINALIZE.value}"
            ),
            classification_version=CLASSIFICATION_VERSION,
            classification_provenance=(
                "typed completion-audit rejection (P11) over the P1 "
                f"rejection vocabulary; recovery_models {CLASSIFICATION_VERSION}"
            ),
            diagnostics_state="captured",
        ),
        deployment=EnvelopeDeployment(),
        request=EnvelopeRequest(),
        progress=EnvelopeProgress(),
        recovery=EnvelopeRecovery(
            eligibility=RecoveryEligibility.PROHIBITED,
            reason_codes=["completion_rejection_no_auto_recovery"],
        ),
    )


def _bound_and_redact(message: str) -> str:
    """Bound and redact a failure message for public carriage."""
    try:
        from chartreux.core.tools.secret_redaction import redact

        # Redact BEFORE bounding (review R4): truncating first would leave
        # a secret straddling the cut as an unredactable fragment.
        return redact(message)[:_MAX_MESSAGE_CHARS]
    except Exception:
        # Fail closed: a redaction failure must never carry the original
        # text publicly — a fixed marker, never the bounded original.
        return "[REDACTED]"


def _sanitize_fingerprint_code(code: str | None) -> str:
    """Reduce a provider-controlled error code to a bounded identifier.

    The structured code is interpolated into the PUBLIC fingerprint
    (review R4): it is reduced to a bounded identifier charset so the
    fingerprint can never carry raw message text, field separators, or
    unbounded payloads.
    """
    if not code:
        return "none"
    sanitized = "".join(char for char in code if char in _FINGERPRINT_CODE_CHARS)
    sanitized = sanitized[:_MAX_FINGERPRINT_CODE_CHARS]
    return sanitized or "none"


def _bound_request_id(request_id: str | None) -> str | None:
    """Bound and redact a request id for public carriage (review R4).

    The request id is provider-controlled text copied onto the public
    envelope: it is redacted and bounded, and a redaction failure drops it
    (null = unavailable) rather than carrying possibly-unredacted text.
    """
    if not request_id:
        return None
    try:
        from chartreux.core.tools.secret_redaction import redact

        return redact(request_id)[:_MAX_REQUEST_ID_CHARS]
    except Exception:
        return None


# ---------------------------------------------------------------------------
# The compact typed terminal projection of one failed ROOT turn (R2)
# ---------------------------------------------------------------------------

#: Cause categories with MEASURED transient provenance — the recovery-layer
#: eligible forms (mirroring RecoveryEligibleClass; launch_failure is the
#: admission-phase form handled separately via measured zero transport
#: starts).
_TRANSIENT_CAUSE_CATEGORIES = frozenset({
    FailureCategory.CONNECTION,
    FailureCategory.TIMEOUT,
    FailureCategory.SERVER_ERROR,
    FailureCategory.OVERLOAD,
    FailureCategory.RATE_LIMIT,
    FailureCategory.INCOMPLETE_STREAM,
    FailureCategory.EMPTY_RESPONSE,
})

#: Root-level stop reasons that are NEVER auto-retryable (budget stops end
#: the turn through the normal limit path, but a failure observed with one
#: of these stop reasons is protected regardless).
_PROTECTED_STOP_REASONS = frozenset({
    "budget_exceeded",
    "budget_unverifiable",
    "user_cancelled",
    "orchestrator_cancelled",
    "retasked",
})


class RootTurnFailure(BaseModel):
    """The compact typed terminal projection of one failed ROOT turn (R2).

    Captured by the app-server at the terminalization boundary — classified
    BEFORE ``public_error()`` destroys the exception context, and with the
    interaction/effect barriers captured BEFORE ``_finalize_turn`` cancels
    callbacks (absence of callbacks afterwards proves nothing). The
    app-server recovery gate consumes this projection for its eligibility,
    scheduling, and admission decisions; it never crosses the wire.

    ``transient_cause_confirmed`` is the trusted-origin requirement: the
    failure carries MEASURED logical-call evidence (attached by the gateway
    for a model call, or by the coordinator for an admission-phase launch
    failure) AND sits in a model-call phase. A tool's network exception
    carries no call evidence and is never a root model-call failure; a
    finalization/monitor failure never qualifies either.
    """

    model_config = ConfigDict(extra="forbid")

    turn_id: str
    observed_at: float
    cause_category: FailureCategory | None = None
    phase: FailurePhase
    failure_class: FailureClass
    transient_cause_confirmed: bool
    # The ATTEMPTED deployment, frozen at the completion boundary.
    attempted_base_model: str | None = None
    attempted_provider: str | None = None
    attempted_wire_name: str | None = None
    endpoint: str | None = None
    retry_after_seconds: float | None = None
    logical_call_id: str | None = None
    transport_attempts: int | None = None
    # Measured elapsed seconds of the failed logical call (the episode
    # deadline anchors to the call's START: capture time minus this).
    call_elapsed_seconds: float | None = None
    semantic_output_published: bool | None = None
    # Effect/interaction barriers captured BEFORE callback cancellation.
    effects_settled: bool = True
    interaction_barrier: bool = False
    barrier_detail: str = ""
    stop_reason: str | None = None


def _observed_attempted_deployment(
    observation: Any | None,
) -> tuple[str, str, str] | None:
    """The attempted-deployment identity attached to the observed error, if any.

    The coordinator attaches the identity that served the failed attempt
    BEFORE the rollback restores the committed model (review R11); the
    projection prefers it over the (possibly restored) committed identity.
    """
    identity = (
        getattr(observation, "attempted_deployment", None)
        if observation is not None
        else None
    )
    if (
        isinstance(identity, tuple)
        and len(identity) == _ATTEMPTED_DEPLOYMENT_ARITY
        and all(isinstance(part, str) for part in identity)
    ):
        return cast("tuple[str, str, str]", identity)
    return None


#: Permanent configuration/auth-adjacent categories (never auto-retryable).
_PERMANENT_CONFIG_CATEGORIES = frozenset({
    FailureCategory.BILLING,
    FailureCategory.QUOTA,
    FailureCategory.INVALID_REQUEST,
    FailureCategory.CONTEXT_SIZE,
})

#: The transient cause categories mapped to their operational class.
_TRANSIENT_CLASS_BY_CATEGORY: dict[FailureCategory, FailureClass] = {
    FailureCategory.RATE_LIMIT: FailureClass.PROVIDER_RATE_LIMIT,
    FailureCategory.OVERLOAD: FailureClass.PROVIDER_OVERLOADED,
    FailureCategory.TIMEOUT: FailureClass.PROVIDER_STALL,
    FailureCategory.CONNECTION: FailureClass.PROVIDER_STALL,
    FailureCategory.SERVER_ERROR: FailureClass.PROVIDER_STALL,
    FailureCategory.INCOMPLETE_STREAM: FailureClass.INCOMPLETE_STREAM,
    FailureCategory.EMPTY_RESPONSE: FailureClass.EMPTY_RESPONSE,
}


def _derive_root_failure_class(
    *,
    stop_reason: str | None,
    cause_category: FailureCategory | None,
    phase: FailurePhase,
    exception_type: str | None,
    transport_attempts: int | None,
) -> FailureClass:
    """The operational class of a failed root turn (protected classes dominate).

    ``launch_failure`` requires admission/init evidence: a measured zero
    transport starts (or the admission-phase exhaustion error itself).
    """
    is_admission_error = exception_type == "AllDeploymentsUnavailableError"
    if (
        stop_reason in _PROTECTED_STOP_REASONS
        or cause_category is FailureCategory.CANCELLATION
    ):
        return FailureClass.CANCELLATION
    if phase is FailurePhase.MONITOR:
        return FailureClass.INTERNAL
    if cause_category is FailureCategory.AUTH:
        return FailureClass.AUTHORIZATION
    if cause_category in _PERMANENT_CONFIG_CATEGORIES:
        return FailureClass.PERMANENT_CONFIG
    if (
        phase is FailurePhase.ADMISSION
        or (is_admission_error and transport_attempts == 0)
    ) and (transport_attempts == 0 or is_admission_error):
        return FailureClass.LAUNCH_FAILURE
    # None (no classification) degrades honestly to UNKNOWN, never a guess.
    transient = (
        _TRANSIENT_CLASS_BY_CATEGORY.get(cause_category)
        if cause_category is not None
        else None
    )
    return transient if transient is not None else FailureClass.UNKNOWN


def build_root_turn_failure(
    *,
    observation: Any | None,
    turn_id: str,
    committed: Any | None = None,
    stop_reason: str | None = None,
    effects_settled: bool = True,
    interaction_barrier: bool = False,
    barrier_detail: str = "",
    clock: Any = time.time,
) -> RootTurnFailure:
    """Build the compact typed terminal projection for one failed root turn.

    Total by contract: a capture failure must never mask the primary
    failure, so missing evidence degrades honestly to null/False rather
    than raising.
    """
    call_evidence: CallEvidence | None = None
    if observation is not None:
        evidence = getattr(observation, "call_evidence", None)
        if isinstance(evidence, CallEvidence):
            call_evidence = evidence

    transport_attempts: int | None = None
    if call_evidence is not None:
        transport_attempts = call_evidence.transport_attempts

    phase = _derive_phase(
        getattr(observation, "phase", None) if observation is not None else None,
        transport_attempts,
    )

    cause_category: FailureCategory | None = None
    classification: Any = getattr(observation, "classification", None)
    if classification is not None:
        category = getattr(classification, "category", None)
        if isinstance(category, FailureCategory):
            cause_category = category

    exception_type: str | None = (
        getattr(observation, "exception_type", None)
        if observation is not None
        else None
    )
    failure_class = _derive_root_failure_class(
        stop_reason=stop_reason,
        cause_category=cause_category,
        phase=phase,
        exception_type=exception_type,
        transport_attempts=transport_attempts,
    )

    # --- trusted-origin transient confirmation ---
    measured_call = call_evidence is not None
    model_call_phase = phase in {FailurePhase.LLM_CALL, FailurePhase.ADMISSION}
    transient_cause_confirmed = (
        measured_call
        and model_call_phase
        and (
            phase is FailurePhase.ADMISSION
            or cause_category in _TRANSIENT_CAUSE_CATEGORIES
        )
    )

    # Review R11: the attempted deployment frozen at the completion
    # boundary — the identity that actually served the failed attempt —
    # preferred over the committed identity the rollback restored.
    base_model, provider, wire_name = _observed_attempted_deployment(observation) or (
        getattr(committed, "base_model", None),
        getattr(committed, "provider", None),
        getattr(committed, "wire_name", None),
    )

    return RootTurnFailure(
        turn_id=turn_id,
        observed_at=(
            observation.observed_at
            if observation is not None and getattr(observation, "observed_at", None)
            else clock()
        ),
        cause_category=cause_category,
        phase=phase,
        failure_class=failure_class,
        transient_cause_confirmed=transient_cause_confirmed,
        attempted_base_model=base_model if isinstance(base_model, str) else None,
        attempted_provider=provider if isinstance(provider, str) else None,
        attempted_wire_name=wire_name if isinstance(wire_name, str) else None,
        endpoint=canonical_endpoint_identity(
            getattr(observation, "endpoint", None) if observation is not None else None
        ),
        retry_after_seconds=(
            classification.retry_after if classification is not None else None
        ),
        logical_call_id=(
            call_evidence.logical_call_id if call_evidence is not None else None
        ),
        transport_attempts=transport_attempts,
        call_elapsed_seconds=(
            call_evidence.elapsed_seconds if call_evidence is not None else None
        ),
        semantic_output_published=(
            call_evidence.semantic_output_published
            if call_evidence is not None
            else None
        ),
        effects_settled=effects_settled,
        interaction_barrier=interaction_barrier,
        barrier_detail=barrier_detail,
        stop_reason=stop_reason,
    )
